#!/usr/bin/env python3
"""
projections.py — DraftKings Points Projection Model

Trains position-specific ElasticNet models on historical lag features to
project dk_points for each player-week, then backtests against 2023–2024
actuals.

Training:           seasons 2009–2022
Validation/Backtest: 2023–2024

Outputs:
  data/processed/projections_backtest.parquet  — predicted vs actual (2023–2024)
  data/models/model_<pos>.pkl                  — trained pipeline per position

Usage:
  python3 src/projections.py                           # train + backtest only
  python3 src/projections.py --season 2025 --week 18  # + current-week predictions
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS    = ROOT / "data" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

TRAIN_MIN = 2009
TRAIN_MAX = 2022
VAL_MIN   = 2023
VAL_MAX   = 2024

PARAM_GRID = {
    "elasticnet__alpha":    [0.01, 0.05, 0.1, 0.5, 1.0],
    "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

# ── Feature column definitions ────────────────────────────────────────────────
#
# All features are lag-based (prior games) to avoid data leakage.
# implied_team_total from the betting market is the only forward-looking input
# and is available before game time.

FEATURES: dict[str, list[str]] = {
    "QB": [
        # Prior-game passing efficiency
        "yards_per_attempt_lag1",  "yards_per_attempt_lag2",
        "completion_pct_lag1",     "completion_pct_lag2",
        "pass_td_rate_lag1",       "pass_td_rate_lag2",
        "int_rate_lag1",           "int_rate_lag2",
        # Prior-game passing volume
        "passing_att_lag1",        "passing_att_lag2",
        "passing_yds_lag1",
        "snap_pct_lag1",
        # QB scrambles contribute DK points
        "rushing_att_lag1",        "rushing_yds_lag1",
        # Game context
        "implied_team_total",
    ],
    "RB": [
        # Prior-game rushing usage and efficiency
        "carry_share_lag1",        "carry_share_lag2",
        "yards_per_carry_lag1",    "yards_per_carry_lag2",
        "rush_td_rate_lag1",       "rush_td_rate_lag2",
        "rushing_att_lag1",        "rushing_att_lag2",
        "rushing_yds_lag1",
        "snap_pct_lag1",
        # RB receiving role
        "target_share_lag1",       "target_share_lag2",
        "catch_rate_lag1",
        "yards_per_target_lag1",
        "rec_td_rate_lag1",
        "receiving_tar_lag1",
        # Game context
        "implied_team_total",
    ],
    "WR": [
        # Prior-game receiving usage and efficiency
        "target_share_lag1",       "target_share_lag2",
        "catch_rate_lag1",         "catch_rate_lag2",
        "yards_per_target_lag1",   "yards_per_target_lag2",
        "rec_td_rate_lag1",        "rec_td_rate_lag2",
        "receiving_tar_lag1",      "receiving_tar_lag2",
        "receiving_yds_lag1",
        "snap_pct_lag1",
        # Game context
        "implied_team_total",
        # Opponent pass defense quality
        "opp_pass_yds_allowed_ytd",
        "opp_pass_yds_allowed_roll4",
    ],
    "TE": [
        "target_share_lag1",       "target_share_lag2",
        "catch_rate_lag1",         "catch_rate_lag2",
        "yards_per_target_lag1",   "yards_per_target_lag2",
        "rec_td_rate_lag1",        "rec_td_rate_lag2",
        "receiving_tar_lag1",      "receiving_tar_lag2",
        "receiving_yds_lag1",
        "snap_pct_lag1",
        "implied_team_total",
        # Opponent pass defense quality
        "opp_pass_yds_allowed_ytd",
        "opp_pass_yds_allowed_roll4",
    ],
}

# Primary lag1 column: rows are excluded from training if this equals -1
# (no prior history — the player's first ever game)
PRIMARY_LAG: dict[str, str] = {
    "QB": "yards_per_attempt_lag1",
    "RB": "carry_share_lag1",
    "WR": "target_share_lag1",
    "TE": "target_share_lag1",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_schedule_implied() -> pd.DataFrame:
    """
    Expand schedule (one row per game) to one row per team per game with
    implied_team_total derived from spread + total.

    Spread sign convention (confirmed):
      POSITIVE spread_line = home team FAVORED
      home_implied = (total_line + spread_line) / 2
      away_implied = (total_line - spread_line) / 2
    """
    sched = pd.read_parquet(PROCESSED / "schedule.parquet")

    home = sched[["season", "week", "home_pfr_code", "spread_line", "total_line"]].copy()
    home = home.rename(columns={"home_pfr_code": "pfr_code"})
    home["implied_team_total"] = (home["total_line"] + home["spread_line"]) / 2

    away = sched[["season", "week", "away_pfr_code", "spread_line", "total_line"]].copy()
    away = away.rename(columns={"away_pfr_code": "pfr_code"})
    away["implied_team_total"] = (away["total_line"] - away["spread_line"]) / 2

    ctx = pd.concat([home, away], ignore_index=True)
    ctx = ctx[["pfr_code", "season", "week", "implied_team_total"]].dropna(
        subset=["implied_team_total"]
    )
    return ctx


def _load_position_data(pos: str, sched_ctx: pd.DataFrame) -> pd.DataFrame:
    """
    Load and merge the relevant feature parquets for a position, then join
    schedule implied totals for game-context.

    QB  — passing_features (QB rows) + rushing_features (QB rows, for scrambles)
    RB  — rushing_features (RB rows) + subset of passing_features (RB receiving)
    WR  — passing_features (WR rows) only
    TE  — passing_features (TE rows) only
    """
    rush = pd.read_parquet(PROCESSED / "rushing_features.parquet")
    pf   = pd.read_parquet(PROCESSED / "passing_features.parquet")

    if pos == "QB":
        base = pf[pf["position_id"] == "QB"].copy()
        rush_qb = (
            rush[rush["position_id"] == "QB"][
                ["player_id", "season", "week",
                 "rushing_att_lag1", "rushing_yds_lag1", "rushing_att_lag2"]
            ].copy()
        )
        base = base.merge(rush_qb, on=["player_id", "season", "week"], how="left")
        for col in ["rushing_att_lag1", "rushing_yds_lag1", "rushing_att_lag2"]:
            base[col] = base[col].fillna(0)

    elif pos == "RB":
        # Rushing is primary for RBs; dk_points from rushing_features is the
        # player's total for the week (rushing + receiving combined)
        base = rush[rush["position_id"] == "RB"].copy()
        rb_recv = (
            pf[pf["position_id"] == "RB"][
                ["player_id", "season", "week",
                 "target_share_lag1", "target_share_lag2",
                 "catch_rate_lag1",
                 "yards_per_target_lag1",
                 "rec_td_rate_lag1",
                 "receiving_tar_lag1"]
            ].copy()
        )
        base = base.merge(rb_recv, on=["player_id", "season", "week"], how="left")
        for col in ["target_share_lag1", "target_share_lag2", "catch_rate_lag1",
                    "yards_per_target_lag1", "rec_td_rate_lag1", "receiving_tar_lag1"]:
            base[col] = base[col].fillna(0)

    elif pos in ("WR", "TE"):
        base = pf[pf["position_id"] == pos].copy()

    else:
        raise ValueError(f"Unknown position: {pos}")

    # Join game context (implied total from betting market)
    base = base.merge(sched_ctx, on=["pfr_code", "season", "week"], how="left")

    # Fill games with no spread data (pre-2003 or data gaps) with the global median
    med = base["implied_team_total"].median()
    base["implied_team_total"] = base["implied_team_total"].fillna(med)

    return base


# ── Feature matrix construction ───────────────────────────────────────────────

def _align_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame with exactly the expected feature columns in order.
    Columns absent from df are added as zeros (handles missing features in
    current-week files that were present in training).
    """
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    return df[features].copy()


def _prepare_Xy(
    pos: str, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Filter rows, replace -1 sentinels, and return (X, y, meta).

    Rows where the primary lag1 feature equals -1 are excluded — these are
    a player's very first game where no prior history exists. Remaining -1
    values (lag2 for a player in their second ever game) are replaced with 0.
    """
    features = FEATURES[pos]
    primary  = PRIMARY_LAG[pos]

    df = df[df[primary] != -1].copy()
    df = df.dropna(subset=["dk_points"])

    # Align to expected features (adds missing cols as 0)
    X    = _align_features(df, features)
    X    = X.replace(-1, 0)        # replace remaining lag2 sentinels
    y    = df["dk_points"].reset_index(drop=True)
    meta = df[["player_id", "name", "pfr_code", "position_id",
               "season", "week"]].reset_index(drop=True)
    X    = X.reset_index(drop=True)

    return X, y, meta


# ── Model training ─────────────────────────────────────────────────────────────

def _train_model(pos: str, X_train: pd.DataFrame,
                 y_train: pd.Series) -> Pipeline:
    """
    Fit StandardScaler → ElasticNet via 5-fold GridSearchCV.
    Returns the best refitted Pipeline.
    """
    pipe = Pipeline([
        ("scaler",     StandardScaler()),
        ("elasticnet", ElasticNet(max_iter=5000, random_state=42)),
    ])
    cv   = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe, PARAM_GRID, cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, refit=True,
    )
    grid.fit(X_train, y_train)
    best = grid.best_params_
    log.info(
        f"  {pos} best params: alpha={best['elasticnet__alpha']:.3f}  "
        f"l1_ratio={best['elasticnet__l1_ratio']:.2f}  "
        f"(CV MAE: {-grid.best_score_:.2f})"
    )
    return grid.best_estimator_


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(
    pos: str,
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_val: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict on the validation set, print per-position metrics, and return
    a backtest DataFrame (player_id, name, season, week, predicted_dk, actual_dk).
    """
    y_pred = np.clip(model.predict(X_val), 0, None)

    mae  = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    r,   _ = pearsonr(y_val, y_pred)
    rho, _ = spearmanr(y_val, y_pred)

    n_weeks  = meta_val["week"].nunique() * meta_val["season"].nunique()
    # Per-week Pearson r to show consistency
    week_rs = []
    for (ssn, wk), grp_meta in meta_val.groupby(["season", "week"]):
        idx = grp_meta.index
        if len(idx) < 3:
            continue
        gr, _ = pearsonr(y_val.loc[idx], y_pred[idx])
        week_rs.append(gr)
    median_week_r = float(np.median(week_rs)) if week_rs else float("nan")

    print(f"\n{'─'*60}")
    print(f"  {pos}  Backtest  ({VAL_MIN}–{VAL_MAX})")
    print(f"  N              = {len(y_val):,} player-games")
    print(f"  MAE            = {mae:.2f} DK pts")
    print(f"  RMSE           = {rmse:.2f} DK pts")
    print(f"  Pearson r      = {r:.3f}")
    print(f"  Spearman ρ     = {rho:.3f}")
    print(f"  Median wk r    = {median_week_r:.3f}  (across {len(week_rs)} weeks)")
    print(f"{'─'*60}")

    result = meta_val.copy()
    result["predicted_dk"] = np.round(y_pred, 2)
    result["actual_dk"]    = y_val.values
    result["error"]        = np.round(result["predicted_dk"] - result["actual_dk"], 2)
    return result


# ── Current-week predictions ──────────────────────────────────────────────────

def _predict_current_week(
    season: int, week: int, models: dict[str, Pipeline]
) -> pd.DataFrame:
    """
    Load rushing_current.parquet + passing_current.parquet (produced by rushing.py
    and passing.py with --season/--week args) and generate dk_points predictions
    for every active player using the trained position models.
    """
    rush_cur_path = PROCESSED / "rushing_current.parquet"
    pass_cur_path = PROCESSED / "passing_current.parquet"

    if not rush_cur_path.exists() or not pass_cur_path.exists():
        log.error(
            "rushing_current or passing_current not found. Run first:\n"
            f"  python3 src/rushing.py --season {season} --week {week}\n"
            f"  python3 src/passing.py --season {season} --week {week}"
        )
        return pd.DataFrame()

    rush_cur = pd.read_parquet(rush_cur_path)
    pass_cur = pd.read_parquet(pass_cur_path)

    all_preds = []

    for pos in ["QB", "RB", "WR", "TE"]:
        if pos not in models:
            continue
        model    = models[pos]
        features = FEATURES[pos]

        if pos == "QB":
            df = pass_cur[pass_cur["position_id"] == "QB"].copy()
            rush_qb = rush_cur[rush_cur["position_id"] == "QB"][
                ["player_id", "rushing_att_lag1", "rushing_yds_lag1"]
            ]
            if not rush_qb.empty:
                df = df.merge(rush_qb, on="player_id", how="left")

        elif pos == "RB":
            df = rush_cur[rush_cur["position_id"] == "RB"].copy()
            rb_recv_cols = [
                "player_id",
                "target_share_lag1", "target_share_lag2",
                "catch_rate_lag1",   "yards_per_target_lag1",
                "rec_td_rate_lag1",  "receiving_tar_lag1",
            ]
            available_recv = [c for c in rb_recv_cols if c in pass_cur.columns]
            rb_recv = pass_cur[pass_cur["position_id"] == "RB"][available_recv]
            if not rb_recv.empty:
                df = df.merge(rb_recv, on="player_id", how="left")

        elif pos in ("WR", "TE"):
            df = pass_cur[pass_cur["position_id"] == pos].copy()

        else:
            continue

        if df.empty:
            log.warning(f"  No current-week rows for {pos}")
            continue

        # implied_team_total is already in both current-week files
        X_cur = _align_features(df.copy(), features).fillna(0).replace(-1, 0)
        preds = np.clip(model.predict(X_cur), 0, None)

        out = pd.DataFrame({
            "player_id":   df["player_id"].values,
            "name":        df["name"].values,
            "pfr_code":    df["pfr_code"].values,
            "position_id": pos,
            "season":      season,
            "week":        week,
            "predicted_dk": np.round(preds, 2),
        })
        all_preds.append(out)

    if not all_preds:
        log.warning("No current-week predictions generated.")
        return pd.DataFrame()

    combined = pd.concat(all_preds, ignore_index=True)
    combined = combined.sort_values("predicted_dk", ascending=False)

    out_path = PROCESSED / "projections_current.parquet"
    combined.to_parquet(out_path, index=False)
    log.info(f"Saved projections_current.parquet ({len(combined):,} players)")

    print(f"\n{'═'*60}")
    print(f"  CURRENT WEEK PROJECTIONS — Season {season}  Week {week}")
    print(f"{'═'*60}")
    for pos in ["QB", "RB", "WR", "TE"]:
        top = combined[combined["position_id"] == pos].head(12)
        if top.empty:
            continue
        print(f"\n  {pos}:")
        print(top[["name", "pfr_code", "predicted_dk"]].to_string(index=False))

    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def run(season: int | None = None, week: int | None = None) -> None:
    log.info("Loading schedule context ...")
    sched_ctx = _load_schedule_implied()
    log.info(f"  {len(sched_ctx):,} team-game rows with implied totals")

    models:   dict[str, Pipeline]   = {}
    backtest: list[pd.DataFrame]    = []

    for pos in ["QB", "RB", "WR", "TE"]:
        log.info(f"\n{'─'*50}")
        log.info(f"Position: {pos}")

        df_all = _load_position_data(pos, sched_ctx)
        X_all, y_all, meta_all = _prepare_Xy(pos, df_all)

        seasons    = meta_all["season"]
        train_mask = seasons.between(TRAIN_MIN, TRAIN_MAX)
        val_mask   = seasons.between(VAL_MIN,   VAL_MAX)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
        meta_val         = meta_all[val_mask].reset_index(drop=True)
        y_val            = y_val.reset_index(drop=True)

        log.info(f"  Train rows: {train_mask.sum():,}  |  Val rows: {val_mask.sum():,}")

        # Train
        model        = _train_model(pos, X_train, y_train)
        models[pos]  = model

        # Persist model
        model_path = MODELS / f"model_{pos}.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump({"model": model, "features": FEATURES[pos]}, fh)
        log.info(f"  Saved {model_path.name}")

        # Backtest
        if val_mask.sum() > 0:
            result = _evaluate(pos, model, X_val, y_val, meta_val)
            backtest.append(result)

    # Save combined backtest output
    if backtest:
        bt_df    = pd.concat(backtest, ignore_index=True)
        bt_path  = PROCESSED / "projections_backtest.parquet"
        bt_df.to_parquet(bt_path, index=False)
        log.info(f"\nSaved projections_backtest.parquet ({len(bt_df):,} rows)")

        print(f"\n{'═'*60}")
        print(f"  BACKTEST SUMMARY  ({VAL_MIN}–{VAL_MAX})")
        print(f"{'═'*60}")
        print(f"  {'Pos':<5} {'N':>7}  {'MAE':>6}  {'RMSE':>6}  {'r':>6}  {'Spearman':>8}")
        print(f"  {'─'*50}")
        for pos in ["QB", "RB", "WR", "TE"]:
            sub = bt_df[bt_df["position_id"] == pos]
            if sub.empty:
                continue
            r,   _ = pearsonr(sub["actual_dk"], sub["predicted_dk"])
            rho, _ = spearmanr(sub["actual_dk"], sub["predicted_dk"])
            mae    = mean_absolute_error(sub["actual_dk"], sub["predicted_dk"])
            rmse   = mean_squared_error(sub["actual_dk"], sub["predicted_dk"]) ** 0.5
            print(f"  {pos:<5} {len(sub):>7,}  {mae:>6.2f}  {rmse:>6.2f}  {r:>6.3f}  {rho:>8.3f}")
        print(f"{'═'*60}")

    # Current-week predictions if requested
    if season is not None and week is not None:
        _predict_current_week(season, week, models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train position-specific DK point projection models and backtest."
    )
    parser.add_argument("--season", type=int, default=None,
                        help="Season year for current-week prediction output")
    parser.add_argument("--week",   type=int, default=None,
                        help="Week number for current-week prediction output")
    args = parser.parse_args()

    if (args.season is None) != (args.week is None):
        parser.error("--season and --week must be provided together.")

    run(args.season, args.week)
