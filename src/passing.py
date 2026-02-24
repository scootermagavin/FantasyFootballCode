#!/usr/bin/env python3
"""
passing.py — Player Passing/Receiving Feature Module

Builds per-player passing and receiving features for both:

  1. Historical training dataset (all seasons 2000–present)
     Output: data/processed/passing_features.parquet

  2. Current-week projection inputs (one upcoming game week)
     Output: data/processed/passing_current.parquet
     Requires: team_game_context.parquet (run team_scoring.py first)

Features computed per player-week:

  RECEIVERS (WR, TE, RB):
    target_share       — player receiving_tar / team targets
    catch_rate         — receiving_rec / receiving_tar
    yards_per_target   — receiving_yds / receiving_tar
    rec_td_rate        — receiving_td / receiving_tar
    snap_pct           — offensive_snapcount_percentage / 100

  QBs:
    completion_pct     — passing_cmp / passing_att
    yards_per_attempt  — passing_yds / passing_att
    pass_td_rate       — passing_td / passing_att
    int_rate           — passing_int / passing_att
    snap_pct           — offensive_snapcount_percentage / 100

Lag features (1-week and 2-week prior):
  *_lag1, *_lag2      — prior-game values per player (grouped by player_id)
                        Sentinel value -1 indicates no prior history.

Projection columns (current week only, receivers):
  projected_targets    — target_share_lag1 × expected_pass_att × pass_def_adj
  projected_rec        — projected_targets × catch_rate_lag1
  projected_rec_yds    — projected_targets × yards_per_target_lag1
  projected_rec_td     — projected_targets × rec_td_rate_lag1

Projection columns (current week only, QBs):
  projected_pass_att   — expected_pass_att × pass_def_adj
  projected_pass_yds   — projected_pass_att × yards_per_attempt_lag1
  projected_pass_td    — projected_pass_att × pass_td_rate_lag1
  projected_pass_int   — projected_pass_att × int_rate_lag1

Usage:
  python3 src/passing.py                           # historical features only
  python3 src/passing.py --season 2024 --week 14   # historical + current week
"""

import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"

RECEIVER_POSITIONS = {"WR", "TE", "RB"}
QB_POSITIONS = {"QB"}
ALL_PASS_POSITIONS = RECEIVER_POSITIONS | QB_POSITIONS

MIN_TAR_FOR_RATE = 1    # minimum targets to compute per-target receiver rates
MIN_ATT_FOR_RATE = 1    # minimum pass attempts to compute per-attempt QB rates
MIN_SEASON = 2009       # exclude pre-2009 data (targets unavailable 2003-2008, different era)


# ── Shared lag helper ────────────────────────────────────────────────────────

def _add_lag_features(df: pd.DataFrame, lag_cols: list[str],
                      n_lags: int = 2) -> pd.DataFrame:
    """
    Add n-week lag features for each column in lag_cols, grouped by player_id.

    Players are sorted by season then week before shifting so that week 1 of a
    new season correctly lags from the prior season's final week.

    Missing lag values (start of career, gaps) are filled with -1 as a sentinel
    that downstream models can recognise as "no prior history."
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()

    # Fill NaN metrics with 0 before lagging (zero targets is genuine information,
    # distinct from -1 "no history")
    for col in lag_cols:
        df[col] = df[col].fillna(0)

    for lag in range(1, n_lags + 1):
        shifted = df.groupby("player_id")[lag_cols].shift(lag)
        for col in lag_cols:
            df[f"{col}_lag{lag}"] = shifted[col].fillna(-1)

    return df


# ── Receiver features ─────────────────────────────────────────────────────────

def _receiver_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute target_share, catch_rate, yards_per_target, rec_td_rate, snap_pct
    for receiver rows (WR, TE, RB).
    """
    df = df.copy()

    df["target_share"] = np.where(
        df["team_targets"] > 0,
        df["receiving_tar"] / df["team_targets"],
        np.nan,
    )

    df["catch_rate"] = np.where(
        df["receiving_tar"] >= MIN_TAR_FOR_RATE,
        df["receiving_rec"] / df["receiving_tar"],
        np.nan,
    )

    df["yards_per_target"] = np.where(
        df["receiving_tar"] >= MIN_TAR_FOR_RATE,
        df["receiving_yds"] / df["receiving_tar"],
        np.nan,
    )

    df["rec_td_rate"] = np.where(
        df["receiving_tar"] >= MIN_TAR_FOR_RATE,
        df["receiving_td"] / df["receiving_tar"],
        np.nan,
    )

    df["snap_pct"] = df["offensive_snapcount_percentage"].fillna(0) / 100.0

    return df


# ── QB features ───────────────────────────────────────────────────────────────

def _qb_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute completion_pct, yards_per_attempt, pass_td_rate, int_rate, snap_pct
    for QB rows.
    """
    df = df.copy()

    df["completion_pct"] = np.where(
        df["passing_att"] >= MIN_ATT_FOR_RATE,
        df["passing_cmp"] / df["passing_att"],
        np.nan,
    )

    df["yards_per_attempt"] = np.where(
        df["passing_att"] >= MIN_ATT_FOR_RATE,
        df["passing_yds"] / df["passing_att"],
        np.nan,
    )

    df["pass_td_rate"] = np.where(
        df["passing_att"] >= MIN_ATT_FOR_RATE,
        df["passing_td"] / df["passing_att"],
        np.nan,
    )

    df["int_rate"] = np.where(
        df["passing_att"] >= MIN_ATT_FOR_RATE,
        df["passing_int"] / df["passing_att"],
        np.nan,
    )

    df["snap_pct"] = df["offensive_snapcount_percentage"].fillna(0) / 100.0

    return df


# ── Opponent pass defense lookup ──────────────────────────────────────────────

def _build_pass_defense_lookup() -> pd.DataFrame:
    """
    Build a per-team, per-week lookup of opponent passing yards allowed.

    Source: defense_context.parquet column `passing_yds_opp`, which records
    passing yards the opponent (i.e. the other team's offense) generated
    against this team — equivalent to passing yards allowed by this team.

    Returned features (both shifted back 1 week so week-W row reflects W-1):
      opp_pass_yds_allowed_ytd   — season-to-date expanding mean through W-1
      opp_pass_yds_allowed_roll4 — rolling 4-game mean through W-1

    Week-1 / insufficient-history NaN → filled with league-season median.
    """
    dc = pd.read_parquet(PROCESSED / "defense_context.parquet",
                         columns=["pfr_code", "season", "week", "passing_yds_opp"])
    dc = dc.sort_values(["pfr_code", "season", "week"]).copy()

    dc["opp_pass_yds_allowed_ytd"] = (
        dc.groupby(["pfr_code", "season"])["passing_yds_opp"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    dc["opp_pass_yds_allowed_roll4"] = (
        dc.groupby(["pfr_code", "season"])["passing_yds_opp"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean().shift(1))
    )

    # Fill week-1 NaN with league-season median
    for col in ["opp_pass_yds_allowed_ytd", "opp_pass_yds_allowed_roll4"]:
        season_med = dc.groupby("season")[col].transform("median")
        dc[col] = dc[col].fillna(season_med)

    return dc[["pfr_code", "season", "week",
               "opp_pass_yds_allowed_ytd", "opp_pass_yds_allowed_roll4"]].copy()


def _build_opponent_map() -> pd.DataFrame:
    """
    Expand schedule (one row per game) to one row per team per game,
    returning pfr_code, season, week, opponent_pfr_code.
    """
    sched = pd.read_parquet(PROCESSED / "schedule.parquet",
                            columns=["season", "week", "home_pfr_code", "away_pfr_code"])
    home = sched.rename(columns={"home_pfr_code": "pfr_code",
                                  "away_pfr_code": "opponent_pfr_code"})
    away = sched.rename(columns={"away_pfr_code": "pfr_code",
                                  "home_pfr_code": "opponent_pfr_code"})
    opp_map = pd.concat(
        [home[["pfr_code", "season", "week", "opponent_pfr_code"]],
         away[["pfr_code", "season", "week", "opponent_pfr_code"]]],
        ignore_index=True,
    )
    return opp_map


# ── Main builders ─────────────────────────────────────────────────────────────

def build_historical_features() -> pd.DataFrame:
    """
    Build the full historical passing/receiving feature dataset covering all seasons.

    Steps:
      1. Load players and team_stats parquets.
      2. Merge team targets (receiver denominator) onto each player-week row.
      3. Compute position-specific per-play metrics for receivers and QBs.
      4. Add 1- and 2-week lag features per player.
      5. Concatenate receiver and QB DataFrames into one output.

    Returns the feature DataFrame and saves to passing_features.parquet.
    """
    log.info("Building historical passing features ...")

    players = pd.read_parquet(PROCESSED / "players.parquet")
    team_stats = pd.read_parquet(PROCESSED / "team_stats.parquet")

    passers = players[
        players["position_id"].isin(ALL_PASS_POSITIONS) &
        (players["season"] >= MIN_SEASON)
    ].copy()
    log.info(f"  {len(passers):,} player-week rows for positions {ALL_PASS_POSITIONS} (seasons {MIN_SEASON}+)")

    # Team-level denominators: targets for receivers, attempts for QB context
    team_pass_ctx = team_stats[["pfr_code", "season", "week", "targets", "attempts"]].rename(
        columns={"targets": "team_targets", "attempts": "team_pass_att"}
    )

    passers = passers.merge(team_pass_ctx, on=["pfr_code", "season", "week"], how="left")

    unmatched = passers["team_targets"].isnull().sum()
    if unmatched > 0:
        log.warning(f"  {unmatched} rows missing team targets after merge")

    # ── Receivers ─────────────────────────────────────────────────────────────
    receivers = passers[passers["position_id"].isin(RECEIVER_POSITIONS)].copy()

    for col in ["receiving_tar", "receiving_rec", "receiving_yds", "receiving_td"]:
        receivers[col] = receivers[col].fillna(0)

    receivers = _receiver_metrics(receivers)

    rec_lag_cols = [
        "target_share", "catch_rate", "yards_per_target", "rec_td_rate",
        "snap_pct", "receiving_tar", "receiving_yds",
    ]
    receivers = _add_lag_features(receivers, rec_lag_cols, n_lags=2)

    # ── Opponent pass defense features ────────────────────────────────────────
    def_lookup = _build_pass_defense_lookup()
    opp_map    = _build_opponent_map()

    # Join opponent team onto each receiver row
    receivers = receivers.merge(opp_map, on=["pfr_code", "season", "week"], how="left")

    # Join defense lookup keyed on the opponent (defensive team)
    receivers = receivers.merge(
        def_lookup.rename(columns={"pfr_code": "opponent_pfr_code"}),
        on=["opponent_pfr_code", "season", "week"],
        how="left",
    )

    # Fill any remaining NaN with global median (no spread data / missing games)
    for col in ["opp_pass_yds_allowed_ytd", "opp_pass_yds_allowed_roll4"]:
        med = def_lookup[col].median()
        receivers[col] = receivers[col].fillna(med)

    rec_keep = [
        "player_id", "name", "pfr_code", "position_id", "season", "week",
        "receiving_tar", "receiving_rec", "receiving_yds", "receiving_td",
        "target_share", "catch_rate", "yards_per_target", "rec_td_rate", "snap_pct",
        "team_targets",
        "target_share_lag1", "catch_rate_lag1", "yards_per_target_lag1",
        "rec_td_rate_lag1", "snap_pct_lag1", "receiving_tar_lag1", "receiving_yds_lag1",
        "target_share_lag2", "catch_rate_lag2", "yards_per_target_lag2",
        "rec_td_rate_lag2", "snap_pct_lag2", "receiving_tar_lag2", "receiving_yds_lag2",
        "opp_pass_yds_allowed_ytd", "opp_pass_yds_allowed_roll4",
        "dk_points",
    ]
    receivers = receivers[[c for c in rec_keep if c in receivers.columns]]

    log.info(f"  Receiver rows: {len(receivers):,}")

    # ── QBs ───────────────────────────────────────────────────────────────────
    qbs = passers[passers["position_id"].isin(QB_POSITIONS)].copy()

    for col in ["passing_att", "passing_cmp", "passing_yds", "passing_td", "passing_int"]:
        qbs[col] = qbs[col].fillna(0)

    qbs = _qb_metrics(qbs)

    qb_lag_cols = [
        "completion_pct", "yards_per_attempt", "pass_td_rate", "int_rate",
        "snap_pct", "passing_att", "passing_yds",
    ]
    qbs = _add_lag_features(qbs, qb_lag_cols, n_lags=2)

    qb_keep = [
        "player_id", "name", "pfr_code", "position_id", "season", "week",
        "passing_att", "passing_cmp", "passing_yds", "passing_td", "passing_int",
        "completion_pct", "yards_per_attempt", "pass_td_rate", "int_rate", "snap_pct",
        "team_pass_att",
        "completion_pct_lag1", "yards_per_attempt_lag1", "pass_td_rate_lag1",
        "int_rate_lag1", "snap_pct_lag1", "passing_att_lag1", "passing_yds_lag1",
        "completion_pct_lag2", "yards_per_attempt_lag2", "pass_td_rate_lag2",
        "int_rate_lag2", "snap_pct_lag2", "passing_att_lag2", "passing_yds_lag2",
        "dk_points",
    ]
    qbs = qbs[[c for c in qb_keep if c in qbs.columns]]

    log.info(f"  QB rows: {len(qbs):,}")

    df = pd.concat([receivers, qbs], ignore_index=True)

    out = PROCESSED / "passing_features.parquet"
    df.to_parquet(out, index=False)
    log.info(f"Saved passing_features.parquet ({len(df):,} rows, {len(df.columns)} cols)")

    return df


def build_current_week(season: int, week: int) -> pd.DataFrame:
    """
    Build projected passing/receiving inputs for the upcoming game week.

    Uses each player's most recent game as the lag-feature baseline, then
    scales projected volume using team_game_context (expected_pass_att and
    pass_def_adj from team_scoring.py).

    Players without prior passing/receiving history receive projected volume
    of 0 and are flagged with has_pass_history=False.

    Returns the projection DataFrame and saves to passing_current.parquet.
    """
    ctx_path = PROCESSED / "team_game_context.parquet"
    if not ctx_path.exists():
        log.error("team_game_context.parquet not found — run team_scoring.py first.")
        return pd.DataFrame()

    log.info(f"Building current passing features — season {season} week {week} ...")

    pass_hist = pd.read_parquet(PROCESSED / "passing_features.parquet")
    ctx = pd.read_parquet(ctx_path)

    active_teams = set(ctx["pfr_code"].unique())

    candidates = pass_hist[
        pass_hist["pfr_code"].isin(active_teams) &
        (
            ((pass_hist["season"] == season) & (pass_hist["week"] < week)) |
            (pass_hist["season"] == season - 1)
        )
    ].copy()

    # Most recent game per player
    most_recent = (
        candidates
        .sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .last()
        .reset_index()
    )

    if season in candidates["season"].values:
        max_prior_week = candidates[candidates["season"] == season]["week"].max()
    else:
        max_prior_week = 0

    most_recent["weeks_since_last_game"] = most_recent.apply(
        lambda r: (max_prior_week - r["week"]) if r["season"] == season
        else (max_prior_week + (18 - r["week"])),
        axis=1,
    )

    proj = most_recent.merge(
        ctx[["pfr_code", "expected_pass_att", "pass_def_adj",
             "implied_team_total", "opponent_pfr_code", "is_home"]],
        on="pfr_code",
        how="inner",
    )

    # ── Opponent pass defense features ────────────────────────────────────────
    # Join the target week's defense stats for each player's upcoming opponent.
    # def_lookup row (pfr_code, season, week=W) holds stats through W-1 via shift(1).
    # Drop any stale values carried forward from most_recent (historical game data).
    def_lookup = _build_pass_defense_lookup()
    opp_def_cols = ["opp_pass_yds_allowed_ytd", "opp_pass_yds_allowed_roll4"]
    proj = proj.drop(columns=[c for c in opp_def_cols if c in proj.columns])

    def_week = (
        def_lookup[(def_lookup["season"] == season) & (def_lookup["week"] == week)]
        .rename(columns={"pfr_code": "opponent_pfr_code"})
        [["opponent_pfr_code"] + opp_def_cols]
    )

    proj = proj.merge(def_week, on="opponent_pfr_code", how="left")

    for col in opp_def_cols:
        med = def_lookup[col].median()
        proj[col] = proj[col].fillna(med)

    # ── Receiver projections ──────────────────────────────────────────────────
    rec_proj = proj[proj["position_id"].isin(RECEIVER_POSITIONS)].copy()

    rec_proj["has_pass_history"] = rec_proj["target_share_lag1"] != -1
    rec_proj["proj_target_share"] = rec_proj["target_share_lag1"].clip(lower=0)

    rec_proj["projected_targets"] = (
        rec_proj["proj_target_share"] *
        rec_proj["expected_pass_att"] *
        rec_proj["pass_def_adj"]
    ).round(1)

    cr  = rec_proj["catch_rate_lag1"].clip(lower=0)
    ypt = rec_proj["yards_per_target_lag1"].clip(lower=0)
    tdr = rec_proj["rec_td_rate_lag1"].clip(lower=0)

    rec_proj["projected_rec"]     = (rec_proj["projected_targets"] * cr).round(1)
    rec_proj["projected_rec_yds"] = (rec_proj["projected_targets"] * ypt).round(1)
    rec_proj["projected_rec_td"]  = (rec_proj["projected_targets"] * tdr).round(3)

    rec_proj = rec_proj[
        (rec_proj["projected_targets"] > 0) | rec_proj["has_pass_history"]
    ].copy()

    rec_keep = [
        "player_id", "name", "pfr_code", "opponent_pfr_code", "position_id",
        "is_home", "implied_team_total",
        "target_share_lag1", "target_share_lag2",
        "catch_rate_lag1", "yards_per_target_lag1", "rec_td_rate_lag1",
        "snap_pct_lag1",
        "expected_pass_att", "pass_def_adj",
        "projected_targets", "projected_rec", "projected_rec_yds", "projected_rec_td",
        "opp_pass_yds_allowed_ytd", "opp_pass_yds_allowed_roll4",
        "has_pass_history", "weeks_since_last_game",
    ]
    rec_proj = rec_proj[[c for c in rec_keep if c in rec_proj.columns]]

    # ── QB projections ────────────────────────────────────────────────────────
    qb_proj = proj[proj["position_id"].isin(QB_POSITIONS)].copy()

    qb_proj["has_pass_history"] = qb_proj["yards_per_attempt_lag1"] != -1
    qb_proj["projected_pass_att"] = (
        qb_proj["expected_pass_att"] * qb_proj["pass_def_adj"]
    ).round(1)

    ypa  = qb_proj["yards_per_attempt_lag1"].clip(lower=0)
    tdr  = qb_proj["pass_td_rate_lag1"].clip(lower=0)
    intr = qb_proj["int_rate_lag1"].clip(lower=0)

    qb_proj["projected_pass_yds"] = (qb_proj["projected_pass_att"] * ypa).round(1)
    qb_proj["projected_pass_td"]  = (qb_proj["projected_pass_att"] * tdr).round(3)
    qb_proj["projected_pass_int"] = (qb_proj["projected_pass_att"] * intr).round(3)

    qb_proj = qb_proj[
        (qb_proj["projected_pass_att"] > 0) | qb_proj["has_pass_history"]
    ].copy()

    qb_keep = [
        "player_id", "name", "pfr_code", "opponent_pfr_code", "position_id",
        "is_home", "implied_team_total",
        "yards_per_attempt_lag1", "yards_per_attempt_lag2",
        "completion_pct_lag1", "pass_td_rate_lag1", "int_rate_lag1",
        "snap_pct_lag1",
        "expected_pass_att", "pass_def_adj",
        "projected_pass_att", "projected_pass_yds", "projected_pass_td", "projected_pass_int",
        "has_pass_history", "weeks_since_last_game",
    ]
    qb_proj = qb_proj[[c for c in qb_keep if c in qb_proj.columns]]

    # ── Combine and save ──────────────────────────────────────────────────────
    out_df = pd.concat([rec_proj, qb_proj], ignore_index=True)

    out = PROCESSED / "passing_current.parquet"
    out_df.to_parquet(out, index=False)
    log.info(f"Saved passing_current.parquet ({len(out_df):,} player-games)")

    # Readable summaries
    top_rec = rec_proj[rec_proj["projected_targets"] >= 4].sort_values(
        "projected_targets", ascending=False
    ).head(20)
    print("\n── Top Projected Receivers ─────────────────────────────────────────")
    print(top_rec[[
        "name", "position_id", "pfr_code", "opponent_pfr_code",
        "target_share_lag1", "yards_per_target_lag1",
        "expected_pass_att", "projected_targets", "projected_rec_yds",
    ]].to_string(index=False))

    top_qb = qb_proj.sort_values("projected_pass_att", ascending=False).head(10)
    print("\n── Top Projected QBs ───────────────────────────────────────────────")
    print(top_qb[[
        "name", "pfr_code", "opponent_pfr_code",
        "yards_per_attempt_lag1", "completion_pct_lag1",
        "expected_pass_att", "projected_pass_att", "projected_pass_yds",
    ]].to_string(index=False))

    return out_df


def run(season: int | None = None, week: int | None = None):
    build_historical_features()

    if season is not None and week is not None:
        build_current_week(season, week)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build player passing/receiving features.")
    parser.add_argument("--season", type=int, default=None, help="Season year for current-week projection")
    parser.add_argument("--week",   type=int, default=None, help="Week number for current-week projection")
    args = parser.parse_args()

    if (args.season is None) != (args.week is None):
        parser.error("--season and --week must be provided together.")

    run(args.season, args.week)
