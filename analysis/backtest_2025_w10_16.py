#!/usr/bin/env python3
"""
backtest_2025_w10_16.py — 2025 Weeks 10–16 Projection Backtest

Loads trained position models (data/models/model_{pos}.pkl) and generates
predictions for 2025 weeks 10–16 directly from the pre-built feature parquets,
then joins with actual dk_points from players.parquet.

Outputs (analysis/charts/):
  backtest_2025_w{week}.png  — per-week: projected vs actual + error distributions
  backtest_2025_summary.png  — per-position summary across all 7 weeks
  backtest_2025_stats.csv    — per-week/position summary statistics
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS    = ROOT / "data" / "models"
CHARTS    = Path(__file__).parent / "charts"
CHARTS.mkdir(exist_ok=True)

WEEKS     = list(range(10, 17))
POSITIONS = ["QB", "RB", "WR", "TE"]

# Feature sets (must match projections.py exactly)
FEATURES: dict[str, list[str]] = {
    "QB": [
        "yards_per_attempt_lag1",  "yards_per_attempt_lag2",
        "completion_pct_lag1",     "completion_pct_lag2",
        "pass_td_rate_lag1",       "pass_td_rate_lag2",
        "int_rate_lag1",           "int_rate_lag2",
        "passing_att_lag1",        "passing_att_lag2",
        "passing_yds_lag1",
        "snap_pct_lag1",
        "rushing_att_lag1",        "rushing_yds_lag1",
        "implied_team_total",
    ],
    "RB": [
        "carry_share_lag1",        "carry_share_lag2",
        "yards_per_carry_lag1",    "yards_per_carry_lag2",
        "rush_td_rate_lag1",       "rush_td_rate_lag2",
        "rushing_att_lag1",        "rushing_att_lag2",
        "rushing_yds_lag1",
        "snap_pct_lag1",
        "target_share_lag1",       "target_share_lag2",
        "catch_rate_lag1",
        "yards_per_target_lag1",
        "rec_td_rate_lag1",
        "receiving_tar_lag1",
        "implied_team_total",
    ],
    "WR": [
        "target_share_lag1",       "target_share_lag2",
        "catch_rate_lag1",         "catch_rate_lag2",
        "yards_per_target_lag1",   "yards_per_target_lag2",
        "rec_td_rate_lag1",        "rec_td_rate_lag2",
        "receiving_tar_lag1",      "receiving_tar_lag2",
        "receiving_yds_lag1",
        "snap_pct_lag1",
        "implied_team_total",
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
        "opp_pass_yds_allowed_ytd",
        "opp_pass_yds_allowed_roll4",
    ],
}

PRIMARY_LAG: dict[str, str] = {
    "QB": "yards_per_attempt_lag1",
    "RB": "carry_share_lag1",
    "WR": "target_share_lag1",
    "TE": "target_share_lag1",
}

# ── Style ──────────────────────────────────────────────────────────────────────

COLORS = {
    "projected": "#2563EB",   # blue
    "actual":    "#DC2626",   # red
    "error":     "#7C3AED",   # purple
    "zero":      "#6B7280",   # grey
}
POS_COLORS = {"QB": "#F59E0B", "RB": "#10B981", "WR": "#3B82F6", "TE": "#EF4444"}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F9FAFB",
    "axes.edgecolor":    "#D1D5DB",
    "axes.labelcolor":   "#374151",
    "xtick.color":       "#374151",
    "ytick.color":       "#374151",
    "text.color":        "#374151",
    "font.family":       "sans-serif",
    "axes.grid":         True,
    "grid.color":        "#E5E7EB",
    "grid.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_models() -> dict:
    models = {}
    for pos in POSITIONS:
        path = MODELS / f"model_{pos}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        models[pos] = obj["model"]
        print(f"  Loaded model_{pos}.pkl")
    return models


def build_sched_ctx() -> pd.DataFrame:
    """Schedule → one row per team per game with implied_team_total."""
    sched = pd.read_parquet(PROCESSED / "schedule.parquet")
    home = sched[["season", "week", "home_pfr_code", "spread_line", "total_line"]].copy()
    home = home.rename(columns={"home_pfr_code": "pfr_code"})
    home["implied_team_total"] = (home["total_line"] + home["spread_line"]) / 2
    away = sched[["season", "week", "away_pfr_code", "spread_line", "total_line"]].copy()
    away = away.rename(columns={"away_pfr_code": "pfr_code"})
    away["implied_team_total"] = (away["total_line"] - away["spread_line"]) / 2
    ctx = pd.concat([home, away], ignore_index=True)
    return ctx[["pfr_code", "season", "week", "implied_team_total"]].dropna(
        subset=["implied_team_total"]
    )


def load_actuals() -> pd.DataFrame:
    """Load actual dk_points for 2025 w10-16."""
    return pd.read_parquet(
        PROCESSED / "players.parquet",
        columns=["player_id", "season", "week", "dk_points"],
    )


def align_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    return df[features].copy()


# ── Prediction for one week ────────────────────────────────────────────────────

def predict_week(
    week: int,
    models: dict,
    rush_all: pd.DataFrame,
    pass_all: pd.DataFrame,
    sched_ctx: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions for all positions for a single week; join actuals."""
    sched_wk = sched_ctx[(sched_ctx["season"] == 2025) & (sched_ctx["week"] == week)]
    med_total = sched_ctx["implied_team_total"].median()

    rush_wk = rush_all[(rush_all["season"] == 2025) & (rush_all["week"] == week)]
    pass_wk = pass_all[(pass_all["season"] == 2025) & (pass_all["week"] == week)]
    act_wk  = actuals[(actuals["season"] == 2025) & (actuals["week"] == week)]

    rows = []
    for pos in POSITIONS:
        model    = models[pos]
        features = FEATURES[pos]
        primary  = PRIMARY_LAG[pos]

        # Assemble feature frame
        if pos == "QB":
            df = pass_wk[pass_wk["position_id"] == "QB"].copy()
            rush_qb = rush_wk[rush_wk["position_id"] == "QB"][
                ["player_id", "rushing_att_lag1", "rushing_yds_lag1"]
            ]
            if not rush_qb.empty:
                df = df.merge(rush_qb, on="player_id", how="left")

        elif pos == "RB":
            df = rush_wk[rush_wk["position_id"] == "RB"].copy()
            recv_cols = [
                "player_id",
                "target_share_lag1", "target_share_lag2",
                "catch_rate_lag1",   "yards_per_target_lag1",
                "rec_td_rate_lag1",  "receiving_tar_lag1",
            ]
            avail = [c for c in recv_cols if c in pass_wk.columns]
            rb_recv = pass_wk[pass_wk["position_id"] == "RB"][avail]
            if not rb_recv.empty:
                df = df.merge(rb_recv, on="player_id", how="left")

        else:  # WR, TE
            df = pass_wk[pass_wk["position_id"] == pos].copy()

        if df.empty:
            continue

        # Join implied team total
        df = df.merge(sched_wk[["pfr_code", "implied_team_total"]],
                      on="pfr_code", how="left")
        df["implied_team_total"] = df["implied_team_total"].fillna(med_total)

        # Drop first-ever-game rows (no lag history)
        if primary in df.columns:
            df = df[df[primary] != -1].copy()

        if df.empty:
            continue

        X = align_features(df.copy(), features).fillna(0).replace(-1, 0)
        preds = np.clip(model.predict(X), 0, None)

        out = pd.DataFrame({
            "player_id":    df["player_id"].values,
            "name":         df["name"].values,
            "position_id":  pos,
            "week":         week,
            "predicted_dk": np.round(preds, 2),
        })
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)

    # Join actuals
    result = result.merge(
        act_wk[["player_id", "dk_points"]],
        on="player_id", how="left",
    )
    result["error"] = result["predicted_dk"] - result["dk_points"]
    return result


# ── Charting helpers ───────────────────────────────────────────────────────────

def kde_plot(ax, data, color, label, bw=0.3):
    """Plot a KDE on ax. data must have ≥2 non-NaN points."""
    data = data.dropna()
    if len(data) < 3:
        return
    kde = gaussian_kde(data, bw_method=bw)
    x   = np.linspace(data.min() - 2, data.max() + 2, 300)
    ax.fill_between(x, kde(x), alpha=0.25, color=color)
    ax.plot(x, kde(x), color=color, lw=1.8, label=label)
    ax.axvline(data.median(), color=color, lw=1.2, ls="--", alpha=0.8)


def stat_box(ax, pred, actual, err):
    """Add a small stats annotation inside the axis."""
    pred   = pred.dropna()
    actual = actual.dropna()
    err    = err.dropna()
    if len(pred) < 3 or len(actual) < 3:
        return
    mae  = mean_absolute_error(actual, pred[:len(actual)])
    bias = float(err.mean())
    r, _ = pearsonr(actual.values[:len(pred)], pred.values[:len(actual)])
    txt  = f"MAE={mae:.1f}  bias={bias:+.1f}  r={r:.2f}  n={len(pred)}"
    ax.text(
        0.98, 0.97, txt,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=6.5, color="#374151",
        bbox=dict(facecolor="white", edgecolor="#D1D5DB", alpha=0.8, pad=2),
    )


# ── Per-week figure ────────────────────────────────────────────────────────────

def plot_week(week: int, df: pd.DataFrame) -> None:
    """
    2 rows × 4 columns per week.
    Row 0: projected (blue) vs actual (red) KDE, with dashed medians.
    Row 1: error distribution (projected − actual), zero line.
    """
    fig, axes = plt.subplots(
        2, 4, figsize=(18, 7),
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    fig.suptitle(
        f"2025  Week {week} — Projection Distributions by Position",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for col, pos in enumerate(POSITIONS):
        sub = df[df["position_id"] == pos]
        pred   = sub["predicted_dk"]
        actual = sub["dk_points"]
        err    = sub["error"]

        # ── Row 0: projected vs actual KDE ─────────────────────────────────
        ax0 = axes[0, col]
        kde_plot(ax0, pred,   COLORS["projected"], "Projected")
        kde_plot(ax0, actual, COLORS["actual"],    "Actual")
        ax0.set_title(pos, fontsize=12, fontweight="bold",
                      color=POS_COLORS[pos], pad=4)
        ax0.set_xlabel("DK Points", fontsize=8)
        ax0.set_ylabel("Density", fontsize=8)
        ax0.tick_params(labelsize=7)
        ax0.legend(fontsize=7, framealpha=0.9)
        stat_box(ax0, pred, actual, err)

        # ── Row 1: error distribution ───────────────────────────────────────
        ax1 = axes[1, col]
        kde_plot(ax1, err, COLORS["error"], "Error (pred−act)")
        ax1.axvline(0, color=COLORS["zero"], lw=1.2, ls="-", alpha=0.7, label="Zero")
        ax1.axvline(float(err.median()), color=COLORS["error"],
                    lw=1.2, ls="--", alpha=0.8)
        ax1.set_xlabel("Error (DK pts)", fontsize=8)
        ax1.set_ylabel("Density", fontsize=8)
        ax1.tick_params(labelsize=7)
        ax1.legend(fontsize=7, framealpha=0.9)

        n_pos = sub["dk_points"].notna().sum()
        ax1.set_title(
            f"n={n_pos}  median err={float(err.median()):+.1f}",
            fontsize=8, color="#6B7280", pad=3,
        )

    out = CHARTS / f"backtest_2025_w{week:02d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Summary figure (all weeks, per position) ───────────────────────────────────

def plot_summary(all_df: pd.DataFrame) -> None:
    """
    4 rows (positions) × 2 columns.
    Left:  per-week box plot of projected vs actual DK pts.
    Right: per-week box plot of error (projected − actual).
    """
    fig, axes = plt.subplots(
        4, 2, figsize=(16, 20),
        gridspec_kw={"hspace": 0.55, "wspace": 0.30},
    )
    fig.suptitle(
        "2025 Weeks 10–16 — Projection Summary by Position",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for row, pos in enumerate(POSITIONS):
        sub = all_df[all_df["position_id"] == pos].copy()
        col_color = POS_COLORS[pos]

        # ── Left: side-by-side box per week ────────────────────────────────
        ax_l = axes[row, 0]
        week_labels = [f"W{w}" for w in WEEKS]
        pred_data   = [sub[sub["week"] == w]["predicted_dk"].dropna().values for w in WEEKS]
        act_data    = [sub[sub["week"] == w]["dk_points"].dropna().values   for w in WEEKS]

        positions_x = np.arange(len(WEEKS))
        offset = 0.22

        bp_pred = ax_l.boxplot(
            pred_data, positions=positions_x - offset,
            widths=0.38, patch_artist=True,
            medianprops=dict(color="white", lw=1.8),
            whiskerprops=dict(color=COLORS["projected"], lw=1.2),
            capprops=dict(color=COLORS["projected"], lw=1.2),
            flierprops=dict(marker=".", markersize=2, color=COLORS["projected"], alpha=0.4),
            boxprops=dict(facecolor=COLORS["projected"], alpha=0.7,
                          edgecolor=COLORS["projected"]),
        )
        bp_act = ax_l.boxplot(
            act_data, positions=positions_x + offset,
            widths=0.38, patch_artist=True,
            medianprops=dict(color="white", lw=1.8),
            whiskerprops=dict(color=COLORS["actual"], lw=1.2),
            capprops=dict(color=COLORS["actual"], lw=1.2),
            flierprops=dict(marker=".", markersize=2, color=COLORS["actual"], alpha=0.4),
            boxprops=dict(facecolor=COLORS["actual"], alpha=0.7,
                          edgecolor=COLORS["actual"]),
        )

        ax_l.set_xticks(positions_x)
        ax_l.set_xticklabels(week_labels, fontsize=9)
        ax_l.set_ylabel("DK Points", fontsize=9)
        ax_l.set_title(f"{pos} — Projected vs Actual", fontsize=11,
                       fontweight="bold", color=col_color)
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor=COLORS["projected"], alpha=0.7, label="Projected"),
            Patch(facecolor=COLORS["actual"],    alpha=0.7, label="Actual"),
        ]
        ax_l.legend(handles=legend_elems, fontsize=8, loc="upper right")
        ax_l.tick_params(axis="y", labelsize=8)

        # ── Right: error box per week ───────────────────────────────────────
        ax_r = axes[row, 1]
        err_data = [sub[sub["week"] == w]["error"].dropna().values for w in WEEKS]

        bp_err = ax_r.boxplot(
            err_data, positions=positions_x,
            widths=0.55, patch_artist=True,
            medianprops=dict(color="white", lw=1.8),
            whiskerprops=dict(color=COLORS["error"], lw=1.2),
            capprops=dict(color=COLORS["error"], lw=1.2),
            flierprops=dict(marker=".", markersize=2, color=COLORS["error"], alpha=0.4),
            boxprops=dict(facecolor=COLORS["error"], alpha=0.65,
                          edgecolor=COLORS["error"]),
        )
        ax_r.axhline(0, color=COLORS["zero"], lw=1.3, ls="-", alpha=0.8)

        ax_r.set_xticks(positions_x)
        ax_r.set_xticklabels(week_labels, fontsize=9)
        ax_r.set_ylabel("Error: Projected − Actual (DK pts)", fontsize=9)
        ax_r.set_title(f"{pos} — Projection Error per Week", fontsize=11,
                       fontweight="bold", color=col_color)
        ax_r.tick_params(axis="y", labelsize=8)

    out = CHARTS / "backtest_2025_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Stats CSV ──────────────────────────────────────────────────────────────────

def build_stats_csv(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for week in WEEKS:
        for pos in POSITIONS:
            sub = all_df[
                (all_df["week"] == week) & (all_df["position_id"] == pos)
            ].dropna(subset=["dk_points", "predicted_dk"])

            if len(sub) < 3:
                continue

            pred   = sub["predicted_dk"]
            actual = sub["dk_points"]
            err    = sub["error"]
            r, _   = pearsonr(actual, pred)

            rows.append({
                "week":           week,
                "position":       pos,
                "n":              len(sub),
                "mae":            round(mean_absolute_error(actual, pred), 2),
                "rmse":           round(mean_squared_error(actual, pred) ** 0.5, 2),
                "pearson_r":      round(r, 3),
                "bias":           round(float(err.mean()), 2),
                "median_error":   round(float(err.median()), 2),
                "mean_predicted": round(float(pred.mean()), 2),
                "mean_actual":    round(float(actual.mean()), 2),
            })

    df = pd.DataFrame(rows)
    out = CHARTS / "backtest_2025_stats.csv"
    df.to_csv(out, index=False)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Loading data ===")
    models    = load_models()
    sched_ctx = build_sched_ctx()
    actuals   = load_actuals()

    print("\n=== Loading feature parquets ===")
    rush_all = pd.read_parquet(PROCESSED / "rushing_features.parquet")
    pass_all = pd.read_parquet(PROCESSED / "passing_features.parquet")
    print(f"  Rush: {len(rush_all):,} rows  |  Pass: {len(pass_all):,} rows")

    print("\n=== Generating predictions for 2025 w10-16 ===")
    all_results = []
    for week in WEEKS:
        df = predict_week(week, models, rush_all, pass_all, sched_ctx, actuals)
        n_pred   = df["predicted_dk"].notna().sum()
        n_actual = df["dk_points"].notna().sum()
        print(f"  Week {week}: {n_pred} predictions, {n_actual} actual scores joined")
        all_results.append(df)

    all_df = pd.concat(all_results, ignore_index=True)

    print("\n=== Summary statistics ===")
    stats = build_stats_csv(all_df)
    print(stats.to_string(index=False))

    print("\n=== Generating per-week charts ===")
    for week in WEEKS:
        plot_week(week, all_df[all_df["week"] == week])

    print("\n=== Generating summary chart ===")
    plot_summary(all_df)

    print(f"\nAll outputs saved to {CHARTS}/")


if __name__ == "__main__":
    main()
