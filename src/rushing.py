#!/usr/bin/env python3
"""
rushing.py — Player Rushing Feature Module

Builds per-player rushing features for both:

  1. Historical training dataset (all seasons 2000–present)
     Output: data/processed/rushing_features.parquet

  2. Current-week projection inputs (one upcoming game week)
     Output: data/processed/rushing_current.parquet
     Requires: team_game_context.parquet (run team_scoring.py first)

Features computed per player-week:
  carry_share        — player rushing_att / team carries
  yards_per_carry    — rushing_yds / rushing_att
  rush_td_rate       — rushing_td / rushing_att
  snap_pct           — offensive_snapcount_percentage / 100

Lag features (1-week and 2-week prior):
  *_lag1, *_lag2     — prior-game values per player (grouped by player_id)
                       Sentinel value -1 indicates no prior history.

Projection columns (current week only):
  projected_rush_att — carry_share_lag1 × expected_rush_att × rush_def_adj
  projected_rush_yds — projected_rush_att × ypc_lag1
  projected_rush_td  — projected_rush_att × td_rate_lag1

Usage:
  python3 src/rushing.py                           # historical features only
  python3 src/rushing.py --season 2024 --week 14   # historical + current week
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

RUSH_POSITIONS = {"QB", "RB", "WR"}   # TE rarely rush; exclude to keep clean
MIN_ATT_FOR_RATE = 1                   # minimum attempts to compute per-carry rates
MIN_SEASON = 2009                      # exclude pre-2009 data (different game era, data gaps)


# ── Feature computation ────────────────────────────────────────────────────

def _per_play_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute carry_share, yards_per_carry, rush_td_rate, snap_pct
    for each player-week row.

    Rows with zero team carries or zero player attempts produce NaN
    for rate metrics, which are later filled with 0 before lagging.
    """
    df = df.copy()

    # Carry share: what fraction of team rushes went to this player
    df["carry_share"] = np.where(
        df["team_carries"] > 0,
        df["rushing_att"] / df["team_carries"],
        np.nan,
    )

    # Efficiency metrics — only meaningful when the player actually rushed
    df["yards_per_carry"] = np.where(
        df["rushing_att"] >= MIN_ATT_FOR_RATE,
        df["rushing_yds"] / df["rushing_att"],
        np.nan,
    )

    df["rush_td_rate"] = np.where(
        df["rushing_att"] >= MIN_ATT_FOR_RATE,
        df["rushing_td"] / df["rushing_att"],
        np.nan,
    )

    # Snap percentage normalised to 0–1
    df["snap_pct"] = df["offensive_snapcount_percentage"].fillna(0) / 100.0

    return df


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

    # Fill NaN metrics with 0 before lagging (a week where a player had 0 carries
    # is genuine information — distinguish from -1 "no history")
    for col in lag_cols:
        df[col] = df[col].fillna(0)

    for lag in range(1, n_lags + 1):
        shifted = df.groupby("player_id")[lag_cols].shift(lag)
        for col in lag_cols:
            df[f"{col}_lag{lag}"] = shifted[col].fillna(-1)

    return df


# ── Main builders ──────────────────────────────────────────────────────────

def build_historical_features() -> pd.DataFrame:
    """
    Build the full historical rushing feature dataset covering all seasons.

    Steps:
      1. Load players and team_stats parquets.
      2. Filter to rushing-relevant positions with at least some activity.
      3. Merge team carries onto each player-week row.
      4. Compute per-play metrics.
      5. Add 1- and 2-week lag features per player.

    Returns the feature DataFrame and saves to rushing_features.parquet.
    """
    log.info("Building historical rushing features ...")

    players = pd.read_parquet(PROCESSED / "players.parquet")
    team_stats = pd.read_parquet(PROCESSED / "team_stats.parquet")

    # Filter to positions that rush meaningfully; restrict to modern era
    rushers = players[
        players["position_id"].isin(RUSH_POSITIONS) &
        (players["season"] >= MIN_SEASON)
    ].copy()
    log.info(f"  {len(rushers):,} player-week rows for positions {RUSH_POSITIONS} (seasons {MIN_SEASON}+)")

    # Pull team carries denominator
    team_carries = team_stats[["pfr_code", "season", "week", "carries"]].rename(
        columns={"carries": "team_carries"}
    )

    # Merge — left join so players without a team_stats row get NaN carries
    df = rushers.merge(team_carries, on=["pfr_code", "season", "week"], how="left")

    unmatched = df["team_carries"].isnull().sum()
    if unmatched > 0:
        log.warning(f"  {unmatched} rows missing team carries after merge")

    df["rushing_att"] = df["rushing_att"].fillna(0)
    df["rushing_yds"] = df["rushing_yds"].fillna(0)
    df["rushing_td"]  = df["rushing_td"].fillna(0)

    df = _per_play_metrics(df)

    lag_cols = ["carry_share", "yards_per_carry", "rush_td_rate", "snap_pct",
                "rushing_att", "rushing_yds"]
    df = _add_lag_features(df, lag_cols, n_lags=2)

    # Keep only the columns needed downstream
    keep = [
        "player_id", "name", "pfr_code", "position_id", "season", "week",
        "rushing_att", "rushing_yds", "rushing_td",
        "carry_share", "yards_per_carry", "rush_td_rate", "snap_pct",
        "team_carries",
        "carry_share_lag1", "yards_per_carry_lag1", "rush_td_rate_lag1",
        "snap_pct_lag1", "rushing_att_lag1", "rushing_yds_lag1",
        "carry_share_lag2", "yards_per_carry_lag2", "rush_td_rate_lag2",
        "snap_pct_lag2", "rushing_att_lag2", "rushing_yds_lag2",
        "dk_points",
    ]
    df = df[[c for c in keep if c in df.columns]]

    out = PROCESSED / "rushing_features.parquet"
    df.to_parquet(out, index=False)
    log.info(f"Saved rushing_features.parquet ({len(df):,} rows, {len(df.columns)} cols)")

    return df


def build_current_week(season: int, week: int) -> pd.DataFrame:
    """
    Build projected rushing inputs for the upcoming game week.

    Uses each player's most recent two games as the lag-feature baseline,
    then scales projected attempts using team_game_context (expected_rush_att
    and rush_def_adj from team_scoring.py).

    Players without at least one prior game of rushing history receive
    a projected_rush_att of 0 and are flagged with has_rush_history=False.

    Returns the projection DataFrame and saves to rushing_current.parquet.
    """
    ctx_path = PROCESSED / "team_game_context.parquet"
    if not ctx_path.exists():
        log.error("team_game_context.parquet not found — run team_scoring.py first.")
        return pd.DataFrame()

    log.info(f"Building current rushing features — season {season} week {week} ...")

    rush_hist = pd.read_parquet(PROCESSED / "rushing_features.parquet")
    ctx = pd.read_parquet(ctx_path)

    # Teams playing this week
    active_teams = set(ctx["pfr_code"].unique())

    # For each player, find their most recent game BEFORE this week
    # Use up to the last 4 weeks to confirm they're still on the roster
    recent_cutoff_weeks = 4
    candidates = rush_hist[
        rush_hist["pfr_code"].isin(active_teams) &
        (
            (rush_hist["season"] == season) & (rush_hist["week"] < week) |
            (rush_hist["season"] == season - 1)
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

    # Flag players who haven't appeared in the last `recent_cutoff_weeks` weeks
    # (likely inactive / injured / off roster)
    if season in candidates["season"].values:
        max_prior_week = candidates[candidates["season"] == season]["week"].max()
    else:
        max_prior_week = 0

    most_recent["weeks_since_last_game"] = most_recent.apply(
        lambda r: (max_prior_week - r["week"]) if r["season"] == season
        else (max_prior_week + (18 - r["week"])),
        axis=1,
    )
    most_recent["has_rush_history"] = most_recent["carry_share_lag1"] != -1

    # Merge game context (expected volume + defensive adjustments)
    proj = most_recent.merge(
        ctx[["pfr_code", "expected_rush_att", "rush_def_adj", "implied_team_total",
             "opponent_pfr_code", "is_home"]],
        on="pfr_code",
        how="inner",
    )

    # Projected rushing attempts
    # Use lag1 carry share as the best single estimate of usage
    proj["proj_carry_share"] = proj["carry_share_lag1"].clip(lower=0)

    proj["projected_rush_att"] = (
        proj["proj_carry_share"] *
        proj["expected_rush_att"] *
        proj["rush_def_adj"]
    ).round(1)

    # Projected yards and TDs using lag1 efficiency
    ypc = proj["yards_per_carry_lag1"].clip(lower=0)
    tdr = proj["rush_td_rate_lag1"].clip(lower=0)

    proj["projected_rush_yds"] = (proj["projected_rush_att"] * ypc).round(1)
    proj["projected_rush_td"]  = (proj["projected_rush_att"] * tdr).round(3)

    # Remove players with no projected carries and no rushing history
    proj = proj[
        (proj["projected_rush_att"] > 0) | (proj["has_rush_history"])
    ].copy()

    keep = [
        "player_id", "name", "pfr_code", "opponent_pfr_code", "position_id",
        "is_home", "implied_team_total",
        "carry_share_lag1", "carry_share_lag2",
        "yards_per_carry_lag1", "yards_per_carry_lag2",
        "rush_td_rate_lag1", "rush_td_rate_lag2",
        "rushing_att_lag1", "rushing_att_lag2",
        "rushing_yds_lag1", "rushing_yds_lag2",
        "snap_pct_lag1",
        "expected_rush_att", "rush_def_adj",
        "projected_rush_att", "projected_rush_yds", "projected_rush_td",
        "has_rush_history", "weeks_since_last_game",
    ]
    proj = proj[[c for c in keep if c in proj.columns]]

    out = PROCESSED / "rushing_current.parquet"
    proj.to_parquet(out, index=False)
    log.info(f"Saved rushing_current.parquet ({len(proj):,} player-games)")

    # Print a readable summary — top projected rushers
    summary = proj[proj["projected_rush_att"] >= 5].sort_values(
        "projected_rush_att", ascending=False
    ).head(20)
    print("\n── Top Projected Rushers ───────────────────────────────────────────")
    print(summary[[
        "name", "position_id", "pfr_code", "opponent_pfr_code",
        "carry_share_lag1", "yards_per_carry_lag1",
        "expected_rush_att", "projected_rush_att", "projected_rush_yds"
    ]].to_string(index=False))

    return proj


def run(season: int | None = None, week: int | None = None):
    build_historical_features()

    if season is not None and week is not None:
        build_current_week(season, week)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build player rushing features.")
    parser.add_argument("--season", type=int, default=None, help="Season year for current-week projection")
    parser.add_argument("--week",   type=int, default=None, help="Week number for current-week projection")
    args = parser.parse_args()

    if (args.season is None) != (args.week is None):
        parser.error("--season and --week must be provided together.")

    run(args.season, args.week)
