#!/usr/bin/env python3
"""
team_scoring.py — Team Scoring Context Module

For a given (season, week), derives the game-level context that all player
projection modules depend on:

  - implied_team_total     : expected points scored, from Vegas O/U + spread
  - expected_pass_att      : projected pass attempts, from pace + implied total
  - expected_rush_att      : projected rush attempts, from pace + implied total
  - pass_def_adj           : opponent pass defense multiplier (1.0 = league avg)
  - rush_def_adj           : opponent rush defense multiplier (1.0 = league avg)
  - game_script_factor     : pass-rate bias from spread (favorites run more)
  - rest_days              : days of rest heading into the game

Spread sign convention (team_schedule.csv spread_line):
  Positive = home team is FAVORED (gives points)   ← opposite of standard ATS notation
  Negative = home team is UNDERDOG (gets points)

  home_implied = (total_line + spread_line) / 2
  away_implied = (total_line - spread_line) / 2

  Example: PHI home, spread = +14, O/U = 44.5
    → home_implied = (44.5 + 14) / 2 = 29.25  ✓
    → away_implied = (44.5 - 14) / 2 = 15.25  ✓

Rolling features are pulled from week W-1 to avoid any data leakage into
the upcoming game being predicted.

Output: data/processed/team_game_context.parquet
"""

import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"

# ── Thresholds for game-script pass-rate adjustment ────────────────────────
# Large favorites tend to run more (protect lead); large underdogs pass more.
# Dataset convention: positive spread_line = home team is FAVORED.
BIG_FAVORITE_THRESHOLD = 7.0    # home spread ABOVE this → home team runs more
BIG_UNDERDOG_THRESHOLD = 7.0    # home spread BELOW -this → home team passes more
GAME_SCRIPT_ADJUSTMENT = 0.07   # fraction of pass plays shifted to/from run

# ── Default implied total when Vegas line is missing ──────────────────────
LEAGUE_AVG_SCORE = 23.0          # approximate per-team average (2015–2024)


# ── Helpers ────────────────────────────────────────────────────────────────

def _implied_totals(spread_line: float, total_line: float) -> tuple[float, float]:
    """
    Derive per-team implied totals from Vegas spread and over/under.

    Dataset convention: positive spread = home team is FAVORED.
    Returns (home_implied, away_implied).
    """
    home = (total_line + spread_line) / 2
    away = (total_line - spread_line) / 2
    return home, away


def _game_script_factor(spread_line: float, is_home: bool) -> float:
    """
    Return a pass-play fraction adjustment for a team based on spread.

    Positive factor = lean toward more passing (underdog mode).
    Negative factor = lean toward more running (favorite mode).

    A factor of +0.07 means shift 7% of plays from run to pass.

    Dataset convention: positive spread_line = home team favored.
    """
    # From the home team's perspective: large positive = big favorite → run more
    if is_home:
        if spread_line > BIG_FAVORITE_THRESHOLD:
            return -GAME_SCRIPT_ADJUSTMENT    # home is big favorite → run more
        elif spread_line < -BIG_UNDERDOG_THRESHOLD:
            return +GAME_SCRIPT_ADJUSTMENT    # home is big underdog → pass more
    else:
        # Away team: negate the spread (their perspective is flipped)
        effective_spread = -spread_line
        if effective_spread > BIG_FAVORITE_THRESHOLD:
            return -GAME_SCRIPT_ADJUSTMENT
        elif effective_spread < -BIG_UNDERDOG_THRESHOLD:
            return +GAME_SCRIPT_ADJUSTMENT
    return 0.0


def _get_prior_rolling(rolling: pd.DataFrame, pfr_code: str,
                       season: int, week: int) -> pd.Series | None:
    """
    Return the rolling feature row for a team from their most recent game
    strictly before (season, week).

    Falls back to any available row within the same season if week-1 is missing
    (e.g. early-season gaps), then to the prior season's final week.
    Returns None if no data is available.
    """
    team_df = rolling[rolling["pfr_code"] == pfr_code].copy()

    # Most recent game before the target week in this season
    prior = team_df[
        (team_df["season"] == season) & (team_df["week"] < week)
    ].sort_values("week")

    if not prior.empty:
        return prior.iloc[-1]

    # Fall back: last game of the prior season
    prev_season = team_df[team_df["season"] == season - 1].sort_values("week")
    if not prev_season.empty:
        log.debug(f"  {pfr_code} week {week}: using prior season fallback")
        return prev_season.iloc[-1]

    log.warning(f"  No rolling data found for {pfr_code} before season {season} week {week}")
    return None


def _defensive_adjustment(opp_rolling: pd.Series | None,
                           league_pass_def_avg: float,
                           league_rush_def_avg: float) -> tuple[float, float]:
    """
    Compute pass and rush defensive adjustment multipliers for an opponent.

    Returns (pass_def_adj, rush_def_adj) where 1.0 = league average.
    Values < 1.0 mean the defense suppresses that type of production.

    Uses:
      def_pressure_rate_3_game_avg  → suppresses passing (positive = harder to pass)
      def_sacks_3_game_avg          → suppresses passing
      def_pass_defended_3_game_avg  → suppresses receiving
      def_tackles_for_loss_3_game_avg → suppresses rushing
    """
    if opp_rolling is None:
        return 1.0, 1.0

    def _safe(col: str, default: float = 0.0) -> float:
        val = opp_rolling.get(col, default)
        return default if (val is None or val == -1 or pd.isna(val)) else float(val)

    pressure = _safe("def_pressure_rate_3_game_avg", league_pass_def_avg)
    sacks    = _safe("def_sacks_3_game_avg", 2.5)
    pass_def = _safe("def_pass_defended_3_game_avg", 6.0)
    tfl      = _safe("def_tackles_for_loss_3_game_avg", 3.5)

    # Normalise each metric to a 0-centred score (positive = above avg = harder for offense)
    pressure_z = (pressure - league_pass_def_avg) / max(league_pass_def_avg * 0.3, 0.01)
    sacks_z    = (sacks - 2.5) / 1.0
    pass_def_z = (pass_def - 6.0) / 2.0
    tfl_z      = (tfl - 3.5) / 1.2

    # Pass adjustment: blend pressure and coverage signals; cap at ±20%
    pass_composite = (pressure_z * 0.5 + sacks_z * 0.25 + pass_def_z * 0.25)
    pass_def_adj   = np.clip(1.0 - pass_composite * 0.10, 0.80, 1.20)

    # Rush adjustment: driven by tackles-for-loss; cap at ±15%
    rush_def_adj = np.clip(1.0 - tfl_z * 0.10, 0.85, 1.15)

    return float(pass_def_adj), float(rush_def_adj)


def _expected_plays(team_rolling: pd.Series | None,
                    implied_total: float,
                    game_script_factor: float) -> tuple[float, float]:
    """
    Estimate expected pass attempts and rush attempts for a team.

    Method:
      1. Pull baseline pass/rush pace from rolling 3-game averages.
      2. Scale total play volume proportionally to implied team total vs.
         the team's own historical scoring average.
      3. Apply game-script factor to shift the pass/rush split.

    Returns (expected_pass_att, expected_rush_att).
    """
    if team_rolling is None:
        # No history: use league-average baseline
        return 35.0, 25.0

    def _safe(col: str, default: float) -> float:
        val = team_rolling.get(col, default)
        return default if (val is None or val == -1 or pd.isna(val)) else float(val)

    pass_att_avg  = _safe("attempts_3_game_avg", 35.0)
    rush_att_avg  = _safe("carries_3_game_avg", 25.0)
    total_plays   = pass_att_avg + rush_att_avg

    # Historical pass rate for this team
    base_pass_rate = pass_att_avg / total_plays if total_plays > 0 else 0.58

    # Scale total plays to implied team scoring
    # Empirical baseline: ~60 plays → ~23 pts (LEAGUE_AVG_SCORE)
    scaling_ratio  = implied_total / LEAGUE_AVG_SCORE
    scaled_plays   = total_plays * scaling_ratio

    # Apply game-script factor (shifts pass/rush split)
    adjusted_pass_rate = np.clip(base_pass_rate + game_script_factor, 0.40, 0.75)

    expected_pass = scaled_plays * adjusted_pass_rate
    expected_rush = scaled_plays * (1.0 - adjusted_pass_rate)

    return float(expected_pass), float(expected_rush)


# ── Main builder ───────────────────────────────────────────────────────────

def build_team_game_context(season: int, week: int) -> pd.DataFrame:
    """
    Build team-level game context for every team playing in (season, week).

    Returns a DataFrame with one row per team per game.
    """
    schedule = pd.read_parquet(PROCESSED / "schedule.parquet")
    rolling  = pd.read_parquet(PROCESSED / "team_rolling.parquet")

    # Current week's matchups
    week_games = schedule[
        (schedule["season"] == season) & (schedule["week"] == week)
    ].copy()

    if week_games.empty:
        log.warning(f"No schedule data found for season {season} week {week}")
        return pd.DataFrame()

    log.info(f"Building context for {len(week_games)} games — season {season} week {week}")

    # League-average defensive pressure rate (for normalisation in adj function)
    valid_pressure = rolling[rolling["def_pressure_rate_3_game_avg"] != -1][
        "def_pressure_rate_3_game_avg"
    ]
    league_pass_def_avg = float(valid_pressure.mean()) if not valid_pressure.empty else 0.10
    league_rush_def_avg = 3.5  # approximate league avg TFL rate

    rows = []

    for _, game in week_games.iterrows():
        spread     = game.get("spread_line", np.nan)
        total      = game.get("total_line", np.nan)
        home_code  = game.get("home_pfr_code")
        away_code  = game.get("away_pfr_code")
        home_rest  = game.get("home_rest", 7)
        away_rest  = game.get("away_rest", 7)

        # Handle missing Vegas lines gracefully
        if pd.isna(spread) or pd.isna(total):
            log.warning(f"  Missing Vegas line for {home_code} vs {away_code} — using league averages")
            home_implied = LEAGUE_AVG_SCORE
            away_implied = LEAGUE_AVG_SCORE
            spread = 0.0
        else:
            home_implied, away_implied = _implied_totals(spread, total)

        # Fetch rolling features for both teams (from prior week)
        home_roll = _get_prior_rolling(rolling, home_code, season, week)
        away_roll = _get_prior_rolling(rolling, away_code, season, week)

        # Game script factors
        home_gs = _game_script_factor(spread, is_home=True)
        away_gs = _game_script_factor(spread, is_home=False)

        # Expected plays
        home_pass, home_rush = _expected_plays(home_roll, home_implied, home_gs)
        away_pass, away_rush = _expected_plays(away_roll, away_implied, away_gs)

        # Defensive adjustments (opponent's defense affects the team's production)
        home_pass_adj, home_rush_adj = _defensive_adjustment(
            away_roll, league_pass_def_avg, league_rush_def_avg
        )
        away_pass_adj, away_rush_adj = _defensive_adjustment(
            home_roll, league_pass_def_avg, league_rush_def_avg
        )

        for (pfr_code, opp_code, implied, exp_pass, exp_rush,
             pass_adj, rush_adj, gs_factor, rest, is_home) in [
            (home_code, away_code, home_implied, home_pass, home_rush,
             home_pass_adj, home_rush_adj, home_gs, home_rest, True),
            (away_code, home_code, away_implied, away_pass, away_rush,
             away_pass_adj, away_rush_adj, away_gs, away_rest, False),
        ]:
            rows.append({
                "season":              season,
                "week":                week,
                "pfr_code":            pfr_code,
                "opponent_pfr_code":   opp_code,
                "is_home":             is_home,
                "spread_line":         spread if is_home else -spread,  # from team's own perspective
                "total_line":          total,
                "implied_team_total":  round(implied, 2),
                "expected_pass_att":   round(exp_pass, 1),
                "expected_rush_att":   round(exp_rush, 1),
                "pass_def_adj":        round(pass_adj, 4),
                "rush_def_adj":        round(rush_adj, 4),
                "game_script_factor":  round(gs_factor, 4),
                "rest_days":           int(rest) if not pd.isna(rest) else 7,
            })

    ctx = pd.DataFrame(rows)
    log.info(f"Built context for {len(ctx)} team-games")
    return ctx


def run(season: int, week: int):
    ctx = build_team_game_context(season, week)
    if ctx.empty:
        log.error("No context produced — check schedule data for the requested week.")
        return

    out = PROCESSED / "team_game_context.parquet"
    ctx.to_parquet(out, index=False)
    log.info(f"Saved team_game_context.parquet ({len(ctx)} rows)")

    # Print a readable summary
    print("\n── Team Game Context ──────────────────────────────────────────────")
    summary = ctx[[
        "pfr_code", "opponent_pfr_code", "is_home",
        "implied_team_total", "expected_pass_att", "expected_rush_att",
        "pass_def_adj", "rush_def_adj", "rest_days"
    ]].sort_values("implied_team_total", ascending=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build team game context for a given week.")
    parser.add_argument("--season", type=int, required=True, help="NFL season year (e.g. 2024)")
    parser.add_argument("--week",   type=int, required=True, help="NFL week number (e.g. 14)")
    args = parser.parse_args()
    run(args.season, args.week)
