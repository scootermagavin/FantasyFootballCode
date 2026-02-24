#!/usr/bin/env python3
"""
ingest.py — Data Ingestion and Normalization Module

Loads all raw data sources, applies consistent filtering and joins, and writes
clean parquet files to data/processed/ for use by downstream pipeline modules.

Design decisions:
  - draftkings_points (actual DK platform points) is the fantasy scoring target.
    This sidesteps the -1 vs. -2 INT discrepancy in legacy scoring weight code
    and ensures projections are calibrated to the platform we optimize for.
  - franchise_id in player_data is the PFR code — the primary team key used
    natively by both player_data.csv and fanduels_defense_data.csv.
  - team_data.csv uses normalized current full team names throughout history.
    Relocated franchises (Rams, Chargers, Raiders) appear under their current name.
  - team_schedule.csv uses abbreviations that have changed with relocations.
    The crosswalk maps both current and legacy abbreviations to the PFR code.

Outputs (data/processed/):
  team_crosswalk.parquet       — pfr_code ↔ full_name ↔ sched_abbrev mapping
  players.parquet              — skill-position game logs, regular season only
  team_stats.parquet           — per-team per-game stats with crosswalk keys
  team_rolling.parquet         — pre-computed 3-game rolling team features
  schedule.parquet             — game schedule with Vegas lines and rest days
  defense_context.parquet      — per-team-game defensive scheme + EPA context
  current_projections.parquet  — current week third-party player projections
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw" / "historical"
PROCESSED = ROOT / "data" / "processed"

# ── Constants ──────────────────────────────────────────────────────────────

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

# ── Team Crosswalk ─────────────────────────────────────────────────────────
# Each tuple: (pfr_code, full_name, current_sched_abbrev, legacy_sched_abbrevs)
#
# pfr_code          — franchise_id in player_data; player_team_pfr_franchise_code_id
#                     in fanduels_defense_data. The stable primary key across history.
# full_name         — as used in team_data.csv and team_data_rolling_three_game.csv.
#                     Already normalized to current name in that dataset.
# current_sched_abbrev — abbreviation used in current DK/FD salary CSVs.
# legacy_sched_abbrevs — historical abbreviations from team_schedule.csv rows
#                        covering years before a relocation.

_CROSSWALK = [
    # pfr_code  full_name                     current  legacy
    ("CRD", "Arizona Cardinals",              "ARI",   []),
    ("ATL", "Atlanta Falcons",                "ATL",   []),
    ("RAV", "Baltimore Ravens",               "BAL",   ["BAL"]),
    ("BUF", "Buffalo Bills",                  "BUF",   []),
    ("CAR", "Carolina Panthers",              "CAR",   []),
    ("CHI", "Chicago Bears",                  "CHI",   []),
    ("CIN", "Cincinnati Bengals",             "CIN",   []),
    ("CLE", "Cleveland Browns",               "CLE",   []),
    ("DAL", "Dallas Cowboys",                 "DAL",   []),
    ("DEN", "Denver Broncos",                 "DEN",   []),
    ("DET", "Detroit Lions",                  "DET",   []),
    ("GNB", "Green Bay Packers",              "GB",    ["GB", "GNB"]),
    ("HTX", "Houston Texans",                 "HOU",   ["HOU"]),
    ("CLT", "Indianapolis Colts",             "IND",   ["IND"]),
    ("JAX", "Jacksonville Jaguars",           "JAX",   []),
    ("KAN", "Kansas City Chiefs",             "KC",    ["KC"]),
    ("RAM", "Los Angeles Rams",               "LA",    ["STL", "LA", "LAR"]),
    ("SDG", "Los Angeles Chargers",           "LAC",   ["SD", "LAC"]),
    ("RAI", "Las Vegas Raiders",              "LV",    ["OAK", "LV"]),
    ("MIA", "Miami Dolphins",                 "MIA",   []),
    ("MIN", "Minnesota Vikings",              "MIN",   []),
    ("NWE", "New England Patriots",           "NE",    ["NE"]),
    ("NOR", "New Orleans Saints",             "NO",    ["NO"]),
    ("NYG", "New York Giants",                "NYG",   []),
    ("NYJ", "New York Jets",                  "NYJ",   []),
    ("PHI", "Philadelphia Eagles",            "PHI",   []),
    ("PIT", "Pittsburgh Steelers",            "PIT",   []),
    ("SFO", "San Francisco 49ers",            "SF",    ["SF"]),
    ("SEA", "Seattle Seahawks",               "SEA",   []),
    ("TAM", "Tampa Bay Buccaneers",           "TB",    ["TB"]),
    ("OTI", "Tennessee Titans",               "TEN",   ["TEN"]),
    ("WAS", "Washington Commanders",          "WAS",   ["WAS"]),
]


def build_crosswalk() -> pd.DataFrame:
    """
    Build the team ID crosswalk DataFrame.

    Returns a DataFrame with columns:
        pfr_code, full_name, sched_abbrev

    Also builds a supplementary abbrev_to_pfr dict (returned separately via
    get_abbrev_lookup) covering both current and legacy abbreviations.
    """
    rows = []
    for pfr_code, full_name, current_abbrev, _ in _CROSSWALK:
        rows.append({
            "pfr_code":     pfr_code,
            "full_name":    full_name,
            "sched_abbrev": current_abbrev,
        })
    return pd.DataFrame(rows)


def get_abbrev_lookup() -> dict:
    """
    Return a dict mapping every known schedule abbreviation (current and legacy)
    to its PFR franchise code.

    Used to join team_schedule.csv onto team-keyed DataFrames.
    """
    lookup = {}
    for pfr_code, _, current_abbrev, legacy_abbrevs in _CROSSWALK:
        lookup[current_abbrev] = pfr_code
        for leg in legacy_abbrevs:
            lookup[leg] = pfr_code
    return lookup


# ── Loader functions ────────────────────────────────────────────────────────

def load_players() -> pd.DataFrame:
    """
    Load combined player game logs (player_data.csv).

    Filters to:
      - Regular season only (playoffs == 0)
      - Skill positions: QB, RB, WR, TE

    Renames franchise_id → pfr_code and draftkings_points → dk_points for
    consistency with the crosswalk and downstream modules.

    The dk_points column (actual DraftKings platform points) is the training
    target for projections.py. It is used directly rather than re-deriving from
    raw stats to avoid the legacy -2 INT weight discrepancy.
    """
    path = RAW / "player_data.csv"
    log.info(f"Loading player data from {path.name} ...")
    df = pd.read_csv(path, low_memory=False)
    before = len(df)

    df = df[
        (df["playoffs"] == 0) &
        df["position_id"].isin(SKILL_POSITIONS)
    ].copy()

    log.info(f"  {before:,} total rows → {len(df):,} after skill-position + regular-season filter")

    df = df.rename(columns={
        "franchise_id":     "pfr_code",
        "draftkings_points": "dk_points",
        "fanduel_points":    "fd_points",
    })

    null_rate = df["dk_points"].isnull().mean()
    if null_rate > 0.05:
        log.warning(f"  dk_points null rate is {null_rate:.1%} — investigate source data")
    else:
        log.info(f"  dk_points null rate: {null_rate:.1%} ✓")

    seasons = f"{int(df['season'].min())}–{int(df['season'].max())}"
    log.info(f"  Seasons: {seasons} | Positions: {sorted(df['position_id'].unique())}")

    return df


def load_team_stats(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Load per-team per-game stats from team_data.csv.

    Joins the crosswalk on full team name to add pfr_code and sched_abbrev.
    Logs a warning if any team names in the file do not match the crosswalk.
    """
    path = RAW / "team_data.csv"
    log.info(f"Loading team stats from {path.name} ...")
    df = pd.read_csv(path, low_memory=False)

    null_name_count = df["name"].isnull().sum()
    if null_name_count > 0:
        log.warning(f"  Dropping {null_name_count} rows with null team name (source data artifact)")
        df = df[df["name"].notna()].copy()

    df = df.merge(
        crosswalk[["pfr_code", "full_name", "sched_abbrev"]],
        left_on="name",
        right_on="full_name",
        how="left",
    ).drop(columns=["full_name"])

    unmatched = df["pfr_code"].isnull().sum()
    if unmatched > 0:
        missing_names = df.loc[df["pfr_code"].isnull(), "name"].unique()
        log.warning(f"  {unmatched} rows unmatched in crosswalk: {missing_names}")
    else:
        log.info(f"  {len(df):,} rows, all matched to crosswalk ✓")

    return df


def load_team_rolling(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Load pre-computed 3-game rolling team features from team_data_rolling_three_game.csv.

    Same team name join as load_team_stats. Adds pfr_code and sched_abbrev.
    This file contains 479 columns covering rolling avg, std, max, and coefficient
    of variation for every team-level stat — ready to use as model features.
    """
    path = RAW / "team_data_rolling_three_game.csv"
    log.info(f"Loading rolling team features from {path.name} ...")
    df = pd.read_csv(path, low_memory=False)

    null_name_count = df["name"].isnull().sum()
    if null_name_count > 0:
        log.warning(f"  Dropping {null_name_count} rows with null team name (source data artifact)")
        df = df[df["name"].notna()].copy()

    df = df.merge(
        crosswalk[["pfr_code", "full_name", "sched_abbrev"]],
        left_on="name",
        right_on="full_name",
        how="left",
    ).drop(columns=["full_name"])

    unmatched = df["pfr_code"].isnull().sum()
    if unmatched > 0:
        missing_names = df.loc[df["pfr_code"].isnull(), "name"].unique()
        log.warning(f"  {unmatched} rows unmatched in crosswalk: {missing_names}")
    else:
        log.info(f"  {len(df):,} rows, all matched to crosswalk ✓")

    return df


def load_schedule() -> pd.DataFrame:
    """
    Load game schedule with Vegas lines and rest days from team_schedule.csv.

    Adds pfr_code for both home and away teams using the full abbrev lookup
    (which covers historical abbreviations like STL, OAK, SD as well as current
    ones), enabling joins to player and team data.
    """
    path = RAW / "team_schedule.csv"
    log.info(f"Loading schedule from {path.name} ...")
    df = pd.read_csv(path, low_memory=False)

    abbrev_lookup = get_abbrev_lookup()

    df["home_pfr_code"] = df["home_team"].map(abbrev_lookup)
    df["away_pfr_code"] = df["away_team"].map(abbrev_lookup)

    unmatched_home = df["home_pfr_code"].isnull().sum()
    unmatched_away = df["away_pfr_code"].isnull().sum()
    if unmatched_home + unmatched_away > 0:
        missing_h = df.loc[df["home_pfr_code"].isnull(), "home_team"].unique()
        missing_a = df.loc[df["away_pfr_code"].isnull(), "away_team"].unique()
        log.warning(f"  Unmatched home abbrevs: {missing_h}")
        log.warning(f"  Unmatched away abbrevs: {missing_a}")
    else:
        log.info(f"  {len(df):,} games, seasons {int(df['season'].min())}–{int(df['season'].max())}, all abbrevs matched ✓")

    return df


def load_defense_context() -> pd.DataFrame:
    """
    Load per-team-game defensive and scheme context from fanduels_defense_data.csv.

    This file uses PFR codes natively (player_team_pfr_franchise_code_id), so no
    crosswalk join is needed. Renames to pfr_code / opponent_pfr_code for consistency.

    Unique value vs. team_data.csv: coaching staff, offensive/defensive scheme,
    weather, roof/surface type, time of possession, EPA breakdowns, and full
    opponent-side mirror stats on the same row.
    """
    path = RAW / "fanduels_defense_data.csv"
    log.info(f"Loading defense context from {path.name} ...")
    df = pd.read_csv(path, low_memory=False)

    df = df.rename(columns={
        "player_team_pfr_franchise_code_id":   "pfr_code",
        "opponent_team_pfr_franchise_code_id": "opponent_pfr_code",
        "player_team_name":                    "team_name",
    })

    log.info(f"  {len(df):,} team-game rows, seasons {int(df['season'].min())}–{int(df['season'].max())}")

    return df


def load_current_projections() -> pd.DataFrame:
    """
    Load current-week third-party player projections from projections_raw.csv.

    Contains projected rushing/receiving/passing stat lines plus cross-platform
    player IDs (draft_kings_player_id, fan_duel_player_id) needed to match
    players to DK/FD salary CSVs in optimizer.py.
    """
    path = RAW / "projections_raw.csv"
    log.info(f"Loading current projections from {path.name} ...")
    df = pd.read_csv(path, low_memory=False)

    season = df["season"].iloc[0]
    week = df["week"].iloc[0]
    log.info(f"  {len(df):,} player projections — season {int(season)}, week {int(week)}")

    return df


# ── Main ────────────────────────────────────────────────────────────────────

def run():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Crosswalk first — required by team loaders
    crosswalk = build_crosswalk()
    crosswalk.to_parquet(PROCESSED / "team_crosswalk.parquet", index=False)
    log.info(f"Saved team_crosswalk.parquet ({len(crosswalk)} teams)")

    # Player data
    players = load_players()
    players.to_parquet(PROCESSED / "players.parquet", index=False)
    log.info(f"Saved players.parquet ({len(players):,} rows)")

    # Team stats
    team_stats = load_team_stats(crosswalk)
    team_stats.to_parquet(PROCESSED / "team_stats.parquet", index=False)
    log.info(f"Saved team_stats.parquet ({len(team_stats):,} rows)")

    # Rolling team features
    team_rolling = load_team_rolling(crosswalk)
    team_rolling.to_parquet(PROCESSED / "team_rolling.parquet", index=False)
    log.info(f"Saved team_rolling.parquet ({len(team_rolling):,} rows)")

    # Schedule
    schedule = load_schedule()
    schedule.to_parquet(PROCESSED / "schedule.parquet", index=False)
    log.info(f"Saved schedule.parquet ({len(schedule):,} rows)")

    # Defense context
    defense_ctx = load_defense_context()
    defense_ctx.to_parquet(PROCESSED / "defense_context.parquet", index=False)
    log.info(f"Saved defense_context.parquet ({len(defense_ctx):,} rows)")

    # Current week projections
    projections = load_current_projections()
    projections.to_parquet(PROCESSED / "current_projections.parquet", index=False)
    log.info(f"Saved current_projections.parquet ({len(projections):,} rows)")

    log.info("\ningest.py complete ✓")
    log.info(f"All outputs written to: {PROCESSED}")


if __name__ == "__main__":
    run()
