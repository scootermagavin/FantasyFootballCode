#!/usr/bin/env python3
"""
optimizer.py — DraftKings Lineup Optimizer

Generates optimal DraftKings NFL lineups by combining:
  - Our model's projections (projections_current.parquet)
  - DraftKings salary export CSV (downloaded from the DK contest page)
  - Optional DST adjustment via team_game_context.parquet

Player matching order:
  1. DK player ID (draft_kings_player_id from current_projections.parquet)
  2. Name + team fuzzy match (fallback for players missing DK ID)

DK classic roster: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DST  |  $50,000 cap

Usage:
  # Download salary CSV from DraftKings, then:
  python3 src/optimizer.py --salary DKSalaries.csv --season 2025 --week 18
  python3 src/optimizer.py --salary DKSalaries.csv --season 2025 --week 18 \\
      --lineups 150 --lock "Josh Allen" --exclude "Derrick Henry"
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pydfs_lineup_optimizer import get_optimizer, Site, Sport
from pydfs_lineup_optimizer.stacks import PositionsStack

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"

N_LINEUPS_DEFAULT = 20
MAX_EXPOSURE      = 0.60   # no player in more than 60% of lineups by default
DST_LEAGUE_AVG    = 8.0    # approximate DK DST average pts/game used for scaling


# ── Player-ID lookup ──────────────────────────────────────────────────────────

def _build_projection_lookup(season: int, week: int) -> dict[str, float]:
    """
    Build a dict mapping DK player ID (string) → predicted_dk from the model.

    Primary join: projections_current.player_id → current_projections.draft_kings_player_id
    Fallback key: normalised "Firstname Lastname" for players without a DK ID mapping.
    """
    proj_path = PROCESSED / "projections_current.parquet"
    curr_path = PROCESSED / "current_projections.parquet"

    if not proj_path.exists():
        log.error(
            "projections_current.parquet not found.\n"
            f"  Run first: python3 src/projections.py --season {season} --week {week}"
        )
        return {}

    proj = pd.read_parquet(proj_path)

    # ID bridge: internal player_id → DK player ID
    id_lookup: dict[str, float] = {}   # str(dk_id) → predicted_dk
    name_lookup: dict[str, float] = {} # "firstname lastname" → predicted_dk

    if curr_path.exists():
        curr = pd.read_parquet(curr_path)
        # Filter to Avg projections only (projections_raw may have Min/Max rows too)
        if "Version" in curr.columns:
            curr = curr[curr["Version"] == "Avg"]
        id_map = (
            curr[["player_id", "draft_kings_player_id"]]
            .dropna(subset=["draft_kings_player_id"])
            .drop_duplicates("player_id")
        )
        proj_with_id = proj.merge(id_map, on="player_id", how="left")
    else:
        log.warning("current_projections.parquet not found — will match by name only.")
        proj_with_id = proj.copy()
        proj_with_id["draft_kings_player_id"] = np.nan

    for _, row in proj_with_id.iterrows():
        predicted = float(row["predicted_dk"])
        name_key  = str(row["name"]).strip().lower()
        name_lookup[name_key] = predicted

        dk_id = row.get("draft_kings_player_id")
        if pd.notna(dk_id):
            id_lookup[str(int(dk_id))] = predicted

    log.info(f"  Projection lookup: {len(id_lookup)} by DK ID, {len(name_lookup)} by name")
    return id_lookup, name_lookup


# ── DST adjustment ────────────────────────────────────────────────────────────

def _build_dst_lookup() -> dict[str, float]:
    """
    Compute a DST projection adjustment multiplier from team_game_context.

    A lower opponent implied total → better DST game environment.
    Returns dict: {team_abbrev → adjustment_multiplier}.
    If team_game_context is unavailable, returns empty dict (use DK avg as-is).
    """
    ctx_path = PROCESSED / "team_game_context.parquet"
    xwalk_path = PROCESSED / "team_crosswalk.parquet"
    if not ctx_path.exists():
        return {}

    ctx   = pd.read_parquet(ctx_path)
    xwalk = pd.read_parquet(xwalk_path) if xwalk_path.exists() else None

    # Build pfr_code → DK-style abbreviation mapping (use sched_abbrev as proxy)
    if xwalk is not None and "sched_abbrev" in xwalk.columns:
        abbrev_map = dict(zip(xwalk["pfr_code"], xwalk["sched_abbrev"]))
    else:
        abbrev_map = {}

    # For each team, compute the opponent's implied total (lower = better DST)
    # team_game_context has one row per team with opponent_pfr_code and implied_team_total
    opp_implied: dict[str, float] = {}
    for _, row in ctx.iterrows():
        opp_code = row.get("opponent_pfr_code", None)
        if opp_code is None:
            continue
        # Find opponent's implied total from the other row
        opp_row = ctx[ctx["pfr_code"] == opp_code]
        if opp_row.empty:
            continue
        opp_total = float(opp_row["implied_team_total"].iloc[0])
        team_abbrev = abbrev_map.get(row["pfr_code"], row["pfr_code"])
        opp_implied[team_abbrev] = opp_total

    if not opp_implied:
        return {}

    league_avg = float(np.mean(list(opp_implied.values())))
    multipliers = {team: league_avg / opp for team, opp in opp_implied.items()}
    log.info(f"  DST adjustment built for {len(multipliers)} teams  (avg opp implied={league_avg:.1f})")
    return multipliers


# ── GridironAI salary format adapter ─────────────────────────────────────────

def _is_gridironai_format(path: Path) -> bool:
    """Return True if the CSV uses GridironAI format (has a display_name column)."""
    with open(path) as f:
        header = f.readline()
    return "display_name" in header.split(",")


def _build_name_to_dkid(curr_path: Path) -> dict[str, str]:
    """
    Build name.lower() → str(int(dk_id)) from current_projections.parquet.
    Filters to Version=="Avg" rows if Version column exists.
    """
    if not curr_path.exists():
        log.warning("current_projections.parquet missing — GridironAI ID bridge unavailable")
        return {}
    curr = pd.read_parquet(curr_path)
    if "Version" in curr.columns:
        curr = curr[curr["Version"] == "Avg"]
    result: dict[str, str] = {}
    for _, row in curr.iterrows():
        dk_id = row.get("draft_kings_player_id")
        name  = str(row.get("name", "")).strip().lower()
        if pd.notna(dk_id) and name:
            result[name] = str(int(dk_id))
    return result


def _convert_gridironai_to_dk_csv(salary_path: Path, curr_path: Path) -> Path:
    """
    Convert GridironAI salary CSV to a DK-compatible temp CSV for pydfs.

    - Non-DST players: name-matched to current_projections.parquet for DK player ID.
      Players without a match get a synthetic unique ID (name-based hash).
    - DST players: position=="DST", display_name = team abbrev. ID = "DST_{team}".
    - K (kicker) rows filtered — no DK Classic slot.
    - Game Info constructed as "AWAY@HOME" from team/opp/game_location.

    Returns path to temp CSV (caller should delete after pydfs loads it).
    """
    df = pd.read_csv(salary_path)
    name_to_dkid = _build_name_to_dkid(curr_path)

    matched = 0
    rows = []
    for _, row in df.iterrows():
        pos = str(row["position"]).strip().upper()
        if pos == "K":
            continue  # DK Classic has no kicker slot

        display  = str(row["display_name"]).strip()
        team     = str(row["team"]).strip()
        opp      = str(row["opp"]).strip()
        loc      = str(row.get("game_location", "away")).strip().lower()

        raw_sal = row.get("salary")
        if pd.isna(raw_sal):
            log.debug(f"  Skipping {display} — no salary in source CSV")
            continue
        salary   = int(raw_sal)
        proj_val = row.get("Projection")
        avg_pts  = float(proj_val) if pd.notna(proj_val) else 0.0

        # Game Info: pydfs parses "AWAY@HOME ..." for game-based stacking
        game_info = f"{opp}@{team}" if loc == "home" else f"{team}@{opp}"

        if pos == "DST":
            dk_id      = f"DST_{team}"
            roster_pos = "DST"
        else:
            dk_id = name_to_dkid.get(display.lower(), "")
            if dk_id:
                matched += 1
            else:
                # Stable synthetic ID so pydfs doesn't collapse duplicates
                dk_id = f"SYN_{display.lower().replace(' ', '_')}"
            roster_pos = pos

        rows.append({
            "ID":               dk_id,
            "Name+ID":          f"{display} ({dk_id})",
            "Name":             display,
            "Position":         pos,
            "TeamAbbrev":       team,
            "Salary":           salary,
            "AvgPointsPerGame": avg_pts,
            "Game Info":        game_info,
            "Roster Position":  roster_pos,
        })

    non_dst = sum(1 for r in rows if r["Position"] != "DST")
    log.info(f"  GridironAI bridge: {matched}/{non_dst} non-DST players matched to DK ID")

    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", prefix="dfs_dk_"
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    return tmp_path


# ── Projection injection ──────────────────────────────────────────────────────

def _inject_projections(
    optimizer,
    id_lookup:  dict[str, float],
    name_lookup: dict[str, float],
    dst_mult:   dict[str, float],
) -> tuple[int, int]:
    """
    Iterate over pydfs Player objects and override fppg with model projections.
    DSTs are scaled by opponent implied total if available, otherwise left as-is.

    Returns (n_matched_by_id, n_matched_by_name).
    """
    matched_id   = 0
    matched_name = 0
    unmatched    = []

    for player in optimizer.players:
        # DST: apply game-environment multiplier, skip model projection
        if "DST" in player.positions or "D" in player.positions:
            mult = dst_mult.get(player.team, 1.0)
            player.fppg = round(player.fppg * mult, 2)
            continue

        # Try DK player ID first
        if player.id in id_lookup:
            player.fppg = round(id_lookup[player.id], 2)
            matched_id += 1
            continue

        # Fallback: normalised name match
        full_name = f"{player.first_name} {player.last_name}".strip().lower()
        if full_name in name_lookup:
            player.fppg = round(name_lookup[full_name], 2)
            matched_name += 1
            continue

        # No match found — zero out projection so the optimizer ignores them
        unmatched.append(f"{player.first_name} {player.last_name} ({player.team})")
        player.fppg = 0.0

    if unmatched:
        log.warning(f"  {len(unmatched)} players with no projection (zeroed out):")
        for name in unmatched[:20]:
            log.warning(f"    {name}")

    return matched_id, matched_name


# ── Lineup printing ───────────────────────────────────────────────────────────

_SLOT_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "DST"]

def _print_lineup(idx: int, lineup, salary_cap: int = 50_000) -> None:
    total_proj = sum(p.fppg for p in lineup.players)
    total_sal  = sum(p.salary for p in lineup.players)

    # Sort by DK slot order
    def slot_key(p):
        pos = p.positions[0] if p.positions else "?"
        try:
            return _SLOT_ORDER.index(pos)
        except ValueError:
            return 99

    players = sorted(lineup.players, key=slot_key)

    print(f"\n  Lineup {idx:>3}  |  Proj: {total_proj:.1f} pts  |  Salary: ${total_sal:,}")
    print(f"  {'Pos':<5} {'Name':<25} {'Team':<5} {'Salary':>8}  {'Proj':>6}")
    print(f"  {'─'*55}")
    for p in players:
        pos = "/".join(p.positions[:2])
        print(f"  {pos:<5} {p.first_name+' '+p.last_name:<25} {p.team:<5} ${p.salary:>7,.0f}  {p.fppg:>6.1f}")


# ── Export ────────────────────────────────────────────────────────────────────

def _export_lineups(lineups: list, season: int, week: int) -> Path:
    """
    Save lineups to CSV with one row per lineup.

    Columns per slot: <SLOT>, <SLOT>_proj, <SLOT>_sal
    Duplicate slots get a numeric suffix: RB/RB2, WR/WR2/WR3.
    Value in each cell: "Firstname Lastname (ID)" (DK upload format).
    """
    from collections import defaultdict
    rows = []
    for idx, lineup in enumerate(lineups, 1):
        row = {"lineup": idx}
        slot_count: dict[str, int] = defaultdict(int)
        for lp in lineup.players:
            slot = lp.lineup_position          # e.g. "QB", "RB", "WR", "TE", "FLEX", "DST"
            slot_count[slot] += 1
            count = slot_count[slot]
            col = slot if count == 1 else f"{slot}{count}"
            row[col]            = f"{lp.first_name} {lp.last_name} ({lp.id})"
            row[f"{col}_proj"]  = round(lp.fppg, 2)
            row[f"{col}_sal"]   = lp.salary
        row["projected_total"] = round(sum(lp.fppg for lp in lineup.players), 1)
        row["salary_used"]     = sum(lp.salary for lp in lineup.players)
        rows.append(row)

    out_path = PROCESSED / f"lineups_{season}_w{week:02d}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info(f"Saved {out_path.name} ({len(rows)} lineups)")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    salary_csv: str,
    season: int,
    week: int,
    n_lineups: int = N_LINEUPS_DEFAULT,
    lock: list[str] | None = None,
    exclude: list[str] | None = None,
    max_exposure: float = MAX_EXPOSURE,
) -> list:
    salary_path = Path(salary_csv)
    if not salary_path.exists():
        log.error(f"Salary CSV not found: {salary_path}")
        return []

    log.info(f"Season {season} Week {week} — {n_lineups} lineups from {salary_path.name}")

    # ── Build projection lookup ───────────────────────────────────────────────
    id_lookup, name_lookup = _build_projection_lookup(season, week)
    if not id_lookup and not name_lookup:
        return []
    dst_mult = _build_dst_lookup()

    # ── Load salary CSV (DK native or GridironAI) ─────────────────────────────
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
    if _is_gridironai_format(salary_path):
        log.info("  Detected GridironAI salary format — converting to DK-compatible CSV")
        tmp_path = _convert_gridironai_to_dk_csv(
            salary_path, PROCESSED / "current_projections.parquet"
        )
        try:
            optimizer.load_players_from_csv(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        optimizer.load_players_from_csv(str(salary_path))
    log.info(f"  Loaded {len(list(optimizer.players))} players from salary file")

    # ── Inject model projections ──────────────────────────────────────────────
    n_id, n_name = _inject_projections(optimizer, id_lookup, name_lookup, dst_mult)
    log.info(f"  Matched {n_id} by DK ID, {n_name} by name")

    # ── Exposure cap ──────────────────────────────────────────────────────────
    for player in optimizer.players:
        if player.max_exposure is None:
            player.max_exposure = max_exposure

    # ── Lock / exclude ────────────────────────────────────────────────────────
    if lock:
        for name in lock:
            try:
                p = optimizer.get_player_by_name(name)
                optimizer.add_player_to_lineup(p)
                log.info(f"  Locked: {name}")
            except Exception as e:
                log.warning(f"  Could not lock '{name}': {e}")

    if exclude:
        for name in exclude:
            try:
                p = optimizer.get_player_by_name(name)
                optimizer.remove_player(p)
                log.info(f"  Excluded: {name}")
            except Exception as e:
                log.warning(f"  Could not exclude '{name}': {e}")

    # ── Stacking: QB must pair with ≥1 WR or TE from same team ───────────────
    optimizer.add_stack(PositionsStack(["QB", ("WR", "TE")]))

    # ── Optimize ──────────────────────────────────────────────────────────────
    log.info(f"Optimizing {n_lineups} lineups ...")
    lineups = list(optimizer.optimize(n_lineups))
    log.info(f"Generated {len(lineups)} lineups")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  {n_lineups} LINEUPS — Season {season}  Week {week}")
    print(f"{'═'*60}")
    for i, lineup in enumerate(lineups, 1):
        _print_lineup(i, lineup)

    # ── Exposure summary ──────────────────────────────────────────────────────
    from collections import Counter
    exposure: Counter = Counter()
    for lineup in lineups:
        for p in lineup.players:
            exposure[f"{p.first_name} {p.last_name}"] += 1

    print(f"\n── Player Exposure (top 25) ────────────────────────────────────")
    print(f"  {'Player':<28} {'Lineups':>7}  {'Exposure':>9}")
    for name, count in exposure.most_common(25):
        print(f"  {name:<28} {count:>7}  {count/n_lineups:>9.1%}")

    # ── Export ────────────────────────────────────────────────────────────────
    _export_lineups(lineups, season, week)

    return lineups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate optimal DraftKings NFL lineups using model projections."
    )
    parser.add_argument("--salary",   required=True,
                        help="Path to salary CSV (GridironAI gridironai-dfs-*.csv or standard DK export)")
    parser.add_argument("--season",   type=int, required=True,
                        help="Season year (must match projections_current.parquet)")
    parser.add_argument("--week",     type=int, required=True,
                        help="Week number (must match projections_current.parquet)")
    parser.add_argument("--lineups",  type=int, default=N_LINEUPS_DEFAULT,
                        help=f"Number of lineups to generate (default: {N_LINEUPS_DEFAULT})")
    parser.add_argument("--lock",     nargs="*", default=None,
                        help="Player names to force into every lineup")
    parser.add_argument("--exclude",  nargs="*", default=None,
                        help="Player names to exclude from all lineups")
    parser.add_argument("--exposure", type=float, default=MAX_EXPOSURE,
                        help=f"Max player exposure fraction (default: {MAX_EXPOSURE})")
    args = parser.parse_args()

    run(
        salary_csv   = args.salary,
        season       = args.season,
        week         = args.week,
        n_lineups    = args.lineups,
        lock         = args.lock,
        exclude      = args.exclude,
        max_exposure = args.exposure,
    )
