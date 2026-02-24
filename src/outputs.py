#!/usr/bin/env python3
"""
outputs.py — Final DFS Deliverable Generator

Reads optimizer lineups + model projections + DK salary CSV → produces:
  1. dk_upload_{season}_w{week:02d}.csv   — DK mass entry format
  2. rankings_{season}_w{week:02d}.csv    — all players ranked by predicted_dk
  3. divergence_{season}_w{week:02d}.csv  — top bullish/bearish vs consensus
  4. slate_{season}_w{week:02d}.txt       — game-by-game context + top plays

Salary CSV is optional; without it, value and divergence columns are omitted.

Usage:
  python3 src/outputs.py --season 2025 --week 22
  python3 src/outputs.py --season 2025 --week 22 --salary DKSalaries.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS   = ROOT / "data" / "outputs"

# Slot order used in lineups CSV (must match optimizer.py export)
DK_SLOTS = ["QB", "RB", "RB2", "WR", "WR2", "WR3", "TE", "FLEX", "DST"]

SLATE_TOP_N   = 3   # top plays shown per team in slate report
DIVERGENCE_N  = 15  # top N bullish / bearish plays in divergence report


# ── Salary loader ──────────────────────────────────────────────────────────────

def _build_grai_name_to_dkid(curr_path: Path) -> dict[str, str]:
    """name.lower() → str(int(dk_id)) from current_projections.parquet (Version==Avg)."""
    if not curr_path.exists():
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


def _load_salary_df(salary_csv: str | None, curr_path: Path | None = None) -> pd.DataFrame:
    """
    Load salary CSV → standardised DataFrame.

    Handles two formats:
      - Standard DK export: Name, ID, Position, TeamAbbrev, Salary, AvgPointsPerGame
      - GridironAI:         display_name, position, team, salary, Projection,
                            ProjectedOwnership, PotentialLeverage

    Returns DataFrame with at minimum: dk_id, name, name_lower, position, team,
    salary, avg_pts.  GridironAI format also includes: projected_ownership,
    potential_leverage.
    """
    empty = pd.DataFrame(columns=["dk_id", "name", "name_lower",
                                   "position", "team", "salary", "avg_pts"])
    if salary_csv is None:
        return empty
    path = Path(salary_csv)
    if not path.exists():
        log.warning(f"Salary CSV not found: {path} — value/divergence will be empty")
        return empty

    df = pd.read_csv(path)

    # ── GridironAI format ─────────────────────────────────────────────────────
    if "display_name" in df.columns:
        name_to_dkid = _build_grai_name_to_dkid(
            curr_path or PROCESSED / "current_projections.parquet"
        )
        df = df.rename(columns={
            "display_name":       "name",
            "position":           "position",
            "team":               "team",
            "salary":             "salary",
            "Projection":         "avg_pts",
            "ProjectedOwnership": "projected_ownership",
            "PotentialLeverage":  "potential_leverage",
        })
        df["name_lower"] = df["name"].str.strip().str.lower()
        df["dk_id"]      = df["name_lower"].map(name_to_dkid).fillna("").astype(str)
        keep = ["dk_id", "name", "name_lower", "position", "team", "salary", "avg_pts"]
        for col in ("projected_ownership", "potential_leverage"):
            if col in df.columns:
                keep.append(col)
        return df[keep]

    # ── Standard DK format ────────────────────────────────────────────────────
    df = df.rename(columns={
        "Name":             "name",
        "ID":               "dk_id",
        "Position":         "position",
        "TeamAbbrev":       "team",
        "Salary":           "salary",
        "AvgPointsPerGame": "avg_pts",
    })
    df["dk_id"]      = df["dk_id"].astype(str)
    df["name_lower"] = df["name"].str.strip().str.lower()
    return df[["dk_id", "name", "name_lower", "position", "team", "salary", "avg_pts"]]


# ── Rankings ───────────────────────────────────────────────────────────────────

def _build_rankings(proj_path: Path, salary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join projections_current + current_projections (DK ID bridge) + salary CSV.
    Returns DataFrame sorted by predicted_dk descending.

    Output columns: name, position_id, team, salary, predicted_dk, value, avg_pts, divergence
    """
    proj = pd.read_parquet(proj_path)
    # Schema: player_id, name, pfr_code, position_id, season, week, predicted_dk

    # ── DK ID bridge ─────────────────────────────────────────────────────────
    curr_path = PROCESSED / "current_projections.parquet"
    if curr_path.exists():
        curr = pd.read_parquet(curr_path)
        if "Version" in curr.columns:
            curr = curr[curr["Version"] == "Avg"]
        bridge = (
            curr[["player_id", "draft_kings_player_id"]]
            .dropna(subset=["draft_kings_player_id"])
            .drop_duplicates("player_id")
            .copy()
        )
        bridge["dk_id"] = bridge["draft_kings_player_id"].apply(lambda x: str(int(x)))
        proj = proj.merge(bridge[["player_id", "dk_id"]], on="player_id", how="left")
    else:
        proj["dk_id"] = None

    proj["name_lower"] = proj["name"].str.strip().str.lower()

    # ── Join salary data ──────────────────────────────────────────────────────
    if not salary_df.empty:
        base_sal_cols  = ["salary", "avg_pts", "team", "position"]
        extra_sal_cols = [c for c in salary_df.columns
                          if c not in base_sal_cols + ["dk_id", "name", "name_lower"]]
        sal_cols = base_sal_cols + extra_sal_cols

        sal_by_id   = (salary_df.drop_duplicates(subset=["dk_id"])
                       .set_index("dk_id")[sal_cols])
        sal_by_name = (salary_df.drop_duplicates(subset=["name_lower"])
                       .set_index("name_lower")[sal_cols])

        # Step 1: join by DK player ID (primary)
        has_id = proj["dk_id"].notna() & (proj["dk_id"] != "")
        id_matched   = proj[has_id].join(sal_by_id,   on="dk_id",      how="left")
        name_matched = proj[~has_id].join(sal_by_name, on="name_lower", how="left")

        # Step 2: name fallback for ID-matched rows that didn't hit salary file
        need_fallback = id_matched["salary"].isna()
        if need_fallback.any():
            drop_on_fallback = [c for c in sal_cols if c in id_matched.columns]
            fallback = (
                id_matched[need_fallback]
                .drop(columns=drop_on_fallback)
                .join(sal_by_name, on="name_lower", how="left")
            )
            id_matched = pd.concat(
                [id_matched[~need_fallback], fallback], ignore_index=True
            )

        proj = pd.concat([id_matched, name_matched], ignore_index=True)
    else:
        extra_sal_cols = []
        proj["salary"]   = None
        proj["avg_pts"]  = None
        proj["team"]     = proj["pfr_code"]
        proj["position"] = proj["position_id"]

    # ── Derived metrics ───────────────────────────────────────────────────────
    proj["value"] = proj.apply(
        lambda r: round(r["predicted_dk"] / (r["salary"] / 1000), 2)
        if pd.notna(r.get("salary")) and r["salary"] > 0 else None,
        axis=1,
    )
    proj["divergence"] = proj.apply(
        lambda r: round(r["predicted_dk"] - r["avg_pts"], 2)
        if pd.notna(r.get("avg_pts")) and r["avg_pts"] > 0 else None,
        axis=1,
    )

    # Keep pfr_code for slate join; drop helper columns
    drop_cols = [c for c in ["dk_id", "name_lower", "player_id", "season", "week"]
                 if c in proj.columns]
    proj = proj.drop(columns=drop_cols)

    out_cols = ["name", "position_id", "pfr_code", "team", "salary",
                "predicted_dk", "value", "avg_pts", "divergence"]
    out_cols += [c for c in extra_sal_cols if c not in out_cols]
    out_cols = [c for c in out_cols if c in proj.columns]
    return (
        proj[out_cols]
        .sort_values("predicted_dk", ascending=False)
        .reset_index(drop=True)
    )


# ── DK upload ──────────────────────────────────────────────────────────────────

def _write_dk_upload(lineups_path: Path, season: int, week: int) -> Path:
    """
    Convert lineups CSV (QB, RB, RB2, WR, WR2, WR3, TE, FLEX, DST columns)
    to DK mass-entry format. Header uses duplicate position names as DK requires.
    """
    lineups = pd.read_csv(lineups_path)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"dk_upload_{season}_w{week:02d}.csv"

    with open(out_path, "w") as f:
        f.write("QB,RB,RB,WR,WR,WR,TE,FLEX,DST\n")
        for _, row in lineups.iterrows():
            vals = [
                row.get("QB",   ""), row.get("RB",   ""), row.get("RB2",  ""),
                row.get("WR",   ""), row.get("WR2",  ""), row.get("WR3",  ""),
                row.get("TE",   ""), row.get("FLEX", ""), row.get("DST",  ""),
            ]
            f.write(",".join(f'"{v}"' for v in vals) + "\n")

    log.info(f"Saved {out_path.name} ({len(lineups)} lineups)")
    return out_path


# ── Rankings CSV ───────────────────────────────────────────────────────────────

def _write_rankings(rankings: pd.DataFrame, season: int, week: int) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"rankings_{season}_w{week:02d}.csv"
    rankings.to_csv(out_path, index=False)
    log.info(f"Saved {out_path.name} ({len(rankings)} players)")
    return out_path


# ── Divergence CSV ─────────────────────────────────────────────────────────────

def _write_divergence(rankings: pd.DataFrame, season: int, week: int) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"divergence_{season}_w{week:02d}.csv"

    div_df = rankings.dropna(subset=["divergence"]).copy()
    if div_df.empty:
        log.warning("  No divergence data — salary CSV required for this report")
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path

    bullish = div_df.nlargest(DIVERGENCE_N, "divergence").assign(signal="BULLISH")
    bearish = div_df.nsmallest(DIVERGENCE_N, "divergence").assign(signal="BEARISH")
    out = pd.concat([bullish, bearish], ignore_index=True)
    out.to_csv(out_path, index=False)
    log.info(f"Saved {out_path.name} ({len(bullish)} bullish, {len(bearish)} bearish)")
    return out_path


# ── Slate TXT ──────────────────────────────────────────────────────────────────

def _write_slate(rankings: pd.DataFrame, season: int, week: int) -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"slate_{season}_w{week:02d}.txt"

    ctx_path   = PROCESSED / "team_game_context.parquet"
    xwalk_path = PROCESSED / "team_crosswalk.parquet"

    if not ctx_path.exists():
        log.warning("  team_game_context.parquet not found — slate report unavailable")
        out_path.write_text("Slate data unavailable.\n")
        return out_path

    ctx   = pd.read_parquet(ctx_path)
    xwalk = pd.read_parquet(xwalk_path) if xwalk_path.exists() else pd.DataFrame()

    # pfr_code → display name
    name_map: dict[str, str] = {}
    if "full_name" in xwalk.columns and "pfr_code" in xwalk.columns:
        name_map = dict(zip(xwalk["pfr_code"], xwalk["full_name"]))

    # Filter to current season/week (ctx is overwritten each run but may include old weeks)
    if "season" in ctx.columns and "week" in ctx.columns:
        ctx = ctx[(ctx["season"] == season) & (ctx["week"] == week)]

    lines = [
        f"SLATE REPORT — Season {season}  Week {week}",
        "=" * 60,
        "",
    ]

    # Each pair of teams appears twice (home and away rows); display each game once
    seen: set[frozenset] = set()
    for _, row in ctx.iterrows():
        team = row["pfr_code"]
        opp  = row.get("opponent_pfr_code", "")
        pair = frozenset([team, opp])
        if pair in seen:
            continue
        seen.add(pair)

        home_rows = ctx[ctx["pfr_code"] == team]
        away_rows = ctx[ctx["pfr_code"] == opp] if opp else pd.DataFrame()

        home_impl = float(home_rows["implied_team_total"].iloc[0]) if not home_rows.empty else 0.0
        away_impl = float(away_rows["implied_team_total"].iloc[0]) if not away_rows.empty else 0.0
        total = home_impl + away_impl

        is_home = bool(home_rows["is_home"].iloc[0]) if not home_rows.empty else True
        if is_home:
            home_pfr, away_pfr   = team, opp
            home_total, away_total = home_impl, away_impl
        else:
            home_pfr, away_pfr   = opp, team
            home_total, away_total = away_impl, home_impl

        home_name = name_map.get(home_pfr, home_pfr)
        away_name = name_map.get(away_pfr, away_pfr)

        lines.append(f"  {away_name} @ {home_name}")
        lines.append(
            f"  O/U: {total:.1f}  |  {home_name}: {home_total:.1f}"
            f"  |  {away_name}: {away_total:.1f}"
        )
        lines.append("")

        for pfr, tname in [(home_pfr, home_name), (away_pfr, away_name)]:
            if not pfr:
                continue
            # Match players via pfr_code (from projections_current)
            team_plays = rankings[rankings["pfr_code"] == pfr].head(SLATE_TOP_N)
            if team_plays.empty:
                continue
            lines.append(f"    {tname}:")
            for _, p in team_plays.iterrows():
                pos = p.get("position_id") or p.get("position") or ""
                sal = f"${p['salary']:,.0f}" if pd.notna(p.get("salary")) else "     "
                val = f"  val={p['value']:.1f}" if pd.notna(p.get("value")) else ""
                lines.append(
                    f"      {p['name']:<26} {pos:<3}  {p['predicted_dk']:>5.1f} pts"
                    f"  {sal}{val}"
                )

        lines.append("")
        lines.append("─" * 60)
        lines.append("")

    out_path.write_text("\n".join(lines))
    log.info(f"Saved {out_path.name}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def run(season: int, week: int, salary_csv: str | None = None) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    proj_path    = PROCESSED / "projections_current.parquet"
    lineups_path = PROCESSED / f"lineups_{season}_w{week:02d}.csv"

    if not proj_path.exists():
        log.error(
            "projections_current.parquet not found.\n"
            f"  Run first: python3 src/projections.py --season {season} --week {week}"
        )
        return

    log.info(f"Season {season}  Week {week}")

    # ── Salary data ───────────────────────────────────────────────────────────
    salary_df = _load_salary_df(salary_csv, PROCESSED / "current_projections.parquet")
    if salary_df.empty:
        log.info("  No salary CSV — value/divergence columns will be empty")
    else:
        log.info(f"  Loaded {len(salary_df)} players from salary CSV")

    # ── Rankings ──────────────────────────────────────────────────────────────
    log.info("Building rankings ...")
    rankings = _build_rankings(proj_path, salary_df)
    _write_rankings(rankings, season, week)

    # ── Divergence ────────────────────────────────────────────────────────────
    _write_divergence(rankings, season, week)

    # ── Slate ─────────────────────────────────────────────────────────────────
    _write_slate(rankings, season, week)

    # ── DK upload ─────────────────────────────────────────────────────────────
    if lineups_path.exists():
        log.info("Building DK upload file ...")
        _write_dk_upload(lineups_path, season, week)
    else:
        log.warning(
            f"  No lineups file at {lineups_path.name} — skipping dk_upload.\n"
            f"  Run first: python3 src/optimizer.py --salary <DKSalaries.csv>"
            f" --season {season} --week {week}"
        )

    log.info(f"Outputs written to {OUTPUTS}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate final DFS deliverables from optimizer lineups and projections."
    )
    parser.add_argument("--season", type=int, required=True, help="Season year")
    parser.add_argument("--week",   type=int, required=True, help="Week number")
    parser.add_argument(
        "--salary", default=None,
        help="Path to DraftKings salary export CSV (enables salary, value, divergence)"
    )
    args = parser.parse_args()

    run(season=args.season, week=args.week, salary_csv=args.salary)
