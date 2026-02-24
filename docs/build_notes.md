# Build Notes — DFS Pipeline

Running reference for decisions made, bugs fixed, and context needed to resume.
Update this file whenever a non-obvious finding is confirmed.

---

## Environment

```
Python 3.11.3
venv: /Users/claude-dev/projects/financial-models/dfs-football/venv/
Activate: source venv/bin/activate
pandas 3.0.1 | scikit-learn | pyarrow | pydfs-lineup-optimizer
```

Run any module from the project root:
```bash
source venv/bin/activate
python3 src/ingest.py
python3 src/team_scoring.py --season 2024 --week 14
```

---

## Data Layer

### Raw data location
`data/raw/historical/` — all source files

### Processed outputs
`data/processed/` — parquet files written by ingest.py, consumed by all downstream modules

| File | Rows | Source |
|---|---|---|
| `team_crosswalk.parquet` | 32 | Hardcoded in ingest.py |
| `players.parquet` | 171,017 | player_data.csv (skill pos, regular season only) |
| `team_stats.parquet` | 14,512 | team_data.csv + crosswalk |
| `team_rolling.parquet` | 14,512 | team_data_rolling_three_game.csv + crosswalk |
| `schedule.parquet` | 7,276 | team_schedule.csv + abbrev lookup |
| `defense_context.parquet` | 14,034 | fanduels_defense_data.csv |
| `current_projections.parquet` | 2,070 | projections_raw.csv |
| `team_game_context.parquet` | varies | written by team_scoring.py |

---

## Confirmed Data Conventions (Critical)

### 1. Fantasy scoring target: use `dk_points` directly
- `player_data.csv` has `draftkings_points` (actual DK platform points, renamed to `dk_points` in ingest.py)
- **Do NOT recalculate from raw stats.** The legacy `player_data_processing.py` used `-2` for INT; DraftKings is `-1`. The actual dk_points column is already correct.
- `dk_points` null rate: 0.0% for skill positions in regular season. Safe to use as training target.

### 2. Team ID system
- **Primary key: PFR franchise codes** (`CRD`, `GNB`, `CLT`, `OTI`, `HTX`, `RAV`, etc.)
- `player_data.csv` `franchise_id` column = PFR codes → renamed to `pfr_code` in ingest.py
- `fanduels_defense_data.csv` `player_team_pfr_franchise_code_id` = same PFR codes
- `team_data.csv` and `team_data_rolling_three_game.csv` use **full current team names** (normalized — Rams = "Los Angeles Rams" even for 1999 St. Louis data)
- `team_schedule.csv` uses abbreviations with historical variants (STL, OAK, SD for relocated teams)
- Full crosswalk is hardcoded in `ingest.py: _CROSSWALK`. 32 teams, each with pfr_code + full_name + sched_abbrev.
- Crosswalk handles historical schedule abbreviations via `get_abbrev_lookup()` dict.
- 16 null-name rows in team_data and team_rolling (seasons 2001-2002) — dropped in ingest.py as data artifact.

### 3. Spread sign convention in team_schedule.csv
**OPPOSITE of standard ATS notation:**
- `spread_line` POSITIVE = home team is FAVORED (gives points)
- `spread_line` NEGATIVE = home team is UNDERDOG (gets points)

Correct implied total formulas:
```python
home_implied = (total_line + spread_line) / 2
away_implied = (total_line - spread_line) / 2
```

Verified against PHI vs CAR 2024 Week 14: spread=14.0, total=44.5
→ PHI (home) implied = (44.5 + 14) / 2 = 29.25 ✓
→ CAR (away) implied = (44.5 - 14) / 2 = 15.25 ✓

### 4. Rolling feature timing
- `team_data_rolling_three_game.csv` rows are indexed by the ACTUAL game week
- Rolling averages at week N include week N's data (retrospective)
- For predicting week W, use rolling features from week W-1: `rolling[(season==S) & (week < W)].sort_values('week').iloc[-1]`
- Week 1 (no prior data) falls back to prior season's last row

### 5. `off_n` column in team_data
- NOT total plays. Appears to be drive count or similar. Value ~28-32 vs actual plays of ~60-80.
- For play-volume estimation, use `attempts_3_game_avg + carries_3_game_avg`

---

## Modules Built

### `ingest.py` ✅
- Runs once per season (or when source files are refreshed)
- No CLI args — just `python3 src/ingest.py`
- All downstream modules read from `data/processed/` parquet files
- Re-run this whenever any raw data file is updated

### `team_scoring.py` ✅
- CLI: `python3 src/team_scoring.py --season YYYY --week W`
- Outputs `data/processed/team_game_context.parquet`
- One row per team per game (2 rows per game)
- **Bug fixed:** `BIG_FAVORITE_THRESHOLD` was `-7.0`, triggering on almost every game. Corrected to `+7.0` to match dataset's positive-favorite convention.
- Game-script factor only fires when |spread| > 7 (meaningful favorite/underdog situation)
- Play-volume estimates: `(attempts_3_game_avg + carries_3_game_avg) × (implied_total / 23.0) × pass_rate_adj`
- Defensive adjustment multipliers (`pass_def_adj`, `rush_def_adj`) are passed downstream to rushing.py and passing.py — they are NOT applied inside team_scoring.py

---

## Modules Pending

| Module | Depends On | Key Task |
|---|---|---|
| `rushing.py` | ingest.py, team_scoring.py | Carry share, YPC, TD rate, lag features per RB |
| `passing.py` | ingest.py, team_scoring.py | Target share, catch rate, YPT lag features per WR/TE/QB |
| `projections.py` | rushing.py, passing.py | ElasticNet model, train on dk_points, predict weekly output |
| `optimizer.py` | projections.py | pydfs, DK salary CSV, stacking constraints |
| `outputs.py` | optimizer.py | DK upload CSV, rankings report, divergence report |

---

## Notes for Resuming `rushing.py`

Start point: `data/processed/players.parquet` + `data/processed/team_game_context.parquet`

Key features to build:
- `carry_share` = player `rushing_att` / team total `rushing_att` for that week (need team-level denominator from team_stats)
- `yards_per_carry` = `rushing_yds` / `rushing_att` (handle div-by-zero)
- `rushing_td_rate` = `rushing_td` / `rushing_att`
- `snap_pct` = `offensive_snapcount_percentage` (already in players.parquet)
- Lag features: 1-week and 2-week lags of the above, grouped by player_id
- Apply `rush_def_adj` from team_game_context to scale projected rush production

Join: `players.parquet` → join on `pfr_code, season, week` → `team_stats.parquet` to get team-level rushing totals for denominator
Then join to `team_game_context.parquet` on `pfr_code, season, week` for the upcoming game context.

Watch out: team totals include ALL rushers (QB sneaks, etc.). May want to filter to just official rushes.

---

## Notes for Resuming `passing.py`

Key features:
- `target_share` = player `receiving_tar` / team total `passing_att`
- `catch_rate` = `receiving_rec` / `receiving_tar`
- `yards_per_target` = `receiving_yds` / `receiving_tar`
- `air_yard_share` = `receiving_air_yards` / team total (check if air_yards exists in player data)
- QB: `completion_pct`, `yards_per_attempt`, `td_rate`, `int_rate`, `yard_per_completion`
- Lag features: 1- and 2-week lags, by player_id

---

## Notes for Resuming `projections.py`

- Training target: `dk_points` (already in players.parquet)
- Train/test split: train on seasons 2000–2022, validate on 2023–2024
- Position-specific models: QB, RB, WR, TE separately
- Feature matrix: rushing/passing features + team_game_context (implied_total, expected_pass_att, expected_rush_att, pass_def_adj, rush_def_adj) + PCA features (PC01–PC66) from players.parquet
- Pipeline: StandardScaler → ElasticNet with GridSearchCV over alpha and l1_ratio
- Cross-check output against `current_projections.parquet` (third-party, col `proj_mean_fanduel_points`)
- Note: projections_raw.csv Version column has "Avg"/"Min"/"Max" variants — filter to "Avg" for comparison

---

## Notes for Resuming `optimizer.py`

- Join DK salary CSV to projections via `draft_kings_player_id` from `current_projections.parquet`
- `projections_raw.csv` has `draft_kings_player_id` — this is the bridge between internal player_id and DK's player identification
- DK roster: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DST — salary cap $50,000
- pydfs init: `get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)`
- Stacking: QB paired with ≥1 WR or TE from same team

---

## Known Issues / Open Questions

1. **PHI carry estimate high:** PHI week 14 2024 estimated at ~47 carries but actual was ~20. This is because PHI's rolling carry average (39/game) × high implied total (29.25) inflates the count. projections.py will calibrate this when trained against actual outcomes — not a blocker.

2. **play volume floor at 40% pass:** PHI (big favorite, run-heavy) hits the 40% pass floor. Consider reviewing this floor if extreme offenses consistently clip it.

3. **projections_raw.csv is 2025 Week 22 (Super Bowl):** Only covers one week. Will need a refresh process for each week of the 2025 season. The `load_current_projections()` function in ingest.py is designed for weekly replacement — just overwrite the file and re-run ingest.py.

4. **team_game_context.parquet is overwritten each run:** It only holds one week at a time. If you want to store historical context for model training, team_scoring.py would need a loop mode.
