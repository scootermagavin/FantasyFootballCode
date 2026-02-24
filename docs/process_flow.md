# DFS Football Pipeline — Process Flow & Architecture

## Ultimate Goal

Generate **profitable weekly DFS lineups** on DraftKings and FanDuel by building a projection
model that estimates each skill-position player's fantasy point output for an upcoming NFL
game, then optimally allocating salary budget across roster slots to maximize projected score.

The model works bottom-up:
1. Start with what Vegas says each **team** will score (macro context)
2. Estimate each team's **play volume** (pass attempts, rush attempts)
3. Allocate those plays to **individual players** based on historical usage rates
4. Convert allocated plays × efficiency rates into **projected fantasy points**
5. Feed projections + salaries into a **constrained optimizer** that picks the best lineup

---

## System Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          WEEKLY DFS PIPELINE                                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────┐
│                    RAW DATA LAYER                    │
│                                                      │
│  player_data_YYYY.csv (2000–2025)  ──────────────┐  │
│  player_data.csv (combined + PCA)  ──────────────┤  │
│  team_data.csv (per-team game log) ──────────────┤  │
│  team_data_rolling_three_game.csv  ──────────────┤  │
│  team_schedule.csv (Vegas lines)   ──────────────┤  │
│  fanduels_defense_data.csv         ──────────────┤  │
│  projections_raw.csv (weekly)      ──────────────┤  │
│  DraftKings salary CSV (weekly)    ──────────────┘  │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 1 │ ingest.py                               │
│                                                      │
│  • Load + validate all raw sources                   │
│  • Filter to skill positions + regular season        │
│  • Calculate PPR fantasy points                      │
│  • Output clean, typed DataFrames                    │
└────────────────────────┬────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
┌─────────────────────┐   ┌──────────────────────────┐
│ MODULE 2            │   │ MODULE 3 │ rushing.py      │
│ team_scoring.py     │   │                            │
│                     │   │ • Carry share per player   │
│ • Implied team      │   │ • Yards-per-carry trend    │
│   totals from Vegas │   │ • TD rate                  │
│ • Expected pass /   │   │ • Goal-line usage          │
│   rush play volume  │   │ • Lag features (2 wks)     │
│ • Opponent defense  │   └────────────┬───────────────┘
│   adjustments       │                │
│ • Pace & game       │   ┌────────────┴───────────────┐
│   script factors    │   │ MODULE 4 │ passing.py       │
└──────────┬──────────┘   │                             │
           │              │ • Target share per player   │
           │              │ • Air yards, YPT, catch %   │
           │              │ • Snap % / route rate        │
           │              │ • Depth-of-target trends    │
           │              │ • Lag features (2 wks)      │
           │              └────────────┬────────────────┘
           │                           │
           └──────────────┬────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 5 │ projections.py                          │
│                                                      │
│  • Combine team context + player usage features      │
│  • ElasticNet model (trained on historical data)     │
│  • Produce mean / downside / upside point estimates  │
│  • Validate against third-party projections_raw.csv  │
│  • Output: ranked player projections for the week    │
└────────────────────────┬────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 6 │ optimizer.py                            │
│                                                      │
│  • Load DraftKings salary CSV                        │
│  • Inject projected points per player                │
│  • Apply roster constraints (QB/RB/WR/TE/FLEX/DST)  │
│  • Apply stacking rules (QB + WR/TE same team)       │
│  • Apply exposure limits (max % across N lineups)    │
│  • Generate N optimized lineups                      │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 7 │ outputs.py                              │
│                                                      │
│  • Format lineups for DraftKings upload CSV          │
│  • Generate ranked projections table by position     │
│  • Generate model vs. third-party comparison report  │
│  • Log run parameters and model metrics              │
└─────────────────────────────────────────────────────┘
```

---

## Data Layer — Sources and Roles

| File | Grain | Key Columns | Role in Pipeline |
|---|---|---|---|
| `player_data_YYYY.csv` | Player × Game | player_id, franchise_id, position_id, season, week, all stat columns | Source for ingest.py; annual ETL input |
| `player_data.csv` | Player × Game | All above + PC01–PC66 PCA features | Primary model input with pre-computed athleticism features |
| `team_data.csv` | Team × Game | name, season, week, passing_epa, rushing_epa, successful_play_rate, def_pressure_rate | Team efficiency baseline for team_scoring.py |
| `team_data_rolling_three_game.csv` | Team × Game | All team_data cols × 4 rolling variants (avg, std, max, coef_var) | Ready-to-use feature matrix for team_scoring.py |
| `team_schedule.csv` | Game | season, week, home_team, away_team, spread_line, total_line, home_rest, away_rest | Vegas line → implied team totals; rest-day adjustment |
| `fanduels_defense_data.csv` | Team-Game | head_coach, offensive_scheme, defensive_alignment, EPA, time_of_possession, weather | Scheme/coordinator context; secondary opponent adjustment |
| `projections_raw.csv` | Player × Week | player_id, position_id, stat projections, draft_kings_player_id, fan_duel_player_id | (1) Third-party benchmark; (2) DK/FD platform ID keys |
| `DraftKings salary CSV` | Player × Week | Player Name, Position, Salary, Team, GameInfo | Salary constraint input to optimizer |

---

## Module Deep Dives

---

### MODULE 1 — `ingest.py`
**When it runs:** Once per season update, plus once each week to pick up projections_raw.csv

**What it does:**

```
For each year in [2000 .. current]:
    Load player_data_{year}.csv
    Filter: regular season only (playoffs == 0)
    Filter: skill positions only (drop OL, DEF, P, K)
    Calculate PPR fantasy points:
        FP = (passing_yds × 0.04) + (passing_td × 4) + (passing_int × -2)
           + (rushing_yds × 0.1) + (rushing_td × 6)
           + (receiving_rec × 1.0) + (receiving_yds × 0.1) + (receiving_td × 6)

Combine all years → combined_players.parquet
Load player_data.csv (PCA version) → players_with_pca.parquet
Load team_schedule.csv → schedule.parquet
Load projections_raw.csv → current_week_projections.parquet

Build team_id crosswalk:
    Map franchise codes (SEA, PIT...) ↔ full names (Seattle Seahawks)
    ↔ PFR codes (OTI, CLT...) ↔ DK abbreviations
    → team_crosswalk.parquet
```

**Output files:**
- `data/processed/combined_players.parquet`
- `data/processed/players_with_pca.parquet`
- `data/processed/schedule.parquet`
- `data/processed/current_week_projections.parquet`
- `data/processed/team_crosswalk.parquet`

**Why this matters:** Every downstream module depends on a single clean data contract.
Bad data caught here never corrupts the model.

---

### MODULE 2 — `team_scoring.py`
**When it runs:** Each week, after team_schedule.csv is refreshed with the current week's Vegas lines

**The core idea:** Before you can project a player's points, you must know how many
scoring opportunities their team will generate. Vegas lines are the most efficient
single signal for this.

**Key logic:**

```
For each game in the current week's schedule:

    1. DERIVE IMPLIED TEAM TOTALS from Vegas lines:
       spread = how many points home team is favored by (positive = home favorite)
       total  = expected combined score

       home_implied = (total + spread) / 2
       away_implied = (total - spread) / 2

       Example: O/U = 47, spread = -6.5 (home favored)
           home_implied = (47 + 6.5) / 2 = 26.75
           away_implied = (47 - 6.5) / 2 = 20.25

    2. ESTIMATE PLAY VOLUME from implied scoring:
       Historical avg: ~1.0 pass play per point of implied total (rough baseline)
       Use team_data_rolling_three_game.csv to find each team's actual pace:
           - attempts_3_game_avg
           - carries_3_game_avg
           - pass/rush split

       expected_pass_attempts = team_pass_rate × total_expected_plays
       expected_rush_attempts = team_rush_rate × total_expected_plays

    3. ADJUST FOR OPPONENT DEFENSE:
       Use def_pressure_rate_3_game_avg → QB under pressure → fewer completions
       Use def_tackles_for_loss_3_game_avg → RB efficiency suppression
       Use def_pass_defended_3_game_avg → WR/TE target suppression

    4. GAME SCRIPT FACTOR:
       Large favorites tend to run more in 2nd half
       Large underdogs tend to pass more (garbage time)
       home_rest / away_rest → rest advantage adjustment

Output per team per game:
    expected_pass_attempts, expected_rush_attempts,
    opponent_pass_def_rank, opponent_rush_def_rank,
    game_script_factor, implied_team_total
```

**Output file:** `data/processed/team_game_context.parquet`

---

### MODULE 3 — `rushing.py`
**When it runs:** Each week, after ingest.py and team_scoring.py complete

**The core idea:** A running back's expected output = (his share of team rushes) ×
(team's expected rush volume) × (his per-carry efficiency).

**Key logic:**

```
For each player in the current week's roster (RBs + rushing QBs):

    USAGE FEATURES (rolling 2-game window):
        carry_share     = player_rushing_att / team_rushing_att
        snap_pct        = offensive_snapcount_percentage
        goal_line_share = rushing_att near goal line / team goal_line_carries
        td_rate         = rushing_td / rushing_att
        yards_per_carry = rushing_yds / rushing_att

    OPPONENT ADJUSTMENT:
        apply opponent_rush_def_rank from team_scoring.py
        → scale yards_per_carry and td_rate expectation

    PROJECTION INPUTS (fed to projections.py):
        expected_carries      = carry_share × team_expected_rush_attempts
        projected_rush_yds    = expected_carries × adjusted_yards_per_carry
        projected_rush_td     = expected_carries × adjusted_td_rate
        projected_rush_points = (projected_rush_yds × 0.1) + (projected_rush_td × 6)

    LAG FEATURES (for model training context):
        lag_rushing_att_1, lag_rushing_yds_1, lag_carry_share_1
        lag_rushing_att_2, lag_rushing_yds_2, lag_carry_share_2
```

**Output file:** `data/processed/rushing_features.parquet`

---

### MODULE 4 — `passing.py`
**When it runs:** Each week, after ingest.py and team_scoring.py complete

**The core idea:** A receiver's output = (his share of team targets) × (team's expected
pass volume) × (his per-target efficiency). QB output = (team's expected pass attempts)
× (his per-attempt efficiency).

**Key logic:**

```
For each player in the current week's roster (WR, TE, QB + receiving RBs):

    USAGE FEATURES (rolling 2-game window):
        target_share    = player_receiving_tar / team_passing_att
        air_yard_share  = player_receiving_air_yards / team_passing_air_yards
        snap_pct        = offensive_snapcount_percentage
        route_rate      = routes_run / team_routes (if available)
        catch_rate      = receiving_rec / receiving_tar
        yards_per_target = receiving_yds / receiving_tar
        td_rate         = receiving_td / receiving_tar
        yard_per_completion = passing_yds / passing_cmp  (QB only)

    OPPONENT ADJUSTMENT:
        apply opponent_pass_def_rank from team_scoring.py
        → scale yards_per_target and td_rate expectation

    PROJECTION INPUTS (fed to projections.py):
        expected_targets       = target_share × team_expected_pass_attempts
        projected_rec          = expected_targets × catch_rate
        projected_rec_yds      = expected_targets × yards_per_target
        projected_rec_td       = expected_targets × adjusted_td_rate
        projected_rec_points   = (projected_rec × 1.0)
                               + (projected_rec_yds × 0.1)
                               + (projected_rec_td × 6)

    LAG FEATURES:
        lag_receiving_tar_1, lag_receiving_yds_1, lag_target_share_1
        lag_receiving_tar_2, lag_receiving_yds_2, lag_target_share_2

For QBs:
    expected_pass_att = team_expected_pass_attempts × starter_probability
    projected_pass_yds = expected_pass_att × completion_rate × yard_per_completion
    projected_pass_td  = expected_pass_att × td_rate
    projected_pass_int = expected_pass_att × int_rate
    projected_qb_points = (projected_pass_yds × 0.04)
                        + (projected_pass_td × 4)
                        + (projected_pass_int × -2)
                        + projected_rush_points (from rushing.py)
```

**Output file:** `data/processed/passing_features.parquet`

---

### MODULE 5 — `projections.py`
**When it runs:** Each week, after rushing.py and passing.py complete

**The core idea:** Combine the bottom-up usage projections (Modules 3 & 4) with a
machine learning model trained on historical weekly outcomes. The model learns which
signals most reliably predict fantasy performance.

**Key logic:**

```
TRAINING PHASE (run once per season, or on demand):
    Load historical combined_players.parquet (2000–2024)
    Build feature matrix per player-week:
        From rushing.py output: carry_share lags, yards_per_carry lags
        From passing.py output: target_share lags, yards_per_target lags
        From team_scoring.py:   implied_team_total, opponent_def_rank
        From player_data.csv:   PC01–PC66 PCA athleticism features,
                                snap%, vegas_line, game_location
    Target variable: actual FantasyPoints (PPR)

    Pipeline:
        StandardScaler → ElasticNet (alpha, l1_ratio via GridSearchCV)

    Train/test split:
        Train on seasons 2000–2022
        Validate on seasons 2023–2024

    Evaluate: Mean Absolute Error per position
    Save trained model → models/projection_model_{position}.pkl

PREDICTION PHASE (run weekly):
    For each active player in current week:
        Build same feature vector (using current-week data)
        Predict: proj_mean_fp, proj_downside_fp, proj_upside_fp

    Cross-check against projections_raw.csv (third-party):
        Flag players where model diverges > 30% from consensus
        These are the highest-leverage decisions

    Output per player:
        player_id, name, position, team, opponent,
        model_proj_fp, model_proj_downside, model_proj_upside,
        consensus_proj_fp, value_score (proj_fp / salary × 1000)
```

**Output file:** `data/processed/weekly_projections.parquet`

---

### MODULE 6 — `optimizer.py`
**When it runs:** Each week, after projections.py, with the freshly downloaded DK salary CSV

**The core idea:** Given a set of projections, find the combination of players that
maximizes total projected points while satisfying all DraftKings roster and salary rules.
This is an integer programming problem solved by pydfs-lineup-optimizer.

**DraftKings NFL Roster:**
```
Slot        Count   Eligible Positions
────────────────────────────────────────
QB            1     QB
RB            2     RB
WR            3     WR
TE            1     TE
FLEX          1     RB, WR, or TE
DST           1     Team Defense
────────────────────────────────────────
Salary Cap:       $50,000
```

**Key logic:**

```
SETUP:
    Load DK salary CSV → map players using projections_raw.draft_kings_player_id
    Inject model_proj_fp as each player's projected points
    Initialize: get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)

CONSTRAINTS TO APPLY:
    1. Salary cap: $50,000 hard limit
    2. Stacking rule: QB must share a team with at least 1 WR or TE
       (correlation — QB and his receiver score together on touchdowns)
    3. Game stack option: include a player from the opposing team
       (exposure to a high-scoring game from both sides)
    4. Max player exposure: each player appears in ≤ 35% of lineups
       (for tournament entry diversity)
    5. Lock / exclude: manually pin or ban specific players

LINEUP GENERATION:
    Generate N lineups (default: 20 for GPP tournaments, 3–5 for cash games)
    Each lineup uses the constraint set above

VALUE SCORE CALCULATION:
    value_score = projected_fp / (salary / 1000)
    Players with value_score > 2.5 are "value plays"
    Players with value_score < 1.5 are "salary traps"
```

**Output file:** `data/processed/optimized_lineups.parquet`

---

### MODULE 7 — `outputs.py`
**When it runs:** After optimizer.py, as the final step before submission

**What it produces:**

```
1. DraftKings upload CSV (lineups.csv)
   Format: QB,RB,RB,WR,WR,WR,TE,FLEX,DST
   One row per lineup, player names exactly as DK expects them

2. Projections report (rankings.csv)
   Columns: Rank, Player, Pos, Team, Opp, Salary, ModelProj,
            ConsensusProj, ValueScore, CarryShare/TargetShare,
            ImpliedTeamTotal, OpponentRank
   Sorted by value_score within each position

3. Model vs. consensus comparison (divergence_report.csv)
   Flags players where model and third-party projections diverge > 30%
   These are the key judgment calls before submitting

4. Run log (run_log.json)
   Records: season, week, model version, MAE on recent weeks,
            parameters used, timestamp
```

---

## Join Key Map — How the Tables Connect

```
                         ┌────────────────────────────────────┐
                         │         player_data.csv             │
                         │  player_id, franchise_id, season,   │
                         │  week, position_id, all stats        │
                         └──────────────┬──────────────────────┘
                                        │ franchise_id
                                        │ (e.g. "SEA")
                                        ▼
                         ┌────────────────────────────────────┐
                         │        team_crosswalk.parquet       │
                         │  franchise_id ↔ full_name          │
                         │  ↔ pfr_code ↔ dk_abbrev            │
                         └──┬──────────┬───────────┬──────────┘
                            │          │           │
                            ▼          ▼           ▼
              ┌─────────────────┐  ┌──────────┐  ┌────────────────────────┐
              │  team_data.csv   │  │ schedule │  │ fanduels_defense_data  │
              │  name (full)     │  │ home/away│  │ team_id (pfr_code)     │
              │  season, week    │  │ season,  │  │ season, week           │
              └─────────────────┘  │ week     │  └────────────────────────┘
                                   └──────────┘
                                        │
                                        │ draft_kings_player_id
                                        ▼
                         ┌────────────────────────────────────┐
                         │       projections_raw.csv           │
                         │  player_id, draft_kings_player_id  │
                         │  fan_duel_player_id, season, week  │
                         └──────────────┬──────────────────────┘
                                        │ draft_kings_player_id
                                        ▼
                         ┌────────────────────────────────────┐
                         │       DK Salary CSV (weekly)        │
                         │  Player Name, Salary, Position      │
                         └────────────────────────────────────┘
```

**Critical note:** Team identifiers are inconsistent across sources:
- `player_data.csv` → short franchise codes: `SEA`, `PIT`, `KAN`
- `team_data.csv` → full names: `Seattle Seahawks`
- `fanduels_defense_data.csv` → PFR codes: `OTI` (Titans), `CLT` (Colts)
- `team_schedule.csv` → both formats side by side

The team_crosswalk table built in `ingest.py` is the linchpin that makes all joins work.
Without it, every other module will produce incorrect merged data.

---

## Weekly Run Process (Game Week Checklist)

```
TUESDAY (after MNF):
    [ ] Download DraftKings salary CSV for the upcoming week
    [ ] Refresh projections_raw.csv from your data source
    [ ] Confirm team_schedule.csv has updated Vegas lines for the week
    [ ] Run: python ingest.py --week {W} --season {YYYY}

WEDNESDAY–THURSDAY:
    [ ] Run: python team_scoring.py --week {W}
    [ ] Run: python rushing.py --week {W}
    [ ] Run: python passing.py --week {W}
    [ ] Review rankings.csv — identify flagged divergences
    [ ] Manual overrides: injury reports, weather, late scratches

FRIDAY:
    [ ] Run: python projections.py --week {W}
    [ ] Run: python optimizer.py --week {W} --lineups 20
    [ ] Review optimized_lineups.parquet for sanity

SATURDAY:
    [ ] Final injury check (official injury reports drop ~4pm)
    [ ] Re-run optimizer.py if any projected starters are ruled out
    [ ] Run: python outputs.py --week {W}

SUNDAY MORNING (before noon kickoff):
    [ ] Upload lineups.csv to DraftKings
    [ ] Confirm submission accepted
```

---

## Key Parameters & Scoring Weights

```python
# PPR Fantasy Scoring (DraftKings)
FANTASY_WEIGHTS = {
    "passing_yds":    0.04,   # 1 pt per 25 yds
    "passing_td":     4,
    "passing_int":    -1,     # DK is -1, not -2
    "rushing_yds":    0.1,    # 1 pt per 10 yds
    "rushing_td":     6,
    "receiving_rec":  1,      # PPR: 1 pt per reception
    "receiving_yds":  0.1,
    "receiving_td":   6,
    "passing_two_pt": 2,
    "rushing_two_pt": 2,
    "fumbles_lost":   -1,
}

# NOTE: player_data_processing.py uses passing_int: -2
# DraftKings is actually -1. This must be corrected in ingest.py.

# DraftKings Salary Cap
DK_SALARY_CAP = 50_000

# Optimizer Settings
N_LINEUPS      = 20        # GPP tournament entry count
MAX_EXPOSURE   = 0.35      # Max fraction of lineups a player appears in
MIN_STACK_SAME_TEAM = 1    # Min WR/TE paired with QB from same team
```

---

## What Could Go Wrong (Risk Register)

| Risk | Where it Occurs | Mitigation |
|---|---|---|
| Team ID mismatch across sources | Every join | Build crosswalk in ingest.py; validate row counts after merge |
| Stale injury data | projections.py, optimizer.py | Pull official injury report before final run; support player lock/exclude |
| Vegas line not updated | team_scoring.py | Validate team_schedule.csv has current week data before running |
| Player in projections_raw but not DK salary CSV | optimizer.py | Left-join DK CSV; log unmatched players |
| Model trained on pre-rule-change data | projections.py | Weight recent seasons more heavily; retrain at season start |
| DK CSV column format change | optimizer.py | Schema validation step on CSV load |
| passing_int scoring weight mismatch (-2 vs -1) | ingest.py → everywhere | Use DK-correct weight (-1) in ingest.py; document legacy discrepancy |
