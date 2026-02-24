#!/usr/bin/env bash
# run_week.sh — DFS Football Weekly Pipeline
#
# Downloads GridironAI data, runs the full pipeline, and produces final outputs.
#
# Usage:
#   ./run_week.sh <season> <week> [/path/to/gridironai-dfs-{season}-{week}.csv]
#
# Examples:
#   ./run_week.sh 2025 14                          # no salary CSV (skips optimizer)
#   ./run_week.sh 2025 14 "legacy/DK 2025 Weekly Salary/gridironai-dfs-2025-14.csv"
#
# Salary CSV: GridironAI format (gridironai-dfs-*.csv) or standard DK export.

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <season> <week> [gridironai-dfs-{season}-{week}.csv]"
    exit 1
fi

SEASON=$1
WEEK=$2
SALARY_CSV=${3:-""}

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/data/raw/historical"
VENV="${SCRIPT_DIR}/venv/bin/activate"

# GridironAI download directory.
# GridironAI always downloads to this exact subfolder inside ~/Downloads.
# Override with GRAI_DIR env var if your path differs.
GRAI_DIR="${GRAI_DIR:-${HOME}/Downloads/Users/andytroiano/football-ai/R_Code/model_project/data/website_load}"

# ── Helpers ───────────────────────────────────────────────────────────────────
step=0
total_steps=7
[[ -z "$SALARY_CSV" ]] && total_steps=6

log_step() {
    step=$((step + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${step}/${total_steps}] $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

start_time=$(date +%s)
elapsed() { echo $(( $(date +%s) - start_time ))s; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DFS Football Pipeline  |  Season ${SEASON}  Week ${WEEK}"
[[ -n "$SALARY_CSV" ]] && echo "  Salary CSV: $(basename "$SALARY_CSV")"
echo "════════════════════════════════════════════════════════════"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: virtualenv not found at ${VENV}"
    echo "       Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
# shellcheck disable=SC1090
source "$VENV"

# ── Step 1: Sync GridironAI data ──────────────────────────────────────────────
log_step "Syncing GridironAI data"

if [[ ! -d "$GRAI_DIR" ]]; then
    echo "WARNING: GridironAI directory not found: ${GRAI_DIR}"
    echo "         Set GRAI_DIR env var or download files manually to ${RAW_DIR}/"
    echo "         Skipping sync — using existing raw data."
else
    echo "  Copying from: ${GRAI_DIR}"
    cp "${GRAI_DIR}"/*.csv "${RAW_DIR}/"
    echo "  Synced $(ls "${GRAI_DIR}"/*.csv | wc -l | tr -d ' ') CSV files"
fi

# ── Step 2: Ingest ────────────────────────────────────────────────────────────
log_step "Ingesting raw data → parquets"
python3 "${SCRIPT_DIR}/src/ingest.py"

# ── Step 3: Team game context (Vegas lines + expected volume) ─────────────────
log_step "Building team game context (Season ${SEASON} Week ${WEEK})"
python3 "${SCRIPT_DIR}/src/team_scoring.py" --season "$SEASON" --week "$WEEK"

# ── Step 4: Rushing features ──────────────────────────────────────────────────
log_step "Building rushing features"
python3 "${SCRIPT_DIR}/src/rushing.py" --season "$SEASON" --week "$WEEK"

# ── Step 5: Passing features ──────────────────────────────────────────────────
log_step "Building passing features"
python3 "${SCRIPT_DIR}/src/passing.py" --season "$SEASON" --week "$WEEK"

# ── Step 6: Projections ───────────────────────────────────────────────────────
log_step "Generating model projections"
python3 "${SCRIPT_DIR}/src/projections.py" --season "$SEASON" --week "$WEEK"

# ── Step 7: Optimizer + Outputs ───────────────────────────────────────────────
if [[ -n "$SALARY_CSV" ]]; then
    if [[ ! -f "$SALARY_CSV" ]]; then
        echo ""
        echo "ERROR: Salary CSV not found: ${SALARY_CSV}"
        exit 1
    fi
    log_step "Optimizing lineups + generating outputs"
    python3 "${SCRIPT_DIR}/src/optimizer.py" \
        --salary "$SALARY_CSV" \
        --season "$SEASON" \
        --week "$WEEK"
    python3 "${SCRIPT_DIR}/src/outputs.py" \
        --season "$SEASON" \
        --week "$WEEK" \
        --salary "$SALARY_CSV"
else
    log_step "Generating outputs (no salary CSV — optimizer skipped)"
    python3 "${SCRIPT_DIR}/src/outputs.py" \
        --season "$SEASON" \
        --week "$WEEK"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done in $(elapsed)  |  Outputs: ${SCRIPT_DIR}/data/outputs/"
echo "════════════════════════════════════════════════════════════"
echo ""
