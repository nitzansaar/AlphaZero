#!/bin/bash
# train.sh — Continuous AlphaZero selfplay → train loop.
# Runs forever until interrupted (Ctrl-C / kill).
# Run from the go/ directory: bash train.sh

set -euo pipefail
cd "$(dirname "$0")"

LOGDIR=$(python3 -c "from config import Config as cfg; print(cfg.LOGDIR)")

START_TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')
NUM_ITERATIONS=0

trap 'echo ""; echo "=== Stopped after $NUM_ITERATIONS iterations ==="; exit 0' INT TERM

while true; do
    ((++NUM_ITERATIONS))
    echo "=== Iteration $NUM_ITERATIONS at $(date) ==="

    echo "Phase 1/2: Self-play..."
    python3 selfplay_cpp_runner.py || { echo "ERROR: selfplay failed"; exit 1; }

    echo "Phase 2/2: Training..."
    python3 train.py || { echo "ERROR: training failed"; exit 1; }

    NEW_ITER=$(cat "${LOGDIR}/current_iteration.txt")
    echo "=== Iteration $NEW_ITER complete at $(date) ==="
done
