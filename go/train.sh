#!/bin/bash
# train.sh — Full selfplay → train → gate loop.
# Run from the go/ directory: bash train.sh

set -euo pipefail
cd "$(dirname "$0")"

while true; do
    echo "Phase 1/3: Self-play..."
    python3 selfplay_cpp_runner.py || { echo "ERROR: selfplay failed"; exit 1; }

    echo "Phase 2/3: Training..."
    python3 train.py || { echo "ERROR: training failed"; exit 1; }

    NEW_ITER=$(cat logs_9x9/current_iteration.txt)
    BEST_ITER=$(cat models_9x9/current_best_iter.txt 2>/dev/null || echo "$NEW_ITER")

    echo "Phase 3/3: Gating iter $NEW_ITER vs $BEST_ITER..."

    # Export TorchScript models if not already present
    NEW_TS="models_9x9/${NEW_ITER}_ts.pt"
    BEST_TS="models_9x9/${BEST_ITER}_ts.pt"

    [ -f "$NEW_TS"  ] || BOARD_SIZE=9 python3 export_model.py \
        "models_9x9/${NEW_ITER}_best_model.pt" "$NEW_TS"
    [ -f "$BEST_TS" ] || BOARD_SIZE=9 python3 export_model.py \
        "models_9x9/${BEST_ITER}_best_model.pt" "$BEST_TS"

    if python3 gatekeeper_runner.py "$NEW_TS" "$BEST_TS" "$NEW_ITER" "$BEST_ITER"; then
        echo "iter $NEW_ITER ACCEPTED as new selfplay model"
    else
        echo "iter $NEW_ITER REJECTED — keeping iter $BEST_ITER"
    fi

    echo "=== Iteration $NEW_ITER complete at $(date) ==="
done
