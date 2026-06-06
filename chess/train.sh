#!/bin/bash
#
# Chess AlphaZero training loop: self-play -> train -> repeat.
# Usage: ./train.sh [num_iterations]   (default 10)

export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDNN_BENCHMARK=1

cd "$(dirname "$0")" || exit

NUM_ITERATIONS=${1:-10}

if ! [[ "$NUM_ITERATIONS" =~ ^[0-9]+$ ]] || [ "$NUM_ITERATIONS" -lt 1 ]; then
    echo "Error: Number of iterations must be a positive integer"
    echo "Usage: $0 [number_of_iterations]"
    exit 1
fi

echo "============================================"
echo "Chess AlphaZero Training Pipeline"
echo "============================================"
echo "Iterations to run: $NUM_ITERATIONS"
echo "============================================"

mkdir -p output_chess/logs
TIMING_LOG="output_chess/logs/timing_tracking.csv"
if [ ! -f "$TIMING_LOG" ]; then
    echo "iteration,start_time,end_time,duration_seconds,selfplay_seconds,training_seconds" > "$TIMING_LOG"
fi

START_TIME=$(date +%s)

for iteration in $(seq 1 "$NUM_ITERATIONS"); do
    ITER_START=$(date +%s)
    ITER_START_ISO=$(date '+%Y-%m-%d %H:%M:%S')

    echo ""
    echo "============================================"
    echo "ITERATION $iteration / $NUM_ITERATIONS"
    echo "Started at: $ITER_START_ISO"
    echo "============================================"

    echo "Phase 1/2: Generating self-play games..."
    SELFPLAY_START=$(date +%s)
    python3 selfplay.py
    if [ $? -ne 0 ]; then
        echo "Error: Self-play failed at iteration $iteration"
        exit 1
    fi
    SELFPLAY_END=$(date +%s)
    SELFPLAY_DURATION=$((SELFPLAY_END - SELFPLAY_START))

    echo ""
    echo "Phase 2/2: Training neural network..."
    TRAINING_START=$(date +%s)
    python3 train.py
    if [ $? -ne 0 ]; then
        echo "Error: Training failed at iteration $iteration"
        exit 1
    fi
    TRAINING_END=$(date +%s)
    TRAINING_DURATION=$((TRAINING_END - TRAINING_START))

    ITER_END=$(date +%s)
    ITER_DURATION=$((ITER_END - ITER_START))
    ITER_END_ISO=$(date '+%Y-%m-%d %H:%M:%S')

    actual_iter_num=$iteration
    iter_file="output_chess/logs/current_iteration.txt"
    if [ -f "$iter_file" ]; then
        actual_iter_num=$(cat "$iter_file")
    fi

    echo "$actual_iter_num,$ITER_START_ISO,$ITER_END_ISO,$ITER_DURATION,$SELFPLAY_DURATION,$TRAINING_DURATION" >> "$TIMING_LOG"

    echo ""
    echo "Iteration $iteration complete in $((ITER_DURATION / 60))m $((ITER_DURATION % 60))s"
    echo "  Self-play: $((SELFPLAY_DURATION / 60))m $((SELFPLAY_DURATION % 60))s | Training: $((TRAINING_DURATION / 60))m $((TRAINING_DURATION % 60))s"
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
echo ""
echo "============================================"
echo "TRAINING PIPELINE COMPLETE!"
echo "Total Duration: $((TOTAL_DURATION / 3600))h $(((TOTAL_DURATION % 3600) / 60))m $((TOTAL_DURATION % 60))s"
echo "Models saved in: output_chess/models/"
echo "============================================"
