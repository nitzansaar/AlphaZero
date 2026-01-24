#!/bin/bash

# RTX 5090 Optimization: Environment Variables
# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
export NVIDIA_TF32_OVERRIDE=1

# Optimize CUDA memory allocation for better performance
# Use the new CUDA memory allocator with expandable segments
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# Enable cuDNN auto-tuning
export CUDNN_BENCHMARK=1

# Change to src directory for relative imports
cd "$(dirname "$0")" || exit

# Number of iterations to run (default: 10, recommended for strong play)
NUM_ITERATIONS=${1:-10}

# Validate input
if ! [[ "$NUM_ITERATIONS" =~ ^[0-9]+$ ]] || [ "$NUM_ITERATIONS" -lt 1 ]; then
    echo "Error: Number of iterations must be a positive integer"
    echo "Usage: $0 [number_of_iterations]"
    echo "Example: $0 10  (runs 10 iterations)"
    exit 1
fi

echo "============================================"
echo "AlphaGo Zero Training Pipeline"
echo "============================================"
echo "Iterations to run: $NUM_ITERATIONS"
echo "Recommended: 10 iterations for strong play"
echo "Expected time: ~$((NUM_ITERATIONS * 1))-$(($NUM_ITERATIONS * 2)) hours"
echo "============================================"
echo ""

START_TIME=$(date +%s)
START_TIME_ISO=$(date -d "@$START_TIME" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')

# Create timing log file
TIMING_LOG="output_tictac/logs/timing_tracking.csv"
mkdir -p output_tictac/logs
if [ ! -f "$TIMING_LOG" ]; then
    echo "iteration,start_time,end_time,duration_seconds,selfplay_seconds,training_seconds,eval_seconds" > "$TIMING_LOG"
fi

# Run iterations
for iteration in $(seq 1 $NUM_ITERATIONS); do
    ITER_START=$(date +%s)
    ITER_START_ISO=$(date -d "@$ITER_START" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')
    
    echo ""
    echo "============================================"
    echo "ITERATION $iteration / $NUM_ITERATIONS"
    echo "============================================"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Self-play phase
    echo "Phase 1/3: Generating self-play games..."
    SELFPLAY_START=$(date +%s)
    python3 selfplay.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Self-play failed at iteration $iteration"
        exit 1
    fi
    SELFPLAY_END=$(date +%s)
    SELFPLAY_DURATION=$((SELFPLAY_END - SELFPLAY_START))
    
    echo ""
    echo "Self-play complete. Starting training..."
    
    # Training phase
    echo "Phase 2/3: Training neural network..."
    TRAINING_START=$(date +%s)
    python3 train.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Training failed at iteration $iteration"
        exit 1
    fi
    TRAINING_END=$(date +%s)
    TRAINING_DURATION=$((TRAINING_END - TRAINING_START))
    
    # Evaluation phase (compare with previous model)
    echo ""
    echo "Phase 3/3: Evaluating model (ELO calculation)..."
    EVAL_START=$(date +%s)
    # Get the actual iteration number from the file saved by train.py
    iter_file="output_tictac/logs/current_iteration.txt"
    if [ -f "$iter_file" ]; then
        actual_iter=$(cat "$iter_file")
        python3 evaluate_model.py $actual_iter 2>/dev/null || echo "Evaluation skipped (no previous model to compare)"
    else
        # Fallback: try to detect from latest model
        python3 evaluate_model.py 2>/dev/null || echo "Evaluation skipped (no previous model to compare)"
    fi
    EVAL_END=$(date +%s)
    EVAL_DURATION=$((EVAL_END - EVAL_START))
    
    ITER_END=$(date +%s)
    ITER_DURATION=$((ITER_END - ITER_START))
    ITER_MIN=$((ITER_DURATION / 60))
    ITER_SEC=$((ITER_DURATION % 60))
    
    # Get actual iteration number for logging
    actual_iter_num=$iteration
    if [ -f "$iter_file" ]; then
        actual_iter_num=$(cat "$iter_file")
    fi
    
    # Save timing data
    ITER_END_ISO=$(date -d "@$ITER_END" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')
    echo "$actual_iter_num,$ITER_START_ISO,$ITER_END_ISO,$ITER_DURATION,$SELFPLAY_DURATION,$TRAINING_DURATION,$EVAL_DURATION" >> "$TIMING_LOG"
    
    echo ""
    echo "Iteration $iteration complete in ${ITER_MIN}m ${ITER_SEC}s"
    echo "  Breakdown: Self-play: $((SELFPLAY_DURATION / 60))m $((SELFPLAY_DURATION % 60))s | Training: $((TRAINING_DURATION / 60))m $((TRAINING_DURATION % 60))s | Eval: $((EVAL_DURATION / 60))m $((EVAL_DURATION % 60))s"
    
    # Show progress
    if [ $iteration -lt $NUM_ITERATIONS ]; then
        REMAINING=$((NUM_ITERATIONS - iteration))
        ELAPSED=$((ITER_END - START_TIME))
        AVG_TIME=$((ELAPSED / iteration))
        ESTIMATED=$((AVG_TIME * REMAINING))
        EST_MIN=$((ESTIMATED / 60))
        echo "Progress: $iteration/$NUM_ITERATIONS iterations complete"
        echo "Estimated time remaining: ~${EST_MIN} minutes"
        echo ""
    fi
done

END_TIME=$(date +%s)
END_TIME_ISO=$(date -d "@$END_TIME" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MIN=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SEC=$((TOTAL_DURATION % 60))
AVG_ITER_MIN=$((TOTAL_DURATION / NUM_ITERATIONS / 60))
AVG_ITER_SEC=$(((TOTAL_DURATION / NUM_ITERATIONS) % 60))

# Save overall timing summary
TIMING_SUMMARY="output_tictac/logs/training_summary.txt"
cat > "$TIMING_SUMMARY" << EOF
============================================
TRAINING SESSION SUMMARY
============================================
Start Time:     $START_TIME_ISO
End Time:       $END_TIME_ISO
Total Duration: ${TOTAL_HOURS}h ${TOTAL_MIN}m ${TOTAL_SEC}s
Total Seconds:  $TOTAL_DURATION

Iterations:     $NUM_ITERATIONS
Avg per Iter:   ${AVG_ITER_MIN}m ${AVG_ITER_SEC}s
============================================
EOF

echo ""
echo "============================================"
echo "TRAINING PIPELINE COMPLETE!"
echo "============================================"
echo "Start Time:     $START_TIME_ISO"
echo "End Time:       $END_TIME_ISO"
echo "Total Duration: ${TOTAL_HOURS}h ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "Total Seconds:  $TOTAL_DURATION"
echo ""
echo "Iterations:     $NUM_ITERATIONS"
echo "Avg per Iter:   ${AVG_ITER_MIN}m ${AVG_ITER_SEC}s"
echo ""
echo "📊 View progress summary:"
echo "  python3 src/track_progress.py"
echo ""
echo "📁 Check training logs in: output_tictac/logs/"
echo "📁 Latest model saved in: output_tictac/models/"
echo ""
echo "🎮 To test your model, run:"
echo "  python3 src/play_human_vs_bot.py"
echo "============================================"

# Show final progress summary
echo ""
python3 track_progress.py
