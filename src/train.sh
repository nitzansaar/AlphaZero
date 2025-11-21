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

echo "Starting self-play game generation..."
python3 selfplay.py

echo ""
echo "Self-play complete. Starting training..."
python3 train.py

echo ""
echo "============================================"
echo "Training pipeline complete!"
echo "============================================"
