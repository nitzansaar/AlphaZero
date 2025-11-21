#!/bin/bash

# RTX 5090 Optimization: Environment Variables
# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
export NVIDIA_TF32_OVERRIDE=1

# Optimize CUDA memory allocation for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable cuDNN auto-tuning
export CUDNN_BENCHMARK=1

# Use the new CUDA memory allocator (if PyTorch >= 1.10)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting self-play game generation..."
python3 selfplay.py

echo ""
echo "Self-play complete. Starting training..."
python3 train.py

echo ""
echo "============================================"
echo "Training pipeline complete!"
echo "============================================"
