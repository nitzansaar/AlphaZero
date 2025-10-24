#!/bin/bash
# Quick setup script for training on Google Cloud GPU VM

set -e

echo "=========================================="
echo "AlphaZero Training - Cloud GPU Setup"
echo "=========================================="

# 1. Check for NVIDIA GPU
echo -e "\n[1/5] Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ GPU detected"
else
    echo "⚠ WARNING: nvidia-smi not found. GPU may not be available."
    echo "   Training will run on CPU (much slower)."
fi

# 2. Install PyTorch with CUDA support
echo -e "\n[2/5] Installing PyTorch with CUDA support..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

# 3. Install other dependencies
echo -e "\n[3/5] Installing dependencies..."
pip3 install matplotlib numpy --quiet

# 4. Verify PyTorch can see GPU
echo -e "\n[4/5] Verifying PyTorch GPU setup..."
python3 << EOF
import torch
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   WARNING: PyTorch cannot access GPU. Training will be slow.")
EOF

# 5. Ready to train
echo -e "\n[5/5] Setup complete!"
echo ""
echo "=========================================="
echo "Ready to train!"
echo "=========================================="
echo ""
echo "To start training (100k episodes):"
echo "  python3 -m src.trainer.train_self_play"
echo ""
echo "To run in background (recommended for long training):"
echo "  nohup python3 -m src.trainer.train_self_play > training.log 2>&1 &"
echo "  tail -f training.log  # Monitor progress"
echo ""
echo "Or use tmux for easy reconnection:"
echo "  tmux new -s training"
echo "  python3 -m src.trainer.train_self_play"
echo "  # Press Ctrl+B then D to detach"
echo "  # Later: tmux attach -t training"
echo ""
echo "Training will take approximately:"
echo "  - CPU: 3-6 hours"
echo "  - T4 GPU: 20-40 minutes"
echo "=========================================="
