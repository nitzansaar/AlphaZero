# Training on Google Cloud GPU

## Why Use GPU?

Training 100k episodes on different hardware:
- **CPU (M1/M2 Mac)**: 3-6 hours ⏱️
- **T4 GPU (Google Cloud)**: 20-40 minutes ⚡ **10x faster!**

The GPU acceleration comes from:
- Parallel neural network forward passes
- Batch operations in MCTS
- GPU-optimized matrix operations

## Setup Steps

### 1. Create a Google Cloud VM with GPU

```bash
# Option A: Using gcloud CLI
gcloud compute instances create tictactoe-trainer \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --metadata="install-nvidia-driver=True"

# Option B: Use Google Cloud Console
# 1. Go to Compute Engine → VM instances
# 2. Click "Create Instance"
# 3. Select:
#    - Machine type: n1-standard-4 (4 vCPUs, 15 GB memory)
#    - GPUs: 1 x NVIDIA T4
#    - Boot disk: Deep Learning VM (PyTorch)
#    - Disk size: 50 GB
# 4. Click "Create"
```

### 2. SSH into Your VM

```bash
# Find your VM's external IP in the Cloud Console
gcloud compute ssh tictactoe-trainer --zone=us-central1-a

# Or use the SSH button in Cloud Console
```

### 3. Transfer Your Code

**Option A: Using SCP**
```bash
# From your local machine
cd /Users/nitzansaar/Desktop/EE542
tar -czf tictactoeMCTS.tar.gz tictactoeMCTS
gcloud compute scp tictactoeMCTS.tar.gz tictactoe-trainer:~ --zone=us-central1-a

# On the VM
tar -xzf tictactoeMCTS.tar.gz
cd tictactoeMCTS
```

**Option B: Using Git**
```bash
# On the VM
git clone YOUR_REPO_URL
cd tictactoeMCTS
```

### 4. Run Setup Script

```bash
chmod +x cloud_training_setup.sh
./cloud_training_setup.sh
```

This will:
- Check GPU availability
- Install PyTorch with CUDA
- Install dependencies
- Verify GPU is working

### 5. Start Training

**Option 1: Direct execution (simplest)**
```bash
python3 -m src.trainer.train_self_play
```

**Option 2: Background process (recommended)**
```bash
nohup python3 -m src.trainer.train_self_play > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Check if still running
ps aux | grep train_self_play
```

**Option 3: Using tmux (best for long sessions)**
```bash
# Start tmux session
tmux new -s training

# Run training
python3 -m src.trainer.train_self_play

# Detach: Press Ctrl+B, then D
# You can now disconnect from SSH

# Later, reconnect and attach
tmux attach -t training
```

### 6. Monitor GPU Usage

```bash
# In another SSH session or tmux pane
watch -n 1 nvidia-smi

# Or just once
nvidia-smi
```

You should see:
- **GPU Utilization**: 60-95%
- **Memory Usage**: ~500-1000 MB
- **Power Usage**: ~40-70W (T4 max is 70W)

### 7. Download Results

After training completes:

```bash
# From your local machine
gcloud compute scp tictactoe-trainer:~/tictactoeMCTS/tictactoe_selfplay_final.pth . --zone=us-central1-a
gcloud compute scp tictactoe-trainer:~/tictactoeMCTS/training_progress.png . --zone=us-central1-a
gcloud compute scp tictactoe-trainer:~/tictactoeMCTS/training_history.json . --zone=us-central1-a

# Copy all checkpoint files
gcloud compute scp 'tictactoe-trainer:~/tictactoeMCTS/*.pth' . --zone=us-central1-a
```

### 8. Stop/Delete the VM

**Important: Stop the VM when done to avoid charges!**

```bash
# Stop (can restart later)
gcloud compute instances stop tictactoe-trainer --zone=us-central1-a

# Delete (permanent)
gcloud compute instances delete tictactoe-trainer --zone=us-central1-a
```

## Troubleshooting

### GPU Not Detected

```bash
# Install NVIDIA drivers
sudo /opt/deeplearning/install-driver.sh

# Reboot
sudo reboot
```

### Out of Memory

Reduce batch size in code or increase VM memory:
```python
batch_size = 64  # Instead of 128
```

### Training Too Slow

Check GPU usage:
```bash
nvidia-smi
```

If GPU util is low (<30%), the bottleneck might be CPU (MCTS simulations). The neural network is small, so GPU benefits are moderate for tic-tac-toe.

### Can't Install PyTorch with CUDA

```bash
# Try different CUDA version
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Or use conda
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Cost Estimate

Google Cloud GPU pricing (as of 2024):
- **T4 GPU**: ~$0.35/hour
- **VM (n1-standard-4)**: ~$0.19/hour
- **Total**: ~$0.54/hour

For 100k episodes (~30 min on GPU):
- **Cost**: ~$0.27 per training run

**Tips to save money:**
1. Use preemptible instances (70% cheaper, but can be interrupted)
2. Stop VM immediately after training
3. Use cloud storage for long-term model storage

## Alternative: Use Colab (Free GPU)

If you don't want to use Google Cloud:

1. Upload code to Google Drive
2. Open Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Upload and run training:

```python
from google.colab import drive
drive.mount('/content/drive')

!cd /content/drive/MyDrive/tictactoeMCTS && python3 -m src.trainer.train_self_play
```

**Note**: Colab sessions timeout after ~12 hours and may disconnect randomly.

## Expected Results

After 100k episodes with T4 GPU training:
- **Draw rate**: >90% (optimal tic-tac-toe)
- **Policy loss**: <0.5
- **Value loss**: <0.3
- **Model makes perfect or near-perfect tactical moves**

The trained model will:
- ✅ Block all winning threats
- ✅ Find all winning moves
- ✅ Play optimal openings
- ✅ Force draws against perfect play
