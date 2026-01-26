#!/usr/bin/env python3
"""
Reset training for fresh start with new model architecture.

This script:
1. Backs up existing models and dataset
2. Clears the dataset for fresh data collection
3. Resets iteration counter

Run this before starting training with the new architecture.
"""
import os
import shutil
from datetime import datetime
from config import Config as cfg

def reset_training():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"

    print("=" * 60)
    print("TRAINING RESET SCRIPT")
    print("=" * 60)

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    print(f"\nBackup directory: {backup_dir}")

    # Backup models
    if os.path.exists(cfg.SAVE_MODEL_PATH):
        model_backup = os.path.join(backup_dir, "models")
        shutil.copytree(cfg.SAVE_MODEL_PATH, model_backup)
        print(f"✓ Backed up models to {model_backup}")

        # Clear models directory
        for f in os.listdir(cfg.SAVE_MODEL_PATH):
            os.remove(os.path.join(cfg.SAVE_MODEL_PATH, f))
        print(f"✓ Cleared {cfg.SAVE_MODEL_PATH}/")

    # Backup dataset
    dataset_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)
    if os.path.exists(dataset_path):
        dataset_backup = os.path.join(backup_dir, cfg.DATASET_PATH)
        shutil.copy2(dataset_path, dataset_backup)
        print(f"✓ Backed up dataset to {dataset_backup}")

        # Remove old dataset
        os.remove(dataset_path)
        print(f"✓ Removed old dataset")

    # Backup logs
    if os.path.exists(cfg.LOGDIR):
        logs_backup = os.path.join(backup_dir, "logs")
        shutil.copytree(cfg.LOGDIR, logs_backup)
        print(f"✓ Backed up logs to {logs_backup}")

        # Clear iteration counter
        iter_file = os.path.join(cfg.LOGDIR, "current_iteration.txt")
        if os.path.exists(iter_file):
            os.remove(iter_file)
            print(f"✓ Reset iteration counter")

    print("\n" + "=" * 60)
    print("RESET COMPLETE")
    print("=" * 60)
    print(f"""
Next steps:
1. Run selfplay.py to generate fresh training data
2. Run train.py to train the new model
3. Repeat the selfplay -> train cycle

The new configuration:
- Model: 6 residual blocks, 128 channels (~1.86M params)
- MCTS simulations: {cfg.NUM_SIMULATIONS}
- Replay buffer size: {cfg.DATASET_QUEUE_SIZE}
- Learning rate: {cfg.LEARNING_RATE}
- Temperature threshold: {cfg.TEMP_THRESHOLD} moves
""")

if __name__ == "__main__":
    response = input("This will backup and clear existing training data. Continue? [y/N]: ")
    if response.lower() == 'y':
        reset_training()
    else:
        print("Aborted.")
