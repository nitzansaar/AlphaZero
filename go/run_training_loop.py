"""
Run multiple iterations of selfplay + training.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run training loop')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of training iterations')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='Run evaluation every N iterations (0 = never)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("ALPHAZERO TRAINING LOOP")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Evaluate every: {args.eval_every if args.eval_every > 0 else 'Never'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    for i in range(args.iterations):
        iter_start = time.time()
        print(f"\n{'#'*60}")
        print(f"# ITERATION {i + 1} / {args.iterations}")
        print(f"{'#'*60}")

        # Self-play
        if not run_command("python selfplay.py", "Self-play: Generating training data"):
            print("ERROR: Self-play failed!")
            return 1

        # Training
        if not run_command("python train.py", "Training: Updating neural network"):
            print("ERROR: Training failed!")
            return 1

        iter_time = time.time() - iter_start
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iteration {i + 1} complete in {iter_time/60:.1f} minutes")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {args.iterations}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average per iteration: {total_time/60/args.iterations:.1f} minutes")
    print(f"\nTo evaluate all models: python evaluate_training_progress.py")


if __name__ == "__main__":
    sys.exit(main() or 0)
