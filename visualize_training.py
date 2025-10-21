#!/usr/bin/env python3
"""
Standalone script to visualize training history from a JSON file.

Usage:
    python visualize_training.py [history_file]
"""

import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_progress(history: dict, save_path: str = "training_progress.png"):
    """
    Plot training progress including win rates and losses.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AlphaZero Training Progress', fontsize=16, fontweight='bold')

    episodes = history['episodes']

    # Plot 1: Win rates over time
    ax1 = axes[0, 0]
    ax1.plot(episodes, history['win_rate_p1'], label='Player 1 (X)', color='blue', linewidth=2)
    ax1.plot(episodes, history['win_rate_p2'], label='Player 2 (O)', color='red', linewidth=2)
    ax1.plot(episodes, history['draw_rate'], label='Draws', color='green', linewidth=2)
    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_title('Game Outcomes Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    # Plot 2: Total loss over time
    ax2 = axes[0, 1]
    ax2.plot(episodes, history['total_loss'], color='purple', linewidth=2)
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Total Loss', fontsize=12)
    ax2.set_title('Total Loss (Policy + Value)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Policy loss over time
    ax3 = axes[1, 0]
    ax3.plot(episodes, history['policy_loss'], color='orange', linewidth=2)
    ax3.set_xlabel('Episodes', fontsize=12)
    ax3.set_ylabel('Policy Loss', fontsize=12)
    ax3.set_title('Policy Loss (Cross-Entropy)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Value loss over time
    ax4 = axes[1, 1]
    ax4.plot(episodes, history['value_loss'], color='brown', linewidth=2)
    ax4.set_xlabel('Episodes', fontsize=12)
    ax4.set_ylabel('Value Loss', fontsize=12)
    ax4.set_title('Value Loss (MSE)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Training progress plot saved: {save_path}")
    plt.show()


def main():
    """Main entry point."""
    # Get history file path
    if len(sys.argv) > 1:
        history_file = sys.argv[1]
    else:
        history_file = "training_history.json"

    # Check if file exists
    if not Path(history_file).exists():
        print(f"❌ Error: Training history file not found: {history_file}")
        print(f"\nUsage: python visualize_training.py [history_file]")
        print(f"Default: python visualize_training.py  (looks for training_history.json)")
        sys.exit(1)

    # Load training history
    print(f"Loading training history from: {history_file}")
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total episodes: {history['episodes'][-1]}")
    print(f"Final win rates:")
    print(f"  Player 1: {history['win_rate_p1'][-1]:.1f}%")
    print(f"  Player 2: {history['win_rate_p2'][-1]:.1f}%")
    print(f"  Draws: {history['draw_rate'][-1]:.1f}%")
    print(f"\nFinal losses:")
    print(f"  Policy loss: {history['policy_loss'][-1]:.4f}")
    print(f"  Value loss: {history['value_loss'][-1]:.4f}")
    print(f"  Total loss: {history['total_loss'][-1]:.4f}")
    print(f"{'='*60}\n")

    # Plot
    plot_training_progress(history)


if __name__ == "__main__":
    main()
