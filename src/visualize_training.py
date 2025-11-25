"""
Create visualizations from training history data.
Generates a single combined loss graph showing all training iterations.
"""
import os
import sys

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("\nPlease install matplotlib, pandas, and numpy:")
    print("  pip install matplotlib pandas numpy")
    print("\nOr if using system packages:")
    print("  sudo apt-get install python3-matplotlib python3-pandas python3-numpy")
    sys.exit(1)

from track_progress import load_training_history
from config import Config as cfg

# Fix path resolution - ensure LOGDIR is relative to script location
_script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(cfg.LOGDIR):
    # If LOGDIR is relative, make it relative to script directory
    cfg.LOGDIR = os.path.join(_script_dir, cfg.LOGDIR)

def create_combined_loss_graph():
    """Create a single graph showing combined loss across all training iterations."""
    history = load_training_history()
    
    if history is None or len(history) == 0:
        print("No training history found")
        return
    
    # Sort by iteration and epoch
    history = history.sort_values(['iteration', 'Epoch'])
    
    # Create a continuous epoch index across all iterations
    # Each iteration starts where the previous one ended
    epochs_per_iteration = history.groupby('iteration')['Epoch'].max() + 1
    cumulative_epochs = []
    current_epoch = 0
    
    for iter_num in sorted(history['iteration'].unique()):
        iter_data = history[history['iteration'] == iter_num]
        num_epochs = len(iter_data)
        cumulative_epochs.extend(range(current_epoch, current_epoch + num_epochs))
        current_epoch += num_epochs
    
    history['Cumulative_Epoch'] = cumulative_epochs
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot total loss
    ax.plot(history['Cumulative_Epoch'], history['Tr_Loss'], 
           label='Total Loss', linewidth=2, color='#2E86AB', alpha=0.9)
    
    # Plot value and policy loss components
    ax.plot(history['Cumulative_Epoch'], history['Value_Loss'], 
           label='Value Loss', linewidth=1.5, color='#06A77D', alpha=0.7)
    ax.plot(history['Cumulative_Epoch'], history['Policy_Loss'], 
           label='Policy Loss', linewidth=1.5, color='#F18F01', alpha=0.7)
    
    # Add vertical lines to separate iterations
    iterations = sorted(history['iteration'].unique())
    current_pos = 0
    for i, iter_num in enumerate(iterations):
        iter_data = history[history['iteration'] == iter_num]
        num_epochs = len(iter_data)
        if i > 0:  # Don't draw line at the start
            ax.axvline(x=current_pos, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        # Add iteration label in the middle of each iteration
        mid_point = current_pos + num_epochs / 2
        ax.text(mid_point, ax.get_ylim()[1] * 0.95, f'Iter {iter_num}', 
               ha='center', va='top', fontsize=9, alpha=0.7,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
        current_pos += num_epochs
    
    ax.set_xlabel('Cumulative Epoch (across all iterations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Progression - All Iterations Combined', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(cfg.LOGDIR, 'training_loss_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Combined loss graph saved to: {output_path}")
    plt.close()

def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("GENERATING COMBINED TRAINING LOSS GRAPH")
    print("=" * 70)
    
    try:
        create_combined_loss_graph()
        print("\n✓ Visualization generated successfully!")
        print(f"\nView visualization in: {cfg.LOGDIR}")
    except Exception as e:
        print(f"\n✗ Error generating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
