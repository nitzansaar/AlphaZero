"""
Create visualizations from training history data.
Generates a graph showing the final training loss for each iteration.
"""
import os
import sys

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    sys.exit(1)

from config import Config as cfg

# Fix path resolution
_script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(cfg.LOGDIR):
    cfg.LOGDIR = os.path.join(_script_dir, cfg.LOGDIR)

output_dir = os.path.join(_script_dir, cfg.TEST_OUTPUT_PATH)
os.makedirs(output_dir, exist_ok=True)

def load_training_history():
    """Load and aggregate the final loss of each training iteration."""
    import glob

    pattern = os.path.join(cfg.LOGDIR, "*_history.csv")
    history_files = glob.glob(pattern)

    if not history_files:
        print(f"No history files found in {cfg.LOGDIR}")
        return None

    all_iteration_finals = []
    for filepath in history_files:
        filename = os.path.basename(filepath)
        try:
            iteration = int(filename.split('_')[0])
        except ValueError:
            continue

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                continue
            
            # Sort by epoch to ensure we get the actual final state
            df = df.sort_values('Epoch')
            # Extract only the last row (final epoch of this iteration)
            final_state = df.iloc[[-1]].copy()
            final_state['iteration'] = iteration
            all_iteration_finals.append(final_state)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    if not all_iteration_finals:
        return None

    # Concatenate and sort by iteration number
    combined = pd.concat(all_iteration_finals, ignore_index=True)
    combined = combined.sort_values('iteration')
    return combined


def create_iteration_loss_graph():
    """Create a graph showing only the final loss per iteration."""
    history = load_training_history()
    
    if history is None or len(history) == 0:
        print("No training history found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotting lines with markers to emphasize specific iteration points
    ax.plot(history['iteration'], history['Tr_Loss'], 
           label='Final Total Loss', marker='o', linewidth=2, color='#2E86AB')
    
    ax.plot(history['iteration'], history['Value_Loss'], 
           label='Final Value Loss', marker='s', linestyle='--', linewidth=1.5, color='#06A77D', alpha=0.7)
    
    ax.plot(history['iteration'], history['Policy_Loss'], 
           label='Final Policy Loss', marker='^', linestyle='--', linewidth=1.5, color='#F18F01', alpha=0.7)
    
    # Labeling
    ax.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss at Iteration End', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Trend (Final Loss per Iteration)', fontsize=14, fontweight='bold')
    
    # Ensure x-axis only shows integer iteration numbers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Save figure
    output_path = os.path.join(output_dir, 'final_loss_per_iteration.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Iteration loss graph saved to: {output_path}")
    plt.close()

def main():
    print("\n" + "=" * 50)
    print("GENERATING ITERATION-LEVEL LOSS VISUALIZATION")
    print("=" * 50)
    
    try:
        create_iteration_loss_graph()
        print("\n✓ Success!")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()