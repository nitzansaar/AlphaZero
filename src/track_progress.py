"""
Track and display training progress across iterations.
Shows loss history, ELO progression, and overall training statistics.
"""
import os
import pandas as pd
import numpy as np
from glob import glob
from config import Config as cfg

# Fix path resolution - ensure paths are relative to script location
_script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(cfg.LOGDIR):
    cfg.LOGDIR = os.path.join(_script_dir, cfg.LOGDIR)
if not os.path.isabs(cfg.SAVE_MODEL_PATH):
    cfg.SAVE_MODEL_PATH = os.path.join(_script_dir, cfg.SAVE_MODEL_PATH)

def load_training_history():
    """Load all training history files."""
    history_files = glob(os.path.join(cfg.LOGDIR, "*_history.csv"))
    all_history = []
    
    for f in history_files:
        try:
            iter_num = int(os.path.basename(f).split("_")[0])
            df = pd.read_csv(f)
            df['iteration'] = iter_num
            all_history.append(df)
        except (ValueError, pd.errors.EmptyDataError):
            continue
    
    if not all_history:
        return None
    
    combined = pd.concat(all_history, ignore_index=True)
    return combined.sort_values('iteration')

def load_elo_tracking():
    """Load ELO tracking data."""
    elo_path = os.path.join(cfg.LOGDIR, "elo_tracking.csv")
    if os.path.exists(elo_path):
        return pd.read_csv(elo_path)
    return None

def load_timing_tracking():
    """Load timing tracking data."""
    timing_path = os.path.join(cfg.LOGDIR, "timing_tracking.csv")
    if os.path.exists(timing_path):
        try:
            df = pd.read_csv(timing_path)
            # Convert duration columns to numeric
            if 'duration_seconds' in df.columns:
                df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce')
            if 'selfplay_seconds' in df.columns:
                df['selfplay_seconds'] = pd.to_numeric(df['selfplay_seconds'], errors='coerce')
            if 'training_seconds' in df.columns:
                df['training_seconds'] = pd.to_numeric(df['training_seconds'], errors='coerce')
            if 'eval_seconds' in df.columns:
                df['eval_seconds'] = pd.to_numeric(df['eval_seconds'], errors='coerce')
            return df
        except Exception as e:
            print(f"Warning: Could not load timing data: {e}")
            return None
    return None

def load_training_summary():
    """Load overall training summary."""
    summary_path = os.path.join(cfg.LOGDIR, "training_summary.txt")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                return f.read()
        except Exception:
            return None
    return None

def get_latest_stats():
    """Get latest statistics from training."""
    history = load_training_history()
    elo_df = load_elo_tracking()
    timing_df = load_timing_tracking()
    
    stats = {
        'total_iterations': 0,
        'latest_loss': None,
        'latest_elo': None,
        'best_loss': None,
        'best_elo': None,
        'loss_trend': None,
        'elo_trend': None,
        'total_time': None,
        'avg_time_per_iter': None,
        'start_time': None,
        'end_time': None
    }
    
    if history is not None and len(history) > 0:
        stats['total_iterations'] = history['iteration'].max() + 1
        
        # Latest loss (final epoch of latest iteration)
        latest_iter = history['iteration'].max()
        latest_data = history[history['iteration'] == latest_iter]
        if len(latest_data) > 0:
            stats['latest_loss'] = latest_data['Tr_Loss'].iloc[-1]
            if 'Value_Loss' in latest_data.columns:
                stats['latest_value_loss'] = latest_data['Value_Loss'].iloc[-1]
                stats['latest_policy_loss'] = latest_data['Policy_Loss'].iloc[-1]
        
        # Best loss
        stats['best_loss'] = history['Tr_Loss'].min()
        
        # Loss trend (comparing first and last iterations)
        if stats['total_iterations'] > 1:
            first_iter = history['iteration'].min()
            first_loss = history[history['iteration'] == first_iter]['Tr_Loss'].iloc[-1]
            last_loss = stats['latest_loss']
            if last_loss is not None:
                stats['loss_trend'] = first_loss - last_loss
    
    if elo_df is not None and len(elo_df) > 0:
        stats['latest_elo'] = elo_df['elo'].iloc[-1]
        stats['best_elo'] = elo_df['elo'].max()
        
        if len(elo_df) > 1:
            first_elo = elo_df['elo'].iloc[0]
            last_elo = elo_df['elo'].iloc[-1]
            stats['elo_trend'] = last_elo - first_elo
    
    # Load timing statistics
    if timing_df is not None and len(timing_df) > 0:
        total_seconds = timing_df['duration_seconds'].sum()
        stats['total_time'] = total_seconds
        stats['avg_time_per_iter'] = timing_df['duration_seconds'].mean()
        
        if 'start_time' in timing_df.columns and len(timing_df) > 0:
            stats['start_time'] = timing_df['start_time'].iloc[0]
        if 'end_time' in timing_df.columns and len(timing_df) > 0:
            stats['end_time'] = timing_df['end_time'].iloc[-1]
    
    return stats

def print_progress_summary():
    """Print a formatted progress summary."""
    stats = get_latest_stats()
    
    print("\n" + "=" * 70)
    print("TRAINING PROGRESS SUMMARY")
    print("=" * 70)
    
    print(f"\nIterations Completed: {stats['total_iterations']}")
    
    # Timing information
    if stats['total_time'] is not None:
        total_sec = int(stats['total_time'])
        total_hours = total_sec // 3600
        total_min = (total_sec % 3600) // 60
        total_sec_remain = total_sec % 60
        
        avg_sec = int(stats['avg_time_per_iter'])
        avg_min = avg_sec // 60
        avg_sec_remain = avg_sec % 60
        
        print(f"\n⏱️  Timing Statistics:")
        if stats['start_time']:
            print(f"  Start Time:    {stats['start_time']}")
        if stats['end_time']:
            print(f"  End Time:      {stats['end_time']}")
        print(f"  Total Time:     {total_hours}h {total_min}m {total_sec_remain}s")
        print(f"  Avg per Iter:   {avg_min}m {avg_sec_remain}s")
    
    if stats['latest_loss'] is not None:
        print(f"\n📊 Loss Statistics:")
        print(f"  Latest Loss:  {stats['latest_loss']:.6f}")
        if 'latest_value_loss' in stats:
            print(f"  Value Loss:   {stats['latest_value_loss']:.6f}")
            print(f"  Policy Loss:  {stats['latest_policy_loss']:.6f}")
        print(f"  Best Loss:    {stats['best_loss']:.6f}")
        if stats['loss_trend'] is not None:
            trend_str = "📈" if stats['loss_trend'] > 0 else "📉"
            print(f"  Loss Change:  {stats['loss_trend']:+.6f} {trend_str}")
    
    if stats['latest_elo'] is not None:
        print(f"\n🏆 ELO Statistics:")
        print(f"  Current ELO:  {stats['latest_elo']:.1f}")
        print(f"  Best ELO:     {stats['best_elo']:.1f}")
        if stats['elo_trend'] is not None:
            trend_str = "📈" if stats['elo_trend'] > 0 else "📉"
            print(f"  ELO Change:   {stats['elo_trend']:+.1f} {trend_str}")
    
    print("\n" + "=" * 70)
    
    # Show iteration-by-iteration progress
    history = load_training_history()
    elo_df = load_elo_tracking()
    
    if history is not None and len(history) > 0:
        print("\n📈 Loss by Iteration:")
        iter_summary = history.groupby('iteration')['Tr_Loss'].agg(['min', 'max', 'last'])
        iter_summary.columns = ['Min', 'Max', 'Final']
        print(iter_summary.to_string())
    
    if elo_df is not None and len(elo_df) > 0:
        print("\n🏆 ELO by Iteration:")
        print(elo_df[['iteration', 'elo', 'elo_change', 'win_rate']].to_string(index=False))
    
    # Show timing breakdown by iteration
    timing_df = load_timing_tracking()
    if timing_df is not None and len(timing_df) > 0:
        print("\n⏱️  Timing by Iteration:")
        timing_display = timing_df.copy()
        # Convert seconds to minutes:seconds format
        if 'duration_seconds' in timing_display.columns:
            timing_display['Duration'] = timing_display['duration_seconds'].apply(
                lambda x: f"{int(x//60)}m {int(x%60)}s" if pd.notna(x) else "N/A"
            )
        if 'selfplay_seconds' in timing_display.columns:
            timing_display['Self-play'] = timing_display['selfplay_seconds'].apply(
                lambda x: f"{int(x//60)}m {int(x%60)}s" if pd.notna(x) else "N/A"
            )
        if 'training_seconds' in timing_display.columns:
            timing_display['Training'] = timing_display['training_seconds'].apply(
                lambda x: f"{int(x//60)}m {int(x%60)}s" if pd.notna(x) else "N/A"
            )
        if 'eval_seconds' in timing_display.columns:
            timing_display['Eval'] = timing_display['eval_seconds'].apply(
                lambda x: f"{int(x//60)}m {int(x%60)}s" if pd.notna(x) else "N/A"
            )
        
        display_cols = ['iteration']
        if 'Duration' in timing_display.columns:
            display_cols.append('Duration')
        if 'Self-play' in timing_display.columns:
            display_cols.append('Self-play')
        if 'Training' in timing_display.columns:
            display_cols.append('Training')
        if 'Eval' in timing_display.columns:
            display_cols.append('Eval')
        
        print(timing_display[display_cols].to_string(index=False))
    
    print("\n" + "=" * 70)

def main():
    """Main function."""
    print_progress_summary()

if __name__ == "__main__":
    main()

