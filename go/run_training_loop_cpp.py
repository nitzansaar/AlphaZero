"""
run_training_loop_cpp.py — Training loop using the C++ selfplay binary.

Identical to run_training_loop.py except it calls selfplay_cpp_runner.py
instead of selfplay.py.  Switch back to the Python version at any time:

    python run_training_loop.py      # Python MCTS selfplay
    python run_training_loop_cpp.py  # C++ selfplay binary
"""

import os
import json
import subprocess
import sys
import time
from datetime import datetime


def _make_pie(ax, timings, title, colormap):
    """Draw a single pie chart with a clean side legend instead of overlapping labels."""
    import matplotlib.pyplot as plt

    sorted_items = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in sorted_items]
    sizes = [v for _, v in sorted_items]
    total = sum(sizes)

    colors = [colormap(i) for i in range(len(labels))]

    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
    )

    legend_labels = []
    for name, secs in zip(labels, sizes):
        pct = (secs / total * 100) if total > 0 else 0
        if secs >= 60:
            time_str = f"{secs/60:.1f}m"
        else:
            time_str = f"{secs:.1f}s"
        legend_labels.append(f"{name}  —  {time_str} ({pct:.1f}%)")

    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.05, 0.5),
              fontsize=10, frameon=False)

    if total >= 60:
        total_str = f"{total/60:.1f} min"
    else:
        total_str = f"{total:.1f}s"
    ax.set_title(f"{title}\n(Total: {total_str})", fontsize=13, fontweight='bold')


def generate_timing_pie_charts(logdir, output_dir):
    """Read timing JSONs from selfplay and train, generate pie charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    selfplay_path = os.path.join(logdir, "selfplay_timing.json")
    train_path    = os.path.join(logdir, "train_timing.json")

    has_selfplay = os.path.exists(selfplay_path)
    has_train    = os.path.exists(train_path)

    if not has_selfplay and not has_train:
        print("No timing data found, skipping pie chart generation.")
        return

    selfplay_timings = {}
    train_timings    = {}

    if has_selfplay:
        with open(selfplay_path) as f:
            selfplay_timings = json.load(f)["timings"]
    if has_train:
        with open(train_path) as f:
            train_timings = json.load(f)["timings"]

    os.makedirs(output_dir, exist_ok=True)

    num_charts = sum([bool(selfplay_timings), bool(train_timings)])
    fig, axes = plt.subplots(1, num_charts, figsize=(10 * num_charts, 6))
    if num_charts == 1:
        axes = [axes]

    chart_idx = 0
    if selfplay_timings:
        _make_pie(axes[chart_idx], selfplay_timings, "Self-Play Time Breakdown (C++)", plt.cm.Set3)
        chart_idx += 1
    if train_timings:
        _make_pie(axes[chart_idx], train_timings, "Training Time Breakdown", plt.cm.Pastel1)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "timing_pie_charts.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Timing pie charts saved to {save_path}")


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


# ── Model Gating ──────────────────────────────────────────────────────────


def _read_best_iter(model_dir):
    """Read current_best_iter.txt; return -1 if absent or unreadable."""
    path = os.path.join(model_dir, "current_best_iter.txt")
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return -1


def _write_best_iter(model_dir, iteration):
    """Write iteration to current_best_iter.txt in model_dir."""
    path = os.path.join(model_dir, "current_best_iter.txt")
    with open(path, "w") as f:
        f.write(str(iteration))


def _read_new_iter(logdir):
    """Read the iteration number that train.py just saved (current_iteration.txt)."""
    path = os.path.join(logdir, "current_iteration.txt")
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return -1


def gate_model(new_iter, prev_best_iter):
    """
    Evaluate new model (new_iter) against previous best (prev_best_iter).

    Plays cfg.GATE_GAMES games sequentially with cfg.GATE_SIMULATIONS MCTS sims
    per move.  Returns True if the new model wins >= cfg.GATE_WIN_RATE fraction.

    Auto-passes when there is no previous model to compare against.
    """
    import torch
    from config import Config as cfg
    from game import Go
    from value_policy_function import ValuePolicyNetwork
    from test_model_vs_model import run_matchup

    if prev_best_iter < 0:
        print("[Gate] No previous model; auto-passing gate.")
        return True

    model_dir = cfg.SAVE_MODEL_PATH
    new_path = os.path.join(model_dir, cfg.BEST_MODEL.format(new_iter))
    old_path = os.path.join(model_dir, cfg.BEST_MODEL.format(prev_best_iter))

    if not os.path.exists(old_path):
        print(f"[Gate] Previous model not found ({old_path}); auto-passing.")
        return True
    if not os.path.exists(new_path):
        print(f"[Gate] New model not found ({new_path}); failing gate.")
        return False

    print(f"\n[Gate] Evaluating iter_{new_iter} vs iter_{prev_best_iter} "
          f"({cfg.GATE_GAMES} games, {cfg.GATE_SIMULATIONS} sims/move)")

    vpn_new = ValuePolicyNetwork(new_path, use_compile=False)
    vpn_old = ValuePolicyNetwork(old_path, use_compile=False)
    game = Go()

    result = run_matchup(
        game, vpn_new, vpn_old,
        num_games=cfg.GATE_GAMES,
        num_simulations1=cfg.GATE_SIMULATIONS,
        num_simulations2=cfg.GATE_SIMULATIONS,
        label1=f"iter_{new_iter}",
        label2=f"iter_{prev_best_iter}",
        temperature_moves=cfg.GATE_TEMPERATURE_MOVES,
    )

    wins_new = result["wins1"]
    win_rate = wins_new / cfg.GATE_GAMES
    passes = win_rate >= cfg.GATE_WIN_RATE
    status = "PASS" if passes else "FAIL"
    print(f"[Gate] {status}: iter_{new_iter} win rate = {win_rate:.0%} "
          f"(threshold {cfg.GATE_WIN_RATE:.0%}; {wins_new}/{cfg.GATE_GAMES} wins)")

    del vpn_new, vpn_old, game
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return passes


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run training loop (C++ selfplay)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of training iterations')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='Run evaluation every N iterations (0 = never)')
    parser.add_argument('--no-gate', action='store_true',
                        help='Disable model gating (always use latest model for selfplay)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("ALPHAZERO TRAINING LOOP  (C++ selfplay)")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Evaluate every: {args.eval_every if args.eval_every > 0 else 'Never'}")
    print(f"Model gating:   {'disabled (--no-gate)' if args.no_gate else 'enabled'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Bootstrap: create the initial random-weight model if no checkpoint exists.
    # selfplay_cpp_runner.py needs a model on disk before it can generate data.
    from glob import glob
    from config import Config as cfg
    model_files = [f for f in glob(os.path.join(cfg.SAVE_MODEL_PATH, "*.pt"))
                   if not f.endswith("_ts.pt")]
    if not model_files:
        print("\n[Bootstrap] No model found — creating initial random-weight model...")
        import torch
        from model import NeuralNetwork
        os.makedirs(cfg.SAVE_MODEL_PATH, exist_ok=True)
        init_model = NeuralNetwork()
        init_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(0))
        torch.save(init_model.state_dict(), init_path)
        print(f"[Bootstrap] Saved: {init_path}\n")

    for i in range(args.iterations):
        iter_start = time.time()
        print(f"\n{'#'*60}")
        print(f"# ITERATION {i + 1} / {args.iterations}")
        print(f"{'#'*60}")

        # Self-play (C++ binary)
        if not run_command(f"{sys.executable} selfplay_cpp_runner.py",
                           "Self-play: Generating training data (C++)"):
            print("ERROR: C++ self-play failed!")
            return 1

        # Training (unchanged)
        if not run_command(f"{sys.executable} train.py",
                           "Training: Updating neural network"):
            print("ERROR: Training failed!")
            return 1

        # Model gating: only promote new model to selfplay if it beats the current best
        if not args.no_gate:
            from config import Config as cfg
            new_iter = _read_new_iter(cfg.LOGDIR)
            prev_best = _read_best_iter(cfg.SAVE_MODEL_PATH)
            if new_iter < 0:
                print("[Gate] Could not determine new iteration number; skipping gate.")
            elif new_iter == prev_best:
                # First iteration or same number (shouldn't happen normally)
                _write_best_iter(cfg.SAVE_MODEL_PATH, new_iter)
            elif gate_model(new_iter, prev_best):
                _write_best_iter(cfg.SAVE_MODEL_PATH, new_iter)
                print(f"[Gate] iter_{new_iter} is the new selfplay model.")
            else:
                print(f"[Gate] Keeping iter_{prev_best} for selfplay "
                      f"(iter_{new_iter} did not pass).")

        iter_time = time.time() - iter_start
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"Iteration {i + 1} complete in {iter_time/60:.1f} minutes")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {args.iterations}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average per iteration: {total_time/60/args.iterations:.1f} minutes")

    from config import Config as cfg
    generate_timing_pie_charts(cfg.LOGDIR, cfg.TEST_OUTPUT_PATH)

    print(f"\nTo evaluate all models: python evaluate_training_progress.py")


if __name__ == "__main__":
    sys.exit(main() or 0)
