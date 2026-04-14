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


# ── Iteration tracking ────────────────────────────────────────────────────


def _read_new_iter(logdir):
    """Read the iteration number that train.py just saved (current_iteration.txt)."""
    path = os.path.join(logdir, "current_iteration.txt")
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return -1


# ── Periodic evaluation (monitoring only, does not affect selfplay) ────────


def run_periodic_eval(new_iter, compare_iter):
    """
    Play cfg.GATE_GAMES games between iter_new and iter_compare and print the
    win rate.  Result is for progress monitoring only — it does not affect
    which model is used for selfplay (selfplay always uses the latest).

    Exports both models to TorchScript if the .ts files are not already cached.
    """
    from config import Config as cfg

    new_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(new_iter))
    cmp_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(compare_iter))

    if not os.path.exists(cmp_path):
        print(f"[Eval] Comparison model iter_{compare_iter} not found — skipping.")
        return
    if not os.path.exists(new_path):
        print(f"[Eval] New model iter_{new_iter} not found — skipping.")
        return

    new_ts = os.path.join(cfg.SAVE_MODEL_PATH, f"{new_iter}_ts.pt")
    cmp_ts = os.path.join(cfg.SAVE_MODEL_PATH, f"{compare_iter}_ts.pt")
    env    = f"BOARD_SIZE={cfg.BOARD_SIZE}"

    if not os.path.exists(new_ts):
        run_command(f"{env} {sys.executable} export_model.py {new_path} {new_ts}",
                    f"Export iter_{new_iter} to TorchScript")
    if not os.path.exists(cmp_ts):
        run_command(f"{env} {sys.executable} export_model.py {cmp_path} {cmp_ts}",
                    f"Export iter_{compare_iter} to TorchScript")

    cmd = (f"{sys.executable} gatekeeper_runner.py "
           f"{new_ts} {cmp_ts} {new_iter} {compare_iter} "
           f"--games {cfg.GATE_GAMES} --sims {cfg.GATE_SIMULATIONS}")
    run_command(cmd,
                f"[Eval] iter_{new_iter} vs iter_{compare_iter} "
                f"({cfg.GATE_GAMES} games — monitoring only)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run training loop (C++ selfplay)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of training iterations (ignored when --forever is set)')
    parser.add_argument('--forever', action='store_true',
                        help='Run indefinitely until killed (Ctrl-C / SIGTERM)')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='Evaluate against model N iters ago every N iterations (0 = never)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("ALPHAZERO TRAINING LOOP  (C++ selfplay)")
    print("=" * 60)
    print(f"Mode: {'continuous (--forever)' if args.forever else f'{args.iterations} iterations'}")
    print(f"Evaluate every: {args.eval_every if args.eval_every > 0 else 'Never'} iterations")
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

    i = 0
    try:
        while args.forever or i < args.iterations:
            iter_start = time.time()
            print(f"\n{'#'*60}")
            if args.forever:
                print(f"# ITERATION {i + 1}")
            else:
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

            # Periodic evaluation for progress monitoring (does not affect selfplay).
            if args.eval_every > 0 and (i + 1) % args.eval_every == 0:
                from config import Config as cfg
                new_iter = _read_new_iter(cfg.LOGDIR)
                if new_iter >= 0:
                    run_periodic_eval(new_iter, new_iter - args.eval_every)

            iter_time = time.time() - iter_start
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Iteration {i + 1} complete in {iter_time/60:.1f} minutes")
            i += 1

    except KeyboardInterrupt:
        print(f"\n\nInterrupted after {i} iteration(s).")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING STOPPED")
    print(f"{'='*60}")
    print(f"Total iterations: {i}")
    print(f"Total time: {total_time/3600:.2f} hours")
    if i > 0:
        print(f"Average per iteration: {total_time/60/i:.1f} minutes")

    from config import Config as cfg
    generate_timing_pie_charts(cfg.LOGDIR, cfg.TEST_OUTPUT_PATH)


if __name__ == "__main__":
    sys.exit(main() or 0)
