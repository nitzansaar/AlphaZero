"""
gatekeeper_runner.py — Python process manager for C++ gate evaluation.

Mirrors selfplay_cpp_runner.py: launches os.cpu_count() worker processes,
each running ./gatekeeper over a slice of games with its own LibTorch context.
Shows one tqdm progress bar per worker.  Exits 0 (ACCEPTED) or 1 (REJECTED).

Usage (called by train.sh):
    python3 gatekeeper_runner.py <new_ts.pt> <best_ts.pt> <new_iter> <best_iter>

Options:
    --games N       total gate games        (default: 100)
    --sims  N       MCTS sims per move      (default: 400)
    --win-rate F    required win fraction   (default: 0.55)
    --workers N     parallel processes      (default: cpu_count)
"""

import os, sys, subprocess, argparse, threading, tempfile
import torch
from tqdm import tqdm
from config import Config as cfg

_HERE       = os.path.dirname(os.path.abspath(__file__))
GATE_BINARY = os.path.join(_HERE, "gatekeeper")
USE_CUDA    = torch.cuda.is_available()


def _monitor(worker_dirs, games_per_worker, stop_event, bars):
    """Background thread: poll each worker's progress file and update tqdm."""
    last = [0] * len(worker_dirs)

    def flush():
        for i, (wdir, bar) in enumerate(zip(worker_dirs, bars)):
            try:
                count = int(open(os.path.join(wdir, "progress")).read().strip())
            except (FileNotFoundError, ValueError):
                count = last[i]
            if count > last[i]:
                bar.update(count - last[i])
                last[i] = count

    while not stop_event.wait(0.5):
        flush()
    flush()  # final update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_ts",    help="new model TorchScript path")
    ap.add_argument("best_ts",   help="best model TorchScript path")
    ap.add_argument("new_iter",  type=int)
    ap.add_argument("best_iter", type=int)
    ap.add_argument("--games",    type=int,   default=20)
    ap.add_argument("--sims",     type=int,   default=200)
    ap.add_argument("--win-rate", type=float, default=0.55)
    ap.add_argument("--workers",  type=int,   default=os.cpu_count() or 1)
    args = ap.parse_args()

    if not os.path.isfile(GATE_BINARY):
        print(f"ERROR: {GATE_BINARY} not found — run: make gatekeeper")
        sys.exit(1)

    num_workers = min(args.workers, args.games)
    base        = args.games // num_workers
    remainder   = args.games % num_workers

    print(f"\n=== gatekeeper_runner ===")
    print(f"New model : {args.new_ts} (iter {args.new_iter})")
    print(f"Best model: {args.best_ts} (iter {args.best_iter})")
    print(f"Games     : {args.games} across {num_workers} workers")
    print(f"Sims      : {args.sims}  (batch=1, full strength)")
    print(f"Device    : {'GPU' if USE_CUDA else 'CPU'}")
    print(f"WinRate   : {args.win_rate:.0%}\n")

    with tempfile.TemporaryDirectory(prefix="gatekeeper_") as tmp_dir:
        procs       = []
        worker_dirs = []
        log_files   = []
        games_list  = []

        for w in range(num_workers):
            game_start = w * base + min(w, remainder)
            game_count = base + (1 if w < remainder else 0)
            games_list.append(game_count)

            wdir = os.path.join(tmp_dir, f"worker_{w}")
            os.makedirs(wdir)
            worker_dirs.append(wdir)

            cmd = [
                GATE_BINARY,
                args.new_ts, args.best_ts,
                str(args.new_iter), str(args.best_iter),
                "--games",       str(game_count),
                "--sims",        str(args.sims),
                "--batch",       "1",
                "--game-offset", str(game_start),
                "--worker-dir",  wdir,
                "--seed",        str(42 + w * 1000),
            ]
            if USE_CUDA:
                cmd.append("--cuda")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"]   = "1"
            env["MKL_NUM_THREADS"]   = "1"
            env["TORCH_NUM_THREADS"] = "1"

            log_f = open(os.path.join(wdir, "worker.log"), "w")
            log_files.append(log_f)
            procs.append(subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=log_f,
                text=True,
                env=env,
            ))

        # One tqdm bar per worker, exactly like selfplay_cpp_runner.py
        bars = [
            tqdm(total=games_list[w],
                 desc=f"Worker {w:2d}",
                 unit="game",
                 position=w,
                 dynamic_ncols=True,
                 leave=True)
            for w in range(num_workers)
        ]

        stop_event = threading.Event()
        monitor    = threading.Thread(
            target=_monitor,
            args=(worker_dirs, games_list, stop_event, bars),
            daemon=True,
        )
        monitor.start()

        # Collect results
        total_wins = 0
        for proc in procs:
            stdout, _ = proc.communicate()
            for line in stdout.splitlines():
                if line.startswith("WINS:"):
                    total_wins += int(line.split()[1])

        stop_event.set()
        monitor.join()
        for bar in bars:
            bar.close()
        for f in log_files:
            f.close()

    pct = total_wins / args.games
    print(f"\nResult: new={total_wins}/{args.games} ({pct:.1%}), threshold {args.win_rate:.0%}")

    if pct >= args.win_rate:
        best_file = os.path.join(_HERE, cfg.SAVE_MODEL_PATH, "current_best_iter.txt")
        with open(best_file, "w") as f:
            f.write(f"{args.new_iter}\n")
        print("ACCEPTED")
        sys.exit(0)
    else:
        print("REJECTED")
        sys.exit(1)


if __name__ == "__main__":
    main()
