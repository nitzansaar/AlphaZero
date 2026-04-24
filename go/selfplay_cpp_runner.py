"""
selfplay_cpp_runner.py — C++-backed self-play data generator.

    python selfplay_cpp_runner.py

Requires:
    ./selfplay_cpp   — compiled from selfplay_cpp.cpp
    export_model.py  — included in this repo

How it works:
    1. Find the latest state-dict checkpoint in SAVE_MODEL_PATH
    2. Export it to TorchScript (.pt) so the C++ binary can load it
    3. Launch NUM_WORKERS C++ processes in parallel (1 thread each, own CUDA context)
    4. Load the .npy output from each worker and merge into the pickle dataset
    5. Save the merged dataset back to disk
"""

import os
import sys
import json
import time
import threading
import subprocess
import tempfile
import torch
from glob import glob
from tqdm import tqdm
from config import Config as cfg
from dataset import TrainingDataset

# ── Constants ────────────────────────────────────────────────────────────

# Path to the compiled C++ binary (same directory as this script).
_HERE = os.path.dirname(os.path.abspath(__file__))

# Select board-size-specific binary: keep "selfplay_cpp" for 9x9 (backward
# compat with any pre-compiled binary); use "selfplay_cpp_N" for other sizes.
_BINARY_NAME = "selfplay_cpp" if cfg.BOARD_SIZE == 9 else f"selfplay_cpp_{cfg.BOARD_SIZE}"
SELFPLAY_BINARY = os.path.join(_HERE, _BINARY_NAME)

# Cap workers from config (Config19x19Base sets NUM_SELFPLAY_WORKERS=4 because
# each 19x19 worker allocates ~374 MB for the node pool).
_worker_cap = getattr(cfg, 'NUM_SELFPLAY_WORKERS', None)
NUM_WORKERS = min(os.cpu_count() or 1, _worker_cap) if _worker_cap else (os.cpu_count() or 1)

# Whether to pass --cuda to the binary.  Falls back to CPU automatically
# if CUDA is unavailable on the target machine.
USE_CUDA = torch.cuda.is_available()

# ── Helper functions ──────────────────────────────────────────────────────

def get_latest_model_path():
    """Return the path of the highest-numbered model checkpoint, or None."""
    all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*.pt"))
    all_models = [m for m in all_models if not m.endswith("_ts.pt")]
    if not all_models:
        return None

    def _parse_iter(path):
        stem = os.path.basename(path).split("_")[0]
        try:
            return int(stem)
        except ValueError:
            return None

    files = [n for n in (_parse_iter(f) for f in all_models) if n is not None]
    if not files:
        return None
    latest_num = max(files)
    path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(latest_num))
    print(f"[Selfplay] Using latest model: iter_{latest_num} ({path})")
    return path


def _monitor_progress(worker_dirs, games_per_worker_list, stop_event, bars):
    """Background thread: poll each worker's progress file and update its tqdm bar."""
    last_counts = [0] * len(worker_dirs)

    def _flush():
        for i, (wdir, bar) in enumerate(zip(worker_dirs, bars)):
            try:
                with open(os.path.join(wdir, "progress")) as f:
                    count = int(f.read().strip())
            except (FileNotFoundError, ValueError):
                count = last_counts[i]
            if count > last_counts[i]:
                bar.update(count - last_counts[i])
                last_counts[i] = count

    while not stop_event.wait(0.5):
        _flush()
    _flush()  # final update after all processes finish


def export_torchscript(state_dict_path):
    """Run export_model.py to convert state_dict → TorchScript; return ts path."""
    ts_path = state_dict_path.replace("_best_model.pt", "_ts.pt")
    env = os.environ.copy()
    env["BOARD_SIZE"] = str(cfg.BOARD_SIZE)
    subprocess.run(
        [sys.executable, os.path.join(_HERE, "export_model.py"),
         state_dict_path, ts_path],
        env=env,
        check=True,
    )
    return ts_path


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if cfg.BOARD_SIZE == 5:
        print("ERROR: 5x5 uses Python selfplay — C++ binary not built for 5x5.")
        print("Use:  python run_training_loop.py   (or selfplay.py directly)")
        sys.exit(1)

    total_start = time.time()

    # Verify binary exists.
    if not os.path.isfile(SELFPLAY_BINARY):
        print(f"ERROR: C++ binary not found: {SELFPLAY_BINARY}")
        if cfg.BOARD_SIZE == 9:
            print("Compile it with:  make selfplay_cpp")
        else:
            print(f"Compile it with:  make selfplay_cpp_{cfg.BOARD_SIZE}")
        sys.exit(1)

    os.makedirs(cfg.SAVE_PICKLES, exist_ok=True)
    os.makedirs(cfg.LOGDIR, exist_ok=True)
    save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)

    # ── 1. Export TorchScript model ──────────────────────────────────────
    model_path = get_latest_model_path()
    if model_path is None:
        print("ERROR: No trained model found in", cfg.SAVE_MODEL_PATH)
        print("Run train.py first to generate an initial model.")
        sys.exit(1)

    print(f"State-dict model : {model_path}")
    export_start = time.time()
    ts_path = export_torchscript(model_path)
    export_time = time.time() - export_start
    print(f"TorchScript model: {ts_path}  ({export_time:.1f}s)")

    # ── 2. Run C++ selfplay processes in parallel ─────────────────────────
    # Never spawn more workers than games (avoids 0-game processes).
    effective_workers = min(NUM_WORKERS, cfg.SELFPLAY_GAMES)
    games_per_worker  = cfg.SELFPLAY_GAMES // effective_workers
    remainder         = cfg.SELFPLAY_GAMES % effective_workers

    print(f"\nRunning {cfg.SELFPLAY_GAMES} games across {effective_workers} workers "
          f"({'GPU' if USE_CUDA else 'CPU'}):")

    with tempfile.TemporaryDirectory(prefix="selfplay_cpp_") as tmp_dir:
        # Launch all workers in parallel.
        # Redirect each worker's stdout/stderr to a per-worker log file so
        # their output doesn't garble the tqdm progress bar.
        procs       = []
        worker_dirs = []
        log_files   = []
        for i in range(effective_workers):
            games      = games_per_worker + (1 if i < remainder else 0)
            worker_dir = os.path.join(tmp_dir, f"worker_{i}")
            os.makedirs(worker_dir)

            cmd = [
                SELFPLAY_BINARY,
                ts_path,
                "--games",          str(games),
                "--sims",           str(cfg.NUM_SIMULATIONS),
                "--batch",          "64",
                "--threads",        "1",
                "--output",         worker_dir,
                "--temp-moves",     str(cfg.TEMP_THRESHOLD),
                "--max-moves",      str(getattr(cfg, 'MAX_MOVES', 200)),
                "--seed",           str(i * 1000),
                "--full-prob",      str(cfg.PLAYOUT_CAP_PROB),
                "--fast-sims",      str(cfg.FAST_SIMS),
                "--min-pass-move",  str(getattr(cfg, 'MIN_PASS_MOVE', 0)),
            ]
            if USE_CUDA:
                cmd.append("--cuda")

            log_f = open(os.path.join(worker_dir, "worker.log"), "w")
            log_files.append(log_f)
            procs.append(subprocess.Popen(cmd, stdout=log_f, stderr=log_f))
            worker_dirs.append(worker_dir)

        print(f"  {games_per_worker}–{games_per_worker + (1 if remainder else 0)} "
              f"games per worker  (logs in each worker dir)\n")

        selfplay_start = time.time()

        # One tqdm bar per worker, just like the Python selfplay version.
        games_list = [games_per_worker + (1 if i < remainder else 0)
                      for i in range(effective_workers)]
        bars = [
            tqdm(total=games_list[i],
                 desc=f"Worker {i:2d}",
                 unit="game",
                 position=i,
                 dynamic_ncols=True,
                 leave=True)
            for i in range(effective_workers)
        ]

        stop_event = threading.Event()
        monitor    = threading.Thread(
            target=_monitor_progress,
            args=(worker_dirs, games_list, stop_event, bars),
            daemon=True,
        )
        monitor.start()

        # Wait for all workers and check for failures.
        failed = []
        for i, proc in enumerate(procs):
            proc.wait()
            if proc.returncode != 0:
                failed.append(i)

        stop_event.set()
        monitor.join()
        for bar in bars:
            bar.close()

        for f in log_files:
            f.close()

        selfplay_time = time.time() - selfplay_start

        if failed:
            # Print the last few lines of each failed worker's log to help debug.
            for i in failed:
                log_path = os.path.join(worker_dirs[i], "worker.log")
                try:
                    with open(log_path) as lf:
                        lines = lf.readlines()
                    print(f"\n--- worker {i} log (last 20 lines) ---")
                    print("".join(lines[-20:]))
                except OSError:
                    pass
            print(f"ERROR: workers {failed} exited with non-zero status.")
            sys.exit(1)

        # ── 3. Merge into the existing pickle dataset ─────────────────────
        training_dataset = TrainingDataset()
        if os.path.exists(save_path):
            training_dataset.load(save_path)
            print(f"Existing dataset : {len(training_dataset.training_dataset)} samples")
        else:
            print("Starting with empty dataset")

        for worker_dir in worker_dirs:
            training_dataset.load_from_npy(worker_dir)

    training_dataset.save(save_path)
    total_time = time.time() - total_start

    print(f"\nTotal training samples: {len(training_dataset.training_dataset)}")
    print(f"Self-play time   : {selfplay_time:.1f}s")
    print(f"Total time       : {total_time:.1f}s")

    # ── 4. Save timing data (same format as selfplay.py) ─────────────────
    # Aggregate per-phase timings written by each C++ worker.
    agg_timings = {}
    for wdir in worker_dirs:
        timing_path = os.path.join(wdir, "timing.json")
        try:
            with open(timing_path) as f:
                worker_t = json.load(f)["timings"]
            for k, v in worker_t.items():
                agg_timings[k] = agg_timings.get(k, 0.0) + v
        except (FileNotFoundError, KeyError, ValueError):
            pass
    # Round for readability
    agg_timings = {k: round(v, 4) for k, v in agg_timings.items()}

    timing_data = {
        "timings": agg_timings if agg_timings else {
            "selfplay_cpp": round(selfplay_time, 3),
        }
    }
    timing_path = os.path.join(cfg.LOGDIR, "selfplay_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)
    print(f"Timing saved to  : {timing_path}")
