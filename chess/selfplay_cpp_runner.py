"""
selfplay_cpp_runner.py — C++-backed self-play data generator for chess.

Drop-in replacement for selfplay.py that uses the compiled ./chess_selfplay
binary (30-100x faster than the pure-Python self-play).

    python selfplay_cpp_runner.py

How it works:
    1. Find the latest state-dict checkpoint in SAVE_MODEL_PATH.
    2. Export it to TorchScript so the C++ binary can load it (export_model.py).
    3. Launch NUM_WORKERS C++ processes in parallel (1 thread each).
    4. Merge each worker's .npy output into the pickle dataset.
    5. Save the merged dataset back to disk.

Requires the binary to be built first:  make all
"""
import os
import sys
import glob
import json
import time
import threading
import subprocess
import tempfile

import torch
from tqdm import tqdm

from config import Config as cfg
from dataset import TrainingDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
SELFPLAY_BINARY = os.path.join(_HERE, "chess_selfplay")

# Each worker holds its own ~180 MB node pool plus a model instance.  Default to
# a modest worker count; override via Config.NUM_SELFPLAY_WORKERS if present.
_worker_cap = getattr(cfg, "NUM_SELFPLAY_WORKERS", 4)
NUM_WORKERS = max(1, min(os.cpu_count() or 1, _worker_cap))

USE_CUDA = torch.cuda.is_available()

# C++ source files the binary depends on (freshness check).
_BINARY_SOURCES = [
    "chess_selfplay.cpp", "mcts.cpp", "mcts.h",
    "chess_encoding.cpp", "chess_encoding.h", "chess.hpp",
    "nn_inference.cpp", "nn_inference.h", "npy_writer.c", "npy_writer.h",
]


def find_latest_model():
    """Return (path, iteration) of the newest {n}_best_model.pt, or (None, -1)."""
    nums = []
    for f in glob.glob(os.path.join(cfg.SAVE_MODEL_PATH, "*.pt")):
        prefix = os.path.basename(f).split("_")[0]
        if prefix.lstrip("-").isdigit():
            nums.append(int(prefix))
    if not nums:
        return None, -1
    latest = max(nums)
    return os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(latest)), latest


def verify_binary_fresh():
    """Warn/exit if the binary is older than the sources it was built from."""
    if not os.path.isfile(SELFPLAY_BINARY):
        print(f"ERROR: C++ binary not found: {SELFPLAY_BINARY}")
        print("Build it with:  make all")
        sys.exit(1)
    binary_mtime = os.path.getmtime(SELFPLAY_BINARY)
    newest = 0.0
    for src in _BINARY_SOURCES:
        p = os.path.join(_HERE, src)
        if os.path.exists(p):
            newest = max(newest, os.path.getmtime(p))
    if newest > binary_mtime:
        print(f"ERROR: {SELFPLAY_BINARY} is older than its sources. Rebuild: make all")
        sys.exit(1)


def export_torchscript(state_dict_path):
    """Convert state_dict -> TorchScript next to it; return the ts path."""
    if state_dict_path.endswith("_best_model.pt"):
        ts_path = state_dict_path.replace("_best_model.pt", "_ts.pt")
    else:
        root, _ = os.path.splitext(state_dict_path)
        ts_path = f"{root}_ts.pt"
    subprocess.run(
        [sys.executable, os.path.join(_HERE, "export_model.py"),
         state_dict_path, ts_path],
        check=True,
    )
    return ts_path


def _monitor_progress(worker_dirs, stop_event, bars):
    """Poll each worker's progress file and advance its tqdm bar."""
    last = [0] * len(worker_dirs)

    def flush():
        for i, (wdir, bar) in enumerate(zip(worker_dirs, bars)):
            try:
                with open(os.path.join(wdir, "progress")) as f:
                    count = int(f.read().strip())
            except (FileNotFoundError, ValueError):
                count = last[i]
            if count > last[i]:
                bar.update(count - last[i])
                last[i] = count

    while not stop_event.wait(0.5):
        flush()
    flush()


def main():
    total_start = time.time()
    verify_binary_fresh()

    os.makedirs(cfg.SAVE_PICKLES, exist_ok=True)
    os.makedirs(cfg.LOGDIR, exist_ok=True)
    save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)

    # ── 1. Export TorchScript model ──────────────────────────────────────
    model_path, model_iter = find_latest_model()
    if model_path is None:
        print("ERROR: No trained model found in", cfg.SAVE_MODEL_PATH)
        print("Run train.py first to generate an initial model.")
        sys.exit(1)
    print(f"[Selfplay] Latest model: iter {model_iter} ({model_path})")
    ts_path = export_torchscript(model_path)
    print(f"[Selfplay] TorchScript : {ts_path}")

    seed_base = int(time.time()) ^ (os.getpid() << 16) ^ (max(model_iter, 0) * 1000003)
    seed_base &= 0x7FFFFFFF

    # ── 2. Run C++ selfplay workers in parallel ──────────────────────────
    effective_workers = max(1, min(NUM_WORKERS, cfg.SELFPLAY_GAMES))
    games_per_worker = cfg.SELFPLAY_GAMES // effective_workers
    remainder = cfg.SELFPLAY_GAMES % effective_workers

    print(f"\nRunning {cfg.SELFPLAY_GAMES} games across {effective_workers} workers "
          f"({'GPU' if USE_CUDA else 'CPU'}), {cfg.NUM_SIMULATIONS} sims/move:\n")

    agg_timings, agg_metrics = {}, {}
    with tempfile.TemporaryDirectory(prefix="chess_selfplay_") as tmp_dir:
        procs, worker_dirs, log_files, games_list = [], [], [], []
        for i in range(effective_workers):
            games = games_per_worker + (1 if i < remainder else 0)
            games_list.append(games)
            wdir = os.path.join(tmp_dir, f"worker_{i}")
            os.makedirs(wdir)
            worker_dirs.append(wdir)

            cmd = [
                SELFPLAY_BINARY, ts_path,
                "--games", str(games),
                "--sims", str(cfg.NUM_SIMULATIONS),
                "--batch", "32",
                "--threads", "1",
                "--output", wdir,
                "--temp-moves", str(cfg.TEMP_THRESHOLD),
                "--final-temp", str(cfg.FINAL_TEMP),
                "--max-moves", str(cfg.MAX_MOVES),
                "--seed", str(seed_base + i * 1000),
                "--c-puct", str(cfg.MCTS_UCB_C),
                "--dirichlet-alpha", str(cfg.DIRICHLET_ALPHA),
                "--dirichlet-frac", str(cfg.DIRICHLET_EPSILON),
            ]
            if USE_CUDA:
                cmd.append("--cuda")

            log_f = open(os.path.join(wdir, "worker.log"), "w")
            log_files.append(log_f)
            procs.append(subprocess.Popen(cmd, stdout=log_f, stderr=log_f))

        selfplay_start = time.time()
        bars = [tqdm(total=games_list[i], desc=f"Worker {i:2d}", unit="game",
                     position=i, dynamic_ncols=True, leave=True)
                for i in range(effective_workers)]
        stop_event = threading.Event()
        monitor = threading.Thread(target=_monitor_progress,
                                   args=(worker_dirs, stop_event, bars), daemon=True)
        monitor.start()

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
            for i in failed:
                try:
                    with open(os.path.join(worker_dirs[i], "worker.log")) as lf:
                        print(f"\n--- worker {i} log (last 20 lines) ---")
                        print("".join(lf.readlines()[-20:]))
                except OSError:
                    pass
            print(f"ERROR: workers {failed} exited with non-zero status.")
            sys.exit(1)

        # ── 3. Merge into the pickle dataset ─────────────────────────────
        training_dataset = TrainingDataset()
        if os.path.exists(save_path):
            training_dataset.load(save_path)
            print(f"\nExisting dataset : {len(training_dataset.training_dataset)} samples")
        else:
            print("\nStarting with empty dataset")

        for wdir in worker_dirs:
            training_dataset.load_from_npy(wdir)

        for wdir in worker_dirs:
            try:
                with open(os.path.join(wdir, "timing.json")) as f:
                    data = json.load(f)
                for k, v in data.get("timings", {}).items():
                    agg_timings[k] = agg_timings.get(k, 0.0) + v
                for k, v in data.get("metrics", {}).items():
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v
            except (FileNotFoundError, KeyError, ValueError):
                pass

    training_dataset.save(save_path)
    total_time = time.time() - total_start

    print(f"\nTotal training samples: {len(training_dataset.training_dataset)}")
    print(f"Self-play time : {selfplay_time:.1f}s   Total: {total_time:.1f}s")

    games_done = int(agg_metrics.get("completed_games", 0) or 0)
    moves = agg_metrics.get("total_game_moves", 0) or 0
    if games_done:
        agg_metrics["avg_game_moves"] = round(moves / games_done, 1)
    print(f"Games: {games_done}  metrics: {agg_metrics}")

    timing_path = os.path.join(cfg.LOGDIR, "selfplay_timing.json")
    with open(timing_path, "w") as f:
        json.dump({"timings": {k: round(v, 4) for k, v in agg_timings.items()},
                   "metrics": agg_metrics,
                   "selfplay_seconds": round(selfplay_time, 1)}, f, indent=2)
    print(f"Timing saved to : {timing_path}")


if __name__ == "__main__":
    main()
