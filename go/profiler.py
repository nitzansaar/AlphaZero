"""
Simple profiling utility for tracking time spent in each phase of training.

Usage:
    timer = Timer()
    with timer.track("mcts_simulation"):
        mcts.run_simulation_batched(...)

    # Get results as dict
    print(timer.summary())

    # Save to JSON
    timer.save("timing.json")
"""

import time
import json
import os
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.timings = {}  # name -> total seconds
        self.counts = {}   # name -> call count

    @contextmanager
    def track(self, name):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.timings[name] = self.timings.get(name, 0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self):
        """Return dict of {name: {"total": seconds, "calls": count, "avg": avg_seconds}}"""
        result = {}
        for name in self.timings:
            total = self.timings[name]
            calls = self.counts[name]
            result[name] = {
                "total": round(total, 4),
                "calls": calls,
                "avg": round(total / calls, 6) if calls > 0 else 0,
            }
        return result

    def to_dict(self):
        """Return flat dict of {name: total_seconds} for pie chart."""
        return {k: round(v, 4) for k, v in self.timings.items()}

    def save(self, path):
        """Save timing data to JSON."""
        data = {
            "timings": self.to_dict(),
            "details": self.summary(),
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def merge(self, other_dict):
        """Accumulate another timing dict (from a worker) for later averaging."""
        for name, seconds in other_dict.items():
            self.timings[name] = self.timings.get(name, 0) + seconds
            self.counts[name] = self.counts.get(name, 0) + 1

    def average(self):
        """Average accumulated timings by number of merges (workers). Call after all merges."""
        for name in self.timings:
            if self.counts[name] > 0:
                self.timings[name] /= self.counts[name]

    def print_summary(self, title="Timing Summary"):
        """Print a formatted summary."""
        total = sum(self.timings.values())
        print(f"\n{'='*50}")
        print(f" {title}")
        print(f"{'='*50}")
        # Sort by time descending
        sorted_items = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        for name, secs in sorted_items:
            pct = (secs / total * 100) if total > 0 else 0
            calls = self.counts.get(name, 0)
            print(f"  {name:30s}  {secs:8.2f}s  ({pct:5.1f}%)  [{calls} calls]")
        print(f"  {'TOTAL':30s}  {total:8.2f}s")
        print(f"{'='*50}")
