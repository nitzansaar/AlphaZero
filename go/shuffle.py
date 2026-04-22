"""
KataGo-style shuffle buffer for self-play training data.

Enforces:
  - Minimum row count before training begins (MIN_ROWS)
  - Linear temporal taper: oldest sample → weight 0.1, newest → weight 1.0
  - Optional hard cap on buffer size (max_rows)

Usage:
    from shuffle import apply_shuffle, MIN_ROWS

    rows, weights = apply_shuffle(training_dataset)
    if rows is None:
        print(f"Only {len(training_dataset)} rows — need {MIN_ROWS} to start training.")
        return
    # rows is a (possibly capped) slice of training_dataset, newest samples last
    # weights is a float32 array of the same length for WeightedRandomSampler
"""

import numpy as np

MIN_ROWS = 10_000


def apply_shuffle(training_dataset, max_rows=None):
    """
    Apply KataGo-style temporal taper to a training dataset.

    Args:
        training_dataset: list of samples in chronological order (oldest first).
        max_rows: if given, cap the buffer to the newest max_rows samples before
                  computing weights.  Defaults to None (use all samples).

    Returns:
        (rows, weights)  if len(training_dataset) >= MIN_ROWS
        (None, None)     if dataset is too small — caller should skip training.
    """
    n = len(training_dataset)
    if n < MIN_ROWS:
        return None, None

    rows = training_dataset if max_rows is None else training_dataset[-max_rows:]
    n = len(rows)

    # Linear ramp: oldest sample gets weight 0.1, newest gets 1.0.
    # This ensures new data drives the gradient signal while old data still
    # contributes, matching KataGo's tapered-window approach.
    weights = np.linspace(0.1, 1.0, n, dtype=np.float32)

    return rows, weights
