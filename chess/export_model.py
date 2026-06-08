#!/usr/bin/env python3
"""
export_model.py — Convert a chess state_dict .pt to TorchScript for C++ inference.

The chess training loop (train.py) saves raw state_dicts; the C++ self-play
binary loads TorchScript.  This bridges the two.

Usage:
    python export_model.py <state_dict.pt> <output_ts.pt>
"""
import sys
import torch

from config import Config as cfg
from model import NeuralNetwork


def strip_orig_mod(state_dict):
    """Remove the '_orig_mod.' prefix added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <state_dict.pt> <output_ts.pt>",
              file=sys.stderr)
        sys.exit(1)

    state_path, output_path = sys.argv[1], sys.argv[2]

    model = NeuralNetwork()
    raw = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(strip_orig_mod(raw))
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(output_path)

    # Sanity check: shapes match what the C++ binary expects.
    with torch.no_grad():
        dummy = torch.zeros(1, cfg.NUM_INPUT_PLANES, 8, 8)
        v, p = scripted(dummy)
        assert v.shape == (1, 1), f"unexpected value shape {tuple(v.shape)}"
        assert p.shape == (1, cfg.ACTION_SIZE), f"unexpected policy shape {tuple(p.shape)}"

    print(f"Saved TorchScript model to: {output_path}")
    print(f"  Input planes : {cfg.NUM_INPUT_PLANES}")
    print(f"  Action size  : {cfg.ACTION_SIZE}")
    print(f"  Value shape  : {list(v.shape)}")
    print(f"  Policy shape : {list(p.shape)}")


if __name__ == "__main__":
    main()
