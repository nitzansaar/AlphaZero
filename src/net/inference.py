# src/net/inference.py
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from src.env.game_env import GameEnv
from src.net.model import Net  # use the Net you implemented earlier

def make_canonical_planes(env: GameEnv) -> torch.Tensor:
    """
    Build (3, N, N) planes from the current player's perspective:
    - plane 0: current player's stones (1)
    - plane 1: opponent stones (-1)
    - plane 2: empty
    """
    board = env.get_canonical_board()  # current player as +1
    n = env.n
    planes = np.zeros((3, n, n), dtype=np.float32)
    planes[0] = (board == 1).astype(np.float32)
    planes[1] = (board == -1).astype(np.float32)
    planes[2] = (board == 0).astype(np.float32)
    return torch.from_numpy(planes)  # (3, N, N)

@torch.no_grad()
def predict_from_env(
    model: Net,
    env: GameEnv,
    device: torch.device | str = "cpu",
    temperature: float = 1.0
) -> Tuple[Dict[Tuple[int, int], float], float]:
    """
    Returns:
      - move_probs: dict mapping (row, col) -> probability (legal moves, sums to 1)
      - win_pct: current-player win percentage in [0, 1]
    """
    model.eval()
    n = env.n

    # Build input
    planes = make_canonical_planes(env).unsqueeze(0).to(device)  # (1, 3, N, N)

    # Forward
    logits, value = model(planes)  # logits: (1, N*N), value: (1, 1) in [-1, 1]

    # Mask illegal moves
    legal = env.get_legal_moves()
    mask = torch.full((n * n,), float("-inf"), device=logits.device)
    for (r, c) in legal:
        mask[r * n + c] = 0.0  # unmask legal indices

    masked_logits = logits[0] + mask  # (N*N,)

    # Temperature softmax over ONLY legal moves
    if len(legal) == 0:
        move_probs = {}
    else:
        if temperature <= 0:
            # Hard argmax among legal
            best_idx = torch.argmax(masked_logits)
            probs = torch.zeros_like(masked_logits)
            probs[best_idx] = 1.0
        else:
            probs = F.softmax(masked_logits / temperature, dim=-1)

        # Map back to (row, col) -> prob for legal moves
        move_probs = {(i // n, i % n): float(probs[i]) for i in range(n * n) if mask[i] == 0.0}

        # Numerical renorm to ensure sum to 1 on legal (softmax already does, but safe)
        s = sum(move_probs.values())
        if s > 0:
            for k in list(move_probs.keys()):
                move_probs[k] /= s

    # Convert value in [-1, 1] to winning percentage in [0, 1]
    win_pct = float((value.item() + 1.0) / 2.0)

    return move_probs, win_pct