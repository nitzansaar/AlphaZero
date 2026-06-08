import numpy as np
import torch

from config import Config as cfg
from model import NeuralNetwork
import encoding

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    torch.set_num_threads(1)


def _strip_orig_mod(state_dict):
    """Drop the _orig_mod. prefix added by torch.compile when saving."""
    new_state = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_state[key[len("_orig_mod."):]] = value
        else:
            new_state[key] = value
    return new_state


class ValuePolicyNetwork:
    def __init__(self, path=None, use_compile=True):
        self.original_model = NeuralNetwork().to(device)

        if path:
            try:
                loaded_state = torch.load(path, map_location=device)
                loaded_state = _strip_orig_mod(loaded_state)
                self.original_model.load_state_dict(loaded_state)
                print(f"Loaded model from {path}")
            except RuntimeError as e:
                print(f"Warning: Could not load model from {path} (architecture mismatch)")
                print(f"Error details: {e}")
                print("Using randomly initialized model instead")

        self.original_model.eval()

        if use_compile and device == "cuda" and hasattr(torch, "compile"):
            print("Compiling inference model with torch.compile...")
            self.model = torch.compile(self.original_model, mode="reduce-overhead")
            print("Inference model compilation complete!")
        else:
            self.model = self.original_model

    def get_vp(self, board):
        """Return (value, policy) for a chess.Board.

        value:  float in [-1, 1] from the side-to-move's perspective.
        policy: length-ACTION_SIZE np.array, masked to legal moves and
                normalized to sum to 1.
        """
        planes = encoding.board_to_planes(board)
        state_tensor = torch.from_numpy(planes).unsqueeze(0).to(device)

        with torch.no_grad():
            value, policy_logits = self.model(state_tensor)

        value = float(value.cpu().numpy().flatten()[0])
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()

        # Mask illegal moves and renormalize.
        mask = encoding.legal_policy_mask(board)
        policy = policy * mask
        total = policy.sum()
        if total > 0:
            policy = policy / total
        elif mask.sum() > 0:
            # Network gave all illegal mass; fall back to uniform over legal.
            policy = mask / mask.sum()

        return value, policy

    def get_vp_batch(self, boards, pad_to=None):
        """Vectorized get_vp over a list of chess.Board.

        Runs a single batched forward pass and returns a list of
        (value, policy) tuples, one per input board, in order.

        pad_to: if set, the network input is zero-padded up to this many rows
        before the forward pass (extra outputs are discarded). This keeps the
        input shape constant across calls so torch.compile's CUDA-graph
        (reduce-overhead) capture is reused instead of recompiling per batch
        size. Pass the max concurrency (cfg.NUM_PARALLEL_GAMES).
        """
        n = len(boards)
        if n == 0:
            return []

        planes = np.stack([encoding.board_to_planes(b) for b in boards])
        if pad_to is not None and pad_to > n:
            pad = np.zeros((pad_to - n, *planes.shape[1:]), dtype=planes.dtype)
            planes = np.concatenate([planes, pad], axis=0)

        state_tensor = torch.from_numpy(planes).to(device)
        with torch.no_grad():
            value, policy_logits = self.model(state_tensor)

        # One GPU->CPU sync for the whole batch (vs. one per board in get_vp).
        values = value.detach().cpu().numpy().reshape(-1)[:n]
        policies = torch.softmax(policy_logits[:n], dim=1).cpu().numpy()

        results = []
        for i, board in enumerate(boards):
            policy = policies[i]
            mask = encoding.legal_policy_mask(board)
            policy = policy * mask
            total = policy.sum()
            if total > 0:
                policy = policy / total
            elif mask.sum() > 0:
                policy = mask / mask.sum()
            results.append((float(values[i]), policy))

        return results
