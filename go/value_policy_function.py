from config import Config as cfg

import torch
import numpy as np
from model import NeuralNetwork
from game import board_to_canonical_17, board_to_planes_17_with_history, NUM_POSITIONS

device = "cuda" if torch.cuda.is_available() else "cpu"

# RTX 5090 Optimizations for inference
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

class ValuePolicyNetwork:
    def __init__(self, path=None, use_compile=True):
        self.original_model = NeuralNetwork().to(device)
        self.model_path = path  # kept so parallel workers can reload the model
        self._game_hist = []  # game-level history set by set_history()

        if path:
            try:
                loaded_state = torch.load(path, map_location=device)
                # Handle models saved with _orig_mod prefix (from torch.compile)
                if any(key.startswith('_orig_mod.') for key in loaded_state.keys()):
                    new_state_dict = {}
                    for key, value in loaded_state.items():
                        if key.startswith('_orig_mod.'):
                            new_key = key[len('_orig_mod.'):]
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    loaded_state = new_state_dict

                self.original_model.load_state_dict(loaded_state)
                print(f"Loaded model from {path}")
            except RuntimeError as e:
                print(f"Warning: Could not load model from {path} (architecture mismatch)")
                print(f"Error details: {e}")
                print("Using randomly initialized model instead")

        self.original_model.eval()

        # Compile model for faster inference
        if use_compile and device == "cuda" and hasattr(torch, 'compile'):
            print("Compiling inference model with torch.compile...")
            self.model = torch.compile(self.original_model, mode="reduce-overhead")
            print("Inference model compilation complete!")
        else:
            self.model = self.original_model

    def set_history(self, hist_boards_abs):
        """
        Set the game-level board history (boards before the current root position).

        Args:
            hist_boards_abs: List of absolute boards (Black=+1, White=-1),
                             newest first, capturing positions before the root.
        """
        self._game_hist = list(hist_boards_abs)

    def _build_history(self, node):
        """
        Build up to 7 prior absolute boards for a leaf node.

        Walks the parent chain (newest first) then falls back to _game_hist.

        Args:
            node: MCTS leaf node being evaluated.

        Returns:
            List of up to 7 absolute boards (Black=+1, White=-1), newest first.
        """
        hist = []
        ancestor = node.parent if node is not None else None
        while ancestor is not None and len(hist) < 7:
            if ancestor.state is not None:
                abs_board = ancestor.state[:NUM_POSITIONS] * ancestor.player
                hist.append(abs_board)
            ancestor = ancestor.parent
        remaining = 7 - len(hist)
        hist.extend(self._game_hist[:remaining])
        return hist

    def get_vp(self, state, player=1, node=None):
        """
        Get value and policy predictions for a board state.

        Args:
            state: Game state array in absolute form (Black=+1, White=-1)
            player: Current player (1=Black, -1=White)
            node: Optional MCTS node; when provided, history is built from the
                  parent chain + _game_hist for full 17-plane input.

        Returns:
            value: Position evaluation (float)
            policy: Move probabilities (array of ACTION_SIZE values)
        """
        hist = self._build_history(node) if node is not None else []
        planes = board_to_planes_17_with_history(state, player, hist)

        # Convert to tensor and add batch dimension: (17, B, B) -> (1, 17, B, B)
        state_tensor = torch.from_numpy(planes).unsqueeze(0).to(device)

        with torch.no_grad():
            value, policy = self.model(state_tensor)

        value = value.cpu().numpy().flatten()[0]
        policy = torch.nn.functional.softmax(policy, dim=1)
        policy = policy.cpu().numpy().flatten()

        return value, policy

    def get_vp_batch(self, states, players, nodes=None):
        """
        Get value and policy predictions for multiple board states in a single batch.
        Much more efficient for GPU utilization than calling get_vp multiple times.

        Args:
            states: List of game state arrays in absolute form
            players: List of current players (1 or -1) for each state
            nodes: Optional list of MCTS nodes; when provided, per-leaf history
                   is built from the parent chain + _game_hist.

        Returns:
            values: List of position evaluations (floats)
            policies: List of move probability arrays
        """
        if len(states) == 0:
            return [], []

        # Convert all states to 17-plane representation (with history if available)
        if nodes is not None:
            canonical_states = np.stack([
                board_to_planes_17_with_history(state, player, self._build_history(node))
                for state, player, node in zip(states, players, nodes)
            ])
        else:
            canonical_states = np.stack([
                board_to_planes_17_with_history(state, player, [])
                for state, player in zip(states, players)
            ])

        # Convert to tensor: (batch, 17, B, B)
        state_tensor = torch.from_numpy(canonical_states).to(device)

        with torch.no_grad():
            values, policies = self.model(state_tensor)

        # Single GPU->CPU sync for the entire batch
        values = values.cpu().numpy().flatten()
        policies = torch.nn.functional.softmax(policies, dim=1)
        policies = policies.cpu().numpy()

        return list(values), list(policies)
