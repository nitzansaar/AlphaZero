"""
Simple agents for testing and evaluation.
"""

import os
from typing import Tuple, List

import torch
import torch.nn.functional as F

try:
    # Use the exact model architecture defined in the trainer
    from src.trainer.train_neural_net import TicTacToeNet  # type: ignore
except Exception:
    TicTacToeNet = None  # type: ignore


class NNAgent:
    """Agent that plays using a trained NN (3x3 supported)."""
    
    def __init__(self, name: str = "NN Agent"):
        """
        Initialize agent.
        
        Args:
            name: Agent name for display
        """
        self.name = name
        self._model = None
        self._device = torch.device("cpu")
        self._maybe_load_model()
    
    def _maybe_load_model(self) -> None:
        """Load the NN weights if the architecture and checkpoint exist."""
        checkpoint_path = "tictactoe_model.pth"
        if TicTacToeNet is None:
            return
        if not os.path.exists(checkpoint_path):
            return
        try:
            model = TicTacToeNet()  # 9->64->9 MLP from trainer
            state = torch.load(checkpoint_path, map_location=self._device)
            model.load_state_dict(state)
            model.eval()
            self._model = model.to(self._device)
        except Exception:
            # Leave model as None to gracefully fall back to random
            self._model = None
    
    def _board_to_canonical_3d(self, env) -> torch.Tensor:
        """
        Convert board to canonical 3-plane representation.

        Returns tensor of shape (1, 3, 3, 3) for batched conv2d input:
        - Plane 0: Current player positions (1s)
        - Plane 1: Opponent positions (1s)
        - Plane 2: Empty positions (1s)
        """
        import numpy as np
        board = env.board  # shape (3, 3), values in {-1, 0, 1}
        current_player = env.current_player  # 1 or -1

        # Convert to canonical view (current player = +1, opponent = -1)
        canonical = board * current_player

        # Create 3-plane representation
        planes = np.zeros((3, 3, 3), dtype=np.float32)
        planes[0] = (canonical == 1).astype(np.float32)   # current player
        planes[1] = (canonical == -1).astype(np.float32)  # opponent
        planes[2] = (canonical == 0).astype(np.float32)   # empty

        # Convert to torch and add batch dimension: (3, 3, 3) -> (1, 3, 3, 3)
        return torch.from_numpy(planes).unsqueeze(0).to(self._device)

    def _nn_select_move(self, env) -> Tuple[int, int]:
        """Select move via NN logits masked by legality (3x3 only)."""
        if env.n != 3:
            raise RuntimeError("NN policy only supports 3x3 boards.")
        # Convert board to canonical 3-plane representation
        board_3d = self._board_to_canonical_3d(env)  # shape (1, 3, 3, 3)
        with torch.no_grad():
            logits = self._model(board_3d).squeeze(0)  # shape (9,)
        # Mask illegal moves
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
        mask = torch.full((9,), float("-inf"), device=logits.device)
        for (r, c) in legal_moves:
            idx = r * env.n + c
            mask[idx] = 0.0
        masked_logits = logits + mask
        # Softmax over masked logits to get probabilities on legal moves only
        probs = F.softmax(masked_logits, dim=-1)
        # Collect and print probabilities for legal moves
        dist = [((i // env.n, i % env.n), float(probs[i])) for i in range(9) if mask[i] == 0.0]
        dist.sort(key=lambda x: x[1], reverse=True)
        print("NN move probabilities:")
        for (r, c), p in dist:
            print(f"  ({r}, {c}): {p:.3f}")
        best_idx = int(torch.argmax(probs).item())
        return best_idx // env.n, best_idx % env.n
    
    def get_move(self, env) -> Tuple[int, int]:
        """
        Select a legal move using the neural network.
        
        Args:
            env: GameEnv instance
        
        Returns:
            (row, col) tuple of selected move
        """
        if self._model is None:
            raise RuntimeError("NN model not loaded. Train via 'python -m src.trainer.train_neural_net' to create tictactoe_model.pth.")
        return self._nn_select_move(env)


class HumanAgent:
    """Agent that asks for human input."""
    
    def __init__(self, name: str = "Human"):
        """
        Initialize human agent.
        
        Args:
            name: Agent name for display
        """
        self.name = name
    
    def get_move(self, env) -> Tuple[int, int]:
        """
        Get move from human player via command line input.
        
        Args:
            env: GameEnv instance
        
        Returns:
            (row, col) tuple of selected move
        """
        legal_moves = env.get_legal_moves()
        
        while True:
            try:
                move_input = input(f"\n{self.name}, enter your move (row col): ").strip()
                parts = move_input.split()
                
                if len(parts) != 2:
                    print("Please enter two numbers separated by space (e.g., '0 1')")
                    continue
                
                row, col = int(parts[0]), int(parts[1])
                
                if (row, col) not in legal_moves:
                    print(f"Invalid move ({row}, {col}). Legal moves: {legal_moves}")
                    continue
                
                return (row, col)
            
            except ValueError:
                print("Please enter valid integers")
            except KeyboardInterrupt:
                print("\nGame interrupted by user")
                raise
            except Exception as e:
                print(f"Error: {e}")

