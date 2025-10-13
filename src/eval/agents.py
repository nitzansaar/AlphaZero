"""
Simple agents for testing and evaluation.
"""

import os
import random
from typing import Tuple, List

import numpy as np
import torch

try:
    # Use the exact model architecture defined in the trainer
    from src.trainer.train_neural_net import TicTacToeNet  # type: ignore
except Exception:
    TicTacToeNet = None  # type: ignore


class RandomAgent:
    """Agent that plays using a trained NN if available, otherwise random."""
    
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
    
    def _nn_select_move(self, env) -> Tuple[int, int]:
        """Select move via NN logits masked by legality (3x3 only)."""
        # Only defined for 3x3 as per training setup
        if env.n != 3:
            raise RuntimeError("NN policy only supports 3x3 boards. Falling back to random.")
        # Flatten board values (-1, 0, 1) to length-9 float32 tensor
        board_flat = torch.tensor(env.board.reshape(-1), dtype=torch.float32, device=self._device)
        with torch.no_grad():
            logits = self._model(board_flat)  # shape (9,)
        # Mask illegal moves
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
        mask = torch.full((9,), float("-inf"), device=logits.device)
        for (r, c) in legal_moves:
            idx = r * env.n + c
            mask[idx] = 0.0
        masked_logits = logits + mask
        best_idx = int(torch.argmax(masked_logits).item())
        return best_idx // env.n, best_idx % env.n
    
    def get_move(self, env) -> Tuple[int, int]:
        """
        Select a legal move using the neural network if available; else random.
        
        Args:
            env: GameEnv instance
        
        Returns:
            (row, col) tuple of selected move
        """
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if self._model is not None:
            try:
                return self._nn_select_move(env)
            except Exception:
                # Fallback to random if NN selection fails for any reason
                pass
        
        return random.choice(legal_moves)


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

