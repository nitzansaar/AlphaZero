"""
Simple agents for testing and evaluation.
"""

import os
from typing import Tuple, List

import torch
import torch.nn.functional as F

try:
    # Use the exact model architecture and MCTS from self-play trainer
    from src.trainer.train_self_play import TicTacToeNet, MCTS  # type: ignore
except Exception:
    TicTacToeNet = None  # type: ignore
    MCTS = None  # type: ignore


class NNAgent:
    """Agent that plays using AlphaZero-style MCTS + NN (3x3 supported)."""

    def __init__(self, name: str = "NN Agent", num_simulations: int = 500):
        """
        Initialize agent.

        Args:
            name: Agent name for display
            num_simulations: Number of MCTS simulations per move (higher = stronger but slower)
        """
        self.name = name
        self._model = None
        self._mcts = None
        self._device = torch.device("cpu")
        self._num_simulations = num_simulations
        self._maybe_load_model()
    
    def _maybe_load_model(self) -> None:
        """Load the NN weights and initialize MCTS if the architecture and checkpoint exist."""
        checkpoint_path = os.path.join("models", "tictactoe_selfplay_final.pth")
        if TicTacToeNet is None or MCTS is None:
            return
        if not os.path.exists(checkpoint_path):
            # Backward compatibility: fall back to root if models/ not used yet
            fallback = "tictactoe_selfplay_final.pth"
            if os.path.exists(fallback):
                checkpoint_path = fallback
            else:
                return
        try:
            model = TicTacToeNet()  # AlphaZero-style Conv2d model with policy + value heads
            state = torch.load(checkpoint_path, map_location=self._device)
            model.load_state_dict(state)
            model.eval()
            self._model = model.to(self._device)

            # Initialize MCTS with the loaded model (debug=True to see visit counts)
            self._mcts = MCTS(self._model, str(self._device), num_simulations=self._num_simulations, debug=True)
        except Exception:
            # Leave model as None to gracefully fall back to error
            self._model = None
            self._mcts = None
    
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

    def _mcts_select_move(self, env) -> Tuple[int, int]:
        """Select move using MCTS search (3x3 only)."""
        if env.n != 3:
            raise RuntimeError("MCTS policy only supports 3x3 boards.")

        # Convert board to flat list format for MCTS
        board_flat = env.board.flatten().tolist()
        current_player = env.current_player

        print(f"Running MCTS with {self._num_simulations} simulations...")

        # Run MCTS to get action probabilities
        # During actual gameplay (not training), use pure exploitation:
        # - temperature=0: Select only the most-visited move (greedy/deterministic)
        # - add_noise=False: No exploration noise, use network policy as-is
        # This gives the strongest play based on what the network learned during training
        action_probs, _ = self._mcts.get_action_probs(
            board_flat, current_player,
            temperature=0,      # Greedy: always pick best move (most visits)
            add_noise=False     # No exploration during gameplay
        )

        # Print MCTS move probabilities for legal moves
        legal_moves = env.get_legal_moves()
        dist = []
        for r, c in legal_moves:
            idx = r * env.n + c
            dist.append(((r, c), action_probs[idx]))
        dist.sort(key=lambda x: x[1], reverse=True)

        print("MCTS move probabilities:")
        for (r, c), p in dist:
            print(f"  ({r}, {c}): {p:.3f}")

        # Select best move (highest probability from MCTS)
        best_idx = int(max(range(9), key=lambda i: action_probs[i]))
        return best_idx // env.n, best_idx % env.n
    
    def get_move(self, env) -> Tuple[int, int]:
        """
        Select a legal move using MCTS + neural network.

        Args:
            env: GameEnv instance

        Returns:
            (row, col) tuple of selected move
        """
        if self._model is None or self._mcts is None:
            raise RuntimeError("AlphaZero model not loaded. Train via 'python -m src.trainer.train_self_play' to create tictactoe_selfplay_final.pth.")
        return self._mcts_select_move(env)


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

