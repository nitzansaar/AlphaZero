"""
Random player that selects moves uniformly at random from legal moves.

This player serves as a baseline for evaluating the performance of trained agents.
"""

import random
from typing import List


class RandomPlayer:
    """A player that makes completely random moves."""

    def __init__(self):
        """Initialize the random player."""
        self.name = "Random Player"

    def get_move(self, board: List[int]) -> int:
        """
        Select a random legal move from the board.

        Args:
            board: Flat list of 9 elements representing the board state
                   (0 = empty, 1 = X, -1 = O)

        Returns:
            Index of the selected move (0-8)
        """
        legal_moves = [i for i in range(len(board)) if board[i] == 0]

        if not legal_moves:
            raise ValueError("No legal moves available")

        return random.choice(legal_moves)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"RandomPlayer()"
