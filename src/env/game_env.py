"""
Generalized N×N K-in-a-row game environment.

This module provides the game logic for a configurable board size and win condition.
Players are represented as: 1 (Player X), -1 (Player O), 0 (Empty)
"""

import numpy as np
from typing import List, Tuple, Optional


class GameEnv:
    """N×N board with K-in-a-row win condition."""
    
    def __init__(self, n: int = 3, k: int = 3):
        """
        Initialize the game environment.
        
        Args:
            n: Board size (N×N)
            k: Number of consecutive pieces needed to win
        """
        if k > n:
            raise ValueError(f"K ({k}) cannot be greater than N ({n})")
        if n < 1:
            raise ValueError(f"N must be at least 1, got {n}")
        if k < 1:
            raise ValueError(f"K must be at least 1, got {k}")
        
        self.n = n
        self.k = k
        self.board = np.zeros((n, n), dtype=np.int8)
        self.current_player = 1  # Player 1 (X) starts
        self.move_count = 0
        self.game_over = False
        self.winner = None
    
    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.board = np.zeros((self.n, self.n), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
        self.game_over = False
        self.winner = None
        return self.board.copy()
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Get all legal moves (empty positions).
        
        Returns:
            List of (row, col) tuples representing empty positions
        """
        if self.game_over:
            return []
        
        # Find all positions with value 0 (empty)
        empty_positions = np.argwhere(self.board == 0)
        return [(int(row), int(col)) for row, col in empty_positions]
    
    def apply_move(self, row: int, col: int) -> Tuple[np.ndarray, float, bool]:
        """
        Apply a move to the board.
        
        Args:
            row: Row index (0-indexed)
            col: Column index (0-indexed)
        
        Returns:
            Tuple of (new_board_state, reward, done)
            - reward: 1 if current player wins, -1 if loses, 0 otherwise
            - done: True if game is over
        
        Raises:
            ValueError: If move is illegal
        """
        if self.game_over:
            raise ValueError("Game is already over")
        
        if row < 0 or row >= self.n or col < 0 or col >= self.n:
            raise ValueError(f"Move ({row}, {col}) is out of bounds for {self.n}×{self.n} board")
        
        if self.board[row, col] != 0:
            raise ValueError(f"Position ({row}, {col}) is already occupied")
        
        # Apply the move
        self.board[row, col] = self.current_player
        self.move_count += 1
        
        # Check for winner
        winner = self.check_winner()
        if winner is not None:
            self.game_over = True
            self.winner = winner
            reward = 1.0 if winner == self.current_player else -1.0
            return self.board.copy(), reward, True
        
        # Check for draw
        if self.is_draw():
            self.game_over = True
            return self.board.copy(), 0.0, True
        
        # Switch player
        self.current_player *= -1
        
        return self.board.copy(), 0.0, False
    
    def check_winner(self) -> Optional[int]:
        """
        Check if there's a winner.
        
        Returns:
            1 if Player 1 wins, -1 if Player 2 wins, None if no winner yet
        """
        # Check all possible winning positions for both players
        for player in [1, -1]:
            # Check horizontal
            for row in range(self.n):
                for col in range(self.n - self.k + 1):
                    if np.all(self.board[row, col:col + self.k] == player):
                        return player
            
            # Check vertical
            for row in range(self.n - self.k + 1):
                for col in range(self.n):
                    if np.all(self.board[row:row + self.k, col] == player):
                        return player
            
            # Check diagonal (top-left to bottom-right)
            for row in range(self.n - self.k + 1):
                for col in range(self.n - self.k + 1):
                    diagonal = np.array([self.board[row + i, col + i] for i in range(self.k)])
                    if np.all(diagonal == player):
                        return player
            
            # Check anti-diagonal (top-right to bottom-left)
            for row in range(self.n - self.k + 1):
                for col in range(self.k - 1, self.n):
                    anti_diagonal = np.array([self.board[row + i, col - i] for i in range(self.k)])
                    if np.all(anti_diagonal == player):
                        return player
        
        return None
    
    def is_draw(self) -> bool:
        """
        Check if the game is a draw (board full with no winner).
        
        Returns:
            True if game is a draw, False otherwise
        """
        return self.move_count == self.n * self.n and self.check_winner() is None
    
    def get_board_encoding(self) -> np.ndarray:
        """
        Encode the board as one-hot planes for neural network input.
        
        Returns:
            np.ndarray of shape (3, N, N) with planes:
                - Plane 0: Player 1 (X) positions
                - Plane 1: Player 2 (O) positions
                - Plane 2: Empty positions
        """
        encoding = np.zeros((3, self.n, self.n), dtype=np.float32)
        
        # Plane 0: Player 1 positions
        encoding[0] = (self.board == 1).astype(np.float32)
        
        # Plane 1: Player 2 positions
        encoding[1] = (self.board == -1).astype(np.float32)
        
        # Plane 2: Empty positions
        encoding[2] = (self.board == 0).astype(np.float32)
        
        return encoding
    
    def get_canonical_board(self) -> np.ndarray:
        """
        Get board from current player's perspective.
        
        For neural network training, we want the board always represented
        from the current player's perspective (current player as 1, opponent as -1).
        
        Returns:
            Board with current player's pieces as 1, opponent's as -1
        """
        return self.board * self.current_player
    
    def clone(self) -> 'GameEnv':
        """
        Create a deep copy of the current game state.
        
        Returns:
            New GameEnv instance with identical state
        """
        new_env = GameEnv(self.n, self.k)
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.move_count = self.move_count
        new_env.game_over = self.game_over
        new_env.winner = self.winner
        return new_env
    
    def render(self) -> str:
        """
        Render the board as a string for display.
        
        Returns:
            String representation of the board
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        
        # Column numbers header
        col_header = '  ' + ' '.join(str(i) for i in range(self.n))
        lines.append(col_header)
        lines.append('  ' + '-' * (2 * self.n - 1))
        
        # Board rows
        for i, row in enumerate(self.board):
            row_str = f"{i}|" + ' '.join(symbols[cell] for cell in row)
            lines.append(row_str)
        
        return '\n'.join(lines)
    
    def __str__(self) -> str:
        """String representation of the game state."""
        return self.render()
    
    def __repr__(self) -> str:
        """Detailed representation of the game state."""
        return f"GameEnv(n={self.n}, k={self.k}, player={self.current_player}, moves={self.move_count})"

