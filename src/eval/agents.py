"""
Simple agents for testing and evaluation.
"""

import random
from typing import Tuple, List


class RandomAgent:
    """Agent that plays random legal moves."""
    
    def __init__(self, name: str = "Random"):
        """
        Initialize random agent.
        
        Args:
            name: Agent name for display
        """
        self.name = name
    
    def get_move(self, env) -> Tuple[int, int]:
        """
        Select a random legal move.
        
        Args:
            env: GameEnv instance
        
        Returns:
            (row, col) tuple of selected move
        """
        # just pick a random move from the legal moves
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
        

        # instead of random choice, use the policy network to get the move
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

