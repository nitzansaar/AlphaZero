#!/usr/bin/env python3
"""
Command-line interface for playing N×N K-in-a-row.

Supports:
- Human vs Human
- Human vs Random AI
- Random vs Random (for testing)
"""

import argparse
import sys
from src.env.game_env import GameEnv
from src.eval.agents import RandomAgent, HumanAgent


def play_game(n: int, k: int, player1_type: str, player2_type: str):
    """
    Play a game with specified players.
    
    Args:
        n: Board size
        k: Win condition (k in a row)
        player1_type: 'human' or 'random'
        player2_type: 'human' or 'random'
    """
    # Create environment
    env = GameEnv(n, k)
    
    # Create agents
    agents = {}
    if player1_type == 'human':
        agents[1] = HumanAgent("Player 1 (X)")
    else:
        agents[1] = RandomAgent("Random 1 (X)")
    
    if player2_type == 'human':
        agents[-1] = HumanAgent("Player 2 (O)")
    else:
        agents[-1] = RandomAgent("Random 2 (O)")
    
    # Game loop
    print(f"\n{'='*40}")
    print(f"  {n}×{n} K-in-a-row Game (K={k})")
    print(f"{'='*40}")
    print(f"\nPlayer 1 (X): {agents[1].name}")
    print(f"Player 2 (O): {agents[-1].name}")
    print(f"\n{'='*40}\n")
    
    move_count = 0
    
    while not env.game_over:
        # Display board
        print(f"\nMove {move_count + 1}")
        print(env.render())
        
        # Get current player and agent
        current_player = env.current_player
        current_agent = agents[current_player]
        player_symbol = 'X' if current_player == 1 else 'O'
        
        print(f"\n{current_agent.name}'s turn ({player_symbol})")
        
        # Get move from agent
        try:
            row, col = current_agent.get_move(env)
            print(f"→ Playing move: ({row}, {col})")
            
            # Apply move
            _, reward, done = env.apply_move(row, col)
            move_count += 1
            
            if done:
                # Display final board
                print(f"\n{'='*40}")
                print("GAME OVER")
                print(f"{'='*40}")
                print(env.render())
                
                if env.winner is not None:
                    winner_symbol = 'X' if env.winner == 1 else 'O'
                    winner_name = agents[env.winner].name
                    print(f"\n{winner_name} ({winner_symbol}) wins!")
                else:
                    print(f"\nIt's a draw!")
                
                print(f"\nTotal moves: {move_count}")
                print(f"{'='*40}\n")
                break
        
        except KeyboardInterrupt:
            print("\n\nGame interrupted. Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Play N×N K-in-a-row game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 3×3 Tic-Tac-Toe, human vs random
  python play_cli.py
  
  # 5×5 with 4-in-a-row, human vs human
  python play_cli.py -n 5 -k 4 -p1 human -p2 human
  
  # Random vs Random (for testing)
  python play_cli.py -p1 random -p2 random
        """
    )
    
    parser.add_argument(
        '-n', '--board-size',
        type=int,
        default=3,
        help='Board size (default: 3)'
    )
    
    parser.add_argument(
        '-k', '--win-length',
        type=int,
        default=3,
        help='Number in a row to win (default: 3)'
    )
    
    parser.add_argument(
        '-p1', '--player1',
        type=str,
        choices=['human', 'random'],
        default='human',
        help='Player 1 type (default: human)'
    )
    
    parser.add_argument(
        '-p2', '--player2',
        type=str,
        choices=['human', 'random'],
        default='random',
        help='Player 2 type (default: random)'
    )
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.win_length > args.board_size:
        print(f"Error: Win length ({args.win_length}) cannot be greater than board size ({args.board_size})")
        sys.exit(1)
    
    try:
        play_game(args.board_size, args.win_length, args.player1, args.player2)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


if __name__ == '__main__':
    main()

