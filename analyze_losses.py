"""
Analyze saved loss games to understand why the neural network lost.

This script reads the saved loss files from the nn_losses directory
and displays them in a human-readable format to help diagnose issues.
"""

import json
import os
from typing import List


def print_board(board: List[int], position_labels: bool = False) -> None:
    """
    Print the board in a readable format.

    Args:
        board: Flat list of 9 elements representing the board state
        position_labels: If True, show position numbers instead of empty cells
    """
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("\n  0 1 2")
    print("  -----")
    for i in range(3):
        row = board[i*3:(i+1)*3]
        if position_labels:
            print(f"{i}|" + ' '.join(str(i*3 + j) if cell == 0 else symbols[cell]
                                     for j, cell in enumerate(row)))
        else:
            print(f"{i}|" + ' '.join(symbols[cell] for cell in row))
    print()


def analyze_loss_file(filepath: str, verbose: bool = True) -> None:
    """
    Analyze a single loss file.

    Args:
        filepath: Path to the JSON loss file
        verbose: If True, print detailed move-by-move analysis
    """
    with open(filepath, 'r') as f:
        loss_data = json.load(f)

    print("=" * 70)
    print(f"LOSS ANALYSIS - Game #{loss_data['game_number']}")
    print("=" * 70)
    print(f"File: {filepath}")
    print(f"Timestamp: {loss_data['timestamp']}")
    print(f"Neural Network played as: {loss_data['nn_played_as']}")
    print(f"Total moves in game: {loss_data['total_moves']}")
    print()

    if verbose:
        print("Move-by-move replay:")
        print("-" * 70)

        for move in loss_data['moves']:
            is_nn_move = (move['player_symbol'] == loss_data['nn_played_as'])
            player_name = "NN" if is_nn_move else "Random"

            print(f"\nMove {move['move_number']}: {player_name} ({move['player_symbol']}) -> position {move['move_position']}")
            print("Board before move:")
            print_board(move['board_before'])

        # Show final board state
        print("\n" + "=" * 70)
        print("FINAL BOARD STATE (NN Lost):")
        # Apply last move to show final state
        final_board = loss_data['moves'][-1]['board_before'].copy()
        final_board[loss_data['moves'][-1]['move_position']] = loss_data['moves'][-1]['player']
        print_board(final_board)
        print("=" * 70)

    return loss_data


def list_all_losses(loss_dir: str = "nn_losses") -> List[str]:
    """
    List all loss files in the directory.

    Args:
        loss_dir: Directory containing loss files

    Returns:
        List of file paths
    """
    if not os.path.exists(loss_dir):
        print(f"No losses directory found at: {loss_dir}")
        return []

    loss_files = [f for f in os.listdir(loss_dir) if f.endswith('.json')]
    loss_files.sort()

    return [os.path.join(loss_dir, f) for f in loss_files]


def summarize_losses(loss_dir: str = "nn_losses") -> None:
    """
    Print a summary of all losses.

    Args:
        loss_dir: Directory containing loss files
    """
    loss_files = list_all_losses(loss_dir)

    if not loss_files:
        print("No loss files found.")
        return

    print(f"\nFound {len(loss_files)} loss files in '{loss_dir}/'")
    print("=" * 70)

    losses_as_x = 0
    losses_as_o = 0

    for filepath in loss_files:
        with open(filepath, 'r') as f:
            loss_data = json.load(f)

        filename = os.path.basename(filepath)
        if loss_data['nn_played_as'] == 'X':
            losses_as_x += 1
        else:
            losses_as_o += 1

        print(f"  {filename}")
        print(f"    Game #{loss_data['game_number']}, NN as {loss_data['nn_played_as']}, {loss_data['total_moves']} moves")

    print("\n" + "=" * 70)
    print(f"Summary: {losses_as_x} losses as X (first), {losses_as_o} losses as O (second)")
    print("=" * 70)


def main():
    """Main function to analyze losses."""
    import sys

    loss_dir = "nn_losses"

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--summary":
            # Show summary only
            summarize_losses(loss_dir)
        elif sys.argv[1] == "--all":
            # Analyze all losses in detail
            loss_files = list_all_losses(loss_dir)
            for i, filepath in enumerate(loss_files):
                if i > 0:
                    input("\nPress Enter to view next loss...")
                analyze_loss_file(filepath, verbose=True)
        elif sys.argv[1] == "--file":
            # Analyze a specific file
            if len(sys.argv) < 3:
                print("Usage: python analyze_losses.py --file <filepath>")
                sys.exit(1)
            analyze_loss_file(sys.argv[2], verbose=True)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python analyze_losses.py                  # Show summary")
            print("  python analyze_losses.py --summary        # Show summary")
            print("  python analyze_losses.py --all            # Analyze all losses")
            print("  python analyze_losses.py --file <path>    # Analyze specific loss file")
            print("  python analyze_losses.py --help           # Show this help")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Run 'python analyze_losses.py --help' for usage information")
    else:
        # Default: show summary
        summarize_losses(loss_dir)


if __name__ == "__main__":
    main()
