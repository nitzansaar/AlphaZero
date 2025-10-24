"""
Evaluate the trained neural network player against a random player.

This script simulates 1000 games between the MCTS-trained neural network
and a random player to measure the performance of the trained agent.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trainer.train_self_play import TicTacToeNet, MCTS, check_winner
from players.random_player import RandomPlayer


class NeuralNetPlayer:
    """Neural network player using MCTS for move selection."""

    def __init__(self, model: torch.nn.Module, device: str, num_simulations: int = 50):
        """
        Initialize the neural network player.

        Args:
            model: Trained TicTacToeNet model
            device: 'cpu' or 'cuda'
            num_simulations: Number of MCTS simulations per move
        """
        self.model = model
        self.device = device
        self.mcts = MCTS(model, device, num_simulations=num_simulations, debug=False)
        self.name = f"Neural Net Player (MCTS {num_simulations} sims)"

    def get_move(self, board: List[int], player: int) -> int:
        """
        Select the best move using MCTS.

        Args:
            board: Flat list of 9 elements representing the board state
            player: Current player (1 or -1)

        Returns:
            Index of the selected move (0-8)
        """
        # Use temperature=0 for deterministic, greedy move selection
        action_probs, _ = self.mcts.get_action_probs(
            board, player, temperature=0.0, add_noise=False
        )

        # Select move with highest probability
        move = int(np.argmax(action_probs))
        return move

    def __str__(self) -> str:
        return self.name


def play_game(player1, player2, verbose: bool = False, track_history: bool = False) -> Tuple[int, List[dict]]:
    """
    Play a single game between two players.

    Args:
        player1: First player (plays X, moves first)
        player2: Second player (plays O)
        verbose: Whether to print game progress
        track_history: Whether to track the full game history

    Returns:
        Tuple of (game_result, game_history)
        - game_result: 1 if player1 wins, -1 if player2 wins, 0 for draw
        - game_history: List of move records (empty if track_history=False)
    """
    board = [0] * 9
    current_player = 1
    move_count = 0
    game_history = [] if track_history else None

    if verbose:
        print(f"\n{'='*50}")
        print(f"Game: {player1} (X) vs {player2} (O)")
        print(f"{'='*50}")

    while True:
        # Check if game is over
        winner = check_winner(board)
        if winner is not None:
            if verbose:
                print_board(board)
                result_str = "Player 1 (X) wins" if winner == 1 else "Player 2 (O) wins" if winner == -1 else "Draw"
                print(f"\nResult: {result_str}")
            return winner, game_history if track_history else []

        # Get move from current player
        if current_player == 1:
            if hasattr(player1, 'get_move'):
                if 'player' in player1.get_move.__code__.co_varnames:
                    move = player1.get_move(board, current_player)
                else:
                    move = player1.get_move(board)
            else:
                raise ValueError("Player 1 does not have get_move method")
        else:
            if hasattr(player2, 'get_move'):
                if 'player' in player2.get_move.__code__.co_varnames:
                    move = player2.get_move(board, current_player)
                else:
                    move = player2.get_move(board)
            else:
                raise ValueError("Player 2 does not have get_move method")

        # Track move in history if requested
        if track_history:
            game_history.append({
                'move_number': move_count + 1,
                'player': current_player,
                'player_symbol': 'X' if current_player == 1 else 'O',
                'board_before': board.copy(),
                'move_position': move,
                'move_by': 'player1' if current_player == 1 else 'player2'
            })

        # Apply move
        board[move] = current_player

        if verbose:
            print(f"\nMove {move_count + 1}: Player {current_player} ({'X' if current_player == 1 else 'O'}) -> position {move}")
            print_board(board)

        # Switch player
        current_player = -current_player
        move_count += 1


def print_board(board: List[int]) -> None:
    """Print the board in a readable format."""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("\n  0 1 2")
    print("  -----")
    for i in range(3):
        row = board[i*3:(i+1)*3]
        print(f"{i}|" + ' '.join(symbols[cell] for cell in row))
    print()


def save_loss_history(game_history: List[dict], game_number: int, nn_player_symbol: str,
                      loss_dir: str = "nn_losses") -> None:
    """
    Save the game history when NN loses to a file.

    Args:
        game_history: List of move records from the game
        game_number: Game number in the evaluation sequence
        nn_player_symbol: 'X' or 'O' depending on which player was NN
        loss_dir: Directory to save loss files
    """
    # Create directory if it doesn't exist
    os.makedirs(loss_dir, exist_ok=True)

    # Generate filename with timestamp and game number
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"loss_game{game_number}_{nn_player_symbol}_{timestamp}.json"
    filepath = os.path.join(loss_dir, filename)

    # Prepare data to save
    loss_data = {
        'game_number': game_number,
        'nn_played_as': nn_player_symbol,
        'timestamp': timestamp,
        'total_moves': len(game_history),
        'moves': game_history
    }

    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(loss_data, f, indent=2)

    print(f"   Loss saved to: {filepath}")


def simulate_games(nn_player, random_player, num_games: int = 1000,
                   nn_plays_first: bool = True, verbose_games: int = 0,
                   save_losses: bool = True) -> dict:
    """
    Simulate multiple games between neural net player and random player.

    Args:
        nn_player: Neural network player
        random_player: Random player
        num_games: Number of games to simulate
        nn_plays_first: If True, NN plays as X (first), otherwise as O (second)
        verbose_games: Number of initial games to show in detail
        save_losses: If True, save game history when NN loses

    Returns:
        Dictionary with game statistics
    """
    results = {
        'nn_wins': 0,
        'random_wins': 0,
        'draws': 0,
        'game_results': [],  # Store individual game results for plotting
        'losses_saved': 0
    }

    player1 = nn_player if nn_plays_first else random_player
    player2 = random_player if nn_plays_first else nn_player
    nn_symbol = 'X' if nn_plays_first else 'O'

    print(f"\nSimulating {num_games} games...")
    print(f"Neural Net plays as: {'X (first)' if nn_plays_first else 'O (second)'}")
    print(f"Random Player plays as: {'O (second)' if nn_plays_first else 'X (first)'}")
    print(f"Save losses: {'Yes' if save_losses else 'No'}")
    print("-" * 60)

    for i in range(num_games):
        verbose = i < verbose_games
        # Track history only if we need to save losses
        result, game_history = play_game(player1, player2, verbose=verbose,
                                         track_history=save_losses)

        # Store result from neural net's perspective
        if nn_plays_first:
            nn_result = result
        else:
            nn_result = -result

        results['game_results'].append(nn_result)

        # Update statistics
        if nn_result == 1:
            results['nn_wins'] += 1
        elif nn_result == -1:
            results['random_wins'] += 1
            # Save the game history if NN lost
            if save_losses and game_history:
                save_loss_history(game_history, i + 1, nn_symbol)
                results['losses_saved'] += 1
        else:
            results['draws'] += 1

        # Print progress
        if (i + 1) % 100 == 0:
            win_rate = 100.0 * results['nn_wins'] / (i + 1)
            loss_info = f", Losses saved: {results['losses_saved']}" if save_losses else ""
            print(f"Games {i+1}/{num_games} - NN wins: {results['nn_wins']} ({win_rate:.1f}%), "
                  f"Random wins: {results['random_wins']}, Draws: {results['draws']}{loss_info}")

    return results


def plot_results(results_first: dict, results_second: dict, save_path: str = "nn_vs_random_results.png"):
    """
    Plot the results of games against random player.

    Args:
        results_first: Results when NN plays first (as X)
        results_second: Results when Random plays first (as X)
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Neural Network vs Random Player Evaluation', fontsize=16, fontweight='bold')

    # Plot 1: Results when NN plays first (as X)
    ax1 = axes[0]
    categories = ['NN Wins', 'Random Wins', 'Draws']
    values_first = [results_first['nn_wins'], results_first['random_wins'], results_first['draws']]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    bars1 = ax1.bar(categories, values_first, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Games', fontsize=12)
    ax1.set_title('NN Playing First (as X)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, max(values_first) * 1.2])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add percentage labels below category names
    total_games_first = sum(values_first)
    percentages_first = [f'({100*v/total_games_first:.1f}%)' for v in values_first]
    for i, (cat, pct) in enumerate(zip(categories, percentages_first)):
        ax1.text(i, -max(values_first) * 0.08, pct, ha='center', fontsize=10, style='italic')

    # Plot 2: Results when Random plays first (as X)
    ax2 = axes[1]
    values_second = [results_second['nn_wins'], results_second['random_wins'], results_second['draws']]
    bars2 = ax2.bar(categories, values_second, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Number of Games', fontsize=12)
    ax2.set_title('Random Playing First (as X)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(values_second) * 1.2])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add percentage labels below category names
    total_games_second = sum(values_second)
    percentages_second = [f'({100*v/total_games_second:.1f}%)' for v in values_second]
    for i, (cat, pct) in enumerate(zip(categories, percentages_second)):
        ax2.text(i, -max(values_second) * 0.08, pct, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nResults plot saved: {save_path}")
    plt.close()


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("NEURAL NETWORK vs RANDOM PLAYER EVALUATION")
    print("=" * 60)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load trained model
    print("\n[1/4] Loading trained neural network model...")
    model = TicTacToeNet()

    # Try to load the final model from models/, or fall back to latest checkpoint
    model_path = os.path.join("models", "tictactoe_selfplay_final.pth")
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found, looking for checkpoint in models/...")
        checkpoints = []
        if os.path.isdir("models"):
            checkpoints = [f for f in os.listdir('models') if f.startswith('tictactoe_alphazero_ep') and f.endswith('.pth')]
        if checkpoints:
            # Sort by episode number and select the newest
            checkpoints.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]), reverse=True)
            model_path = os.path.join('models', checkpoints[0])
            print(f"Found checkpoint: {model_path}")
        else:
            # Backward compatibility: look in repo root
            root_final = "tictactoe_selfplay_final.pth"
            if os.path.exists(root_final):
                model_path = root_final
            else:
                root_checkpoints = [f for f in os.listdir('.') if f.startswith('tictactoe_alphazero_ep') and f.endswith('.pth')]
                if root_checkpoints:
                    root_checkpoints.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]), reverse=True)
                    model_path = root_checkpoints[0]
                    print(f"Found checkpoint: {model_path}")
                else:
                    raise FileNotFoundError("No trained model found. Please run train_self_play.py first.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"   Model loaded from: {model_path}")

    # Create players
    print("\n[2/4] Creating players...")
    nn_player = NeuralNetPlayer(model, device, num_simulations=50)
    random_player = RandomPlayer()
    print(f"   Neural Net Player: {nn_player}")
    print(f"   Random Player: {random_player}")

    # Simulate games with NN playing first
    print("\n[3/4] Running simulations...")
    print("\n--- Scenario 1: Neural Net plays FIRST (as X) ---")
    results_first = simulate_games(nn_player, random_player, num_games=1000,
                                   nn_plays_first=True, verbose_games=0, save_losses=True)

    # Simulate games with NN playing second
    print("\n--- Scenario 2: Neural Net plays SECOND (as O) ---")
    results_second = simulate_games(nn_player, random_player, num_games=1000,
                                    nn_plays_first=False, verbose_games=0, save_losses=True)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    total_games_first = len(results_first['game_results'])
    print(f"\nScenario 1: NN playing FIRST (as X) - {total_games_first} games")
    print(f"  NN Wins:     {results_first['nn_wins']:4d} ({100.0 * results_first['nn_wins'] / total_games_first:.1f}%)")
    print(f"  Random Wins: {results_first['random_wins']:4d} ({100.0 * results_first['random_wins'] / total_games_first:.1f}%)")
    print(f"  Draws:       {results_first['draws']:4d} ({100.0 * results_first['draws'] / total_games_first:.1f}%)")

    total_games_second = len(results_second['game_results'])
    print(f"\nScenario 2: NN playing SECOND (as O) - {total_games_second} games")
    print(f"  NN Wins:     {results_second['nn_wins']:4d} ({100.0 * results_second['nn_wins'] / total_games_second:.1f}%)")
    print(f"  Random Wins: {results_second['random_wins']:4d} ({100.0 * results_second['random_wins'] / total_games_second:.1f}%)")
    print(f"  Draws:       {results_second['draws']:4d} ({100.0 * results_second['draws'] / total_games_second:.1f}%)")

    overall_nn_wins = results_first['nn_wins'] + results_second['nn_wins']
    overall_random_wins = results_first['random_wins'] + results_second['random_wins']
    overall_draws = results_first['draws'] + results_second['draws']
    overall_losses_saved = results_first['losses_saved'] + results_second['losses_saved']
    total_games = total_games_first + total_games_second

    print(f"\nOverall Results - {total_games} total games")
    print(f"  NN Wins:     {overall_nn_wins:4d} ({100.0 * overall_nn_wins / total_games:.1f}%)")
    print(f"  Random Wins: {overall_random_wins:4d} ({100.0 * overall_random_wins / total_games:.1f}%)")
    print(f"  Draws:       {overall_draws:4d} ({100.0 * overall_draws / total_games:.1f}%)")
    print(f"\nLosses saved to files: {overall_losses_saved} (in 'nn_losses/' directory)")

    # Plot results
    print("\n[4/4] Generating visualization...")
    plot_results(results_first, results_second)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
