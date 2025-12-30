#!/usr/bin/env python3
"""
Play MCTS-trained model against a random opponent.
Tests model performance against baseline random play.

Usage examples:
    # Run 100 games with default settings
    python3 play_v_random.py --games 100

    # Test with specific model and more simulations
    python3 play_v_random.py --model output_3x3/models/5_best_model.pt --simulations 1000 --games 50

    # Watch individual games in detail
    python3 play_v_random.py --games 5 --verbose

    # Quick test with fewer simulations
    python3 play_v_random.py --games 20 --simulations 200
"""
import numpy as np
import argparse
from game import TicTacToe
from mcts import MonteCarloTreeSearch
from value_policy_function import ValuePolicyNetwork
from config import Config as cfg

def random_move(game, state):
    """Select a random valid move."""
    valid_moves = game.get_valid_moves(state)
    valid_indices = np.where(valid_moves == 1)[0]
    action_index = np.random.choice(valid_indices)
    action = np.zeros(cfg.ACTION_SIZE)
    action[action_index] = 1
    return action

def mcts_move(mcts, node, state, player, num_simulations):
    """Select a move using MCTS."""
    node = mcts.run_simulation(node, num_simulations=num_simulations, player=player)
    action, subtree, _ = mcts.select_move(node, mode="exploit")
    return action, subtree

def print_board(state):
    """Pretty print the board state."""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    board = state.reshape(3, 3)
    print("\n  0 1 2")
    for i, row in enumerate(board):
        print(f"{i} " + " ".join(symbols[int(val)] for val in row))
    print()

def play_single_game(game, mcts, model_first=True, num_simulations=500, verbose=False):
    """
    Play a single game between MCTS model and random opponent.

    Args:
        game: TicTacToe game instance
        mcts: MonteCarloTreeSearch instance
        model_first: If True, model plays first (X), otherwise second (O)
        num_simulations: Number of MCTS simulations per move
        verbose: Print game progress

    Returns:
        1 if model wins, -1 if random wins, 0 if draw
    """
    state = np.zeros(cfg.ACTION_SIZE)
    current_player = 1  # X always starts
    node = mcts.init_root_node()

    if verbose:
        print("\n" + "=" * 40)
        print(f"New Game: Model plays {'X (first)' if model_first else 'O (second)'}")
        print("=" * 40)
        print_board(state)

    move_count = 0
    while True:
        move_count += 1

        # Determine who's playing this turn
        is_model_turn = (model_first and current_player == 1) or (not model_first and current_player == -1)

        if is_model_turn:
            # MCTS model's turn
            if verbose:
                print(f"Move {move_count}: Model's turn ({'X' if current_player == 1 else 'O'})")
            action, node = mcts_move(mcts, node, state, current_player, num_simulations)
        else:
            # Random player's turn
            if verbose:
                print(f"Move {move_count}: Random's turn ({'X' if current_player == 1 else 'O'})")
            action = random_move(game, state)

        # Apply the move
        action_index = np.argmax(action)
        state, winner, current_player = game.play(state, current_player, action_index)

        # Update/reset MCTS tree with new state
        if not is_model_turn:
            # After random move, reset tree with updated state
            node = mcts.init_root_node()
            node.set_state(state.copy())
        else:
            # After model move, update the node state
            node.set_state(state.copy())

        if verbose:
            print_board(state)

        # Check if game is over
        if winner is not None:
            if verbose:
                if winner == 0:
                    print("Game Result: DRAW")
                else:
                    winner_name = "Model" if ((winner == 1 and model_first) or (winner == -1 and not model_first)) else "Random"
                    print(f"Game Result: {winner_name} WINS!")

            # Return result from model's perspective
            if winner == 0:
                return 0  # Draw
            elif (winner == 1 and model_first) or (winner == -1 and not model_first):
                return 1  # Model wins
            else:
                return -1  # Random wins

def main():
    parser = argparse.ArgumentParser(description='Play MCTS model vs random opponent')
    parser.add_argument('--model', type=str, default='output_3x3/models/10_best_model.pt',
                        help='Path to model file (default: output_3x3/models/10_best_model.pt)')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games to play (default: 100)')
    parser.add_argument('--simulations', type=int, default=500,
                        help='MCTS simulations per move (default: 500)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print each game')
    args = parser.parse_args()

    print("=" * 60)
    print("MCTS Model vs Random Opponent")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Games: {args.games}")
    print(f"MCTS Simulations: {args.simulations}")
    print("=" * 60)

    # Initialize game and model
    game = TicTacToe()
    policy_value_network = ValuePolicyNetwork(path=args.model, use_compile=False)
    mcts = MonteCarloTreeSearch(game, policy_value_network.get_vp)

    # Track statistics
    model_wins = 0
    random_wins = 0
    draws = 0

    model_wins_as_first = 0
    model_wins_as_second = 0
    random_wins_vs_first = 0
    random_wins_vs_second = 0
    draws_model_first = 0
    draws_model_second = 0

    # Play games
    for i in range(args.games):
        model_first = (i % 2 == 0)  # Alternate who goes first

        if not args.verbose and (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{args.games} games completed...")

        result = play_single_game(game, mcts, model_first, args.simulations, args.verbose)

        if result == 1:
            model_wins += 1
            if model_first:
                model_wins_as_first += 1
            else:
                model_wins_as_second += 1
        elif result == -1:
            random_wins += 1
            if model_first:
                random_wins_vs_first += 1
            else:
                random_wins_vs_second += 1
        else:
            draws += 1
            if model_first:
                draws_model_first += 1
            else:
                draws_model_second += 1

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Games: {args.games}")
    print()
    print(f"Model Wins:  {model_wins:3d} ({100 * model_wins / args.games:5.1f}%)")
    print(f"Random Wins: {random_wins:3d} ({100 * random_wins / args.games:5.1f}%)")
    print(f"Draws:       {draws:3d} ({100 * draws / args.games:5.1f}%)")
    print()
    print("Breakdown by Position:")
    print(f"  Model as X (first):  {model_wins_as_first:3d} wins, {random_wins_vs_first:3d} losses, {draws_model_first:3d} draws")
    print(f"  Model as O (second): {model_wins_as_second:3d} wins, {random_wins_vs_second:3d} losses, {draws_model_second:3d} draws")
    print("=" * 60)

    # Performance assessment
    if model_wins > random_wins * 2:
        print("\n✓ Excellent! Model strongly dominates random play.")
    elif model_wins > random_wins:
        print("\n✓ Good! Model performs better than random.")
    elif model_wins == random_wins:
        print("\n⚠ Model performs similarly to random play.")
    else:
        print("\n✗ Warning: Model underperforms random play!")

if __name__ == "__main__":
    main()
