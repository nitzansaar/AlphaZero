"""
Test script to verify self-play data generation.
Runs 3 games and prints the generated dataset for inspection.
"""
import os
import numpy as np
import torch
from config import Config as cfg
from game import Go, BOARD_SIZE, PASS_ACTION, NUM_POSITIONS, board_to_canonical_3d
from mcts import MonteCarloTreeSearch, Node
from dataset import TrainingDataset, GoDataset
from value_policy_function import ValuePolicyNetwork
from copy import copy
from glob import glob

# Use fewer simulations for faster testing
NUM_TEST_SIMULATIONS = 100
NUM_TEST_GAMES = 3
MAX_MOVES_PER_GAME = 50

def print_board(state, title="Board"):
    """Print a board state in a readable format."""
    board = state[:25].reshape(BOARD_SIZE, BOARD_SIZE)
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print(f"\n{title}:")
    print("    " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for i in range(BOARD_SIZE):
        row = " ".join(symbols[int(board[i, j])] for j in range(BOARD_SIZE))
        print(f"  {i} {row}")

def print_canonical_planes(canonical_3d, title="Canonical 3-Plane Input"):
    """Print the 3-plane canonical representation that goes into the neural network."""
    print(f"\n{title}:")
    plane_names = ["Plane 0 (Current Player)", "Plane 1 (Opponent)", "Plane 2 (Empty)"]
    for plane_idx, plane_name in enumerate(plane_names):
        print(f"  {plane_name}:")
        print("      " + " ".join(str(i) for i in range(BOARD_SIZE)))
        for i in range(BOARD_SIZE):
            row = " ".join(f"{int(canonical_3d[plane_idx, i, j])}" for j in range(BOARD_SIZE))
            print(f"    {i} {row}")

def print_policy(policy, title="Policy"):
    """Print policy distribution as a board."""
    board_policy = policy[:25].reshape(BOARD_SIZE, BOARD_SIZE)
    print(f"\n{title}:")
    print("    " + "  ".join(str(i) for i in range(BOARD_SIZE)))
    for i in range(BOARD_SIZE):
        row = " ".join(f"{board_policy[i, j]:.2f}" for j in range(BOARD_SIZE))
        print(f"  {i} {row}")
    print(f"  Pass: {policy[PASS_ACTION]:.2f}")

def action_to_str(action_probs):
    """Convert action probs to best move string."""
    best_action = np.argmax(action_probs)
    if best_action == PASS_ACTION:
        return "PASS"
    row, col = best_action // BOARD_SIZE, best_action % BOARD_SIZE
    return f"({row},{col})"

def main():
    print("=" * 70)
    print("SELF-PLAY DATA GENERATION TEST")
    print("=" * 70)

    game = Go()

    # Load the latest trained model
    all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*.pt"))
    if all_models:
        files = [int(os.path.basename(f).split("_")[0]) for f in all_models
                 if os.path.basename(f).split("_")[0].lstrip('-').isdigit()]
        if files:
            latest_num = max(files)
            model_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(latest_num))
            print(f"Loading model: {model_path}")
            vpn = ValuePolicyNetwork(path=model_path, use_compile=False)
        else:
            print("Using randomly initialized network.")
            vpn = ValuePolicyNetwork(path=None, use_compile=False)
    else:
        print("Using randomly initialized network.")
        vpn = ValuePolicyNetwork(path=None, use_compile=False)

    policy_value_network = vpn.get_vp
    mcts = MonteCarloTreeSearch(game, policy_value_network)

    # Storage for all games' data
    all_games_data = []

    print(f"\nRunning {NUM_TEST_GAMES} self-play games with {NUM_TEST_SIMULATIONS} MCTS simulations each...")
    print("-" * 70)

    for game_num in range(NUM_TEST_GAMES):
        print(f"\n{'='*70}")
        print(f"GAME {game_num + 1}")
        print("=" * 70)

        # Initialize root node for new game
        root_node = Node(prior_prob=0, player=1, action_index=None)
        root_node.set_state(game.state.copy())

        node = root_node
        dataset = []
        player = 1
        move_count = 0

        while game.winner(node.state, perspective=player) is None:
            if move_count >= MAX_MOVES_PER_GAME:
                print(f"\n[Game force-ended at {move_count} moves]")
                break

            # Convert from relative form to absolute form for storing in dataset
            # Node stores state in relative form (current player's perspective)
            # Dataset should store absolute form (Black=+1, White=-1 always)
            parent_state = copy(node.state)
            parent_state[:NUM_POSITIONS] *= player

            # Run MCTS
            node = mcts.run_simulation(root_node=node, num_simulations=NUM_TEST_SIMULATIONS, player=player)

            # Temperature for move selection
            temperature = 1.0 if move_count < 6 else 0.1

            # Select move
            action, node, action_probs = mcts.select_move(node=node, mode="explore", temperature=temperature)

            # Store data point
            dataset.append([parent_state, action_probs, player])

            # Print move info
            best_move = action_to_str(action_probs)
            player_name = "Black(X)" if player == 1 else "White(O)"
            print(f"\nMove {move_count + 1}: {player_name} plays {best_move}")

            player = -1 * player
            move_count += 1

        # Determine winner
        winner = game.get_winner(node.state, perspective=player)
        winner_name = "Black(X)" if winner == 1 else "White(O)" if winner == -1 else "Draw"
        print(f"\nGame {game_num + 1} Result: {winner_name} wins!" if winner != 0 else f"\nGame {game_num + 1} Result: Draw!")

        # Print final board
        print_board(node.state, "Final Board")

        # Calculate values for this game's dataset
        training_dataset = TrainingDataset()
        training_dataset.add_game_to_training_dataset(dataset, winner)

        all_games_data.append({
            'game_num': game_num + 1,
            'winner': winner,
            'num_moves': len(dataset),
            'data': training_dataset.training_dataset.copy()
        })

    # Print detailed dataset information
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    for game_data in all_games_data:
        print(f"\n{'='*70}")
        print(f"GAME {game_data['game_num']} DATA ({game_data['num_moves']} positions)")
        print(f"Winner: {'Black(X)' if game_data['winner'] == 1 else 'White(O)' if game_data['winner'] == -1 else 'Draw'}")
        print("=" * 70)

        # Create GoDataset to show processed data
        go_dataset = GoDataset(game_data['data'], use_augmentation=False)

        for i, datapoint in enumerate(game_data['data']):
            state = datapoint[0]
            policy = datapoint[1]
            player = datapoint[2]
            value = datapoint[3]

            player_name = "Black(X)" if player == 1 else "White(O)"

            print(f"\n{'─'*70}")
            print(f"POSITION {i + 1}")
            print(f"{'─'*70}")

            print(f"\n[RAW DATA FROM SELF-PLAY]")
            print(f"Player to move: {player_name} (player={player})")
            print(f"Value target: {value} ({'win' if value == 1 else 'loss' if value == -1 else 'draw'})")

            print_board(state, "Raw Board State (state[:25])")

            # Show ko and pass info
            ko_point = int(state[25])
            consecutive_passes = int(state[26])
            print(f"\nKo point (state[25]): {ko_point}")
            print(f"Consecutive passes (state[26]): {consecutive_passes}")

            # Show top 5 moves from policy
            top_moves = np.argsort(policy)[::-1][:5]
            print("\nPolicy target (MCTS visit distribution):")
            print(f"  Sum: {policy.sum():.4f}")
            print("  Top 5 moves:")
            for rank, move in enumerate(top_moves, 1):
                move_str = "PASS" if move == PASS_ACTION else f"({move // BOARD_SIZE},{move % BOARD_SIZE})"
                print(f"    {rank}. {move_str}: {policy[move]:.4f}")

            # Now show what actually goes into the neural network
            print(f"\n[PROCESSED DATA FOR NEURAL NETWORK]")

            # Get processed data from GoDataset
            state_tensor, value_tensor, policy_tensor = go_dataset[i]

            print(f"\nState tensor shape: {state_tensor.shape} (channels, height, width)")
            print(f"Value tensor: {value_tensor.item():.4f}")
            print(f"Policy tensor shape: {policy_tensor.shape}, sum: {policy_tensor.sum().item():.4f}")

            # Show the canonical 3-plane representation
            canonical_3d = state_tensor.numpy()
            print_canonical_planes(canonical_3d, "Canonical 3-Plane Input (what NN sees)")

            # Verify the planes are correct
            print("\n  Plane verification:")
            print(f"    Plane 0 sum (current player stones): {canonical_3d[0].sum():.0f}")
            print(f"    Plane 1 sum (opponent stones): {canonical_3d[1].sum():.0f}")
            print(f"    Plane 2 sum (empty squares): {canonical_3d[2].sum():.0f}")
            print(f"    Total (should be 25): {canonical_3d.sum():.0f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_positions = sum(g['num_moves'] for g in all_games_data)
    print(f"Total games: {NUM_TEST_GAMES}")
    print(f"Total positions generated: {total_positions}")
    print(f"Average positions per game: {total_positions / NUM_TEST_GAMES:.1f}")

    # Check data integrity
    print("\nData Integrity Checks:")
    all_ok = True

    for game_data in all_games_data:
        for i, dp in enumerate(game_data['data']):
            state, policy, player, value = dp

            # Check state shape
            if len(state) != 27:
                print(f"  ERROR: Game {game_data['game_num']}, pos {i}: state length {len(state)} != 27")
                all_ok = False

            # Check policy shape and sum
            if len(policy) != 26:
                print(f"  ERROR: Game {game_data['game_num']}, pos {i}: policy length {len(policy)} != 26")
                all_ok = False
            if not np.isclose(policy.sum(), 1.0, atol=0.01):
                print(f"  ERROR: Game {game_data['game_num']}, pos {i}: policy sum {policy.sum():.4f} != 1.0")
                all_ok = False

            # Check player
            if player not in [1, -1]:
                print(f"  ERROR: Game {game_data['game_num']}, pos {i}: invalid player {player}")
                all_ok = False

            # Check value
            if value not in [1, -1, 0]:
                print(f"  ERROR: Game {game_data['game_num']}, pos {i}: invalid value {value}")
                all_ok = False

    if all_ok:
        print("  All checks passed!")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
