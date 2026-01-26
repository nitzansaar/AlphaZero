"""
Verification script to inspect self-play data generation.
Runs a few games and validates the training data structure and content.
"""
import numpy as np
from copy import copy

from game import Go, PASS_ACTION, ACTION_SIZE, NUM_POSITIONS, BOARD_SIZE
from mcts import MonteCarloTreeSearch, Node
from dataset import TrainingDataset


def mock_policy_value_network(state, player):
    """Simple mock network for testing."""
    value = 0.0
    policy = np.ones(ACTION_SIZE) / ACTION_SIZE
    return value, policy


def render_board(state):
    """Render board state as string."""
    board = state[:NUM_POSITIONS].reshape(BOARD_SIZE, BOARD_SIZE)
    symbols = {0: '.', 1: 'X', -1: 'O'}
    lines = []
    lines.append('  ' + ' '.join(str(i) for i in range(BOARD_SIZE)))
    for i in range(BOARD_SIZE):
        row_str = ' '.join(symbols[int(board[i, j])] for j in range(BOARD_SIZE))
        lines.append(f'{i} {row_str}')
    return '\n'.join(lines)


def action_to_str(action_probs):
    """Convert action probabilities to readable string."""
    top_actions = np.argsort(action_probs)[-5:][::-1]
    parts = []
    for a in top_actions:
        if action_probs[a] > 0.01:
            if a == PASS_ACTION:
                parts.append(f"PASS:{action_probs[a]:.3f}")
            else:
                row, col = a // BOARD_SIZE, a % BOARD_SIZE
                parts.append(f"({row},{col}):{action_probs[a]:.3f}")
    return ', '.join(parts)


def verify_data_point(data_point, idx):
    """Verify a single data point has correct structure."""
    errors = []

    state, policy, player, value = data_point

    # Check state
    if len(state) != NUM_POSITIONS + 2:
        errors.append(f"State length: expected {NUM_POSITIONS + 2}, got {len(state)}")

    board = state[:NUM_POSITIONS]
    if not all(v in [-1, 0, 1] for v in board):
        errors.append(f"Board contains invalid values (not in -1, 0, 1)")

    ko_point = state[NUM_POSITIONS]
    if ko_point != -1 and not (0 <= ko_point < NUM_POSITIONS):
        errors.append(f"Invalid ko point: {ko_point}")

    passes = state[NUM_POSITIONS + 1]
    if passes not in [0, 1, 2]:
        errors.append(f"Invalid consecutive passes: {passes}")

    # Check policy
    if len(policy) != ACTION_SIZE:
        errors.append(f"Policy length: expected {ACTION_SIZE}, got {len(policy)}")

    if not np.isclose(np.sum(policy), 1.0, atol=0.01):
        errors.append(f"Policy doesn't sum to 1: {np.sum(policy):.4f}")

    if np.any(policy < 0):
        errors.append("Policy contains negative values")

    # Check player
    if player not in [1, -1]:
        errors.append(f"Invalid player: {player}")

    # Check value
    if value not in [-1, 0, 1]:
        errors.append(f"Invalid value: {value}")

    return errors


def run_verification(num_games=3, num_simulations=50, verbose=True):
    """Run self-play games and verify the data."""
    print("=" * 60)
    print("SELF-PLAY DATA VERIFICATION")
    print("=" * 60)

    game = Go()
    mcts = MonteCarloTreeSearch(game, mock_policy_value_network)
    root_node = mcts.init_root_node()
    training_dataset = TrainingDataset()

    total_moves = 0
    total_errors = 0
    game_results = []

    for game_num in range(num_games):
        print(f"\n{'='*60}")
        print(f"GAME {game_num + 1}")
        print("=" * 60)

        node = root_node
        player = 1
        game_data = []
        move_count = 0
        max_moves = 100

        if verbose:
            print(f"\nInitial board:")
            print(render_board(node.state))

        while game.win_or_draw(node.state, perspective=player) is None:
            if move_count >= max_moves:
                print(f"  [Force ended at {max_moves} moves]")
                break

            parent_state = copy(node.state)
            node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)

            # Temperature decay
            temperature = 1.0 if move_count < 10 else 0.1

            action, node, action_probs = mcts.select_move(node, mode="explore", temperature=temperature)
            action_idx = np.argmax(action)

            game_data.append([parent_state, action_probs, player])

            if verbose and move_count < 10:  # Show first 10 moves
                player_str = "Black (X)" if player == 1 else "White (O)"
                if action_idx == PASS_ACTION:
                    print(f"\nMove {move_count + 1}: {player_str} PASSES")
                else:
                    row, col = action_idx // BOARD_SIZE, action_idx % BOARD_SIZE
                    print(f"\nMove {move_count + 1}: {player_str} plays at ({row}, {col})")
                print(f"  Top actions: {action_to_str(action_probs)}")
                print(render_board(node.state))

            player = -1 * player
            move_count += 1

        total_moves += move_count

        # Determine winner
        if game.game_ended(node.state):
            winner = game.get_winner(node.state, perspective=player)
        else:
            winner = game.get_winner(node.state, perspective=player)

        winner_str = "Black" if winner == 1 else ("White" if winner == -1 else "Draw")
        game_results.append((winner, move_count))
        print(f"\nGame {game_num + 1} finished: {winner_str} wins, {move_count} moves")

        # Add to training dataset
        training_dataset.add_game_to_training_dataset(game_data, winner)

    # Verify all data points
    print("\n" + "=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)

    print(f"\nTotal data points collected: {len(training_dataset.training_dataset)}")

    # Check each data point
    for idx, data_point in enumerate(training_dataset.training_dataset):
        errors = verify_data_point(data_point, idx)
        if errors:
            print(f"\nData point {idx} has errors:")
            for e in errors:
                print(f"  - {e}")
            total_errors += len(errors)

    if total_errors == 0:
        print("\n✓ All data points passed verification!")
    else:
        print(f"\n✗ Found {total_errors} errors in data")

    # Statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    # Value distribution
    values = [d[3] for d in training_dataset.training_dataset]
    print(f"\nValue distribution:")
    print(f"  Wins (1):   {values.count(1):4d} ({100*values.count(1)/len(values):.1f}%)")
    print(f"  Draws (0):  {values.count(0):4d} ({100*values.count(0)/len(values):.1f}%)")
    print(f"  Losses (-1):{values.count(-1):4d} ({100*values.count(-1)/len(values):.1f}%)")

    # Player distribution
    players = [d[2] for d in training_dataset.training_dataset]
    print(f"\nPlayer distribution:")
    print(f"  Black (1):  {players.count(1):4d} ({100*players.count(1)/len(players):.1f}%)")
    print(f"  White (-1): {players.count(-1):4d} ({100*players.count(-1)/len(players):.1f}%)")

    # Game results
    print(f"\nGame results:")
    for i, (winner, moves) in enumerate(game_results):
        winner_str = "Black" if winner == 1 else ("White" if winner == -1 else "Draw")
        print(f"  Game {i+1}: {winner_str} wins, {moves} moves")

    # Policy statistics
    print(f"\nPolicy statistics:")
    policies = [d[1] for d in training_dataset.training_dataset]
    avg_entropy = np.mean([-np.sum(p * np.log(p + 1e-10)) for p in policies])
    avg_max_prob = np.mean([np.max(p) for p in policies])
    print(f"  Average entropy: {avg_entropy:.3f}")
    print(f"  Average max probability: {avg_max_prob:.3f}")

    # Sample data points
    print("\n" + "=" * 60)
    print("SAMPLE DATA POINTS")
    print("=" * 60)

    for i in [0, len(training_dataset.training_dataset)//2, -1]:
        idx = i if i >= 0 else len(training_dataset.training_dataset) + i
        data_point = training_dataset.training_dataset[idx]
        state, policy, player, value = data_point

        print(f"\n--- Data point {idx} ---")
        print(f"Player: {'Black (X)' if player == 1 else 'White (O)'}")
        print(f"Value: {value}")
        print(f"Top actions: {action_to_str(policy)}")
        print(f"Board state:")
        print(render_board(state))
        print(f"Ko point: {int(state[NUM_POSITIONS])}")
        print(f"Consecutive passes: {int(state[NUM_POSITIONS + 1])}")

    return total_errors == 0


if __name__ == '__main__':
    success = run_verification(num_games=3, num_simulations=50, verbose=True)
    print("\n" + "=" * 60)
    if success:
        print("VERIFICATION PASSED - Data is being created properly!")
    else:
        print("VERIFICATION FAILED - Check errors above")
    print("=" * 60)
