"""
Inspect the actual training data to verify labels are correct.
"""

import os
import pickle
import numpy as np
from config import Config as cfg
from game import Go, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION

def print_board(state_flat):
    board = state_flat[:NUM_POSITIONS].reshape(BOARD_SIZE, BOARD_SIZE)
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("  " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for i in range(BOARD_SIZE):
        row = " ".join(symbols[int(board[i, j])] for j in range(BOARD_SIZE))
        print(f"{i} {row}")

def action_to_str(policy):
    idx = np.argmax(policy)
    if idx == PASS_ACTION:
        return "pass"
    return f"({idx // BOARD_SIZE}, {idx % BOARD_SIZE})"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, cfg.SAVE_PICKLES, cfg.DATASET_PATH)

    if not os.path.exists(save_path):
        print(f"Dataset not found at {save_path}")
        return

    with open(save_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Total samples: {len(dataset)}")
    print(f"Sample format: [state, policy, player, value]")

    # Analyze value distribution
    values = [d[3] for d in dataset]
    players = [d[2] for d in dataset]

    print(f"\nValue distribution:")
    print(f"  +1 (wins): {values.count(1)} ({values.count(1)/len(values)*100:.1f}%)")
    print(f"   0 (draws): {values.count(0)} ({values.count(0)/len(values)*100:.1f}%)")
    print(f"  -1 (losses): {values.count(-1)} ({values.count(-1)/len(values)*100:.1f}%)")

    print(f"\nPlayer distribution:")
    print(f"  Black (1): {players.count(1)} ({players.count(1)/len(players)*100:.1f}%)")
    print(f"  White (-1): {players.count(-1)} ({players.count(-1)/len(players)*100:.1f}%)")

    # Find some winning and losing examples
    print("\n" + "=" * 60)
    print("SAMPLE WINNING POSITIONS (value=+1)")
    print("=" * 60)

    wins = [(i, d) for i, d in enumerate(dataset) if d[3] == 1]
    for i, (idx, sample) in enumerate(wins[:3]):
        state, policy, player, value = sample
        player_str = "Black" if player == 1 else "White"
        print(f"\nSample {idx} - {player_str} to move, value={value}")
        print(f"This means {player_str} eventually WON this game")
        print_board(state)
        print(f"Top move: {action_to_str(policy)}")

        # Count stones
        board = state[:NUM_POSITIONS]
        black_stones = np.sum(board == 1)
        white_stones = np.sum(board == -1)
        print(f"Black stones: {black_stones}, White stones: {white_stones}")

    print("\n" + "=" * 60)
    print("SAMPLE LOSING POSITIONS (value=-1)")
    print("=" * 60)

    losses = [(i, d) for i, d in enumerate(dataset) if d[3] == -1]
    for i, (idx, sample) in enumerate(losses[:3]):
        state, policy, player, value = sample
        player_str = "Black" if player == 1 else "White"
        print(f"\nSample {idx} - {player_str} to move, value={value}")
        print(f"This means {player_str} eventually LOST this game")
        print_board(state)
        print(f"Top move: {action_to_str(policy)}")

        board = state[:NUM_POSITIONS]
        black_stones = np.sum(board == 1)
        white_stones = np.sum(board == -1)
        print(f"Black stones: {black_stones}, White stones: {white_stones}")

    # Check for potential issues
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # Check if states are in absolute form (Black should be +1)
    # Sample some states
    issues = []

    for i in range(min(1000, len(dataset))):
        state, policy, player, value = dataset[i]
        board = state[:NUM_POSITIONS]

        # In absolute form, we expect to see both +1 and -1 values
        unique_vals = set(board) - {0}

        if player == 1:  # Black to move
            # State should have Black=+1 (absolute form)
            pass
        else:  # White to move (player=-1)
            # State should still have Black=+1 (absolute form)
            # The canonical conversion happens in GoDataset
            pass

    # Look for late-game positions to verify value correctness
    print("\nLooking for late-game positions (many stones)...")

    late_game = []
    for i, sample in enumerate(dataset):
        state, policy, player, value = sample
        board = state[:NUM_POSITIONS]
        total_stones = np.sum(board != 0)
        if total_stones >= 15:
            late_game.append((i, sample, total_stones))

    late_game.sort(key=lambda x: x[2], reverse=True)

    print(f"Found {len(late_game)} positions with 15+ stones")

    for i, (idx, sample, num_stones) in enumerate(late_game[:5]):
        state, policy, player, value = sample
        player_str = "Black" if player == 1 else "White"
        value_str = "WON" if value == 1 else ("LOST" if value == -1 else "DRAW")

        print(f"\nSample {idx} - {num_stones} stones, {player_str} to move, {player_str} {value_str}")
        print_board(state)

        board = state[:NUM_POSITIONS]
        black_stones = np.sum(board == 1)
        white_stones = np.sum(board == -1)
        print(f"Black: {black_stones}, White: {white_stones}")

        # Check if the value makes sense
        # In late game, the player with more territory/stones usually wins
        game = Go()
        black_score, white_score = game.count_territory(board)
        print(f"Territory - Black: {black_score}, White: {white_score + 2.5} (with komi)")

        expected_winner = 1 if black_score > white_score + 2.5 else -1
        if player == expected_winner and value == 1:
            print("[OK] Value matches expected winner")
        elif player != expected_winner and value == -1:
            print("[OK] Value matches expected winner")
        else:
            print(f"[CHECK] Player={player}, Value={value}, Expected winner based on territory={expected_winner}")

if __name__ == "__main__":
    main()
