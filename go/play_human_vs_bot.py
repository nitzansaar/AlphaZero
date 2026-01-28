import os
import numpy as np
from glob import glob
import torch
from config import Config as cfg
from game import Go, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"


def format_board_state(state):
    """Convert board state to a readable 2D representation."""
    board = state[:NUM_POSITIONS]
    board_2d = board.reshape(BOARD_SIZE, BOARD_SIZE)
    formatted = []
    for row in board_2d:
        formatted_row = []
        for cell in row:
            if cell == 1:
                formatted_row.append('X')
            elif cell == -1:
                formatted_row.append('O')
            else:
                formatted_row.append('.')
        formatted.append(formatted_row)
    return formatted


def display_board(state, game):
    """Display the board in traditional Go intersection style."""
    board_2d = format_board_state(state)

    # Map cell values to intersection symbols (Black=empty circle, White=filled circle)
    symbols = {'.': '+', 'X': '○', 'O': '●'}

    # Column headers
    print("\n     ", end="")
    for col in range(BOARD_SIZE):
        print(f"{col}   ", end="")
    print("\n")

    for row_idx, row in enumerate(board_2d):
        # Print the row with intersections connected by horizontal lines
        print(f" {row_idx}   ", end="")
        for col_idx, cell in enumerate(row):
            symbol = symbols[cell]
            if col_idx < BOARD_SIZE - 1:
                print(f"{symbol}───", end="")
            else:
                print(f"{symbol}", end="")
        print(f"   {row_idx}")

        # Print vertical connectors (except after last row)
        if row_idx < BOARD_SIZE - 1:
            print("     ", end="")
            for col_idx in range(BOARD_SIZE):
                if col_idx < BOARD_SIZE - 1:
                    print("│   ", end="")
                else:
                    print("│", end="")
            print()

    # Column headers at bottom
    print("\n     ", end="")
    for col in range(BOARD_SIZE):
        print(f"{col}   ", end="")

    # Show game info
    ko = game.get_ko_point(state)
    passes = game.get_consecutive_passes(state)
    print(f"\n\n  Ko point: {ko if ko >= 0 else 'None'}, Consecutive passes: {passes}")
    print()


def get_human_move(game, state, player, can_undo=False):
    """Get a valid move from the human player."""
    valid_moves = game.get_valid_moves(state, player)

    while True:
        try:
            undo_hint = ", 'undo'" if can_undo else ""
            user_input = input(f"Enter your move (row col), 'pass'{undo_hint}: ").strip().lower()

            if user_input in ['quit', 'q', 'exit']:
                return None

            if user_input == 'undo':
                if can_undo:
                    return 'undo'
                else:
                    print("Cannot undo - no moves to undo.")
                    continue

            if user_input == 'pass':
                return PASS_ACTION

            parts = user_input.split()
            if len(parts) != 2:
                print(f"Invalid input. Enter row and column (0-{BOARD_SIZE-1}) or 'pass'")
                continue

            row, col = int(parts[0]), int(parts[1])

            if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
                print(f"Invalid coordinates. Row and column must be 0-{BOARD_SIZE-1}.")
                continue

            action_index = row * BOARD_SIZE + col

            if valid_moves[action_index] != 1:
                print("Invalid move (occupied, suicide, or ko). Choose another position.")
                continue

            return action_index

        except ValueError:
            print("Invalid input. Please enter numbers (e.g., '2 2') or 'pass'")
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
            return None


def get_bot_move(game, mcts, state, player, num_simulations=200):
    """Get the bot's move using MCTS."""
    print(f"\nBot is thinking... ({num_simulations} MCTS simulations)")

    node = Node(prior_prob=0, player=player, action_index=None)
    node.set_state(state.copy())

    root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)

    action, _, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=0.1)
    action_index = np.argmax(action)

    visit_counts = np.zeros(ACTION_SIZE)
    for k, v in root_node.children.items():
        visit_counts[k] = v.total_visits_N

    top_indices = np.argsort(visit_counts)[::-1][:3]
    print("\nBot's top 3 moves:")
    for i, idx in enumerate(top_indices, 1):
        if visit_counts[idx] > 0:
            if idx == PASS_ACTION:
                print(f"  {i}. Pass: {int(visit_counts[idx])} visits")
            else:
                row, col = idx // BOARD_SIZE, idx % BOARD_SIZE
                print(f"  {i}. Position ({row}, {col}): {int(visit_counts[idx])} visits")

    if action_index == PASS_ACTION:
        print(f"\nBot passes")
    else:
        row, col = action_index // BOARD_SIZE, action_index % BOARD_SIZE
        print(f"\nBot plays at ({row}, {col})")

    return action_index


def play_game(game, mcts, human_player, num_simulations=200):
    """Play a single game of human vs bot."""
    state = game.state.copy()
    current_player = 1  # Black always goes first

    # Track game history for undo: list of (state, player) tuples
    history = []

    print("\n" + "=" * 60)
    print("GAME START - 5x5 Go")
    print("=" * 60)
    print(f"You are playing as: {'Black (X, first)' if human_player == 1 else 'White (O, second)'}")
    print(f"Bot is playing as: {'White (O, second)' if human_player == 1 else 'Black (X, first)'}")
    print("\nRules: Capture opponent stones by surrounding them.")
    print("Game ends when both players pass consecutively.")
    print("Scoring: Area scoring with 2.5 komi for White.")
    print("Enter moves as 'row col' (e.g., '2 2'), 'pass', or 'undo'")
    print("Type 'quit' to exit")
    print("=" * 60)

    display_board(state, game)

    move_count = 0

    while True:
        move_count += 1

        if current_player == human_player:
            print(f"\n--- Move {move_count} ---")
            print(f"Your turn ({'Black' if human_player == 1 else 'White'})")

            can_undo = len(history) >= 2  # Need at least 2 moves to undo (human + bot)
            action_index = get_human_move(game, state, current_player, can_undo)

            if action_index is None:
                print("\nGame ended by user.")
                return None

            if action_index == 'undo':
                # Undo both the bot's last move and the human's last move
                if len(history) >= 2:
                    history.pop()  # Remove bot's pre-move state
                    state, current_player = history.pop()  # Restore to human's pre-move state
                    move_count = len(history)
                    print("\nUndid last two moves.")
                    display_board(state, game)
                    continue
                else:
                    print("Cannot undo - not enough moves.")
                    move_count -= 1
                    continue

            # Save state before applying move
            history.append((state.copy(), current_player))
            state = game.apply_move(state, action_index, current_player)

        else:
            print(f"\n--- Move {move_count} ---")
            print(f"Bot's turn ({'Black' if current_player == 1 else 'White'})")

            # Save state before applying move
            history.append((state.copy(), current_player))
            action_index = get_bot_move(game, mcts, state, current_player, num_simulations)
            state = game.apply_move(state, action_index, current_player)

        display_board(state, game)

        result = game.win_or_draw(state)

        if result is not None:
            print("\n" + "=" * 60)
            black_score, white_score = game.count_territory(game.get_board(state))
            print(f"Final Score: Black {black_score} - White {white_score + 2.5} (with 2.5 komi)")

            if result == 1:
                if human_player == 1:
                    print("Congratulations! You won!")
                else:
                    print("Bot (Black) wins!")
            elif result == -1:
                if human_player == -1:
                    print("Congratulations! You won!")
                else:
                    print("Bot (White) wins!")
            else:
                print("It's a draw!")
            print("=" * 60)
            return result

        current_player *= -1


def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for model_dir in [os.path.join(script_dir, cfg.SAVE_MODEL_PATH), cfg.SAVE_MODEL_PATH]:
        if os.path.isdir(model_dir):
            models = glob(os.path.join(model_dir, "*_best_model.pt"))
            if models:
                models.sort(key=os.path.getmtime, reverse=True)
                return models[0]
    return None


def main():
    """Main function to run the human vs bot game."""
    print("\n" + "=" * 60)
    print("Welcome to 5x5 Go")
    print("Human vs AlphaZero Bot")
    print("=" * 60)

    model_path = load_model()

    if model_path is None:
        print("\nERROR: No trained model found!")
        print("Please train a model first.")
        print(f"Looking in: {cfg.SAVE_MODEL_PATH}")
        return

    print(f"Loading model from: {model_path}")

    game = Go()
    vpn = ValuePolicyNetwork(model_path, use_compile=False)
    policy_value_network = vpn.get_vp
    mcts = MonteCarloTreeSearch(game, policy_value_network)

    print("\n" + "=" * 60)
    print("GAME SETTINGS")
    print("=" * 60)

    while True:
        choice = input("\nDo you want to go first (play Black)? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            human_player = 1
            break
        elif choice in ['n', 'no']:
            human_player = -1
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'")

    # Always use a fixed MCTS budget per move (no difficulty prompt).
    num_simulations = 200
    print(f"\nBot settings: {num_simulations} MCTS simulations per move")

    while True:
        result = play_game(game, mcts, human_player, num_simulations)

        if result is None:
            break

        print("\n" + "=" * 60)
        play_again = input("\nPlay again? (y/n): ").strip().lower()

        if play_again not in ['y', 'yes']:
            break

        switch = input("Switch sides? (y/n): ").strip().lower()
        if switch in ['y', 'yes']:
            human_player *= -1

    print("\nThanks for playing! Goodbye!")


if __name__ == "__main__":
    main()
