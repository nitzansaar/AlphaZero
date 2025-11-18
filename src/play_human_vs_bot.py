import os
import numpy as np
from glob import glob
import torch
from config import Config as cfg
from game import TicTacToe
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

def format_board_state(state):
    """
    Convert board state to a readable 2D representation.
    Returns a 9x9 grid with 'X' for player 1, 'O' for player -1, '.' for empty
    """
    board_2d = state.reshape(9, 9)
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

def display_board(state):
    """
    Display the board in a user-friendly format with row/column numbers.
    """
    board_2d = format_board_state(state)

    print("\n    ", end="")
    for col in range(9):
        print(f"  {col} ", end="")
    print("\n   +" + "---+" * 9)

    for row_idx, row in enumerate(board_2d):
        print(f" {row_idx} |", end="")
        for cell in row:
            print(f" {cell} |", end="")
        print(f" {row_idx}")
        print("   +" + "---+" * 9)

    print("    ", end="")
    for col in range(9):
        print(f"  {col} ", end="")
    print("\n")

def get_human_move(game, state):
    """
    Get a valid move from the human player.
    Returns the action index (0-80).
    """
    valid_moves = game.get_valid_moves(state)

    while True:
        try:
            user_input = input("Enter your move (row col), e.g., '4 4': ").strip()

            if user_input.lower() in ['quit', 'q', 'exit']:
                return None

            parts = user_input.split()
            if len(parts) != 2:
                print("Invalid input. Please enter row and column separated by space (e.g., '4 4')")
                continue

            row, col = int(parts[0]), int(parts[1])

            if row < 0 or row > 8 or col < 0 or col > 8:
                print("Invalid coordinates. Row and column must be between 0 and 8.")
                continue

            action_index = row * 9 + col

            if valid_moves[action_index] != 1:
                print("That position is already taken. Please choose an empty position.")
                continue

            return action_index

        except ValueError:
            print("Invalid input. Please enter numbers only (e.g., '4 4')")
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
            return None

def get_bot_move(game, mcts, state, player, num_simulations=800):
    """
    Get the bot's move using MCTS.
    Returns the action index.
    """
    print(f"\nBot is thinking... (running {num_simulations} MCTS simulations)")

    # Create node for current state
    node = Node(prior_prob=0, player=player, action_index=None)
    node.set_state(state.copy())

    # Run MCTS
    root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)

    # Select best move (exploit mode for competitive play)
    action, _, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=0.1)
    action_index = np.argmax(action)

    # Show bot's top moves
    visit_counts = np.zeros(cfg.ACTION_SIZE)
    for k, v in root_node.children.items():
        visit_counts[k] = v.total_visits_N

    top_indices = np.argsort(visit_counts)[::-1][:3]
    print("\nBot's top 3 moves:")
    for i, idx in enumerate(top_indices, 1):
        if visit_counts[idx] > 0:
            row, col = idx // 9, idx % 9
            print(f"  {i}. Position ({row}, {col}): {int(visit_counts[idx])} visits")

    row, col = action_index // 9, action_index % 9
    print(f"\nBot plays at ({row}, {col})")

    return action_index

def check_winner(game, state):
    """
    Check if there's a winner or draw.
    Returns: 1 (player 1 wins), -1 (player -1 wins), 0 (draw), None (game continues)
    """
    return game.win_or_draw(state)

def play_game(game, mcts, human_player, num_simulations=800):
    """
    Play a single game of human vs bot.

    Args:
        game: TicTacToe instance
        mcts: MonteCarloTreeSearch instance
        human_player: 1 if human plays X (goes first), -1 if human plays O (goes second)
        num_simulations: Number of MCTS simulations for bot
    """
    state = np.zeros(cfg.ACTION_SIZE)  # Absolute board state
    current_player = 1  # Player 1 (X) always goes first

    print("\n" + "=" * 60)
    print("GAME START")
    print("=" * 60)
    print(f"You are playing as: {'X (first)' if human_player == 1 else 'O (second)'}")
    print(f"Bot is playing as: {'O (second)' if human_player == 1 else 'X (first)'}")
    print("\nGoal: Get 5 in a row (horizontally, vertically, or diagonally)")
    print("Enter moves as 'row col' (e.g., '4 4' for center)")
    print("Type 'quit' to exit the game")
    print("=" * 60)

    display_board(state)

    move_count = 0

    while True:
        move_count += 1

        # Determine if it's human's turn or bot's turn
        if current_player == human_player:
            # Human's turn
            print(f"\n--- Move {move_count} ---")
            print(f"Your turn ({'X' if human_player == 1 else 'O'})")

            action_index = get_human_move(game, state)

            if action_index is None:
                print("\nGame ended by user.")
                return None

            # Update state
            state[action_index] = current_player

        else:
            # Bot's turn
            print(f"\n--- Move {move_count} ---")
            print(f"Bot's turn ({'X' if current_player == 1 else 'O'})")

            # For bot, we need the canonicalized state (from current player's perspective)
            canonical_state = state.copy() * current_player
            action_index = get_bot_move(game, mcts, canonical_state, current_player, num_simulations)

            # Update state
            state[action_index] = current_player

        # Display updated board
        display_board(state)

        # Check for winner or draw
        result = check_winner(game, state)

        if result is not None:
            print("\n" + "=" * 60)
            if result == 1:
                winner = "X (Player 1)"
                if human_player == 1:
                    print("🎉 CONGRATULATIONS! You won! 🎉")
                else:
                    print("Bot (X) wins!")
            elif result == -1:
                winner = "O (Player -1)"
                if human_player == -1:
                    print("🎉 CONGRATULATIONS! You won! 🎉")
                else:
                    print("Bot (O) wins!")
            else:
                winner = "Draw"
                print("It's a draw!")
            print("=" * 60)
            return result

        # Switch player
        current_player *= -1

def load_model():
    """
    Load the latest trained model.
    Returns the model path or None if no model found.
    """
    all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*_best_model.pt"))
    model_path = None

    if all_models:
        # Get modification time for each model
        models_with_time = []
        for f in all_models:
            try:
                mtime = os.path.getmtime(f)
                models_with_time.append((mtime, f))
            except OSError:
                continue

        if models_with_time:
            # Sort by modification time (most recent first)
            models_with_time.sort(reverse=True)

            # Try loading models starting from most recent until one works
            for mtime, model_file in models_with_time:
                try:
                    # Quick test: try to load state dict to check architecture
                    test_model = NeuralNetwork().to(device)
                    test_state = torch.load(model_file, map_location=device)
                    test_model.load_state_dict(test_state)
                    # If we get here, architecture matches!
                    model_path = model_file
                    model_name = os.path.basename(model_file)
                    print(f"Found compatible model: {model_name}")
                    break
                except (RuntimeError, FileNotFoundError) as e:
                    # Architecture mismatch or file error, try next
                    continue
                finally:
                    # Clean up test model
                    del test_model
                    if 'test_state' in locals():
                        del test_state
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # If no compatible model found, fall back to highest number
            if model_path is None:
                files_with_numbers = []
                for f in all_models:
                    basename = os.path.basename(f)
                    if "_best_model.pt" in basename:
                        try:
                            num = int(basename.split("_")[0])
                            files_with_numbers.append((num, f))
                        except ValueError:
                            continue

                if files_with_numbers:
                    latest_num, model_path = max(files_with_numbers, key=lambda x: x[0])
                    print(f"Using highest numbered model: {latest_num}")

    if model_path and os.path.exists(model_path):
        return model_path

    return None

def main():
    """
    Main function to run the human vs bot game.
    """
    print("\n" + "=" * 60)
    print("Welcome to 9x9 Tic-Tac-Toe (5-in-a-row)")
    print("Human vs AlphaZero Bot")
    print("=" * 60)

    # Load model
    model_path = load_model()

    if model_path is None:
        print("\nERROR: No trained model found!")
        print("Please train a model first using: ./train.sh")
        print(f"Looking in: {cfg.SAVE_MODEL_PATH}")
        return

    print(f"Loading model from: {model_path}")

    # Initialize game and MCTS
    game = TicTacToe()
    vpn = ValuePolicyNetwork(model_path, use_compile=False)  # Disable compile for interactive play
    policy_value_network = vpn.get_vp
    mcts = MonteCarloTreeSearch(game, policy_value_network)

    # Get game settings
    print("\n" + "=" * 60)
    print("GAME SETTINGS")
    print("=" * 60)

    # Choose who goes first
    while True:
        choice = input("\nDo you want to go first? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            human_player = 1  # Human is X (goes first)
            break
        elif choice in ['n', 'no']:
            human_player = -1  # Human is O (goes second)
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'")

    # Choose difficulty (number of MCTS simulations)
    print("\nChoose difficulty:")
    print("  1. Easy (200 simulations)")
    print("  2. Medium (400 simulations)")
    print("  3. Hard (800 simulations)")
    print("  4. Expert (1600 simulations)")

    while True:
        choice = input("\nEnter difficulty (1-4): ").strip()
        if choice == '1':
            num_simulations = 200
            break
        elif choice == '2':
            num_simulations = 400
            break
        elif choice == '3':
            num_simulations = 800
            break
        elif choice == '4':
            num_simulations = 1600
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4")

    print(f"\nDifficulty set: {num_simulations} MCTS simulations per move")

    # Play games in a loop
    while True:
        result = play_game(game, mcts, human_player, num_simulations)

        if result is None:
            # User quit mid-game
            break

        # Ask if user wants to play again
        print("\n" + "=" * 60)
        play_again = input("\nPlay again? (y/n): ").strip().lower()

        if play_again not in ['y', 'yes']:
            break

        # Ask if user wants to switch sides
        switch = input("Switch sides? (y/n): ").strip().lower()
        if switch in ['y', 'yes']:
            human_player *= -1

    print("\nThanks for playing! Goodbye!")

if __name__ == "__main__":
    main()
