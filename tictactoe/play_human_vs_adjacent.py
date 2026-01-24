import numpy as np
from config import Config as cfg
from game import TicTacToe

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

class AdjacentBot:
    """
    A simple heuristic bot that places pieces adjacent to its own existing pieces.
    This is a non-neural network baseline opponent.
    """
    def __init__(self, game):
        self.game = game
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # orthogonal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonal
        ]

    def get_adjacent_empty_cells(self, state, player_value):
        """
        Find all empty cells adjacent to cells occupied by player_value.
        """
        board_2d = state.reshape(9, 9)
        adjacent_cells = set()
        player_cells = np.argwhere(board_2d == player_value)

        for row, col in player_cells:
            for dr, dc in self.directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 9 and 0 <= new_col < 9:
                    if board_2d[new_row, new_col] == 0:
                        flat_index = new_row * 9 + new_col
                        adjacent_cells.add(flat_index)
        return adjacent_cells

    def get_action(self, state, player):
        """
        Get bot's move: prefer cells adjacent to its own pieces, fallback to random.
        Returns action index.
        """
        valid_moves = self.game.get_valid_moves(state)
        valid_indices = np.where(valid_moves == 1)[0]

        if len(valid_indices) == 0:
            return None

        # Canonicalize state to current player's perspective
        canonical_state = state * player

        # Find adjacent empty cells to current player's pieces (value 1 in canonical state)
        adjacent_cells = self.get_adjacent_empty_cells(canonical_state, player_value=1)
        adjacent_valid = list(adjacent_cells.intersection(set(valid_indices)))

        if adjacent_valid:
            # Choose randomly from adjacent cells
            action_index = np.random.choice(adjacent_valid)
        else:
            # No adjacent cells (likely first move), choose random
            action_index = np.random.choice(valid_indices)

        return action_index

def get_bot_move(adjacent_bot, state, player):
    """
    Get the adjacent bot's move.
    Returns the action index.
    """
    print(f"\nAdjacent Bot is thinking...")

    action_index = adjacent_bot.get_action(state, player)

    row, col = action_index // 9, action_index % 9
    print(f"Adjacent Bot plays at ({row}, {col})")

    return action_index

def check_winner(game, state):
    """
    Check if there's a winner or draw.
    Returns: 1 (player 1 wins), -1 (player -1 wins), 0 (draw), None (game continues)
    """
    return game.win_or_draw(state)

def play_game(game, adjacent_bot, human_player):
    """
    Play a single game of human vs adjacent bot.

    Args:
        game: TicTacToe instance
        adjacent_bot: AdjacentBot instance
        human_player: 1 if human plays X (goes first), -1 if human plays O (goes second)
    """
    state = np.zeros(cfg.ACTION_SIZE)  # Absolute board state
    current_player = 1  # Player 1 (X) always goes first

    print("\n" + "=" * 60)
    print("GAME START")
    print("=" * 60)
    print(f"You are playing as: {'X (first)' if human_player == 1 else 'O (second)'}")
    print(f"Adjacent Bot (heuristic) is playing as: {'O (second)' if human_player == 1 else 'X (first)'}")
    print("\nGoal: Get 5 in a row (horizontally, vertically, or diagonally)")
    print("Enter moves as 'row col' (e.g., '4 4' for center)")
    print("Type 'quit' to exit the game")
    print("\nNote: Adjacent Bot uses a simple heuristic strategy (places next to its pieces)")
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
            print(f"Adjacent Bot's turn ({'X' if current_player == 1 else 'O'})")

            action_index = get_bot_move(adjacent_bot, state, current_player)

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
                    print("CONGRATULATIONS! You won!")
                else:
                    print("Adjacent Bot (X) wins!")
            elif result == -1:
                winner = "O (Player -1)"
                if human_player == -1:
                    print("CONGRATULATIONS! You won!")
                else:
                    print("Adjacent Bot (O) wins!")
            else:
                winner = "Draw"
                print("It's a draw!")
            print("=" * 60)
            return result

        # Switch player
        current_player *= -1

def main():
    """
    Main function to run the human vs adjacent bot game.
    """
    print("\n" + "=" * 60)
    print("Welcome to 9x9 Tic-Tac-Toe (5-in-a-row)")
    print("Human vs Adjacent Bot (Heuristic Strategy)")
    print("=" * 60)
    print("\nThe Adjacent Bot uses a simple heuristic strategy:")
    print("- Places pieces adjacent to its own existing pieces")
    print("- No neural network or MCTS involved")
    print("- Good baseline opponent for testing")
    print("=" * 60)

    # Initialize game and adjacent bot
    game = TicTacToe()
    adjacent_bot = AdjacentBot(game)

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

    # Play games in a loop
    while True:
        result = play_game(game, adjacent_bot, human_player)

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
