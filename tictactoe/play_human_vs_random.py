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

class RandomBot:
    """
    A bot that randomly selects from valid moves.
    This is the simplest possible baseline opponent.
    """
    def __init__(self, game):
        self.game = game

    def get_action(self, state, player):
        """
        Get bot's move: randomly select from valid moves.
        Returns action index.
        """
        valid_moves = self.game.get_valid_moves(state)
        valid_indices = np.where(valid_moves == 1)[0]

        if len(valid_indices) == 0:
            return None

        action_index = np.random.choice(valid_indices)
        return action_index

def get_bot_move(random_bot, state, player):
    """
    Get the random bot's move.
    Returns the action index.
    """
    print(f"\nRandom Bot is thinking...")

    action_index = random_bot.get_action(state, player)

    row, col = action_index // 9, action_index % 9
    print(f"Random Bot plays at ({row}, {col})")

    return action_index

def check_winner(game, state):
    """
    Check if there's a winner or draw.
    Returns: 1 (player 1 wins), -1 (player -1 wins), 0 (draw), None (game continues)
    """
    return game.win_or_draw(state)

def play_game(game, random_bot, human_player):
    """
    Play a single game of human vs random bot.

    Args:
        game: TicTacToe instance
        random_bot: RandomBot instance
        human_player: 1 if human plays X (goes first), -1 if human plays O (goes second)
    """
    state = np.zeros(cfg.ACTION_SIZE)  # Absolute board state
    current_player = 1  # Player 1 (X) always goes first

    print("\n" + "=" * 60)
    print("GAME START")
    print("=" * 60)
    print(f"You are playing as: {'X (first)' if human_player == 1 else 'O (second)'}")
    print(f"Random Bot is playing as: {'O (second)' if human_player == 1 else 'X (first)'}")
    print("\nGoal: Get 5 in a row (horizontally, vertically, or diagonally)")
    print("Enter moves as 'row col' (e.g., '4 4' for center)")
    print("Type 'quit' to exit the game")
    print("\nNote: Random Bot selects moves completely at random")
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
            print(f"Random Bot's turn ({'X' if current_player == 1 else 'O'})")

            action_index = get_bot_move(random_bot, state, current_player)

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
                    print("Random Bot (X) wins!")
            elif result == -1:
                winner = "O (Player -1)"
                if human_player == -1:
                    print("CONGRATULATIONS! You won!")
                else:
                    print("Random Bot (O) wins!")
            else:
                winner = "Draw"
                print("It's a draw!")
            print("=" * 60)
            return result

        # Switch player
        current_player *= -1

def main():
    """
    Main function to run the human vs random bot game.
    """
    print("\n" + "=" * 60)
    print("Welcome to 9x9 Tic-Tac-Toe (5-in-a-row)")
    print("Human vs Random Bot")
    print("=" * 60)
    print("\nThe Random Bot:")
    print("- Selects moves completely at random from valid positions")
    print("- No strategy or intelligence")
    print("- Easiest possible opponent")
    print("=" * 60)

    # Initialize game and random bot
    game = TicTacToe()
    random_bot = RandomBot(game)

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
        result = play_game(game, random_bot, human_player)

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
