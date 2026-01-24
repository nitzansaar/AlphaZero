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

def get_human_move(game, state, player_name):
    """
    Get a valid move from a human player.
    Returns the action index (0-80).
    """
    valid_moves = game.get_valid_moves(state)

    while True:
        try:
            user_input = input(f"{player_name}, enter your move (row col), e.g., '4 4': ").strip()

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

def check_winner(game, state):
    """
    Check if there's a winner or draw.
    Returns: 1 (player 1 wins), -1 (player -1 wins), 0 (draw), None (game continues)
    """
    return game.win_or_draw(state)

def play_game(game, player1_name="Player 1 (X)", player2_name="Player 2 (O)"):
    """
    Play a single game of human vs human.

    Args:
        game: TicTacToe instance
        player1_name: Name for player 1 (X)
        player2_name: Name for player 2 (O)
    """
    state = np.zeros(cfg.ACTION_SIZE)  # Absolute board state
    current_player = 1  # Player 1 (X) always goes first

    print("\n" + "=" * 60)
    print("GAME START")
    print("=" * 60)
    print(f"{player1_name} plays X (goes first)")
    print(f"{player2_name} plays O (goes second)")
    print("\nGoal: Get 5 in a row (horizontally, vertically, or diagonally)")
    print("Enter moves as 'row col' (e.g., '4 4' for center)")
    print("Type 'quit' to exit the game")
    print("=" * 60)

    display_board(state)

    move_count = 0

    while True:
        move_count += 1

        # Determine which player's turn it is
        if current_player == 1:
            current_player_name = player1_name
            symbol = 'X'
        else:
            current_player_name = player2_name
            symbol = 'O'

        print(f"\n--- Move {move_count} ---")
        print(f"{current_player_name}'s turn ({symbol})")

        action_index = get_human_move(game, state, current_player_name)

        if action_index is None:
            print("\nGame ended by user.")
            return None

        # Update state
        state[action_index] = current_player

        # Display updated board
        display_board(state)

        # Check for winner or draw
        result = check_winner(game, state)

        if result is not None:
            print("\n" + "=" * 60)
            if result == 1:
                print(f"{player1_name} wins!")
            elif result == -1:
                print(f"{player2_name} wins!")
            else:
                print("It's a draw!")
            print("=" * 60)
            return result

        # Switch player
        current_player *= -1

def main():
    """
    Main function to run the human vs human game.
    """
    print("\n" + "=" * 60)
    print("Welcome to 9x9 Tic-Tac-Toe (5-in-a-row)")
    print("Human vs Human")
    print("=" * 60)

    # Initialize game
    game = TicTacToe()

    # Use default player names
    player1_name = "Player 1"
    player2_name = "Player 2"

    # Play games in a loop
    wins = {player1_name: 0, player2_name: 0, "draws": 0}

    while True:
        result = play_game(game, player1_name, player2_name)

        if result is None:
            # User quit mid-game
            break

        # Update win counter
        if result == 1:
            wins[player1_name] += 1
        elif result == -1:
            wins[player2_name] += 1
        else:
            wins["draws"] += 1

        # Display score
        print("\n" + "=" * 60)
        print("SCORE:")
        print(f"  {player1_name}: {wins[player1_name]} wins")
        print(f"  {player2_name}: {wins[player2_name]} wins")
        print(f"  Draws: {wins['draws']}")
        print("=" * 60)

        # Ask if users want to play again
        play_again = input("\nPlay again? (y/n): ").strip().lower()

        if play_again not in ['y', 'yes']:
            break

    # Display final score
    print("\n" + "=" * 60)
    print("FINAL SCORE:")
    print(f"  {player1_name}: {wins[player1_name]} wins")
    print(f"  {player2_name}: {wins[player2_name]} wins")
    print(f"  Draws: {wins['draws']}")
    print("=" * 60)
    print("\nThanks for playing! Goodbye!")

if __name__ == "__main__":
    main()
