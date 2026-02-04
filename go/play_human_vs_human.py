from game import Go, BOARD_SIZE, PASS_ACTION, NUM_POSITIONS

# Column letters (skip I, standard Go convention)
COL_LETTERS = 'ABCDEFGHJ'[:BOARD_SIZE]


def col_to_letter(c):
    return COL_LETTERS[c]


def letter_to_col(ch):
    ch = ch.upper()
    if ch in COL_LETTERS:
        return COL_LETTERS.index(ch)
    return -1


def move_name(idx):
    """Convert an action index to a display string like 'C3' or 'pass'."""
    if idx == PASS_ACTION:
        return 'pass'
    row, col = idx // BOARD_SIZE, idx % BOARD_SIZE
    return f'{col_to_letter(col)}{BOARD_SIZE - row}'


def display_board(state, game):
    """Display the board in traditional Go intersection style."""
    board = game.get_board(state).reshape(BOARD_SIZE, BOARD_SIZE)
    symbols = {0: '+', 1: '○', -1: '●'}

    # Column headers
    print("\n     " + "   ".join(COL_LETTERS) + "\n")

    for row_idx in range(BOARD_SIZE):
        row_num = BOARD_SIZE - row_idx
        print(f"{row_num:>2}   ", end="")
        for col_idx in range(BOARD_SIZE):
            symbol = symbols[int(board[row_idx, col_idx])]
            if col_idx < BOARD_SIZE - 1:
                print(f"{symbol}───", end="")
            else:
                print(f"{symbol}", end="")
        print(f"   {row_num}")

        if row_idx < BOARD_SIZE - 1:
            print("     ", end="")
            for col_idx in range(BOARD_SIZE):
                if col_idx < BOARD_SIZE - 1:
                    print("│   ", end="")
                else:
                    print("│", end="")
            print()

    # Column headers at bottom
    print("\n     " + "   ".join(COL_LETTERS))

    ko = game.get_ko_point(state)
    passes = game.get_consecutive_passes(state)
    print(f"\n  Ko point: {move_name(int(ko)) if ko >= 0 else 'None'}, Consecutive passes: {passes}")
    print()


def parse_move(move_str):
    """
    Parse human input into an action index.
    Accepts: 'pass', 'p', or coordinates like 'C3', 'a1'
    """
    move_str = move_str.strip()

    if move_str.lower() in ('pass', 'p'):
        return PASS_ACTION

    # Try parsing as letter+number
    try:
        move_str = move_str.upper()
        if len(move_str) < 2:
            return None
        col_ch = move_str[0]
        row_num = int(move_str[1:])
        col = letter_to_col(col_ch)
        row = BOARD_SIZE - row_num
        if col >= 0 and 0 <= row < BOARD_SIZE:
            return row * BOARD_SIZE + col
    except ValueError:
        pass

    return None


def main():
    game = Go()
    state = game.state.copy()
    player = 1  # Black starts
    player_names = {1: "Black (○)", -1: "White (●)"}

    print(f"=== {BOARD_SIZE}x{BOARD_SIZE} Go: Human vs Human ===")
    print(f"Enter moves as column letter + row number (e.g. 'C3') or 'pass'")
    print(f"Columns: A-{COL_LETTERS[-1]}, Rows: 1-{BOARD_SIZE} (1 at bottom)")
    print()

    while True:
        display_board(state, game)

        # Check if game ended
        result = game.win_or_draw(state, perspective=1)
        if result is not None:
            if result == 1:
                print("Black (○) wins!")
            elif result == -1:
                print("White (●) wins!")
            else:
                print("It's a draw!")

            # Show final scores
            board = game.get_board(state)
            black_score, white_score = game.count_territory(board)
            from config import KOMI
            print(f"Final scores - Black: {black_score}, White: {white_score + KOMI} (includes {KOMI} komi)")
            break

        # Get valid moves
        valid_moves = game.get_valid_moves(state, player)

        # Get move from current player
        while True:
            move_str = input(f"{player_names[player]}'s turn: ")
            action = parse_move(move_str)

            if action is None:
                print(f"Invalid input. Enter column letter + row number (e.g. 'C3') or 'pass'")
                continue

            if valid_moves[action] != 1:
                print("Invalid move. That position is occupied, suicide, or ko violation.")
                continue

            break

        # Apply move
        state, _, player = game.play(state, player, action, perspective=1)


if __name__ == "__main__":
    main()
