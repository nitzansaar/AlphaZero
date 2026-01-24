from game import Go, BOARD_SIZE, PASS_ACTION, NUM_POSITIONS


def parse_move(move_str):
    """
    Parse human input into an action index.
    Accepts: 'pass', 'p', or coordinates like '2,3' or '2 3'
    """
    move_str = move_str.strip().lower()

    if move_str in ('pass', 'p'):
        return PASS_ACTION

    # Try parsing as coordinates
    try:
        if ',' in move_str:
            parts = move_str.split(',')
        else:
            parts = move_str.split()

        if len(parts) == 2:
            row, col = int(parts[0]), int(parts[1])
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                return row * BOARD_SIZE + col
    except ValueError:
        pass

    return None


def main():
    game = Go()
    state = game.state.copy()
    player = 1  # Black starts
    player_names = {1: "Black (X)", -1: "White (O)"}

    print("=== 5x5 Go: Human vs Human ===")
    print("Enter moves as 'row col' (e.g., '2 3') or 'pass'")
    print("Rows and columns are 0-indexed (0-4)")
    print()

    while True:
        game.render(state)
        print()

        # Check if game ended
        result = game.win_or_draw(state, perspective=1)
        if result is not None:
            if result == 1:
                print("Black (X) wins!")
            elif result == -1:
                print("White (O) wins!")
            else:
                print("It's a draw!")

            # Show final scores
            board = game.get_board(state)
            black_score, white_score = game.count_territory(board)
            print(f"Final scores - Black: {black_score}, White: {white_score + 2.5} (includes 2.5 komi)")
            break

        # Get valid moves
        valid_moves = game.get_valid_moves(state, player)

        # Get move from current player
        while True:
            move_str = input(f"{player_names[player]}'s turn: ")
            action = parse_move(move_str)

            if action is None:
                print("Invalid input. Enter 'row col' (e.g., '2 3') or 'pass'")
                continue

            if valid_moves[action] != 1:
                print("Invalid move. That position is occupied, suicide, or ko violation.")
                continue

            break

        # Apply move
        state, _, player = game.play(state, player, action, perspective=1)


if __name__ == "__main__":
    main()
