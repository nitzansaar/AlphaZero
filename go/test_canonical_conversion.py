"""
Test that canonical board conversion is working correctly.
This is critical - if the network sees inconsistent representations, it can't learn.
"""

import numpy as np
from game import Go, board_to_canonical_3d, BOARD_SIZE, NUM_POSITIONS

def print_planes(planes):
    """Print the 3 canonical planes."""
    names = ["Current Player (plane 0)", "Opponent (plane 1)", "Empty (plane 2)"]
    for i, name in enumerate(names):
        print(f"\n{name}:")
        for row in range(BOARD_SIZE):
            print("  " + " ".join(str(int(planes[i, row, col])) for col in range(BOARD_SIZE)))

def main():
    print("=" * 60)
    print("CANONICAL CONVERSION TEST")
    print("=" * 60)

    # Create a test position
    # Black stones at (0,0), (1,1), (2,2)
    # White stones at (0,1), (1,0)
    board = np.array([
        [1, -1, 0, 0, 0],
        [-1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

    state = np.zeros(NUM_POSITIONS + 2)
    state[:NUM_POSITIONS] = board.flatten()

    print("\nOriginal board (absolute form):")
    print("  Black (1): X, White (-1): O")
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("  " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for i in range(BOARD_SIZE):
        row = " ".join(symbols[int(board[i, j])] for j in range(BOARD_SIZE))
        print(f"  {i} {row}")

    print("\n" + "-" * 60)
    print("TEST 1: Black to move (player=1)")
    print("-" * 60)

    planes_black = board_to_canonical_3d(state, player=1)
    print_planes(planes_black)

    # Verify
    print("\nVerification:")
    print(f"  Plane 0 should show BLACK stones (current player)")
    plane0_matches_black = np.array_equal(planes_black[0], (board == 1).astype(np.float32))
    print(f"  Plane 0 matches Black positions: {plane0_matches_black}")

    print(f"  Plane 1 should show WHITE stones (opponent)")
    plane1_matches_white = np.array_equal(planes_black[1], (board == -1).astype(np.float32))
    print(f"  Plane 1 matches White positions: {plane1_matches_white}")

    if plane0_matches_black and plane1_matches_white:
        print("  [PASS] Black to move conversion is CORRECT")
    else:
        print("  [FAIL] Black to move conversion is WRONG")

    print("\n" + "-" * 60)
    print("TEST 2: White to move (player=-1)")
    print("-" * 60)

    planes_white = board_to_canonical_3d(state, player=-1)
    print_planes(planes_white)

    # Verify
    print("\nVerification:")
    print(f"  Plane 0 should show WHITE stones (current player)")
    plane0_matches_white = np.array_equal(planes_white[0], (board == -1).astype(np.float32))
    print(f"  Plane 0 matches White positions: {plane0_matches_white}")

    print(f"  Plane 1 should show BLACK stones (opponent)")
    plane1_matches_black = np.array_equal(planes_white[1], (board == 1).astype(np.float32))
    print(f"  Plane 1 matches Black positions: {plane1_matches_black}")

    if plane0_matches_white and plane1_matches_black:
        print("  [PASS] White to move conversion is CORRECT")
    else:
        print("  [FAIL] White to move conversion is WRONG")

    print("\n" + "-" * 60)
    print("TEST 3: Consistency check")
    print("-" * 60)

    # The key insight: in canonical form, current player should ALWAYS be in plane 0
    # If we flip who's playing, planes 0 and 1 should swap

    planes_swapped = planes_black[0].copy() == planes_white[1]
    planes_swapped2 = planes_black[1].copy() == planes_white[0]

    print(f"  Black's plane 0 == White's plane 1: {np.all(planes_swapped)}")
    print(f"  Black's plane 1 == White's plane 0: {np.all(planes_swapped2)}")

    if np.all(planes_swapped) and np.all(planes_swapped2):
        print("  [PASS] Planes correctly swap when player changes")
    else:
        print("  [FAIL] Planes do not correctly swap")

    print("\n" + "-" * 60)
    print("TEST 4: Training data simulation")
    print("-" * 60)

    # Simulate what happens in selfplay.py
    print("\nSimulating selfplay data creation:")

    # Case 1: Black's turn in selfplay
    # In selfplay, node.state is in canonical form (current player = 1)
    # parent_state = node.state * player to convert to absolute
    canonical_black_turn = board.copy()  # Black=1 (canonical, Black's turn)
    canonical_black_turn[canonical_black_turn == 1] = 99  # temp
    canonical_black_turn[canonical_black_turn == -1] = -1  # opponent
    canonical_black_turn[canonical_black_turn == 99] = 1  # current player
    # Actually, in canonical form current player is always 1
    # So canonical_black_turn already has Black=1

    player = 1  # Black's turn
    saved_state = canonical_black_turn.flatten() * player  # Convert to "absolute"
    # saved_state *= 1 = Black=1 (still absolute)

    print(f"\n  Black's turn (player=1):")
    print(f"    Canonical state: Black=1, White=-1")
    print(f"    Saved state (×player): Black={1*player}, White={-1*player}")
    print(f"    In GoDataset, canonical = saved × player = Black={1*player*player}, White={-1*player*player}")
    print(f"    Network sees: plane 0 = Black (correct, current player)")

    # Case 2: White's turn in selfplay
    # After Black moves, state is flipped so White's stones = 1
    canonical_white_turn = -board.copy()  # Flipped: White=1, Black=-1

    player = -1  # White's turn
    saved_state_white = canonical_white_turn.flatten() * player
    # saved_state_white = (White=1, Black=-1) × (-1) = (White=-1, Black=1) = absolute form

    print(f"\n  White's turn (player=-1):")
    print(f"    Canonical state: White=1, Black=-1")
    print(f"    Saved state (×player): White={1*player}, Black={-1*player}")
    print(f"    In GoDataset, canonical = saved × player = White={1*player*player}, Black={-1*player*player}")
    print(f"    Network sees: plane 0 = White (correct, current player)")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The canonical conversion logic is:
1. Selfplay saves states in ABSOLUTE form (Black=+1, White=-1 always)
   - Done by: parent_state[:NUM_POSITIONS] *= player

2. GoDataset converts to CANONICAL form for network input
   - Done by: canonical = state × player
   - Result: current player's stones = +1, opponent = -1

3. Network always sees:
   - Plane 0: current player's stones
   - Plane 1: opponent's stones
   - Plane 2: empty positions

If this is working correctly, the network learns:
   - "When plane 0 dominates → value should be positive"
   - This is true regardless of whether Black or White is playing
""")

if __name__ == "__main__":
    main()
