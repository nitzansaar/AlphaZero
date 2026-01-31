"""
Comprehensive sanity tests for the Value and Policy network.

Tests:
1. Output ranges and validity
2. Policy prefers legal moves
3. Value predicts won/lost positions correctly
4. Policy-Value consistency
5. Symmetry tests
6. Position understanding tests
"""

import os
import numpy as np
import torch
from glob import glob
from config import Config as cfg
from game import Go, ACTION_SIZE, NUM_POSITIONS, BOARD_SIZE, PASS_ACTION
from value_policy_function import ValuePolicyNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"


def create_position(board_2d, player=1, ko=-1, passes=0):
    """Helper to create a game state from a 2D board."""
    state = np.zeros(NUM_POSITIONS + 2)
    state[:NUM_POSITIONS] = board_2d.flatten()
    state[NUM_POSITIONS] = ko
    state[NUM_POSITIONS + 1] = passes
    return state


def print_board(state):
    """Print board state."""
    board = state[:NUM_POSITIONS].reshape(BOARD_SIZE, BOARD_SIZE)
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("  " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for i in range(BOARD_SIZE):
        row = " ".join(symbols[int(board[i, j])] for j in range(BOARD_SIZE))
        print(f"{i} {row}")


def test_output_validity(vpn, game):
    """Test that outputs are in valid ranges."""
    print("\n" + "=" * 60)
    print("TEST 1: Output Validity")
    print("=" * 60)

    passed = True

    # Test on empty board
    state = game.state.copy()
    value, policy = vpn.get_vp(state, player=1)

    # Check value range
    if -1 <= value <= 1:
        print(f"[PASS] Value in range [-1, 1]: {value:.4f}")
    else:
        print(f"[FAIL] Value out of range: {value:.4f}")
        passed = False

    # Check policy sums to 1
    policy_sum = np.sum(policy)
    if 0.99 <= policy_sum <= 1.01:
        print(f"[PASS] Policy sums to ~1: {policy_sum:.6f}")
    else:
        print(f"[FAIL] Policy doesn't sum to 1: {policy_sum:.6f}")
        passed = False

    # Check no NaN or Inf
    if not np.any(np.isnan(policy)) and not np.any(np.isinf(policy)):
        print(f"[PASS] No NaN/Inf in policy")
    else:
        print(f"[FAIL] NaN or Inf found in policy")
        passed = False

    # Check all policy values non-negative
    if np.all(policy >= 0):
        print(f"[PASS] All policy values >= 0")
    else:
        print(f"[FAIL] Negative policy values found")
        passed = False

    return passed


def test_policy_prefers_legal_moves(vpn, game):
    """Test that policy assigns higher probability to legal moves."""
    print("\n" + "=" * 60)
    print("TEST 2: Policy Prefers Legal Moves")
    print("=" * 60)

    passed = True

    # Create a position with some illegal moves
    board = np.array([
        [1, -1, 0, 0, 0],
        [-1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    state = create_position(board, player=1)

    print("Test position (X=Black, O=White):")
    print_board(state)

    value, policy = vpn.get_vp(state, player=1)
    valid_moves = game.get_valid_moves(state, player=1)

    # Position (0,0) is occupied - should be illegal
    # Position (1,1) might be suicide - check

    legal_prob = np.sum(policy * valid_moves)
    illegal_prob = np.sum(policy * (1 - valid_moves))

    print(f"\nProbability on legal moves: {legal_prob:.4f}")
    print(f"Probability on illegal moves: {illegal_prob:.4f}")

    if legal_prob > 0.9:
        print(f"[PASS] >90% probability on legal moves")
    elif legal_prob > 0.7:
        print(f"[WARN] Only {legal_prob:.1%} on legal moves (should be >90%)")
        passed = False
    else:
        print(f"[FAIL] Only {legal_prob:.1%} on legal moves")
        passed = False

    # Check specific illegal move
    occupied_idx = 0  # Position (0,0) is occupied
    if policy[occupied_idx] < 0.01:
        print(f"[PASS] Occupied position (0,0) has low probability: {policy[occupied_idx]:.6f}")
    else:
        print(f"[WARN] Occupied position has probability: {policy[occupied_idx]:.4f}")

    return passed


def test_value_on_terminal_positions(vpn, game):
    """Test that value correctly identifies won/lost positions."""
    print("\n" + "=" * 60)
    print("TEST 3: Value on Terminal/Near-Terminal Positions")
    print("=" * 60)

    passed = True

    # Position where Black dominates (should have positive value for Black)
    black_winning = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ])
    state_black_wins = create_position(black_winning, player=1)

    print("Black dominant position:")
    print_board(state_black_wins)

    value_for_black, _ = vpn.get_vp(state_black_wins, player=1)
    value_for_white, _ = vpn.get_vp(state_black_wins, player=-1)

    print(f"Value (Black's perspective): {value_for_black:.4f}")
    print(f"Value (White's perspective): {value_for_white:.4f}")

    if value_for_black > 0:
        print(f"[PASS] Black sees winning position as positive")
    else:
        print(f"[FAIL] Black should see this as winning (value > 0)")
        passed = False

    if value_for_white < 0:
        print(f"[PASS] White sees losing position as negative")
    else:
        print(f"[FAIL] White should see this as losing (value < 0)")
        passed = False

    # Position where White dominates
    white_winning = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0],
        [-1, -1, -1, 0, 0],
        [-1, -1, 0, 0, 0],
        [-1, 0, 0, 0, 0]
    ])
    state_white_wins = create_position(white_winning, player=1)

    print("\nWhite dominant position:")
    print_board(state_white_wins)

    value_for_black, _ = vpn.get_vp(state_white_wins, player=1)
    value_for_white, _ = vpn.get_vp(state_white_wins, player=-1)

    print(f"Value (Black's perspective): {value_for_black:.4f}")
    print(f"Value (White's perspective): {value_for_white:.4f}")

    if value_for_black < 0:
        print(f"[PASS] Black sees losing position as negative")
    else:
        print(f"[FAIL] Black should see this as losing (value < 0)")
        passed = False

    return passed


def test_value_empty_board(vpn, game):
    """Test value on empty board (should be close to 0 or slight Black advantage)."""
    print("\n" + "=" * 60)
    print("TEST 4: Value on Empty Board")
    print("=" * 60)

    state = game.state.copy()

    print("Empty board:")
    print_board(state)

    value_black, _ = vpn.get_vp(state, player=1)
    value_white, _ = vpn.get_vp(state, player=-1)

    print(f"Value (Black to play): {value_black:.4f}")
    print(f"Value (White to play): {value_white:.4f}")

    # Empty board should be roughly neutral, maybe slight advantage for Black (first move)
    if -0.5 <= value_black <= 0.5:
        print(f"[PASS] Empty board value is reasonable for Black")
    else:
        print(f"[WARN] Empty board value seems extreme: {value_black:.4f}")

    return True


def test_policy_center_preference(vpn, game):
    """Test if policy prefers center moves on empty board (common Go heuristic)."""
    print("\n" + "=" * 60)
    print("TEST 5: Policy Center Preference (Empty Board)")
    print("=" * 60)

    state = game.state.copy()
    value, policy = vpn.get_vp(state, player=1)

    # Get top 5 moves
    top_indices = np.argsort(policy)[::-1][:5]

    print("Top 5 moves on empty board:")
    for i, idx in enumerate(top_indices):
        if idx == PASS_ACTION:
            move_str = "pass"
        else:
            row, col = idx // BOARD_SIZE, idx % BOARD_SIZE
            move_str = f"({row}, {col})"
        print(f"  {i+1}. {move_str}: {policy[idx]:.4f}")

    # Center is (2, 2) for 5x5 board
    center_idx = 2 * BOARD_SIZE + 2
    center_prob = policy[center_idx]

    print(f"\nCenter (2,2) probability: {center_prob:.4f}")
    print(f"Center rank: {list(top_indices).index(center_idx) + 1 if center_idx in top_indices else '>5'}")

    if center_prob == max(policy[:NUM_POSITIONS]):
        print("[PASS] Center is the top move (good Go intuition)")
    elif center_prob > 0.1:
        print("[PASS] Center has reasonable probability")
    else:
        print(f"[WARN] Center has low probability: {center_prob:.4f}")

    # Check if pass is unreasonably high on empty board
    pass_prob = policy[PASS_ACTION]
    if pass_prob < 0.1:
        print(f"[PASS] Pass probability low on empty board: {pass_prob:.4f}")
    else:
        print(f"[WARN] Pass probability too high on empty board: {pass_prob:.4f}")

    return True


def test_symmetry(vpn, game):
    """Test that symmetric positions give symmetric policy."""
    print("\n" + "=" * 60)
    print("TEST 6: Rotational Symmetry")
    print("=" * 60)

    # Simple position with one stone in corner
    board1 = np.zeros((BOARD_SIZE, BOARD_SIZE))
    board1[0, 0] = 1  # Black stone in top-left
    state1 = create_position(board1, player=-1)  # White to play

    # Rotated 180 degrees
    board2 = np.zeros((BOARD_SIZE, BOARD_SIZE))
    board2[4, 4] = 1  # Black stone in bottom-right
    state2 = create_position(board2, player=-1)

    print("Position 1 (stone at 0,0):")
    print_board(state1)

    print("\nPosition 2 (stone at 4,4 - rotated 180°):")
    print_board(state2)

    value1, policy1 = vpn.get_vp(state1, player=-1)
    value2, policy2 = vpn.get_vp(state2, player=-1)

    print(f"\nValue 1: {value1:.4f}")
    print(f"Value 2: {value2:.4f}")
    print(f"Value difference: {abs(value1 - value2):.4f}")

    if abs(value1 - value2) < 0.1:
        print("[PASS] Values are similar for symmetric positions")
    else:
        print(f"[WARN] Values differ for symmetric positions")

    # Check if top move in pos1 corresponds to rotated top move in pos2
    top1 = np.argmax(policy1[:NUM_POSITIONS])
    top2 = np.argmax(policy2[:NUM_POSITIONS])

    row1, col1 = top1 // BOARD_SIZE, top1 % BOARD_SIZE
    row2, col2 = top2 // BOARD_SIZE, top2 % BOARD_SIZE

    # 180° rotation: (r, c) -> (4-r, 4-c)
    expected_row2, expected_col2 = BOARD_SIZE - 1 - row1, BOARD_SIZE - 1 - col1

    print(f"\nTop move in pos1: ({row1}, {col1})")
    print(f"Top move in pos2: ({row2}, {col2})")
    print(f"Expected (rotated): ({expected_row2}, {expected_col2})")

    if (row2, col2) == (expected_row2, expected_col2):
        print("[PASS] Top moves are rotationally symmetric")
    else:
        print("[WARN] Top moves are not symmetric (may be OK if multiple good moves)")

    return True


def test_value_policy_consistency(vpn, game):
    """Test if high-policy moves lead to positions with good value."""
    print("\n" + "=" * 60)
    print("TEST 7: Policy-Value Consistency")
    print("=" * 60)

    state = game.state.copy()
    value, policy = vpn.get_vp(state, player=1)

    # Get top 3 and bottom 3 moves
    valid_moves = game.get_valid_moves(state, player=1)
    masked_policy = policy * valid_moves

    sorted_indices = np.argsort(masked_policy)[::-1]
    top_moves = [i for i in sorted_indices if masked_policy[i] > 0][:3]
    bottom_moves = [i for i in sorted_indices if masked_policy[i] > 0][-3:]

    print("Checking if high-policy moves lead to better positions...\n")

    top_values = []
    print("Top 3 policy moves:")
    for idx in top_moves:
        if idx == PASS_ACTION:
            move_str = "pass"
        else:
            row, col = idx // BOARD_SIZE, idx % BOARD_SIZE
            move_str = f"({row}, {col})"

        # Apply move and get value for opponent
        new_state = game.apply_move(state.copy(), idx, player=1)
        opp_value, _ = vpn.get_vp(new_state, player=-1)
        our_value = -opp_value  # Flip for our perspective

        print(f"  {move_str}: policy={masked_policy[idx]:.4f}, resulting_value={our_value:.4f}")
        top_values.append(our_value)

    bottom_values = []
    print("\nBottom 3 policy moves:")
    for idx in bottom_moves:
        if idx == PASS_ACTION:
            move_str = "pass"
        else:
            row, col = idx // BOARD_SIZE, idx % BOARD_SIZE
            move_str = f"({row}, {col})"

        new_state = game.apply_move(state.copy(), idx, player=1)
        opp_value, _ = vpn.get_vp(new_state, player=-1)
        our_value = -opp_value

        print(f"  {move_str}: policy={masked_policy[idx]:.4f}, resulting_value={our_value:.4f}")
        bottom_values.append(our_value)

    avg_top = np.mean(top_values)
    avg_bottom = np.mean(bottom_values)

    print(f"\nAverage value after top moves: {avg_top:.4f}")
    print(f"Average value after bottom moves: {avg_bottom:.4f}")

    if avg_top > avg_bottom:
        print("[PASS] High-policy moves lead to better positions")
    else:
        print("[WARN] High-policy moves don't lead to better positions")
        print("       This suggests policy and value are not well aligned")

    return avg_top > avg_bottom


def test_response_to_threats(vpn, game):
    """Test if the network responds to obvious threats."""
    print("\n" + "=" * 60)
    print("TEST 8: Response to Threats (Capture Detection)")
    print("=" * 60)

    # Position where White can capture a Black stone
    # Black stone at (2,2) with only one liberty at (2,3)
    board = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, -1, 1, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    state = create_position(board, player=-1)  # White to play

    print("Position (White to play, can capture at 2,3):")
    print_board(state)

    value, policy = vpn.get_vp(state, player=-1)

    # The capture move is at (2, 3)
    capture_idx = 2 * BOARD_SIZE + 3
    capture_prob = policy[capture_idx]

    # Get top moves
    top_indices = np.argsort(policy)[::-1][:5]

    print(f"\nCapture move (2,3) probability: {capture_prob:.4f}")
    print(f"Top 5 moves:")
    for i, idx in enumerate(top_indices):
        if idx == PASS_ACTION:
            move_str = "pass"
        else:
            row, col = idx // BOARD_SIZE, idx % BOARD_SIZE
            move_str = f"({row}, {col})"
        marker = " <-- CAPTURE" if idx == capture_idx else ""
        print(f"  {i+1}. {move_str}: {policy[idx]:.4f}{marker}")

    capture_rank = list(top_indices).index(capture_idx) + 1 if capture_idx in top_indices else ">5"

    if capture_rank == 1:
        print("[PASS] Capture move is top choice")
    elif capture_rank <= 3:
        print(f"[PASS] Capture move is in top 3 (rank {capture_rank})")
    else:
        print(f"[WARN] Capture move ranked {capture_rank} - network may not understand captures")

    return True


def main():
    print("=" * 60)
    print("NEURAL NETWORK SANITY TESTS")
    print("=" * 60)

    game = Go()

    # Find model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, cfg.SAVE_MODEL_PATH)
    model_files = glob(os.path.join(model_dir, "*_best_model.pt"))

    if not model_files:
        print("No trained models found!")
        return

    # Find best model (iteration 1 based on earlier tests)
    models_with_iter = []
    for f in model_files:
        try:
            iter_num = int(os.path.basename(f).split("_")[0])
            models_with_iter.append((iter_num, f))
        except ValueError:
            continue

    models_with_iter.sort(key=lambda x: x[0])

    # Use iteration 1 if available, else latest
    best_path = None
    for iter_num, path in models_with_iter:
        if iter_num == 1:
            best_path = path
            break
    if best_path is None:
        best_path = models_with_iter[-1][1]

    print(f"\nUsing model: {os.path.basename(best_path)}")
    vpn = ValuePolicyNetwork(best_path, use_compile=False)

    # Run all tests
    results = []

    results.append(("Output Validity", test_output_validity(vpn, game)))
    results.append(("Legal Move Preference", test_policy_prefers_legal_moves(vpn, game)))
    results.append(("Terminal Position Values", test_value_on_terminal_positions(vpn, game)))
    results.append(("Empty Board Value", test_value_empty_board(vpn, game)))
    results.append(("Center Preference", test_policy_center_preference(vpn, game)))
    results.append(("Symmetry", test_symmetry(vpn, game)))
    results.append(("Policy-Value Consistency", test_value_policy_consistency(vpn, game)))
    results.append(("Threat Response", test_response_to_threats(vpn, game)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL/WARN"
        print(f"  {name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\nThe network appears to be functioning correctly.")
    elif passed >= total - 2:
        print("\nThe network has minor issues but is mostly functional.")
    else:
        print("\nThe network has significant issues that may affect learning.")


if __name__ == "__main__":
    main()
