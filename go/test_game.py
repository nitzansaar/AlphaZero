import unittest
import numpy as np
from game import (
    Go, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE,
    idx_to_coord, coord_to_idx, get_neighbors, board_to_canonical_3d
)


class TestCoordinateConversions(unittest.TestCase):
    """Test coordinate conversion functions."""

    def test_idx_to_coord(self):
        # Corners
        self.assertEqual(idx_to_coord(0), (0, 0))  # top-left
        self.assertEqual(idx_to_coord(4), (0, 4))  # top-right
        self.assertEqual(idx_to_coord(20), (4, 0))  # bottom-left
        self.assertEqual(idx_to_coord(24), (4, 4))  # bottom-right

    def test_coord_to_idx(self):
        self.assertEqual(coord_to_idx(0, 0), 0)
        self.assertEqual(coord_to_idx(0, 4), 4)
        self.assertEqual(coord_to_idx(4, 0), 20)
        self.assertEqual(coord_to_idx(4, 4), 24)

    def test_roundtrip(self):
        for idx in range(NUM_POSITIONS):
            row, col = idx_to_coord(idx)
            self.assertEqual(coord_to_idx(row, col), idx)


class TestNeighbors(unittest.TestCase):
    """Test neighbor finding."""

    def test_corner_neighbors(self):
        # Top-left corner (0) has 2 neighbors
        neighbors = get_neighbors(0)
        self.assertEqual(set(neighbors), {1, 5})

        # Bottom-right corner (24) has 2 neighbors
        neighbors = get_neighbors(24)
        self.assertEqual(set(neighbors), {23, 19})

    def test_edge_neighbors(self):
        # Top edge middle (2) has 3 neighbors
        neighbors = get_neighbors(2)
        self.assertEqual(set(neighbors), {1, 3, 7})

        # Left edge middle (10) has 3 neighbors
        neighbors = get_neighbors(10)
        self.assertEqual(set(neighbors), {5, 15, 11})

    def test_center_neighbors(self):
        # Center (12) has 4 neighbors
        neighbors = get_neighbors(12)
        self.assertEqual(set(neighbors), {7, 17, 11, 13})


class TestBoardToCanonical3D(unittest.TestCase):
    """Test the 3D canonical board representation."""

    def test_empty_board(self):
        board_flat = np.zeros(NUM_POSITIONS)
        planes = board_to_canonical_3d(board_flat, player=1)

        self.assertEqual(planes.shape, (3, BOARD_SIZE, BOARD_SIZE))
        self.assertEqual(np.sum(planes[0]), 0)  # no current player stones
        self.assertEqual(np.sum(planes[1]), 0)  # no opponent stones
        self.assertEqual(np.sum(planes[2]), NUM_POSITIONS)  # all empty

    def test_with_stones_black_perspective(self):
        board_flat = np.zeros(NUM_POSITIONS)
        board_flat[0] = 1   # black stone
        board_flat[1] = -1  # white stone

        planes = board_to_canonical_3d(board_flat, player=1)

        self.assertEqual(planes[0, 0, 0], 1)  # current player (black)
        self.assertEqual(planes[1, 0, 1], 1)  # opponent (white)
        self.assertEqual(planes[2, 0, 0], 0)  # not empty
        self.assertEqual(planes[2, 0, 1], 0)  # not empty
        self.assertEqual(planes[2, 0, 2], 1)  # empty

    def test_with_stones_white_perspective(self):
        board_flat = np.zeros(NUM_POSITIONS)
        board_flat[0] = 1   # black stone
        board_flat[1] = -1  # white stone

        planes = board_to_canonical_3d(board_flat, player=-1)

        # From white's perspective, white is "current player"
        self.assertEqual(planes[0, 0, 1], 1)  # current player (white)
        self.assertEqual(planes[1, 0, 0], 1)  # opponent (black)


class TestGoBasics(unittest.TestCase):
    """Test basic Go class functionality."""

    def test_initial_state(self):
        game = Go()
        self.assertEqual(len(game.state), NUM_POSITIONS + 2)
        self.assertTrue(np.all(game.get_board(game.state) == 0))
        self.assertEqual(game.get_ko_point(game.state), -1)
        self.assertEqual(game.get_consecutive_passes(game.state), 0)

    def test_get_set_ko_point(self):
        game = Go()
        state = game.state.copy()
        game.set_ko_point(state, 12)
        self.assertEqual(game.get_ko_point(state), 12)

    def test_get_set_consecutive_passes(self):
        game = Go()
        state = game.state.copy()
        game.set_consecutive_passes(state, 1)
        self.assertEqual(game.get_consecutive_passes(state), 1)


class TestGroupFinding(unittest.TestCase):
    """Test group and liberty finding."""

    def test_single_stone_liberties(self):
        game = Go()
        state = game.state.copy()
        board = game.get_board(state)

        # Place a stone in center
        board[12] = 1
        group, liberties = game.find_group(board, 12)
        self.assertEqual(group, {12})
        self.assertEqual(liberties, 4)  # center has 4 neighbors

    def test_corner_stone_liberties(self):
        game = Go()
        state = game.state.copy()
        board = game.get_board(state)

        # Place a stone in corner
        board[0] = 1
        group, liberties = game.find_group(board, 0)
        self.assertEqual(group, {0})
        self.assertEqual(liberties, 2)  # corner has 2 neighbors

    def test_connected_group(self):
        game = Go()
        state = game.state.copy()
        board = game.get_board(state)

        # Create a horizontal line of 3 stones
        board[11] = 1
        board[12] = 1
        board[13] = 1

        group, liberties = game.find_group(board, 12)
        self.assertEqual(group, {11, 12, 13})
        self.assertEqual(liberties, 8)  # each end has 3, middle contributes 2

    def test_surrounded_stone(self):
        game = Go()
        state = game.state.copy()
        board = game.get_board(state)

        # Place black stone in center
        board[12] = 1
        # Surround with white stones
        board[7] = -1   # above
        board[17] = -1  # below
        board[11] = -1  # left
        board[13] = -1  # right

        group, liberties = game.find_group(board, 12)
        self.assertEqual(group, {12})
        self.assertEqual(liberties, 0)


class TestCaptures(unittest.TestCase):
    """Test stone capture mechanics."""

    def test_capture_single_stone(self):
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Place white stone in corner
        board[0] = -1
        # Surround with black (just need 2 stones for corner)
        board[1] = 1
        board[5] = 1

        # White has 0 liberties, should be captured
        captured = game.capture_dead_stones(board, player=1)
        self.assertEqual(captured, 1)
        self.assertEqual(board[0], 0)  # stone removed

    def test_capture_group(self):
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Place white group in corner (2 stones)
        board[0] = -1
        board[1] = -1
        # Surround with black
        board[2] = 1
        board[5] = 1
        board[6] = 1

        captured = game.capture_dead_stones(board, player=1)
        self.assertEqual(captured, 2)
        self.assertEqual(board[0], 0)
        self.assertEqual(board[1], 0)

    def test_no_capture_with_liberties(self):
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Place white stone in corner
        board[0] = -1
        # Only partially surround
        board[1] = 1

        captured = game.capture_dead_stones(board, player=1)
        self.assertEqual(captured, 0)
        self.assertEqual(board[0], -1)  # stone remains


class TestSuicide(unittest.TestCase):
    """Test suicide rule."""

    def test_suicide_single_stone(self):
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Create a surrounded empty point in corner
        board[1] = -1
        board[5] = -1

        # Black playing at 0 would be suicide
        self.assertTrue(game.is_suicide(board, 0, player=1))

    def test_suicide_in_eye(self):
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Create black eye in corner area
        # Black stones surrounding position 0
        board[1] = 1
        board[5] = 1

        # White playing at 0 would be suicide
        self.assertTrue(game.is_suicide(board, 0, player=-1))

    def test_not_suicide_with_liberty(self):
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Partially surrounded position
        board[1] = -1  # only one side blocked

        # Black playing at 0 has a liberty at position 5
        self.assertFalse(game.is_suicide(board, 0, player=1))

    def test_not_suicide_when_capturing(self):
        """Playing a stone that captures opponent is not suicide."""
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # White stone in corner about to be captured
        board[0] = -1
        # Black partially surrounds
        board[1] = 1

        # Black plays at 5 to complete the capture - not suicide
        self.assertFalse(game.is_suicide(board, 5, player=1))


class TestKo(unittest.TestCase):
    """Test ko rule."""

    def test_ko_point_blocks_move(self):
        game = Go()
        state = game.state.copy()

        # Set ko point to position 12
        game.set_ko_point(state, 12)

        # Playing at ko point should be invalid
        self.assertTrue(game.would_be_ko(game.get_board(state), 12, 1, 12))
        self.assertFalse(game.is_valid_move(state, 12, player=1))

    def test_ko_created_after_capture(self):
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Set up a proper ko position where capturing creates ko
        # For ko: after capture, the capturing stone must have exactly 1 liberty
        # Pattern (@ = where black will play):
        #   . X O .
        #   X O @ O    <-- black plays at position 7, captures white at 6
        #   . X O .
        #
        # Positions:
        #   0  1  2  3
        #   5  6  7  8
        #   10 11 12 13
        #
        # After capture, black at 7 has neighbors: 2(W), 6(empty), 8(W), 12(W)
        # So black at 7 has exactly 1 liberty (position 6) -> ko!

        board[1] = 1    # black
        board[2] = -1   # white (surrounds position 7)
        board[5] = 1    # black
        board[6] = -1   # white (will be captured)
        board[8] = -1   # white (surrounds position 7)
        board[11] = 1   # black
        board[12] = -1  # white (surrounds position 7)

        # Black plays at 7, capturing white at 6
        new_state = game.apply_move(state, 7, player=1)

        # Now there should be a ko point at 6
        ko_point = game.get_ko_point(new_state)
        self.assertEqual(ko_point, 6)

    def test_ko_cleared_after_other_move(self):
        """Ko point should be cleared when playing elsewhere."""
        game = Go()
        state = game.state.copy()

        # Set a ko point
        game.set_ko_point(state, 6)

        # Play at a different position
        new_state = game.apply_move(state, 20, player=1)

        # Ko should be cleared (set to -1)
        self.assertEqual(game.get_ko_point(new_state), -1)


class TestValidMoves(unittest.TestCase):
    """Test valid move detection."""

    def test_pass_always_valid(self):
        game = Go()
        self.assertTrue(game.is_valid_move(game.state, PASS_ACTION, player=1))
        self.assertTrue(game.is_valid_move(game.state, PASS_ACTION, player=-1))

    def test_occupied_invalid(self):
        game = Go()
        state = game.state.copy()
        state[12] = 1  # place black stone

        self.assertFalse(game.is_valid_move(state, 12, player=1))
        self.assertFalse(game.is_valid_move(state, 12, player=-1))

    def test_empty_valid(self):
        game = Go()
        self.assertTrue(game.is_valid_move(game.state, 12, player=1))

    def test_get_valid_moves_empty_board(self):
        game = Go()
        valid = game.get_valid_moves(game.state)

        # All positions should be valid on empty board
        self.assertEqual(np.sum(valid[:NUM_POSITIONS]), NUM_POSITIONS)
        self.assertEqual(valid[PASS_ACTION], 1)
        self.assertEqual(np.sum(valid), ACTION_SIZE)

    def test_get_valid_moves_with_stones(self):
        game = Go()
        state = game.state.copy()
        state[12] = 1  # place stone

        valid = game.get_valid_moves(state)

        self.assertEqual(valid[12], 0)  # occupied position invalid
        self.assertEqual(np.sum(valid[:NUM_POSITIONS]), NUM_POSITIONS - 1)


class TestApplyMove(unittest.TestCase):
    """Test move application."""

    def test_place_stone(self):
        game = Go()
        state = game.state.copy()

        new_state = game.apply_move(state, 12, player=1)

        self.assertEqual(new_state[12], 1)
        self.assertEqual(game.get_consecutive_passes(new_state), 0)
        self.assertEqual(game.get_ko_point(new_state), -1)

    def test_pass_increments_counter(self):
        game = Go()
        state = game.state.copy()

        new_state = game.apply_move(state, PASS_ACTION, player=1)
        self.assertEqual(game.get_consecutive_passes(new_state), 1)

        new_state2 = game.apply_move(new_state, PASS_ACTION, player=-1)
        self.assertEqual(game.get_consecutive_passes(new_state2), 2)

    def test_stone_resets_pass_counter(self):
        game = Go()
        state = game.state.copy()

        # Pass once
        state = game.apply_move(state, PASS_ACTION, player=1)
        self.assertEqual(game.get_consecutive_passes(state), 1)

        # Place stone
        state = game.apply_move(state, 12, player=-1)
        self.assertEqual(game.get_consecutive_passes(state), 0)

    def test_apply_move_does_not_modify_original(self):
        game = Go()
        state = game.state.copy()
        original = state.copy()

        game.apply_move(state, 12, player=1)

        # Original should be unchanged
        np.testing.assert_array_equal(state, original)


class TestGameEnd(unittest.TestCase):
    """Test game ending conditions."""

    def test_game_not_ended_initially(self):
        game = Go()
        self.assertFalse(game.game_ended(game.state))

    def test_game_ends_two_passes(self):
        game = Go()
        state = game.state.copy()

        state = game.apply_move(state, PASS_ACTION, player=1)
        self.assertFalse(game.game_ended(state))

        state = game.apply_move(state, PASS_ACTION, player=-1)
        self.assertTrue(game.game_ended(state))

    def test_game_not_ended_after_one_pass(self):
        game = Go()
        state = game.state.copy()

        state = game.apply_move(state, PASS_ACTION, player=1)
        state = game.apply_move(state, 12, player=-1)  # stone placement

        self.assertFalse(game.game_ended(state))


class TestScoring(unittest.TestCase):
    """Test territory counting and scoring."""

    def test_empty_board_scoring(self):
        game = Go()
        board = np.zeros(NUM_POSITIONS)

        black_score, white_score = game.count_territory(board)

        # Empty board: no stones, all territory is neutral (dame)
        self.assertEqual(black_score, 0)
        self.assertEqual(white_score, 0)

    def test_stones_counted(self):
        game = Go()
        board = np.zeros(NUM_POSITIONS)

        # Place some stones
        board[0] = 1   # black
        board[1] = 1   # black
        board[24] = -1  # white

        black_score, white_score = game.count_territory(board)

        # At minimum, stones should be counted
        self.assertGreaterEqual(black_score, 2)
        self.assertGreaterEqual(white_score, 1)

    def test_surrounded_territory(self):
        game = Go()
        board = np.zeros(NUM_POSITIONS)

        # Create a completely surrounded empty point
        # . X .
        # X . X
        # . X .
        board[1] = 1   # top
        board[5] = 1   # left
        board[7] = 1   # right
        board[11] = 1  # bottom

        black_score, white_score = game.count_territory(board)

        # Black has 4 stones + position 6 is surrounded (should be territory)
        # Total should be at least 5
        self.assertGreaterEqual(black_score, 5)

    def test_winner_with_komi(self):
        game = Go()
        state = game.state.copy()

        # End game with empty board
        game.set_consecutive_passes(state, 2)

        # With 2.5 komi, white should win on empty board
        winner = game.get_winner(state)
        self.assertEqual(winner, -1)  # white wins due to komi


class TestFullGameScenarios(unittest.TestCase):
    """Test complete game scenarios."""

    def test_capture_in_corner(self):
        """Test that corner capture works correctly."""
        game = Go()
        state = game.state.copy()

        # Black plays corner
        state = game.apply_move(state, 0, player=1)
        # White surrounds
        state = game.apply_move(state, 1, player=-1)
        state = game.apply_move(state, 5, player=-1)

        # Black stone should be captured
        board = game.get_board(state)
        self.assertEqual(board[0], 0)  # captured
        self.assertEqual(board[1], -1)  # white remains
        self.assertEqual(board[5], -1)  # white remains

    def test_capture_removes_opponent_first(self):
        """Capturing happens before checking own liberties."""
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Setup: mutual atari situation
        # White stone at 0, black at 1 and 5
        # If black plays somewhere that captures white first, it's valid

        board[0] = -1   # white in corner
        board[1] = 1    # black

        # Black plays at 5 to capture white - this is valid even though
        # without the capture black would have no liberties
        self.assertFalse(game.is_suicide(board, 5, player=1))

    def test_atari_and_escape(self):
        """Test that a group in atari can escape by extending."""
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Setup: Black stone with one liberty
        # . X .
        # X B X
        # . . .
        # B = black at position 6

        board[6] = 1   # black
        board[1] = -1  # white
        board[5] = -1  # white
        board[7] = -1  # white
        # Position 11 is black's last liberty

        # Black extends to escape atari
        state = game.apply_move(state, 11, player=1)

        # Black group should survive (has new liberties)
        board = game.get_board(state)
        self.assertEqual(board[6], 1)
        self.assertEqual(board[11], 1)

    def test_two_eyes_live(self):
        """Test that a group with two eyes cannot be captured."""
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Create black group with two eyes
        # X X X X X
        # X . X . X
        # X X X X X

        # Top row (0-4)
        for i in range(5):
            board[i] = 1

        # Middle row with two eyes (5-9)
        board[5] = 1
        # board[6] = 0  # eye 1
        board[7] = 1
        # board[8] = 0  # eye 2
        board[9] = 1

        # Bottom row (10-14)
        for i in range(10, 15):
            board[i] = 1

        # White cannot fill either eye without dying (suicide)
        self.assertTrue(game.is_suicide(board, 6, player=-1))
        self.assertTrue(game.is_suicide(board, 8, player=-1))


class TestCanonicalForm(unittest.TestCase):
    """Test canonical form transformations."""

    def test_next_state_flips_perspective(self):
        game = Go()
        state = game.state.copy()

        # Place a stone as black (player 1)
        action = np.zeros(ACTION_SIZE)
        action[12] = 1  # play at center

        new_state = game.get_next_state_from_next_player_prespective(state, action, player=1)

        # From new player's perspective, the stone should be opponent's (-1)
        self.assertEqual(new_state[12], -1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_all_positions_filled(self):
        """Test behavior when board is nearly full."""
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Fill board alternating (this won't be a legal game, just testing)
        for i in range(NUM_POSITIONS):
            board[i] = 1 if i % 2 == 0 else -1

        valid = game.get_valid_moves(state, player=1)

        # Only pass should be valid
        self.assertEqual(np.sum(valid[:NUM_POSITIONS]), 0)
        self.assertEqual(valid[PASS_ACTION], 1)

    def test_action_index_bounds(self):
        game = Go()

        self.assertFalse(game.is_valid_move(game.state, -1, player=1))
        self.assertFalse(game.is_valid_move(game.state, NUM_POSITIONS + 10, player=1))

    def test_check_one_hot_action(self):
        game = Go()

        # Valid one-hot action
        action = np.zeros(ACTION_SIZE)
        action[12] = 1
        self.assertTrue(game.check_if_action_is_valid(game.state, action, player=1))

        # Invalid: multiple 1s
        action2 = np.zeros(ACTION_SIZE)
        action2[12] = 1
        action2[13] = 1
        self.assertFalse(game.check_if_action_is_valid(game.state, action2, player=1))


class TestPlayMethod(unittest.TestCase):
    """Test the play() method that combines move application with result checking."""

    def test_play_returns_correct_tuple(self):
        game = Go()
        state = game.state.copy()

        new_state, result, next_player = game.play(state, player=1, action_index=12)

        self.assertEqual(new_state[12], 1)
        self.assertIsNone(result)  # game not ended
        self.assertEqual(next_player, -1)

    def test_play_detects_game_end(self):
        game = Go()
        state = game.state.copy()

        # First pass
        state, result, next_player = game.play(state, player=1, action_index=PASS_ACTION)
        self.assertIsNone(result)

        # Second pass ends game
        state, result, next_player = game.play(state, player=-1, action_index=PASS_ACTION)
        self.assertIsNotNone(result)  # game ended


class TestLadder(unittest.TestCase):
    """Test ladder (shicho) scenarios."""

    def test_ladder_escape_at_edge(self):
        """A stone being chased in a ladder should be capturable if no escape."""
        game = Go()
        state = game.state.copy()
        board = state[:NUM_POSITIONS]

        # Set up a simple ladder starting position
        # White stone at (1,1) = index 6, black chasing
        board[6] = -1   # white
        board[5] = 1    # black
        board[1] = 1    # black

        # White has 2 liberties (7 and 11)
        group, liberties = game.find_group(board, 6)
        self.assertEqual(liberties, 2)


class TestSeki(unittest.TestCase):
    """Test seki (mutual life) scenarios - simplified."""

    def test_shared_liberties(self):
        """Groups sharing liberties may both live in seki."""
        game = Go()
        board = np.zeros(NUM_POSITIONS)

        # Two groups sharing liberties (simplified)
        board[0] = 1   # black
        board[1] = -1  # white

        # Both have liberties
        b_group, b_lib = game.find_group(board, 0)
        w_group, w_lib = game.find_group(board, 1)

        self.assertGreater(b_lib, 0)
        self.assertGreater(w_lib, 0)


class TestRender(unittest.TestCase):
    """Test the render function doesn't crash."""

    def test_render_empty_board(self):
        game = Go()
        # Just make sure it doesn't raise an exception
        try:
            game.render(game.state)
        except Exception as e:
            self.fail(f"render() raised {e}")

    def test_render_with_stones(self):
        game = Go()
        state = game.state.copy()
        state[12] = 1
        state[6] = -1

        try:
            game.render(state)
        except Exception as e:
            self.fail(f"render() raised {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
