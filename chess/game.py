"""
Thin chess environment wrapping python-chess.

State is a chess.Board (not a flat array as in tictactoe). The MCTS copies the
board when descending into children, so callers should treat boards returned by
apply_move as fresh objects.

Result convention (from White's perspective):
    +1  White wins
    -1  Black wins
     0  draw
"""
import chess

import encoding


class ChessGame:
    def get_initial_board(self):
        return chess.Board()

    def get_valid_moves(self, board):
        """Length-ACTION_SIZE float mask of legal moves."""
        return encoding.legal_policy_mask(board)

    def legal_move_index_map(self, board):
        """Dict {action_index: chess.Move} for the current position."""
        return encoding.legal_move_index_map(board)

    def apply_move(self, board, move):
        """Return a new board with `move` applied (does not mutate input)."""
        next_board = board.copy(stack=False)
        next_board.push(move)
        return next_board

    def apply_action_index(self, board, action_index):
        move = encoding.index_to_move(action_index, board)
        return self.apply_move(board, move)

    def is_terminal(self, board):
        return board.is_game_over(claim_draw=True)

    def get_result(self, board):
        """+1 White win, -1 Black win, 0 draw. Assumes terminal board."""
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1
        if result == "0-1":
            return -1
        return 0

    def to_planes(self, board):
        return encoding.board_to_planes(board)
