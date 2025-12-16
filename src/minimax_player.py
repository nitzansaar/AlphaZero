import numpy as np


WIN_SCORE = 1_000_000


def _precompute_windows_5(board_size=9, k=5):
    windows = []
    n = board_size

    # Rows
    for r in range(n):
        for c in range(n - k + 1):
            windows.append([r * n + (c + i) for i in range(k)])

    # Cols
    for c in range(n):
        for r in range(n - k + 1):
            windows.append([(r + i) * n + c for i in range(k)])

    # Diagonals (top-left -> bottom-right)
    for r in range(n - k + 1):
        for c in range(n - k + 1):
            windows.append([(r + i) * n + (c + i) for i in range(k)])

    # Anti-diagonals (top-right -> bottom-left)
    for r in range(n - k + 1):
        for c in range(k - 1, n):
            windows.append([(r + i) * n + (c - i) for i in range(k)])

    return windows


WINDOWS_5 = _precompute_windows_5()


def _pattern_score_in_windows(board_flat, player):
    weights = {
        1: 1,
        2: 10,
        3: 200,
        4: 20_000,
        5: WIN_SCORE,
    }

    score = 0
    for window in WINDOWS_5:
        window_vals = board_flat[window]
        if np.any(window_vals == -player):
            continue
        count = int(np.sum(window_vals == player))
        if count == 0:
            continue
        score += weights.get(count, 0)
    return score


def _heuristic(board_flat, player):
    return _pattern_score_in_windows(board_flat, player) - 1.2 * _pattern_score_in_windows(board_flat, -player)


def _candidate_moves(board_flat, radius=1, max_candidates=24):
    empty = np.where(board_flat == 0)[0]
    if len(empty) == len(board_flat):
        return [40]  # center of 9x9

    occupied = np.where(board_flat != 0)[0]
    n = 9
    candidates = set()

    for idx in occupied:
        r, c = divmod(int(idx), n)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    cand = rr * n + cc
                    if board_flat[cand] == 0:
                        candidates.add(cand)

    candidates = list(candidates)
    if not candidates:
        return list(empty)

    # Simple ordering: prefer moves with more occupied neighbors + center bias.
    def move_key(move_idx):
        r, c = divmod(int(move_idx), n)
        center_bias = -abs(r - 4) - abs(c - 4)
        neighbor_count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < n and 0 <= cc < n and board_flat[rr * n + cc] != 0:
                    neighbor_count += 1
        return (neighbor_count, center_bias)

    candidates.sort(key=move_key, reverse=True)
    return candidates[:max_candidates]


class MinimaxPlayer:
    """
    Depth-limited minimax/alpha-beta opponent for 9x9 "5-in-a-row" TicTacToe.

    Intended as a stronger-than-random baseline, not a perfect solver.
    """

    def __init__(self, game, depth=2, radius=1, max_candidates=24):
        self.game = game
        self.depth = depth
        self.radius = radius
        self.max_candidates = max_candidates

    def _negamax(self, board_flat, player, depth, alpha, beta):
        winner = self.game.win_or_draw(board_flat)
        if winner is not None:
            if winner == 0:
                return 0
            return WIN_SCORE if winner == player else -WIN_SCORE

        if depth <= 0:
            return _heuristic(board_flat, player)

        best = -float("inf")
        moves = _candidate_moves(board_flat, radius=self.radius, max_candidates=self.max_candidates)

        for move in moves:
            board_flat[move] = player
            score = -self._negamax(board_flat, -player, depth - 1, -beta, -alpha)
            board_flat[move] = 0

            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best

    def get_action_index(self, board_flat, player):
        valid_moves = self.game.get_valid_moves(board_flat)
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) == 0:
            return None

        # Immediate win
        for move in valid_indices:
            board_flat[move] = player
            if self.game.win_or_draw(board_flat) == player:
                board_flat[move] = 0
                return int(move)
            board_flat[move] = 0

        # Immediate block
        for move in valid_indices:
            board_flat[move] = -player
            if self.game.win_or_draw(board_flat) == -player:
                board_flat[move] = 0
                return int(move)
            board_flat[move] = 0

        best_move = None
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        candidates = _candidate_moves(board_flat, radius=self.radius, max_candidates=self.max_candidates)
        candidates = [m for m in candidates if valid_moves[m] == 1]
        if not candidates:
            candidates = [int(m) for m in valid_indices]

        for move in candidates:
            board_flat[move] = player
            score = -self._negamax(board_flat, -player, self.depth - 1, -beta, -alpha)
            board_flat[move] = 0

            if score > best_score:
                best_score = score
                best_move = int(move)
            if score > alpha:
                alpha = score

        return best_move

    def get_action(self, board_flat, player):
        action_index = self.get_action_index(board_flat, player)
        if action_index is None:
            return None
        action = np.zeros_like(board_flat, dtype=np.float32)
        action[action_index] = 1.0
        return action

