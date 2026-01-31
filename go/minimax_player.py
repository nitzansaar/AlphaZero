"""
Minimax player with alpha-beta pruning for 5x5 Go.
"""

import numpy as np
from game import Go, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE


class MinimaxPlayer:
    """
    Minimax player with alpha-beta pruning.

    Uses a heuristic evaluation function based on:
    - Territory control
    - Stone count
    - Liberties (breathing room)
    - Capture threats
    """

    def __init__(self, max_depth=3):
        self.game = Go()
        self.max_depth = max_depth
        self.nodes_searched = 0

    def get_liberties(self, board, row, col):
        """Count liberties (empty adjacent points) for a stone."""
        liberties = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr, nc] == 0:
                    liberties += 1
        return liberties

    def get_group_liberties(self, board, row, col, visited=None):
        """Count total liberties for a connected group of stones."""
        if visited is None:
            visited = set()

        color = board[row, col]
        if color == 0:
            return 0

        liberties = set()
        stack = [(row, col)]
        group = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr, nc] == 0:
                        liberties.add((nr, nc))
                    elif board[nr, nc] == color and (nr, nc) not in group:
                        stack.append((nr, nc))

        return len(liberties)

    def evaluate(self, state, player):
        """
        Evaluate position from player's perspective.
        Returns value in range [-1, 1].

        Args:
            state: Game state
            player: Player to evaluate for (1=Black, -1=White)
        """
        board = state[:NUM_POSITIONS].reshape(BOARD_SIZE, BOARD_SIZE)

        # Check for game end
        if self.game.game_ended(state):
            winner = self.game.get_winner(state, perspective=1)
            if winner == player:
                return 1.0
            elif winner == -player:
                return -1.0
            else:
                return 0.0

        score = 0.0

        # Stone count (simple)
        my_stones = np.sum(board == player)
        opp_stones = np.sum(board == -player)
        score += (my_stones - opp_stones) * 0.1

        # Territory estimation using flood fill from empty points
        my_territory = 0
        opp_territory = 0
        visited = set()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] == 0 and (r, c) not in visited:
                    # Flood fill to find connected empty region
                    region = []
                    borders = set()
                    stack = [(r, c)]

                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in visited:
                            continue
                        if board[cr, cc] == 0:
                            visited.add((cr, cc))
                            region.append((cr, cc))
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                                    if board[nr, nc] == 0 and (nr, nc) not in visited:
                                        stack.append((nr, nc))
                                    elif board[nr, nc] != 0:
                                        borders.add(board[nr, nc])

                    # If region is bordered by only one color, it's that color's territory
                    if len(borders) == 1:
                        color = list(borders)[0]
                        if color == player:
                            my_territory += len(region)
                        else:
                            opp_territory += len(region)

        score += (my_territory - opp_territory) * 0.15

        # Liberty count (more liberties = safer stones)
        my_liberties = 0
        opp_liberties = 0
        counted = set()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) not in counted and board[r, c] != 0:
                    libs = self.get_group_liberties(board, r, c)
                    if board[r, c] == player:
                        my_liberties += libs
                    else:
                        opp_liberties += libs
                    # Mark group as counted (simplified)
                    counted.add((r, c))

        score += (my_liberties - opp_liberties) * 0.05

        # Stones in atari (1 liberty) - bad for owner
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] != 0:
                    libs = self.get_group_liberties(board, r, c)
                    if libs == 1:
                        if board[r, c] == player:
                            score -= 0.3  # My stone in atari is bad
                        else:
                            score += 0.3  # Opponent stone in atari is good

        # Center control bonus
        center = BOARD_SIZE // 2
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] == player:
                    dist = abs(r - center) + abs(c - center)
                    score += (2 - dist) * 0.02
                elif board[r, c] == -player:
                    dist = abs(r - center) + abs(c - center)
                    score -= (2 - dist) * 0.02

        # Komi adjustment (White gets 2.5 points)
        if player == -1:  # White
            score += 0.1  # Small bonus for komi
        else:
            score -= 0.1

        # Clamp to [-1, 1]
        return np.clip(score, -1.0, 1.0)

    def get_valid_actions(self, state, player):
        """Get list of valid actions for player."""
        # State is in absolute form, need to convert for valid move check
        valid = self.game.get_valid_moves(state, player)
        return [i for i in range(ACTION_SIZE) if valid[i] == 1]

    def apply_action(self, state, action, player):
        """Apply action and return new state."""
        new_state = self.game.apply_move(state, action, player)
        return new_state

    def alphabeta(self, state, depth, alpha, beta, maximizing_player, player):
        """
        Alpha-beta pruning minimax search.

        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: True if maximizing, False if minimizing
            player: Current player to move (1 or -1)

        Returns:
            (value, best_action)
        """
        self.nodes_searched += 1

        # Terminal or depth limit
        if depth == 0 or self.game.game_ended(state):
            # Evaluate from the perspective of the original maximizing player
            eval_player = player if maximizing_player else -player
            return self.evaluate(state, eval_player), None

        valid_actions = self.get_valid_actions(state, player)

        if not valid_actions:
            valid_actions = [PASS_ACTION]

        best_action = valid_actions[0]

        if maximizing_player:
            max_eval = float('-inf')
            for action in valid_actions:
                new_state = self.apply_action(state, action, player)
                eval_score, _ = self.alphabeta(
                    new_state, depth - 1, alpha, beta, False, -player
                )
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in valid_actions:
                new_state = self.apply_action(state, action, player)
                eval_score, _ = self.alphabeta(
                    new_state, depth - 1, alpha, beta, True, -player
                )
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval, best_action

    def get_action(self, state, player):
        """
        Get the best action for the given state and player.

        Args:
            state: Game state in absolute form (Black=1, White=-1)
            player: Current player (1 or -1)

        Returns:
            Best action index
        """
        self.nodes_searched = 0
        _, action = self.alphabeta(
            state, self.max_depth, float('-inf'), float('inf'), True, player
        )
        return action


class IterativeDeepeningMinimaxPlayer(MinimaxPlayer):
    """
    Minimax with iterative deepening for better time management.
    """

    def __init__(self, max_depth=4, time_limit=None):
        super().__init__(max_depth)
        self.time_limit = time_limit

    def get_action(self, state, player):
        """Get best action using iterative deepening."""
        best_action = None

        for depth in range(1, self.max_depth + 1):
            self.nodes_searched = 0
            _, action = self.alphabeta(
                state, depth, float('-inf'), float('inf'), True, player
            )
            if action is not None:
                best_action = action

        return best_action


if __name__ == "__main__":
    # Quick test
    game = Go()
    player = MinimaxPlayer(max_depth=3)

    state = np.zeros(NUM_POSITIONS + 2)
    state[NUM_POSITIONS] = -1  # No ko
    state[NUM_POSITIONS + 1] = 0  # No passes

    print("Testing Minimax Player on empty board...")
    action = player.get_action(state, 1)
    print(f"Best move for Black: ({action // BOARD_SIZE}, {action % BOARD_SIZE})")
    print(f"Nodes searched: {player.nodes_searched}")
