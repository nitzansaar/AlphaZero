import numpy as np
from config import BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE, KOMI


def board_to_canonical_3d(board_flat, player):
    """
    Convert flat board state to canonical 3-plane representation.

    Args:
        board_flat: Flat array of NUM_POSITIONS values (from game.state)
        player: Current player (1 or -1)

    Returns:
        3-plane numpy array of shape (3, BOARD_SIZE, BOARD_SIZE):
        - Plane 0: Current player positions
        - Plane 1: Opponent positions
        - Plane 2: Empty positions
    """
    board_2d = np.array(board_flat[:NUM_POSITIONS]).reshape(BOARD_SIZE, BOARD_SIZE) # reshape the board to 5x5
    canonical = board_2d * player # transform the board to the current player's perspective

    planes = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    planes[0] = (canonical == 1).astype(np.float32)   # current player
    planes[1] = (canonical == -1).astype(np.float32)  # opponent
    planes[2] = (canonical == 0).astype(np.float32)   # empty

    return planes


def idx_to_coord(idx):
    """Convert flat index to (row, col) coordinates."""
    return idx // BOARD_SIZE, idx % BOARD_SIZE


def coord_to_idx(row, col):
    """Convert (row, col) coordinates to flat index."""
    return row * BOARD_SIZE + col


def get_neighbors(idx):
    """Get list of neighboring indices for a position."""
    row, col = idx_to_coord(idx)
    neighbors = []
    if row > 0:
        neighbors.append(coord_to_idx(row - 1, col))
    if row < BOARD_SIZE - 1:
        neighbors.append(coord_to_idx(row + 1, col))
    if col > 0:
        neighbors.append(coord_to_idx(row, col - 1))
    if col < BOARD_SIZE - 1:
        neighbors.append(coord_to_idx(row, col + 1))
    return neighbors


class Go:
    """
    Go game implementation (supports multiple board sizes).

    State representation:
    - Positions 0 to NUM_POSITIONS-1: Board state (0=empty, 1=black, -1=white)
    - Position NUM_POSITIONS: Ko point (-1 if no ko, else the forbidden position index)
    - Position NUM_POSITIONS+1: Consecutive passes count (0, 1, or 2)

    Actions:
    - 0 to NUM_POSITIONS-1: Place stone at that position
    - PASS_ACTION: Pass
    """

    def __init__(self):
        # State: 25 board positions + ko point + consecutive passes
        self.state = np.zeros(NUM_POSITIONS + 2)
        self.state[NUM_POSITIONS] = -1  # No ko point initially
        self.state[NUM_POSITIONS + 1] = 0  # No consecutive passes

    def get_board(self, state):
        """Extract just the board portion of state."""
        return state[:NUM_POSITIONS]

    def get_ko_point(self, state):
        """Get the ko point from state (-1 if none)."""
        return int(state[NUM_POSITIONS])

    def get_consecutive_passes(self, state):
        """Get consecutive pass count from state."""
        return int(state[NUM_POSITIONS + 1])

    def set_ko_point(self, state, ko_point):
        """Set the ko point in state."""
        state[NUM_POSITIONS] = ko_point

    def set_consecutive_passes(self, state, count):
        """Set consecutive passes in state."""
        state[NUM_POSITIONS + 1] = count

    def find_group(self, board, idx):
        """
        Find all stones connected to the stone at idx.
        Returns (group_indices, liberty_count).
        """
        if board[idx] == 0: # if the position is empty, return an empty group and 0 liberties
            return set(), 0

        color = board[idx]
        group = set()
        liberties = set()
        queue = [idx]

        while queue:
            current = queue.pop()
            if current in group:
                continue
            group.add(current)

            for neighbor in get_neighbors(current):
                if board[neighbor] == 0:
                    liberties.add(neighbor)
                elif board[neighbor] == color and neighbor not in group:
                    queue.append(neighbor)

        return group, len(liberties)

    def remove_group(self, board, group):
        """Remove a group of stones from the board."""
        for idx in group:
            board[idx] = 0

    def capture_dead_stones(self, board, player):
        """
        Remove any opponent groups with zero liberties.
        Returns the number of stones captured.
        """
        opponent = -player
        captured = 0
        visited = set()

        for idx in range(NUM_POSITIONS):
            if board[idx] == opponent and idx not in visited:
                group, liberties = self.find_group(board, idx)
                visited.update(group)
                if liberties == 0:
                    self.remove_group(board, group)
                    captured += len(group)

        return captured

    def is_suicide(self, board, idx, player):
        """
        Check if placing a stone at idx would be suicide.
        A move is suicide if it results in the placed stone's group having no liberties
        AND it doesn't capture any opponent stones.
        """
        # Make a copy and place the stone
        test_board = board.copy()
        test_board[idx] = player

        # First check if this captures any opponent stones
        opponent = -player
        for neighbor in get_neighbors(idx):
            if test_board[neighbor] == opponent:
                group, liberties = self.find_group(test_board, neighbor)
                if liberties == 0:
                    # This move captures stones, so it's not suicide
                    return False

        # Check if our group has liberties
        group, liberties = self.find_group(test_board, idx)
        return liberties == 0

    def would_be_ko(self, board, idx, player, ko_point):
        """Check if playing at idx would violate the ko rule."""
        return idx == ko_point

    def is_valid_move(self, state, action, player):
        """Check if an action is valid for the given player."""
        if action == PASS_ACTION:
            return True

        if action < 0 or action >= NUM_POSITIONS:
            return False

        board = self.get_board(state)
        ko_point = self.get_ko_point(state)

        # Position must be empty
        if board[action] != 0:
            return False

        # Check ko rule
        if self.would_be_ko(board, action, player, ko_point):
            return False

        # Check suicide rule
        if self.is_suicide(board, action, player):
            return False

        return True

    def get_valid_moves(self, state, player=1):
        """
        Get valid moves for the current player.
        Returns array of size ACTION_SIZE (26) with 1 for valid moves, 0 otherwise.
        """
        valid = np.zeros(ACTION_SIZE)

        for action in range(NUM_POSITIONS):
            if self.is_valid_move(state, action, player):
                valid[action] = 1

        # Pass is always valid
        valid[PASS_ACTION] = 1

        return valid

    def check_if_action_is_valid(self, state, action, player=1):
        """Check if a one-hot encoded action is valid."""
        action_index = np.where(action == 1)[0]
        if len(action_index) != 1:
            return False
        action_index = action_index[0]
        return self.is_valid_move(state, action_index, player)

    def apply_move(self, state, action, player):
        """
        Apply a move and return the new state.
        Does not modify the input state.
        """
        new_state = state.copy()
        board = new_state[:NUM_POSITIONS]

        if action == PASS_ACTION:
            # Pass move
            self.set_ko_point(new_state, -1)  # Clear ko
            self.set_consecutive_passes(new_state, self.get_consecutive_passes(state) + 1)
            return new_state

        # Reset consecutive passes on a stone placement
        self.set_consecutive_passes(new_state, 0)

        # Place the stone
        board[action] = player

        # Capture opponent stones
        captured = self.capture_dead_stones(board, player)

        # Determine ko point
        # Ko occurs when exactly one stone is captured and the capturing stone
        # has exactly one liberty (the just-captured position)
        ko_point = -1
        if captured == 1:
            group, liberties = self.find_group(board, action)
            if len(group) == 1 and liberties == 1:
                # Find the liberty (the captured position)
                for neighbor in get_neighbors(action):
                    if board[neighbor] == 0:
                        ko_point = neighbor
                        break

        self.set_ko_point(new_state, ko_point)

        return new_state

    def get_next_state_from_next_player_prespective(self, state, action, player):
        """
        Apply move and return state from next player's perspective.
        Board values are negated so current player is always +1.
        """
        action_idx = np.argmax(action) if isinstance(action, np.ndarray) and len(action) > 1 else action
        new_state = self.apply_move(state, action_idx, 1)  # Current player is always 1 in canonical form

        # Flip board perspective
        new_state[:NUM_POSITIONS] *= -1

        return new_state

    def count_territory(self, board):
        """
        Count territory for scoring using area scoring (Chinese rules).
        Returns (black_score, white_score).

        Area scoring: count stones + surrounded empty points.
        """
        black_score = 0
        white_score = 0

        # Count stones
        black_score += np.sum(board == 1)
        white_score += np.sum(board == -1)

        # Find empty regions and determine ownership
        visited = set()

        for idx in range(NUM_POSITIONS):
            if board[idx] == 0 and idx not in visited:
                # BFS to find connected empty region
                region = set()
                borders_black = False
                borders_white = False
                frontier = [idx]

                while frontier:
                    current = frontier.pop()
                    if current in region:
                        continue
                    if board[current] == 0:
                        region.add(current)
                        for neighbor in get_neighbors(current):
                            if neighbor not in region:
                                frontier.append(neighbor)
                    elif board[current] == 1:
                        borders_black = True
                    else:
                        borders_white = True

                visited.update(region)

                # Territory belongs to a player only if it borders only their stones
                if borders_black and not borders_white:
                    black_score += len(region)
                elif borders_white and not borders_black:
                    white_score += len(region)
                # If borders both or neither, it's neutral (dame)

        return black_score, white_score

    def game_ended(self, state):
        """Check if the game has ended (two consecutive passes)."""
        return self.get_consecutive_passes(state) >= 2

    def get_winner(self, state, perspective=1):
        """
        Determine the winner of the game.
        Returns 1 (black wins), -1 (white wins), or 0 (draw).
        Uses area scoring with komi for white.

        Args:
            state: Game state (may be in canonical form)
            perspective: Whose perspective the state is from (1=Black, -1=White).
                        In canonical form, current player's stones are 1.
                        If perspective=-1, the board is flipped to absolute form before scoring.
        """
        board = self.get_board(state).copy()

        # Convert from canonical to absolute form if needed
        # In canonical form: 1 = current player, -1 = opponent
        # In absolute form: 1 = Black, -1 = White
        if perspective == -1:
            board = board * -1

        black_score, white_score = self.count_territory(board)

        # Apply komi (compensation for white going second)
        white_score += KOMI

        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return -1
        else:
            return 0

    def winner(self, state, perspective=1):
        """
        Check game state and return result.
        Returns 1 (black wins), -1 (white wins), 0 (draw), or None (game ongoing).

        Args:
            state: Game state (may be in canonical form)
            perspective: Whose perspective the state is from (1=Black, -1=White)
        """
        if self.game_ended(state):
            return self.get_winner(state, perspective)
        return None

    def get_reward_for_next_player(self, state, player):
        """
        Get the game result for use in MCTS backup.

        Args:
            state: Game state in canonical form
            player: Whose perspective the state is from (1=Black, -1=White)

        Returns:
            1 (Black wins), -1 (White wins), 0 (draw), or None (game ongoing)
        """
        result = self.winner(state, perspective=player)
        return result

    def play(self, board_state, player, action_index, perspective=1):
        """
        Execute a move and return (new_state, result, next_player).

        Args:
            board_state: Current game state
            player: Player making the move (1 or -1)
            action_index: Action to take
            perspective: Whose perspective the state is from (default 1 for absolute form)
        """
        new_state = self.apply_move(board_state, action_index, player)
        # After the move, perspective shifts to the next player
        next_player = -player
        next_perspective = -perspective if perspective != 1 else 1
        result = self.winner(new_state, perspective=next_perspective)
        return new_state, result, next_player

    def render(self, state):
        """Print the board state for debugging."""
        board = self.get_board(state).reshape(BOARD_SIZE, BOARD_SIZE)
        symbols = {0: '.', 1: 'X', -1: 'O'}

        # Column headers
        print('  ' + ' '.join(str(i) for i in range(BOARD_SIZE)))

        for i in range(BOARD_SIZE):
            row_str = ' '.join(symbols[int(board[i, j])] for j in range(BOARD_SIZE))
            print(f'{i} {row_str}')

        ko = self.get_ko_point(state)
        passes = self.get_consecutive_passes(state)
        print(f'Ko point: {ko}, Consecutive passes: {passes}')


# Alias for backward compatibility
TicTacToe = Go
