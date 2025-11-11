import numpy as np

def board_to_canonical_3d(board_flat, player):
    """
    Convert flat board state to canonical 3-plane representation.

    Args:
        board_flat: Flat array of 25 values (from game.state)
        player: Current player (1 or -1)

    Returns:
        3-plane numpy array of shape (3, 5, 5):
        - Plane 0: Current player positions
        - Plane 1: Opponent positions
        - Plane 2: Empty positions
    """
    # Reshape to 2D
    board_2d = np.array(board_flat).reshape(5, 5)

    # Make canonical (current player is always +1)
    canonical = board_2d * player

    # Create 3 planes
    planes = np.zeros((3, 5, 5), dtype=np.float32)
    planes[0] = (canonical == 1).astype(np.float32)   # current player
    planes[1] = (canonical == -1).astype(np.float32)  # opponent
    planes[2] = (canonical == 0).astype(np.float32)   # empty

    return planes

class TicTacToe:
    def __init__(self):

        self.state = np.zeros(25) # initial state is all zeros (5x5 board)
        
    def get_valid_moves(self,state):
        valid_actions = np.zeros(25)
        valid_actions[np.where(state==0)]=1
        return valid_actions
    def check_if_action_is_valid(self,state,action):
        valid_actions = self.get_valid_moves(state)
        action_index = np.where(action==1)[0]
        if len(action_index)!=1:
            return False
        action_index= action_index[0]
        if valid_actions[action_index]!=1:
            return False
        return True
    def get_next_state_from_next_player_prespective(self,state,action,player):
        next_state = state.copy()
        next_state[np.argmax(action)]=1
        return next_state *-1
    def win_or_draw(self,state):
        state = state.reshape(5, 5)

        # Check rows for 4 in a row
        for i in range(5):
            for j in range(2):  # positions 0-3 and 1-4
                row_sum = state[i, j:j+4].sum()
                if row_sum == 4:
                    return 1
                if row_sum == -4:
                    return -1

        # Check columns for 4 in a row
        for j in range(5):
            for i in range(2):  # positions 0-3 and 1-4
                col_sum = state[i:i+4, j].sum()
                if col_sum == 4:
                    return 1
                if col_sum == -4:
                    return -1

        # Check diagonals (top-left to bottom-right)
        for i in range(2):  # start row: 0, 1
            for j in range(2):  # start col: 0, 1
                diag_sum = sum(state[i+k, j+k] for k in range(4))
                if diag_sum == 4:
                    return 1
                if diag_sum == -4:
                    return -1

        # Check anti-diagonals (top-right to bottom-left)
        for i in range(2):  # start row: 0, 1
            for j in range(3, 5):  # start col: 3, 4
                anti_diag_sum = sum(state[i+k, j-k] for k in range(4))
                if anti_diag_sum == 4:
                    return 1
                if anti_diag_sum == -4:
                    return -1

        # Check draw
        if len(np.where(state==0)[0])==0:
            return 0
        return None
    def get_reward_for_next_player(self,state,player):
        winner = self.win_or_draw(state)
        if winner:
            if winner in [-1,1]:
                return -1 * player # return the reward for the next player
            return 0
        return winner
    def play(self,board_state,player,action_index):
        board_state[action_index] = player
        return board_state,self.win_or_draw(board_state),-1 * player