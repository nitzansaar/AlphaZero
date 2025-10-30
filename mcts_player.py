import numpy as np
import neural_network
import board as b
import copy

class MCTSNode:
    """A node in the MCTS search tree."""
    
    def __init__(self, state, parent=None, prior=0, player=1):
        self.state = state  # Board state (numpy array)
        self.parent = parent
        self.children = {}  # Dict mapping moves to child nodes
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.player = player  # Which player's turn it is at this node
        
    def value(self):
        """Average value of this node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def is_expanded(self):
        """Check if this node has been expanded."""
        return len(self.children) > 0
    
    def select_child(self, c_puct=1.0):
        """Select child with highest UCB score."""
        best_score = -np.inf
        best_child = None
        best_move = None
        
        for move, child in self.children.items():
            score = self._ucb_score(child, c_puct)
            if score > best_score:
                best_score = score
                best_child = child
                best_move = move
                
        return best_move, best_child
    
    def _ucb_score(self, child, c_puct):
        """Calculate UCB score for a child node."""
        prior_score = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
        if child.visit_count > 0:
            # Value from opponent's perspective, so negate
            value_score = -child.value()
        else:
            value_score = 0
        return value_score + prior_score


class MCTS_Player:
    """AlphaGo Zero style player that runs MCTS with neural network guidance."""
    
    def __init__(self, type, name, nn_type, num_simulations=100, c_puct=1.0, temperature=1.0):
        """
        Args:
            type: 'x' or 'o'
            name: Player name
            nn_type: Neural network checkpoint to load
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for UCB
            temperature: Temperature for move selection (0 = greedy, 1 = stochastic)
        """
        self.type = b.Board.STR2INT_MAP[type]
        self.name = name
        self.nn_predictor = neural_network.nn_predictor(nn_type)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        
    def turn(self, board):
        """Select move using MCTS."""
        # Create root node
        root = MCTSNode(board.board.copy(), player=self.type)
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree using UCB
            while node.is_expanded() and not self._is_terminal(node.state):
                move, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Check if game is over
            winner = b.Board.winner(node.state)
            if winner != 0 or not self._has_legal_moves(node.state):
                # Terminal node
                value = self._get_terminal_value(node.state, root.player)
            else:
                # Expansion and evaluation
                value = self._expand_and_evaluate(node)
            
            # Backpropagation
            self._backpropagate(search_path, value, root.player)
        
        # Select move based on visit counts
        move = self._select_move_from_root(root)
        row, col = move
        return self.type, row, col
    
    def _expand_and_evaluate(self, node):
        """Expand node and return value estimate from neural network."""
        # Get neural network prediction
        pred_winner, prior_probs = self.nn_predictor.predict(node.state)
        
        # Value estimate (from current player's perspective at this node)
        # pred_winner is [p_loss, p_draw, p_win] for player 1
        if node.player == 1:
            value = pred_winner[2] - pred_winner[0]  # p_win - p_loss
        else:
            value = pred_winner[0] - pred_winner[2]  # p_loss - p_win (reversed)
        
        # Expand: add children for all legal moves
        legal_moves = self._get_legal_moves(node.state)
        
        # Normalize prior probabilities over legal moves
        prior_probs = np.maximum(prior_probs, 1e-8)  # Avoid zeros
        legal_priors = np.zeros(9)
        for move in legal_moves:
            move_idx = move[0] * 3 + move[1]
            legal_priors[move_idx] = prior_probs[move_idx]
        
        prior_sum = legal_priors.sum()
        if prior_sum > 0:
            legal_priors /= prior_sum
        else:
            # If all priors are zero, use uniform distribution
            for move in legal_moves:
                move_idx = move[0] * 3 + move[1]
                legal_priors[move_idx] = 1.0 / len(legal_moves)
        
        # Create child nodes
        next_player = -node.player
        for move in legal_moves:
            move_idx = move[0] * 3 + move[1]
            new_state = node.state.copy()
            new_state[move[0], move[1]] = node.player
            child = MCTSNode(new_state, parent=node, prior=legal_priors[move_idx], player=next_player)
            node.children[move] = child
        
        return value
    
    def _backpropagate(self, search_path, value, root_player):
        """Backpropagate value up the search path."""
        for node in reversed(search_path):
            node.visit_count += 1
            # Value is from root player's perspective
            # Negate if current node is opponent's turn
            node_value = value if node.player == root_player else -value
            node.value_sum += node_value
    
    def _select_move_from_root(self, root):
        """Select move from root based on visit counts and temperature."""
        if not root.children:
            # No valid moves (shouldn't happen in tic-tac-toe)
            legal_moves = self._get_legal_moves(root.state)
            return legal_moves[0]
        
        moves = list(root.children.keys())
        visit_counts = np.array([root.children[move].visit_count for move in moves])
        
        if self.temperature < 1e-6:
            # Greedy selection
            best_idx = np.argmax(visit_counts)
            return moves[best_idx]
        else:
            # Stochastic selection with temperature
            visit_counts_temp = np.power(visit_counts, 1.0 / self.temperature)
            probs = visit_counts_temp / visit_counts_temp.sum()
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
    
    def _get_legal_moves(self, state):
        """Get list of legal moves (empty positions)."""
        moves = []
        for row in range(3):
            for col in range(3):
                if state[row, col] == 0:
                    moves.append((row, col))
        return moves
    
    def _has_legal_moves(self, state):
        """Check if there are any legal moves."""
        return np.any(state == 0)
    
    def _is_terminal(self, state):
        """Check if state is terminal (game over)."""
        return b.Board.winner(state) != 0 or not self._has_legal_moves(state)
    
    def _get_terminal_value(self, state, player):
        """Get value of terminal state from player's perspective."""
        winner = b.Board.winner(state)
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        else:
            return 0.0
    
    def close(self):
        """Close the neural network predictor to free resources."""
        if hasattr(self, 'nn_predictor') and self.nn_predictor is not None:
            self.nn_predictor.close()
    
    def __del__(self):
        """Destructor to ensure resources are freed."""
        self.close()


def main():
    print("MCTS Player module")

if __name__ == '__main__':
    main()

