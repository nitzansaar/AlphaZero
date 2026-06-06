"""
MCTS for chess. Same PUCT / backup / temperature logic as the tictactoe
version, but each node owns a real chess.Board instead of a flat array, and
state transitions go through python-chess instead of "negate the array".

The policy/value network is a callable: network(board) -> (value, policy)
  - value:  float in [-1, 1] from the perspective of the side to move at `board`
  - policy: length-ACTION_SIZE np.array, already masked to legal moves and
            normalized to sum to 1.
"""
import math
import numpy as np

from config import Config as cfg


class Node:
    def __init__(self, board, player, prior_prob, parent=None, action_index=None):
        self.board = board            # chess.Board at this node
        self.player = player          # +1 if White to move here, -1 if Black
        self.prior_probs_P = prior_prob
        self.total_visits_N = 0
        self.total_action_value_W = 0.0
        self.mean_action_value_Q = 0.0
        self.children = {}            # action_index -> Node
        self.parent = parent
        self.action_index = action_index

    def is_leaf_node(self):
        return len(self.children) == 0

    def expand(self, action_probs, game):
        """Create one child per legal move with its prior probability."""
        child_player = -self.player
        for action_index, move in game.legal_move_index_map(self.board).items():
            child_board = game.apply_move(self.board, move)
            self.children[action_index] = Node(
                child_board, child_player, action_probs[action_index],
                parent=self, action_index=action_index,
            )

    def select_best_child(self):
        best_score = float("-inf")
        best_index = None
        best_child = None
        sqrt_N = math.sqrt(self.total_visits_N)
        for action_index, child in self.children.items():
            # child.Q is from the child mover's (opponent's) perspective, so we
            # negate it to score from the current node's perspective.
            q = -child.mean_action_value_Q
            u = cfg.MCTS_UCB_C * child.prior_probs_P * sqrt_N / (1 + child.total_visits_N)
            score = q + u
            if score > best_score:
                best_score = score
                best_index = action_index
                best_child = child
        return best_index, best_child


class MonteCarloTreeSearch:
    def __init__(self, game, policy_value_network):
        self.game = game
        self.policy_value_network = policy_value_network

    def init_root_node(self):
        board = self.game.get_initial_board()
        root = Node(board, player=1, prior_prob=0.0)
        return root

    def make_root(self, board):
        # White to move -> +1, Black to move -> -1.
        player = 1 if board.turn else -1
        return Node(board, player=player, prior_prob=0.0)

    def _add_dirichlet_noise(self, root):
        indices = list(root.children.keys())
        if not indices:
            return
        noise = np.random.dirichlet([cfg.DIRICHLET_ALPHA] * len(indices))
        eps = cfg.DIRICHLET_EPSILON
        for i, action_index in enumerate(indices):
            child = root.children[action_index]
            child.prior_probs_P = (1 - eps) * child.prior_probs_P + eps * noise[i]

    def backup(self, path, leaf_value, leaf_player):
        for node in reversed(path):
            node.total_visits_N += 1
            signed = leaf_value if node.player == leaf_player else -leaf_value
            node.total_action_value_W += signed
            node.mean_action_value_Q = node.total_action_value_W / node.total_visits_N

    def run_simulation(self, root_node, num_simulations=None, add_noise=True):
        if num_simulations is None:
            num_simulations = cfg.NUM_SIMULATIONS

        # Expand the root if needed (first time it is used).
        if root_node.is_leaf_node() and not self.game.is_terminal(root_node.board):
            _, probs = self.policy_value_network(root_node.board)
            root_node.expand(probs, self.game)

        if add_noise:
            self._add_dirichlet_noise(root_node)

        for _ in range(num_simulations):
            path = [root_node]
            node = root_node
            while not node.is_leaf_node():
                _, node = node.select_best_child()
                path.append(node)

            leaf = node
            if self.game.is_terminal(leaf.board):
                # Result is from White's perspective; convert to leaf mover view.
                result = self.game.get_result(leaf.board)
                leaf_value = result * leaf.player
            else:
                value, probs = self.policy_value_network(leaf.board)
                leaf.expand(probs, self.game)
                leaf_value = value

            self.backup(path, leaf_value, leaf.player)

        return root_node

    def select_move(self, node, temperature=1.0):
        """Return (action_index, child_node, action_probs over ACTION_SIZE)."""
        indices = list(node.children.keys())
        visits = np.array(
            [node.children[i].total_visits_N for i in indices], dtype=np.float64
        )

        action_probs = np.zeros(cfg.ACTION_SIZE, dtype=np.float32)
        total = visits.sum()
        if total > 0:
            action_probs[indices] = (visits / total).astype(np.float32)

        if temperature <= 1e-3 or total == 0:
            chosen = indices[int(np.argmax(visits))]
        else:
            dist = visits ** (1.0 / temperature)
            dist = dist / dist.sum()
            chosen = int(np.random.choice(indices, p=dist))

        return chosen, node.children[chosen], action_probs
