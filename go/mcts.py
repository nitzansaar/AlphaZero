import numpy as np
import math
from copy import copy
from config import Config as cfg
from game import NUM_POSITIONS, PASS_ACTION, ACTION_SIZE

class Node:
    def __init__(self, prior_prob, player, parent=None, action_index=None):
        self.state = None
        self.player = player
        self.total_visits_N = 0
        self.total_action_value_of_next_state_W = 0
        self.mean_action_value_of_next_state_Q = 0
        self.prior_probs_P = prior_prob
        self.children = {}
        self.parent = parent
        if action_index is not None:
            state = copy(parent.state)
            # Flip board perspective (only the board portion, not ko/passes)
            state[:NUM_POSITIONS] = state[:NUM_POSITIONS] * -1
            if action_index != PASS_ACTION:
                state[action_index] = -1
            self.state = copy(state)

    def set_state(self, state):
        self.state = state

    def expand(self, action_probs, player, parent):
        for i, action_prob in enumerate(action_probs):
            if action_prob != 0:
                self.children[i] = Node(action_prob, player, parent, i)

    def is_leaf_node(self):
        return len(self.children) == 0

    def select_best_child(self):
        best_uscore = float('-inf')
        best_child_index = None
        for i, child in self.children.items():
            psa = child.prior_probs_P
            Ns = self.total_visits_N
            Nsa = child.total_visits_N
            Cs = cfg.MCTS_UCB_C
            Q = child.mean_action_value_of_next_state_Q
            Uscore = Q + Cs * psa * math.sqrt(Ns) / (1 + Nsa)
            if best_uscore < Uscore:
                best_uscore = Uscore
                best_child_index = i
        return best_child_index, self.children[best_child_index]


class MonteCarloTreeSearch:
    def __init__(self, game, policy_value_network):
        self.game = game
        self.policy_value_network = policy_value_network

    def init_root_node(self):
        # State size: 25 board positions + ko point + consecutive passes = 27
        root_state = np.zeros(NUM_POSITIONS + 2)
        root_state[NUM_POSITIONS] = -1  # No ko point
        root_state[NUM_POSITIONS + 1] = 0  # No consecutive passes
        root_node = Node(prior_prob=0, player=1, action_index=None)
        root_node.set_state(root_state)
        return root_node

    def backup(self, mtc_steps, winner, player, value):
        for node in reversed(mtc_steps):
            node.total_visits_N += 1
            if winner is None:
                pass  # game not over, use nn estimated value
            elif winner == 0:
                value = 0
            else:
                value = -1 if winner == node.player else 1
            node.total_action_value_of_next_state_W = node.total_action_value_of_next_state_W + value
            node.mean_action_value_of_next_state_Q = node.total_action_value_of_next_state_W / node.total_visits_N

    def run_simulation(self, root_node, num_simulations=1600, player=1):
        root_state = root_node.state
        next_player = -1 * player
        value, action_probs = self.policy_value_network(root_state, player)
        valid_moves = self.game.get_valid_moves(root_state, player)
        action_probs = action_probs * valid_moves
        root_node.expand(action_probs=action_probs, player=next_player, parent=root_node)

        for _ in range(num_simulations):
            backup_steps = [root_node]
            node = root_node
            while not node.is_leaf_node():
                action_index, node = node.select_best_child()
                backup_steps.append(node)
            leaf_node = node
            parent_node = backup_steps[-2]

            action = np.zeros(ACTION_SIZE)
            action[action_index] = 1
            leaf_node_state = self.game.get_next_state_from_next_player_prespective(
                parent_node.state, action, player
            )
            leaf_node.set_state(leaf_node_state)

            value, action_probs = self.policy_value_network(leaf_node_state, leaf_node.player)
            winner = self.game.get_reward_for_next_player(leaf_node_state, leaf_node.player)

            if winner is None:
                valid_moves = self.game.get_valid_moves(leaf_node_state, leaf_node.player)
                action_probs = action_probs * valid_moves
                next_player = leaf_node.player * -1
                leaf_node.expand(action_probs=action_probs, player=next_player, parent=leaf_node)

            self.backup(backup_steps, winner, player, value)
        return root_node

    def select_move(self, node, mode="exploit", temperature=1):
        visits = [(k, v.total_visits_N) for k, v in node.children.items()]

        if mode == "exploit":
            action_index = max(visits, key=lambda t: t[1])[0]
        elif mode == "explore":
            visit_options = [k for k, v in node.children.items()]
            probs = [v.total_visits_N ** (1 / temperature) for k, v in node.children.items()]
            probs = [t / sum(probs) for t in probs]
            action_index = np.random.choice(visit_options, 1, p=probs)[0]

        action = np.zeros(ACTION_SIZE)
        action[action_index] = 1
        subtree = node.children[action_index]

        action_probs = np.zeros(ACTION_SIZE)
        for k, v in node.children.items():
            action_probs[k] = v.total_visits_N

        total_visits = np.sum(action_probs)
        if total_visits > 0:
            action_probs = action_probs / total_visits
        else:
            action_probs = action_probs / len(node.children) if len(node.children) > 0 else action_probs

        return action, subtree, action_probs
