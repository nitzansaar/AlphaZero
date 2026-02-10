import numpy as np
import math
from copy import copy
from config import Config as cfg
from game import NUM_POSITIONS, PASS_ACTION, ACTION_SIZE

class Node:
    """
    MCTS tree node.

    State representation:
    - States are stored in CANONICAL form where current player's stones = 1
    - node.player stores the ABSOLUTE player identity (1=Black, -1=White)
    - To convert canonical to absolute: multiply board by node.player
    """
    def __init__(self, prior_prob, player, parent=None, action_index=None):
        self.state = None  # State is set later via set_state()
        self.player = player  # Absolute player (1=Black, -1=White)
        self.total_visits_N = 0
        self.total_action_value_of_next_state_W = 0
        self.mean_action_value_of_next_state_Q = 0
        self.prior_probs_P = prior_prob
        self.children = {}
        self.parent = parent
        self.virtual_loss = 0  # For parallel MCTS - temporarily makes node look worse

    def set_state(self, state):
        self.state = state

    def expand(self, action_probs, player, parent):
        for i, action_prob in enumerate(action_probs):
            if action_prob != 0:
                self.children[i] = Node(action_prob, player, parent, i)

    def is_leaf_node(self):
        return len(self.children) == 0

    def select_best_child(self):
        """
        Select child with highest UCB score.
        Virtual loss is incorporated to discourage multiple parallel traversals
        from selecting the same path.

        Note: Q values are stored from the child's player perspective.
        Since players alternate, parent wants to MINIMIZE child's value,
        so we use -Q in the UCB formula.
        """
        best_uscore = float('-inf')
        best_child_index = None
        for i, child in self.children.items():
            psa = child.prior_probs_P
            # Include virtual loss in visit counts
            Ns = self.total_visits_N + self.virtual_loss
            Nsa = child.total_visits_N + child.virtual_loss
            Cs = cfg.MCTS_UCB_C
            # Adjust Q for virtual loss (treat virtual visits as losses)
            if child.total_visits_N + child.virtual_loss > 0:
                Q = (child.total_action_value_of_next_state_W - child.virtual_loss) / (child.total_visits_N + child.virtual_loss)
            else:
                Q = 0
            # Use -Q because Q is from child's perspective, but parent wants to maximize parent's value
            Uscore = -Q + Cs * psa * math.sqrt(Ns) / (1 + Nsa)
            if best_uscore < Uscore:
                best_uscore = Uscore
                best_child_index = i
        return best_child_index, self.children[best_child_index]


class MonteCarloTreeSearch:
    def __init__(self, game, policy_value_network, policy_value_network_batch=None):
        self.game = game
        self.policy_value_network = policy_value_network
        self.policy_value_network_batch = policy_value_network_batch

    def init_root_node(self):
        # State size: n*n board positions + ko point + consecutive passes
        root_state = np.zeros(NUM_POSITIONS + 2)
        root_state[NUM_POSITIONS] = -1  # No ko point
        root_state[NUM_POSITIONS + 1] = 0  # No consecutive passes
        root_node = Node(prior_prob=0, player=1, action_index=None)
        root_node.set_state(root_state)
        return root_node

    def backup(self, mtc_steps, winner, leaf_player, value):
        """
        Backup value through the tree from leaf to root.

        Args:
            mtc_steps: List of nodes from root to leaf
            winner: Game result (1=Black wins, -1=White wins, 0=draw, None=ongoing)
            leaf_player: The player at the leaf node (whose perspective the NN value is from)
            value: NN's value estimate (from leaf_player's perspective)
        """
        # Start with the value from the leaf's perspective
        # We need to track whose perspective the current value is from
        current_value_perspective = leaf_player

        for node in reversed(mtc_steps):
            node.total_visits_N += 1

            if winner is not None:
                # Terminal state: use actual game outcome
                if winner == 0:
                    node_value = 0
                else:
                    # Value from node's player perspective: +1 if they won, -1 if they lost
                    node_value = 1 if winner == node.player else -1
            else:
                # Non-terminal: use NN value, but adjust for perspective
                # If current_value_perspective matches node.player, use value directly
                # Otherwise, negate it
                if current_value_perspective == node.player:
                    node_value = value
                else:
                    node_value = -value

            node.total_action_value_of_next_state_W += node_value
            node.mean_action_value_of_next_state_Q = node.total_action_value_of_next_state_W / node.total_visits_N

    def run_simulation(self, root_node, num_simulations=1600, player=1, add_noise=True):
        root_state = root_node.state
        next_player = -1 * player

        # Convert state to absolute form for neural network if needed
        # Child nodes (parent is not None) store state in relative form
        # Fresh nodes (parent is None) already have absolute form
        if root_node.parent is not None:
            # Child node: convert relative form to absolute form
            absolute_state = root_state.copy()
            absolute_state[:NUM_POSITIONS] *= player
        else:
            # Fresh node: already in absolute form
            absolute_state = root_state

        value, action_probs = self.policy_value_network(absolute_state, player)
        # In canonical form, current player's stones = 1, so use player=1 for valid move check
        # For fresh root nodes (absolute form), player is already 1 (Black starts)
        valid_moves_player = 1 if root_node.parent is not None else player
        valid_moves = self.game.get_valid_moves(root_state, valid_moves_player)
        action_probs = action_probs * valid_moves

        # Add Dirichlet noise at root for exploration (AlphaGo Zero style)
        if add_noise:
            noise = np.random.dirichlet([0.03] * len(action_probs))
            action_probs = 0.75 * action_probs + 0.25 * noise
            # Re-mask invalid moves after adding noise
            action_probs = action_probs * valid_moves
            # Renormalize
            if np.sum(action_probs) > 0:
                action_probs = action_probs / np.sum(action_probs)

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

            # Convert from relative form to absolute form for neural network
            absolute_leaf_state = leaf_node_state.copy()
            absolute_leaf_state[:NUM_POSITIONS] *= leaf_node.player
            value, action_probs = self.policy_value_network(absolute_leaf_state, leaf_node.player)
            winner = self.game.get_reward_for_next_player(leaf_node_state, leaf_node.player)

            if winner is None:
                # leaf_node_state is in canonical form (current player's stones = 1)
                # So always use player=1 for valid move checking
                valid_moves = self.game.get_valid_moves(leaf_node_state, 1)
                action_probs = action_probs * valid_moves
                next_player = leaf_node.player * -1
                leaf_node.expand(action_probs=action_probs, player=next_player, parent=leaf_node)

            self.backup(backup_steps, winner, leaf_node.player, value)
        return root_node

    def run_simulation_batched(self, root_node, num_simulations=1600, player=1, add_noise=True, batch_size=32):
        """
        AlphaGo Zero style batched MCTS simulation.

        Uses virtual loss to collect multiple leaf nodes, evaluates them in a single
        batched NN call, then backs up all results. Much more efficient GPU utilization.

        Args:
            root_node: The root node to start search from
            num_simulations: Total number of simulations to run
            player: Current player (1 or -1)
            add_noise: Whether to add Dirichlet noise at root
            batch_size: Number of leaf nodes to collect before batched NN evaluation
        """
        if self.policy_value_network_batch is None:
            raise ValueError("Batch network function not provided. Use run_simulation instead.")

        root_state = root_node.state
        next_player = -1 * player

        # Convert state to absolute form for neural network if needed
        if root_node.parent is not None:
            absolute_state = root_state.copy()
            absolute_state[:NUM_POSITIONS] *= player
        else:
            absolute_state = root_state

        # Initial expansion of root node (single NN call)
        value, action_probs = self.policy_value_network(absolute_state, player)
        valid_moves_player = 1 if root_node.parent is not None else player
        valid_moves = self.game.get_valid_moves(root_state, valid_moves_player)
        action_probs = action_probs * valid_moves

        # Add Dirichlet noise at root for exploration
        if add_noise:
            noise = np.random.dirichlet([0.03] * len(action_probs))
            action_probs = 0.75 * action_probs + 0.25 * noise
            action_probs = action_probs * valid_moves
            if np.sum(action_probs) > 0:
                action_probs = action_probs / np.sum(action_probs)

        root_node.expand(action_probs=action_probs, player=next_player, parent=root_node)

        # Run simulations in batches
        sim_count = 0
        while sim_count < num_simulations:
            current_batch_size = min(batch_size, num_simulations - sim_count)

            # Collect leaf nodes with virtual loss
            leaf_data = []  # List of (leaf_node, backup_steps, action_index, parent_node)

            for _ in range(current_batch_size):
                backup_steps = [root_node]
                node = root_node

                # Apply virtual loss as we traverse down
                node.virtual_loss += 1

                while not node.is_leaf_node():
                    action_index, node = node.select_best_child()
                    node.virtual_loss += 1
                    backup_steps.append(node)

                leaf_node = node
                parent_node = backup_steps[-2] if len(backup_steps) > 1 else root_node

                # Get the action that led to this leaf
                action_index = None
                for idx, child in parent_node.children.items():
                    if child is leaf_node:
                        action_index = idx
                        break

                leaf_data.append((leaf_node, backup_steps, action_index, parent_node))

            # Deduplicate leaves - track unique leaves and their first occurrence
            seen_leaves = {}  # leaf_node id -> index in unique_leaf_data
            unique_leaf_data = []  # List of (leaf_node, action_index, parent_node)
            leaf_to_paths = {}  # leaf_node id -> list of backup_steps

            for leaf_node, backup_steps, action_index, parent_node in leaf_data:
                leaf_id = id(leaf_node)
                if leaf_id not in seen_leaves:
                    seen_leaves[leaf_id] = len(unique_leaf_data)
                    unique_leaf_data.append((leaf_node, action_index, parent_node))
                    leaf_to_paths[leaf_id] = [backup_steps]
                else:
                    leaf_to_paths[leaf_id].append(backup_steps)

            # Compute states for unique leaf nodes only
            states_to_eval = []
            players_to_eval = []
            valid_moves_list = []

            for leaf_node, action_index, parent_node in unique_leaf_data:
                # Compute leaf state if not already set
                if leaf_node.state is None and action_index is not None:
                    action = np.zeros(ACTION_SIZE)
                    action[action_index] = 1
                    leaf_node_state = self.game.get_next_state_from_next_player_prespective(
                        parent_node.state, action, player
                    )
                    leaf_node.set_state(leaf_node_state)

                if leaf_node.state is not None:
                    # Convert from canonical to absolute form for NN
                    absolute_leaf_state = leaf_node.state.copy()
                    absolute_leaf_state[:NUM_POSITIONS] *= leaf_node.player
                    states_to_eval.append(absolute_leaf_state)
                    players_to_eval.append(leaf_node.player)
                    # Canonical form: current player = 1
                    valid_moves = self.game.get_valid_moves(leaf_node.state, 1)
                    valid_moves_list.append(valid_moves)

            # Batch evaluate unique leaf nodes with single NN call
            if states_to_eval:
                values, policies = self.policy_value_network_batch(states_to_eval, players_to_eval)
            else:
                values, policies = [], []

            # Process results: expand once per leaf, backup all paths
            eval_idx = 0
            for leaf_node, action_index, parent_node in unique_leaf_data:
                leaf_id = id(leaf_node)
                all_paths = leaf_to_paths[leaf_id]

                # Remove virtual loss from all paths to this leaf
                for backup_steps in all_paths:
                    for node in backup_steps:
                        node.virtual_loss -= 1

                if leaf_node.state is None:
                    continue

                value = values[eval_idx] if eval_idx < len(values) else 0
                action_probs = policies[eval_idx] if eval_idx < len(policies) else np.zeros(ACTION_SIZE)
                valid_moves = valid_moves_list[eval_idx] if eval_idx < len(valid_moves_list) else np.ones(ACTION_SIZE)
                eval_idx += 1

                winner = self.game.get_reward_for_next_player(leaf_node.state, leaf_node.player)

                # Expand only once per unique leaf
                if winner is None:
                    action_probs = action_probs * valid_moves
                    next_player = leaf_node.player * -1
                    leaf_node.expand(action_probs=action_probs, player=next_player, parent=leaf_node)

                # Backup all paths that led to this leaf
                for backup_steps in all_paths:
                    self.backup(backup_steps, winner, leaf_node.player, value)

            sim_count += current_batch_size

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
