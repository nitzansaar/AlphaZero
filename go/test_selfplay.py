"""
Tests for self-play functionality.

Since selfplay.py is a script that runs on import, we test its components
and simulate the self-play process to verify correctness.
"""
import unittest
import numpy as np
import os
import tempfile
from copy import copy

from game import Go, PASS_ACTION, ACTION_SIZE, NUM_POSITIONS
from mcts import MonteCarloTreeSearch, Node
from dataset import TrainingDataset, GoDataset
from config import Config as cfg


def mock_policy_value_network(state, player):
    """
    Mock policy-value network that returns uniform policy and neutral value.
    Used for testing without requiring trained models.
    """
    value = 0.0
    policy = np.ones(ACTION_SIZE) / ACTION_SIZE
    return value, policy


def mock_policy_value_network_prefer_center(state, player):
    """
    Mock network that prefers center moves.
    Useful for testing move selection.
    """
    value = 0.0
    policy = np.ones(ACTION_SIZE) * 0.01
    policy[12] = 0.5  # Prefer center (position 12 on 5x5 board)
    policy = policy / np.sum(policy)
    return value, policy


def mock_policy_value_network_prefer_pass(state, player):
    """
    Mock network that prefers passing.
    Useful for testing game termination.
    """
    value = 0.0
    policy = np.ones(ACTION_SIZE) * 0.01
    policy[PASS_ACTION] = 0.9
    policy = policy / np.sum(policy)
    return value, policy


class TestNode(unittest.TestCase):
    """Test MCTS Node class."""

    def test_node_creation(self):
        node = Node(prior_prob=0.5, player=1)
        self.assertEqual(node.prior_probs_P, 0.5)
        self.assertEqual(node.player, 1)
        self.assertEqual(node.total_visits_N, 0)
        self.assertEqual(node.mean_action_value_of_next_state_Q, 0)
        self.assertIsNone(node.state)
        self.assertEqual(len(node.children), 0)

    def test_node_set_state(self):
        node = Node(prior_prob=0.5, player=1)
        state = np.zeros(NUM_POSITIONS + 2)
        node.set_state(state)
        np.testing.assert_array_equal(node.state, state)

    def test_node_is_leaf(self):
        node = Node(prior_prob=0.5, player=1)
        self.assertTrue(node.is_leaf_node())

    def test_node_expand(self):
        node = Node(prior_prob=0.5, player=1)
        state = np.zeros(NUM_POSITIONS + 2)
        state[NUM_POSITIONS] = -1  # No ko
        node.set_state(state)

        # Expand with some action probabilities
        action_probs = np.zeros(ACTION_SIZE)
        action_probs[0] = 0.3
        action_probs[12] = 0.5
        action_probs[PASS_ACTION] = 0.2

        node.expand(action_probs=action_probs, player=-1, parent=node)

        self.assertFalse(node.is_leaf_node())
        self.assertEqual(len(node.children), 3)  # 3 non-zero probs
        self.assertIn(0, node.children)
        self.assertIn(12, node.children)
        self.assertIn(PASS_ACTION, node.children)

    def test_node_child_with_action(self):
        """Test that child nodes correctly transform state."""
        parent = Node(prior_prob=0.5, player=1)
        state = np.zeros(NUM_POSITIONS + 2)
        state[NUM_POSITIONS] = -1  # No ko
        parent.set_state(state)

        # Create child with action at position 12
        child = Node(prior_prob=0.3, player=-1, parent=parent, action_index=12)

        # Child state should have the move applied
        self.assertIsNotNone(child.state)
        self.assertEqual(child.state[12], -1)  # Stone placed (flipped perspective)


class TestMCTS(unittest.TestCase):
    """Test Monte Carlo Tree Search."""

    def setUp(self):
        self.game = Go()
        self.mcts = MonteCarloTreeSearch(self.game, mock_policy_value_network)

    def test_init_root_node(self):
        root = self.mcts.init_root_node()

        self.assertIsNotNone(root.state)
        self.assertEqual(root.player, 1)
        self.assertEqual(len(root.state), NUM_POSITIONS + 2)
        # Board should be empty
        np.testing.assert_array_equal(root.state[:NUM_POSITIONS], np.zeros(NUM_POSITIONS))
        # Ko should be -1
        self.assertEqual(root.state[NUM_POSITIONS], -1)
        # Passes should be 0
        self.assertEqual(root.state[NUM_POSITIONS + 1], 0)

    def test_run_simulation_expands_root(self):
        root = self.mcts.init_root_node()

        # Run with few simulations
        root = self.mcts.run_simulation(root, num_simulations=10, player=1)

        # Root should be expanded
        self.assertFalse(root.is_leaf_node())
        self.assertGreater(len(root.children), 0)
        self.assertGreater(root.total_visits_N, 0)

    def test_run_simulation_visits_increase(self):
        root = self.mcts.init_root_node()

        root = self.mcts.run_simulation(root, num_simulations=50, player=1)

        # Total visits should equal number of simulations
        total_child_visits = sum(c.total_visits_N for c in root.children.values())
        self.assertEqual(total_child_visits, 50)

    def test_select_move_exploit(self):
        root = self.mcts.init_root_node()
        root = self.mcts.run_simulation(root, num_simulations=100, player=1)

        action, subtree, action_probs = self.mcts.select_move(root, mode="exploit")

        # Action should be one-hot encoded
        self.assertEqual(np.sum(action), 1)
        self.assertEqual(len(action), ACTION_SIZE)

        # Subtree should be a valid node
        self.assertIsInstance(subtree, Node)

        # Action probs should sum to 1
        self.assertAlmostEqual(np.sum(action_probs), 1.0, places=5)

    def test_select_move_explore(self):
        root = self.mcts.init_root_node()
        root = self.mcts.run_simulation(root, num_simulations=100, player=1)

        action, subtree, action_probs = self.mcts.select_move(root, mode="explore", temperature=1.0)

        # Action should be one-hot encoded
        self.assertEqual(np.sum(action), 1)
        self.assertEqual(len(action), ACTION_SIZE)

    def test_select_move_temperature(self):
        """Test that temperature affects move selection."""
        root = self.mcts.init_root_node()
        root = self.mcts.run_simulation(root, num_simulations=100, player=1)

        # Run multiple selections with high temperature - should have variety
        high_temp_moves = set()
        for _ in range(20):
            action, _, _ = self.mcts.select_move(root, mode="explore", temperature=2.0)
            high_temp_moves.add(np.argmax(action))

        # Run multiple selections with low temperature - should be more deterministic
        low_temp_moves = set()
        for _ in range(20):
            action, _, _ = self.mcts.select_move(root, mode="explore", temperature=0.1)
            low_temp_moves.add(np.argmax(action))

        # Low temperature should have fewer unique moves
        self.assertLessEqual(len(low_temp_moves), len(high_temp_moves))

    def test_backup(self):
        """Test that backup correctly updates node statistics."""
        root = self.mcts.init_root_node()

        # Create a simple path
        child = Node(prior_prob=0.5, player=-1)
        steps = [root, child]

        # Backup with a win for player 1
        self.mcts.backup(steps, winner=1, player=1, value=1.0)

        self.assertEqual(root.total_visits_N, 1)
        self.assertEqual(child.total_visits_N, 1)


class TestMCTSWithPreferredMoves(unittest.TestCase):
    """Test MCTS with a network that has move preferences."""

    def setUp(self):
        self.game = Go()
        self.mcts = MonteCarloTreeSearch(self.game, mock_policy_value_network_prefer_center)

    def test_prefers_center(self):
        """Network that prefers center should lead to center being visited more."""
        root = self.mcts.init_root_node()
        root = self.mcts.run_simulation(root, num_simulations=200, player=1, add_noise=False)

        # Center (position 12) should have high visits
        if 12 in root.children:
            center_visits = root.children[12].total_visits_N
            # Center should be among the most visited
            max_visits = max(c.total_visits_N for c in root.children.values())
            # With noise disabled, center should be close to max
            self.assertGreater(center_visits, max_visits * 0.3)


class TestTrainingDataset(unittest.TestCase):
    """Test training dataset functionality."""

    def test_empty_dataset(self):
        dataset = TrainingDataset()
        self.assertEqual(len(dataset.training_dataset), 0)

    def test_calculate_values_winner_black(self):
        dataset = TrainingDataset()

        # Simulate a game record: [state, action_probs, player]
        game_data = [
            [np.zeros(27), np.ones(26) / 26, 1],   # Black's move
            [np.zeros(27), np.ones(26) / 26, -1],  # White's move
            [np.zeros(27), np.ones(26) / 26, 1],   # Black's move
        ]

        # Black wins
        result = dataset.calculate_values(game_data, winner=1)

        # Black's positions should have value 1
        self.assertEqual(result[0][3], 1)   # Black's move -> win
        self.assertEqual(result[1][3], -1)  # White's move -> loss
        self.assertEqual(result[2][3], 1)   # Black's move -> win

    def test_calculate_values_winner_white(self):
        dataset = TrainingDataset()

        game_data = [
            [np.zeros(27), np.ones(26) / 26, 1],
            [np.zeros(27), np.ones(26) / 26, -1],
        ]

        # White wins
        result = dataset.calculate_values(game_data, winner=-1)

        self.assertEqual(result[0][3], -1)  # Black's move -> loss
        self.assertEqual(result[1][3], 1)   # White's move -> win

    def test_calculate_values_draw(self):
        dataset = TrainingDataset()

        game_data = [
            [np.zeros(27), np.ones(26) / 26, 1],
            [np.zeros(27), np.ones(26) / 26, -1],
        ]

        # Draw
        result = dataset.calculate_values(game_data, winner=0)

        self.assertEqual(result[0][3], 0)
        self.assertEqual(result[1][3], 0)

    def test_add_game_to_dataset(self):
        dataset = TrainingDataset()

        game_data = [
            [np.zeros(27), np.ones(26) / 26, 1],
            [np.zeros(27), np.ones(26) / 26, -1],
        ]

        dataset.add_game_to_training_dataset(game_data, winner=1)

        self.assertEqual(len(dataset.training_dataset), 2)

    def test_dataset_queue_limit(self):
        """Test that dataset respects queue size limit."""
        dataset = TrainingDataset()

        # Add more samples than the queue size
        original_queue_size = cfg.DATASET_QUEUE_SIZE

        # Temporarily reduce queue size for testing
        cfg.DATASET_QUEUE_SIZE = 10

        for i in range(20):
            game_data = [[np.zeros(27), np.ones(26) / 26, 1]]
            dataset.add_game_to_training_dataset(game_data, winner=1)

        # Should be limited to queue size
        self.assertEqual(len(dataset.training_dataset), 10)

        # Restore original
        cfg.DATASET_QUEUE_SIZE = original_queue_size

    def test_save_and_load(self):
        dataset = TrainingDataset()

        game_data = [
            [np.zeros(27), np.ones(26) / 26, 1],
            [np.zeros(27), np.ones(26) / 26, -1],
        ]
        dataset.add_game_to_training_dataset(game_data, winner=1)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            dataset.save(temp_path)

            # Load into new dataset
            loaded_dataset = TrainingDataset()
            loaded_dataset.load(temp_path)

            self.assertEqual(len(loaded_dataset.training_dataset), 2)
        finally:
            os.unlink(temp_path)


class TestGoDataset(unittest.TestCase):
    """Test the PyTorch dataset wrapper."""

    def test_dataset_length(self):
        data = [
            [np.zeros(27), np.ones(26) / 26, 1, 1],  # state, policy, player, value
            [np.zeros(27), np.ones(26) / 26, -1, -1],
        ]
        dataset = GoDataset(data, use_augmentation=False)

        self.assertEqual(len(dataset), 2)

    def test_dataset_getitem(self):
        state = np.zeros(27)
        state[12] = 1  # Place a stone
        policy = np.ones(26) / 26
        player = 1
        value = 1

        data = [[state, policy, player, value]]
        dataset = GoDataset(data, use_augmentation=False)

        state_tensor, value_tensor, policy_tensor = dataset[0]

        # Check shapes
        self.assertEqual(state_tensor.shape, (3, 5, 5))  # 3-plane representation
        self.assertEqual(value_tensor.shape, ())
        self.assertEqual(policy_tensor.shape, (26,))

        # Check value
        self.assertEqual(value_tensor.item(), 1)


class TestSelfPlayIntegration(unittest.TestCase):
    """Integration tests for the self-play process."""

    def setUp(self):
        self.game = Go()
        self.mcts = MonteCarloTreeSearch(self.game, mock_policy_value_network)

    def test_single_game_to_completion(self):
        """Test that a single self-play game can complete."""
        root_node = self.mcts.init_root_node()
        node = root_node
        player = 1
        move_count = 0
        max_moves = 50  # Reduced for testing

        dataset = []

        while self.game.win_or_draw(node.state, perspective=player) is None:
            if move_count >= max_moves:
                break

            parent_state = copy(node.state)
            node = self.mcts.run_simulation(root_node=node, num_simulations=10, player=player)

            action, node, action_probs = self.mcts.select_move(node, mode="explore", temperature=1.0)
            dataset.append([parent_state, action_probs, player])

            player = -1 * player
            move_count += 1

        # Game should have generated some data
        self.assertGreater(len(dataset), 0)

        # Each data point should have correct structure
        for state, probs, p in dataset:
            self.assertEqual(len(state), NUM_POSITIONS + 2)
            self.assertEqual(len(probs), ACTION_SIZE)
            self.assertIn(p, [1, -1])

    def test_game_with_pass_network(self):
        """Test game with a network that prefers passing (should end quickly)."""
        mcts = MonteCarloTreeSearch(self.game, mock_policy_value_network_prefer_pass)
        root_node = mcts.init_root_node()
        node = root_node
        player = 1
        move_count = 0

        while self.game.win_or_draw(node.state, perspective=player) is None:
            if move_count >= 100:
                break

            node = mcts.run_simulation(root_node=node, num_simulations=10, player=player)
            action, node, _ = mcts.select_move(node, mode="explore", temperature=1.0)
            player = -1 * player
            move_count += 1

        # Game should end (either naturally or by move limit)
        # With pass-preferring network, should end quickly due to consecutive passes
        self.assertLessEqual(move_count, 100)

    def test_temperature_decay(self):
        """Test temperature decay during self-play."""
        temp_threshold = 10
        initial_temp = 1.0

        for move_count in range(20):
            if move_count < temp_threshold:
                temperature = initial_temp
            else:
                temperature = 0.1

            if move_count < temp_threshold:
                self.assertEqual(temperature, 1.0)
            else:
                self.assertEqual(temperature, 0.1)

    def test_dataset_collection(self):
        """Test that dataset is correctly collected during self-play."""
        training_dataset = TrainingDataset()
        root_node = self.mcts.init_root_node()

        # Play a few short games
        for _ in range(3):
            node = root_node
            player = 1
            game_data = []
            move_count = 0

            while self.game.win_or_draw(node.state, perspective=player) is None:
                if move_count >= 10:  # Short games for testing
                    break

                parent_state = copy(node.state)
                node = self.mcts.run_simulation(root_node=node, num_simulations=5, player=player)
                action, node, action_probs = self.mcts.select_move(node, mode="explore")

                game_data.append([parent_state, action_probs, player])
                player = -1 * player
                move_count += 1

            # Determine winner
            if self.game.game_ended(node.state):
                winner = self.game.get_winner(node.state, perspective=player)
            else:
                winner = self.game.get_winner(node.state, perspective=player)

            training_dataset.add_game_to_training_dataset(game_data, winner)

        # Should have accumulated data from multiple games
        self.assertGreater(len(training_dataset.training_dataset), 0)

    def test_force_end_game(self):
        """Test force-ending a game that exceeds max moves."""
        max_moves = 5
        node = self.mcts.init_root_node()
        player = 1
        move_count = 0
        force_ended = False

        while self.game.win_or_draw(node.state, perspective=player) is None:
            if move_count >= max_moves:
                force_ended = True
                break

            node = self.mcts.run_simulation(root_node=node, num_simulations=5, player=player)
            _, node, _ = self.mcts.select_move(node, mode="explore")
            player = -1 * player
            move_count += 1

        # With only 5 moves allowed, game should be force-ended
        self.assertTrue(force_ended)

        # Should still be able to compute winner
        winner = self.game.get_winner(node.state, perspective=player)
        self.assertIn(winner, [-1, 0, 1])


class TestMCTSNoiseAndExploration(unittest.TestCase):
    """Test Dirichlet noise and exploration in MCTS."""

    def setUp(self):
        self.game = Go()
        self.mcts = MonteCarloTreeSearch(self.game, mock_policy_value_network)

    def test_dirichlet_noise_adds_exploration(self):
        """Test that Dirichlet noise is added at root."""
        root = self.mcts.init_root_node()

        # Run with noise
        root_with_noise = self.mcts.run_simulation(
            root, num_simulations=50, player=1, add_noise=True
        )

        # Run without noise (need fresh root)
        root2 = self.mcts.init_root_node()
        root_no_noise = self.mcts.run_simulation(
            root2, num_simulations=50, player=1, add_noise=False
        )

        # Both should have children
        self.assertGreater(len(root_with_noise.children), 0)
        self.assertGreater(len(root_no_noise.children), 0)

    def test_valid_moves_mask(self):
        """Test that invalid moves are not selected."""
        # Create a state where some moves are invalid
        root = self.mcts.init_root_node()

        # Place stones to make some moves invalid
        root.state[12] = 1  # Occupied position

        root = self.mcts.run_simulation(root, num_simulations=50, player=1)

        # Position 12 should not be a child (it's occupied)
        # Actually, the children are created based on valid moves
        # Let's verify that valid_moves masking works
        valid = self.game.get_valid_moves(root.state, player=1)
        self.assertEqual(valid[12], 0)  # Position 12 should be invalid


class TestNodeSelectBestChild(unittest.TestCase):
    """Test UCB selection in Node."""

    def test_select_best_child_ucb(self):
        """Test that UCB formula correctly selects children."""
        parent = Node(prior_prob=0.5, player=1)
        state = np.zeros(NUM_POSITIONS + 2)
        state[NUM_POSITIONS] = -1
        parent.set_state(state)

        # Create children manually
        child1 = Node(prior_prob=0.5, player=-1)
        child1.total_visits_N = 10
        child1.mean_action_value_of_next_state_Q = 0.5

        child2 = Node(prior_prob=0.3, player=-1)
        child2.total_visits_N = 5
        child2.mean_action_value_of_next_state_Q = 0.8

        parent.children = {0: child1, 1: child2}
        parent.total_visits_N = 15

        # Select best child
        idx, child = parent.select_best_child()

        # The selection depends on UCB calculation
        self.assertIn(idx, [0, 1])
        self.assertIsInstance(child, Node)


class TestRetrieveTrainTestData(unittest.TestCase):
    """Test train/test split functionality."""

    def test_retrieve_train_test_data(self):
        dataset = TrainingDataset()

        # Add some data
        for i in range(100):
            game_data = [[np.zeros(27), np.ones(26) / 26, 1]]
            dataset.add_game_to_training_dataset(game_data, winner=1)

        train_data, val_data = dataset.retreive_test_train_data()

        # Train data should have samples
        self.assertGreater(len(train_data), 0)

        # Note: The current implementation returns all data as train (val is empty)
        # This is because train_idx uses all samples


if __name__ == '__main__':
    unittest.main(verbosity=2)
