"""
Test whether the neural network actually contributes to playing strength.

Compares:
1. Pure MCTS (uniform random policy, no value guidance) vs Random
2. NN-guided MCTS (your trained model) vs Random
3. Pure NN policy (no MCTS, just pick highest probability move) vs Random

If #1 and #2 have similar win rates, the NN isn't helping.
"""

import os
import numpy as np
from tqdm import tqdm
from glob import glob
import torch
from config import Config as cfg
from game import Go, ACTION_SIZE, NUM_POSITIONS
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_GAMES = 50  # Games per test
MCTS_BATCH_SIZE = 32


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def get_action(self, state, player=1):
        valid_moves = self.game.get_valid_moves(state, player)
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) == 0:
            return None
        return np.random.choice(valid_indices)


class PureMCTS:
    """MCTS with uniform random policy and no value network."""

    def __init__(self, game):
        self.game = game

    def uniform_policy_value(self, state, player):
        """Return uniform policy and neutral value."""
        policy = np.ones(ACTION_SIZE) / ACTION_SIZE
        value = 0.0  # Neutral - let MCTS figure it out
        return value, policy

    def uniform_policy_value_batch(self, states, players):
        """Batch version."""
        n = len(states)
        values = [0.0] * n
        policies = [np.ones(ACTION_SIZE) / ACTION_SIZE for _ in range(n)]
        return values, policies


class PureNNPlayer:
    """Player that just uses NN policy directly, no MCTS."""

    def __init__(self, vpn, game):
        self.vpn = vpn
        self.game = game

    def get_action(self, state, player):
        value, policy = self.vpn.get_vp(state, player)
        valid_moves = self.game.get_valid_moves(state, player)

        # Mask invalid moves
        policy = policy * valid_moves
        if np.sum(policy) > 0:
            policy = policy / np.sum(policy)
        else:
            # Fallback to random valid move
            policy = valid_moves / np.sum(valid_moves)

        return np.argmax(policy)


def play_game_mcts(game, mcts, random_player, bot_plays_first, num_simulations):
    """Play using MCTS-based bot."""
    player = 1
    state = game.state.copy()
    bot_player = 1 if bot_plays_first else -1

    while game.winner(state) is None:
        if player == bot_player:
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation_batched(
                root_node=node, num_simulations=num_simulations,
                player=player, batch_size=MCTS_BATCH_SIZE, add_noise=False
            )
            action, _, _ = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = np.argmax(action)
        else:
            action_index = random_player.get_action(state, player)
            if action_index is None:
                break

        state = game.apply_move(state, action_index, player)
        player = -player

    winner = game.get_winner(state)
    if winner == bot_player:
        return 1
    elif winner == -bot_player:
        return -1
    return 0


def play_game_pure_nn(game, nn_player, random_player, bot_plays_first):
    """Play using pure NN policy (no MCTS)."""
    player = 1
    state = game.state.copy()
    bot_player = 1 if bot_plays_first else -1

    while game.winner(state) is None:
        if player == bot_player:
            action_index = nn_player.get_action(state, player)
        else:
            action_index = random_player.get_action(state, player)
            if action_index is None:
                break

        state = game.apply_move(state, action_index, player)
        player = -player

    winner = game.get_winner(state)
    if winner == bot_player:
        return 1
    elif winner == -bot_player:
        return -1
    return 0


def test_configuration(name, play_func, num_games):
    """Run games and return win rate."""
    wins = 0
    for i in range(num_games):
        bot_first = (i % 2 == 0)
        result = play_func(bot_first)
        if result == 1:
            wins += 1
    return wins / num_games * 100


def main():
    print("=" * 70)
    print("NEURAL NETWORK CONTRIBUTION TEST")
    print("=" * 70)
    print(f"Games per configuration: {NUM_GAMES}")
    print()

    game = Go()
    random_player = RandomPlayer(game)

    # Find the best trained model - handle running from different directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, cfg.SAVE_MODEL_PATH)
    model_files = glob(os.path.join(model_dir, "*_best_model.pt"))

    if not model_files:
        print("No trained models found!")
        return

    # Use iteration 1 (the best one based on earlier evaluation)
    # Or find the latest
    models_with_iter = []
    for f in model_files:
        try:
            iter_num = int(os.path.basename(f).split("_")[0])
            models_with_iter.append((iter_num, f))
        except ValueError:
            continue

    models_with_iter.sort(key=lambda x: x[0])

    # Test with multiple simulation counts
    sim_counts = [100, 400, 800]

    results = []

    for num_sims in sim_counts:
        print(f"\n{'='*70}")
        print(f"TESTING WITH {num_sims} MCTS SIMULATIONS")
        print(f"{'='*70}")

        # Test 1: Pure MCTS (no NN)
        print(f"\n[1/3] Pure MCTS (uniform policy, no value network)...")
        pure_mcts_obj = PureMCTS(game)
        pure_mcts = MonteCarloTreeSearch(
            game,
            pure_mcts_obj.uniform_policy_value,
            pure_mcts_obj.uniform_policy_value_batch
        )

        pure_wins = 0
        for i in tqdm(range(NUM_GAMES), desc="Pure MCTS"):
            bot_first = (i % 2 == 0)
            result = play_game_mcts(game, pure_mcts, random_player, bot_first, num_sims)
            if result == 1:
                pure_wins += 1
        pure_win_rate = pure_wins / NUM_GAMES * 100
        print(f"Pure MCTS win rate: {pure_win_rate:.1f}%")

        # Test 2: NN-guided MCTS (trained model)
        print(f"\n[2/3] NN-guided MCTS (trained model)...")

        # Use iteration 1 (best) if available, otherwise latest
        best_model_path = None
        for iter_num, path in models_with_iter:
            if iter_num == 1:
                best_model_path = path
                break
        if best_model_path is None:
            best_model_path = models_with_iter[-1][1]

        print(f"Using model: {os.path.basename(best_model_path)}")
        vpn = ValuePolicyNetwork(best_model_path, use_compile=False)
        nn_mcts = MonteCarloTreeSearch(game, vpn.get_vp, vpn.get_vp_batch)

        nn_wins = 0
        for i in tqdm(range(NUM_GAMES), desc="NN MCTS"):
            bot_first = (i % 2 == 0)
            result = play_game_mcts(game, nn_mcts, random_player, bot_first, num_sims)
            if result == 1:
                nn_wins += 1
        nn_win_rate = nn_wins / NUM_GAMES * 100
        print(f"NN-guided MCTS win rate: {nn_win_rate:.1f}%")

        # Calculate NN contribution
        nn_contribution = nn_win_rate - pure_win_rate

        results.append({
            'simulations': num_sims,
            'pure_mcts': pure_win_rate,
            'nn_mcts': nn_win_rate,
            'nn_contribution': nn_contribution
        })

        # Cleanup
        del vpn, nn_mcts
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 3: Pure NN (no MCTS at all)
    print(f"\n{'='*70}")
    print("TESTING PURE NN (NO MCTS)")
    print(f"{'='*70}")

    vpn = ValuePolicyNetwork(best_model_path, use_compile=False)
    pure_nn_player = PureNNPlayer(vpn, game)

    pure_nn_wins = 0
    for i in tqdm(range(NUM_GAMES), desc="Pure NN"):
        bot_first = (i % 2 == 0)
        result = play_game_pure_nn(game, pure_nn_player, random_player, bot_first)
        if result == 1:
            pure_nn_wins += 1
    pure_nn_win_rate = pure_nn_wins / NUM_GAMES * 100
    print(f"Pure NN (no MCTS) win rate: {pure_nn_win_rate:.1f}%")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Simulations':<12} | {'Pure MCTS':<12} | {'NN + MCTS':<12} | {'NN Contribution':<15}")
    print("-" * 60)
    for r in results:
        contrib_str = f"+{r['nn_contribution']:.1f}%" if r['nn_contribution'] >= 0 else f"{r['nn_contribution']:.1f}%"
        print(f"{r['simulations']:<12} | {r['pure_mcts']:<11.1f}% | {r['nn_mcts']:<11.1f}% | {contrib_str:<15}")

    print(f"\nPure NN (0 sims): {pure_nn_win_rate:.1f}%")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    avg_contribution = np.mean([r['nn_contribution'] for r in results])

    if avg_contribution < 5:
        print("\nThe NN is NOT significantly helping. MCTS is doing most of the work.")
        print("Possible reasons:")
        print("  - 5x5 board is small enough that MCTS can solve it")
        print("  - NN policy/value predictions aren't accurate enough")
        print("  - Training data quality issues")
    elif avg_contribution < 15:
        print("\nThe NN provides MODEST benefit over pure MCTS.")
        print("More training might help, but diminishing returns expected.")
    else:
        print("\nThe NN provides SIGNIFICANT benefit over pure MCTS.")
        print("The training is working as expected.")

    if pure_nn_win_rate > 60:
        print(f"\nPure NN ({pure_nn_win_rate:.0f}%) shows the network has learned something useful.")
    else:
        print(f"\nPure NN ({pure_nn_win_rate:.0f}%) is weak - the network hasn't learned good policy.")


if __name__ == "__main__":
    main()
