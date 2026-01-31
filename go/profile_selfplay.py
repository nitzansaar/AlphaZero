#!/usr/bin/env python3
"""
Profile selfplay to identify bottlenecks.

Usage:
    python profile_selfplay.py              # Run with cProfile (batched)
    python profile_selfplay.py --games 5    # Profile 5 games
    python profile_selfplay.py --sequential # Use original sequential MCTS
    python profile_selfplay.py --compare    # Compare sequential vs batched

For visual flame graph (requires py-spy):
    pip install py-spy
    py-spy record -o profile.svg -- python selfplay.py
"""

import cProfile
import pstats
import io
import os
import time
import argparse
from config import Config as cfg
from game import Go, PASS_ACTION, NUM_POSITIONS
from mcts import MonteCarloTreeSearch
from dataset import TrainingDataset
from value_policy_function import ValuePolicyNetwork
from copy import copy
from glob import glob

def run_selfplay_games(num_games=5, use_batched=True, batch_size=32):
    """Run selfplay games for profiling."""
    game = Go()

    # Load the latest trained model if available
    all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*.pt"))
    if all_models:
        files = [int(os.path.basename(f).split("_")[0]) for f in all_models if os.path.basename(f).split("_")[0].isdigit()]
        if files:
            latest_num = max(files)
            model_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(latest_num))
            print(f"Loading trained model: {model_path}")
            vpn = ValuePolicyNetwork(path=model_path)
        else:
            print("No trained models found. Using randomly initialized network.")
            vpn = ValuePolicyNetwork(path=None)
    else:
        print("No trained models found. Using randomly initialized network.")
        vpn = ValuePolicyNetwork(path=None)

    policy_value_network = vpn.get_vp
    policy_value_network_batch = vpn.get_vp_batch
    mcts = MonteCarloTreeSearch(game, policy_value_network, policy_value_network_batch)
    root_node = mcts.init_root_node()

    MAX_MOVES_PER_GAME = 100

    mode_str = f"BATCHED (batch_size={batch_size})" if use_batched else "SEQUENTIAL"
    print(f"\nProfiling {num_games} selfplay games with {cfg.NUM_SIMULATIONS} MCTS simulations per move...")
    print(f"Mode: {mode_str}\n")

    total_moves = 0
    start_time = time.time()

    for game_number in range(num_games):
        node = root_node
        dataset = []
        player = 1
        move_count = 0

        while game.winner(node.state, perspective=player) is None:
            if move_count >= MAX_MOVES_PER_GAME:
                break

            parent_state = copy(node.state)
            parent_state[:NUM_POSITIONS] *= player

            if use_batched:
                node = mcts.run_simulation_batched(
                    root_node=node,
                    num_simulations=cfg.NUM_SIMULATIONS,
                    player=player,
                    batch_size=batch_size
                )
            else:
                node = mcts.run_simulation(
                    root_node=node,
                    num_simulations=cfg.NUM_SIMULATIONS,
                    player=player
                )

            if move_count < cfg.TEMP_THRESHOLD:
                temperature = cfg.INITIAL_TEMP
            else:
                temperature = 0.1

            action, node, action_probs = mcts.select_move(node=node, mode="explore", temperature=temperature)
            dataset.append([parent_state, action_probs, player])
            player = -1 * player
            move_count += 1

        total_moves += move_count
        print(f"  Game {game_number + 1}/{num_games}: {move_count} moves")

    elapsed = time.time() - start_time
    print(f"\nDone! Total: {total_moves} moves in {elapsed:.2f}s")
    print(f"Speed: {total_moves / elapsed:.2f} moves/sec, {total_moves * cfg.NUM_SIMULATIONS / elapsed:.0f} sims/sec")
    return elapsed, total_moves

def main():
    parser = argparse.ArgumentParser(description='Profile selfplay')
    parser.add_argument('--games', type=int, default=3, help='Number of games to profile')
    parser.add_argument('--top', type=int, default=30, help='Show top N functions')
    parser.add_argument('--sequential', action='store_true', help='Use sequential MCTS (original)')
    parser.add_argument('--compare', action='store_true', help='Compare sequential vs batched')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for batched MCTS')
    parser.add_argument('--no-profile', action='store_true', help='Skip detailed profiling, just time')
    args = parser.parse_args()

    if args.compare:
        print("="*80)
        print("COMPARISON: Sequential vs Batched MCTS")
        print("="*80)

        print("\n--- Sequential MCTS ---")
        seq_time, seq_moves = run_selfplay_games(num_games=args.games, use_batched=False)

        print("\n--- Batched MCTS ---")
        batch_time, batch_moves = run_selfplay_games(num_games=args.games, use_batched=True, batch_size=args.batch_size)

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Sequential: {seq_time:.2f}s for {seq_moves} moves ({seq_moves/seq_time:.2f} moves/sec)")
        print(f"Batched:    {batch_time:.2f}s for {batch_moves} moves ({batch_moves/batch_time:.2f} moves/sec)")
        speedup = seq_time / batch_time if batch_time > 0 else float('inf')
        print(f"\nSpeedup: {speedup:.2f}x faster with batched MCTS")
        return

    use_batched = not args.sequential

    if args.no_profile:
        run_selfplay_games(num_games=args.games, use_batched=use_batched, batch_size=args.batch_size)
        return

    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    run_selfplay_games(num_games=args.games, use_batched=use_batched, batch_size=args.batch_size)

    profiler.disable()

    # Print results
    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by cumulative time)")
    print("="*80 + "\n")

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(args.top)
    print(s.getvalue())

    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by total time in function)")
    print("="*80 + "\n")

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    stats.print_stats(args.top)
    print(s.getvalue())

    # Save to file for later analysis
    profiler.dump_stats('selfplay_profile.prof')
    print("\nProfile saved to selfplay_profile.prof")
    print("View with: python -m pstats selfplay_profile.prof")
    print("Or visualize with: pip install snakeviz && snakeviz selfplay_profile.prof")

if __name__ == "__main__":
    main()
