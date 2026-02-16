import os
import pickle
import tempfile
import torch
import torch.multiprocessing as mp
from config import Config as cfg
from game import Go, PASS_ACTION, NUM_POSITIONS
from mcts import MonteCarloTreeSearch
from dataset import TrainingDataset
from tqdm import tqdm
from value_policy_function import ValuePolicyNetwork
from copy import copy
from glob import glob
from profiler import Timer

# Maximum moves per game - force end if exceeded
MAX_MOVES_PER_GAME = 100

# Batch size for MCTS (number of leaf nodes to evaluate in parallel)
MCTS_BATCH_SIZE = 32

# Number of parallel worker processes
NUM_WORKERS = os.cpu_count()


def get_latest_model_path():
    """Find the latest trained model path, or return None for random init."""
    all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*.pt"))
    if all_models:
        files = [int(os.path.basename(f).split("_")[0]) for f in all_models if os.path.basename(f).split("_")[0].isdigit()]
        if files:
            latest_num = max(files)
            return os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(latest_num))
    return None


def play_games_worker(worker_id, num_games, model_path, output_dir):
    """Worker function: loads model, plays games, saves results to disk."""
    torch.set_num_threads(1)  # prevent thread oversubscription
    timer = Timer()

    game = Go()
    vpn = ValuePolicyNetwork(path=model_path, use_compile=False)
    mcts = MonteCarloTreeSearch(game, vpn.get_vp, vpn.get_vp_batch)
    root_node = mcts.init_root_node()

    results = []

    for _ in tqdm(range(num_games), desc=f"Worker {worker_id}", position=worker_id):
        node = root_node
        dataset = []
        player = 1
        move_count = 0
        force_ended = False

        while game.winner(node.state, perspective=player) is None:
            if move_count >= MAX_MOVES_PER_GAME:
                force_ended = True
                break

            with timer.track("state_copy"):
                parent_state = copy(node.state)
                parent_state[:NUM_POSITIONS] *= player

            with timer.track("mcts_simulation"):
                node = mcts.run_simulation_batched(
                    root_node=node,
                    num_simulations=cfg.NUM_SIMULATIONS,
                    player=player,
                    batch_size=MCTS_BATCH_SIZE
                )

            if move_count < cfg.TEMP_THRESHOLD:
                temperature = cfg.INITIAL_TEMP
            else:
                temperature = 0.1

            with timer.track("move_selection"):
                action, node, action_probs = mcts.select_move(node=node, mode="explore", temperature=temperature)
            dataset.append([parent_state, action_probs, player])
            player = -1 * player
            move_count += 1

        with timer.track("winner_check"):
            if force_ended:
                winner = game.get_winner(node.state, perspective=player)
            else:
                winner = game.winner(node.state, perspective=player)

        results.append((dataset, winner))

    # Save results to disk instead of returning through pipe (avoids deadlock)
    output_path = os.path.join(output_dir, f"worker_{worker_id}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump((results, timer.to_dict()), f)

    return output_path


if __name__ == '__main__':
    mp.set_start_method('spawn')

    os.makedirs(cfg.SAVE_PICKLES, exist_ok=True)
    save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)

    model_path = get_latest_model_path()
    if model_path:
        print(f"Loading trained model: {model_path}")
    else:
        print("No trained models found. Using randomly initialized network.")

    num_games = cfg.SELFPLAY_GAMES
    games_per_worker = num_games // NUM_WORKERS
    remainder = num_games % NUM_WORKERS

    # Temp directory for worker output files
    tmp_dir = tempfile.mkdtemp(prefix="selfplay_")

    # Distribute games evenly, giving one extra to the first `remainder` workers
    worker_args = []
    for i in range(NUM_WORKERS):
        n = games_per_worker + (1 if i < remainder else 0)
        worker_args.append((i, n, model_path, tmp_dir))

    print(f"Starting {num_games} self-play games across {NUM_WORKERS} workers")

    with mp.Pool(NUM_WORKERS) as pool:
        output_paths = pool.starmap(play_games_worker, worker_args)

    # Read results from disk
    combined_timer = Timer()
    training_dataset = TrainingDataset()
    if os.path.exists(save_path):
        training_dataset.load(save_path)
        print(f"Loaded existing dataset with {len(training_dataset.training_dataset)} samples")
    else:
        print("Starting with empty dataset")

    for path in output_paths:
        with open(path, 'rb') as f:
            worker_results, worker_timings = pickle.load(f)
        for dataset, winner in worker_results:
            training_dataset.add_game_to_training_dataset(dataset, winner)
        combined_timer.merge(worker_timings)
        os.remove(path)  # clean up temp file

    os.rmdir(tmp_dir)

    # Aggregate timing data from all workers (average across parallel workers)
    combined_timer.average()
    combined_timer.print_summary("Self-Play Timing (avg per worker)")
    timing_path = os.path.join(cfg.LOGDIR, "selfplay_timing.json")
    combined_timer.save(timing_path)
    print(f"Timing data saved to {timing_path}")

    training_dataset.save(save_path)
    print(f"Total training samples: {len(training_dataset.training_dataset)}")
