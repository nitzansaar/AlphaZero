import os
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


def play_games_worker(worker_id, num_games, model_path):
    """Worker function: loads model, plays games, returns results."""
    torch.set_num_threads(1)  # prevent thread oversubscription

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

            parent_state = copy(node.state)
            parent_state[:NUM_POSITIONS] *= player

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

            action, node, action_probs = mcts.select_move(node=node, mode="explore", temperature=temperature)
            dataset.append([parent_state, action_probs, player])
            player = -1 * player
            move_count += 1

        if force_ended:
            winner = game.get_winner(node.state, perspective=player)
        else:
            winner = game.winner(node.state, perspective=player)

        results.append((dataset, winner))

    return results


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

    # Distribute games evenly, giving one extra to the first `remainder` workers
    worker_args = []
    for i in range(NUM_WORKERS):
        n = games_per_worker + (1 if i < remainder else 0)
        worker_args.append((i, n, model_path))

    print(f"Starting {num_games} self-play games across {NUM_WORKERS} workers")

    with mp.Pool(NUM_WORKERS) as pool:
        all_results = pool.starmap(play_games_worker, worker_args)

    # Collect results into training dataset
    training_dataset = TrainingDataset()
    if os.path.exists(save_path):
        training_dataset.load(save_path)
        print(f"Loaded existing dataset with {len(training_dataset.training_dataset)} samples")
    else:
        print("Starting with empty dataset")

    for worker_results in all_results:
        for dataset, winner in worker_results:
            training_dataset.add_game_to_training_dataset(dataset, winner)

    training_dataset.save(save_path)
    print(f"Total training samples: {len(training_dataset.training_dataset)}")
