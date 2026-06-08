import os
from glob import glob

import numpy as np
from tqdm import tqdm

from config import Config as cfg
from game import ChessGame
from mcts import MonteCarloTreeSearch
from dataset import TrainingDataset
from value_policy_function import ValuePolicyNetwork


def find_latest_model():
    """Return (path, iteration) of the newest checkpoint, or (None, -1)."""
    all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*.pt"))
    nums = []
    for f in all_models:
        prefix = os.path.basename(f).split("_")[0]
        if prefix.lstrip("-").isdigit():
            nums.append(int(prefix))
    if not nums:
        return None, -1
    latest = max(nums)
    return os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(latest)), latest


def sparse_policy(action_probs):
    """Compress a dense ACTION_SIZE distribution to (indices, probs)."""
    indices = np.nonzero(action_probs)[0]
    return indices.astype(np.int32), action_probs[indices].astype(np.float32)


def play_game(game, mcts):
    """Play one self-play game; return list of [fen, sparse_policy, player] and winner."""
    board = game.get_initial_board()
    root = mcts.make_root(board)

    records = []
    move_count = 0

    while not game.is_terminal(board) and move_count < cfg.MAX_MOVES:
        mcts.run_simulation(root, num_simulations=cfg.NUM_SIMULATIONS, add_noise=True)

        temperature = cfg.INITIAL_TEMP if move_count < cfg.TEMP_THRESHOLD else cfg.FINAL_TEMP
        _, child, action_probs = mcts.select_move(root, temperature=temperature)

        records.append([board.fen(), sparse_policy(action_probs), root.player])

        # Advance: the chosen child becomes the new root.
        board = child.board
        child.parent = None
        root = child
        move_count += 1

    if game.is_terminal(board):
        winner = game.get_result(board)  # +1 White, -1 Black, 0 draw
    else:
        winner = 0  # hit the move cap -> treat as a draw

    return records, winner


def main():
    os.makedirs(cfg.SAVE_PICKLES, exist_ok=True)
    save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)

    game = ChessGame()

    model_path, latest = find_latest_model()
    if model_path is not None:
        print(f"Loading trained model: {model_path}")
        vpn = ValuePolicyNetwork(path=model_path)
    else:
        print("No trained model found. Bootstrapping self-play with a random network.")
        vpn = ValuePolicyNetwork(path=None)

    mcts = MonteCarloTreeSearch(game, vpn.get_vp)

    training_dataset = TrainingDataset()
    num_games = cfg.SELFPLAY_GAMES

    for game_number in tqdm(range(num_games), total=num_games):
        records, winner = play_game(game, mcts)
        training_dataset.add_game_to_training_dataset(records, winner)

        if game_number % 50 == 0:
            training_dataset.save(save_path)
            print(f"saving.... game {game_number}, samples={len(training_dataset.training_dataset)}")

    training_dataset.save(save_path)
    print(f"Self-play complete. Total samples: {len(training_dataset.training_dataset)}")


if __name__ == "__main__":
    main()
