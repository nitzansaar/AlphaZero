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


# ----------------------------------------------------------------------------
# Batched self-play
# ----------------------------------------------------------------------------
# Many games run concurrently in one process. Each game is a generator (a
# coroutine) that yields a board whenever its MCTS needs an evaluation and is
# sent back (value, policy). A driver gathers the boards yielded by every active
# game, runs ONE batched forward pass, and sends each result back — turning
# thousands of batch-1 calls into a few batched calls per step. The MCTS math is
# identical to play_game / run_simulation; only *when* inference happens changes.


def play_game_coro(mcts, game, allow_resign):
    """Generator self-play game.

    Yields a chess.Board each time an evaluation is needed; the driver sends
    back (value, policy). Returns (records, winner) via StopIteration.value.
    """
    board = game.get_initial_board()
    root = mcts.make_root(board)

    records = []
    move_count = 0
    resign_streak = 0
    winner = None

    while not game.is_terminal(board) and move_count < cfg.MAX_MOVES:
        # Expand the root the first time it is used.
        if root.is_leaf_node() and not game.is_terminal(root.board):
            _, probs = yield root.board
            root.expand(probs, game)

        mcts._add_dirichlet_noise(root)

        for _ in range(cfg.NUM_SIMULATIONS):
            path = [root]
            node = root
            while not node.is_leaf_node():
                _, node = node.select_best_child()
                path.append(node)

            leaf = node
            if game.is_terminal(leaf.board):
                result = game.get_result(leaf.board)
                leaf_value = result * leaf.player
            else:
                value, probs = yield leaf.board
                leaf.expand(probs, game)
                leaf_value = value

            mcts.backup(path, leaf_value, leaf.player)

        temperature = cfg.INITIAL_TEMP if move_count < cfg.TEMP_THRESHOLD else cfg.FINAL_TEMP
        _, child, action_probs = mcts.select_move(root, temperature=temperature)

        records.append([board.fen(), sparse_policy(action_probs), root.player])

        # Resignation: root.mean_action_value_Q is the searched value from the
        # side-to-move's perspective. If it stays very negative, abandon the game.
        if allow_resign:
            if root.mean_action_value_Q < cfg.RESIGN_THRESHOLD:
                resign_streak += 1
            else:
                resign_streak = 0
            if resign_streak >= cfg.RESIGN_CONSECUTIVE:
                winner = -root.player  # side to move resigns; opponent wins
                break

        # Advance: the chosen child becomes the new root.
        board = child.board
        child.parent = None
        root = child
        move_count += 1

    if winner is None:
        if game.is_terminal(board):
            winner = game.get_result(board)  # +1 White, -1 Black, 0 draw
        else:
            winner = 0  # hit the move cap -> treat as a draw

    return records, winner


def run_batched_selfplay(vpn, game, num_games, num_parallel):
    """Play num_games via num_parallel concurrent coroutines, batching every
    evaluation step into one forward pass. Yields (records, winner) as each
    game finishes."""
    mcts = MonteCarloTreeSearch(game, None)  # network is reached via yield, not this

    def new_coro():
        # A fraction of games never resign (calibration / full-game data).
        allow_resign = np.random.random() >= cfg.RESIGN_PLAYTHROUGH_FRAC
        coro = play_game_coro(mcts, game, allow_resign)
        board = coro.send(None)  # prime: advance to first eval request
        return coro, board

    num_parallel = max(1, min(num_parallel, num_games))
    started = 0
    coros = []        # active coroutines
    boards = []       # board each active coroutine is currently waiting on

    for _ in range(num_parallel):
        if started >= num_games:
            break
        coro, board = new_coro()
        coros.append(coro)
        boards.append(board)
        started += 1

    while coros:
        # One batched forward pass for every active game's pending board.
        # Pad to num_parallel so the input shape stays constant (keeps the
        # torch.compile CUDA-graph capture reusable across steps).
        results = vpn.get_vp_batch(boards, pad_to=num_parallel)

        next_coros = []
        next_boards = []
        for coro, result in zip(coros, results):
            try:
                board = coro.send(result)
                next_coros.append(coro)
                next_boards.append(board)
            except StopIteration as stop:
                yield stop.value  # (records, winner)
                if started < num_games:
                    ncoro, nboard = new_coro()
                    next_coros.append(ncoro)
                    next_boards.append(nboard)
                    started += 1

        coros = next_coros
        boards = next_boards


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

    training_dataset = TrainingDataset()
    num_games = cfg.SELFPLAY_GAMES
    num_parallel = cfg.NUM_PARALLEL_GAMES

    print(f"Self-play: {num_games} games, {num_parallel} concurrent (batched inference).")

    completed = run_batched_selfplay(vpn, game, num_games, num_parallel)
    for game_number, (records, winner) in enumerate(tqdm(completed, total=num_games)):
        training_dataset.add_game_to_training_dataset(records, winner)

        if game_number % 50 == 0:
            training_dataset.save(save_path)
            print(f"saving.... game {game_number}, samples={len(training_dataset.training_dataset)}")

    training_dataset.save(save_path)
    print(f"Self-play complete. Total samples: {len(training_dataset.training_dataset)}")


if __name__ == "__main__":
    main()
