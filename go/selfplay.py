import os
from config import Config as cfg
from game import Go, PASS_ACTION
from mcts import MonteCarloTreeSearch
from dataset import TrainingDataset
from tqdm import tqdm
from value_policy_function import ValuePolicyNetwork
from copy import copy
from glob import glob

# Maximum moves per game - force end if exceeded (5x5 board shouldn't need more than this)
MAX_MOVES_PER_GAME = 100

os.makedirs(cfg.SAVE_PICKLES, exist_ok=True)
save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)

game = Go()

# Load the latest trained model if available, or use random initialization
all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*.pt"))
if all_models:
    files = [int(os.path.basename(f).split("_")[0]) for f in all_models if os.path.basename(f).split("_")[0].isdigit()]
    if files:
        latest_num = max(files) # get the latest file number
        model_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(latest_num)) # get the path to the latest trained model
        print(f"Loading trained model: {model_path}")
        vpn = ValuePolicyNetwork(path=model_path)
    else:
        print("No trained models found. Using randomly initialized network.")
        vpn = ValuePolicyNetwork(path=None)
else:
    print("No trained models found. Using randomly initialized network.")
    vpn = ValuePolicyNetwork(path=None)

policy_value_network = vpn.get_vp
mcts = MonteCarloTreeSearch(game, policy_value_network)
root_node = mcts.init_root_node()
num_games = cfg.SELFPLAY_GAMES

training_dataset = TrainingDataset()
force_ended_games = 0

for game_number in tqdm(range(num_games), total=num_games):
    node = root_node  # start with an empty board
    dataset = []
    player = 1  # initialize player (game starts with player 1 / black)
    move_count = 0
    force_ended = False

    while game.win_or_draw(node.state, perspective=player) is None:
        # Check if we've exceeded max moves
        if move_count >= MAX_MOVES_PER_GAME:
            force_ended = True
            break

        parent_state = copy(node.state)
        node = mcts.run_simulation(root_node=node, num_simulations=cfg.NUM_SIMULATIONS, player=player)

        # Temperature decay: use high temperature early, low temperature later
        if move_count < cfg.TEMP_THRESHOLD:
            temperature = cfg.INITIAL_TEMP
        else:
            temperature = 0.1  # Near-deterministic for later moves

        action, node, action_probs = mcts.select_move(node=node, mode="explore", temperature=temperature)
        dataset.append([parent_state, action_probs, player])
        player = -1 * player  # switch player
        move_count += 1

    # Skip force-ended games, only save completed games
    if force_ended:
        continue

    winner = game.win_or_draw(node.state, perspective=player)
    training_dataset.add_game_to_training_dataset(dataset, winner)

    if game_number % 500 == 0:
        training_dataset.save(save_path)
        print("saving....", game_number)

training_dataset.save(save_path)
