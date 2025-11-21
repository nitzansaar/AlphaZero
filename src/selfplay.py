import os
from config import Config as cfg
from game import TicTacToe
from mcts import MonteCarloTreeSearch
from dataset import TrainingDataset
from tqdm import tqdm
from value_policy_function import ValuePolicyNetwork
from copy import copy
from glob import glob

os.makedirs(cfg.SAVE_PICKLES, exist_ok=True) # create the save path if it doesn't exist
save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)


game = TicTacToe()

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
        # vpn = ValuePolicyNetwork()
        exit()
else:
    print("No trained models found. Using randomly initialized network.")
    exit()
    # vpn = ValuePolicyNetwork()

policy_value_network = vpn.get_vp
mcts = MonteCarloTreeSearch(game, policy_value_network)
root_node = mcts.init_root_node()
num_games = cfg.SELFPLAY_GAMES


training_dataset = TrainingDataset()
for game_number in tqdm(range(num_games),total=num_games): # play 2500 games
    node = root_node # start with an empty board
    dataset = []
    player = 1  # initialize player (game starts with player 1)
    while game.win_or_draw(node.state) == None: # while the game is not over
        parent_state = copy(node.state)
        node = mcts.run_simulation(root_node=node, num_simulations=1600, player=player) # run mcts to find the best action
        action, node, action_probs = mcts.select_move(node=node, mode="explore", temperature=1) # select the action with the probability of the visits
        dataset.append([parent_state, action_probs, player]) # board state, action probabilities, player
        player = -1 * player # switch player
    winner = game.get_reward_for_next_player(node.state, player) # assign value based on the winner
    training_dataset.add_game_to_training_dataset(dataset, winner)
    if game_number % 500 == 0: # save the training dataset every 500 games
        training_dataset.save(save_path) 
        print("saving....", game_number)

    
training_dataset.save(save_path)