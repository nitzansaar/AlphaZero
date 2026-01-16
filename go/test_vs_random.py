import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
import torch
from config import Config as cfg
from game import Go, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE
from mcts import MonteCarloTreeSearch
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def format_board_state(state):
    """Convert board state to a readable 2D representation."""
    board = state[:NUM_POSITIONS]
    board_2d = board.reshape(BOARD_SIZE, BOARD_SIZE)
    formatted = []
    for row in board_2d:
        formatted_row = []
        for cell in row:
            if cell == 1:
                formatted_row.append('X')
            elif cell == -1:
                formatted_row.append('O')
            else:
                formatted_row.append('.')
        formatted.append(formatted_row)
    return formatted


def action_index_to_coords(action_index):
    """Convert action index to row, col coordinates or 'pass'"""
    if action_index == PASS_ACTION:
        return "pass"
    row = action_index // BOARD_SIZE
    col = action_index % BOARD_SIZE
    return (row, col)


class RandomPlayer:
    """A player that makes completely random moves"""
    def __init__(self, game):
        self.game = game

    def get_action(self, state, player=1):
        """Select a random valid action"""
        valid_moves = self.game.get_valid_moves(state, player)
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) == 0:
            return None
        action_index = np.random.choice(valid_indices)
        action = np.zeros(ACTION_SIZE)
        action[action_index] = 1
        return action


def play_game_bot_first(game, mcts, random_player, num_simulations=1600):
    """Play a single game with the bot going first (player 1)"""
    from mcts import Node

    player = 1
    state = game.state.copy()
    move_number = 0

    while game.win_or_draw(state) is None:
        move_number += 1
        if player == 1:  # Bot's turn
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)
            action, node, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = np.argmax(action)
            state = game.apply_move(state, action_index, player)
        else:  # Random player's turn
            action = random_player.get_action(state, player)
            if action is None:
                break
            action_index = np.argmax(action)
            state = game.apply_move(state, action_index, player)

        player = -1 * player

    winner = game.get_winner(state)
    if winner == 1:
        result = 1  # Bot wins
    elif winner == -1:
        result = -1  # Random wins
    else:
        result = 0

    return result


def play_game_random_first(game, mcts, random_player, num_simulations=1600):
    """Play a single game with random player going first (player 1), bot is player -1"""
    from mcts import Node

    player = 1
    state = game.state.copy()
    move_number = 0

    while game.win_or_draw(state) is None:
        move_number += 1
        if player == -1:  # Bot's turn
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)
            action, node, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = np.argmax(action)
            state = game.apply_move(state, action_index, player)
        else:  # Random player's turn
            action = random_player.get_action(state, player)
            if action is None:
                break
            action_index = np.argmax(action)
            state = game.apply_move(state, action_index, player)

        player = -1 * player

    winner = game.get_winner(state)
    if winner == -1:
        result = 1  # Bot wins
    elif winner == 1:
        result = -1  # Random wins
    else:
        result = 0

    return result


def main():
    game = Go()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_model_dirs = [
        os.path.join(script_dir, cfg.SAVE_MODEL_PATH),
        cfg.SAVE_MODEL_PATH,
        os.path.join(script_dir, "..", cfg.SAVE_MODEL_PATH),
    ]

    all_models = []
    for model_dir in possible_model_dirs:
        if os.path.isdir(model_dir):
            found_models = glob(os.path.join(model_dir, "*_best_model.pt"))
            if found_models:
                all_models.extend(found_models)
                break

    model_path = None

    if all_models:
        models_with_time = []
        for f in all_models:
            try:
                mtime = os.path.getmtime(f)
                models_with_time.append((mtime, f))
            except OSError:
                continue

        if models_with_time:
            models_with_time.sort(reverse=True)

            for mtime, model_file in models_with_time:
                try:
                    test_model = NeuralNetwork().to(device)
                    test_state = torch.load(model_file, map_location=device)
                    test_model.load_state_dict(test_state)
                    model_path = model_file
                    model_name = os.path.basename(model_file)
                    print(f"Found {len(all_models)} model(s), using most recent compatible: {model_name}")
                    break
                except (RuntimeError, FileNotFoundError):
                    continue
                finally:
                    del test_model
                    if 'test_state' in locals():
                        del test_state
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if model_path is None or not os.path.exists(model_path):
        print(f"ERROR: No model file found!")
        print(f"Searched in:")
        for model_dir in possible_model_dirs:
            print(f"  - {os.path.abspath(model_dir)}")
        return

    print(f"Loading model from: {model_path}")
    vpn = ValuePolicyNetwork(model_path)
    policy_value_network = vpn.get_vp
    mcts = MonteCarloTreeSearch(game, policy_value_network)

    random_player = RandomPlayer(game)

    num_games = cfg.NUM_GAMES
    num_simulations = cfg.NUM_SIMULATIONS

    print(f"\nTesting AlphaZero bot vs Random Player - 5x5 Go")
    print(f"Total games: {num_games}")
    print(f"MCTS simulations per move: {num_simulations}")
    print("=" * 60)

    results = []

    print("\nPlaying bot vs random games (alternating first/second):")
    for game_num in tqdm(range(num_games), total=num_games):
        if game_num % 2 == 0:
            result = play_game_bot_first(game, mcts, random_player, num_simulations)
            bot_position = 'first'
        else:
            result = play_game_random_first(game, mcts, random_player, num_simulations)
            bot_position = 'second'

        outcome = 'bot_win' if result == 1 else ('draw' if result == 0 else 'random_win')

        results.append({
            'game_number': game_num,
            'bot_position': bot_position,
            'result': result,
            'outcome': outcome
        })

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total_games = len(df_results)
    bot_wins = len(df_results[df_results['outcome'] == 'bot_win'])
    draws = len(df_results[df_results['outcome'] == 'draw'])
    random_wins = len(df_results[df_results['outcome'] == 'random_win'])

    print(f"\nOverall Performance:")
    print(f"  Total Games:    {total_games}")
    print(f"  Bot Wins:       {bot_wins} ({bot_wins/total_games*100:.1f}%)")
    print(f"  Draws:          {draws} ({draws/total_games*100:.1f}%)")
    print(f"  Random Wins:    {random_wins} ({random_wins/total_games*100:.1f}%)")

    print(f"\nBot as Black (going first):")
    first_games = df_results[df_results['bot_position'] == 'first']
    first_wins = len(first_games[first_games['outcome'] == 'bot_win'])
    first_draws = len(first_games[first_games['outcome'] == 'draw'])
    first_losses = len(first_games[first_games['outcome'] == 'random_win'])
    if len(first_games) > 0:
        print(f"  Wins:  {first_wins} ({first_wins/len(first_games)*100:.1f}%)")
        print(f"  Draws: {first_draws} ({first_draws/len(first_games)*100:.1f}%)")
        print(f"  Losses: {first_losses} ({first_losses/len(first_games)*100:.1f}%)")

    print(f"\nBot as White (going second):")
    second_games = df_results[df_results['bot_position'] == 'second']
    second_wins = len(second_games[second_games['outcome'] == 'bot_win'])
    second_draws = len(second_games[second_games['outcome'] == 'draw'])
    second_losses = len(second_games[second_games['outcome'] == 'random_win'])
    if len(second_games) > 0:
        print(f"  Wins:  {second_wins} ({second_wins/len(second_games)*100:.1f}%)")
        print(f"  Draws: {second_draws} ({second_draws/len(second_games)*100:.1f}%)")
        print(f"  Losses: {second_losses} ({second_losses/len(second_games)*100:.1f}%)")

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    outcome_counts = df_results['outcome'].value_counts()
    colors = {'bot_win': 'green', 'draw': 'gray', 'random_win': 'red'}
    outcome_colors = [colors.get(outcome, 'green') for outcome in outcome_counts.index]

    ax1.bar(outcome_counts.index, outcome_counts.values, color=outcome_colors)
    ax1.set_xlabel('Outcome')
    ax1.set_ylabel('Number of Games')
    ax1.set_title('5x5 Go: Bot vs Random - Overall Results')

    label_map = {'bot_win': 'Bot Win', 'draw': 'Draw', 'random_win': 'Random Win'}
    ax1.set_xticklabels([label_map.get(outcome, outcome) for outcome in outcome_counts.index])

    for i, (outcome, count) in enumerate(outcome_counts.items()):
        percentage = count / total_games * 100
        ax1.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')

    positions = ['first', 'second']
    outcomes = ['bot_win', 'draw', 'random_win']
    x = np.arange(len(positions))
    width = 0.25

    for i, outcome in enumerate(outcomes):
        counts = [len(df_results[(df_results['bot_position'] == pos) & (df_results['outcome'] == outcome)])
                  for pos in positions]
        offset = (i - 1) * width
        ax2.bar(x + offset, counts, width, label=outcome.replace('_', ' ').title(),
                color=colors.get(outcome, 'green'))

    ax2.set_xlabel('Bot Position')
    ax2.set_ylabel('Number of Games')
    ax2.set_title('Results by Bot Position')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Bot First (Black)', 'Bot Second (White)'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bot_vs_random_results.png"), dpi=150)
    plt.close()
    print(f"\nResults graph saved to: {os.path.join(output_dir, 'bot_vs_random_results.png')}")
    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    main()
