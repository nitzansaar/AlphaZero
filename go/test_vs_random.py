import os
import argparse
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

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Batch size for MCTS (number of leaf nodes to evaluate in parallel)
MCTS_BATCH_SIZE = 32


def format_board_state(state, last_move=None):
    """Convert board state to a readable 2D representation with Go-style grid.

    Args:
        state: The game state
        last_move: Tuple (row, col) of the last move, or 'pass', or None

    Returns:
        List of strings representing the board
    """
    board = state[:NUM_POSITIONS]
    board_2d = board.reshape(BOARD_SIZE, BOARD_SIZE)

    # Symbols: ○ for Black, ● for White, + for empty
    # Last move is highlighted with brackets: [●] or [○]
    lines = []

    # Column headers
    header = "     " + "   ".join([str(i) for i in range(BOARD_SIZE)])
    lines.append(header)
    lines.append("")

    for row_idx in range(BOARD_SIZE):
        row_str = f" {row_idx}   "
        for col_idx in range(BOARD_SIZE):
            cell = board_2d[row_idx, col_idx]
            is_last_move = (last_move == (row_idx, col_idx))

            if cell == 1:
                symbol = "[○]" if is_last_move else " ○ "
            elif cell == -1:
                symbol = "[●]" if is_last_move else " ● "
            else:
                symbol = " + "

            if col_idx < BOARD_SIZE - 1:
                # Remove trailing space and add connector
                row_str += symbol.rstrip() + "──"
            else:
                row_str += symbol.rstrip()

        row_str += f"   {row_idx}"
        lines.append(row_str)

        # Vertical connectors (except after last row)
        if row_idx < BOARD_SIZE - 1:
            connector = "     " + "│   " * (BOARD_SIZE - 1) + "│"
            lines.append(connector)

    lines.append("")
    lines.append(header)

    return lines


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


def get_visit_counts(root_node):
    """Extract visit counts from MCTS root node children."""
    visit_counts = {}
    for action_idx, child in root_node.children.items():
        coords = action_index_to_coords(action_idx)
        visit_counts[coords] = child.total_visits_N
    # Sort by visit count descending
    sorted_visits = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_visits


def play_game_bot_first(game, mcts, random_player, num_simulations=1200):
    """Play a single game with the bot going first (player 1)"""
    from mcts import Node

    player = 1
    state = game.state.copy()
    move_number = 0
    history = []

    # Record initial state
    history.append({
        'move_number': 0,
        'player': None,
        'action': None,
        'board': format_board_state(state, last_move=None),
        'visit_counts': None
    })

    while game.winner(state) is None:
        move_number += 1
        if player == 1:  # Bot's turn
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation_batched(
                root_node=node, num_simulations=num_simulations, player=player, add_noise=False, batch_size=MCTS_BATCH_SIZE
            )
            visit_counts = get_visit_counts(root_node)
            action, node, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = np.argmax(action)
            state = game.apply_move(state, action_index, player)
            last_move = action_index_to_coords(action_index)
            history.append({
                'move_number': move_number,
                'player': 'Bot (Black/○)',
                'action': last_move,
                'board': format_board_state(state, last_move=last_move if last_move != "pass" else None),
                'visit_counts': visit_counts
            })
        else:  # Random player's turn
            action = random_player.get_action(state, player)
            if action is None:
                break
            action_index = np.argmax(action)
            state = game.apply_move(state, action_index, player)
            last_move = action_index_to_coords(action_index)
            history.append({
                'move_number': move_number,
                'player': 'Random (White/●)',
                'action': last_move,
                'board': format_board_state(state, last_move=last_move if last_move != "pass" else None),
                'visit_counts': None
            })

        player = -1 * player

    winner = game.get_winner(state)
    if winner == 1:
        result = 1  # Bot wins
    elif winner == -1:
        result = -1  # Random wins
    else:
        result = 0

    return result, history


def play_game_random_first(game, mcts, random_player, num_simulations=1200):
    """Play a single game with random player going first (player 1), bot is player -1"""
    from mcts import Node

    player = 1
    state = game.state.copy()
    move_number = 0
    history = []

    # Record initial state
    history.append({
        'move_number': 0,
        'player': None,
        'action': None,
        'board': format_board_state(state, last_move=None),
        'visit_counts': None
    })

    while game.winner(state) is None:
        move_number += 1
        if player == -1:  # Bot's turn
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation_batched(
                root_node=node, num_simulations=num_simulations, player=player, add_noise=False, batch_size=MCTS_BATCH_SIZE
            )
            visit_counts = get_visit_counts(root_node)
            action, node, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = np.argmax(action)
            state = game.apply_move(state, action_index, player)
            last_move = action_index_to_coords(action_index)
            history.append({
                'move_number': move_number,
                'player': 'Bot (White/●)',
                'action': last_move,
                'board': format_board_state(state, last_move=last_move if last_move != "pass" else None),
                'visit_counts': visit_counts
            })
        else:  # Random player's turn
            action = random_player.get_action(state, player)
            if action is None:
                break
            action_index = np.argmax(action)
            state = game.apply_move(state, action_index, player)
            last_move = action_index_to_coords(action_index)
            history.append({
                'move_number': move_number,
                'player': 'Random (Black/○)',
                'action': last_move,
                'board': format_board_state(state, last_move=last_move if last_move != "pass" else None),
                'visit_counts': None
            })

        player = -1 * player

    winner = game.get_winner(state)
    if winner == -1:
        result = 1  # Bot wins
    elif winner == 1:
        result = -1  # Random wins
    else:
        result = 0

    return result, history


def main():
    parser = argparse.ArgumentParser(description='Test AlphaZero bot vs Random Player on 5x5 Go')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to the model file to use. If not specified, uses the most recent model.')
    args = parser.parse_args()

    game = Go()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = None

    if args.model:
        # User specified a model path
        if os.path.exists(args.model):
            model_path = args.model
        else:
            # Try relative to script directory
            candidate = os.path.join(script_dir, args.model)
            if os.path.exists(candidate):
                model_path = candidate
            else:
                # Try in the models directory
                candidate = os.path.join(script_dir, cfg.SAVE_MODEL_PATH, args.model)
                if os.path.exists(candidate):
                    model_path = candidate

        if model_path is None:
            print(f"ERROR: Specified model not found: {args.model}")
            return
        print(f"Using specified model: {model_path}")
    else:
        # Default behavior: find the most recent model
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

        if model_path is None:
            print(f"ERROR: No model file found!")
            print(f"Searched in:")
            for model_dir in possible_model_dirs:
                print(f"  - {os.path.abspath(model_dir)}")
            return

    print(f"Loading model from: {model_path}")
    vpn = ValuePolicyNetwork(model_path)
    policy_value_network = vpn.get_vp
    policy_value_network_batch = vpn.get_vp_batch
    mcts = MonteCarloTreeSearch(game, policy_value_network, policy_value_network_batch)

    random_player = RandomPlayer(game)

    num_games = cfg.NUM_GAMES
    num_simulations = cfg.NUM_SIMULATIONS

    print(f"\nTesting AlphaZero bot vs Random Player - 5x5 Go")
    print(f"Total games: {num_games}")
    print(f"MCTS simulations per move: {num_simulations}")
    print("=" * 60)

    results = []
    all_game_histories = []

    print("\nPlaying bot vs random games (alternating first/second):")
    for game_num in tqdm(range(num_games), total=num_games):
        if game_num % 2 == 0:
            result, history = play_game_bot_first(game, mcts, random_player, num_simulations)
            bot_position = 'first'
        else:
            result, history = play_game_random_first(game, mcts, random_player, num_simulations)
            bot_position = 'second'

        outcome = 'bot_win' if result == 1 else ('draw' if result == 0 else 'random_win')

        results.append({
            'game_number': game_num,
            'bot_position': bot_position,
            'result': result,
            'outcome': outcome
        })

        all_game_histories.append({
            'game_number': game_num,
            'bot_position': bot_position,
            'outcome': outcome,
            'history': history
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
    output_dir = os.path.join(script_dir, cfg.TEST_OUTPUT_PATH)
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

    # Save game histories to text file
    history_file = os.path.join(output_dir, "game_history_log.txt")
    with open(history_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GAME HISTORY LOG - Bot vs Random Player (5x5 Go)\n")
        f.write(f"Total Games: {num_games} | MCTS Simulations: {num_simulations}\n")
        f.write("=" * 70 + "\n\n")

        for game_data in all_game_histories:
            game_num = game_data['game_number']
            bot_pos = game_data['bot_position']
            outcome = game_data['outcome']
            history = game_data['history']

            f.write("-" * 70 + "\n")
            f.write(f"GAME {game_num + 1}\n")
            f.write(f"Bot Position: {'First (Black/○)' if bot_pos == 'first' else 'Second (White/●)'}\n")
            f.write(f"Outcome: {outcome.replace('_', ' ').title()}\n")
            f.write(f"Legend: ○ = Black, ● = White, + = Empty, [○]/[●] = Last move\n")
            f.write("-" * 70 + "\n\n")

            for turn in history:
                move_num = turn['move_number']
                player = turn['player']
                action = turn['action']
                board_lines = turn['board']
                visit_counts = turn['visit_counts']

                if move_num == 0:
                    f.write("Initial Board State:\n")
                else:
                    f.write(f"Move {move_num}: {player} plays {action}\n")

                # Show visit counts for bot moves
                if visit_counts is not None:
                    f.write("  MCTS Visit Counts (top moves):\n")
                    for i, (coords, visits) in enumerate(visit_counts[:10]):  # Show top 10
                        f.write(f"    {coords}: {visits} visits\n")
                    f.write("\n")

                # Write the board (list of pre-formatted lines)
                for line in board_lines:
                    f.write(f"  {line}\n")
                f.write("\n")

            f.write("\n")

    print(f"Game history log saved to: {history_file}")
    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    main()
