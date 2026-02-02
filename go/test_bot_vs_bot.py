import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
import torch
import json
from config import Config as cfg
from game import Go, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE
from mcts import MonteCarloTreeSearch
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def format_board_state(state):
    """
    Convert board state to a readable 2D representation.
    Returns a 5x5 grid with 'X' for player 1, 'O' for player -1, '.' for empty
    """
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


def format_board_as_string(board_state):
    """
    Format the board state as a readable 5x5 grid string.
    """
    lines = []
    for i, row in enumerate(board_state):
        row_str = " | ".join(row)
        lines.append(f"  {row_str}")
        if i < len(board_state) - 1:
            lines.append("  " + "-" * 17)
    return "\n".join(lines)


def format_visit_counts(visit_counts, action_index=None):
    """
    Format visit counts as a 5x5 grid string plus pass action.
    """
    board_visits = visit_counts[:NUM_POSITIONS]
    visit_grid = np.array(board_visits).reshape(BOARD_SIZE, BOARD_SIZE)
    lines = []
    for i in range(BOARD_SIZE):
        row_strs = []
        for j in range(BOARD_SIZE):
            idx = i * BOARD_SIZE + j
            count = visit_grid[i, j]
            if action_index is not None and idx == action_index:
                row_strs.append(f"{int(count):4d}*")
            else:
                row_strs.append(f"{int(count):4d}")
        lines.append("  " + " | ".join(row_strs))
        if i < BOARD_SIZE - 1:
            lines.append("  " + "-" * 35)
    # Add pass action
    pass_count = visit_counts[PASS_ACTION] if len(visit_counts) > PASS_ACTION else 0
    pass_marker = "*" if action_index == PASS_ACTION else ""
    lines.append(f"  Pass: {int(pass_count)}{pass_marker}")
    return "\n".join(lines)


def get_top_moves(visit_counts, top_k=5):
    """
    Get top k moves with their visit counts and coordinates.
    """
    indices = np.argsort(visit_counts)[::-1][:top_k]
    result = []
    for idx in indices:
        if visit_counts[idx] > 0:
            if idx == PASS_ACTION:
                result.append((int(idx), int(visit_counts[idx]), "pass"))
            else:
                result.append((int(idx), int(visit_counts[idx]), action_index_to_coords(idx)))
    return result


def action_index_to_coords(action_index):
    """Convert action index to row, col coordinates or 'pass'"""
    if action_index == PASS_ACTION:
        return "pass"
    row = action_index // BOARD_SIZE
    col = action_index % BOARD_SIZE
    return (row, col)


def format_board_with_highlight(board_state, highlight_row=None, highlight_col=None):
    """
    Format board state with visible highlighting for a specific cell.
    """
    formatted_rows = []
    for i, row in enumerate(board_state):
        formatted_cells = []
        for j, cell in enumerate(row):
            if highlight_row is not None and highlight_col is not None and i == highlight_row and j == highlight_col:
                formatted_cells.append(f"*{cell}*")
            else:
                formatted_cells.append(f" {cell} ")
        formatted_rows.append(" ".join(formatted_cells))
    return formatted_rows


def play_game_bot_vs_bot(game, mcts1, mcts2, num_simulations=800, temperature=1.2):
    """
    Play a single game with bot1 going first (player 1) and bot2 as player -1
    """
    from mcts import Node

    player = 1
    state = game.state.copy()  # Use proper Go state (27 values)
    game_history = []
    move_number = 0

    # Record initial state
    board_state_formatted = format_board_state(state)
    game_history.append({
        'move': move_number,
        'player': None,
        'action': None,
        'action_coords': None,
        'board_state': board_state_formatted,
        'board_state_formatted': format_board_as_string(board_state_formatted),
    })

    while game.win_or_draw(state) is None:
        move_number += 1

        current_mcts = mcts1 if player == 1 else mcts2
        bot_name = 'bot1' if player == 1 else 'bot2'

        node = Node(prior_prob=0, player=player, action_index=None)
        node.set_state(state.copy())
        root_node = current_mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)

        visit_counts = np.zeros(ACTION_SIZE)
        for k, v in root_node.children.items():
            visit_counts[k] = v.total_visits_N
        visit_counts_list = [int(v) for v in visit_counts]
        top_moves = get_top_moves(visit_counts, top_k=5)

        action, node, action_probs = current_mcts.select_move(node=root_node, mode="exploit", temperature=temperature)
        action_index = np.argmax(action)

        state = game.apply_move(state, action_index, player)

        board_state_formatted = format_board_state(state)
        coords = action_index_to_coords(action_index)
        game_history.append({
            'move': move_number,
            'player': bot_name,
            'player_number': player,
            'action_index': int(action_index),
            'action_coords': coords,
            'board_state': board_state_formatted,
            'board_state_formatted': format_board_as_string(board_state_formatted),
            'visit_counts': visit_counts_list,
            'visit_counts_formatted': format_visit_counts(visit_counts, action_index),
            'top_moves': top_moves
        })

        player = -1 * player

    winner = game.get_winner(state)
    if winner == 1:
        result = 1
    elif winner == -1:
        result = -1
    else:
        result = 0

    return result, game_history


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

    vpn1 = ValuePolicyNetwork(model_path)
    policy_value_network1 = vpn1.get_vp
    mcts1 = MonteCarloTreeSearch(game, policy_value_network1)

    vpn2 = ValuePolicyNetwork(model_path)
    policy_value_network2 = vpn2.get_vp
    mcts2 = MonteCarloTreeSearch(game, policy_value_network2)

    num_games = cfg.NUM_GAMES
    num_simulations = cfg.NUM_SIMULATIONS
    temperature = 1

    print(f"\nTesting Bot vs Bot (Self-Play) - 5x5 Go")
    print(f"Total games: {num_games}")
    print(f"MCTS simulations per move: {num_simulations}")
    print(f"Temperature: {temperature}")
    print("=" * 60)

    results = []
    all_games_history = []

    print("\nPlaying bot vs bot games:")
    for game_num in tqdm(range(num_games), total=num_games):
        if game_num % 2 == 0:
            result, game_history = play_game_bot_vs_bot(game, mcts1, mcts2, num_simulations, temperature)
            first_bot = 'bot1'
        else:
            result, game_history = play_game_bot_vs_bot(game, mcts2, mcts1, num_simulations, temperature)
            result = -result
            first_bot = 'bot2'

        outcome = 'bot1_win' if result == 1 else ('draw' if result == 0 else 'bot2_win')

        results.append({
            'game_number': game_num,
            'first_player': first_bot,
            'result': result,
            'outcome': outcome
        })

        all_games_history.append({
            'game_number': game_num,
            'first_player': first_bot,
            'result': result,
            'outcome': outcome,
            'moves': game_history
        })

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total_games = len(df_results)
    bot1_wins = len(df_results[df_results['outcome'] == 'bot1_win'])
    draws = len(df_results[df_results['outcome'] == 'draw'])
    bot2_wins = len(df_results[df_results['outcome'] == 'bot2_win'])

    print(f"\nOverall Performance:")
    print(f"  Total Games:    {total_games}")
    print(f"  Bot1 Wins:      {bot1_wins} ({bot1_wins/total_games*100:.1f}%)")
    print(f"  Draws:          {draws} ({draws/total_games*100:.1f}%)")
    print(f"  Bot2 Wins:      {bot2_wins} ({bot2_wins/total_games*100:.1f}%)")

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, cfg.TEST_OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    games_text_file = os.path.join(output_dir, "bot_vs_bot_games_readable.txt")
    if os.path.exists(games_text_file):
        os.remove(games_text_file)
    with open(games_text_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Bot vs Bot Self-Play - 5x5 Go Game Histories\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Games: {num_games}\n")
        f.write(f"MCTS Simulations: {num_simulations} per move\n\n")

        for game_data in all_games_history:
            game_num = game_data['game_number']
            first_player = game_data['first_player']
            outcome = game_data['outcome']
            result = game_data['result']

            f.write("-" * 80 + "\n")
            f.write(f"Game {game_num} | First Player: {first_player} | Outcome: {outcome}\n")
            f.write("-" * 80 + "\n")

            for move_data in game_data['moves']:
                move_num = move_data['move']
                player = move_data['player']
                action_coords = move_data.get('action_coords')
                board_state = move_data['board_state']

                if move_num == 0:
                    f.write(f"\nInitial Board:\n")
                    for row in board_state:
                        f.write("  " + " ".join(row) + "\n")
                else:
                    if action_coords == "pass":
                        f.write(f"\nMove {move_num}: {player.upper()} passes\n")
                    elif action_coords:
                        f.write(f"\nMove {move_num}: {player.upper()} plays at {action_coords}\n")
                    for row in board_state:
                        f.write("  " + " ".join(row) + "\n")

            f.write("\n")

    print(f"Readable game histories saved to: {games_text_file}")

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    outcome_counts = df_results['outcome'].value_counts()
    colors = {'bot1_win': 'blue', 'draw': 'gray', 'bot2_win': 'red'}
    outcome_colors = [colors.get(outcome, 'blue') for outcome in outcome_counts.index]

    ax.bar(outcome_counts.index, outcome_counts.values, color=outcome_colors)
    ax.set_xlabel('Outcome', fontsize=12)
    ax.set_ylabel('Number of Games', fontsize=12)
    ax.set_title('Bot vs Bot - 5x5 Go Results', fontsize=14, fontweight='bold')

    label_map = {'bot1_win': 'Bot1 Win', 'draw': 'Draw', 'bot2_win': 'Bot2 Win'}
    ax.set_xticklabels([label_map.get(outcome, outcome) for outcome in outcome_counts.index])

    for i, (outcome, count) in enumerate(outcome_counts.items()):
        percentage = count / total_games * 100
        ax.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bot_vs_bot_results.png"), dpi=150)
    plt.close()
    print(f"\nResults graph saved to: {os.path.join(output_dir, 'bot_vs_bot_results.png')}")
    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    main()
