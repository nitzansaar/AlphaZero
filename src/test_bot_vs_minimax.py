import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
import torch
from config import Config as cfg
from game import TicTacToe
from mcts import MonteCarloTreeSearch
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork
import matplotlib.pyplot as plt
from minimax_player import MinimaxPlayer

device = "cuda" if torch.cuda.is_available() else "cpu"


def format_board_state(state):
    """
    Convert board state to a readable 2D representation.
    Returns a 9x9 grid with 'X' for player 1, 'O' for player -1, '.' for empty
    """
    board_2d = state.reshape(9, 9)
    formatted = []
    for row in board_2d:
        formatted_row = []
        for cell in row:
            if cell == 1:
                formatted_row.append("X")
            elif cell == -1:
                formatted_row.append("O")
            else:
                formatted_row.append(".")
        formatted.append(formatted_row)
    return formatted


def format_board_as_string(board_state):
    lines = []
    for i, row in enumerate(board_state):
        row_str = " | ".join(row)
        lines.append(f"  {row_str}")
        if i < len(board_state) - 1:
            lines.append("  " + "-" * 33)
    return "\n".join(lines)


def action_index_to_coords(action_index):
    row = action_index // 9
    col = action_index % 9
    return (row, col)


def format_visit_counts(visit_counts, action_index=None):
    visit_grid = np.array(visit_counts).reshape(9, 9)
    lines = []
    for i in range(9):
        row_strs = []
        for j in range(9):
            idx = i * 9 + j
            count = visit_grid[i, j]
            if action_index is not None and idx == action_index:
                row_strs.append(f"{int(count):4d}*")
            else:
                row_strs.append(f"{int(count):4d}")
        lines.append("  " + " | ".join(row_strs))
        if i < 8:
            lines.append("  " + "-" * 60)
    return "\n".join(lines)


def get_top_moves(visit_counts, top_k=5):
    indices = np.argsort(visit_counts)[::-1][:top_k]
    return [(int(idx), int(visit_counts[idx]), action_index_to_coords(idx)) for idx in indices if visit_counts[idx] > 0]


def format_board_with_highlight(board_state, highlight_row=None, highlight_col=None):
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


def play_game_bot_first(game, mcts, minimax_player, num_simulations=1600):
    """
    Bot goes first (player 1), minimax is player -1.
    Returns: (result, game_history) where result is 1 if bot wins, -1 if minimax wins, 0 if draw
    """
    from mcts import Node

    player = 1
    state = np.zeros(cfg.ACTION_SIZE)  # canonicalized state
    absolute_state = np.zeros(cfg.ACTION_SIZE)  # absolute board
    game_history = []
    move_number = 0

    board_state_formatted = format_board_state(absolute_state.copy())
    game_history.append(
        {
            "move": move_number,
            "player": None,
            "action": None,
            "action_coords": None,
            "board_state": board_state_formatted,
            "board_state_formatted": format_board_as_string(board_state_formatted),
            "board_state_flat": [float(x) for x in absolute_state.copy().tolist()],
        }
    )

    while game.win_or_draw(absolute_state) is None:
        move_number += 1
        if player == 1:  # bot
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)

            visit_counts = np.zeros(cfg.ACTION_SIZE)
            for k, v in root_node.children.items():
                visit_counts[k] = v.total_visits_N
            visit_counts_list = [int(v) for v in visit_counts]
            top_moves = get_top_moves(visit_counts_list, top_k=5)

            action, next_node, _ = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = int(np.argmax(action))

            state = next_node.state.copy()
            absolute_state[action_index] = 1

            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append(
                {
                    "move": move_number,
                    "player": "bot",
                    "player_number": 1,
                    "action_index": int(action_index),
                    "action_coords": tuple(action_index_to_coords(action_index)),
                    "board_state": board_state_formatted,
                    "board_state_formatted": format_board_as_string(board_state_formatted),
                    "board_state_flat": [float(x) for x in absolute_state.copy().tolist()],
                    "visit_counts": visit_counts_list,
                    "visit_counts_formatted": format_visit_counts(visit_counts_list, action_index),
                    "top_moves": top_moves,
                }
            )
        else:  # minimax
            action_index = minimax_player.get_action_index(absolute_state, player)
            if action_index is None:
                break
            action = np.zeros(cfg.ACTION_SIZE, dtype=np.float32)
            action[action_index] = 1.0

            state = game.get_next_state_from_next_player_prespective(state, action, player)
            absolute_state[action_index] = -1

            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append(
                {
                    "move": move_number,
                    "player": "minimax",
                    "player_number": -1,
                    "action_index": int(action_index),
                    "action_coords": tuple(action_index_to_coords(action_index)),
                    "board_state": board_state_formatted,
                    "board_state_formatted": format_board_as_string(board_state_formatted),
                    "board_state_flat": [float(x) for x in absolute_state.copy().tolist()],
                }
            )

        player *= -1

    winner = game.get_reward_for_next_player(absolute_state, player)
    if winner == 1:
        result = 1
    elif winner == -1:
        result = -1
    else:
        result = 0

    return result, game_history


def play_game_minimax_first(game, mcts, minimax_player, num_simulations=1600):
    """
    Minimax goes first (player 1), bot is player -1.
    Returns: (result, game_history) where result is 1 if bot wins, -1 if minimax wins, 0 if draw
    """
    from mcts import Node

    player = 1
    state = np.zeros(cfg.ACTION_SIZE)  # canonicalized state
    absolute_state = np.zeros(cfg.ACTION_SIZE)  # absolute board
    game_history = []
    move_number = 0

    board_state_formatted = format_board_state(absolute_state.copy())
    game_history.append(
        {
            "move": move_number,
            "player": None,
            "action": None,
            "action_coords": None,
            "board_state": board_state_formatted,
            "board_state_formatted": format_board_as_string(board_state_formatted),
            "board_state_flat": [float(x) for x in absolute_state.copy().tolist()],
        }
    )

    while game.win_or_draw(absolute_state) is None:
        move_number += 1
        if player == 1:  # minimax
            action_index = minimax_player.get_action_index(absolute_state, player)
            if action_index is None:
                break
            action = np.zeros(cfg.ACTION_SIZE, dtype=np.float32)
            action[action_index] = 1.0

            state = game.get_next_state_from_next_player_prespective(state, action, player)
            absolute_state[action_index] = 1

            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append(
                {
                    "move": move_number,
                    "player": "minimax",
                    "player_number": 1,
                    "action_index": int(action_index),
                    "action_coords": tuple(action_index_to_coords(action_index)),
                    "board_state": board_state_formatted,
                    "board_state_formatted": format_board_as_string(board_state_formatted),
                    "board_state_flat": [float(x) for x in absolute_state.copy().tolist()],
                }
            )
        else:  # bot
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)

            visit_counts = np.zeros(cfg.ACTION_SIZE)
            for k, v in root_node.children.items():
                visit_counts[k] = v.total_visits_N
            visit_counts_list = [int(v) for v in visit_counts]
            top_moves = get_top_moves(visit_counts_list, top_k=5)

            action, next_node, _ = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = int(np.argmax(action))

            state = next_node.state.copy()
            absolute_state[action_index] = -1

            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append(
                {
                    "move": move_number,
                    "player": "bot",
                    "player_number": -1,
                    "action_index": int(action_index),
                    "action_coords": tuple(action_index_to_coords(action_index)),
                    "board_state": board_state_formatted,
                    "board_state_formatted": format_board_as_string(board_state_formatted),
                    "board_state_flat": [float(x) for x in absolute_state.copy().tolist()],
                    "visit_counts": visit_counts_list,
                    "visit_counts_formatted": format_visit_counts(visit_counts_list, action_index),
                    "top_moves": top_moves,
                }
            )

        player *= -1

    winner = game.get_reward_for_next_player(absolute_state, player)
    if winner == -1:  # bot is player -1 in this setup
        result = 1
    elif winner == 1:
        result = -1
    else:
        result = 0

    return result, game_history


def _load_latest_compatible_model(possible_model_dirs):
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
                models_with_time.append((os.path.getmtime(f), f))
            except OSError:
                continue
        models_with_time.sort(reverse=True)

        for _, model_file in models_with_time:
            try:
                test_model = NeuralNetwork().to(device)
                test_state = torch.load(model_file, map_location=device)
                test_model.load_state_dict(test_state)
                model_path = model_file
                break
            except (RuntimeError, FileNotFoundError):
                continue
            finally:
                if "test_model" in locals():
                    del test_model
                if "test_state" in locals():
                    del test_state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if model_path is None:
            files_with_numbers = []
            for f in all_models:
                basename = os.path.basename(f)
                if "_best_model.pt" in basename:
                    try:
                        num = int(basename.split("_")[0])
                        files_with_numbers.append((num, f))
                    except ValueError:
                        continue
            if files_with_numbers:
                _, model_path = max(files_with_numbers, key=lambda x: x[0])
    return model_path


def main():
    game = TicTacToe()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_model_dirs = [
        os.path.join(script_dir, cfg.SAVE_MODEL_PATH),
        cfg.SAVE_MODEL_PATH,
        os.path.join(script_dir, "..", cfg.SAVE_MODEL_PATH),
    ]

    model_path = _load_latest_compatible_model(possible_model_dirs)
    if model_path is None or not os.path.exists(model_path):
        print("ERROR: No compatible model file found!")
        print("Searched in:")
        for model_dir in possible_model_dirs:
            print(f"  - {os.path.abspath(model_dir)}")
        return

    print(f"Loading model from: {model_path}")
    vpn = ValuePolicyNetwork(model_path)
    policy_value_network = vpn.get_vp
    mcts = MonteCarloTreeSearch(game, policy_value_network)

    minimax_depth = 2
    minimax_player = MinimaxPlayer(game, depth=minimax_depth, radius=1, max_candidates=24)

    num_games = cfg.NUM_GAMES
    num_simulations = cfg.NUM_SIMULATIONS

    print(f"\nTesting AlphaZero bot vs Minimax Player (depth={minimax_depth})")
    print(f"Total games: {num_games}")
    print(f"MCTS simulations per move: {num_simulations}")
    print(f"Games per player position: {num_games // 2}")
    print("=" * 60)

    results = []
    all_games_history = []

    print("\nPlaying bot vs minimax games (alternating first/second):")
    for game_num in tqdm(range(num_games), total=num_games):
        if game_num % 2 == 0:
            result, game_history = play_game_bot_first(game, mcts, minimax_player, num_simulations)
            bot_position = "first"
            bot_player = 1
        else:
            result, game_history = play_game_minimax_first(game, mcts, minimax_player, num_simulations)
            bot_position = "second"
            bot_player = -1

        outcome = "bot_win" if result == 1 else ("draw" if result == 0 else "minimax_win")
        results.append(
            {
                "game_number": game_num,
                "bot_position": bot_position,
                "bot_player": bot_player,
                "result": result,
                "outcome": outcome,
            }
        )
        all_games_history.append(
            {
                "game_number": game_num,
                "bot_position": bot_position,
                "bot_player": bot_player,
                "result": result,
                "outcome": outcome,
                "moves": game_history,
            }
        )

    df_results = pd.DataFrame(results)
    total_games = len(df_results)
    bot_wins = len(df_results[df_results["outcome"] == "bot_win"])
    draws = len(df_results[df_results["outcome"] == "draw"])
    minimax_wins = len(df_results[df_results["outcome"] == "minimax_win"])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nOverall Performance:")
    print(f"  Total Games:    {total_games}")
    print(f"  Bot Wins:       {bot_wins} ({bot_wins/total_games*100:.1f}%)")
    print(f"  Draws:          {draws} ({draws/total_games*100:.1f}%)")
    print(f"  Minimax Wins:   {minimax_wins} ({minimax_wins/total_games*100:.1f}%)")
    print(f"  Bot No-Loss:    {(bot_wins + draws)/total_games*100:.1f}%")

    print(f"\nBot as Player 1 (going first):")
    first_games = df_results[df_results["bot_position"] == "first"]
    first_wins = len(first_games[first_games["outcome"] == "bot_win"])
    first_draws = len(first_games[first_games["outcome"] == "draw"])
    first_losses = len(first_games[first_games["outcome"] == "minimax_win"])
    print(f"  Wins:  {first_wins} ({first_wins/len(first_games)*100:.1f}%)")
    print(f"  Draws: {first_draws} ({first_draws/len(first_games)*100:.1f}%)")
    print(f"  Losses: {first_losses} ({first_losses/len(first_games)*100:.1f}%)")

    print(f"\nBot as Player -1 (going second):")
    second_games = df_results[df_results["bot_position"] == "second"]
    second_wins = len(second_games[second_games["outcome"] == "bot_win"])
    second_draws = len(second_games[second_games["outcome"] == "draw"])
    second_losses = len(second_games[second_games["outcome"] == "minimax_win"])
    print(f"  Wins:  {second_wins} ({second_wins/len(second_games)*100:.1f}%)")
    print(f"  Draws: {second_draws} ({second_draws/len(second_games)*100:.1f}%)")
    print(f"  Losses: {second_losses} ({second_losses/len(second_games)*100:.1f}%)")

    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    games_text_file = os.path.join(output_dir, "bot_vs_minimax_games_readable.txt")
    if os.path.exists(games_text_file):
        os.remove(games_text_file)

    with open(games_text_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("AlphaZero Bot vs Minimax Player - Game Histories\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Games: {num_games}\n")
        f.write(f"MCTS Simulations: {num_simulations} per move\n")
        f.write(f"Minimax Depth: {minimax_depth}\n\n")

        for game_data in all_games_history:
            game_num = game_data["game_number"]
            bot_pos = game_data["bot_position"]
            outcome = game_data["outcome"]
            result = game_data["result"]

            f.write("-" * 80 + "\n")
            f.write(f"Game {game_num} | Bot Position: {bot_pos} | Outcome: {outcome} (Result: {result})\n")
            f.write("-" * 80 + "\n")

            for move_data in game_data["moves"]:
                move_num = move_data["move"]
                player_name = move_data["player"]
                action_coords = move_data["action_coords"]
                board_state = move_data["board_state"]

                if move_num == 0:
                    f.write("\nInitial Board:\n")
                    for row in board_state:
                        f.write("  " + " ".join(row) + "\n")
                else:
                    if action_coords:
                        f.write(f"\nMove {move_num}: {player_name.upper()} plays at ({action_coords[0]}, {action_coords[1]})\n")
                    else:
                        f.write(f"\nMove {move_num}: {player_name.upper()}\n")

                    highlighted_rows = format_board_with_highlight(
                        board_state, action_coords[0] if action_coords else None, action_coords[1] if action_coords else None
                    )
                    for row in highlighted_rows:
                        f.write("  " + row + "\n")

                if player_name == "bot" and "visit_counts_formatted" in move_data:
                    f.write("\nBot's MCTS Visit Counts:\n")
                    f.write(move_data["visit_counts_formatted"])
                    f.write("\n")
                    if "top_moves" in move_data and move_data["top_moves"]:
                        f.write("Top 5 moves considered:\n")
                        for idx, (action_idx, visit_count, (r, c)) in enumerate(move_data["top_moves"], 1):
                            marker = " <- SELECTED" if action_idx == move_data["action_index"] else ""
                            f.write(f"  {idx}. Position ({r}, {c}): {visit_count} visits{marker}\n")
                    f.write("\n")

            f.write("\n")

    print(f"Readable game histories saved to: {games_text_file}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    outcome_counts = df_results["outcome"].value_counts()
    colors = {"bot_win": "green", "draw": "gray", "minimax_win": "red"}
    outcome_colors = [colors.get(outcome, "green") for outcome in outcome_counts.index]

    ax1.bar(outcome_counts.index, outcome_counts.values, color=outcome_colors)
    ax1.set_xlabel("Outcome")
    ax1.set_ylabel("Number of Games")
    ax1.set_title("Overall Results Distribution")
    label_map = {"bot_win": "Bot Win", "draw": "Draw", "minimax_win": "Minimax Win"}
    ax1.set_xticklabels([label_map.get(outcome, outcome) for outcome in outcome_counts.index])

    for i, (outcome, count) in enumerate(outcome_counts.items()):
        percentage = count / total_games * 100
        ax1.text(i, count, f"{count}\n({percentage:.1f}%)", ha="center", va="bottom")

    positions = ["first", "second"]
    outcomes = ["bot_win", "draw", "minimax_win"]
    x = np.arange(len(positions))
    width = 0.25

    for i, outcome in enumerate(outcomes):
        counts = [len(df_results[(df_results["bot_position"] == pos) & (df_results["outcome"] == outcome)]) for pos in positions]
        offset = (i - 1) * width
        ax2.bar(x + offset, counts, width, label=outcome.replace("_", " ").title(), color=colors.get(outcome, "green"))

    ax2.set_xlabel("Bot Position")
    ax2.set_ylabel("Number of Games")
    ax2.set_title("Results by Bot Position")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Bot First (X)", "Bot Second (O)"])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    results_plot = os.path.join(output_dir, "bot_vs_minimax_results.png")
    plt.savefig(results_plot, dpi=150)
    plt.close()
    print(f"\nResults graph saved to: {results_plot}")
    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    main()

