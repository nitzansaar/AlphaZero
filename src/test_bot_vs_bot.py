import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
import torch
import json
from config import Config as cfg
from game import TicTacToe
from mcts import MonteCarloTreeSearch
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork
import matplotlib.pyplot as plt

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
                formatted_row.append('X')
            elif cell == -1:
                formatted_row.append('O')
            else:
                formatted_row.append('.')
        formatted.append(formatted_row)
    return formatted

def format_board_as_string(board_state):
    """
    Format the board state as a readable 9x9 grid string.
    Args:
        board_state: List of lists representing the board (from format_board_state)
    Returns:
        Multi-line string showing the board in a formatted way
    """
    lines = []
    for i, row in enumerate(board_state):
        row_str = " | ".join(row)
        lines.append(f"  {row_str}")
        if i < len(board_state) - 1:
            lines.append("  " + "-" * 33)  # Separator between rows (9 cells * 3 chars + 8 separators)
    return "\n".join(lines)

def format_visit_counts(visit_counts, action_index=None):
    """
    Format visit counts as a 9x9 grid string.
    Args:
        visit_counts: List of 81 visit counts
        action_index: Optional action index to highlight
    Returns:
        Multi-line string showing the visit counts
    """
    # Reshape to 9x9
    visit_grid = np.array(visit_counts).reshape(9, 9)
    lines = []
    for i in range(9):
        row_strs = []
        for j in range(9):
            idx = i * 9 + j
            count = visit_grid[i, j]
            if action_index is not None and idx == action_index:
                row_strs.append(f"{int(count):4d}*")  # Mark selected move
            else:
                row_strs.append(f"{int(count):4d}")
        lines.append("  " + " | ".join(row_strs))
        if i < 8:
            lines.append("  " + "-" * 60)  # Separator between rows (9 cells * 4 chars + 8 separators * 3 chars)
    return "\n".join(lines)

def get_top_moves(visit_counts, top_k=5):
    """
    Get top k moves with their visit counts and coordinates.
    Returns list of tuples: (action_index, visit_count, coords)
    """
    indices = np.argsort(visit_counts)[::-1][:top_k]
    return [(int(idx), int(visit_counts[idx]), action_index_to_coords(idx))
            for idx in indices if visit_counts[idx] > 0]

def action_index_to_coords(action_index):
    """Convert action index to row, col coordinates"""
    row = action_index // 9
    col = action_index % 9
    return (row, col)

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def play_game_bot_vs_bot(game, mcts1, mcts2, num_simulations=800, temperature=1.2):
    """
    Play a single game with bot1 going first (player 1) and bot2 as player -1

    Args:
        game: TicTacToe game instance
        mcts1: MCTS instance for bot1
        mcts2: MCTS instance for bot2
        num_simulations: Number of MCTS simulations per move
        temperature: Temperature for move selection (higher = more random, lower = more deterministic)
                    Default 1.2 adds variety while still favoring good moves

    Returns: (result, game_history) where result is 1 if bot1 wins, -1 if bot2 wins, 0 if draw
    """
    from mcts import Node

    player = 1
    state = np.zeros(cfg.ACTION_SIZE)  # Canonicalized state for MCTS
    absolute_state = np.zeros(cfg.ACTION_SIZE)  # Absolute state for display (player 1 = 1, player -1 = -1)
    game_history = []
    move_number = 0

    # Record initial state
    board_state_formatted = format_board_state(absolute_state.copy())
    game_history.append({
        'move': move_number,
        'player': None,
        'action': None,
        'action_coords': None,
        'board_state': board_state_formatted,
        'board_state_formatted': format_board_as_string(board_state_formatted),
        'board_state_flat': [float(x) for x in absolute_state.copy().tolist()]
    })

    while game.win_or_draw(absolute_state) == None:
        move_number += 1

        # Select which bot plays this turn
        current_mcts = mcts1 if player == 1 else mcts2
        bot_name = 'bot1' if player == 1 else 'bot2'

        # Create fresh node for current bot
        node = Node(prior_prob=0, player=player, action_index=None)
        node.set_state(state.copy())
        root_node = current_mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)

        # Extract visit counts directly from root node children BEFORE select_move
        visit_counts = np.zeros(cfg.ACTION_SIZE)
        for k, v in root_node.children.items():
            visit_counts[k] = v.total_visits_N
        visit_counts_list = [int(v) for v in visit_counts]
        top_moves = get_top_moves(visit_counts_list, top_k=5)

        action, node, action_probs = current_mcts.select_move(node=root_node, mode="explore", temperature=temperature)
        action_index = np.argmax(action)

        # Update canonicalized state
        state = node.state.copy()
        # Update absolute state
        absolute_state[action_index] = player

        # Record bot move
        board_state_formatted = format_board_state(absolute_state.copy())
        game_history.append({
            'move': move_number,
            'player': bot_name,
            'player_number': player,
            'action_index': int(action_index),
            'action_coords': tuple(action_index_to_coords(action_index)),
            'board_state': board_state_formatted,
            'board_state_formatted': format_board_as_string(board_state_formatted),
            'board_state_flat': [float(x) for x in absolute_state.copy().tolist()],
            'visit_counts': visit_counts_list,
            'visit_counts_formatted': format_visit_counts(visit_counts_list, action_index),
            'top_moves': top_moves
        })

        player = -1 * player

    # Determine winner from bot1's perspective (bot1 is player 1)
    winner = game.get_reward_for_next_player(absolute_state, player)
    if winner == 1:  # Player 1 (bot1) won
        result = 1
    elif winner == -1:  # Player -1 (bot2) won
        result = -1
    else:  # Draw
        result = 0

    return result, game_history

def main():
    # Initialize game
    game = TicTacToe()

    # Load the latest trained model for both bots
    all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*_best_model.pt"))
    model_path = None

    if all_models:
        # Get modification time for each model
        models_with_time = []
        for f in all_models:
            try:
                mtime = os.path.getmtime(f)
                models_with_time.append((mtime, f))
            except OSError:
                continue

        if models_with_time:
            # Sort by modification time (most recent first)
            models_with_time.sort(reverse=True)

            # Try loading models starting from most recent until one works
            for mtime, model_file in models_with_time:
                try:
                    # Quick test: try to load state dict to check architecture
                    test_model = NeuralNetwork().to(device)
                    test_state = torch.load(model_file, map_location=device)
                    test_model.load_state_dict(test_state)
                    # If we get here, architecture matches!
                    model_path = model_file
                    model_name = os.path.basename(model_file)
                    print(f"Found {len(all_models)} model(s), using most recent compatible: {model_name}")
                    break
                except (RuntimeError, FileNotFoundError) as e:
                    # Architecture mismatch or file error, try next
                    continue
                finally:
                    # Clean up test model
                    del test_model
                    if 'test_state' in locals():
                        del test_state
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # If no compatible model found, fall back to highest number
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
                    latest_num, model_path = max(files_with_numbers, key=lambda x: x[0])
                    print(f"Found {len(all_models)} model(s), using highest numbered: model {latest_num}")
        else:
            # Fallback: try numbered models
            model_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(1))
            if not os.path.exists(model_path):
                model_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(0))
    else:
        # No models found, try default
        model_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(1))
        if not os.path.exists(model_path):
            model_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(0))

    if not os.path.exists(model_path):
        print(f"ERROR: No model found at {model_path}")
        print("Please train a model first before running bot vs bot tests.")
        return

    print(f"Loading model from: {model_path}")

    # Create two separate MCTS instances (both using the same model)
    vpn1 = ValuePolicyNetwork(model_path)
    policy_value_network1 = vpn1.get_vp
    mcts1 = MonteCarloTreeSearch(game, policy_value_network1)

    vpn2 = ValuePolicyNetwork(model_path)
    policy_value_network2 = vpn2.get_vp
    mcts2 = MonteCarloTreeSearch(game, policy_value_network2)

    # Test parameters
    num_games = cfg.NUM_GAMES
    num_simulations = cfg.NUM_SIMULATIONS
    temperature = 1.2  # Add randomness to ensure varied games

    print(f"\nTesting Bot vs Bot (Self-Play)")
    print(f"Total games: {num_games}")
    print(f"MCTS simulations per move: {num_simulations}")
    print(f"Temperature: {temperature} (higher = more variety)")
    print(f"Both bots use the same model")
    print("=" * 60)

    results = []
    all_games_history = []

    # Play games with bots alternating who goes first
    print("\nPlaying bot vs bot games:")
    for game_num in tqdm(range(num_games), total=num_games):
        # Alternate who goes first
        if game_num % 2 == 0:
            result, game_history = play_game_bot_vs_bot(game, mcts1, mcts2, num_simulations, temperature)
            first_bot = 'bot1'
        else:
            # Swap bots so bot2 goes first
            result, game_history = play_game_bot_vs_bot(game, mcts2, mcts1, num_simulations, temperature)
            result = -result  # Flip result since we swapped bots
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

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Calculate statistics
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

    # Statistics by first player
    print(f"\nWhen Bot1 plays first:")
    first_bot1 = df_results[df_results['first_player'] == 'bot1']
    bot1_first_wins = len(first_bot1[first_bot1['outcome'] == 'bot1_win'])
    bot1_first_draws = len(first_bot1[first_bot1['outcome'] == 'draw'])
    bot1_first_losses = len(first_bot1[first_bot1['outcome'] == 'bot2_win'])
    if len(first_bot1) > 0:
        print(f"  Bot1 Wins:  {bot1_first_wins} ({bot1_first_wins/len(first_bot1)*100:.1f}%)")
        print(f"  Draws:      {bot1_first_draws} ({bot1_first_draws/len(first_bot1)*100:.1f}%)")
        print(f"  Bot2 Wins:  {bot1_first_losses} ({bot1_first_losses/len(first_bot1)*100:.1f}%)")

    print(f"\nWhen Bot2 plays first:")
    first_bot2 = df_results[df_results['first_player'] == 'bot2']
    bot2_first_wins = len(first_bot2[first_bot2['outcome'] == 'bot2_win'])
    bot2_first_draws = len(first_bot2[first_bot2['outcome'] == 'draw'])
    bot2_first_losses = len(first_bot2[first_bot2['outcome'] == 'bot1_win'])
    if len(first_bot2) > 0:
        print(f"  Bot2 Wins:  {bot2_first_wins} ({bot2_first_wins/len(first_bot2)*100:.1f}%)")
        print(f"  Draws:      {bot2_first_draws} ({bot2_first_draws/len(first_bot2)*100:.1f}%)")
        print(f"  Bot1 Wins:  {bot2_first_losses} ({bot2_first_losses/len(first_bot2)*100:.1f}%)")

    # Save results
    output_dir = cfg.LOGDIR
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "bot_vs_bot_results.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Save game histories to JSON file
    games_json_file = os.path.join(output_dir, "bot_vs_bot_games.json")
    json_data = {
        'metadata': {
            'total_games': int(num_games),
            'num_simulations': int(num_simulations),
            'temperature': float(temperature),
            'model_path': model_path,
            'description': 'Both bots use the same trained model with temperature for variety'
        },
        'games': convert_numpy_types(all_games_history)
    }
    with open(games_json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Game histories saved to: {games_json_file}")

    # Save a readable text version of games
    games_text_file = os.path.join(output_dir, "bot_vs_bot_games_readable.txt")
    with open(games_text_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Bot vs Bot Self-Play - Game Histories\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Games: {num_games}\n")
        f.write(f"MCTS Simulations: {num_simulations} per move\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Both bots use the same trained model\n\n")

        for game_data in all_games_history:
            game_num = game_data['game_number']
            first_player = game_data['first_player']
            outcome = game_data['outcome']
            result = game_data['result']

            f.write("-" * 80 + "\n")
            f.write(f"Game {game_num} | First Player: {first_player} | Outcome: {outcome} ")
            f.write(f"(Result: {result})\n")
            f.write("-" * 80 + "\n")

            for move_data in game_data['moves']:
                move_num = move_data['move']
                player = move_data['player']
                action_coords = move_data['action_coords']
                board_state = move_data['board_state']

                if move_num == 0:
                    f.write(f"\nInitial Board:\n")
                else:
                    if action_coords:
                        f.write(f"\nMove {move_num}: {player.upper()} plays at ({action_coords[0]}, {action_coords[1]})\n")
                    else:
                        f.write(f"\nMove {move_num}: {player.upper()}\n")

                # Print board in readable format
                for row in board_state:
                    f.write("  " + " ".join(row) + "\n")

                # Print visit counts for bot moves
                if player in ['bot1', 'bot2'] and 'visit_counts_formatted' in move_data:
                    f.write(f"\n{player.upper()}'s MCTS Visit Counts:\n")
                    f.write(move_data['visit_counts_formatted'])
                    f.write("\n")
                    if 'top_moves' in move_data and move_data['top_moves']:
                        f.write("Top 5 moves considered:\n")
                        for idx, (action_idx, visit_count, (r, c)) in enumerate(move_data['top_moves'], 1):
                            marker = " <- SELECTED" if action_idx == move_data['action_index'] else ""
                            f.write(f"  {idx}. Position ({r}, {c}): {visit_count} visits{marker}\n")
                    f.write("\n")

            f.write("\n")
            # Determine final winner
            final_state = game_data['moves'][-1]['board_state']
            winner = None
            # Check rows
            for row in final_state:
                for j in range(2):
                    if row[j:j+4].count('X') == 4:
                        winner = 'X (Player 1)'
                        break
                    if row[j:j+4].count('O') == 4:
                        winner = 'O (Player -1)'
                        break
                if winner:
                    break

            if winner:
                f.write(f"Winner: {winner}\n")
            elif all('.' not in row for row in final_state):
                f.write("Result: Draw\n")
            f.write("\n")

    print(f"Readable game histories saved to: {games_text_file}")

    # Create visualization - Overall results only
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Overall results distribution
    outcome_counts = df_results['outcome'].value_counts()
    colors = {'bot1_win': 'blue', 'draw': 'gray', 'bot2_win': 'red'}
    outcome_colors = [colors.get(outcome, 'blue') for outcome in outcome_counts.index]

    ax.bar(outcome_counts.index, outcome_counts.values, color=outcome_colors)
    ax.set_xlabel('Outcome', fontsize=12)
    ax.set_ylabel('Number of Games', fontsize=12)
    ax.set_title('Bot vs Bot - Overall Results Distribution', fontsize=14, fontweight='bold')

    # Set x-tick labels based on what outcomes actually exist
    label_map = {'bot1_win': 'Bot1 Win', 'draw': 'Draw', 'bot2_win': 'Bot2 Win'}
    ax.set_xticklabels([label_map.get(outcome, outcome) for outcome in outcome_counts.index])

    # Add percentage labels on bars
    for i, (outcome, count) in enumerate(outcome_counts.items()):
        percentage = count / total_games * 100
        ax.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bot_vs_bot_results.png"), dpi=150)
    plt.close()
    print(f"Results graph saved to: {os.path.join(output_dir, 'bot_vs_bot_results.png')}")

    # Save summary
    summary_file = os.path.join(output_dir, "bot_vs_bot_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Bot vs Bot Self-Play - Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Games:      {total_games}\n")
        f.write(f"MCTS Simulations: {num_simulations} per move\n")
        f.write(f"Temperature:      {temperature}\n")
        f.write(f"Model Path:       {model_path}\n\n")
        f.write(f"Overall Performance:\n")
        f.write(f"  Bot1 Wins:      {bot1_wins} ({bot1_wins/total_games*100:.1f}%)\n")
        f.write(f"  Draws:          {draws} ({draws/total_games*100:.1f}%)\n")
        f.write(f"  Bot2 Wins:      {bot2_wins} ({bot2_wins/total_games*100:.1f}%)\n\n")
        if len(first_bot1) > 0:
            f.write(f"Bot1 first: {bot1_first_wins}W-{bot1_first_draws}D-{bot1_first_losses}L\n")
        if len(first_bot2) > 0:
            f.write(f"Bot2 first: {bot2_first_wins}W-{bot2_first_draws}D-{bot2_first_losses}L\n")

    print(f"Summary saved to: {summary_file}")
    print("\n" + "=" * 60)
    print("Testing complete!")

if __name__ == "__main__":
    main()
