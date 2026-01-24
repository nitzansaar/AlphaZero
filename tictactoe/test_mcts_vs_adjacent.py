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

def action_index_to_coords(action_index):
    """Convert action index to row, col coordinates"""
    row = action_index // 9
    col = action_index % 9
    return (row, col)

def format_board_with_highlight(board_state, highlight_row=None, highlight_col=None):
    """
    Format board state with visible highlighting for a specific cell.
    Args:
        board_state: List of lists representing the board
        highlight_row: Row index to highlight (0-8)
        highlight_col: Column index to highlight (0-8)
    Returns:
        List of formatted strings, one per row
    """
    formatted_rows = []
    for i, row in enumerate(board_state):
        formatted_cells = []
        for j, cell in enumerate(row):
            if highlight_row is not None and highlight_col is not None and i == highlight_row and j == highlight_col:
                # Use asterisks and brackets to make the latest move very obvious
                formatted_cells.append(f"*{cell}*")
            else:
                formatted_cells.append(f" {cell} ")
        formatted_rows.append(" ".join(formatted_cells))
    return formatted_rows

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

class AdjacentBot:
    """
    A bot that places pieces adjacent to its own existing pieces.
    Strategy: Find all cells adjacent (8 directions) to current player's pieces
    and randomly select one. If no pieces exist, place randomly.
    """
    def __init__(self, game):
        self.game = game
        # 8 directions: N, S, E, W, NE, NW, SE, SW
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # N, S, W, E
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # NW, NE, SW, SE
        ]

    def get_adjacent_empty_cells(self, state, player_value):
        """
        Get all empty cells adjacent to cells occupied by player_value.

        Args:
            state: Flat array of 81 values
            player_value: 1 or -1 (which player we're finding adjacent cells for)

        Returns:
            Set of indices of empty cells adjacent to player's pieces
        """
        board_2d = state.reshape(9, 9)
        adjacent_cells = set()

        # Find all cells occupied by the player
        player_cells = np.argwhere(board_2d == player_value)

        # For each player cell, check all 8 adjacent positions
        for row, col in player_cells:
            for dr, dc in self.directions:
                new_row, new_col = row + dr, col + dc
                # Check bounds
                if 0 <= new_row < 9 and 0 <= new_col < 9:
                    # Check if empty
                    if board_2d[new_row, new_col] == 0:
                        # Convert to flat index
                        flat_index = new_row * 9 + new_col
                        adjacent_cells.add(flat_index)

        return adjacent_cells

    def get_action(self, state):
        """
        Select an action: prefer adjacent cells, fallback to random if no pieces exist.

        Args:
            state: Flat array representing canonical state (current player is always 1)

        Returns:
            Action array with 1 at selected position, 0 elsewhere
        """
        valid_moves = self.game.get_valid_moves(state)
        valid_indices = np.where(valid_moves == 1)[0]

        if len(valid_indices) == 0:
            return None

        # Find adjacent empty cells to current player's pieces (value 1 in canonical state)
        adjacent_cells = self.get_adjacent_empty_cells(state, player_value=1)

        # Filter to only valid moves
        adjacent_valid = list(adjacent_cells.intersection(set(valid_indices)))

        if adjacent_valid:
            # Choose randomly from adjacent cells
            action_index = np.random.choice(adjacent_valid)
        else:
            # No adjacent cells (likely first move), choose random
            action_index = np.random.choice(valid_indices)

        action = np.zeros(len(valid_moves))
        action[action_index] = 1
        return action

def play_game_mcts_first(game, mcts, adjacent_bot, num_simulations=1600):
    """
    Play a single game with MCTS bot going first (player 1)
    Returns: (result, game_history) where result is 1 if MCTS wins, -1 if adjacent wins, 0 if draw
    """
    from mcts import Node

    player = 1
    state = np.zeros(cfg.ACTION_SIZE)  # Canonicalized state for MCTS
    absolute_state = np.zeros(cfg.ACTION_SIZE)  # Absolute state for display
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
        if player == 1:  # MCTS bot's turn
            # Create fresh node for bot
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)

            action, node, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = np.argmax(action)
            # Update canonicalized state
            state = node.state.copy()
            # Update absolute state (player 1 plays)
            absolute_state[action_index] = 1

            # Record MCTS move
            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append({
                'move': move_number,
                'player': 'mcts',
                'player_number': 1,
                'action_index': int(action_index),
                'action_coords': tuple(action_index_to_coords(action_index)),
                'board_state': board_state_formatted,
                'board_state_formatted': format_board_as_string(board_state_formatted),
                'board_state_flat': [float(x) for x in absolute_state.copy().tolist()]
            })
        else:  # Adjacent bot's turn
            action = adjacent_bot.get_action(state)
            if action is None:
                break
            # Apply action to canonicalized state
            action_index = np.argmax(action)
            state = game.get_next_state_from_next_player_prespective(state, action, player)
            # Update absolute state (player -1 plays)
            absolute_state[action_index] = -1

            # Record adjacent bot move
            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append({
                'move': move_number,
                'player': 'adjacent',
                'player_number': -1,
                'action_index': int(action_index),
                'action_coords': tuple(action_index_to_coords(action_index)),
                'board_state': board_state_formatted,
                'board_state_formatted': format_board_as_string(board_state_formatted),
                'board_state_flat': [float(x) for x in absolute_state.copy().tolist()]
            })

        player = -1 * player

    # Determine winner from MCTS perspective (MCTS is player 1)
    winner = game.get_reward_for_next_player(absolute_state, player)
    if winner == 1:  # Player 1 (MCTS) won
        result = 1
    elif winner == -1:  # Player -1 (adjacent) won
        result = -1
    else:  # Draw
        result = 0

    return result, game_history

def play_game_adjacent_first(game, mcts, adjacent_bot, num_simulations=1600):
    """
    Play a single game with adjacent bot going first (player 1), MCTS is player -1
    Returns: (result, game_history) where result is 1 if MCTS wins, -1 if adjacent wins, 0 if draw
    """
    from mcts import Node

    player = 1
    state = np.zeros(cfg.ACTION_SIZE)  # Canonicalized state for MCTS
    absolute_state = np.zeros(cfg.ACTION_SIZE)  # Absolute state for display
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
        if player == -1:  # MCTS bot's turn
            # Create fresh node for bot
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)

            action, node, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = np.argmax(action)
            # Update canonicalized state
            state = node.state.copy()
            # Update absolute state (player -1 plays)
            absolute_state[action_index] = -1

            # Record MCTS move
            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append({
                'move': move_number,
                'player': 'mcts',
                'player_number': -1,
                'action_index': int(action_index),
                'action_coords': tuple(action_index_to_coords(action_index)),
                'board_state': board_state_formatted,
                'board_state_formatted': format_board_as_string(board_state_formatted),
                'board_state_flat': [float(x) for x in absolute_state.copy().tolist()]
            })
        else:  # Adjacent bot's turn
            action = adjacent_bot.get_action(state)
            if action is None:
                break
            # Apply action to canonicalized state
            action_index = np.argmax(action)
            state = game.get_next_state_from_next_player_prespective(state, action, player)
            # Update absolute state (player 1 plays)
            absolute_state[action_index] = 1

            # Record adjacent bot move
            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append({
                'move': move_number,
                'player': 'adjacent',
                'player_number': 1,
                'action_index': int(action_index),
                'action_coords': tuple(action_index_to_coords(action_index)),
                'board_state': board_state_formatted,
                'board_state_formatted': format_board_as_string(board_state_formatted),
                'board_state_flat': [float(x) for x in absolute_state.copy().tolist()]
            })

        player = -1 * player

    # Determine winner from MCTS perspective (MCTS is player -1)
    winner = game.get_reward_for_next_player(absolute_state, player)
    if winner == -1:  # Player -1 (MCTS) won
        result = 1
    elif winner == 1:  # Player 1 (adjacent) won
        result = -1
    else:  # Draw
        result = 0

    return result, game_history

def main():
    # Initialize game
    game = TicTacToe()

    # Resolve model path - handle both relative paths from script location and current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try paths relative to script location first, then relative to current working directory
    possible_model_dirs = [
        os.path.join(script_dir, cfg.SAVE_MODEL_PATH),  # src/output_tictac/models
        cfg.SAVE_MODEL_PATH,  # output_tictac/models (relative to cwd)
        os.path.join(script_dir, "..", cfg.SAVE_MODEL_PATH),  # ../output_tictac/models
    ]

    # Load the latest trained model
    all_models = []
    for model_dir in possible_model_dirs:
        if os.path.isdir(model_dir):
            found_models = glob(os.path.join(model_dir, "*_best_model.pt"))
            if found_models:
                all_models.extend(found_models)
                break

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
            for model_dir in possible_model_dirs:
                if os.path.isdir(model_dir):
                    test_path = os.path.join(model_dir, cfg.BEST_MODEL.format(1))
                    if os.path.exists(test_path):
                        model_path = test_path
                        break
                    test_path = os.path.join(model_dir, cfg.BEST_MODEL.format(0))
                    if os.path.exists(test_path):
                        model_path = test_path
                        break
    else:
        # No models found, try default paths
        for model_dir in possible_model_dirs:
            if os.path.isdir(model_dir):
                test_path = os.path.join(model_dir, cfg.BEST_MODEL.format(1))
                if os.path.exists(test_path):
                    model_path = test_path
                    break
                test_path = os.path.join(model_dir, cfg.BEST_MODEL.format(0))
                if os.path.exists(test_path):
                    model_path = test_path
                    break

    # Final check: ensure model_path exists before loading
    if model_path is None or not os.path.exists(model_path):
        print(f"ERROR: No model file found!")
        print(f"Searched in:")
        for model_dir in possible_model_dirs:
            print(f"  - {os.path.abspath(model_dir)}")
        print(f"\nPlease ensure you have trained a model first, or check that model files exist.")
        return

    print(f"Loading model from: {model_path}")
    vpn = ValuePolicyNetwork(model_path)
    policy_value_network = vpn.get_vp
    mcts = MonteCarloTreeSearch(game, policy_value_network)

    # Initialize adjacent bot
    adjacent_bot = AdjacentBot(game)

    # Test parameters
    num_games = cfg.NUM_GAMES
    num_simulations = cfg.NUM_SIMULATIONS

    print(f"\nTesting MCTS Neural Network Bot vs Adjacent Strategy Bot")
    print(f"Total games: {num_games}")
    print(f"MCTS simulations per move: {num_simulations}")
    print(f"Adjacent Bot Strategy: Place pieces adjacent to own pieces")
    print("=" * 60)

    results = []
    all_games_history = []

    # Play games with bots alternating going first and second
    print("\nPlaying MCTS vs Adjacent games (alternating first/second):")
    for game_num in tqdm(range(num_games), total=num_games):
        # Alternate who goes first
        if game_num % 2 == 0:
            # MCTS goes first
            result, game_history = play_game_mcts_first(game, mcts, adjacent_bot, num_simulations)
            mcts_position = 'first'
            mcts_player = 1
        else:
            # Adjacent goes first
            result, game_history = play_game_adjacent_first(game, mcts, adjacent_bot, num_simulations)
            mcts_position = 'second'
            mcts_player = -1

        outcome = 'mcts_win' if result == 1 else ('draw' if result == 0 else 'adjacent_win')

        results.append({
            'game_number': game_num,
            'mcts_position': mcts_position,
            'mcts_player': mcts_player,
            'result': result,
            'outcome': outcome
        })

        all_games_history.append({
            'game_number': game_num,
            'mcts_position': mcts_position,
            'mcts_player': mcts_player,
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
    mcts_wins = len(df_results[df_results['outcome'] == 'mcts_win'])
    draws = len(df_results[df_results['outcome'] == 'draw'])
    adjacent_wins = len(df_results[df_results['outcome'] == 'adjacent_win'])

    print(f"\nOverall Performance:")
    print(f"  Total Games:       {total_games}")
    print(f"  MCTS Wins:         {mcts_wins} ({mcts_wins/total_games*100:.1f}%)")
    print(f"  Draws:             {draws} ({draws/total_games*100:.1f}%)")
    print(f"  Adjacent Wins:     {adjacent_wins} ({adjacent_wins/total_games*100:.1f}%)")
    print(f"  MCTS Win Rate:     {(mcts_wins + 0.5*draws)/total_games*100:.1f}%")

    # Statistics by position
    print(f"\nMCTS as Player 1 (going first):")
    first_games = df_results[df_results['mcts_position'] == 'first']
    first_wins = len(first_games[first_games['outcome'] == 'mcts_win'])
    first_draws = len(first_games[first_games['outcome'] == 'draw'])
    first_losses = len(first_games[first_games['outcome'] == 'adjacent_win'])
    if len(first_games) > 0:
        print(f"  Wins:   {first_wins} ({first_wins/len(first_games)*100:.1f}%)")
        print(f"  Draws:  {first_draws} ({first_draws/len(first_games)*100:.1f}%)")
        print(f"  Losses: {first_losses} ({first_losses/len(first_games)*100:.1f}%)")

    print(f"\nMCTS as Player -1 (going second):")
    second_games = df_results[df_results['mcts_position'] == 'second']
    second_wins = len(second_games[second_games['outcome'] == 'mcts_win'])
    second_draws = len(second_games[second_games['outcome'] == 'draw'])
    second_losses = len(second_games[second_games['outcome'] == 'adjacent_win'])
    if len(second_games) > 0:
        print(f"  Wins:   {second_wins} ({second_wins/len(second_games)*100:.1f}%)")
        print(f"  Draws:  {second_draws} ({second_draws/len(second_games)*100:.1f}%)")
        print(f"  Losses: {second_losses} ({second_losses/len(second_games)*100:.1f}%)")

    # Save results to dedicated test output directory
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # Save a readable text version of games
    games_text_file = os.path.join(output_dir, "mcts_vs_adjacent_games_readable.txt")
    # Delete existing file if it exists
    if os.path.exists(games_text_file):
        os.remove(games_text_file)
    with open(games_text_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MCTS Neural Network Bot vs Adjacent Strategy Bot - Game Histories\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Games: {num_games}\n")
        f.write(f"MCTS Simulations: {num_simulations} per move\n")
        f.write(f"Adjacent Strategy: Place pieces adjacent to own pieces\n\n")

        for game_data in all_games_history:
            game_num = game_data['game_number']
            mcts_pos = game_data['mcts_position']
            outcome = game_data['outcome']
            result = game_data['result']

            f.write("-" * 80 + "\n")
            f.write(f"Game {game_num} | MCTS Position: {mcts_pos} | Outcome: {outcome} ")
            f.write(f"(Result: {result})\n")
            f.write("-" * 80 + "\n")

            for move_data in game_data['moves']:
                move_num = move_data['move']
                player = move_data['player']
                action_coords = move_data['action_coords']
                board_state = move_data['board_state']

                if move_num == 0:
                    f.write(f"\nInitial Board:\n")
                    # Print board without highlighting
                    for row in board_state:
                        f.write("  " + " ".join(row) + "\n")
                else:
                    if action_coords:
                        f.write(f"\nMove {move_num}: {player.upper()} plays at ({action_coords[0]}, {action_coords[1]})\n")
                    else:
                        f.write(f"\nMove {move_num}: {player.upper()}\n")

                    # Print board with highlighted move
                    highlighted_rows = format_board_with_highlight(board_state, action_coords[0] if action_coords else None, action_coords[1] if action_coords else None)
                    for row in highlighted_rows:
                        f.write("  " + row + "\n")
                f.write("\n")

            # Determine final winner
            final_state = game_data['moves'][-1]['board_state']
            winner = None
            # Check for winner (5 in a row)
            for row in final_state:
                for j in range(5):
                    if row[j:j+5].count('X') == 5:
                        winner = 'X (Player 1)'
                        break
                    if row[j:j+5].count('O') == 5:
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

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall results distribution
    outcome_counts = df_results['outcome'].value_counts()
    colors = {'mcts_win': 'green', 'draw': 'gray', 'adjacent_win': 'red'}
    outcome_colors = [colors.get(outcome, 'green') for outcome in outcome_counts.index]

    ax1.bar(outcome_counts.index, outcome_counts.values, color=outcome_colors)
    ax1.set_xlabel('Outcome', fontsize=12)
    ax1.set_ylabel('Number of Games', fontsize=12)
    ax1.set_title('MCTS vs Adjacent Bot - Overall Results', fontsize=14, fontweight='bold')

    # Set x-tick labels based on what outcomes actually exist
    label_map = {'mcts_win': 'MCTS Win', 'draw': 'Draw', 'adjacent_win': 'Adjacent Win'}
    ax1.set_xticklabels([label_map.get(outcome, outcome) for outcome in outcome_counts.index])

    # Add percentage labels on bars
    for i, (outcome, count) in enumerate(outcome_counts.items()):
        percentage = count / total_games * 100
        ax1.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.grid(axis='y', alpha=0.3)

    # Results by position (grouped bar chart)
    positions = ['first', 'second']
    outcomes = ['mcts_win', 'draw', 'adjacent_win']
    x = np.arange(len(positions))
    width = 0.25

    for i, outcome in enumerate(outcomes):
        counts = [len(df_results[(df_results['mcts_position'] == pos) & (df_results['outcome'] == outcome)])
                  for pos in positions]
        offset = (i - 1) * width
        ax2.bar(x + offset, counts, width, label=outcome.replace('_', ' ').title(),
                color=colors.get(outcome, 'green'))

    ax2.set_xlabel('MCTS Position', fontsize=12)
    ax2.set_ylabel('Number of Games', fontsize=12)
    ax2.set_title('Results by MCTS Position', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['MCTS First (X)', 'MCTS Second (O)'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mcts_vs_adjacent_results.png"), dpi=150)
    plt.close()
    print(f"\nResults graph saved to: {os.path.join(output_dir, 'mcts_vs_adjacent_results.png')}")
    print("\n" + "=" * 60)
    print("Testing complete!")

if __name__ == "__main__":
    main()
