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
    Returns a 4x4 grid with 'X' for player 1, 'O' for player -1, '.' for empty
    """
    board_2d = state.reshape(4, 4)
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
    Format the board state as a readable 4x4 grid string.
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
            lines.append("  " + "-" * 13)  # Separator between rows
    return "\n".join(lines)

def format_visit_counts(visit_counts, action_index=None):
    """
    Format visit counts as a 4x4 grid string.
    Args:
        visit_counts: List of 16 visit counts
        action_index: Optional action index to highlight
    Returns:
        Multi-line string showing the visit counts
    """
    # Reshape to 4x4
    visit_grid = np.array(visit_counts).reshape(4, 4)
    lines = []
    for i in range(4):
        row_strs = []
        for j in range(4):
            idx = i * 4 + j
            count = visit_grid[i, j]
            if action_index is not None and idx == action_index:
                row_strs.append(f"{int(count):4d}*")  # Mark selected move
            else:
                row_strs.append(f"{int(count):4d}")
        lines.append("  " + " | ".join(row_strs))
        if i < 3:
            lines.append("  " + "-" * 41)  # Separator between rows
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
    row = action_index // 4
    col = action_index % 4
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

class RandomPlayer:
    """A player that makes completely random moves"""
    def __init__(self, game):
        self.game = game
    
    def get_action(self, state):
        """Select a random valid action"""
        valid_moves = self.game.get_valid_moves(state)
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) == 0:
            return None
        action_index = np.random.choice(valid_indices)
        action = np.zeros(len(valid_moves))
        action[action_index] = 1
        return action

def play_game_bot_first(game, mcts, random_player, num_simulations=1600):
    """
    Play a single game with the bot going first (player 1)
    Returns: (result, game_history) where result is 1 if bot wins, -1 if random wins, 0 if draw
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
        if player == 1:  # Bot's turn
            # Create fresh node for bot
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)
            
            # Extract visit counts directly from root node children BEFORE select_move
            visit_counts = np.zeros(cfg.ACTION_SIZE)
            for k, v in root_node.children.items():
                visit_counts[k] = v.total_visits_N
            visit_counts_list = [int(v) for v in visit_counts]
            top_moves = get_top_moves(visit_counts_list, top_k=5)
            
            action, node, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = np.argmax(action)
            # Update canonicalized state
            state = node.state.copy()
            # Update absolute state (player 1 plays)
            absolute_state[action_index] = 1
            
            # Record bot move
            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append({
                'move': move_number,
                'player': 'bot',
                'player_number': 1,
                'action_index': int(action_index),
                'action_coords': tuple(action_index_to_coords(action_index)),
                'board_state': board_state_formatted,
                'board_state_formatted': format_board_as_string(board_state_formatted),
                'board_state_flat': [float(x) for x in absolute_state.copy().tolist()],
                'visit_counts': visit_counts_list,
                'visit_counts_formatted': format_visit_counts(visit_counts_list, action_index),
                'top_moves': top_moves
            })
        else:  # Random player's turn
            action = random_player.get_action(state)
            if action is None:
                break
            # Apply random action to canonicalized state
            action_index = np.argmax(action)
            state = game.get_next_state_from_next_player_prespective(state, action, player)
            # Update absolute state (player -1 plays)
            absolute_state[action_index] = -1
            
            # Record random player move
            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append({
                'move': move_number,
                'player': 'random',
                'player_number': -1,
                'action_index': int(action_index),
                'action_coords': tuple(action_index_to_coords(action_index)),
                'board_state': board_state_formatted,
                'board_state_formatted': format_board_as_string(board_state_formatted),
                'board_state_flat': [float(x) for x in absolute_state.copy().tolist()]
            })
        
        player = -1 * player
    
    # Determine winner from bot's perspective (bot is player 1)
    winner = game.get_reward_for_next_player(absolute_state, player)
    if winner == 1:  # Player 1 (bot) won
        result = 1
    elif winner == -1:  # Player -1 (random) won
        result = -1
    else:  # Draw
        result = 0
    
    return result, game_history

def play_game_random_first(game, mcts, random_player, num_simulations=1600):
    """
    Play a single game with random player going first (player 1), bot is player -1
    Returns: (result, game_history) where result is 1 if bot wins, -1 if random wins, 0 if draw
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
        if player == -1:  # Bot's turn
            # Create fresh node for bot
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)
            
            # Extract visit counts directly from root node children BEFORE select_move
            visit_counts = np.zeros(cfg.ACTION_SIZE)
            for k, v in root_node.children.items():
                visit_counts[k] = v.total_visits_N
            visit_counts_list = [int(v) for v in visit_counts]
            top_moves = get_top_moves(visit_counts_list, top_k=5)
            
            action, node, action_probs = mcts.select_move(node=root_node, mode="exploit", temperature=1)
            action_index = np.argmax(action)
            # Update canonicalized state
            state = node.state.copy()
            # Update absolute state (player -1 plays)
            absolute_state[action_index] = -1
            
            # Record bot move
            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append({
                'move': move_number,
                'player': 'bot',
                'player_number': -1,
                'action_index': int(action_index),
                'action_coords': tuple(action_index_to_coords(action_index)),
                'board_state': board_state_formatted,
                'board_state_formatted': format_board_as_string(board_state_formatted),
                'board_state_flat': [float(x) for x in absolute_state.copy().tolist()],
                'visit_counts': visit_counts_list,
                'visit_counts_formatted': format_visit_counts(visit_counts_list, action_index),
                'top_moves': top_moves
            })
        else:  # Random player's turn
            action = random_player.get_action(state)
            if action is None:
                break
            # Apply random action to canonicalized state
            action_index = np.argmax(action)
            state = game.get_next_state_from_next_player_prespective(state, action, player)
            # Update absolute state (player 1 plays)
            absolute_state[action_index] = 1
            
            # Record random player move
            board_state_formatted = format_board_state(absolute_state.copy())
            game_history.append({
                'move': move_number,
                'player': 'random',
                'player_number': 1,
                'action_index': int(action_index),
                'action_coords': tuple(action_index_to_coords(action_index)),
                'board_state': board_state_formatted,
                'board_state_formatted': format_board_as_string(board_state_formatted),
                'board_state_flat': [float(x) for x in absolute_state.copy().tolist()]
            })
        
        player = -1 * player
    
    # Determine winner from bot's perspective (bot is player -1)
    winner = game.get_reward_for_next_player(absolute_state, player)
    if winner == -1:  # Player -1 (bot) won
        result = 1
    elif winner == 1:  # Player 1 (random) won
        result = -1
    else:  # Draw
        result = 0
    
    return result, game_history

def main():
    # Initialize game
    game = TicTacToe()
    
    # Load the latest trained model
    # Find the most recently modified model file (better than highest number)
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
    
    print(f"Loading model from: {model_path}")
    vpn = ValuePolicyNetwork(model_path)
    policy_value_network = vpn.get_vp
    mcts = MonteCarloTreeSearch(game, policy_value_network)
    
    # Initialize random player
    random_player = RandomPlayer(game)
    
    # Test parameters
    num_games = cfg.NUM_GAMES
    num_simulations = cfg.NUM_SIMULATIONS
    
    print(f"\nTesting AlphaZero bot vs Random Player")
    print(f"Total games: {num_games}")
    print(f"MCTS simulations per move: {num_simulations}")
    print(f"Games per player position: {num_games // 2}")
    print("=" * 60)
    
    results = []
    all_games_history = []
    
    # Play half the games with bot going first
    print("\n[1/2] Bot playing as Player 1 (X - goes first):")
    for game_num in tqdm(range(num_games // 2), total=num_games // 2):
        result, game_history = play_game_bot_first(game, mcts, random_player, num_simulations)
        outcome = 'bot_win' if result == 1 else ('draw' if result == 0 else 'random_win')
        
        results.append({
            'game_number': game_num,
            'bot_position': 'first',
            'bot_player': 1,
            'result': result,
            'outcome': outcome
        })
        
        all_games_history.append({
            'game_number': game_num,
            'bot_position': 'first',
            'bot_player': 1,
            'result': result,
            'outcome': outcome,
            'moves': game_history
        })
    
    # Play half the games with random going first
    print("\n[2/2] Bot playing as Player -1 (O - goes second):")
    for game_num in tqdm(range(num_games // 2, num_games), total=num_games // 2):
        result, game_history = play_game_random_first(game, mcts, random_player, num_simulations)
        outcome = 'bot_win' if result == 1 else ('draw' if result == 0 else 'random_win')
        
        results.append({
            'game_number': game_num,
            'bot_position': 'second',
            'bot_player': -1,
            'result': result,
            'outcome': outcome
        })
        
        all_games_history.append({
            'game_number': game_num,
            'bot_position': 'second',
            'bot_player': -1,
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
    bot_wins = len(df_results[df_results['outcome'] == 'bot_win'])
    draws = len(df_results[df_results['outcome'] == 'draw'])
    random_wins = len(df_results[df_results['outcome'] == 'random_win'])
    
    print(f"\nOverall Performance:")
    print(f"  Total Games:    {total_games}")
    print(f"  Bot Wins:       {bot_wins} ({bot_wins/total_games*100:.1f}%)")
    print(f"  Draws:          {draws} ({draws/total_games*100:.1f}%)")
    print(f"  Random Wins:    {random_wins} ({random_wins/total_games*100:.1f}%)")
    print(f"  Bot Win Rate:   {(bot_wins + draws)/total_games*100:.1f}% (no losses)")
    
    # Statistics by position
    print(f"\nBot as Player 1 (going first):")
    first_games = df_results[df_results['bot_position'] == 'first']
    first_wins = len(first_games[first_games['outcome'] == 'bot_win'])
    first_draws = len(first_games[first_games['outcome'] == 'draw'])
    first_losses = len(first_games[first_games['outcome'] == 'random_win'])
    print(f"  Wins:  {first_wins} ({first_wins/len(first_games)*100:.1f}%)")
    print(f"  Draws: {first_draws} ({first_draws/len(first_games)*100:.1f}%)")
    print(f"  Losses: {first_losses} ({first_losses/len(first_games)*100:.1f}%)")
    
    print(f"\nBot as Player -1 (going second):")
    second_games = df_results[df_results['bot_position'] == 'second']
    second_wins = len(second_games[second_games['outcome'] == 'bot_win'])
    second_draws = len(second_games[second_games['outcome'] == 'draw'])
    second_losses = len(second_games[second_games['outcome'] == 'random_win'])
    print(f"  Wins:  {second_wins} ({second_wins/len(second_games)*100:.1f}%)")
    print(f"  Draws: {second_draws} ({second_draws/len(second_games)*100:.1f}%)")
    print(f"  Losses: {second_losses} ({second_losses/len(second_games)*100:.1f}%)")
    
    # Save results
    output_dir = cfg.LOGDIR
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "bot_vs_random_results.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save game histories to JSON file
    games_json_file = os.path.join(output_dir, "bot_vs_random_games.json")
    # Convert numpy types to native Python types for JSON serialization
    json_data = {
        'metadata': {
            'total_games': int(num_games),
            'num_simulations': int(num_simulations),
            'bot_position_first': 'Player 1 (X)',
            'bot_position_second': 'Player -1 (O)'
        },
        'games': convert_numpy_types(all_games_history)
    }
    with open(games_json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Game histories saved to: {games_json_file}")
    
    # Save a readable text version of games
    games_text_file = os.path.join(output_dir, "bot_vs_random_games_readable.txt")
    with open(games_text_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AlphaZero Bot vs Random Player - Game Histories\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Games: {num_games}\n")
        f.write(f"MCTS Simulations: {num_simulations} per move\n\n")
        
        for game_data in all_games_history:
            game_num = game_data['game_number']
            bot_pos = game_data['bot_position']
            outcome = game_data['outcome']
            result = game_data['result']
            
            f.write("-" * 80 + "\n")
            f.write(f"Game {game_num} | Bot Position: {bot_pos} | Outcome: {outcome} ")
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
                if player == 'bot' and 'visit_counts_formatted' in move_data:
                    f.write(f"\nBot's MCTS Visit Counts:\n")
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
            for row in final_state:
                if row.count('X') == 4:
                    winner = 'X (Player 1)'
                    break
                if row.count('O') == 4:
                    winner = 'O (Player -1)'
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
    colors = {'bot_win': 'green', 'draw': 'gray', 'random_win': 'red'}
    outcome_colors = [colors.get(outcome, 'blue') for outcome in outcome_counts.index]
    
    ax1.bar(outcome_counts.index, outcome_counts.values, color=outcome_colors)
    ax1.set_xlabel('Outcome')
    ax1.set_ylabel('Number of Games')
    ax1.set_title('Overall Results Distribution')
    ax1.set_xticklabels(['Bot Win', 'Draw', 'Random Win'])
    
    # Add percentage labels on bars
    for i, (outcome, count) in enumerate(outcome_counts.items()):
        percentage = count / total_games * 100
        ax1.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    # Results by bot position (grouped bar chart)
    positions = ['first', 'second']
    outcomes = ['bot_win', 'draw', 'random_win']
    x = np.arange(len(positions))
    width = 0.25
    
    for i, outcome in enumerate(outcomes):
        counts = [len(df_results[(df_results['bot_position'] == pos) & (df_results['outcome'] == outcome)]) 
                  for pos in positions]
        offset = (i - 1) * width
        ax2.bar(x + offset, counts, width, label=outcome.replace('_', ' ').title(), 
                color=colors.get(outcome, 'blue'))
    
    ax2.set_xlabel('Bot Position')
    ax2.set_ylabel('Number of Games')
    ax2.set_title('Results by Bot Position')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Bot First (X)', 'Bot Second (O)'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bot_vs_random_results.png"), dpi=150)
    plt.close()
    print(f"Results graph saved to: {os.path.join(output_dir, 'bot_vs_random_results.png')}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "bot_vs_random_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AlphaZero Bot vs Random Player - Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Games:    {total_games}\n")
        f.write(f"MCTS Simulations: {num_simulations} per move\n\n")
        f.write(f"Overall Performance:\n")
        f.write(f"  Bot Wins:       {bot_wins} ({bot_wins/total_games*100:.1f}%)\n")
        f.write(f"  Draws:          {draws} ({draws/total_games*100:.1f}%)\n")
        f.write(f"  Random Wins:    {random_wins} ({random_wins/total_games*100:.1f}%)\n\n")
        f.write(f"Bot as Player 1 (first): {first_wins}W-{first_draws}D-{first_losses}L\n")
        f.write(f"Bot as Player -1 (second): {second_wins}W-{second_draws}D-{second_losses}L\n")
    
    print(f"Summary saved to: {summary_file}")
    print("\n" + "=" * 60)
    print("Testing complete!")

if __name__ == "__main__":
    main()

