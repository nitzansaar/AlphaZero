import os
import numpy as np
from glob import glob
import torch
from flask import Flask, render_template, request, jsonify, session
import secrets

from config import Config as cfg
from game import TicTacToe
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global model cache
_model_cache = {
    'path': None,
    'game': None,
    'mcts': None,
    'policy_value_network': None
}

def load_model():
    """
    Load the latest trained model.
    Returns the model path or None if no model found.
    """
    all_models = glob(os.path.join("output_tictac/models", "*_best_model.pt"))
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

    if model_path and os.path.exists(model_path):
        return model_path

    return None


def initialize_game():
    """Initialize or return cached game components."""
    if _model_cache['game'] is not None:
        return _model_cache['game'], _model_cache['mcts'], _model_cache['policy_value_network']

    model_path = load_model()
    if model_path is None:
        raise Exception("No trained model found")

    # Initialize game components
    game = TicTacToe()
    vpn = ValuePolicyNetwork(model_path, use_compile=False)
    policy_value_network = vpn.get_vp
    mcts = MonteCarloTreeSearch(game, policy_value_network)

    # Cache
    _model_cache['path'] = model_path
    _model_cache['game'] = game
    _model_cache['mcts'] = mcts
    _model_cache['policy_value_network'] = policy_value_network

    return game, mcts, policy_value_network


@app.route('/')
def settings():
    """Show settings page."""
    return render_template('settings.html')


@app.route('/start_game', methods=['POST'])
def start_game():
    """Initialize a new game with the provided settings."""
    try:
        # Get settings
        human_player = int(request.json.get('human_player', 1))
        num_simulations = int(request.json.get('num_simulations', 800))

        # Initialize game components
        game, mcts, policy_value_network = initialize_game()

        # Initialize game state in session
        session['state'] = np.zeros(cfg.ACTION_SIZE).tolist()
        session['current_player'] = 1  # X always starts
        session['game_over'] = False
        session['human_player'] = human_player
        session['num_simulations'] = num_simulations

        # Clear MCTS cache
        session['last_mcts_visit_counts'] = None
        session['last_mcts_chosen_action_index'] = None

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/game')
def game():
    """Show game page."""
    if 'state' not in session:
        return render_template('settings.html')
    return render_template('game.html')


@app.route('/get_state', methods=['GET'])
def get_state():
    """Get current game state."""
    if 'state' not in session:
        return jsonify({'error': 'No game in progress'}), 400

    state = np.array(session['state'])
    game, _, _ = initialize_game()

    valid_moves = game.get_valid_moves(state).tolist()
    current_player = session['current_player']
    human_player = session['human_player']
    is_human_turn = (current_player == human_player)

    result = game.win_or_draw(state)
    game_over = result is not None

    return jsonify({
        'state': session['state'],
        'current_player': current_player,
        'human_player': human_player,
        'is_human_turn': is_human_turn,
        'game_over': game_over,
        'result': result,
        'valid_moves': valid_moves,
        'num_simulations': session.get('num_simulations', 800),
        'last_mcts_visit_counts': session.get('last_mcts_visit_counts'),
        'last_mcts_chosen_action_index': session.get('last_mcts_chosen_action_index')
    })


@app.route('/make_move', methods=['POST'])
def make_move():
    """Process human move."""
    if 'state' not in session:
        return jsonify({'error': 'No game in progress'}), 400

    action_index = int(request.json.get('action_index'))

    state = np.array(session['state'])
    game, _, _ = initialize_game()

    # Validate move
    valid_moves = game.get_valid_moves(state)
    if valid_moves[action_index] != 1:
        return jsonify({'error': 'Invalid move'}), 400

    # Make move
    state[action_index] = session['current_player']
    session['state'] = state.tolist()

    # Check for game end
    result = game.win_or_draw(state)
    if result is not None:
        session['game_over'] = True
        return jsonify({
            'success': True,
            'game_over': True,
            'result': result
        })

    # Switch player
    session['current_player'] *= -1

    return jsonify({'success': True, 'game_over': False})


@app.route('/bot_move', methods=['POST'])
def bot_move():
    """Process bot move."""
    if 'state' not in session:
        return jsonify({'error': 'No game in progress'}), 400

    if session.get('game_over', False):
        return jsonify({'error': 'Game is over'}), 400

    state = np.array(session['state'])
    current_player = session['current_player']
    num_simulations = session['num_simulations']

    game, mcts, policy_value_network = initialize_game()

    # Canonicalize state from bot's perspective
    canonical_state = state.copy() * current_player

    # Create node for current state
    node = Node(prior_prob=0, player=current_player, action_index=None)
    node.set_state(canonical_state)

    # Run MCTS
    root_node = mcts.run_simulation(
        root_node=node,
        num_simulations=num_simulations,
        player=current_player
    )

    # Select best move
    action, _, action_probs = mcts.select_move(
        node=root_node,
        mode="exploit",
        temperature=0.1
    )
    action_index = int(np.argmax(action))

    # Get visit counts for visualization
    visit_counts = np.zeros(cfg.ACTION_SIZE)
    for k, v in root_node.children.items():
        visit_counts[k] = v.total_visits_N

    # Store MCTS data for heatmap
    session['last_mcts_visit_counts'] = visit_counts.tolist()
    session['last_mcts_chosen_action_index'] = int(action_index)

    # Get top moves
    top_indices = np.argsort(visit_counts)[::-1][:3].tolist()
    top_moves = []
    for idx in top_indices:
        if visit_counts[idx] > 0:
            row, col = idx // 9, idx % 9
            top_moves.append({
                'row': int(row),
                'col': int(col),
                'visits': int(visit_counts[idx])
            })

    # Make move
    state[action_index] = current_player
    session['state'] = state.tolist()

    # Check for game end
    result = game.win_or_draw(state)
    if result is not None:
        session['game_over'] = True
        return jsonify({
            'success': True,
            'action_index': int(action_index),
            'top_moves': top_moves,
            'game_over': True,
            'result': result
        })

    # Switch player
    session['current_player'] *= -1

    return jsonify({
        'success': True,
        'action_index': int(action_index),
        'top_moves': top_moves,
        'game_over': False
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
