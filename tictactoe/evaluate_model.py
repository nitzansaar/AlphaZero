"""
Evaluate a model against the previous best model and calculate ELO rating.
This is used to track model improvement across training iterations.
"""
import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import pandas as pd
from config import Config as cfg
from game import TicTacToe
from mcts import MonteCarloTreeSearch
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_elo_change(elo_a, elo_b, result, k_factor=32):
    """
    Calculate ELO rating change after a match.
    
    Args:
        elo_a: Current ELO of player A
        elo_b: Current ELO of player B
        result: 1 if A wins, 0 if draw, -1 if A loses
        k_factor: K-factor for ELO updates (default 32)
    
    Returns:
        (new_elo_a, new_elo_b): Updated ELO ratings
    """
    # Expected score for player A
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    
    # Actual score: 1 for win, 0.5 for draw, 0 for loss
    if result == 1:
        actual_a = 1.0
    elif result == 0:
        actual_a = 0.5
    else:
        actual_a = 0.0
    
    # Update ELO
    new_elo_a = elo_a + k_factor * (actual_a - expected_a)
    new_elo_b = elo_b - k_factor * (actual_a - expected_a)  # Opposite change
    
    return new_elo_a, new_elo_b

# Removed - using inline version in evaluate_models instead

def get_model_iteration_number(model_path):
    """Extract iteration number from model filename."""
    basename = os.path.basename(model_path)
    if "_best_model.pt" in basename:
        try:
            return int(basename.split("_")[0])
        except ValueError:
            return -1
    return -1

def load_model_by_iteration(iteration_num):
    """Load model by iteration number."""
    model_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(iteration_num))
    if os.path.exists(model_path):
        return model_path
    return None

def get_latest_model():
    """Get the latest trained model."""
    all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*_best_model.pt"))
    if not all_models:
        return None
    
    # Get models with iteration numbers
    models_with_iter = []
    for f in all_models:
        iter_num = get_model_iteration_number(f)
        if iter_num >= 0:
            models_with_iter.append((iter_num, f))
    
    if not models_with_iter:
        return None
    
    # Return highest iteration number
    latest_iter, latest_path = max(models_with_iter, key=lambda x: x[0])
    return latest_path, latest_iter

def get_previous_model(current_iteration):
    """Get the previous model for comparison."""
    if current_iteration <= 0:
        return None, -1
    
    prev_iteration = current_iteration - 1
    model_path = load_model_by_iteration(prev_iteration)
    
    if model_path and os.path.exists(model_path):
        return model_path, prev_iteration
    
    return None, -1

def evaluate_models(new_model_path, old_model_path, num_games=50, num_simulations=800):
    """
    Evaluate new model against old model and calculate ELO change.
    
    Returns:
        (win_rate, elo_change, results): Win rate, ELO change, and game results
    """
    game = TicTacToe()
    
    # Load models
    vpn_new = ValuePolicyNetwork(new_model_path, use_compile=False)
    vpn_old = ValuePolicyNetwork(old_model_path, use_compile=False)
    
    policy_value_new = vpn_new.get_vp
    policy_value_old = vpn_old.get_vp
    
    mcts_new = MonteCarloTreeSearch(game, policy_value_new)
    mcts_old = MonteCarloTreeSearch(game, policy_value_old)
    
    # Load ELO tracking
    elo_tracking_path = os.path.join(cfg.LOGDIR, "elo_tracking.csv")
    base_elo = 1500  # Starting ELO
    
    if os.path.exists(elo_tracking_path):
        try:
            elo_df = pd.read_csv(elo_tracking_path)
            if len(elo_df) > 0:
                # Get ELO of previous model
                prev_iter = get_model_iteration_number(old_model_path)
                prev_elos = elo_df[elo_df['iteration'] == prev_iter]
                if len(prev_elos) > 0:
                    base_elo = prev_elos.iloc[-1]['elo'].values[0]
        except Exception as e:
            print(f"Warning: Could not load ELO tracking: {e}")
    
    new_elo = base_elo
    old_elo = base_elo
    
    results = []
    wins = 0
    draws = 0
    losses = 0
    
    print(f"\nEvaluating models:")
    print(f"  New model: {os.path.basename(new_model_path)}")
    print(f"  Old model: {os.path.basename(old_model_path)}")
    print(f"  Games: {num_games}")
    print(f"  Simulations per move: {num_simulations}")
    
    # Play games
    for game_num in tqdm(range(num_games), desc="Playing evaluation games"):
        # Alternate who goes first
        new_goes_first = (game_num % 2 == 0)
        
        # Initialize game state
        state = np.zeros(cfg.ACTION_SIZE)
        current_player = 1
        
        while game.win_or_draw(state) is None:
            # Create node for current state
            from mcts import Node
            node = Node(prior_prob=0, player=current_player, action_index=None)
            node.set_state(state.copy())
            
            # Run MCTS and select move
            if (new_goes_first and current_player == 1) or (not new_goes_first and current_player == -1):
                # New model's turn
                node = mcts_new.run_simulation(root_node=node, num_simulations=num_simulations, player=current_player)
                action, node, _ = mcts_new.select_move(node=node, mode="exploit", temperature=0.1)
            else:
                # Old model's turn
                node = mcts_old.run_simulation(root_node=node, num_simulations=num_simulations, player=current_player)
                action, node, _ = mcts_old.select_move(node=node, mode="exploit", temperature=0.1)
            
            # Apply move
            action_index = np.argmax(action)
            state[action_index] = current_player
            current_player *= -1
            
            # Check for win/draw
            winner = game.win_or_draw(state)
            if winner is not None:
                break
        
        # Determine winner from new model's perspective
        winner = game.win_or_draw(state)
        if winner is None:
            result = 0  # Draw
        elif new_goes_first:
            result = 1 if winner == 1 else (-1 if winner == -1 else 0)
        else:
            result = 1 if winner == -1 else (-1 if winner == 1 else 0)
        
        # Update ELO
        new_elo, old_elo = calculate_elo_change(new_elo, old_elo, result)
        
        # Track results
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
        
        results.append(result)
    
    win_rate = wins / num_games
    elo_change = new_elo - base_elo
    
    return win_rate, elo_change, new_elo, results

def main():
    """Main evaluation function."""
    import sys
    
    # Get current iteration from command line, file, or detect automatically
    current_iteration = None
    
    if len(sys.argv) > 1:
        current_iteration = int(sys.argv[1])
    else:
        # Try to read from file (set by train.py)
        iter_file = os.path.join(cfg.LOGDIR, "current_iteration.txt")
        if os.path.exists(iter_file):
            try:
                with open(iter_file, 'r') as f:
                    current_iteration = int(f.read().strip())
            except (ValueError, IOError):
                pass
    
    if current_iteration is None:
        # Auto-detect latest model
        result = get_latest_model()
        if result is None:
            print("Error: No models found")
            return
        new_model_path, current_iteration = result
    else:
        new_model_path = load_model_by_iteration(current_iteration)
        if not new_model_path:
            print(f"Error: Model for iteration {current_iteration} not found")
            return
    
    # Get previous model
    old_model_path, prev_iteration = get_previous_model(current_iteration)
    
    if old_model_path is None:
        print(f"Iteration {current_iteration}: No previous model to compare against (first iteration)")
        # Initialize ELO tracking
        elo_tracking_path = os.path.join(cfg.LOGDIR, "elo_tracking.csv")
        elo_df = pd.DataFrame({
            'iteration': [current_iteration],
            'elo': [1500],
            'elo_change': [0],
            'win_rate': [0.5]
        })
        elo_df.to_csv(elo_tracking_path, index=False)
        print(f"Initialized ELO tracking: Starting ELO = 1500")
        return
    
    # Evaluate
    win_rate, elo_change, new_elo, results = evaluate_models(
        new_model_path, 
        old_model_path, 
        num_games=cfg.EVAL_GAMES,
        num_simulations=cfg.NUM_SIMULATIONS
    )
    
    # Update ELO tracking
    elo_tracking_path = os.path.join(cfg.LOGDIR, "elo_tracking.csv")
    
    if os.path.exists(elo_tracking_path):
        elo_df = pd.read_csv(elo_tracking_path)
    else:
        elo_df = pd.DataFrame(columns=['iteration', 'elo', 'elo_change', 'win_rate'])
    
    # Add new entry
    new_row = pd.DataFrame({
        'iteration': [current_iteration],
        'elo': [new_elo],
        'elo_change': [elo_change],
        'win_rate': [win_rate]
    })
    elo_df = pd.concat([elo_df, new_row], ignore_index=True)
    elo_df.to_csv(elo_tracking_path, index=False)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Iteration: {current_iteration}")
    print(f"Previous iteration: {prev_iteration}")
    print(f"\nWin Rate: {win_rate:.1%} ({int(win_rate * cfg.EVAL_GAMES)}/{cfg.EVAL_GAMES} games)")
    print(f"Wins: {sum(1 for r in results if r == 1)}")
    print(f"Draws: {sum(1 for r in results if r == 0)}")
    print(f"Losses: {sum(1 for r in results if r == -1)}")
    print(f"\nELO Rating: {new_elo:.1f}")
    print(f"ELO Change: {elo_change:+.1f}")
    print("=" * 60)
    
    # Save summary
    summary_path = os.path.join(cfg.LOGDIR, f"eval_iteration_{current_iteration}.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation Results - Iteration {current_iteration}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Previous iteration: {prev_iteration}\n")
        f.write(f"Win Rate: {win_rate:.1%}\n")
        f.write(f"Wins: {sum(1 for r in results if r == 1)}\n")
        f.write(f"Draws: {sum(1 for r in results if r == 0)}\n")
        f.write(f"Losses: {sum(1 for r in results if r == -1)}\n")
        f.write(f"ELO Rating: {new_elo:.1f}\n")
        f.write(f"ELO Change: {elo_change:+.1f}\n")
        f.write("=" * 60 + "\n")

if __name__ == "__main__":
    main()

