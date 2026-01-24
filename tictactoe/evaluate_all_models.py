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
from minimax_player import MinimaxPlayer

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_iteration_number(model_path):
    """Extract iteration number from model filename"""
    basename = os.path.basename(model_path)
    try:
        num = int(basename.split("_")[0])
        return num
    except ValueError:
        return None

def play_game_bot_first_vs_minimax(game, mcts, minimax_player, num_simulations=1600):
    """
    Bot (MCTS) is player 1, minimax is player -1.
    Returns: 1 if bot wins, -1 if minimax wins, 0 if draw.
    """
    from mcts import Node

    player = 1
    state = np.zeros(cfg.ACTION_SIZE)
    absolute_state = np.zeros(cfg.ACTION_SIZE)

    while game.win_or_draw(absolute_state) is None:
        if player == 1:
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)
            action, next_node, _ = mcts.select_move(node=root_node, mode="explore", temperature=0.1)
            action_index = int(np.argmax(action))
            state = next_node.state.copy()
            absolute_state[action_index] = 1
        else:
            action = minimax_player.get_action(absolute_state, player)
            if action is None:
                break
            action_index = int(np.argmax(action))
            state = game.get_next_state_from_next_player_prespective(state, action, player)
            absolute_state[action_index] = -1
        player *= -1

    winner = game.win_or_draw(absolute_state)
    if winner == 0:
        return 0
    return 1 if winner == 1 else -1


def play_game_minimax_first(game, mcts, minimax_player, num_simulations=1600):
    """
    Minimax is player 1, bot (MCTS) is player -1.
    Returns: 1 if bot wins, -1 if minimax wins, 0 if draw.
    """
    from mcts import Node

    player = 1
    state = np.zeros(cfg.ACTION_SIZE)
    absolute_state = np.zeros(cfg.ACTION_SIZE)

    while game.win_or_draw(absolute_state) is None:
        if player == 1:
            action = minimax_player.get_action(absolute_state, player)
            if action is None:
                break
            action_index = int(np.argmax(action))
            state = game.get_next_state_from_next_player_prespective(state, action, player)
            absolute_state[action_index] = 1
        else:
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            root_node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)
            action, next_node, _ = mcts.select_move(node=root_node, mode="explore", temperature=0.1)
            action_index = int(np.argmax(action))
            state = next_node.state.copy()
            absolute_state[action_index] = -1
        player *= -1

    winner = game.win_or_draw(absolute_state)
    if winner == 0:
        return 0
    return 1 if winner == -1 else -1


def evaluate_model(model_path, game, num_games=100, num_simulations=400):
    """
    Evaluate a single model against minimax opponent.
    Returns: dict with win/loss/draw statistics
    """
    try:
        # Load model
        vpn = ValuePolicyNetwork(model_path)
        policy_value_network = vpn.get_vp
        mcts = MonteCarloTreeSearch(game, policy_value_network)

        opponent_player = MinimaxPlayer(game, depth=2, radius=1, max_candidates=24)
        play_first = lambda: (play_game_bot_first_vs_minimax(game, mcts, opponent_player, num_simulations), None)
        play_second = lambda: (play_game_minimax_first(game, mcts, opponent_player, num_simulations), None)
        opponent_label = "minimax"
        opponent_wins_key = "minimax_wins"

        results = []

        # Play games with bot alternating going first and second
        for game_num in range(num_games):
            if game_num % 2 == 0:
                result, _ = play_first()
                bot_position = 'first'
            else:
                result, _ = play_second()
                bot_position = 'second'

            outcome = 'bot_win' if result == 1 else ('draw' if result == 0 else f'{opponent_label}_win')
            results.append({
                'game_number': game_num,
                'bot_position': bot_position,
                'result': result,
                'outcome': outcome
            })

        # Calculate statistics
        df_results = pd.DataFrame(results)
        total_games = len(df_results)
        bot_wins = len(df_results[df_results['outcome'] == 'bot_win'])
        draws = len(df_results[df_results['outcome'] == 'draw'])
        opponent_wins = len(df_results[df_results['outcome'] == f'{opponent_label}_win'])

        stats = {
            'total_games': total_games,
            'bot_wins': bot_wins,
            'draws': draws,
            opponent_wins_key: opponent_wins,
            'win_rate': bot_wins / total_games * 100,
            'draw_rate': draws / total_games * 100,
            'loss_rate': opponent_wins / total_games * 100,
            'no_loss_rate': (bot_wins + draws) / total_games * 100
        }
        # Back-compat / shared column name for generic plotting.
        stats['opponent'] = opponent_label
        stats['opponent_wins'] = opponent_wins
        return stats
    except Exception as e:
        print(f"Error evaluating model {model_path}: {e}")
        return None

def save_results_and_visualization(evaluation_results, output_dir, opponent_label, opponent_wins_col, csv_name, plot_name):
    if not evaluation_results:
        print(f"\nNo evaluation results to save for opponent={opponent_label}")
        return pd.DataFrame()

    df_eval = pd.DataFrame(evaluation_results)
    csv_path = os.path.join(output_dir, csv_name)
    df_eval.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Single-graph outcome distribution across training iterations (stacked percentages).
    iterations = df_eval["iteration"].to_numpy()
    win = df_eval["win_rate"].to_numpy()
    draw = df_eval["draw_rate"].to_numpy()
    loss = df_eval["loss_rate"].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.stackplot(
        iterations,
        win,
        draw,
        loss,
        labels=["Bot Win", "Draw", f"{opponent_label.title()} Win"],
        colors=["#2ca02c", "#7f7f7f", "#d62728"],
        alpha=0.85,
    )
    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Outcome Distribution (%)", fontsize=12)
    ax.set_title(f"Outcome Distribution vs {opponent_label.title()} Across Training", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: {plot_path}")

    return df_eval


def print_evaluation_summary(df_eval, opponent_label):
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY (vs {opponent_label.upper()})")
    print("=" * 60)

    perfect_wins = df_eval[df_eval['win_rate'] >= 100]
    perfect_no_loss = df_eval[df_eval['no_loss_rate'] >= 100]

    if len(perfect_wins) > 0:
        first_perfect_win = perfect_wins.iloc[0]
        print(f"\nFirst 100% win rate: Iteration {first_perfect_win['iteration']}")
    else:
        print(f"\nNo iteration achieved 100% win rate yet")
        best_win = df_eval.loc[df_eval['win_rate'].idxmax()]
        print(f"Best win rate: {best_win['win_rate']:.1f}% at iteration {best_win['iteration']}")

    if len(perfect_no_loss) > 0:
        first_no_loss = perfect_no_loss.iloc[0]
        print(f"First 100% no-loss rate: Iteration {first_no_loss['iteration']}")
    else:
        print(f"No iteration achieved 100% no-loss rate yet")
        best_no_loss = df_eval.loc[df_eval['no_loss_rate'].idxmax()]
        print(f"Best no-loss rate: {best_no_loss['no_loss_rate']:.1f}% at iteration {best_no_loss['iteration']}")

    print(f"\nProgression from first to last iteration:")
    first_iter = df_eval.iloc[0]
    last_iter = df_eval.iloc[-1]
    print(f"  Iteration {first_iter['iteration']: >3}: Win={first_iter['win_rate']:5.1f}%, Draw={first_iter['draw_rate']:5.1f}%, Loss={first_iter['loss_rate']:5.1f}%")
    print(f"  Iteration {last_iter['iteration']: >3}: Win={last_iter['win_rate']:5.1f}%, Draw={last_iter['draw_rate']:5.1f}%, Loss={last_iter['loss_rate']:5.1f}%")
    print(f"  Improvement: +{last_iter['win_rate'] - first_iter['win_rate']:.1f}% win rate")

def main():
    # Initialize game
    game = TicTacToe()

    # Find all model files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, cfg.SAVE_MODEL_PATH)

    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        return

    all_models = glob(os.path.join(model_dir, "*_best_model.pt"))
    if not all_models:
        print(f"ERROR: No models found in {model_dir}")
        return

    # Sort models by iteration number
    models_with_numbers = []
    for model_path in all_models:
        iteration_num = extract_iteration_number(model_path)
        if iteration_num is not None:
            models_with_numbers.append((iteration_num, model_path))

    models_with_numbers.sort(key=lambda x: x[0])

    print(f"Found {len(models_with_numbers)} models to evaluate")
    print(f"Iteration range: {models_with_numbers[0][0]} to {models_with_numbers[-1][0]}")

    # Evaluation parameters
    num_games = 50  # Games per model (reduced for faster evaluation)
    num_simulations_minimax = 1600  # MCTS simulations per move vs minimax (strong baseline)

    print(f"\nEvaluation settings:")
    print(f"  Games per model: {num_games}")
    print(f"  MCTS simulations vs minimax: {num_simulations_minimax}")
    print(f"  Total games played: {len(models_with_numbers) * num_games}")
    print("=" * 60)

    # Evaluate each model
    evaluation_results_minimax = []

    for iteration_num, model_path in tqdm(models_with_numbers, desc="Evaluating models"):
        print(f"\nEvaluating iteration {iteration_num}...")
        stats_minimax = evaluate_model(
            model_path,
            game,
            num_games=num_games,
            num_simulations=num_simulations_minimax,
        )
        if stats_minimax:
            stats_minimax['iteration'] = iteration_num
            stats_minimax['model_path'] = model_path
            evaluation_results_minimax.append(stats_minimax)
            print(f"  vs Minimax: Win {stats_minimax['win_rate']:.1f}% | Draw {stats_minimax['draw_rate']:.1f}% | Loss {stats_minimax['loss_rate']:.1f}%")

    # Save results
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    df_minimax = save_results_and_visualization(
        evaluation_results_minimax,
        output_dir=output_dir,
        opponent_label="minimax",
        opponent_wins_col="minimax_wins",
        csv_name="model_progression_vs_minimax.csv",
        plot_name="training_progression_vs_minimax.png",
    )

    print_evaluation_summary(df_minimax, "minimax")

    print("\n" + "=" * 60)
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
