"""
Evaluate models using ONLY the raw policy network output (no MCTS search).
This shows the actual improvement in the neural network itself.
"""
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
import torch
import torch.nn.functional as F
from config import Config as cfg
from game import TicTacToe
from model import NeuralNetwork
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_iteration_number(model_path):
    """Extract iteration number from model filename"""
    basename = os.path.basename(model_path)
    try:
        num = int(basename.split("_")[0])
        return num
    except ValueError:
        return None

def state_to_tensor(state, player):
    """
    Convert flat state array to 3-channel tensor for neural network.
    Channels: [current player positions, opponent positions, empty positions]
    """
    state_2d = state.reshape(9, 9)

    # Create 3 channels
    current_player = (state_2d == 1).astype(np.float32)
    opponent = (state_2d == -1).astype(np.float32)
    empty = (state_2d == 0).astype(np.float32)

    # Stack channels: [current_player, opponent, empty]
    tensor = np.stack([current_player, opponent, empty], axis=0)

    # Add batch dimension and convert to torch tensor
    tensor = torch.FloatTensor(tensor).unsqueeze(0).to(device)

    return tensor

class RawPolicyPlayer:
    """A player that uses only the neural network policy (no MCTS)"""
    def __init__(self, model, game, temperature=0.1):
        self.model = model
        self.game = game
        self.temperature = temperature

    def get_action(self, state, player):
        """Select action using only the neural network policy output"""
        # Convert state to tensor
        state_tensor = state_to_tensor(state, player)

        # Get policy from network
        with torch.no_grad():
            _, policy_logits = self.model(state_tensor)
            policy_logits = policy_logits.cpu().numpy().flatten()

        # Mask invalid moves
        valid_moves = self.game.get_valid_moves(state)
        policy_logits = policy_logits * valid_moves
        policy_logits[valid_moves == 0] = -1e9

        # Apply temperature and softmax
        if self.temperature > 0:
            policy_probs = F.softmax(torch.FloatTensor(policy_logits) / self.temperature, dim=0).numpy()
        else:
            # Greedy selection
            policy_probs = np.zeros_like(policy_logits)
            policy_probs[np.argmax(policy_logits)] = 1.0

        # Sample action
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) == 0:
            return None

        # Renormalize to ensure valid probability distribution
        policy_probs = policy_probs / (policy_probs.sum() + 1e-10)

        try:
            action_index = np.random.choice(len(policy_probs), p=policy_probs)
        except:
            # Fallback to valid random move if sampling fails
            action_index = np.random.choice(valid_indices)

        action = np.zeros(len(valid_moves))
        action[action_index] = 1
        return action

class RandomPlayer:
    """A player that makes completely random moves"""
    def __init__(self, game):
        self.game = game

    def get_action(self, state, player):
        """Select a random valid action"""
        valid_moves = self.game.get_valid_moves(state)
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) == 0:
            return None
        action_index = np.random.choice(valid_indices)
        action = np.zeros(len(valid_moves))
        action[action_index] = 1
        return action

def play_game_policy_first(game, policy_player, random_player):
    """
    Play a game with policy network going first (player 1)
    Returns: 1 if policy wins, -1 if random wins, 0 if draw
    """
    player = 1
    state = np.zeros(cfg.ACTION_SIZE)

    while game.win_or_draw(state) is None:
        if player == 1:  # Policy player's turn
            action = policy_player.get_action(state, player)
        else:  # Random player's turn
            action = random_player.get_action(state, player)

        if action is None:
            break

        action_index = np.argmax(action)
        state = game.get_next_state_from_next_player_prespective(state, action, player)
        player = -1 * player

    # Determine winner
    winner = game.get_reward_for_next_player(state, player)
    if winner == 1:  # Player 1 (policy) won
        return 1
    elif winner == -1:  # Player -1 (random) won
        return -1
    else:  # Draw
        return 0

def play_game_random_first(game, policy_player, random_player):
    """
    Play a game with random going first (player 1), policy is player -1
    Returns: 1 if policy wins, -1 if random wins, 0 if draw
    """
    player = 1
    state = np.zeros(cfg.ACTION_SIZE)

    while game.win_or_draw(state) is None:
        if player == -1:  # Policy player's turn
            action = policy_player.get_action(state, player)
        else:  # Random player's turn
            action = random_player.get_action(state, player)

        if action is None:
            break

        action_index = np.argmax(action)
        state = game.get_next_state_from_next_player_prespective(state, action, player)
        player = -1 * player

    # Determine winner
    winner = game.get_reward_for_next_player(state, player)
    if winner == -1:  # Player -1 (policy) won
        return 1
    elif winner == 1:  # Player 1 (random) won
        return -1
    else:  # Draw
        return 0

def evaluate_model_raw_policy(model_path, game, num_games=100, temperature=0.1):
    """
    Evaluate a model using ONLY raw policy output (no MCTS)
    Returns: dict with win/loss/draw statistics
    """
    try:
        # Load model
        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        policy_player = RawPolicyPlayer(model, game, temperature=temperature)
        random_player = RandomPlayer(game)

        results = []

        # Play games alternating first/second
        for game_num in range(num_games):
            if game_num % 2 == 0:
                result = play_game_policy_first(game, policy_player, random_player)
                position = 'first'
            else:
                result = play_game_random_first(game, policy_player, random_player)
                position = 'second'

            outcome = 'policy_win' if result == 1 else ('draw' if result == 0 else 'random_win')
            results.append({
                'game_number': game_num,
                'position': position,
                'result': result,
                'outcome': outcome
            })

        # Calculate statistics
        df_results = pd.DataFrame(results)
        total_games = len(df_results)
        policy_wins = len(df_results[df_results['outcome'] == 'policy_win'])
        draws = len(df_results[df_results['outcome'] == 'draw'])
        random_wins = len(df_results[df_results['outcome'] == 'random_win'])

        return {
            'total_games': total_games,
            'policy_wins': policy_wins,
            'draws': draws,
            'random_wins': random_wins,
            'win_rate': policy_wins / total_games * 100,
            'draw_rate': draws / total_games * 100,
            'loss_rate': random_wins / total_games * 100,
            'no_loss_rate': (policy_wins + draws) / total_games * 100
        }
    except Exception as e:
        print(f"Error evaluating model {model_path}: {e}")
        return None

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
    num_games = 100  # Games per model
    temperature = 0.5  # Temperature for policy sampling (lower = more deterministic)

    print(f"\nEvaluation settings:")
    print(f"  Mode: Raw policy network only (NO MCTS)")
    print(f"  Games per model: {num_games}")
    print(f"  Policy temperature: {temperature}")
    print(f"  Total evaluations: {len(models_with_numbers) * num_games}")
    print("=" * 60)

    # Evaluate each model
    evaluation_results = []

    for iteration_num, model_path in tqdm(models_with_numbers, desc="Evaluating models"):
        stats = evaluate_model_raw_policy(model_path, game, num_games, temperature)

        if stats:
            stats['iteration'] = iteration_num
            stats['model_path'] = model_path
            evaluation_results.append(stats)
            print(f"Iter {iteration_num:3d}: Win={stats['win_rate']:5.1f}% Draw={stats['draw_rate']:5.1f}% Loss={stats['loss_rate']:5.1f}%")

    # Save results
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    df_eval = pd.DataFrame(evaluation_results)
    csv_path = os.path.join(output_dir, "raw_policy_progression_vs_random.csv")
    df_eval.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Win/Draw/Loss rates over iterations
    ax1 = axes[0, 0]
    ax1.plot(df_eval['iteration'], df_eval['win_rate'], 'g-o', label='Win Rate', linewidth=2, markersize=6)
    ax1.plot(df_eval['iteration'], df_eval['draw_rate'], 'gray', label='Draw Rate', linewidth=2, markersize=6, linestyle='--')
    ax1.plot(df_eval['iteration'], df_eval['loss_rate'], 'r-x', label='Loss Rate', linewidth=2, markersize=6)
    ax1.axhline(y=100, color='green', linestyle=':', alpha=0.3, label='100% Win Target')
    ax1.set_xlabel('Training Iteration', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Raw Policy Network Performance vs Random (NO MCTS)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # Plot 2: Absolute game counts (stacked bar)
    ax2 = axes[0, 1]
    width = 0.8
    x_pos = df_eval['iteration']

    ax2.bar(x_pos, df_eval['policy_wins'], width, label='Policy Wins', color='green', alpha=0.8)
    ax2.bar(x_pos, df_eval['draws'], width, bottom=df_eval['policy_wins'], label='Draws', color='gray', alpha=0.8)
    ax2.bar(x_pos, df_eval['random_wins'], width,
            bottom=df_eval['policy_wins'] + df_eval['draws'], label='Random Wins', color='red', alpha=0.8)

    ax2.set_xlabel('Training Iteration', fontsize=12)
    ax2.set_ylabel('Number of Games', fontsize=12)
    ax2.set_title('Game Outcomes by Training Iteration (Raw Policy)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Win rate with trend line
    ax3 = axes[1, 0]
    ax3.scatter(df_eval['iteration'], df_eval['win_rate'], c='green', s=100, alpha=0.6, edgecolors='darkgreen', linewidth=1.5)

    # Add polynomial trend line if we have enough data points
    if len(df_eval) > 3:
        z = np.polyfit(df_eval['iteration'], df_eval['win_rate'], min(3, len(df_eval)-1))
        p = np.poly1d(z)
        x_smooth = np.linspace(df_eval['iteration'].min(), df_eval['iteration'].max(), 100)
        ax3.plot(x_smooth, p(x_smooth), "g--", alpha=0.8, linewidth=2, label='Trend')

    ax3.axhline(y=100, color='green', linestyle=':', linewidth=2, alpha=0.5, label='100% Win Target')
    ax3.set_xlabel('Training Iteration', fontsize=12)
    ax3.set_ylabel('Win Rate (%)', fontsize=12)
    ax3.set_title('Win Rate Progression (Raw Policy Network)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 105)

    # Plot 4: Comparison of all three metrics over time
    ax4 = axes[1, 1]
    ax4.fill_between(df_eval['iteration'], 0, df_eval['win_rate'],
                     color='green', alpha=0.3, label='Win Rate')
    ax4.fill_between(df_eval['iteration'], df_eval['win_rate'],
                     df_eval['win_rate'] + df_eval['draw_rate'],
                     color='gray', alpha=0.3, label='Draw Rate')
    ax4.fill_between(df_eval['iteration'], df_eval['win_rate'] + df_eval['draw_rate'], 100,
                     color='red', alpha=0.3, label='Loss Rate')

    ax4.set_xlabel('Training Iteration', fontsize=12)
    ax4.set_ylabel('Percentage (%)', fontsize=12)
    ax4.set_title('Outcome Distribution Over Training (Stacked Area)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10, loc='right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "raw_policy_training_progression.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {plot_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (RAW POLICY NETWORK - NO MCTS)")
    print("=" * 60)

    # Find key milestones
    perfect_wins = df_eval[df_eval['win_rate'] >= 100]
    no_losses = df_eval[df_eval['loss_rate'] == 0]

    if len(perfect_wins) > 0:
        first_perfect = perfect_wins.iloc[0]
        print(f"\nFirst 100% win rate: Iteration {first_perfect['iteration']}")
    else:
        best_win = df_eval.loc[df_eval['win_rate'].idxmax()]
        print(f"\nBest win rate: {best_win['win_rate']:.1f}% at iteration {best_win['iteration']}")

    if len(no_losses) > 0:
        first_no_loss = no_losses.iloc[0]
        print(f"First 0% loss rate: Iteration {first_no_loss['iteration']}")

    # Show progression
    print(f"\nProgression from first to last iteration:")
    first_iter = df_eval.iloc[0]
    last_iter = df_eval.iloc[-1]
    print(f"  Iteration {first_iter['iteration']: >3}: Win={first_iter['win_rate']:5.1f}%, Draw={first_iter['draw_rate']:5.1f}%, Loss={first_iter['loss_rate']:5.1f}%")
    print(f"  Iteration {last_iter['iteration']: >3}: Win={last_iter['win_rate']:5.1f}%, Draw={last_iter['draw_rate']:5.1f}%, Loss={last_iter['loss_rate']:5.1f}%")
    print(f"  Improvement: +{last_iter['win_rate'] - first_iter['win_rate']:.1f}% win rate")

    print("\n" + "=" * 60)
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
