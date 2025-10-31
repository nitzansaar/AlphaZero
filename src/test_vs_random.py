import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from config import Config as cfg
from game import TicTacToe
from mcts import MonteCarloTreeSearch
from value_policy_function import ValuePolicyNetwork
import matplotlib.pyplot as plt

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
    Returns: 1 if bot wins, -1 if random wins, 0 if draw
    """
    from mcts import Node
    
    player = 1
    state = np.zeros(cfg.ACTION_SIZE)
    
    while game.win_or_draw(state) == None:
        if player == 1:  # Bot's turn
            # Create fresh node for bot
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)
            action, node, action_probs = mcts.select_move(node=node, mode="exploit", temperature=1)
            # Update state
            state = node.state.copy()
        else:  # Random player's turn
            action = random_player.get_action(state)
            if action is None:
                break
            # Apply random action to state
            action_index = np.argmax(action)
            state = game.get_next_state_from_next_player_prespective(state, action, player)
        
        player = -1 * player
    
    # Determine winner from bot's perspective (bot is player 1)
    winner = game.get_reward_for_next_player(state, player)
    if winner == 1:  # Player 1 (bot) won
        return 1
    elif winner == -1:  # Player -1 (random) won
        return -1
    else:  # Draw
        return 0

def play_game_random_first(game, mcts, random_player, num_simulations=1600):
    """
    Play a single game with random player going first (player 1), bot is player -1
    Returns: 1 if bot wins, -1 if random wins, 0 if draw
    """
    from mcts import Node
    
    player = 1
    state = np.zeros(cfg.ACTION_SIZE)
    
    while game.win_or_draw(state) == None:
        if player == -1:  # Bot's turn
            # Create fresh node for bot
            node = Node(prior_prob=0, player=player, action_index=None)
            node.set_state(state.copy())
            node = mcts.run_simulation(root_node=node, num_simulations=num_simulations, player=player)
            action, node, action_probs = mcts.select_move(node=node, mode="exploit", temperature=1)
            # Update state
            state = node.state.copy()
        else:  # Random player's turn
            action = random_player.get_action(state)
            if action is None:
                break
            # Apply random action to state
            action_index = np.argmax(action)
            state = game.get_next_state_from_next_player_prespective(state, action, player)
        
        player = -1 * player
    
    # Determine winner from bot's perspective (bot is player -1)
    winner = game.get_reward_for_next_player(state, player)
    if winner == -1:  # Player -1 (bot) won
        return 1
    elif winner == 1:  # Player 1 (random) won
        return -1
    else:  # Draw
        return 0

def main():
    # Initialize game
    game = TicTacToe()
    
    # Load the latest trained model
    model_path = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(1))
    if not os.path.exists(model_path):
        # Try loading model 0 if model 1 doesn't exist
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
    
    # Play half the games with bot going first
    print("\n[1/2] Bot playing as Player 1 (X - goes first):")
    for game_num in tqdm(range(num_games // 2), total=num_games // 2):
        result = play_game_bot_first(game, mcts, random_player, num_simulations)
        results.append({
            'game_number': game_num,
            'bot_position': 'first',
            'bot_player': 1,
            'result': result,
            'outcome': 'bot_win' if result == 1 else ('draw' if result == 0 else 'random_win')
        })
    
    # Play half the games with random going first
    print("\n[2/2] Bot playing as Player -1 (O - goes second):")
    for game_num in tqdm(range(num_games // 2, num_games), total=num_games // 2):
        result = play_game_random_first(game, mcts, random_player, num_simulations)
        results.append({
            'game_number': game_num,
            'bot_position': 'second',
            'bot_player': -1,
            'result': result,
            'outcome': 'bot_win' if result == 1 else ('draw' if result == 0 else 'random_win')
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

