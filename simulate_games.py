__author__ = 'Florin Bora'

import neural_network
import player
import game
import mcts
import numpy as np
import mcts_player

def simulate_games(num_games=100, checkpoint_step=100000, num_simulations=1600):
    """
    Simulate games between AlphaGo Zero player with MCTS and a random player.
    Uses pure self-play RL with no hard-coded strategies.
    
    Args:
        num_games: Number of games to simulate (default: 1000)
        checkpoint_step: The training step checkpoint to load (default: 100000).
                        Set to None to use the best model from training.py session.
        num_simulations: Number of MCTS simulations per move (default: 1600).
                        Same as AlphaGo Zero used for 19x19 Go. For 3x3 tic-tac-toe,
                        this ensures exhaustive search for perfect play.
    
    Note:
        With 1600 simulations on 3x3 tic-tac-toe, the MCTS will exhaustively explore
        the game tree, guaranteeing zero losses against random play.
    """
    print(f"\n{'='*60}")
    print(f"Simulating {num_games} games: AlphaGo Zero (MCTS) vs Random Player")
    print(f"{'='*60}\n")
    
    try:
        # Determine which model to use
        if checkpoint_step is not None:
            nn_check_pt = neural_network.nn_predictor.CHECK_POINTS_NAME + '-' + str(checkpoint_step)
            print(f"Loading checkpoint: {nn_check_pt}")
        else:
            nn_check_pt = 'best'
            print("Using best model from training session")
        
        # Create AlphaGo Zero player with MCTS
        smart_player = mcts_player.MCTS_Player(
            'x', 
            'AlphaGo_Zero_MCTS', 
            nn_type=nn_check_pt, 
            num_simulations=num_simulations,
            c_puct=1.5,  # Balance exploration and exploitation
            temperature=0  # Greedy selection for best play
        )
        
        random_player = player.Random_Player('o', 'Random_Bot')
        
        print(f"Players initialized successfully")
        print(f"  - {smart_player.name} (using {num_simulations} MCTS simulations per move)")
        print(f"  - {random_player.name} (Random)")
        print(f"\nRunning {num_games} games...\n")
        
        # Create game instance
        zero_vs_random = game.Game(smart_player, random_player)
        
        # Track results
        zero_wins = 0
        random_wins = 0
        draws = 0
        
        # Play games
        for i in range(num_games):
            result = zero_vs_random.run()
            winner = result[1]
            
            if winner == 0:
                draws += 1
            elif winner == 1:  # Player 1 (zero_player) wins
                zero_wins += 1
            else:  # Player 2 (random_player) wins
                random_wins += 1
            
            # Print progress every 10 games
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_games} games completed")
        
        # Print results
        print(f"\n{'='*60}")
        print(f"SIMULATION RESULTS")
        print(f"{'='*60}")
        print(f"Total Games:        {num_games}")
        print(f"AlphaGo Zero Wins:  {zero_wins} ({zero_wins/num_games*100:.1f}%)")
        print(f"Random Player Wins: {random_wins} ({random_wins/num_games*100:.1f}%)")
        print(f"Draws:              {draws} ({draws/num_games*100:.1f}%)")
        print(f"{'='*60}\n")
        
        # Now let AlphaGo Zero play second (Random player plays first)
        print("=" * 60)
        print("SECOND TEST: Random Player First, AlphaGo Zero Second")
        print("=" * 60)
        print("\nRunning games with AlphaGo Zero playing second...\n")
        
        # Create new player with Random going first
        random_player_first = player.Random_Player('x', 'Random_Bot')
        zero_player_second = mcts_player.MCTS_Player(
            'o', 
            'AlphaGo_Zero_MCTS', 
            nn_type=nn_check_pt, 
            num_simulations=num_simulations,
            c_puct=1.5,  # Balance exploration and exploitation
            temperature=0
        )
        
        random_vs_zero = game.Game(random_player_first, zero_player_second)
        
        # Track results for second test
        zero_wins_second = 0
        random_wins_second = 0
        draws_second = 0
        
        for i in range(num_games):
            result = random_vs_zero.run()
            winner = result[1]
            
            if winner == 0:
                draws_second += 1
            elif winner == 1:  # Player 1 (random_player) wins
                random_wins_second += 1
            else:  # Player 2 (zero_player) wins
                zero_wins_second += 1
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_games} games completed")
        
        print(f"\n{'='*60}")
        print(f"RESULTS (AlphaGo Zero Playing Second)")
        print(f"{'='*60}")
        print(f"Total Games:        {num_games}")
        print(f"Random Player Wins: {random_wins_second} ({random_wins_second/num_games*100:.1f}%)")
        print(f"AlphaGo Zero Wins:  {zero_wins_second} ({zero_wins_second/num_games*100:.1f}%)")
        print(f"Draws:              {draws_second} ({draws_second/num_games*100:.1f}%)")
        print(f"{'='*60}\n")
        
        # Combined statistics
        total_zero_wins = zero_wins + zero_wins_second
        total_random_wins = random_wins + random_wins_second
        total_draws = draws + draws_second
        total_games = num_games * 2
        
        print("=" * 60)
        print("COMBINED RESULTS (Both Orders)")
        print("=" * 60)
        print(f"Total Games:        {total_games}")
        print(f"AlphaGo Zero Wins:  {total_zero_wins} ({total_zero_wins/total_games*100:.1f}%)")
        print(f"Random Player Wins: {total_random_wins} ({total_random_wins/total_games*100:.1f}%)")
        print(f"Draws:              {total_draws} ({total_draws/total_games*100:.1f}%)")
        print(f"{'='*60}\n")
        
        if total_zero_wins > total_random_wins:
            print(f"✓ AlphaGo Zero dominated with a win rate of {total_zero_wins/total_games*100:.1f}%")
        elif total_random_wins > total_zero_wins:
            print(f"✗ Random player won more games (might indicate training issues)")
        else:
            print(f"= Both players won equal number of games")
        
        return {
            'total_games': total_games,
            'zero_wins': total_zero_wins,
            'random_wins': total_random_wins,
            'draws': total_draws,
            'zero_win_rate': total_zero_wins / total_games,
            'random_win_rate': total_random_wins / total_games,
            'draw_rate': total_draws / total_games,
            'zero_first': {
                'zero_wins': zero_wins,
                'random_wins': random_wins,
                'draws': draws
            },
            'zero_second': {
                'zero_wins': zero_wins_second,
                'random_wins': random_wins_second,
                'draws': draws_second
            }
        }
        
    except Exception as e:
        import traceback
        print(f"Error during simulation: {e}")
        traceback.print_exc()
        print(f"Make sure you have trained a model first by running training.py")
        return None
    
    finally:
        # Close players to free TensorFlow session
        if 'smart_player' in locals():
            smart_player.close()
        if 'zero_player_second' in locals():
            zero_player_second.close()


def main():
    """Run simulation with default parameters."""
    results = simulate_games(num_games=100, checkpoint_step=100000, num_simulations=1600)
    
    if results:
        print("\nSimulation completed successfully!")
    else:
        print("\nSimulation failed. Check the error messages above.")


if __name__ == '__main__':
    main()

