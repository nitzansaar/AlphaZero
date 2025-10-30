import player
import game
import neural_network
import mcts

def train():
    mcts.MCTS.get_tree_and_edges(reset=True)
    neural_network.nn_predictor.reset_nn_check_pts()
    nn_training_set = None

    iterations = 100  # Increased from 50 for better convergence

    for i in range(iterations):
        print(f'\n=== Training Iteration {i+1}/{iterations} ===')
        
        # Self-play phase
        player1 = player.Zero_Player('x', 'Bot_ONE', nn_type='best', temperature=1)
        player2 = player.Zero_Player('o', 'Bot_ONE', nn_type='best', temperature=1)
        self_play_game = game.Game(player1, player2)
        self_play_results = self_play_game.play(500)
        
        # Close players to free TensorFlow sessions
        player1.close()
        player2.close()
        
        augmented_self_play_results = neural_network.augment_data_set(self_play_results)

        mcts.MCTS.update_mcts_edges(augmented_self_play_results)
        nn_training_set = neural_network.update_nn_training_set(self_play_results, nn_training_set)

        neural_network.train_nn(nn_training_set)

        # Testing phase
        player1 = player.Zero_Player('x', 'Bot_ONE', nn_type='last', temperature=0)
        player2 = player.Zero_Player('o', 'Bot_ONE', nn_type='best', temperature=0)

        nn_test_game = game.Game(player1, player2)
        wins_player1, wins_player2 = nn_test_game.play_symmetric(100)
        
        # Close players to free TensorFlow sessions
        player1.close()
        player2.close()

        if wins_player1 >= wins_player2:
            neural_network.nn_predictor.BEST = neural_network.nn_predictor.LAST
            print(f'New best model! Wins: {wins_player1} vs {wins_player2}')
        else:
            print(f'Keeping old best model. Wins: {wins_player1} vs {wins_player2}')


def zero_vs_random():
    N_games = 100
    # Use the best model instead of a specific checkpoint
    player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type='best', temperature=0)
    player2 = player.Random_Player('o', 'Bot_RANDOM')
    z_vs_r_game = game.Game(player1, player2)
    w1, w2 = z_vs_r_game.play_symmetric(N_games)
    print('{} vs {} summary:'.format(player1.name, player2.name))
    print('wins={}, draws={}, losses={}'.format(w1, N_games-w1-w2, w2))
    
    # Close player to free TensorFlow session
    player1.close()


def main():
    train()
    zero_vs_random()

if __name__ == '__main__':
    main()
