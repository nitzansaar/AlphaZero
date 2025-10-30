
import neural_network
import player
import game
import mcts

def interactive_game(checkpoint_step=50000):
    """
    Play an interactive game against the trained bot.
    
    Args:
        checkpoint_step: The training step checkpoint to load (default: 50000).
                        Set to None to use the best model from training.py session.
    """
    mcts.MCTS.PUCT_CONSTANT = 0.33
    
    # Determine which model to use
    if checkpoint_step is not None:
        # Use a specific checkpoint
        nn_check_pt = neural_network.nn_predictor.CHECK_POINTS_NAME + '-' + str(checkpoint_step)
        print(f"Loading checkpoint: {nn_check_pt}")
    else:
        # Use best model (only works if training.py was run in same session)
        nn_check_pt = 'best'
        print("Using best model from training session")
    
    try:
        player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type=nn_check_pt, temperature=0)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure you have trained a model first by running training.py")
        return
    
    player2 = player.Interactive_Player('o', 'Human')
    z_v_h_game = game.Game(player1, player2)
    
    try:
        outcome = z_v_h_game.run()

        z_v_h_game.board.display(clear=True)
        if outcome[1] == 0:
            print('Game ended in draw!')
        else:
            winner = player1 if outcome[1] == player1.type else player2
            print('{} won the game!'.format(winner.name))
    finally:
        # Close player to free TensorFlow session
        player1.close()


def main():
    interactive_game()

if __name__ == '__main__':
    main()
