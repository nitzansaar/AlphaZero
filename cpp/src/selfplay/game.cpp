#include "../../include/selfplay/game.hpp"
#include "../../include/game/tictactoe.hpp"
#include <random>

namespace mcts::selfplay {

SelfPlayGame::SelfPlayGame(std::shared_ptr<NNPredictor> predictor,
                           int num_simulations,
                           float temperature,
                           bool add_noise)
    : predictor_(predictor),
      num_simulations_(num_simulations),
      temperature_(temperature),
      add_noise_(add_noise),
      game_result_(0) {}

std::vector<Experience> SelfPlayGame::play_game() {
    game::TicTacToe game;
    MCTS mcts(predictor_, num_simulations_);

    std::vector<Experience> experiences;
    int8_t current_player = 1;
    int move_count = 0;

    static std::random_device rd;
    static std::mt19937 gen(rd());

    while (true) {
        // Check if game is over
        auto winner = game.check_winner();
        if (winner.has_value()) {
            game_result_ = winner.value();

            // Assign values to all experiences based on game outcome
            for (size_t i = 0; i < experiences.size(); ++i) {
                // Value is from the perspective of the player who made the move
                // Move 0: player 1, Move 1: player -1, Move 2: player 1, etc.
                int8_t exp_player = (i % 2 == 0) ? 1 : -1;
                experiences[i].value = static_cast<float>(game_result_ * exp_player);
            }

            return experiences;
        }

        // Get canonical state (current player's perspective)
        auto canonical_state = game.get_canonical_state(current_player);

        // Use temperature decay: high temp early, low temp later
        float temp = (move_count < 10) ? temperature_ : 0.1f;

        // Get MCTS policy
        Policy mcts_policy = mcts.get_action_probs(game, current_player, temp, add_noise_);

        // Create experience (value will be filled in later)
        Experience exp;
        exp.state = canonical_state;
        exp.policy = mcts_policy;
        exp.value = 0.0f;  // Will be updated when game ends
        experiences.push_back(exp);

        // Sample move from MCTS policy
        std::discrete_distribution<int> dist(mcts_policy.begin(), mcts_policy.end());
        int move = dist(gen);

        // Make move
        game.make_move(move, current_player);
        current_player = -current_player;
        move_count++;
    }
}

} // namespace mcts::selfplay
