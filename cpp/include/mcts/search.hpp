#pragma once
#include "node.hpp"
#include "../game/tictactoe.hpp"
#include "../nn/predictor.hpp"
#include <memory>
#include <vector>

namespace mcts {

class MCTS {
public:
    MCTS(std::shared_ptr<NNPredictor> predictor, int num_simulations, float c_puct = 2.0f);

    // Main API: Get action probabilities from MCTS search
    Policy get_action_probs(const game::TicTacToe& game,
                           int8_t player,
                           float temperature = 1.0f,
                           bool add_noise = false);

    // Get visit counts (for debugging/analysis)
    std::vector<int> get_visit_counts() const;

private:
    // Recursive MCTS simulation
    float simulate(game::TicTacToe& game, int8_t player, MCTSNode* node);

    // Helper methods
    void expand_node(MCTSNode* node, const game::TicTacToe& game, int8_t player);
    Policy apply_temperature(const std::vector<int>& visits, float temp) const;

    std::shared_ptr<NNPredictor> predictor_;
    int num_simulations_;
    float c_puct_;
};

} // namespace mcts
