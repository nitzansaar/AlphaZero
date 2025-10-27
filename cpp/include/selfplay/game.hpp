#pragma once
#include "../mcts/search.hpp"
#include "../common.hpp"
#include <vector>

namespace mcts::selfplay {

class SelfPlayGame {
public:
    SelfPlayGame(std::shared_ptr<NNPredictor> predictor,
                 int num_simulations,
                 float temperature = 1.0f,
                 bool add_noise = true);

    // Play one full game and return experiences
    std::vector<Experience> play_game();

    // Get game result (-1, 0, or 1)
    int8_t get_game_result() const { return game_result_; }

private:
    std::shared_ptr<NNPredictor> predictor_;
    int num_simulations_;
    float temperature_;
    bool add_noise_;
    int8_t game_result_;
};

} // namespace mcts::selfplay
