#include "../../include/nn/predictor.hpp"
#include <random>

namespace mcts {

NNPredictor::Prediction RandomPredictor::predict(const std::array<float, 27>& state) {
    // Random policy: uniform distribution over all 9 moves
    Prediction pred;
    for (int i = 0; i < 9; i++) {
        pred.policy[i] = 1.0f / 9.0f;
    }

    // Random value between -1 and 1
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    pred.value = dist(gen);

    return pred;
}

} // namespace mcts
