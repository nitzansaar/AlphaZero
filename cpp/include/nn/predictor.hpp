#pragma once
#include "../common.hpp"
#include <utility>

namespace mcts {

// Interface for neural network predictions
class NNPredictor {
public:
    struct Prediction {
        Policy policy;  // Probability distribution over moves
        float value;    // Value estimate for the position
    };

    virtual ~NNPredictor() = default;

    // Predict policy and value for a given state
    virtual Prediction predict(const std::array<float, 27>& state) = 0;
};

// Random policy predictor for testing (Phase 2)
class RandomPredictor : public NNPredictor {
public:
    Prediction predict(const std::array<float, 27>& state) override;
};

} // namespace mcts
