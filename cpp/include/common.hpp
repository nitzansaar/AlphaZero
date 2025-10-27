#pragma once
#include <vector>
#include <array>
#include <cstdint>

namespace mcts {

// Board representation (flat array, row-major)
using Board = std::array<int8_t, 9>;

// Policy is 9 floats (probability for each move)
using Policy = std::array<float, 9>;

// Single training example
struct Experience {
    std::array<float, 27> state;  // 3×3×3 canonical board
    Policy policy;                 // MCTS-improved policy
    float value;                   // Game outcome from player perspective
};

// Batch of experiences from multiple games
struct ExperienceBatch {
    std::vector<Experience> experiences;
    int total_games;
    int player1_wins;
    int player2_wins;
    int draws;
};

} // namespace mcts
