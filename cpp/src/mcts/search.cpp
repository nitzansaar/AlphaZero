#include "../../include/mcts/search.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

namespace mcts {

MCTS::MCTS(std::shared_ptr<NNPredictor> predictor, int num_simulations, float c_puct)
    : predictor_(predictor), num_simulations_(num_simulations), c_puct_(c_puct) {}

Policy MCTS::get_action_probs(const game::TicTacToe& game_state,
                               int8_t player,
                               float temperature,
                               bool add_noise) {
    // Create root node
    MCTSNode root(1.0f);

    // Get valid moves
    std::vector<int> valid_moves = game_state.get_legal_moves();
    if (valid_moves.empty()) {
        // No valid moves, return uniform (shouldn't happen)
        Policy probs;
        probs.fill(1.0f / 9.0f);
        return probs;
    }

    // Initialize root node children with NN policy
    auto canonical_state = game_state.get_canonical_state(player);
    auto prediction = predictor_->predict(canonical_state);

    // Mask invalid moves and normalize
    float policy_sum = 0.0f;
    for (int move : valid_moves) {
        policy_sum += prediction.policy[move];
    }

    // Create child nodes with normalized priors
    for (int move : valid_moves) {
        float prior = (policy_sum > 0) ? prediction.policy[move] / policy_sum : 1.0f / valid_moves.size();

        // Add Dirichlet noise for exploration (if requested)
        if (add_noise && valid_moves.size() > 1) {
            // Simple noise addition (full Dirichlet implementation in Phase 3)
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<float> dist(0.0f, 0.3f);
            float noise = dist(gen);
            prior = 0.75f * prior + 0.25f * noise;
        }

        root.add_child(move, std::make_unique<MCTSNode>(prior));
    }

    // Run simulations
    for (int i = 0; i < num_simulations_; i++) {
        // Make a copy of the game state for this simulation
        game::TicTacToe game_copy = game_state;
        simulate(game_copy, player, &root);
    }

    // Compute visit count distribution
    std::vector<int> visits(9, 0);
    for (const auto& [action, child] : root.children()) {
        visits[action] = child->visit_count();
    }

    // Apply temperature
    return apply_temperature(visits, temperature);
}

float MCTS::simulate(game::TicTacToe& game, int8_t player, MCTSNode* node) {
    // Check terminal state
    auto winner = game.check_winner();
    if (winner.has_value()) {
        // Return value from current player's perspective
        return static_cast<float>(winner.value() * player);
    }

    // Select action using UCB
    int action = node->select_child(c_puct_);
    if (action == -1) {
        // No children (shouldn't happen if properly initialized)
        return 0.0f;
    }

    // Apply action
    game.make_move(action, player);
    int8_t next_player = -player;

    // Get child node
    MCTSNode* child = node->get_child(action);
    if (!child) {
        return 0.0f;  // Shouldn't happen
    }

    float value;

    // Check if this is a leaf node (first visit)
    if (child->visit_count() == 0) {
        // Evaluate with neural network
        auto canonical_state = game.get_canonical_state(next_player);
        auto prediction = predictor_->predict(canonical_state);

        // Expand node
        std::vector<int> valid_moves = game.get_legal_moves();
        if (!valid_moves.empty()) {
            // Normalize policy for valid moves
            float policy_sum = 0.0f;
            for (int move : valid_moves) {
                policy_sum += prediction.policy[move];
            }

            for (int move : valid_moves) {
                float prior = (policy_sum > 0) ? prediction.policy[move] / policy_sum : 1.0f / valid_moves.size();
                child->add_child(move, std::make_unique<MCTSNode>(prior));
            }
        }

        // Value is from next_player's perspective, flip for current player
        value = -prediction.value;
    } else {
        // Recurse
        value = -simulate(game, next_player, child);
    }

    // Backpropagate
    child->update(value);

    return value;
}

Policy MCTS::apply_temperature(const std::vector<int>& visits, float temp) const {
    Policy probs;
    probs.fill(0.0f);

    if (temp == 0.0f) {
        // Greedy: pick most visited
        int best_action = std::distance(visits.begin(), std::max_element(visits.begin(), visits.end()));
        probs[best_action] = 1.0f;
    } else {
        // Boltzmann distribution
        std::vector<float> visits_temp(9);
        float sum = 0.0f;

        for (int i = 0; i < 9; i++) {
            visits_temp[i] = std::pow(static_cast<float>(visits[i]), 1.0f / temp);
            sum += visits_temp[i];
        }

        if (sum > 0.0f) {
            for (int i = 0; i < 9; i++) {
                probs[i] = visits_temp[i] / sum;
            }
        } else {
            // Uniform if no visits (shouldn't happen)
            probs.fill(1.0f / 9.0f);
        }
    }

    return probs;
}

std::vector<int> MCTS::get_visit_counts() const {
    // For debugging/testing
    return std::vector<int>(9, 0);
}

} // namespace mcts
