#pragma once
#include "../common.hpp"
#include "game.hpp"
#include <memory>
#include <vector>
#include <thread>
#include <mutex>

namespace mcts::selfplay {

/**
 * Parallel batch generator for self-play games.
 *
 * This class generates multiple self-play games in parallel using a thread pool,
 * following the KataGo architecture pattern.
 *
 * Key features:
 * - Thread-safe parallel game generation
 * - Efficient predictor sharing across threads
 * - Model update support (reload without recreation)
 * - Statistics collection (wins, draws, etc.)
 */
class BatchGenerator {
public:
    /**
     * Create a batch generator with a neural network predictor.
     *
     * @param predictor Shared pointer to neural network predictor (thread-safe)
     * @param num_threads Number of parallel threads for game generation
     */
    BatchGenerator(std::shared_ptr<NNPredictor> predictor, int num_threads = 8);

    /**
     * Generate a batch of self-play games in parallel.
     *
     * @param num_games Number of games to generate
     * @param num_simulations MCTS simulations per move
     * @param temperature Temperature for move selection
     * @param add_noise Whether to add Dirichlet noise for exploration
     * @return ExperienceBatch with all experiences and statistics
     */
    ExperienceBatch generate_batch(int num_games,
                                   int num_simulations = 50,
                                   float temperature = 1.0f,
                                   bool add_noise = true);

    /**
     * Update the neural network predictor.
     *
     * This allows efficient model updates during training without recreating
     * the BatchGenerator (KataGo-style).
     *
     * @param new_predictor New predictor to use
     */
    void update_predictor(std::shared_ptr<NNPredictor> new_predictor);

    /**
     * Get the number of threads used for parallel generation.
     */
    int num_threads() const { return num_threads_; }

private:
    std::shared_ptr<NNPredictor> predictor_;
    int num_threads_;
    std::mutex predictor_mutex_;  // Protect predictor updates
};

} // namespace mcts::selfplay
