#include "../../include/selfplay/batch_generator.hpp"
#include <future>
#include <algorithm>

namespace mcts::selfplay {

BatchGenerator::BatchGenerator(std::shared_ptr<NNPredictor> predictor, int num_threads)
    : predictor_(predictor), num_threads_(num_threads) {

    if (num_threads_ <= 0) {
        num_threads_ = std::thread::hardware_concurrency();
        if (num_threads_ <= 0) {
            num_threads_ = 4;  // Fallback
        }
    }
}

void BatchGenerator::update_predictor(std::shared_ptr<NNPredictor> new_predictor) {
    std::lock_guard<std::mutex> lock(predictor_mutex_);
    predictor_ = new_predictor;
}

ExperienceBatch BatchGenerator::generate_batch(int num_games,
                                               int num_simulations,
                                               float temperature,
                                               bool add_noise) {
    // GIL is released by py::call_guard in bindings.cpp
    // This allows C++ threads to acquire GIL when calling back to Python

    // Launch parallel game generation tasks
    std::vector<std::future<std::pair<std::vector<Experience>, int8_t>>> futures;

    for (int i = 0; i < num_games; i++) {
        // Launch async task for each game
        futures.push_back(std::async(std::launch::async, [=]() {
            // Get predictor (thread-safe)
            std::shared_ptr<NNPredictor> local_predictor;
            {
                std::lock_guard<std::mutex> lock(predictor_mutex_);
                local_predictor = predictor_;
            }

            // Play one game
            SelfPlayGame game(local_predictor, num_simulations, temperature, add_noise);
            std::vector<Experience> experiences = game.play_game();
            int8_t result = game.get_game_result();

            return std::make_pair(experiences, result);
        }));
    }

    // Collect results from all games
    ExperienceBatch batch;
    batch.total_games = num_games;
    batch.player1_wins = 0;
    batch.player2_wins = 0;
    batch.draws = 0;

    for (auto& future : futures) {
        auto [experiences, result] = future.get();

        // Add experiences to batch
        batch.experiences.insert(batch.experiences.end(),
                                experiences.begin(),
                                experiences.end());

        // Update statistics
        if (result == 1) {
            batch.player1_wins++;
        } else if (result == -1) {
            batch.player2_wins++;
        } else {
            batch.draws++;
        }
    }

    return batch;
}

} // namespace mcts::selfplay
