#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/common.hpp"
#include "../include/game/tictactoe.hpp"
#include "../include/mcts/node.hpp"
#include "../include/mcts/search.hpp"
#include "../include/nn/predictor.hpp"
#include "../include/nn/python_callback_predictor.hpp"
#include "../include/selfplay/game.hpp"
#include "../include/selfplay/batch_generator.hpp"

namespace py = pybind11;

// Helper function to test basic functionality
std::string hello_world() {
    return "Hello from C++!";
}

PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS self-play engine for Tic-Tac-Toe";

    // Test function
    m.def("hello_world", &hello_world, "A simple hello world function");

    // Experience struct
    py::class_<mcts::Experience>(m, "Experience")
        .def(py::init<>())
        .def_readwrite("state", &mcts::Experience::state, "3x3x3 canonical board state")
        .def_readwrite("policy", &mcts::Experience::policy, "MCTS-improved policy")
        .def_readwrite("value", &mcts::Experience::value, "Game outcome from player perspective");

    // ExperienceBatch struct
    py::class_<mcts::ExperienceBatch>(m, "ExperienceBatch")
        .def(py::init<>())
        .def_readwrite("experiences", &mcts::ExperienceBatch::experiences, "List of experiences")
        .def_readwrite("total_games", &mcts::ExperienceBatch::total_games, "Total number of games played")
        .def_readwrite("player1_wins", &mcts::ExperienceBatch::player1_wins, "Player 1 wins")
        .def_readwrite("player2_wins", &mcts::ExperienceBatch::player2_wins, "Player 2 wins")
        .def_readwrite("draws", &mcts::ExperienceBatch::draws, "Number of draws");

    // TicTacToe game class
    py::class_<mcts::game::TicTacToe>(m, "TicTacToe")
        .def(py::init<>())
        .def("reset", &mcts::game::TicTacToe::reset, "Reset the game")
        .def("is_valid_move", &mcts::game::TicTacToe::is_valid_move, "Check if a move is valid")
        .def("make_move", &mcts::game::TicTacToe::make_move, "Make a move")
        .def("check_winner", &mcts::game::TicTacToe::check_winner, "Check for winner")
        .def("is_terminal", &mcts::game::TicTacToe::is_terminal, "Check if game is over")
        .def("get_legal_moves", &mcts::game::TicTacToe::get_legal_moves, "Get list of legal moves")
        .def("get_board", &mcts::game::TicTacToe::get_board, "Get current board state")
        .def("get_canonical_state", &mcts::game::TicTacToe::get_canonical_state, "Get canonical state")
        .def("get_current_player", &mcts::game::TicTacToe::get_current_player, "Get current player");

    // NNPredictor interface
    py::class_<mcts::NNPredictor, std::shared_ptr<mcts::NNPredictor>>(m, "NNPredictor")
        .def("predict", &mcts::NNPredictor::predict, "Predict policy and value");

    // NNPredictor::Prediction struct
    py::class_<mcts::NNPredictor::Prediction>(m, "Prediction")
        .def(py::init<>())
        .def_readwrite("policy", &mcts::NNPredictor::Prediction::policy)
        .def_readwrite("value", &mcts::NNPredictor::Prediction::value);

    // RandomPredictor
    py::class_<mcts::RandomPredictor, mcts::NNPredictor, std::shared_ptr<mcts::RandomPredictor>>(m, "RandomPredictor")
        .def(py::init<>(), "Create a random policy predictor");

    // PythonCallbackPredictor - allows C++ to call Python neural network
    py::class_<mcts::PythonCallbackPredictor, mcts::NNPredictor, std::shared_ptr<mcts::PythonCallbackPredictor>>(m, "PythonCallbackPredictor")
        .def(py::init<py::object>(),
             py::arg("py_predictor"),
             "Create predictor that calls Python model's predict(state) method");

    // MCTS search
    py::class_<mcts::MCTS>(m, "MCTS")
        .def(py::init<std::shared_ptr<mcts::NNPredictor>, int, float>(),
             py::arg("predictor"),
             py::arg("num_simulations"),
             py::arg("c_puct") = 2.0f,
             "Create MCTS search")
        .def("get_action_probs", &mcts::MCTS::get_action_probs,
             py::arg("game"),
             py::arg("player"),
             py::arg("temperature") = 1.0f,
             py::arg("add_noise") = false,
             "Get action probabilities from MCTS");

    // SelfPlayGame
    py::class_<mcts::selfplay::SelfPlayGame>(m, "SelfPlayGame")
        .def(py::init<std::shared_ptr<mcts::NNPredictor>, int, float, bool>(),
             py::arg("predictor"),
             py::arg("num_simulations"),
             py::arg("temperature") = 1.0f,
             py::arg("add_noise") = true,
             "Create a self-play game")
        .def("play_game", &mcts::selfplay::SelfPlayGame::play_game, "Play a full game")
        .def("get_game_result", &mcts::selfplay::SelfPlayGame::get_game_result, "Get game result");

    // BatchGenerator (Phase 4: Parallel batch generation)
    py::class_<mcts::selfplay::BatchGenerator>(m, "BatchGenerator")
        .def(py::init<std::shared_ptr<mcts::NNPredictor>, int>(),
             py::arg("predictor"),
             py::arg("num_threads") = 8,
             "Create a parallel batch generator")
        .def("generate_batch", &mcts::selfplay::BatchGenerator::generate_batch,
             py::arg("num_games"),
             py::arg("num_simulations") = 50,
             py::arg("temperature") = 1.0f,
             py::arg("add_noise") = true,
             "Generate a batch of games in parallel")
        .def("update_predictor", &mcts::selfplay::BatchGenerator::update_predictor,
             py::arg("new_predictor"),
             "Update the neural network predictor")
        .def("num_threads", &mcts::selfplay::BatchGenerator::num_threads,
             "Get number of threads");

    m.attr("__version__") = "0.3.0";
}
