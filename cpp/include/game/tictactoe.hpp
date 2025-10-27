#pragma once
#include "../common.hpp"
#include <optional>
#include <vector>

namespace mcts::game {

class TicTacToe {
public:
    TicTacToe();

    // Core game methods
    void reset();
    bool is_valid_move(int position) const;
    void make_move(int position, int8_t player);
    std::optional<int8_t> check_winner() const;
    bool is_terminal() const;

    // For MCTS
    std::vector<int> get_legal_moves() const;
    Board get_board() const { return board_; }
    void set_board(const Board& board) { board_ = board; }

    // Get canonical state (current player perspective)
    std::array<float, 27> get_canonical_state(int8_t player) const;

    // Get current player
    int8_t get_current_player() const { return current_player_; }
    void set_current_player(int8_t player) { current_player_ = player; }

private:
    Board board_;
    int8_t current_player_;
    int move_count_;
};

// Utility functions
std::optional<int8_t> check_winner(const Board& board);
std::vector<int> get_valid_moves(const Board& board);
std::array<float, 27> board_to_canonical_3d(const Board& board, int8_t player);

} // namespace mcts::game
