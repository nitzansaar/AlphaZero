#include "../../include/game/tictactoe.hpp"
#include <algorithm>

namespace mcts::game {

TicTacToe::TicTacToe() {
    reset();
}

void TicTacToe::reset() {
    board_.fill(0);
    current_player_ = 1;
    move_count_ = 0;
}

bool TicTacToe::is_valid_move(int position) const {
    return position >= 0 && position < 9 && board_[position] == 0;
}

void TicTacToe::make_move(int position, int8_t player) {
    if (!is_valid_move(position)) {
        throw std::invalid_argument("Invalid move");
    }
    board_[position] = player;
    current_player_ = -player;  // Switch player
    move_count_++;
}

std::optional<int8_t> TicTacToe::check_winner() const {
    return mcts::game::check_winner(board_);
}

bool TicTacToe::is_terminal() const {
    return check_winner().has_value();
}

std::vector<int> TicTacToe::get_legal_moves() const {
    return get_valid_moves(board_);
}

std::array<float, 27> TicTacToe::get_canonical_state(int8_t player) const {
    return board_to_canonical_3d(board_, player);
}

// Utility functions

std::optional<int8_t> check_winner(const Board& board) {
    // Check all winning combinations
    const int wins[][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},  // rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},  // cols
        {0, 4, 8}, {2, 4, 6}              // diagonals
    };

    for (const auto& win : wins) {
        int a = win[0], b = win[1], c = win[2];
        if (board[a] == board[b] && board[b] == board[c] && board[a] != 0) {
            return board[a];
        }
    }

    // Check for draw (board full)
    bool has_empty = false;
    for (int i = 0; i < 9; i++) {
        if (board[i] == 0) {
            has_empty = true;
            break;
        }
    }

    if (!has_empty) {
        return 0;  // Draw
    }

    return std::nullopt;  // Game ongoing
}

std::vector<int> get_valid_moves(const Board& board) {
    std::vector<int> moves;
    for (int i = 0; i < 9; i++) {
        if (board[i] == 0) {
            moves.push_back(i);
        }
    }
    return moves;
}

std::array<float, 27> board_to_canonical_3d(const Board& board, int8_t player) {
    /*
     * Convert flat board to canonical 3-plane representation.
     * Canonical means current player is always represented as +1.
     *
     * Returns: 3x3x3 array where:
     *   planes[0] = current player positions (as +1 in original)
     *   planes[1] = opponent positions (as -1 in original)
     *   planes[2] = empty positions
     */
    std::array<float, 27> planes{};

    for (int i = 0; i < 9; i++) {
        // Flip perspective so current player is always +1
        int8_t canonical_value = board[i] * player;

        if (canonical_value == 1) {
            planes[i] = 1.0f;        // Current player
        } else if (canonical_value == -1) {
            planes[9 + i] = 1.0f;    // Opponent
        } else {
            planes[18 + i] = 1.0f;   // Empty
        }
    }

    return planes;
}

} // namespace mcts::game
