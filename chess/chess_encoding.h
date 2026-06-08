#pragma once

/*
 * chess_encoding.h — AlphaZero chess move/board encoding in C++.
 *
 * A faithful C++ port of encoding.py.  Two responsibilities:
 *
 *   1. board_to_planes(board): canonical (NUM_PLANES, 8, 8) network input,
 *      always from the side-to-move's perspective (vertically mirrored and
 *      colors swapped when it is Black to move).
 *   2. move <-> index: bijective map between a chess::Move and an index in
 *      [0, ACTION_SIZE) using the AlphaZero 8x8x73 move-plane encoding, done
 *      in the same perspective frame as the planes.
 *
 * All geometry is computed in the perspective frame:
 *   - White to move: identity.
 *   - Black to move: vertical flip (square index ^ 56), so the mover's pawns
 *     always advance toward increasing rank.
 *
 * NOTE: Disservin's chess-library encodes castling as "king captures own
 * rook" (move.to() is the rook square).  chess_move_to_index converts this to
 * the king's destination square so the geometry matches python-chess, where
 * castling looks like a 2-square king slide.
 */

#include "chess.hpp"

// AlphaZero chess action space: 64 from-squares x 73 move planes = 4672.
static constexpr int CHESS_BOARD_SQ   = 64;
static constexpr int NUM_MOVE_PLANES  = 73;
static constexpr int CHESS_ACTION_SIZE = CHESS_BOARD_SQ * NUM_MOVE_PLANES;  // 4672

// NN input planes (see chess_board_to_planes): 12 piece + 1 stm + 4 castling
// + 1 en-passant + 1 halfmove clock.
static constexpr int CHESS_NUM_PLANES = 19;

// Move-plane layout within the 73 planes.
static constexpr int NUM_QUEEN_PLANES     = 56;  // planes [0, 56)
static constexpr int NUM_KNIGHT_PLANES    = 8;   // planes [56, 64)
static constexpr int NUM_UNDERPROMO_PLANES = 9;  // planes [64, 73)

// Theoretical maximum number of legal moves in any chess position.
static constexpr int MAX_LEGAL_MOVES = 218;

/*
 * Map a legal chess::Move to an action index in [0, CHESS_ACTION_SIZE).
 * `stm` is the side to move at the position the move is played from.
 * Mirrors encoding.move_to_index().
 */
int chess_move_to_index(const chess::Move &move, chess::Color stm);

/*
 * Inverse of chess_move_to_index.  Reconstructs the chess::Move for an action
 * index in the given position (used for tests / parity with Python; the MCTS
 * stores moves directly and does not rely on this in the hot path).
 * Returns a move whose .move() == chess::Move::NO_MOVE if the index does not
 * correspond to a geometrically valid move (off-board target).
 * Mirrors encoding.index_to_move().
 */
chess::Move chess_index_to_move(int action_idx, const chess::Board &board);

/*
 * Compute NN input planes: (CHESS_NUM_PLANES, 8, 8) float32 row-major, i.e.
 * planes_out[plane * 64 + rank * 8 + file].  Caller allocates
 * CHESS_NUM_PLANES * 64 floats.  Mirrors encoding.board_to_planes().
 */
void chess_board_to_planes(const chess::Board &board, float *planes_out);

/*
 * Generate legal moves for `board`, filling:
 *   action_out[i] : action index of move i               (length MAX_LEGAL_MOVES)
 *   move_out[i]   : the chess::Move object of move i      (length MAX_LEGAL_MOVES)
 * Returns the number of legal moves.  Used by expand_node so move generation
 * and index computation happen exactly once per expansion.
 */
int chess_legal_moves(const chess::Board &board,
                      int *action_out, chess::Move *move_out);
