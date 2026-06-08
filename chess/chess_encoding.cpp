#include "chess_encoding.h"

#include <cstring>
#include <cstdlib>

using chess::Board;
using chess::Color;
using chess::Move;
using chess::PieceType;
using chess::Square;

/* ── Move-plane geometry (mirrors encoding.py tables) ─────────────────────── */

// 8 sliding ("queen") directions as (file_delta, rank_delta) unit vectors.
static const int QUEEN_DIR[8][2] = {
    { 0,  1},   // N
    { 1,  1},   // NE
    { 1,  0},   // E
    { 1, -1},   // SE
    { 0, -1},   // S
    {-1, -1},   // SW
    {-1,  0},   // W
    {-1,  1},   // NW
};

// 8 knight move deltas (file_delta, rank_delta).
static const int KNIGHT_DELTA[8][2] = {
    { 1,  2},
    { 2,  1},
    { 2, -1},
    { 1, -2},
    {-1, -2},
    {-2, -1},
    {-2,  1},
    {-1,  2},
};

// Underpromotion encoding: file directions [-1, 0, 1] x pieces {N, B, R}.
// UNDERPROMO_PIECES = [KNIGHT, BISHOP, ROOK] → piece_idx = int(pt) - 1.

static inline int sgn(int x) { return (x > 0) - (x < 0); }

// Vertical mirror of a square index (rank flip); matches chess.square_mirror.
static inline int persp_sq(int sq, Color stm) {
    return (stm == Color::WHITE) ? sq : (sq ^ 56);
}

// Index of a queen direction unit vector, or -1 if not a unit direction.
static int queen_dir_index(int sdf, int sdr) {
    for (int i = 0; i < 8; i++)
        if (QUEEN_DIR[i][0] == sdf && QUEEN_DIR[i][1] == sdr) return i;
    return -1;
}

// Index of a knight delta, or -1.
static int knight_delta_index(int df, int dr) {
    for (int i = 0; i < 8; i++)
        if (KNIGHT_DELTA[i][0] == df && KNIGHT_DELTA[i][1] == dr) return i;
    return -1;
}

/* ── Castling helper ──────────────────────────────────────────────────────
 *
 * Disservin encodes castling as king-captures-rook: move.to() is the rook
 * square.  Return the king's true destination square (g- or c-file) so the
 * geometry matches python-chess's 2-square king slide.
 */
static int castling_king_dest(const Move &move) {
    int from   = move.from().index();
    int rook   = move.to().index();
    int rank   = from >> 3;
    bool kingside = (rook > from);          // rook on h-side has higher index
    int file   = kingside ? 6 /*G*/ : 2 /*C*/;
    return rank * 8 + file;
}

/* ── move -> index ────────────────────────────────────────────────────────── */

int chess_move_to_index(const Move &move, Color stm)
{
    int from_abs = move.from().index();
    int to_abs;

    if (move.typeOf() == Move::CASTLING)
        to_abs = castling_king_dest(move);
    else
        to_abs = move.to().index();

    int from_sq = persp_sq(from_abs, stm);
    int to_sq   = persp_sq(to_abs,   stm);

    int ff = from_sq & 7, fr = from_sq >> 3;
    int tf = to_sq   & 7, tr = to_sq   >> 3;
    int df = tf - ff, dr = tr - fr;

    // Underpromotion (knight / bishop / rook).  Queen promotions fall through.
    if (move.typeOf() == Move::PROMOTION &&
        move.promotionType() != PieceType(PieceType::QUEEN)) {
        int dir_idx   = df + 1;                       // [-1,0,1] -> [0,1,2]
        int piece_idx = (int)move.promotionType() - 1; // KNIGHT(1)->0 ...
        int plane = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + dir_idx * 3 + piece_idx;
        return from_sq * NUM_MOVE_PLANES + plane;
    }

    // Knight move (unique L-shaped geometry).
    int kn = knight_delta_index(df, dr);
    if (kn >= 0) {
        int plane = NUM_QUEEN_PLANES + kn;
        return from_sq * NUM_MOVE_PLANES + plane;
    }

    // Sliding / king / pawn / castling / queen-promotion move.
    int dir_idx  = queen_dir_index(sgn(df), sgn(dr));
    int distance = std::abs(df) > std::abs(dr) ? std::abs(df) : std::abs(dr);
    int plane = dir_idx * 7 + (distance - 1);
    return from_sq * NUM_MOVE_PLANES + plane;
}

/* ── index -> move ────────────────────────────────────────────────────────── */

chess::Move chess_index_to_move(int action_idx, const Board &board)
{
    Color stm   = board.sideToMove();
    int from_sq = action_idx / NUM_MOVE_PLANES;   // perspective from-square
    int plane   = action_idx % NUM_MOVE_PLANES;

    int ff = from_sq & 7, fr = from_sq >> 3;
    int tf, tr;
    PieceType promo = PieceType(PieceType::NONE);

    if (plane < NUM_QUEEN_PLANES) {
        int dir_idx  = plane / 7;
        int distance = plane % 7 + 1;
        tf = ff + QUEEN_DIR[dir_idx][0] * distance;
        tr = fr + QUEEN_DIR[dir_idx][1] * distance;
    } else if (plane < NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES) {
        int k = plane - NUM_QUEEN_PLANES;
        tf = ff + KNIGHT_DELTA[k][0];
        tr = fr + KNIGHT_DELTA[k][1];
    } else {
        int u = plane - (NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES);
        int dir_idx   = u / 3;
        int piece_idx = u % 3;
        tf = ff + (dir_idx - 1);   // [0,1,2] -> [-1,0,1]
        tr = fr + 1;
        static const PieceType::underlying UP[3] = {
            PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK};
        promo = UP[piece_idx];
    }

    if (tf < 0 || tf >= 8 || tr < 0 || tr >= 8)
        return chess::Move(Move::NO_MOVE);

    int persp_from = fr * 8 + ff;
    int persp_to   = tr * 8 + tf;
    int board_from = persp_sq(persp_from, stm);
    int board_to   = persp_sq(persp_to,   stm);

    PieceType moved = board.at<PieceType>(Square(board_from));

    // Queen-plane pawn move reaching the last rank (perspective tr==7) is a
    // queen promotion.
    if (promo == PieceType(PieceType::NONE) && tr == 7 &&
        moved == PieceType(PieceType::PAWN)) {
        promo = PieceType(PieceType::QUEEN);
    }

    Square sq_from(board_from), sq_to(board_to);

    // Promotion move.
    if (promo != PieceType(PieceType::NONE))
        return Move::make<Move::PROMOTION>(sq_from, sq_to, promo);

    // Castling: king slides two files.  Rebuild as king-captures-rook so the
    // move matches what movegen produces (and makeMove expects).
    if (moved == PieceType(PieceType::KING) &&
        std::abs((board_to & 7) - (board_from & 7)) == 2) {
        bool kingside = (board_to & 7) > (board_from & 7);
        int  rank     = board_from >> 3;
        int  rook_sq  = rank * 8 + (kingside ? 7 : 0);
        return Move::make<Move::CASTLING>(sq_from, Square(rook_sq));
    }

    // En passant: pawn moves diagonally onto an empty square.
    if (moved == PieceType(PieceType::PAWN) &&
        (board_to & 7) != (board_from & 7) &&
        board.at(sq_to) == chess::Piece::NONE) {
        return Move::make<Move::ENPASSANT>(sq_from, sq_to);
    }

    return Move::make<Move::NORMAL>(sq_from, sq_to);
}

/* ── board -> planes ──────────────────────────────────────────────────────── */

void chess_board_to_planes(const Board &board, float *planes_out)
{
    std::memset(planes_out, 0, CHESS_NUM_PLANES * CHESS_BOARD_SQ * sizeof(float));

    Color stm  = board.sideToMove();
    Color them = ~stm;

    static const PieceType::underlying PTS[6] = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN,  PieceType::KING};

    // Planes 0-5: our pieces; 6-11: opponent pieces (canonical / mirrored).
    for (int ci = 0; ci < 2; ci++) {
        Color color = (ci == 0) ? stm : them;
        for (int pt = 0; pt < 6; pt++) {
            int plane = ci * 6 + pt;
            chess::Bitboard bb = board.pieces(PieceType(PTS[pt]), color);
            while (bb) {
                int sq  = bb.pop();
                int csq = persp_sq(sq, stm);
                planes_out[plane * 64 + csq] = 1.0f;   // csq = rank*8 + file
            }
        }
    }

    // Plane 12: side-to-move (constant 1 in canonical frame).
    for (int i = 0; i < 64; i++) planes_out[12 * 64 + i] = 1.0f;

    // Planes 13-16: castling rights (own K, own Q, opp K, opp Q).
    auto cr = board.castlingRights();
    using Side = chess::Board::CastlingRights::Side;
    if (cr.has(stm,  Side::KING_SIDE))  for (int i = 0; i < 64; i++) planes_out[13 * 64 + i] = 1.0f;
    if (cr.has(stm,  Side::QUEEN_SIDE)) for (int i = 0; i < 64; i++) planes_out[14 * 64 + i] = 1.0f;
    if (cr.has(them, Side::KING_SIDE))  for (int i = 0; i < 64; i++) planes_out[15 * 64 + i] = 1.0f;
    if (cr.has(them, Side::QUEEN_SIDE)) for (int i = 0; i < 64; i++) planes_out[16 * 64 + i] = 1.0f;

    // Plane 17: en-passant target square.
    Square ep = board.enpassantSq();
    if (ep != Square::NO_SQ) {
        int csq = persp_sq(ep.index(), stm);
        planes_out[17 * 64 + csq] = 1.0f;
    }

    // Plane 18: fifty-move (halfmove) clock, normalized.
    float hm = (float)board.halfMoveClock() / 100.0f;
    for (int i = 0; i < 64; i++) planes_out[18 * 64 + i] = hm;
}

/* ── legal moves + action indices ─────────────────────────────────────────── */

int chess_legal_moves(const Board &board, int *action_out, chess::Move *move_out)
{
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    Color stm = board.sideToMove();

    int n = 0;
    for (const auto &m : moves) {
        move_out[n]   = m;
        action_out[n] = chess_move_to_index(m, stm);
        n++;
    }
    return n;
}
