"""
AlphaZero-style chess encoding.

Two responsibilities:
  1. board_to_planes(board): canonical (NUM_INPUT_PLANES, 8, 8) network input,
     always from the side-to-move's perspective (board is vertically mirrored
     and colors swapped when it is Black to move).
  2. move <-> index: bijective map between a python-chess Move and an index in
     [0, 4672) using the AlphaZero 8x8x73 move-plane encoding. Encoding is done
     in the same canonical (perspective-relative) frame as the planes so the
     policy head is always "from the mover's point of view".

All move geometry is computed in the perspective frame:
  - White to move: identity.
  - Black to move: vertical flip (chess.square_mirror), so the mover's pawns
    always advance toward increasing rank.
"""
import numpy as np
import chess

from config import Config as cfg

# ----------------------------------------------------------------------------
# Move-plane geometry
# ----------------------------------------------------------------------------
# 8 sliding ("queen") directions as (file_delta, rank_delta) unit vectors.
QUEEN_DIRECTIONS = [
    (0, 1),    # N
    (1, 1),    # NE
    (1, 0),    # E
    (1, -1),   # SE
    (0, -1),   # S
    (-1, -1),  # SW
    (-1, 0),   # W
    (-1, 1),   # NW
]

# 8 knight move deltas (file_delta, rank_delta).
KNIGHT_DELTAS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]

# Underpromotion encoding: 3 file directions x 3 promotion pieces.
UNDERPROMO_FILE_DELTAS = [-1, 0, 1]  # capture-left, forward, capture-right
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

NUM_QUEEN_PLANES = 56   # planes [0, 56)
NUM_KNIGHT_PLANES = 8   # planes [56, 64)
NUM_UNDERPROMO_PLANES = 9  # planes [64, 73)
NUM_MOVE_PLANES = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + NUM_UNDERPROMO_PLANES  # 73


def _to_perspective(square, turn):
    """Map a board square into the side-to-move's perspective frame."""
    if turn == chess.WHITE:
        return square
    return chess.square_mirror(square)


# _to_perspective is its own inverse (mirror twice == identity), so use it both ways.
_from_perspective = _to_perspective


def move_to_index(move, board):
    """Map a (legal) chess.Move to an index in [0, ACTION_SIZE)."""
    turn = board.turn
    from_sq = _to_perspective(move.from_square, turn)
    to_sq = _to_perspective(move.to_square, turn)

    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    tf, tr = chess.square_file(to_sq), chess.square_rank(to_sq)
    df, dr = tf - ff, tr - fr

    # Underpromotion (knight / bishop / rook). Queen promotions fall through to
    # the queen-move planes below.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        dir_idx = UNDERPROMO_FILE_DELTAS.index(df)
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        plane = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + dir_idx * 3 + piece_idx
        return from_sq * NUM_MOVE_PLANES + plane

    # Knight move (unique L-shaped geometry).
    if (df, dr) in KNIGHT_DELTAS:
        plane = NUM_QUEEN_PLANES + KNIGHT_DELTAS.index((df, dr))
        return from_sq * NUM_MOVE_PLANES + plane

    # Sliding / king / pawn / castling / queen-promotion move.
    dir_unit = (np.sign(df), np.sign(dr))
    distance = max(abs(df), abs(dr))
    dir_idx = QUEEN_DIRECTIONS.index((int(dir_unit[0]), int(dir_unit[1])))
    plane = dir_idx * 7 + (distance - 1)
    return from_sq * NUM_MOVE_PLANES + plane


def index_to_move(index, board):
    """Inverse of move_to_index. Returns a chess.Move (without legality check)."""
    turn = board.turn
    from_sq = index // NUM_MOVE_PLANES
    plane = index % NUM_MOVE_PLANES

    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    promotion = None

    if plane < NUM_QUEEN_PLANES:
        dir_idx = plane // 7
        distance = plane % 7 + 1
        df_unit, dr_unit = QUEEN_DIRECTIONS[dir_idx]
        tf, tr = ff + df_unit * distance, fr + dr_unit * distance
    elif plane < NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES:
        df, dr = KNIGHT_DELTAS[plane - NUM_QUEEN_PLANES]
        tf, tr = ff + df, fr + dr
    else:
        u = plane - (NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES)
        dir_idx, piece_idx = divmod(u, 3)
        df = UNDERPROMO_FILE_DELTAS[dir_idx]
        tf, tr = ff + df, fr + 1
        promotion = UNDERPROMO_PIECES[piece_idx]

    if not (0 <= tf < 8 and 0 <= tr < 8):
        return None

    persp_from = chess.square(ff, fr)
    persp_to = chess.square(tf, tr)
    board_from = _from_perspective(persp_from, turn)
    board_to = _from_perspective(persp_to, turn)

    # Queen-plane pawn move reaching the last rank is a queen promotion.
    if promotion is None and tr == 7:
        piece = board.piece_at(board_from)
        if piece is not None and piece.piece_type == chess.PAWN:
            promotion = chess.QUEEN

    return chess.Move(board_from, board_to, promotion=promotion)


def legal_policy_mask(board):
    """Length-ACTION_SIZE float mask: 1.0 for every legal move, else 0.0."""
    mask = np.zeros(cfg.ACTION_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = 1.0
    return mask


def legal_move_index_map(board):
    """Dict {index: chess.Move} for the current legal moves (perspective frame)."""
    return {move_to_index(m, board): m for m in board.legal_moves}


# ----------------------------------------------------------------------------
# Board -> input planes
# ----------------------------------------------------------------------------
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]


def board_to_planes(board):
    """Canonical (NUM_INPUT_PLANES, 8, 8) float32 planes from mover's view."""
    # Canonicalize: the side to move always appears as White.
    canonical = board if board.turn == chess.WHITE else board.mirror()

    planes = np.zeros((cfg.NUM_INPUT_PLANES, 8, 8), dtype=np.float32)

    # Planes 0-5: our pieces; 6-11: opponent pieces.
    for color_idx, color in enumerate((chess.WHITE, chess.BLACK)):
        for pt_idx, piece_type in enumerate(_PIECE_TYPES):
            plane = color_idx * 6 + pt_idx
            for sq in canonical.pieces(piece_type, color):
                planes[plane, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    # Plane 12: side-to-move (constant 1 in canonical frame).
    planes[12, :, :] = 1.0

    # Planes 13-16: castling rights (own K, own Q, opp K, opp Q).
    if canonical.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if canonical.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if canonical.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if canonical.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    # Plane 17: en-passant target square.
    if canonical.ep_square is not None:
        planes[17, chess.square_rank(canonical.ep_square),
               chess.square_file(canonical.ep_square)] = 1.0

    # Plane 18: fifty-move (halfmove) clock, normalized.
    planes[18, :, :] = canonical.halfmove_clock / 100.0

    return planes
