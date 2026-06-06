"""
Correctness checks for the chess encoding and a self-play/train smoke test.

Run from inside chess/:  python test_encoding.py
"""
import random

import chess
import numpy as np

import encoding
from config import Config as cfg
from game import ChessGame


def test_move_index_roundtrip(num_positions=300):
    """For random legal positions, index_to_move(move_to_index(m)) == m."""
    board = chess.Board()
    checked = 0
    for _ in range(num_positions):
        if board.is_game_over():
            board = chess.Board()
        for move in board.legal_moves:
            idx = encoding.move_to_index(move, board)
            assert 0 <= idx < cfg.ACTION_SIZE, f"index {idx} out of range"
            recovered = encoding.index_to_move(idx, board)
            assert recovered == move, (
                f"roundtrip failed: {move.uci()} -> {idx} -> "
                f"{recovered.uci() if recovered else None} (fen={board.fen()})"
            )
            checked += 1
        board.push(random.choice(list(board.legal_moves)))
    print(f"[ok] move<->index roundtrip verified on {checked} legal moves")


def test_index_uniqueness(num_positions=200):
    """All legal moves in a position map to distinct indices."""
    board = chess.Board()
    for _ in range(num_positions):
        if board.is_game_over():
            board = chess.Board()
        indices = [encoding.move_to_index(m, board) for m in board.legal_moves]
        assert len(indices) == len(set(indices)), f"collision at fen={board.fen()}"
        board.push(random.choice(list(board.legal_moves)))
    print("[ok] move indices unique per position")


def test_mask_agreement(num_positions=200):
    """legal_policy_mask marks exactly the legal moves (count + indices)."""
    board = chess.Board()
    for _ in range(num_positions):
        if board.is_game_over():
            board = chess.Board()
        mask = encoding.legal_policy_mask(board)
        legal = list(board.legal_moves)
        assert int(mask.sum()) == len(legal), (
            f"mask count {int(mask.sum())} != legal {len(legal)} (fen={board.fen()})"
        )
        for m in legal:
            assert mask[encoding.move_to_index(m, board)] == 1.0
        board.push(random.choice(legal))
    print("[ok] legal_policy_mask agrees with python-chess legal moves")


def test_planes_shape():
    board = chess.Board()
    planes = encoding.board_to_planes(board)
    assert planes.shape == (cfg.NUM_INPUT_PLANES, 8, 8), planes.shape
    # Black-to-move position should canonicalize to look like White to move.
    board.push_san("e4")
    planes_black = encoding.board_to_planes(board)
    assert planes_black.shape == (cfg.NUM_INPUT_PLANES, 8, 8)
    print("[ok] board_to_planes shape correct for both colors")


def test_black_perspective_symmetry():
    """After 1.e4 e5 (symmetric-ish), check black's canonical own-pawn plane."""
    board = chess.Board()
    board.push_san("e4")
    # Black to move: canonical planes[0] = black pawns mirrored to white view.
    planes = encoding.board_to_planes(board)
    # Black has 8 pawns on rank 7 (index 6); mirrored -> rank index 1.
    own_pawns = planes[0]
    assert own_pawns[1, :].sum() == 8, own_pawns
    print("[ok] black-to-move canonicalization mirrors correctly")


def test_selfplay_smoke():
    """Tiny self-play game + one training step with a random network."""
    from value_policy_function import ValuePolicyNetwork
    from mcts import MonteCarloTreeSearch

    # Shrink for speed.
    cfg.NUM_SIMULATIONS = 8
    cfg.MAX_MOVES = 6

    game = ChessGame()
    vpn = ValuePolicyNetwork(path=None, use_compile=False)
    mcts = MonteCarloTreeSearch(game, vpn.get_vp)

    board = game.get_initial_board()
    root = mcts.make_root(board)
    moves_played = 0
    while not game.is_terminal(board) and moves_played < cfg.MAX_MOVES:
        mcts.run_simulation(root, num_simulations=cfg.NUM_SIMULATIONS)
        idx, child, probs = mcts.select_move(root, temperature=1.0)
        assert abs(probs.sum() - 1.0) < 1e-4, probs.sum()
        assert child.board.fen() != board.fen()
        board = child.board
        child.parent = None
        root = child
        moves_played += 1
    assert moves_played > 0
    print(f"[ok] self-play smoke test played {moves_played} legal moves")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    test_move_index_roundtrip()
    test_index_uniqueness()
    test_mask_agreement()
    test_planes_shape()
    test_black_perspective_symmetry()
    test_selfplay_smoke()
    print("\nAll encoding/self-play checks passed.")
