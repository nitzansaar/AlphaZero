"""
Correctness checks for batched self-play, lazy MCTS expansion, batched
inference, and resignation — the pieces that changed when self-play was
parallelized.

Run from inside chess/:  python test_selfplay.py
"""
import random

import chess
import numpy as np

from config import Config as cfg
from game import ChessGame
from mcts import MonteCarloTreeSearch, Node
import encoding
import selfplay


# ----------------------------------------------------------------------------
# Lazy MCTS expansion
# ----------------------------------------------------------------------------
def test_lazy_expand_defers_boards():
    """expand() stores moves but does NOT build child boards; ensure_board
    materializes the correct board only when asked."""
    game = ChessGame()
    board = game.get_initial_board()
    root = Node(board, player=1, prior_prob=0.0)

    probs = encoding.legal_policy_mask(board)
    probs = probs / probs.sum()
    root.expand(probs, game)

    assert len(root.children) == board.legal_moves.count(), "wrong number of children"
    for child in root.children.values():
        assert child.board is None, "child board should be lazy (None) until visited"
        assert child.move is not None, "child must remember its move"

    # Materialize one child and confirm it equals the eager computation.
    child = next(iter(root.children.values()))
    materialized = child.ensure_board(game)
    expected = game.apply_move(board, child.move)
    assert materialized.fen() == expected.fen(), "ensure_board produced wrong board"
    # Idempotent: second call returns the same object.
    assert child.ensure_board(game) is materialized
    print("[ok] expand defers child boards; ensure_board materializes correctly")


def test_only_visited_nodes_get_boards():
    """After a search, every child with visits has a correct board; unvisited
    children stay unmaterialized (the whole point of the optimization)."""
    game = ChessGame()
    vpn = _net()
    mcts = MonteCarloTreeSearch(game, vpn.get_vp)

    root = mcts.make_root(game.get_initial_board())
    mcts.run_simulation(root, num_simulations=16, add_noise=True)

    visited = 0
    unvisited = 0
    for child in root.children.values():
        if child.total_visits_N > 0:
            visited += 1
            assert child.board is not None, "visited child must have a board"
            expected = game.apply_move(root.board, child.move)
            assert child.board.fen() == expected.fen(), "visited child board wrong"
        else:
            unvisited += 1
            assert child.board is None, "unvisited child should not be materialized"
    assert visited >= 1, "search should visit at least one child"
    assert unvisited >= 1, "with 16 sims some children should be unvisited (lazy win)"
    print(f"[ok] only visited nodes materialized ({visited} visited, {unvisited} lazy)")


def test_search_is_deterministic():
    """Same network + same RNG seed -> identical visit distribution. Confirms
    the lazy change did not perturb selection (which uses priors/Q/N, not boards)."""
    game = ChessGame()
    vpn = _net()
    mcts = MonteCarloTreeSearch(game, vpn.get_vp)

    def one_run():
        np.random.seed(123)
        root = mcts.make_root(game.get_initial_board())
        mcts.run_simulation(root, num_simulations=32, add_noise=True)
        _, _, probs = mcts.select_move(root, temperature=1.0)
        return probs

    a = one_run()
    b = one_run()
    assert np.array_equal(a, b), "search not reproducible under fixed seed"
    print("[ok] search is deterministic under a fixed seed")


# ----------------------------------------------------------------------------
# Terminal detection (claim_draw=False change)
# ----------------------------------------------------------------------------
def test_terminal_detection():
    game = ChessGame()

    # Fool's mate: 1. f3 e5 2. g4 Qh4#  -> Black wins.
    mate = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        mate.push_san(san)
    assert game.is_terminal(mate), "checkmate not detected"
    assert game.get_result(mate) == -1, "checkmate result should be Black win (-1)"

    # Stalemate: black king h8 has no legal move and is not in check.
    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stale.is_stalemate(), "test fen is not actually a stalemate"
    assert game.is_terminal(stale), "stalemate not detected"
    assert game.get_result(stale) == 0, "stalemate should be a draw (0)"

    # Fresh board is not terminal.
    assert not game.is_terminal(game.get_initial_board()), "start position flagged terminal"
    print("[ok] terminal detection: checkmate / stalemate / non-terminal")


# ----------------------------------------------------------------------------
# Batched inference
# ----------------------------------------------------------------------------
def test_get_vp_batch_matches_single():
    game = ChessGame()
    vpn = _net()

    boards = _sample_boards(game, n=5)
    batched = vpn.get_vp_batch(boards, pad_to=cfg.NUM_PARALLEL_GAMES)
    assert len(batched) == len(boards)

    for board, (bv, bp) in zip(boards, batched):
        sv, sp = vpn.get_vp(board)
        assert abs(sv - bv) < 1e-4, f"value mismatch {sv} vs {bv}"
        assert np.allclose(sp, bp, atol=1e-4), "policy mismatch single vs batched"
        # Masked + normalized to legal moves.
        mask = encoding.legal_policy_mask(board)
        assert abs(bp.sum() - 1.0) < 1e-4, "policy not normalized"
        assert np.all(bp[mask == 0] == 0.0), "policy has mass on illegal moves"
    print("[ok] get_vp_batch matches get_vp (value, policy, masking, padding)")


def test_get_vp_batch_padding_independent():
    """Padding to a larger fixed size must not change the real outputs."""
    game = ChessGame()
    vpn = _net()
    boards = _sample_boards(game, n=3)

    no_pad = vpn.get_vp_batch(boards, pad_to=None)
    padded = vpn.get_vp_batch(boards, pad_to=64)
    assert len(no_pad) == len(padded) == 3
    for (v0, p0), (v1, p1) in zip(no_pad, padded):
        assert abs(v0 - v1) < 1e-5 and np.allclose(p0, p1, atol=1e-5)
    assert vpn.get_vp_batch([]) == [], "empty batch should return empty list"
    print("[ok] get_vp_batch padding does not affect results; empty batch handled")


# ----------------------------------------------------------------------------
# Batched self-play driver
# ----------------------------------------------------------------------------
def test_driver_counts_and_validity():
    game = ChessGame()
    vpn = _net()
    _shrink(sims=8, max_moves=10)
    cfg.RESIGN_PLAYTHROUGH_FRAC = 1.0  # disable resign for a clean validity check

    games = list(selfplay.run_batched_selfplay(vpn, game, num_games=6, num_parallel=3))
    assert len(games) == 6, f"expected 6 games, got {len(games)}"

    for records, winner in games:
        assert winner in (-1, 0, 1), f"bad winner {winner}"
        assert len(records) >= 1, "empty game"
        # White moves first, players alternate every ply.
        players = [r[2] for r in records]
        assert players[0] == 1, "first mover should be White (+1)"
        for i in range(1, len(players)):
            assert players[i] == -players[i - 1], "players must alternate"
        # Every stored fen is a valid board and every policy is a normalized
        # sparse distribution over legal moves.
        for fen, (idx, probs), player in records:
            b = chess.Board(fen)  # raises if malformed
            assert len(idx) == len(probs)
            assert abs(float(np.sum(probs)) - 1.0) < 1e-3, "policy target not normalized"
            legal = encoding.legal_policy_mask(b)
            assert np.all(legal[np.asarray(idx, dtype=np.int64)] == 1.0), "policy on illegal move"
    print(f"[ok] driver produced 6 valid games (counts, fens, alternation, policy targets)")


def test_driver_refills_slots():
    """num_games > num_parallel must still complete every game (refill works)."""
    game = ChessGame()
    vpn = _net()
    _shrink(sims=6, max_moves=8)
    cfg.RESIGN_PLAYTHROUGH_FRAC = 1.0

    n = 7
    games = list(selfplay.run_batched_selfplay(vpn, game, num_games=n, num_parallel=2))
    assert len(games) == n, f"refill failed: expected {n}, got {len(games)}"
    # num_parallel larger than num_games is clamped, not an error.
    games2 = list(selfplay.run_batched_selfplay(vpn, game, num_games=2, num_parallel=64))
    assert len(games2) == 2
    print("[ok] driver refills finished slots and clamps over-large concurrency")


def test_resignation():
    game = ChessGame()
    vpn = _net()
    _shrink(sims=6, max_moves=40)

    saved = (cfg.RESIGN_THRESHOLD, cfg.RESIGN_CONSECUTIVE, cfg.RESIGN_PLAYTHROUGH_FRAC)
    try:
        # Force resignation on the very first eligible move.
        cfg.RESIGN_THRESHOLD = 1.0       # any value < 1.0 trips it (always true)
        cfg.RESIGN_CONSECUTIVE = 1
        cfg.RESIGN_PLAYTHROUGH_FRAC = 0.0
        np.random.seed(0)
        games = list(selfplay.run_batched_selfplay(vpn, game, num_games=4, num_parallel=4))
        for records, winner in games:
            assert len(records) == 1, f"resign should stop at move 1, got {len(records)}"
            # Move 1 = White to move (player +1) resigns -> Black (-1) wins.
            assert winner == -1, f"resigning side's opponent should win, got {winner}"

        # With playthrough = 1.0, no game resigns even though the threshold trips.
        cfg.RESIGN_PLAYTHROUGH_FRAC = 1.0
        np.random.seed(0)
        games2 = list(selfplay.run_batched_selfplay(vpn, game, num_games=3, num_parallel=3))
        assert all(len(r) > 1 for r, _ in games2), "playthrough games should not resign"
    finally:
        cfg.RESIGN_THRESHOLD, cfg.RESIGN_CONSECUTIVE, cfg.RESIGN_PLAYTHROUGH_FRAC = saved
    print("[ok] resignation stops games early with correct winner; playthrough disables it")


# ----------------------------------------------------------------------------
# Self-play output -> training data compatibility
# ----------------------------------------------------------------------------
def test_training_data_roundtrips_and_trains():
    """A batched self-play game flows through TrainingDataset (save/load) and a
    real forward+backward+optimizer step on the network without shape errors."""
    import torch
    from dataset import TrainingDataset, ChessDataset
    from model import NeuralNetwork

    game = ChessGame()
    vpn = _net()
    _shrink(sims=8, max_moves=10)
    cfg.RESIGN_PLAYTHROUGH_FRAC = 1.0

    td = TrainingDataset()
    for records, winner in selfplay.run_batched_selfplay(vpn, game, num_games=3, num_parallel=3):
        td.add_game_to_training_dataset(records, winner)
    assert len(td.training_dataset) > 0, "no training samples produced"

    # Save / load roundtrip (the on-disk format train.py consumes).
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ds.pkl")
        td.save(path)
        td2 = TrainingDataset()
        td2.load(path)
    assert len(td2.training_dataset) == len(td.training_dataset), "roundtrip lost samples"

    # Each sample carries an appended value label in [-1, 1].
    for sample in td2.training_dataset:
        assert len(sample) == 4, "sample missing value label"
        assert sample[3] in (-1, 0, 1), f"unexpected value label {sample[3]}"

    # Run one real training step exactly as train.py does (without DataLoader).
    ds = ChessDataset(td2.training_dataset)
    X, v, p = ds[0]
    X = X.unsqueeze(0); v = v.unsqueeze(0); p = p.unsqueeze(0)
    assert X.shape == (1, cfg.NUM_INPUT_PLANES, 8, 8), X.shape
    assert p.shape == (1, cfg.ACTION_SIZE), p.shape

    net = NeuralNetwork()
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    before = next(net.parameters()).detach().clone()
    yv, yp = net(X)
    assert yv.shape == (1, 1) and yp.shape == (1, cfg.ACTION_SIZE)
    vloss = torch.nn.functional.mse_loss(yv, v)
    aloss = -(p * torch.nn.functional.log_softmax(yp, dim=1)).sum(dim=1).mean()
    loss = vloss + aloss
    assert torch.isfinite(loss), "loss is not finite"
    opt.zero_grad(); loss.backward(); opt.step()
    after = next(net.parameters()).detach()
    assert not torch.equal(before, after), "optimizer step did not update weights"
    print("[ok] self-play data round-trips and trains the network for one step")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _net():
    """A small random network on CPU (no torch.compile)."""
    from value_policy_function import ValuePolicyNetwork
    return ValuePolicyNetwork(path=None, use_compile=False)


def _shrink(sims, max_moves):
    cfg.NUM_SIMULATIONS = sims
    cfg.MAX_MOVES = max_moves


def _sample_boards(game, n):
    """A handful of distinct, reachable positions (varied side-to-move)."""
    boards = []
    b = game.get_initial_board()
    boards.append(b.copy(stack=False))
    while len(boards) < n:
        moves = list(b.legal_moves)
        if not moves:
            b = game.get_initial_board()
            continue
        b = game.apply_move(b, random.choice(moves))
        boards.append(b.copy(stack=False))
    return boards


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    test_lazy_expand_defers_boards()
    test_only_visited_nodes_get_boards()
    test_search_is_deterministic()
    test_terminal_detection()
    test_get_vp_batch_matches_single()
    test_get_vp_batch_padding_independent()
    test_driver_counts_and_validity()
    test_driver_refills_slots()
    test_resignation()
    test_training_data_roundtrips_and_trains()

    print("\nAll batched self-play checks passed.")
