"""
Play KataGo vs our AlphaZero implementation on 9x9 Go.

KataGo model — pick one of:
  --katago-elo 482          pretrained_katago_models/katago-elo-482.gz  (available: 482, 802)
  --katago-elo 802          pretrained_katago_models/katago-elo-802.gz
  --katago-model <path>     any .bin.gz or .txt.gz model file (full path)

AlphaZero model:
  --az-iter 236             iteration number; loads models_9x9_base/236_best_model.pt by default
  --az-model-dir <dir>      override the model directory (default: cfg.SAVE_MODEL_PATH = models_9x9_base)

Examples:
    # Single game with verbose board output
    BOARD_SIZE=9 python katago_vs_alphazero.py \\
        --katago-elo 482 --az-iter 236 --games 1 --verbose

    # 10-game series, more search budget on both sides
    BOARD_SIZE=9 python katago_vs_alphazero.py \\
        --katago-elo 802 --az-iter 200 --games 10

    # Custom KataGo model by path
    BOARD_SIZE=9 python katago_vs_alphazero.py \\
        --katago-model /path/to/model.bin.gz --az-iter 142 --games 20
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from glob import glob
import torch

from config import Config as cfg, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE, KOMI
from game import Go
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KATAGO_BIN = os.path.join(SCRIPT_DIR, "KataGo", "cpp", "katago")
GTP_CONFIG  = os.path.join(SCRIPT_DIR, "KataGo", "cpp", "configs", "gtp_example.cfg")
PRETRAINED_KATAGO_DIR = os.path.join(SCRIPT_DIR, "pretrained_katago_models")

# GTP column letters: A-H then J (skip I), standard Go convention
GTP_COLS = "ABCDEFGHJ"[:BOARD_SIZE]

MAX_MOVES = BOARD_SIZE * BOARD_SIZE * 3


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def az_to_gtp(action_index):
    """AlphaZero action index → GTP string, e.g. 36 → 'E5', 81 → 'pass'."""
    if action_index == PASS_ACTION:
        return "pass"
    row = action_index // BOARD_SIZE
    col = action_index % BOARD_SIZE
    return f"{GTP_COLS[col]}{BOARD_SIZE - row}"


def gtp_to_az(gtp_move):
    """GTP string → AlphaZero action index. Returns None on resign."""
    gtp_move = gtp_move.strip().upper()
    if gtp_move in ("PASS", ""):
        return PASS_ACTION
    if gtp_move == "RESIGN":
        return None
    col = GTP_COLS.index(gtp_move[0])
    row = BOARD_SIZE - int(gtp_move[1:])
    return row * BOARD_SIZE + col


# ---------------------------------------------------------------------------
# KataGo GTP wrapper
# ---------------------------------------------------------------------------

class KataGoGTP:
    def __init__(self, model_path, visits=200):
        override = f"maxVisits={visits},logToStderr=false,logDir="
        cmd = [
            KATAGO_BIN, "gtp",
            "-model", model_path,
            "-config", GTP_CONFIG,
            "-override-config", override,
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._cmd(f"boardsize {BOARD_SIZE}")
        self._cmd(f"komi {KOMI}")
        self._cmd("clear_board")

    def _cmd(self, command):
        """Send a GTP command, return the response text (stripped)."""
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                stderr_out = self.proc.stderr.read()
                msg = f"KataGo process exited unexpectedly during '{command}'"
                if stderr_out.strip():
                    msg += f"\nKataGo stderr:\n{stderr_out.strip()}"
                raise RuntimeError(msg)
            line = line.rstrip("\n")
            if line.startswith("=") or line.startswith("?"):
                lines.append(line)
                # Responses end with a blank line
                while True:
                    nxt = self.proc.stdout.readline().rstrip("\n")
                    if nxt == "":
                        break
                    lines.append(nxt)
                break
        resp = "\n".join(lines)
        if resp.startswith("?"):
            raise RuntimeError(f"KataGo error for '{command}': {resp}")
        return resp.lstrip("=").strip()

    def reset(self):
        self._cmd("clear_board")

    def play(self, color, action_index):
        """Tell KataGo about a move that was already made."""
        self._cmd(f"play {color} {az_to_gtp(action_index)}")

    def genmove(self, color):
        """Ask KataGo to pick and play a move. Returns action_index or None (resign)."""
        resp = self._cmd(f"genmove {color}")
        return gtp_to_az(resp)

    def close(self):
        try:
            self._cmd("quit")
        except Exception:
            pass
        self.proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_az_model(iteration, model_dir=None):
    if model_dir is None:
        model_dir = os.path.join(SCRIPT_DIR, cfg.SAVE_MODEL_PATH)
    model_path = os.path.join(model_dir, cfg.BEST_MODEL.format(iteration))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"AlphaZero model not found: {model_path}")
    return ValuePolicyNetwork(model_path, use_compile=True)


def _find_katago_model(folder):
    """Return the model file in folder, trying .txt.gz then .bin.gz."""
    for ext in ("model.txt.gz", "model.bin.gz"):
        p = os.path.join(folder, ext)
        if os.path.exists(p):
            return p
    return None


def _find_katago_by_elo(elo):
    """Return path to pretrained_katago_models/katago-elo-{elo}.gz, or raise."""
    path = os.path.join(PRETRAINED_KATAGO_DIR, f"katago-elo-{elo}.gz")
    if os.path.exists(path):
        return path
    available = sorted(os.listdir(PRETRAINED_KATAGO_DIR))
    raise FileNotFoundError(
        f"No pretrained KataGo model for elo={elo}. Available: {available}"
    )


def latest_katago_model():
    base = os.path.join(SCRIPT_DIR, "pretrained_katago_models_9x9")
    folders = sorted(glob(os.path.join(base, "*")))
    for folder in reversed(folders):
        p = _find_katago_model(folder)
        if p:
            return p
    raise FileNotFoundError("No KataGo models found in katago_models_9x9/")


def format_board(node, player):
    """Print board in absolute form (Black=X, White=O)."""
    abs_board = node.state[:NUM_POSITIONS].copy() * node.player
    board_2d = abs_board.reshape(BOARD_SIZE, BOARD_SIZE)
    header = "    " + "  ".join(GTP_COLS)
    lines = [header]
    for r in range(BOARD_SIZE):
        row_label = f"{BOARD_SIZE - r:2d}  "
        cells = " ".join(
            "X" if board_2d[r, c] == 1 else "O" if board_2d[r, c] == -1 else "."
            for c in range(BOARD_SIZE)
        )
        lines.append(row_label + cells)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_one_game(game, katago, mcts_az, vpn, az_color, simulations, verbose):
    """
    Play one game.
      az_color: 1  → AlphaZero plays Black
                -1 → AlphaZero plays White
    Returns: (winner, num_moves)   winner: 1=Black, -1=White, 0=draw
    """
    katago.reset()

    player = 1  # absolute player: 1=Black, -1=White
    node = Node(prior_prob=0, player=player, action_index=None)
    node.set_state(game.state.copy())
    move_count = 0
    # Board history in absolute form (Black=+1, White=-1), newest first.
    # Boards BEFORE the current position — used to fill 17-plane history planes.
    hist_boards_abs = []

    while True:
        result = game.winner(node.state, perspective=player)
        if result is not None:
            return result, move_count
        if move_count >= MAX_MOVES:
            return game.get_winner(node.state, perspective=player), move_count

        # Capture absolute board of current position before the move is made.
        # Formula works uniformly: state * player = absolute (Black=+1, White=-1).
        curr_abs_board = node.state[:NUM_POSITIONS].copy() * player

        gtp_color = "b" if player == 1 else "w"

        if player == az_color:
            # --- AlphaZero's turn ---
            # Give the VPN the game history so _build_history can build correct
            # per-leaf history by walking up the MCTS tree and then appending this.
            vpn.set_history(hist_boards_abs)
            node = mcts_az.run_simulation(
                root_node=node, num_simulations=simulations,
                player=player, add_noise=False,
            )
            # Try moves in MCTS visit-count order until KataGo accepts one.
            # Fallback is needed because game.py uses simple ko while KataGo
            # uses positional superko, so they can disagree on legality.
            sorted_actions = sorted(
                node.children, key=lambda k: node.children[k].total_visits_N, reverse=True
            )
            prev_state = node.state
            action_index = None
            for candidate in sorted_actions + [PASS_ACTION]:
                try:
                    katago.play(gtp_color, candidate)
                    action_index = candidate
                    break
                except RuntimeError:
                    continue
            if action_index is None:
                return -az_color, move_count  # all moves rejected; forfeit
            if action_index in node.children:
                node = node.children[action_index]
                if node.state is None:
                    node.set_state(game.get_next_state_from_next_player_prespective(
                        prev_state, action_index, 1))
            else:
                new_node = Node(prior_prob=0, player=-player, action_index=action_index)
                new_node.set_state(game.get_next_state_from_next_player_prespective(
                    prev_state, action_index, 1))
                node.children[action_index] = new_node
                node = new_node
            who = "AZ"
        else:
            # --- KataGo's turn ---
            action_index = katago.genmove(gtp_color)
            if action_index is None:
                # KataGo resigned — other player wins
                return -player, move_count
            # Always recompute state: MCTS may have created this child node during
            # tree expansion with state=None (only visited leaves get state set).
            prev_state = node.state
            new_state = game.get_next_state_from_next_player_prespective(
                prev_state, action_index, 1
            )
            if action_index in node.children:
                node = node.children[action_index]
                node.set_state(new_state)
            else:
                new_node = Node(prior_prob=0, player=-player, action_index=action_index)
                new_node.set_state(new_state)
                node.children[action_index] = new_node
                node = new_node
            who = "KG"

        # Add the board before this move to history (it's now 1 step in the past).
        hist_boards_abs = [curr_abs_board] + hist_boards_abs[:6]

        move_count += 1
        if verbose:
            color_name = "Black" if player == 1 else "White"
            board_str = format_board(node, -player)
            print(f"  Move {move_count:3d}: {who} ({color_name}) → {az_to_gtp(action_index)}")
            print(board_str)
            print()

        player *= -1




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KataGo vs AlphaZero on 9x9 Go")
    parser.add_argument("--katago-model", default=None,
                        help="Path to KataGo model.bin.gz (overrides --katago-iter)")
    parser.add_argument("--katago-iter", type=int, default=None,
                        help="KataGo sequential model number (e.g. 37 for katago_models_9x9/037/)")
    parser.add_argument("--katago-elo", type=int, default=None,
                        help="ELO of pretrained KataGo model in pretrained_katago_models/ "
                             "(e.g. 482 for katago-elo-482.gz)")
    parser.add_argument("--az-iter", type=int, required=True,
                        help="AlphaZero model iteration number (e.g. 142)")
    parser.add_argument("--az-model-dir", default=None,
                        help="Directory containing AlphaZero checkpoints "
                             "(default: cfg.SAVE_MODEL_PATH = models_9x9_base)")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of games to play (default: 10)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move for AlphaZero")
    parser.add_argument("--katago-visits", type=int, default=10,
                        help="Search visits per move for KataGo")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every move and board state")
    args = parser.parse_args()

    if args.katago_model:
        katago_model = args.katago_model
    elif args.katago_elo is not None:
        katago_model = _find_katago_by_elo(args.katago_elo)
    elif args.katago_iter is not None:
        folder = os.path.join(SCRIPT_DIR, "pretrained_katago_models_9x9", f"{args.katago_iter:03d}")
        katago_model = _find_katago_model(folder)
        if katago_model is None:
            raise FileNotFoundError(f"KataGo model not found in: {folder}")
    else:
        katago_model = latest_katago_model()
    print(f"KataGo model : {katago_model}")
    print(f"AlphaZero    : iter {args.az_iter}")
    print(f"Games        : {args.games}")
    print()

    az_model_dir = args.az_model_dir
    print("Loading AlphaZero model...", end=" ", flush=True)
    vpn = load_az_model(args.az_iter, model_dir=az_model_dir)
    print("done")

    game = Go()
    mcts_az = MonteCarloTreeSearch(game, vpn.get_vp, vpn.get_vp_batch)

    print("Starting KataGo...", end=" ", flush=True)
    katago = KataGoGTP(katago_model, visits=args.katago_visits)
    print("done\n")

    az_wins = 0
    kg_wins = 0
    draws = 0
    total_moves = 0
    katago_label = os.path.basename(katago_model).removesuffix(".gz")

    for i in range(args.games):
        # Alternate colours each game
        az_color = 1 if i % 2 == 0 else -1
        az_color_name = "Black" if az_color == 1 else "White"
        kg_color_name = "White" if az_color == 1 else "Black"

        if args.verbose:
            print(f"=== Game {i+1}/{args.games} | AZ={az_color_name} KG={kg_color_name} ===\n")
        else:
            print(f"Game {i+1}/{args.games} (AZ={az_color_name}) ... ", end="", flush=True)

        # Reset MCTS trees between games
        game.__init__()
        mcts_az = MonteCarloTreeSearch(game, vpn.get_vp, vpn.get_vp_batch)

        winner, num_moves = play_one_game(
            game, katago, mcts_az, vpn, az_color,
            simulations=args.simulations,
            verbose=args.verbose,
        )
        total_moves += num_moves

        if winner == az_color:
            result_str = "AZ wins"
            az_wins += 1
        elif winner == -az_color:
            result_str = "KG wins"
            kg_wins += 1
        else:
            result_str = "Draw"
            draws += 1

        if args.verbose:
            print(f"Result: {result_str} ({num_moves} moves)\n{'='*50}\n")
        else:
            print(f"{result_str} ({num_moves} moves)")

    katago.close()

    total = args.games
    print()
    print("=" * 40)
    print(f"Results over {total} games")
    print(f"  AlphaZero (iter {args.az_iter}): {az_wins} wins ({az_wins/total*100:.0f}%)")
    print(f"  KataGo ({katago_label}): {kg_wins} wins ({kg_wins/total*100:.0f}%)")
    print("=" * 40)



if __name__ == "__main__":
    main()
