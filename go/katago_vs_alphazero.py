"""
Play KataGo (trained .bin.gz model) vs our AlphaZero implementation.

Usage:
    BOARD_SIZE=9 python katago_vs_alphazero.py \
        --katago-model katago_models_9x9/mytraining-s120320-d3911050/model.bin.gz \
        --az-iter 142 \
        --games 20 \
        --simulations 200 \
        --katago-visits 200
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from glob import glob
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config as cfg, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE, KOMI
from game import Go
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KATAGO_BIN = os.path.join(SCRIPT_DIR, "KataGo", "cpp", "katago")
GTP_CONFIG  = os.path.join(SCRIPT_DIR, "KataGo", "cpp", "configs", "gtp_example.cfg")

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
            stderr=subprocess.DEVNULL,
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
                raise RuntimeError(f"KataGo process exited unexpectedly during '{command}'")
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

def load_az_model(iteration):
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


def latest_katago_model():
    base = os.path.join(SCRIPT_DIR, "katago_models_9x9")
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

def play_one_game(game, katago, mcts_az, az_color, simulations, verbose):
    """
    Play one game.
      az_color: 1  → AlphaZero plays Black
                -1 → AlphaZero plays White
    Returns: (winner, num_moves, history)   winner: 1=Black, -1=White, 0=draw
    """
    katago.reset()

    player = 1  # absolute player: 1=Black, -1=White
    node = Node(prior_prob=0, player=player, action_index=None)
    node.set_state(game.state.copy())
    move_count = 0
    history = []

    while True:
        result = game.winner(node.state, perspective=player)
        if result is not None:
            return result, move_count, history
        if move_count >= MAX_MOVES:
            return game.get_winner(node.state, perspective=player), move_count, history

        gtp_color = "b" if player == 1 else "w"

        if player == az_color:
            # --- AlphaZero's turn ---
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
                return -az_color, move_count, history  # all moves rejected; forfeit
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
                return -player, move_count, history
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

        move_count += 1
        color_name = "Black" if player == 1 else "White"
        board_str = format_board(node, -player)  # node is now from the next player's view
        history.append({
            "move": move_count,
            "who": who,
            "color": color_name,
            "gtp": az_to_gtp(action_index),
            "board": board_str,
        })
        if verbose:
            print(f"  Move {move_count:3d}: {who} ({color_name}) → {az_to_gtp(action_index)}")
            print(board_str)
            print()

        player *= -1


# ---------------------------------------------------------------------------
# Log writing
# ---------------------------------------------------------------------------

def write_game_log(filepath, game_logs, az_iter, katago_label, az_wins, kg_wins, draws, args):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"KataGo vs AlphaZero — {BOARD_SIZE}x{BOARD_SIZE} Go\n")
        f.write(f"  AlphaZero iter : {az_iter}  ({args.simulations} simulations/move)\n")
        f.write(f"  KataGo model   : {katago_label}  ({args.katago_visits} visits/move)\n")
        f.write(f"  Games          : {len(game_logs)}\n")
        f.write("=" * 70 + "\n\n")

        total = len(game_logs)
        f.write(f"AlphaZero: {az_wins}/{total} wins ({az_wins/total*100:.0f}%)\n")
        f.write(f"KataGo:    {kg_wins}/{total} wins ({kg_wins/total*100:.0f}%)\n")
        if draws:
            f.write(f"Draws:     {draws}\n")
        f.write("\n")

        for g in game_logs:
            f.write("-" * 70 + "\n")
            f.write(f"Game {g['game_num']:3d} | "
                    f"Black: {g['black']:20s}  White: {g['white']:20s} | "
                    f"{g['outcome']} ({g['num_moves']} moves)\n")
            f.write("-" * 70 + "\n\n")
            for m in g["history"]:
                f.write(f"  Move {m['move']:3d}: {m['who']} ({m['color']}) → {m['gtp']}\n")
                f.write(m["board"] + "\n\n")
            f.write("\n")

    print(f"Game log saved to: {filepath}")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def create_results_graph(filepath, az_iter, katago_label, az_wins, kg_wins, draws, total):
    """Save a bar chart PNG showing wins, losses, and draws."""
    labels = [f"AZ-{az_iter}", f"KG-{katago_label}", "Draw"]
    counts = [az_wins, kg_wins, draws]
    colors = ["#4C72B0", "#DD8452", "#8C8C8C"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{cnt} ({cnt/total*100:.0f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, total * 1.15)
    ax.set_ylabel("Games")
    ax.set_title(f"KataGo ({katago_label}) vs AlphaZero (iter {az_iter}) — {total} games")

    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Results graph saved to: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KataGo vs AlphaZero on 9x9 Go")
    parser.add_argument("--katago-model", default=None,
                        help="Path to KataGo model.bin.gz (overrides --katago-iter)")
    parser.add_argument("--katago-iter", type=int, default=None,
                        help="KataGo sequential model number (e.g. 37 for katago_models_9x9/037/)")
    parser.add_argument("--az-iter", type=int, required=True,
                        help="AlphaZero model iteration number (e.g. 142)")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of games to play (default: 10)")
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations per move for AlphaZero (default: 200)")
    parser.add_argument("--katago-visits", type=int, default=200,
                        help="Search visits per move for KataGo (default: 200)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every move and board state")
    args = parser.parse_args()

    if args.katago_model:
        katago_model = args.katago_model
    elif args.katago_iter is not None:
        folder = os.path.join(SCRIPT_DIR, "katago_models_9x9", f"{args.katago_iter:03d}")
        katago_model = _find_katago_model(folder)
        if katago_model is None:
            raise FileNotFoundError(f"KataGo model not found in: {folder}")
    else:
        katago_model = latest_katago_model()
    print(f"KataGo model : {katago_model}")
    print(f"AlphaZero    : iter {args.az_iter}")
    print(f"Games        : {args.games}")
    print(f"AZ sims      : {args.simulations}    KataGo visits: {args.katago_visits}")
    print(f"Board size   : {BOARD_SIZE}x{BOARD_SIZE}   Komi: {KOMI}")
    print()

    print("Loading AlphaZero model...", end=" ", flush=True)
    vpn = load_az_model(args.az_iter)
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
    game_logs = []
    katago_label = os.path.basename(os.path.dirname(katago_model))

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

        winner, num_moves, history = play_one_game(
            game, katago, mcts_az, az_color,
            simulations=args.simulations,
            verbose=args.verbose,
        )
        total_moves += num_moves

        az_label = f"AZ-{args.az_iter}"
        kg_label = f"KG-{katago_label}"
        black_label = az_label if az_color == 1 else kg_label
        white_label = kg_label if az_color == 1 else az_label

        if winner == az_color:
            result_str = "AZ wins"
            az_wins += 1
        elif winner == -az_color:
            result_str = "KG wins"
            kg_wins += 1
        else:
            result_str = "Draw"
            draws += 1

        game_logs.append({
            "game_num": i + 1,
            "black": black_label,
            "white": white_label,
            "outcome": result_str,
            "num_moves": num_moves,
            "history": history,
        })

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
    if draws:
        print(f"  Draws: {draws}")
    print(f"  Avg moves/game: {total_moves/total:.1f}")
    print("=" * 40)

    log_path = os.path.join(
        SCRIPT_DIR, "test_output_9x9",
        f"katago_{katago_label}_vs_az_{args.az_iter}.txt"
    )
    write_game_log(log_path, game_logs, args.az_iter, katago_label,
                   az_wins, kg_wins, draws, args)

    graph_path = log_path.replace(".txt", ".png")
    create_results_graph(graph_path, args.az_iter, katago_label,
                         az_wins, kg_wins, draws, total)


if __name__ == "__main__":
    main()
