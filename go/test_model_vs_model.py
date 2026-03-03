"""
Test models from different training iterations against each other.

Usage:
    # Play iteration 10 vs iteration 20 (10 games each side)
    BOARD_SIZE=9 python test_model_vs_model.py --model1 10 --model2 20

    # Gauntlet: test every N-th iteration against its neighbors
    BOARD_SIZE=9 python test_model_vs_model.py --gauntlet --step 5

    # Gauntlet with custom range
    BOARD_SIZE=9 python test_model_vs_model.py --gauntlet --start 0 --end 34 --step 5

    # Adjust games per matchup and MCTS simulations
    BOARD_SIZE=9 python test_model_vs_model.py --gauntlet --step 5 --games 20 --simulations 200

"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from config import Config as cfg
from game import Go, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_MOVES_PER_GAME = 150


def find_model_dir():
    """Find the model directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for d in [os.path.join(script_dir, cfg.SAVE_MODEL_PATH), cfg.SAVE_MODEL_PATH]:
        if os.path.isdir(d):
            return d
    return None


def get_available_iterations(model_dir):
    """Return sorted list of available model iteration numbers."""
    models = glob(os.path.join(model_dir, "*_best_model.pt"))
    iterations = []
    for f in models:
        basename = os.path.basename(f)
        num_str = basename.split("_")[0]
        if num_str.isdigit():
            iterations.append(int(num_str))
    return sorted(iterations)


def load_model(model_dir, iteration):
    """Load a model for a given iteration number. Returns a ValuePolicyNetwork."""
    model_path = os.path.join(model_dir, cfg.BEST_MODEL.format(iteration))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    vpn = ValuePolicyNetwork(model_path, use_compile=True)
    return vpn


def format_board(state):
    """Format board state as a string grid. X=Black, O=White, .=empty."""
    board = state[:NUM_POSITIONS].reshape(BOARD_SIZE, BOARD_SIZE)
    lines = []
    col_header = "   " + " ".join(f"{c:2d}" for c in range(BOARD_SIZE))
    lines.append(col_header)
    for r in range(BOARD_SIZE):
        row_str = f"{r:2d} "
        for c in range(BOARD_SIZE):
            v = board[r, c]
            if v == 1:
                row_str += " X "
            elif v == -1:
                row_str += " O "
            else:
                row_str += " . "
        lines.append(row_str)
    return "\n".join(lines)


def action_to_str(action_index):
    """Convert action index to human-readable string."""
    if action_index == PASS_ACTION:
        return "pass"
    row = action_index // BOARD_SIZE
    col = action_index % BOARD_SIZE
    return f"({row},{col})"


def get_absolute_board(node):
    """Convert a node's canonical state to absolute form (Black=+1, White=-1)."""
    state = node.state.copy()
    state[:NUM_POSITIONS] *= node.player
    return state


def play_game(game, mcts1, mcts2, num_simulations1=800, num_simulations2=800, max_moves=MAX_MOVES_PER_GAME):
    """
    Play a single game. mcts1 plays as Black (player 1), mcts2 plays as White (player -1).
    Reuses MCTS subtrees between moves (same pattern as selfplay.py).

    num_simulations1 / num_simulations2 can differ, allowing handicap matches.

    Returns:
        winner: 1 (Black/mcts1 wins), -1 (White/mcts2 wins), 0 (draw)
        num_moves: number of moves played
        history: list of dicts with move-by-move details
    """
    player = 1
    node = Node(prior_prob=0, player=player, action_index=None)
    node.set_state(game.state.copy())
    move_count = 0
    history = []

    while True:
        result = game.winner(node.state, perspective=player)
        if result is not None:
            return result, move_count, history

        if move_count >= max_moves:
            return game.get_winner(node.state, perspective=player), move_count, history

        current_mcts = mcts1 if player == 1 else mcts2
        current_sims = num_simulations1 if player == 1 else num_simulations2

        node = current_mcts.run_simulation(
            root_node=node, num_simulations=current_sims, player=player,
            add_noise=False,
        )

        # Record visit counts before selecting move
        visit_counts = np.zeros(ACTION_SIZE)
        for k, v in node.children.items():
            visit_counts[k] = v.total_visits_N
        top_indices = np.argsort(visit_counts)[::-1][:3]
        top_moves = [(action_to_str(int(idx)), int(visit_counts[idx])) for idx in top_indices if visit_counts[idx] > 0]

        action, node, _ = current_mcts.select_move(node=node, mode="exploit", temperature=1)
        action_index = np.argmax(action)

        move_count += 1
        player *= -1

        # node is now the child subtree (from new player's perspective)
        abs_state = get_absolute_board(node)

        history.append({
            "move": move_count,
            "player": -player,  # the player who just moved (before flip)
            "action": action_to_str(action_index),
            "top_moves": top_moves,
            "board": format_board(abs_state),
        })


def run_matchup(game, vpn1, vpn2, num_games=20, num_simulations1=800, num_simulations2=800,
                label1="Model1", label2="Model2"):
    """
    Run a matchup between two models. Each model plays half the games as Black.

    Returns dict with results.
    """
    mcts1 = MonteCarloTreeSearch(game, vpn1.get_vp, vpn1.get_vp_batch)
    mcts2 = MonteCarloTreeSearch(game, vpn2.get_vp, vpn2.get_vp_batch)

    wins1 = 0
    wins2 = 0
    draws = 0
    total_moves = 0
    game_logs = []

    desc = f"{label1}({num_simulations1}s) vs {label2}({num_simulations2}s)"
    for i in tqdm(range(num_games), desc=desc, leave=False):
        if i % 2 == 0:
            # model1 as Black, model2 as White
            result, moves, history = play_game(game, mcts1, mcts2, num_simulations1, num_simulations2)
            black_label, white_label = label1, label2
            if result == 1:
                wins1 += 1
            elif result == -1:
                wins2 += 1
            else:
                draws += 1
        else:
            # model2 as Black, model1 as White
            result, moves, history = play_game(game, mcts2, mcts1, num_simulations2, num_simulations1)
            black_label, white_label = label2, label1
            if result == 1:
                wins2 += 1
            elif result == -1:
                wins1 += 1
            else:
                draws += 1
        total_moves += moves

        outcome_str = "Black wins" if result == 1 else ("White wins" if result == -1 else "Draw")
        game_logs.append({
            "game_num": i + 1,
            "black": black_label,
            "white": white_label,
            "result": result,
            "outcome": outcome_str,
            "num_moves": moves,
            "history": history,
        })

    return {
        "model1": label1,
        "model2": label2,
        "wins1": wins1,
        "wins2": wins2,
        "draws": draws,
        "games": num_games,
        "avg_moves": total_moves / num_games if num_games > 0 else 0,
        "winrate1": wins1 / num_games * 100 if num_games > 0 else 0,
        "winrate2": wins2 / num_games * 100 if num_games > 0 else 0,
        "game_logs": game_logs,
    }


def write_game_log(filepath, results_list, header_info=""):
    """Write detailed game logs to a text file."""
    with open(filepath, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Model vs Model - {BOARD_SIZE}x{BOARD_SIZE} Go Game Log\n")
        if header_info:
            f.write(header_info)
        f.write("=" * 80 + "\n\n")

        for result in results_list:
            f.write(f"{'='*80}\n")
            f.write(f"MATCHUP: {result['model1']} vs {result['model2']}\n")
            f.write(f"  {result['model1']}: {result['wins1']} wins ({result['winrate1']:.0f}%)\n")
            f.write(f"  {result['model2']}: {result['wins2']} wins ({result['winrate2']:.0f}%)\n")
            f.write(f"  Draws: {result['draws']}  |  Avg moves/game: {result['avg_moves']:.1f}\n")
            f.write(f"{'='*80}\n\n")

            for game in result["game_logs"]:
                f.write(f"----- Game {game['game_num']} "
                        f"| Black: {game['black']}  White: {game['white']} "
                        f"| {game['outcome']} ({game['num_moves']} moves) -----\n\n")

                for move in game["history"]:
                    player_str = "Black(X)" if move["player"] == 1 else "White(O)"
                    top_str = ", ".join(f"{m}:{v}" for m, v in move["top_moves"])
                    f.write(f"  Move {move['move']:3d}: {player_str} plays {move['action']}"
                            f"  [top: {top_str}]\n")
                    f.write(move["board"] + "\n\n")

                f.write("\n")

    print(f"Game log saved to: {filepath}")


def print_matchup_result(result):
    """Print a single matchup result."""
    print(f"\n  {result['model1']} vs {result['model2']}:")
    print(f"    {result['model1']:>12s}: {result['wins1']} wins ({result['winrate1']:.0f}%)")
    print(f"    {result['model2']:>12s}: {result['wins2']} wins ({result['winrate2']:.0f}%)")
    print(f"    {'Draws':>12s}: {result['draws']}")
    print(f"    Avg moves/game: {result['avg_moves']:.1f}")


def run_single_matchup(args):
    """Run a single matchup between two specified iterations."""
    model_dir = find_model_dir()
    if not model_dir:
        print("ERROR: No model directory found!")
        return

    game = Go()

    print(f"Loading iteration {args.model1}...")
    vpn1 = load_model(model_dir, args.model1)
    print(f"Loading iteration {args.model2}...")
    vpn2 = load_model(model_dir, args.model2)

    print(f"\n{'='*60}")
    print(f"Model vs Model: Iteration {args.model1} vs Iteration {args.model2}")
    print(f"Board: {BOARD_SIZE}x{BOARD_SIZE} Go")
    print(f"Games: {args.games} ({args.games//2} per side)")
    print(f"MCTS simulations: {args.simulations1} (model1) / {args.simulations2} (model2)")
    print(f"{'='*60}")

    result = run_matchup(
        game, vpn1, vpn2,
        num_games=args.games,
        num_simulations1=args.simulations1,
        num_simulations2=args.simulations2,
        label1=f"iter_{args.model1}",
        label2=f"iter_{args.model2}",
    )
    print_matchup_result(result)

    # Save game log
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, cfg.TEST_OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"model_vs_model_{args.model1}_vs_{args.model2}.txt")
    header = (f"Iterations: {args.model1} vs {args.model2}\n"
              f"Games: {args.games} | Simulations: {args.simulations1} (model1) / {args.simulations2} (model2)\n")
    write_game_log(log_path, [result], header)


def run_gauntlet(args):
    """Run a gauntlet: consecutive iterations play against each other."""
    model_dir = find_model_dir()
    if not model_dir:
        print("ERROR: No model directory found!")
        return

    available = get_available_iterations(model_dir)
    if not available:
        print("ERROR: No models found!")
        return

    print(f"Available iterations: {available[0]} to {available[-1]} ({len(available)} models)")

    start = args.start if args.start is not None else available[0]
    end = args.end if args.end is not None else available[-1]
    step = args.step

    # Build list of iterations to test
    iterations = [i for i in available if start <= i <= end]
    if step > 1:
        sampled = [iterations[0]]
        for it in iterations:
            if it - sampled[-1] >= step:
                sampled.append(it)
        if sampled[-1] != iterations[-1]:
            sampled.append(iterations[-1])
        iterations = sampled

    if len(iterations) < 2:
        print(f"ERROR: Need at least 2 iterations to compare, got {len(iterations)}")
        return

    print(f"\nTesting iterations: {iterations}")
    print(f"Matchups: {len(iterations) - 1}")
    print(f"Games per matchup: {args.games}")
    print(f"MCTS simulations: {args.simulations1} (older) / {args.simulations2} (newer)")
    print(f"Board: {BOARD_SIZE}x{BOARD_SIZE} Go")
    print(f"{'='*60}")

    game = Go()
    results = []

    # Pre-load all models
    models = {}
    for it in iterations:
        print(f"Loading iteration {it}...")
        models[it] = load_model(model_dir, it)

    print(f"\n{'='*60}")
    print("Running matchups...")
    print(f"{'='*60}")

    for i in range(len(iterations) - 1):
        it_old = iterations[i]
        it_new = iterations[i + 1]

        result = run_matchup(
            game, models[it_old], models[it_new],
            num_games=args.games,
            num_simulations1=args.simulations1,
            num_simulations2=args.simulations2,
            label1=f"iter_{it_old}",
            label2=f"iter_{it_new}",
        )
        results.append(result)
        print_matchup_result(result)

    # Free GPU memory
    del models
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary table
    print(f"\n{'='*60}")
    print("GAUNTLET SUMMARY")
    print(f"{'='*60}")
    print(f"{'Matchup':<25s} {'Newer Wins':>10s} {'Older Wins':>10s} {'Draws':>6s} {'Newer WR':>9s}")
    print("-" * 65)

    newer_wins_total = 0
    older_wins_total = 0
    draws_total = 0

    for r in results:
        newer_wr = r["winrate2"]
        print(f"{r['model1']} vs {r['model2']:<10s} {r['wins2']:>10d} {r['wins1']:>10d} {r['draws']:>6d} {newer_wr:>8.0f}%")
        newer_wins_total += r["wins2"]
        older_wins_total += r["wins1"]
        draws_total += r["draws"]

    total_games = newer_wins_total + older_wins_total + draws_total
    print("-" * 65)
    print(f"{'TOTAL':<25s} {newer_wins_total:>10d} {older_wins_total:>10d} {draws_total:>6d} {newer_wins_total/total_games*100:>8.1f}%")

    # Save game log
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, cfg.TEST_OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "model_vs_model_gauntlet.txt")
    header = (f"Gauntlet: iterations {iterations}\n"
              f"Games per matchup: {args.games} | Simulations: {args.simulations1} (older) / {args.simulations2} (newer)\n")
    write_game_log(log_path, results, header)

    # Plot results
    plot_gauntlet_results(results, iterations, args)


def plot_gauntlet_results(results, iterations, args):
    """Create visualization of gauntlet results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, cfg.TEST_OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # --- Plot 1: Win rate of newer model in each matchup ---
    ax = axes[0]
    matchup_labels = [f"{r['model1']}\nvs\n{r['model2']}" for r in results]
    newer_wr = [r["winrate2"] for r in results]
    older_wr = [r["winrate1"] for r in results]
    draw_pct = [r["draws"] / r["games"] * 100 for r in results]

    x = np.arange(len(results))
    width = 0.25

    bars1 = ax.bar(x - width, newer_wr, width, label="Newer model wins %", color="green", alpha=0.8)
    bars2 = ax.bar(x, older_wr, width, label="Older model wins %", color="red", alpha=0.8)
    bars3 = ax.bar(x + width, draw_pct, width, label="Draw %", color="gray", alpha=0.8)

    ax.set_ylabel("Percentage")
    ax.set_title(f"Model vs Model Gauntlet - {BOARD_SIZE}x{BOARD_SIZE} Go\n({args.games} games, {args.simulations} sims/move)")
    ax.set_xticks(x)
    ax.set_xticklabels(matchup_labels, fontsize=8)
    ax.axhline(y=50, color="black", linestyle="--", alpha=0.3, label="50% baseline")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=7)

    # --- Plot 2: Cumulative newer-model win rate across iterations ---
    ax2 = axes[1]
    # For each iteration (except the first), compute its win rate as the newer model
    iter_labels = []
    cumulative_wr = []

    for i, r in enumerate(results):
        it_new = iterations[i + 1]
        iter_labels.append(f"iter_{it_new}")
        cumulative_wr.append(r["winrate2"])

    ax2.plot(iter_labels, cumulative_wr, "go-", linewidth=2, markersize=8, label="Win rate vs previous iteration")
    ax2.axhline(y=50, color="black", linestyle="--", alpha=0.3, label="50% (no improvement)")
    ax2.set_ylabel("Win Rate vs Previous (%)")
    ax2.set_xlabel("Model Iteration (newer)")
    ax2.set_title("Is the Model Improving?")
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)

    for i, wr in enumerate(cumulative_wr):
        ax2.annotate(f"{wr:.0f}%", (iter_labels[i], wr), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "model_vs_model_results.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nResults chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Go models from different iterations against each other")
    parser.add_argument("--model1", type=int, help="First model iteration number")
    parser.add_argument("--model2", type=int, help="Second model iteration number")
    parser.add_argument("--gauntlet", action="store_true", help="Run gauntlet: consecutive iterations play each other")
    parser.add_argument("--start", type=int, default=None, help="Start iteration for gauntlet")
    parser.add_argument("--end", type=int, default=None, help="End iteration for gauntlet")
    parser.add_argument("--step", type=int, default=5, help="Step size between iterations in gauntlet (default: 5)")
    parser.add_argument("--games", type=int, default=20, help="Number of games per matchup (default: 20)")
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations per move for both models (default: 200)")
    parser.add_argument("--simulations1", type=int, default=None, help="MCTS simulations for model1/older (overrides --simulations)")
    parser.add_argument("--simulations2", type=int, default=None, help="MCTS simulations for model2/newer (overrides --simulations)")

    args = parser.parse_args()

    # Resolve per-model sim counts: --simulations1/2 override --simulations
    if args.simulations1 is None:
        args.simulations1 = args.simulations
    if args.simulations2 is None:
        args.simulations2 = args.simulations

    if args.gauntlet:
        run_gauntlet(args)
    elif args.model1 is not None and args.model2 is not None:
        run_single_matchup(args)
    else:
        # Default: list available models
        model_dir = find_model_dir()
        if model_dir:
            available = get_available_iterations(model_dir)
            print(f"Available model iterations: {available}")
            print(f"\nUsage examples:")
            print(f"  Single matchup:  BOARD_SIZE=9 python test_model_vs_model.py --model1 10 --model2 20")
            print(f"  Gauntlet:        BOARD_SIZE=9 python test_model_vs_model.py --gauntlet --step 5")
            print(f"  Custom range:    BOARD_SIZE=9 python test_model_vs_model.py --gauntlet --start 0 --end 34 --step 5")
        else:
            print("No model directory found.")


if __name__ == "__main__":
    main()
