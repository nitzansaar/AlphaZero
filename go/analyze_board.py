"""
Interactive board analysis tool for Go.
Place stones interactively, then view raw NN policy/value and MCTS visit distribution.

Usage:
    BOARD_SIZE=5 python analyze_board.py                    # 5x5, latest model
    BOARD_SIZE=9 python analyze_board.py                    # 9x9, latest model
    BOARD_SIZE=5 python analyze_board.py --model models_5x5/10_best_model.pt
    BOARD_SIZE=5 python analyze_board.py --sims 800         # custom sim count
"""

import argparse
import sys
import os
import numpy as np

from config import Config as cfg, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE
from game import Go, board_to_canonical_3d, idx_to_coord, coord_to_idx
from value_policy_function import ValuePolicyNetwork
from mcts import MonteCarloTreeSearch, Node


def find_latest_model():
    """Find the highest-numbered model in the models directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, cfg.SAVE_MODEL_PATH)
    if not os.path.exists(model_dir):
        return None
    models = [f for f in os.listdir(model_dir) if f.endswith('_best_model.pt')]
    if not models:
        return None
    def model_num(name):
        try:
            return int(name.split('_')[0])
        except ValueError:
            return -2
    models.sort(key=model_num)
    return os.path.join(model_dir, models[-1])


def render_board(state, game, highlight=None):
    """Render the board as a Go grid with box-drawing characters."""
    board = game.get_board(state).reshape(BOARD_SIZE, BOARD_SIZE)
    symbols = {0: '+', 1: 'X', -1: 'O'}

    col_labels = '     ' + '   '.join(f'{i}' for i in range(BOARD_SIZE))
    print()
    print(col_labels)
    print()
    for r in range(BOARD_SIZE):
        # Row with stones/intersections
        cells = []
        for c in range(BOARD_SIZE):
            cells.append(symbols[int(board[r, c])])
        row_str = '───'.join(cells)
        print(f' {r}   {row_str}   {r}')
        # Connector row (vertical lines between rows)
        if r < BOARD_SIZE - 1:
            vlines = '   '.join('│' for _ in range(BOARD_SIZE))
            print(f'     {vlines}')
    print()
    print(col_labels)
    print()
    ko = game.get_ko_point(state)
    passes = game.get_consecutive_passes(state)
    if ko >= 0:
        kr, kc = idx_to_coord(ko)
        print(f'  Ko point: ({kr},{kc})')
    if passes > 0:
        print(f'  Consecutive passes: {passes}')


def render_heatmap(values, label, board_state, game, include_pass=True):
    """Render a grid of values as a heatmap with the board overlay."""
    board = game.get_board(board_state).reshape(BOARD_SIZE, BOARD_SIZE)
    symbols = {1: 'X', -1: 'O'}

    print(f'\n  {label}')
    col_labels = '    ' + ' '.join(f'{i:>6}' for i in range(BOARD_SIZE))
    print(col_labels)
    print('    ' + '-------' * BOARD_SIZE)

    for r in range(BOARD_SIZE):
        cells = []
        for c in range(BOARD_SIZE):
            idx = coord_to_idx(r, c)
            stone = int(board[r, c])
            if stone != 0:
                cells.append(f'  [{symbols[stone]}] ')
            else:
                v = values[idx]
                if v >= 0.01:
                    cells.append(f'{v:>6.3f}')
                elif v > 0.001:
                    cells.append(f'{v:>6.4f}')
                else:
                    cells.append(f'    . ')
        print(f'{r:>2} | {" ".join(cells)}')

    if include_pass:
        pass_val = values[PASS_ACTION] if len(values) > PASS_ACTION else 0
        print(f'\n  Pass: {pass_val:.4f}')


def render_top_moves(values, label, n=10):
    """Show the top N moves ranked by value."""
    indexed = [(i, v) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: -x[1])
    print(f'\n  Top {n} moves ({label}):')
    print(f'  {"Rank":>4}  {"Move":>8}  {"Value":>8}')
    print(f'  {"----":>4}  {"--------":>8}  {"--------":>8}')
    for rank, (idx, val) in enumerate(indexed[:n], 1):
        if idx == PASS_ACTION:
            move_str = 'pass'
        else:
            r, c = idx_to_coord(idx)
            move_str = f'({r},{c})'
        print(f'  {rank:>4}  {move_str:>8}  {val:>8.4f}')


def run_analysis(state, player, game, vpn, mcts, num_sims):
    """Run NN evaluation and MCTS, then display results."""
    print('\n' + '=' * 60)
    print(f'  Current player: {"Black (X)" if player == 1 else "White (O)"}')
    print('=' * 60)

    # --- Raw NN output ---
    value, policy = vpn.get_vp(state, player)
    valid_moves = game.get_valid_moves(state, player)

    # Mask invalid moves and renormalize
    masked_policy = policy * valid_moves
    policy_sum = np.sum(masked_policy)
    if policy_sum > 0:
        masked_policy = masked_policy / policy_sum

    print(f'\n  NN Value (from current player perspective): {value:+.4f}')

    render_heatmap(masked_policy, 'NN Policy (masked & normalized)', state, game)
    render_top_moves(masked_policy, 'NN Policy')

    # --- MCTS ---
    print(f'\n  Running MCTS with {num_sims} simulations...')

    # MCTS expects the root state in canonical form (current player's stones = 1).
    # analyze_board stores state in absolute form (black=1, white=-1), so we must
    # convert when white is to move. We then run MCTS as player=1 so the internal
    # state tracking stays consistent.
    mcts_state = state.copy()
    if player == -1:
        mcts_state[:NUM_POSITIONS] *= -1
    root_node = Node(prior_prob=0, player=1, action_index=None)
    root_node.set_state(mcts_state)
    root_node = mcts.run_simulation(root_node, num_simulations=num_sims, player=1, add_noise=False)

    # Extract visit distribution
    visit_counts = np.zeros(ACTION_SIZE)
    q_values = np.zeros(ACTION_SIZE)
    for action_idx, child in root_node.children.items():
        visit_counts[action_idx] = child.total_visits_N
        q_values[action_idx] = child.mean_action_value_of_next_state_Q

    total_visits = np.sum(visit_counts)
    visit_dist = visit_counts / total_visits if total_visits > 0 else visit_counts

    render_heatmap(visit_dist, 'MCTS Visit Distribution', state, game)
    render_top_moves(visit_dist, 'MCTS Visits')

    # Q-value display for top visited moves
    print(f'\n  Q-values for top visited moves:')
    indexed = [(i, visit_counts[i], q_values[i]) for i in range(ACTION_SIZE) if visit_counts[i] > 0]
    indexed.sort(key=lambda x: -x[1])
    print(f'  {"Move":>8}  {"Visits":>8}  {"Q-value":>8}')
    print(f'  {"--------":>8}  {"--------":>8}  {"--------":>8}')
    for idx, visits, q in indexed[:10]:
        if idx == PASS_ACTION:
            move_str = 'pass'
        else:
            r, c = idx_to_coord(idx)
            move_str = f'({r},{c})'
        print(f'  {move_str:>8}  {int(visits):>8}  {q:>+8.4f}')

    print()


def main():
    parser = argparse.ArgumentParser(description='Interactive Go board analysis')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (default: latest in models dir)')
    parser.add_argument('--sims', type=int, default=1600,
                        help='Number of MCTS simulations (default: 1600)')
    args = parser.parse_args()

    model_path = args.model or find_latest_model()
    if model_path is None:
        print('No model found. Specify --model path.')
        sys.exit(1)

    print(f'Board size: {BOARD_SIZE}x{BOARD_SIZE}')
    print(f'Loading model: {model_path}')
    vpn = ValuePolicyNetwork(path=model_path, use_compile=False)
    game = Go()
    mcts = MonteCarloTreeSearch(game, vpn.get_vp)

    # Initialize empty board
    state = np.zeros(NUM_POSITIONS + 2)
    state[NUM_POSITIONS] = -1   # no ko
    state[NUM_POSITIONS + 1] = 0  # no passes
    player = 1  # Black starts

    print(f'\nCommands:')
    print(f'  row,col     - Place a stone (e.g. "2,3")')
    print(f'  pass        - Pass')
    print(f'  undo        - Undo last move')
    print(f'  clear       - Clear the board')
    print(f'  swap        - Swap current player without placing')
    print(f'  analyze / a - Run NN + MCTS analysis')
    print(f'  sims N      - Change number of MCTS simulations')
    print(f'  quit / q    - Exit')
    print()

    history = []  # (state, player) tuples for undo

    while True:
        render_board(state, game)
        player_name = 'Black (X)' if player == 1 else 'White (O)'
        try:
            cmd = input(f'[{player_name}] > ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ('quit', 'q'):
            break

        elif cmd in ('analyze', 'a'):
            run_analysis(state, player, game, vpn, mcts, args.sims)

        elif cmd == 'clear':
            state = np.zeros(NUM_POSITIONS + 2)
            state[NUM_POSITIONS] = -1
            state[NUM_POSITIONS + 1] = 0
            player = 1
            history.clear()
            print('Board cleared.\n')

        elif cmd == 'undo':
            if history:
                state, player = history.pop()
                print('Undone.\n')
            else:
                print('Nothing to undo.\n')

        elif cmd == 'swap':
            player *= -1
            print(f'Swapped to {("Black (X)" if player == 1 else "White (O)")}.\n')

        elif cmd == 'pass':
            history.append((state.copy(), player))
            state = game.apply_move(state, PASS_ACTION, player)
            player *= -1
            print('Passed.\n')

        elif cmd.startswith('sims '):
            try:
                args.sims = int(cmd.split()[1])
                print(f'Simulations set to {args.sims}.\n')
            except (ValueError, IndexError):
                print('Usage: sims N\n')

        else:
            # Try to parse as row,col
            try:
                parts = cmd.replace(' ', ',').split(',')
                r, c = int(parts[0]), int(parts[1])
                if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE:
                    print(f'Out of bounds. Use 0-{BOARD_SIZE-1}.\n')
                    continue
                action = coord_to_idx(r, c)
                if not game.is_valid_move(state, action, player):
                    print('Invalid move (occupied, suicide, or ko).\n')
                    continue
                history.append((state.copy(), player))
                state = game.apply_move(state, action, player)
                player *= -1
            except (ValueError, IndexError):
                print('Unknown command. Use "row,col", "analyze", "pass", "undo", "clear", "swap", "quit".\n')


if __name__ == '__main__':
    main()
