import os
import numpy as np
import torch
from glob import glob
from config import Config as cfg
from game import Go, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE
from mcts import MonteCarloTreeSearch, Node
from value_policy_function import ValuePolicyNetwork
from model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for model_dir in [os.path.join(script_dir, cfg.SAVE_MODEL_PATH), cfg.SAVE_MODEL_PATH]:
        if os.path.isdir(model_dir):
            models = glob(os.path.join(model_dir, "*_best_model.pt"))
            if models:
                models.sort(key=os.path.getmtime, reverse=True)
                return models[0]
    return None


def parse_coords(text):
    """Parse coordinates in various formats:
    - '1,1 2,2' (comma-separated pairs)
    - '11 22' (two digits together)
    - '1 1 2 2' (space-separated, pairs of numbers)
    """
    coords = []
    if not text.strip():
        return coords

    # Remove parentheses and replace commas with spaces
    text = text.replace("(", "").replace(")", "").replace(",", " ")
    parts = text.strip().split()

    i = 0
    while i < len(parts):
        part = parts[i]
        # Two digits together like "22"
        if len(part) == 2 and part.isdigit():
            coords.append((int(part[0]), int(part[1])))
            i += 1
        # Single digit - take next part as column
        elif len(part) == 1 and part.isdigit() and i + 1 < len(parts) and parts[i+1].isdigit():
            coords.append((int(part), int(parts[i+1])))
            i += 2
        else:
            i += 1
    return coords


def display_board(board, highlight=None):
    """Display board, optionally highlighting a move with [*]"""
    print("\n     " + "   ".join([str(i) for i in range(BOARD_SIZE)]))
    for row in range(BOARD_SIZE):
        row_str = f" {row}   "
        for col in range(BOARD_SIZE):
            is_highlight = (highlight == (row, col))
            if board[row, col] == 1:
                sym = "[○]" if is_highlight else " ○ "  # Black
            elif board[row, col] == -1:
                sym = "[●]" if is_highlight else " ● "  # White
            else:
                sym = "[*]" if is_highlight else " + "  # Empty / suggested move
            row_str += sym if col == BOARD_SIZE - 1 else sym.rstrip() + "──"
        print(row_str)
        if row < BOARD_SIZE - 1:
            print("     " + "│   " * (BOARD_SIZE - 1) + "│")
    print()


def main():
    model_path = load_model()
    if not model_path:
        print("No model found!")
        return

    print(f"Model: {os.path.basename(model_path)}\n")

    vpn = ValuePolicyNetwork(model_path)
    game = Go()
    mcts = MonteCarloTreeSearch(game, vpn.get_vp)

    while True:
        board = np.zeros((BOARD_SIZE, BOARD_SIZE))

        # Show empty board with coordinates
        print("Board coordinates:")
        display_board(board)

        print("Enter coordinates as: '1,1 2,2' or '1 1 2 2' or '11 22'")
        print("Leave empty for no pieces. Type 'q' to quit.\n")

        black_input = input("Black pieces (○): ").strip()
        if black_input.lower() == 'q':
            break
        for r, c in parse_coords(black_input):
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                board[r, c] = 1

        display_board(board)

        white_input = input("White pieces (●): ").strip()
        if white_input.lower() == 'q':
            break
        for r, c in parse_coords(white_input):
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                board[r, c] = -1

        print("\nFinal board:")
        display_board(board)

        # Current player
        p = input("Whose turn? (b/w) [b]: ").strip().lower()
        if p == 'q':
            break
        player = -1 if p == "w" else 1
        player_name = "Black (○)" if player == 1 else "White (●)"

        # Create state
        state = np.zeros(NUM_POSITIONS + 2)
        state[:NUM_POSITIONS] = board.flatten()
        state[NUM_POSITIONS] = -1
        state[NUM_POSITIONS + 1] = 0

        # Policy-Value Network
        value, policy = vpn.get_vp(state, player)
        valid_moves = game.get_valid_moves(state, player)
        policy = policy * valid_moves
        if policy.sum() > 0:
            policy = policy / policy.sum()

        print(f"\n{'='*50}")
        print(f"Player to move: {player_name}")
        print(f"Value: {value:.4f}")
        print(f"{'='*50}")

        print("\nPolicy (top 10):")
        moves = [(i, policy[i]) for i in range(ACTION_SIZE) if policy[i] > 0]
        moves.sort(key=lambda x: x[1], reverse=True)
        for i, (idx, prob) in enumerate(moves[:10]):
            if idx == PASS_ACTION:
                print(f"  {i+1}. pass: {prob*100:.1f}%")
            else:
                print(f"  {i+1}. ({idx//BOARD_SIZE}, {idx%BOARD_SIZE}): {prob*100:.1f}%")

        # MCTS
        sims = input(f"\nMCTS simulations [{cfg.NUM_SIMULATIONS}, 0=skip]: ").strip()
        if sims.lower() == 'q':
            break
        num_sims = int(sims) if sims else cfg.NUM_SIMULATIONS

        if num_sims > 0:
            print(f"\nRunning {num_sims} simulations...")
            # Convert state to canonical form for MCTS (current player's stones = +1)
            # MCTS internally maintains canonical form, so we need to convert from absolute
            canonical_state = state.copy()
            canonical_state[:NUM_POSITIONS] *= player  # Flip if White's turn
            root = Node(prior_prob=0, player=player, action_index=None)
            root.set_state(canonical_state)
            root = mcts.run_simulation(root, num_sims, player, add_noise=False)

            visits = [(idx, child.total_visits_N, child.mean_action_value_of_next_state_Q)
                      for idx, child in root.children.items()]
            visits.sort(key=lambda x: x[1], reverse=True)

            print("\nMCTS visits (top 10):")
            for i, (idx, v, q) in enumerate(visits[:10]):
                if idx == PASS_ACTION:
                    print(f"  {i+1}. pass: {v} visits, Q={q:.3f}")
                else:
                    print(f"  {i+1}. ({idx//BOARD_SIZE}, {idx%BOARD_SIZE}): {v} visits, Q={q:.3f}")


        print("\n" + "="*50 + "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
