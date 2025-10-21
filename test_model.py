#!/usr/bin/env python3
"""
Test script to verify the trained model is working correctly.
"""

import torch
import numpy as np
from src.trainer.train_self_play import TicTacToeNet, MCTS, board_to_canonical_3d

def test_model():
    """Test the trained model with a specific board position."""

    # Load model
    device = 'cpu'
    model = TicTacToeNet()

    try:
        model.load_state_dict(torch.load('tictactoe_selfplay_final.pth', map_location=device))
        model.eval()
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Test case: Obvious winning move for X
    # Board:
    # X O O
    # X X .   <- X should play (1,2) to win
    # . . .

    board_flat = [
        1, -1, -1,  # Row 0: X O O
        1,  1,  0,  # Row 1: X X .
        0,  0,  0   # Row 2: . . .
    ]
    current_player = 1  # X to move

    print("Testing board position:")
    print("  0 1 2")
    print("  -----")
    for i in range(3):
        row = []
        for j in range(3):
            val = board_flat[i*3 + j]
            symbol = 'X' if val == 1 else ('O' if val == -1 else '.')
            row.append(symbol)
        print(f"{i}|{' '.join(row)}")
    print("\nX to move. Obvious winning move: (1,2)\n")

    # Convert to canonical 3-plane format
    canonical_board = board_to_canonical_3d(board_flat, current_player)
    board_tensor = torch.from_numpy(canonical_board).unsqueeze(0).to(device)

    # Get NN predictions
    print("="*50)
    print("NEURAL NETWORK OUTPUT")
    print("="*50)
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
        policy_logits = policy_logits.squeeze(0)
        value = value.item()

    # Show raw logits
    print("\nRaw policy logits:")
    for i in range(9):
        r, c = i // 3, i % 3
        print(f"  ({r},{c}): {policy_logits[i].item():7.3f}")

    # Show softmax probabilities
    policy_probs = torch.softmax(policy_logits, dim=0)
    print("\nPolicy probabilities (softmax):")
    legal_moves = [(1,0), (1,2), (2,0), (2,1), (2,2)]
    for i in range(9):
        r, c = i // 3, i % 3
        prob = policy_probs[i].item()
        legal = "✓" if (r,c) in legal_moves else "✗"
        print(f"  ({r},{c}): {prob:6.1%} {legal}")

    print(f"\nValue prediction: {value:.3f}")
    print(f"  (Should be close to +1 since X is winning)")

    # Run MCTS
    print("\n" + "="*50)
    print("MCTS SEARCH (500 simulations)")
    print("="*50)

    mcts = MCTS(model, device, num_simulations=500, debug=True)
    action_probs, _ = mcts.get_action_probs(board_flat, current_player, temperature=1.0)

    print("\nFinal MCTS action probabilities:")
    for i in range(9):
        r, c = i // 3, i % 3
        prob = action_probs[i]
        if prob > 0.001:  # Only show moves with >0.1% probability
            legal = "✓" if (r,c) in legal_moves else "✗"
            star = " ← WINNER!" if (r,c) == (1,2) else ""
            print(f"  ({r},{c}): {prob:6.1%} {legal}{star}")

    best_move_idx = max(range(9), key=lambda i: action_probs[i])
    best_move = (best_move_idx // 3, best_move_idx % 3)

    print(f"\nMCTS selected move: {best_move}")
    if best_move == (1, 2):
        print("✓ CORRECT! MCTS found the winning move.")
    else:
        print(f"✗ WRONG! MCTS should have chosen (1,2), but chose {best_move}")
        print("   This indicates the model needs more training.")

if __name__ == "__main__":
    test_model()
