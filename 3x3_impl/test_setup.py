#!/usr/bin/env python3
"""
Quick test to verify 3x3 tic-tac-toe setup is working correctly.
"""
import numpy as np
from game import TicTacToe, board_to_canonical_3d
from config import Config as cfg

print("Testing 3x3 Tic-Tac-Toe Setup")
print("=" * 50)

# Test 1: Config
print("\n1. Testing Config:")
print(f"   ACTION_SIZE = {cfg.ACTION_SIZE} (expected: 9)")
assert cfg.ACTION_SIZE == 9, "ACTION_SIZE should be 9 for 3x3 board"
print("   ✓ Config correct")

# Test 2: Game initialization
print("\n2. Testing Game Initialization:")
game = TicTacToe()
print(f"   Board size = {len(game.state)} (expected: 9)")
assert len(game.state) == 9, "Board should have 9 positions"
print("   ✓ Game initialization correct")

# Test 3: Valid moves
print("\n3. Testing Valid Moves:")
valid_moves = game.get_valid_moves(game.state)
print(f"   Valid moves count = {np.sum(valid_moves)} (expected: 9)")
assert np.sum(valid_moves) == 9, "All positions should be valid initially"
print("   ✓ Valid moves correct")

# Test 4: Canonical board representation
print("\n4. Testing Canonical Board Representation:")
canonical = board_to_canonical_3d(game.state, player=1)
print(f"   Canonical shape = {canonical.shape} (expected: (3, 3, 3))")
assert canonical.shape == (3, 3, 3), "Canonical board should be (3, 3, 3)"
print("   ✓ Canonical representation correct")

# Test 5: Win condition - horizontal
print("\n5. Testing Win Condition (Horizontal):")
test_state = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
result = game.win_or_draw(test_state)
print(f"   Horizontal win result = {result} (expected: 1)")
assert result == 1, "Should detect horizontal win"
print("   ✓ Horizontal win detection correct")

# Test 6: Win condition - vertical
print("\n6. Testing Win Condition (Vertical):")
test_state = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
result = game.win_or_draw(test_state)
print(f"   Vertical win result = {result} (expected: 1)")
assert result == 1, "Should detect vertical win"
print("   ✓ Vertical win detection correct")

# Test 7: Win condition - diagonal
print("\n7. Testing Win Condition (Diagonal):")
test_state = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
result = game.win_or_draw(test_state)
print(f"   Diagonal win result = {result} (expected: 1)")
assert result == 1, "Should detect diagonal win"
print("   ✓ Diagonal win detection correct")

# Test 8: Draw condition
print("\n8. Testing Draw Condition:")
test_state = np.array([1, -1, 1, -1, 1, -1, -1, 1, -1])
result = game.win_or_draw(test_state)
print(f"   Draw result = {result} (expected: 0)")
assert result == 0, "Should detect draw"
print("   ✓ Draw detection correct")

# Test 9: Game in progress
print("\n9. Testing Game in Progress:")
test_state = np.array([1, -1, 0, 0, 1, 0, 0, 0, 0])
result = game.win_or_draw(test_state)
print(f"   In progress result = {result} (expected: None)")
assert result is None, "Should return None for game in progress"
print("   ✓ Game in progress detection correct")

print("\n" + "=" * 50)
print("✓ All tests passed! 3x3 tic-tac-toe setup is correct.")
print("=" * 50)
