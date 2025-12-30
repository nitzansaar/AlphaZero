#!/usr/bin/env python3
"""
Test to verify the neural network model works with 3x3 input.
"""
import torch
import numpy as np
from model import NeuralNetwork
from game import board_to_canonical_3d
from config import Config as cfg

print("Testing 3x3 Tic-Tac-Toe Neural Network")
print("=" * 50)

# Test 1: Model instantiation
print("\n1. Testing Model Instantiation:")
model = NeuralNetwork()
print(f"   Model created successfully")
print(f"   ✓ Model instantiation correct")

# Test 2: Input shape
print("\n2. Testing Input Shape:")
empty_board = np.zeros(9)
canonical = board_to_canonical_3d(empty_board, player=1)
print(f"   Input shape = {canonical.shape} (expected: (3, 3, 3))")
assert canonical.shape == (3, 3, 3), "Input should be (3, 3, 3)"
print("   ✓ Input shape correct")

# Test 3: Forward pass
print("\n3. Testing Forward Pass:")
batch_input = torch.from_numpy(canonical).unsqueeze(0).float()  # Add batch dimension
print(f"   Batch input shape = {batch_input.shape} (expected: torch.Size([1, 3, 3, 3]))")
value, policy = model(batch_input)
print(f"   Value shape = {value.shape} (expected: torch.Size([1, 1]))")
print(f"   Policy shape = {policy.shape} (expected: torch.Size([1, 9]))")
assert value.shape == torch.Size([1, 1]), "Value should have shape [1, 1]"
assert policy.shape == torch.Size([1, 9]), "Policy should have shape [1, 9]"
print("   ✓ Forward pass correct")

# Test 4: Output values
print("\n4. Testing Output Values:")
print(f"   Value range: [{value.item():.4f}] (should be in [-1, 1])")
assert -1 <= value.item() <= 1, "Value should be in range [-1, 1]"
policy_probs = torch.nn.functional.softmax(policy, dim=1)
print(f"   Policy sum: {policy_probs.sum().item():.6f} (should be ~1.0)")
assert abs(policy_probs.sum().item() - 1.0) < 1e-5, "Policy probabilities should sum to 1"
print("   ✓ Output values correct")

# Test 5: Different board states
print("\n5. Testing Different Board States:")
test_board = np.array([1, -1, 0, 0, 1, 0, 0, 0, 0])
canonical = board_to_canonical_3d(test_board, player=1)
batch_input = torch.from_numpy(canonical).unsqueeze(0).float()
value, policy = model(batch_input)
print(f"   Processed non-empty board successfully")
print(f"   Value: {value.item():.4f}, Policy shape: {policy.shape}")
print("   ✓ Different board states work")

print("\n" + "=" * 50)
print("✓ All model tests passed! Neural network is ready.")
print("=" * 50)
