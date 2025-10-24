#!/bin/bash
# Run all tests for the project

echo "Running GameEnv tests..."
python3 -m unittest tests/test_game_env.py -v

echo ""
echo "All tests completed!"

