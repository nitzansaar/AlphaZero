# AlphaZero-style N×N K-in-a-row Agent

A general N×N K-in-a-row game agent that learns to play optimally via self-play, guided by a policy/value neural network and Monte Carlo Tree Search (MCTS).

Currently only support 3 by 3 tictactoe

Never loses to random play

## Project Overview

This project implements a complete AlphaZero-inspired reinforcement learning system that:
- Plays generalized tic-tac-toe on any N×N board (3 by 3 for now)
- Learns optimal strategies through MCTS-guided self-play
- Uses dual-head neural network (policy + value)

## Train model
cd src
chmod +x ./train.sh
./train.sh

## Simulate play versus random player
python3 test_vs_random.py


