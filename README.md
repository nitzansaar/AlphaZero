# AlphaZero-style N×N K-in-a-row Agent

A general N×N K-in-a-row game agent that learns to play optimally via self-play, guided by a policy/value neural network and Monte Carlo Tree Search (MCTS).

Currently support 5 by 5, 4 in a row tictactoe 

Never loses to random play

## Project Overview

This project implements a complete AlphaZero-inspired reinforcement learning system that:
- Plays generalized tic-tac-toe on any N×N board
- Learns optimal strategies through MCTS-guided self-play
- Uses dual-head neural network (policy + value)

## Train model
- cd src
- chmod +x ./train.sh
- ./train.sh

## Simulate play versus random player
- python3 test_vs_random.py


