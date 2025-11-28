# AlphaZero-style N×N K-in-a-row Agent

A general N×N K-in-a-row game agent that learns to play optimally via self-play, guided by a policy/value neural network and Monte Carlo Tree Search (MCTS).

Currently support 9 by 9, 5 in a row tictactoe (Gomoku)

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

## Play against the bot (Human vs Bot)
- cd src
- python3 play_human_vs_bot.py

## Simulate play versus random player
- cd src
- python3 test_vs_random.py

## Simulate play vs bot
- cd src
- python3 test_bot_vs_bot.py


