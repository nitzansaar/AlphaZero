# AlphaZero-style Connect Five Agent

A reinforcement learning agent that learns to play board games optimally without human knowledge via self-play, guided by a policy/value neural network and Monte Carlo Tree Search (MCTS).

Currently supports:
- 9 by 9, 5 in a row Connect Five 
- 5 by 5 Go

## Project Overview

This project implements a complete AlphaZero-inspired reinforcement learning system that:
- Learns optimal strategies through MCTS-guided self-play
- Uses dual-head neural network (policy + value)

## Train model
- chmod +x ./train.sh
- ./train.sh

## Play against the bot (Human vs Bot)
- python3 tictactoe/play_human_vs_bot_flask.py 
- python3 go/play_human_vs_bot.py

## Simulate play versus random player
- python3 test_vs_random.py

## Simulate play vs bot
- python3 test_bot_vs_bot.py

