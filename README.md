# AlphaZero Agent

AlphaZero-style reinforcement learning for board games. Learns through MCTS-guided self-play with a dual-head neural network (policy + value).

Supports:
- Connect Five (9x9 board)
- Go (5x5 and 9x9)

## Go

```bash
cd go

# Training
./run_5x5.sh train [iterations]
./run_9x9.sh train [iterations]

# Testing
./run_5x5.sh test-random [games]
./run_5x5.sh test-minimax [games] [depth]

# Play against bot
python play_human_vs_bot.py
```

## Connect Five (TicTacToe)

```bash
cd tictactoe

# Training
./train.sh

# Play against bot
python play_human_vs_bot_flask.py

# Testing
python test_vs_random.py
python test_bot_vs_bot.py
```
