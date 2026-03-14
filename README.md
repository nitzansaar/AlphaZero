# AlphaZero Agent

AlphaZero-style reinforcement learning for board games. Learns through MCTS-guided self-play with a dual-head neural network (policy + value).

Supports:
- Connect Five (9x9 board)
- Go (5x5 and 9x9)

## Go

```bash
cd go

# Run selfplay training loop
BOARD_SIZE=9 bash train.sh

# Play against bot
BOARD_SIZE=9 python3 play_human_vs_bot.py

# Test bot vs bot
BOARD_SIZE=9 python3 test_bot_vs_bot.py --model1 10 --model2 37

# Test Katago vs AlphaZero
BOARD_SIZE=9 python3 katago_vs_alphazero.py --az-iter 142 --katago-iter 10 --games 100

# Analyze a board position (NN policy/value + MCTS distribution)
BOARD_SIZE=9 python analyze_board.py --model models_9x9/10_best_model.pt --sims 800
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


# To access neural network files
https://drive.google.com/drive/folders/1neejbIjkp-FvL2B9mxFdjVnwautYseVE?usp=drive_link
