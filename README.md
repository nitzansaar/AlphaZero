# AlphaZero Agent

AlphaZero-style reinforcement learning for board games. Learns through MCTS-guided self-play with a dual-head neural network (policy + value).

Supports:
- Connect Five (9x9 board)
- Go (9x9, and 19x19)

## Go

```bash
cd go

# Run selfplay training loop (defaults to 19x19)
bash train.sh

# Override board size
BOARD_SIZE=9 bash train.sh

# Play against bot
BOARD_SIZE=19 python3 play_human_vs_bot.py

# Test bot vs bot
BOARD_SIZE=19 python3 test_bot_vs_bot.py --model1 10 --model2 37

# Test AlphaZero vs KataGo (19x19, 100-game match)
bash run_az20_vs_katago_100.sh
# Override settings: AZ_ITERATION=146 KATAGO_ELO=482 KATAGO_VISITS=10 bash run_az20_vs_katago_100.sh

# Test KataGo vs AlphaZero (9x9)
BOARD_SIZE=9 python3 katago_vs_alphazero.py --az-iter 142 --katago-iter 10 --games 100

# Analyze a board position (NN policy/value + MCTS distribution)
BOARD_SIZE=19 python analyze_board.py --model models_19x19/10_best_model.pt --sims 800
```

### AZ20 vs KataGo notebook

`go/notebooks/az20_vs_pretrained_katago_19x19.ipynb` — interactive 19x19 match runner and move-by-move replay visualizer.

Set `AZ_ITERATION` and `KATAGO_ELO` in the config cell, then run all cells. The notebook will:
- resolve `models_19x19_az20/<iteration>/model_ts.pt` and `pretrained_katago_models/katago-elo-<elo>.gz`
- run a multi-game match via the GTP evaluator and write per-game results to `go/results/`
- play one additional in-process game and save a full move-by-move board image under `go/notebooks/figures/`

Key knobs (edit in the config cell):

| Variable | Default | Meaning |
|---|---|---|
| `AZ_ITERATION` | 29 | AlphaZero checkpoint to load |
| `KATAGO_ELO` | 1070 | Pretrained KataGo ELO to play against |
| `GAMES` | 2 | Number of match games |
| `AZ_SIMS` / `AZ_BATCH` | 160 / 32 | AlphaZero MCTS budget |
| `KATAGO_VISITS` | 60 | KataGo visit budget |
| `AZ_FIRST_COLOR` | `"alternate"` | Color assignment (`"alternate"`, `"black"`, `"white"`) |
| `MAKE_REPLAY_IMAGE` | `True` | Whether to render the move-by-move PNG |

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

