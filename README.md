# AlphaZero-style N×N K-in-a-row Agent

A general N×N K-in-a-row game agent that learns to play optimally via self-play, guided by a policy/value neural network and Monte Carlo Tree Search (MCTS).

## Project Overview

This project implements an AlphaZero-inspired reinforcement learning agent that can:
- Play generalized tic-tac-toe on any N×N board
- Learn optimal strategies through self-play
- Scale from 3×3 standard tic-tac-toe to larger boards like 5×5 with K=4

## Project Structure

```
tictactoeMCTS/
├── src/
│   ├── env/          # Game environment
│   ├── mcts/         # Monte Carlo Tree Search
│   ├── net/          # Neural network architecture
│   ├── trainer/      # Training pipeline
│   └── eval/         # Evaluation and agents
├── tests/            # Unit tests
├── play_cli.py      # Command-line game interface
└── requirements.txt # Python dependencies
```

## Getting Started

### Installation

1. Clone the repository:
```bash
cd tictactoeMCTS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Play the Game

```bash
# Play 3×3 tic-tac-toe against random agent
python play_cli.py

# Play 5×5 with 4-in-a-row against another human
python play_cli.py -n 5 -k 4 -p1 human -p2 human

# Watch random agents play (for testing)
python play_cli.py -p1 random -p2 random
```

### Run Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=src --cov-report=html
```

##  Week 1 Milestone (Completed)

- Implemented generalized `GameEnv` class for N×N K-in-a-row
- Board representation using NumPy arrays
- Win detection (horizontal, vertical, diagonal, anti-diagonal)
- Draw detection and edge cases
- Board encoding for neural network input (one-hot planes)
- Comprehensive unit tests (21 tests, all passing)
- CLI for human vs random gameplay
- Project structure and dependencies setup

## Game Environment Features

The `GameEnv` class supports:

- **Configurable board size**: Any N×N board
- **Flexible win condition**: K pieces in a row (K ≤ N)
- **Player representation**: 1 (X), -1 (O), 0 (empty)
- **Win detection**: All directions (horizontal, vertical, both diagonals)
- **Neural network encoding**: 3-plane one-hot encoding
- **Canonical board**: Current player perspective for training
- **Game state cloning**: For MCTS simulations

### Example Usage

```python
from src.env.game_env import GameEnv

# Create 3×3 tic-tac-toe
env = GameEnv(n=3, k=3)

# Make moves
board, reward, done = env.apply_move(0, 0)  # X at (0,0)
board, reward, done = env.apply_move(1, 1)  # O at (1,1)

# Get board encoding for neural network
encoding = env.get_board_encoding()  # Shape: (3, N, N)

# Check legal moves
legal_moves = env.get_legal_moves()  # List of (row, col) tuples

# Display board
print(env.render())
```

## Testing

The test suite covers:
- Basic game mechanics (initialization, reset, move validation)
- Win detection in all directions
- Draw conditions
- Edge cases (different board sizes, K values)
- Board encoding correctness
- Game state cloning


## Next Steps (Upcoming Weeks)

- **Week 2**: Neural network architecture & supervised learning
- **Week 3**: MCTS implementation
- **Week 4**: Self-play pipeline
- **Week 5**: Training stability & tuning
- **Week 6**: Scaling to larger boards
- **Week 7**: Demo & documentation

