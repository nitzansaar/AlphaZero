# AlphaZero-style N×N K-in-a-row Agent

A general N×N K-in-a-row game agent that learns to play optimally via self-play, guided by a policy/value neural network and Monte Carlo Tree Search (MCTS).

## Project Overview

This project implements a complete AlphaZero-inspired reinforcement learning system that:
- Plays generalized tic-tac-toe on any N×N board (3 by 3 for now)
- Learns optimal strategies through MCTS-guided self-play
- Uses dual-head neural network (policy + value)
- Supports training on GPU for 10x speedup
- Includes visualization and debugging tools

## Project Structure

```
tictactoeMCTS/
├── src/
│   ├── env/                      # Game environment
│   ├── trainer/
│   │   ├── train_self_play.py   # AlphaZero-style MCTS + NN training ✅
│   │   └── train_neural_net.py  # Supervised learning baseline
│   └── eval/
│       └── agents.py             # MCTS-based gameplay agents ✅
├── play_cli.py                   # Command-line game interface
├── visualize_training.py         # Training progress visualization ✅
├── test_model.py                 # Model debugging tool ✅
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

### Quick Start

**Train the AlphaZero model:**
```bash
# Train on CPU (1-2 hours for 10k episodes)
python3 -m src.trainer.train_self_play

# Or train on GPU (5-10 minutes for 10k episodes)
# See CLOUD_TRAINING.md for Google Cloud GPU setup
```

**Play against the AI:**
```bash
# Play 3×3 tic-tac-toe against trained MCTS agent
python play_cli.py

# Human vs Human
python play_cli.py -p1 human -p2 human

# Watch AI vs AI
python play_cli.py -p1 nn -p2 nn
```

### Run Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=src --cov-report=html
```

## Project Milestones

### ✅ Milestone 1: Environment + Infrastructure (Completed 10/7/2025)
- Generalized N×N K-in-a-row game environment
- Board representation and rendering
- Win/draw detection in all directions
- Legal move validation
- Neural network encoding (3-plane one-hot)
- Comprehensive unit tests (21 tests passing)
- CLI for human vs random gameplay

### ✅ Milestone 2: Neural Network Architecture & Baseline Training (Completed 10/14/2025)
- Dual-head neural network (policy + value)
- 3-plane convolutional architecture
- Supervised learning on exhaustive game states
- Minimax-optimal move labeling
- Training achieves 95%+ accuracy on validation set
- Model: `tictactoe_model_best.pth`

### ✅ Milestone 3: MCTS Integration (Completed 10/21/2025)
- Full MCTS implementation with UCB selection
- Neural network prior initialization
- Value-based leaf evaluation
- 500 simulations per move during gameplay
- **Critical bug fix**: UCB exploration formula
- Debug mode for visit count visualization
- Consistently finds tactical moves (blocks, wins)

### 🔄 Milestone 4: Self-Play Pipeline (In Progress - Due 10/28/2025)
**Status**: AlphaZero training implemented, needs retraining with bug fixes

**Completed:**
- ✅ Self-play game engine with MCTS
- ✅ Experience replay buffer (10k positions)
- ✅ AlphaZero-style loss (policy + value)
- ✅ Learning rate scheduling for stability
- ✅ Training visualization (4-panel plots)
- ✅ GPU optimization (10x speedup)
- ✅ Checkpoint saving every 500 episodes
- 
**Remaining:**
- 🔄 Retrain with fixed MCTS exploration bug
- 🔄 Validate model achieves >90% draw rate
- 🔄 Verify tactical move accuracy

**Expected Completion**: This week (once retraining completes)

### 📋 Milestone 5: Debugging, Stability & Improvement (Not Started - Due 11/4/2025)
- Tune hyperparameters (LR, batch size, simulations)
- Compare self-play vs supervised learning
- Measure training stability metrics
- Profile and optimize training speed
- Add early stopping based on performance

### 📋 Milestone 6: Scaling to Larger Boards & Evaluation (Not Started - Due 11/11/2025)
- Generalize to 5×5 K=4 boards (should eventually work N by N K in a row)
- Collect quantitative results
- Compare against baselines
- Measure strategic depth

### 📋 Milestone 7: Demo, Documentation & Final Report (Not Started - Due 11/18/2025)
- Final project demonstration
- Complete documentation
- Video demo of gameplay
- Performance analysis report

## Key Features

### AlphaZero Training System
- **MCTS-guided self-play**: Network plays against itself using tree search
- **Dual-head architecture**: Policy (move probabilities) + Value (position evaluation)
- **Experience replay**: 10k position buffer for stable learning
- **Learning rate scheduling**: Prevents late-stage training instability
- **GPU acceleration**: supports GPU training if one exists on your machine

### MCTS Implementation
- **UCB-based selection**: Balances exploration and exploitation
- **Neural network priors**: Uses policy head to guide search
- **Value-based evaluation**: Uses value head for leaf positions
- **Configurable simulations**: 50 during training, 500 during play
- **Debug visualization**: Shows visit counts and average values

### Gameplay
- **MCTS-powered moves**: 500 simulations per move
- **Tactical play**: Finds blocks, forks, and winning sequences
- **Debug mode**: See exactly what the AI is thinking
- **Probability display**: Shows move evaluation percentages

## Performance

**Training Speed:**
- CPU (M1/M2): 1-2 hours for 10k episodes
- GPU (T4): 5-10 minutes for 10k episodes

**Expected Results** (after proper training):
- Draw rate: >90% (optimal tic-tac-toe)
- Policy loss: <0.20
- Value loss: <0.05
- Tactical accuracy: Near-perfect

## Current Status

✅ **Completed**: Full AlphaZero implementation with MCTS + NN
🔄 **In Progress**: Retraining with bug fixes
📋 **Next**: Hyperparameter tuning and scaling to 5×5 boards

