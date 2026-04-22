"""
Configuration for AlphaZero Go implementation.
Supports multiple board sizes (5x5, 9x9, 19x19).

Usage:
    from config import Config as cfg, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE

    # Access settings
    print(cfg.BOARD_SIZE)  # 5, 9, or 19
    print(cfg.NUM_SIMULATIONS)  # Scaled for board size

To change board size, set environment variable before importing:
    BOARD_SIZE=9 python train.py

Or modify the defaults below.
"""

import os

# Default board size - change this or use BOARD_SIZE env var
DEFAULT_BOARD_SIZE = int(os.environ.get('BOARD_SIZE', '9'))


class Config5x5:
    """Configuration for 5x5 Go (proof of concept)"""

    # Board settings
    BOARD_SIZE = 5
    NUM_POSITIONS = 25
    PASS_ACTION = 25
    ACTION_SIZE = 26  # 25 positions + 1 pass

    # Komi (compensation for white)
    KOMI = 2.5

    # Training settings
    BATCH_SIZE = 512
    TRAIN_STEPS = 15
    SELFPLAY_GAMES = 500
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_DECAY_ITERS = [100, 150]
    LR_DECAY_FACTOR = 0.1

    # MCTS settings
    NUM_SIMULATIONS = 1600
    MCTS_UCB_C = 1.414  # sqrt(2)

    # Playout cap randomization
    PLAYOUT_CAP_PROB = 0.25
    FAST_SIMS        = 100

    # Network architecture
    NUM_RES_BLOCKS = 6
    NUM_CHANNELS = 128
    VALUE_HEAD_HIDDEN = 256

    # Dataset
    DATASET_QUEUE_SIZE = 100000

    # Loss weights
    VALUE_LOSS_WEIGHT = 1.0
    POLICY_LOSS_WEIGHT = 1.0

    # Temperature for exploration
    TEMP_THRESHOLD = 6
    INITIAL_TEMP = 1.0

    # Data augmentation
    USE_AUGMENTATION = True

    # Model gating: new model must win >= GATE_WIN_RATE of GATE_GAMES games
    # against the previous best before replacing it for selfplay data generation.
    GATE_WIN_RATE = 0.55
    GATE_GAMES = 20
    GATE_SIMULATIONS = 200
    GATE_TEMPERATURE_MOVES = 4  # opening moves sampled proportionally; rest greedy

    # Paths (board-size specific)
    SAVE_MODEL_PATH = "models_5x5"
    SAVE_PICKLES = "pickles_5x5"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "logs_5x5"
    TEST_OUTPUT_PATH = "test_output_5x5"

    # Evaluation
    EVAL_GAMES = 40
    NUM_GAMES = 100


class Config9x9Base:
    """Base configuration for 9x9 Go (original scaled up settings)"""

    # Board settings
    BOARD_SIZE = 9
    NUM_POSITIONS = 81
    PASS_ACTION = 81
    ACTION_SIZE = 82  # 81 positions + 1 pass

    # Komi 
    KOMI = 6

    # Training settings - increased for larger board
    BATCH_SIZE = 512
    TRAIN_STEPS = 1500  
    SELFPLAY_GAMES = 500  # decreased to 500 to match katago
    LEARNING_RATE = 0.001  # SGD LR
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_DECAY_ITERS = [30, 60, 200, 400, 700]
    LR_DECAY_FACTOR = 0.1

    # MCTS settings
    NUM_SIMULATIONS = 800
    MCTS_UCB_C = 1.414

    # Playout cap randomization
    # A fraction PLAYOUT_CAP_PROB of moves use the full NUM_SIMULATIONS budget
    # and generate training data; the rest use FAST_SIMS to advance the game
    # cheaply without contributing training examples.
    PLAYOUT_CAP_PROB = 0.25
    FAST_SIMS        = 100

    # Network architecture - larger for 9x9
    NUM_RES_BLOCKS = 10
    NUM_CHANNELS = 256
    VALUE_HEAD_HIDDEN = 512

    # Dataset - larger buffer
    DATASET_QUEUE_SIZE = 500000

    # Loss weights — value loss is scaled up to match the policy cross-entropy
    # magnitude (~0.5 raw vs ~5.0 raw), so both heads contribute equally to
    # the shared backbone gradient.  POLICY_LR_MULTIPLIER is kept at 1.0
    # because the loss-weight balance already achieves the desired signal ratio.
    VALUE_LOSS_WEIGHT = 5.0
    POLICY_LOSS_WEIGHT = 1.0
    POLICY_LR_MULTIPLIER = 1.0

    # Temperature for exploration
    TEMP_THRESHOLD = 15  # Opening moves with temp=1; rest greedy (~17% of ~90-move game)
    INITIAL_TEMP = 1.0

    # Data augmentation
    USE_AUGMENTATION = True

    # Model gating: new model must win >= GATE_WIN_RATE of GATE_GAMES games
    # against the previous best before replacing it for selfplay data generation.
    GATE_WIN_RATE = 0.55
    GATE_GAMES = 20
    GATE_SIMULATIONS = 200
    GATE_TEMPERATURE_MOVES = 4  # opening moves sampled proportionally; rest greedy

    # Paths
    SAVE_MODEL_PATH = "models_9x9_base"
    SAVE_PICKLES = "pickles_9x9_base"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "logs_9x9_base"
    TEST_OUTPUT_PATH = "test_output_9x9_base"

    # Evaluation
    EVAL_GAMES = 40
    NUM_GAMES = 100


class Config19x19Base:
    """Base configuration for 19x19 Go (AlphaZero-style)."""

    # Board settings
    BOARD_SIZE = 19
    NUM_POSITIONS = 361
    PASS_ACTION = 361
    ACTION_SIZE = 362      # 361 positions + 1 pass

    # Komi — standard 19x19 tournament komi
    KOMI = 7.5

    # Training — more steps and larger buffer for bigger action space
    BATCH_SIZE = 128
    TRAIN_STEPS = 1000
    SELFPLAY_GAMES = 500
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_DECAY_ITERS = [30, 60, 200, 400, 700]
    LR_DECAY_FACTOR = 0.1

    NUM_SIMULATIONS = 600
    MCTS_UCB_C = 1.414
    PLAYOUT_CAP_PROB = 0.25
    FAST_SIMS = 100

    # Network — deeper for 19x19
    NUM_RES_BLOCKS = 15
    NUM_CHANNELS = 256
    VALUE_HEAD_HIDDEN = 512

    DATASET_QUEUE_SIZE = 500_000

    # Loss weights — value loss is scaled up to match the policy cross-entropy
    # magnitude (~0.5 raw vs ~5.0 raw), so both heads contribute equally to
    # the shared backbone gradient.  POLICY_LR_MULTIPLIER is kept at 1.0
    # because the loss-weight balance already achieves the desired signal ratio;
    # a separate per-head LR multiplier would overcorrect.
    VALUE_LOSS_WEIGHT = 5.0
    POLICY_LOSS_WEIGHT = 1.0
    POLICY_LR_MULTIPLIER = 1.0

    # Temperature — first 30 moves exploratory (~10% of a 300-move game)
    TEMP_THRESHOLD = 30
    INITIAL_TEMP = 1.0

    # Data augmentation
    USE_AUGMENTATION = True

    # Gating
    GATE_WIN_RATE = 0.55
    GATE_GAMES = 20
    GATE_SIMULATIONS = 200
    GATE_TEMPERATURE_MOVES = 4

    # None → use all CPU cores (os.cpu_count()).
    NUM_SELFPLAY_WORKERS = None

    # Max moves per selfplay game; 19x19 games average 250-350 moves
    MAX_MOVES = 500

    # Paths
    SAVE_MODEL_PATH = "models_19x19_base"
    SAVE_PICKLES = "pickles_19x19_base"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "logs_19x19_base"
    TEST_OUTPUT_PATH = "test_output_19x19_base"

    # Evaluation
    EVAL_GAMES = 40
    NUM_GAMES = 100


# Select configuration based on board size
_configs = {
    5: Config5x5,
    9: Config9x9Base,
    19: Config19x19Base,
}

if DEFAULT_BOARD_SIZE not in _configs:
    raise ValueError(f"Unsupported board size: {DEFAULT_BOARD_SIZE}. Supported: {list(_configs.keys())}")

Config = _configs[DEFAULT_BOARD_SIZE]

# Provide defaults for attributes introduced by this change so older configs
# (5x5, 9x9) don't need to be updated to define them.
if not hasattr(Config, 'NUM_SELFPLAY_WORKERS'):
    Config.NUM_SELFPLAY_WORKERS = None   # None → use os.cpu_count()
if not hasattr(Config, 'MAX_MOVES'):
    Config.MAX_MOVES = 200               # existing 9x9 default
if not hasattr(Config, 'POLICY_LR_MULTIPLIER'):
    Config.POLICY_LR_MULTIPLIER = 1.0   # 5x5 has no per-head LR boost

# Export commonly used constants at module level for convenience
BOARD_SIZE = Config.BOARD_SIZE
NUM_POSITIONS = Config.NUM_POSITIONS
PASS_ACTION = Config.PASS_ACTION
ACTION_SIZE = Config.ACTION_SIZE
KOMI = Config.KOMI


def get_config(board_size=None):
    """Get configuration for a specific board size."""
    if board_size is None:
        board_size = DEFAULT_BOARD_SIZE
    if board_size not in _configs:
        raise ValueError(f"Unsupported board size: {board_size}. Supported: {list(_configs.keys())}")
    return _configs[board_size]


def print_config():
    """Print current configuration."""
    print(f"{'='*50}")
    print(f"AlphaZero Go Configuration")
    print(f"{'='*50}")
    print(f"Board Size: {Config.BOARD_SIZE}x{Config.BOARD_SIZE}")
    print(f"Action Space: {Config.ACTION_SIZE}")
    print(f"Komi: {Config.KOMI}")
    print(f"")
    print(f"Network:")
    print(f"  Residual Blocks: {Config.NUM_RES_BLOCKS}")
    print(f"  Channels: {Config.NUM_CHANNELS}")
    print(f"  Value Head Hidden: {Config.VALUE_HEAD_HIDDEN}")
    print(f"")
    print(f"Training:")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Train Steps: {Config.TRAIN_STEPS}")
    print(f"  Self-play Games: {Config.SELFPLAY_GAMES}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"")
    print(f"MCTS:")
    print(f"  Simulations: {Config.NUM_SIMULATIONS}")
    print(f"  UCB C: {Config.MCTS_UCB_C}")
    print(f"")
    print(f"Paths:")
    print(f"  Models: {Config.SAVE_MODEL_PATH}")
    print(f"  Data: {Config.SAVE_PICKLES}")
    print(f"  Logs: {Config.LOGDIR}")
    print(f"  Test Output: {Config.TEST_OUTPUT_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    print_config()