"""
Configuration for AlphaZero Go implementation.
Supports multiple board sizes (5x5, 9x9, etc.)

Usage:
    from config import Config as cfg, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE

    # Access settings
    print(cfg.BOARD_SIZE)  # 5 or 9
    print(cfg.NUM_SIMULATIONS)  # Scaled for board size

To change board size, set environment variable before importing:
    BOARD_SIZE=9 python train.py

Or modify the DEFAULT_BOARD_SIZE below.
"""

import os

# Default board size - change this or use BOARD_SIZE env var
DEFAULT_BOARD_SIZE = int(os.environ.get('BOARD_SIZE', '5'))


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
    EPOCHS = 50
    SELFPLAY_GAMES = 500
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9

    # MCTS settings
    NUM_SIMULATIONS = 1600
    MCTS_UCB_C = 1.414  # sqrt(2)

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


class Config9x9:
    """Configuration for 9x9 Go (scaled up)"""

    # Board settings
    BOARD_SIZE = 9
    NUM_POSITIONS = 81
    PASS_ACTION = 81
    ACTION_SIZE = 82  # 81 positions + 1 pass

    # Komi (standard for 9x9)
    KOMI = 5.5

    # Training settings - increased for larger board
    BATCH_SIZE = 256  # Smaller batches, more updates
    EPOCHS = 100  # More epochs for convergence
    SELFPLAY_GAMES = 1000  # More games for diversity
    LEARNING_RATE = 0.001  # Start higher, will decay
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9

    # MCTS settings
    NUM_SIMULATIONS = 800
    MCTS_UCB_C = 1.414

    # Network architecture - larger for 9x9
    NUM_RES_BLOCKS = 10
    NUM_CHANNELS = 256
    VALUE_HEAD_HIDDEN = 512

    # Dataset - larger buffer
    DATASET_QUEUE_SIZE = 500000

    # Loss weights — equal weighting; with Adam the ratio is irrelevant because
    # Adam normalises out gradient scale (scale-invariant update rule).
    # To actually bias Adam toward the policy head, we use a higher learning rate
    # for policy head parameters in the optimizer (see train.py).
    VALUE_LOSS_WEIGHT = 1.0
    POLICY_LOSS_WEIGHT = 1.0

    # Learning-rate multiplier for the policy head relative to backbone/value.
    # Adam is scale-invariant w.r.t. loss weights, but not w.r.t. per-param-group LRs.
    POLICY_LR_MULTIPLIER = 3.0

    # Temperature for exploration
    TEMP_THRESHOLD = 15  # More moves before deterministic
    INITIAL_TEMP = 1.0

    # Data augmentation
    USE_AUGMENTATION = True

    # Paths (board-size specific)
    SAVE_MODEL_PATH = "models_9x9"
    SAVE_PICKLES = "pickles_9x9"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "logs_9x9"
    TEST_OUTPUT_PATH = "test_output_9x9"

    # Evaluation
    EVAL_GAMES = 40
    NUM_GAMES = 100


# Select configuration based on board size
_configs = {
    5: Config5x5,
    9: Config9x9,
}

if DEFAULT_BOARD_SIZE not in _configs:
    raise ValueError(f"Unsupported board size: {DEFAULT_BOARD_SIZE}. Supported: {list(_configs.keys())}")

Config = _configs[DEFAULT_BOARD_SIZE]

# Export commonly used constants at module level for convenience
BOARD_SIZE = Config.BOARD_SIZE
NUM_POSITIONS = Config.NUM_POSITIONS
PASS_ACTION = Config.PASS_ACTION
ACTION_SIZE = Config.ACTION_SIZE
KOMI = Config.KOMI


def get_config(board_size=None):
    """Get configuration for a specific board size."""
    if board_size is None:
        return Config
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
    print(f"  Epochs: {Config.EPOCHS}")
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
