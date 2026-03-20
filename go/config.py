"""
Configuration for AlphaZero Go implementation.
Supports multiple board sizes (5x5, 9x9, etc.) and variants for 9x9 (e.g., base, large, fast).

Usage:
    from config import Config as cfg, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE

    # Access settings
    print(cfg.BOARD_SIZE)  # 5 or 9
    print(cfg.NUM_SIMULATIONS)  # Scaled for board size

To change board size, set environment variable before importing:
    BOARD_SIZE=9 python train.py

To change 9x9 variant, set CONFIG_VARIANT (defaults to 'base'):
    BOARD_SIZE=9 CONFIG_VARIANT=large python train.py

Or modify the defaults below.
"""

import os

# Default board size - change this or use BOARD_SIZE env var
DEFAULT_BOARD_SIZE = int(os.environ.get('BOARD_SIZE', '9'))

# Default 9x9 config variant - change this or use CONFIG_VARIANT env var (only for board_size=9)
DEFAULT_CONFIG_VARIANT = os.environ.get('CONFIG_VARIANT', 'base')


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

    # Minimum move number before pass is allowed in C++ selfplay.
    MIN_PASS_MOVE = 15

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
    LR_DECAY_ITERS = [30, 60]
    LR_DECAY_FACTOR = 0.1

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
    TEMP_THRESHOLD = 15  # Opening moves with temp=1; rest greedy (~17% of ~90-move game)
    INITIAL_TEMP = 1.0

    # Minimum move number before pass is allowed in C++ selfplay.
    # Prevents komi exploitation (model learning to pass immediately as White).
    MIN_PASS_MOVE = 30

    # Data augmentation
    USE_AUGMENTATION = True

    # Model gating: new model must win >= GATE_WIN_RATE of GATE_GAMES games
    # against the previous best before replacing it for selfplay data generation.
    GATE_WIN_RATE = 0.55
    GATE_GAMES = 20
    GATE_SIMULATIONS = 200
    GATE_TEMPERATURE_MOVES = 4  # opening moves sampled proportionally; rest greedy

    # Paths (board-size and variant specific) - overridden in selection logic
    SAVE_MODEL_PATH = "models_9x9_base"
    SAVE_PICKLES = "pickles_9x9_base"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "logs_9x9_base"
    TEST_OUTPUT_PATH = "test_output_9x9_base"

    # Evaluation
    EVAL_GAMES = 40
    NUM_GAMES = 100


class Config9x9Large(Config9x9Base):
    """Variant for 9x9 with larger network (deeper/wider for potentially stronger play, but slower training)"""
    NUM_RES_BLOCKS = 15  # More blocks
    NUM_CHANNELS = 384  # Wider channels
    VALUE_HEAD_HIDDEN = 768  # Larger value head
    NUM_SIMULATIONS = 1200  # More MCTS sims for better self-play quality
    DATASET_QUEUE_SIZE = 1000000  # Larger buffer
    TRAIN_STEPS = 2000  # More training steps per iteration


class Config9x9Fast(Config9x9Base):
    """Variant for 9x9 with faster settings (smaller net, fewer sims/steps for quicker experiments)"""
    NUM_RES_BLOCKS = 8  # Fewer blocks
    NUM_CHANNELS = 192  # Narrower channels
    VALUE_HEAD_HIDDEN = 384  # Smaller value head
    NUM_SIMULATIONS = 400  # Fewer MCTS sims
    TRAIN_STEPS = 1000  # Fewer training steps
    SELFPLAY_GAMES = 300  # Fewer self-play games per iteration
    LEARNING_RATE = 0.002  # Higher LR for faster convergence (riskier)
    

# make a variant that is able to change the NN structure through training


# Select configuration based on board size and variant
_configs = {
    5: Config5x5,
    9: {
        'base': Config9x9Base,
        'large': Config9x9Large,
        'fast': Config9x9Fast,
        # more variants to come
    },
}

if DEFAULT_BOARD_SIZE not in _configs:
    raise ValueError(f"Unsupported board size: {DEFAULT_BOARD_SIZE}. Supported: {list(_configs.keys())}")

if DEFAULT_BOARD_SIZE == 5:
    if DEFAULT_CONFIG_VARIANT != 'base':
        print("Warning: CONFIG_VARIANT ignored for board_size=5; using default.")
    Config = _configs[5]
else:
    if DEFAULT_CONFIG_VARIANT not in _configs[DEFAULT_BOARD_SIZE]:
        raise ValueError(f"Unsupported config variant for {DEFAULT_BOARD_SIZE}x{DEFAULT_BOARD_SIZE}: {DEFAULT_CONFIG_VARIANT}. "
                         f"Supported: {list(_configs[DEFAULT_BOARD_SIZE].keys())}")
    Config = _configs[DEFAULT_BOARD_SIZE][DEFAULT_CONFIG_VARIANT]

    # Append variant to paths to isolate experiments
    variant_suffix = f"_{DEFAULT_CONFIG_VARIANT}"
    Config.SAVE_MODEL_PATH = f"models_9x9{variant_suffix}"
    Config.SAVE_PICKLES = f"pickles_9x9{variant_suffix}"
    Config.LOGDIR = f"logs_9x9{variant_suffix}"
    Config.TEST_OUTPUT_PATH = f"test_output_9x9{variant_suffix}"

# Export commonly used constants at module level for convenience
BOARD_SIZE = Config.BOARD_SIZE
NUM_POSITIONS = Config.NUM_POSITIONS
PASS_ACTION = Config.PASS_ACTION
ACTION_SIZE = Config.ACTION_SIZE
KOMI = Config.KOMI


def get_config(board_size=None, variant='base'):
    """Get configuration for a specific board size and variant."""
    if board_size is None:
        board_size = DEFAULT_BOARD_SIZE
    if board_size not in _configs:
        raise ValueError(f"Unsupported board size: {board_size}. Supported: {list(_configs.keys())}")
    
    if board_size == 5:
        if variant != 'base':
            print("Warning: Variant ignored for board_size=5; using default.")
        return _configs[5]
    else:
        if variant not in _configs[board_size]:
            raise ValueError(f"Unsupported variant for {board_size}x{board_size}: {variant}. "
                             f"Supported: {list(_configs[board_size].keys())}")
        return _configs[board_size][variant]


def print_config():
    """Print current configuration."""
    variant_str = f" (Variant: {DEFAULT_CONFIG_VARIANT})" if DEFAULT_BOARD_SIZE == 9 else ""
    print(f"{'='*50}")
    print(f"AlphaZero Go Configuration{variant_str}")
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