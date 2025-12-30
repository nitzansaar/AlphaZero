class Config:
    # Optimized for NVIDIA RTX 5090
    BATCH_SIZE = 512  # Increased from 100 (5090 has plenty of memory)
    
    # Training hyperparameters (AlphaGo Zero style)
    # For 3x3 tic-tac-toe, simpler game than Go
    # Recommended: Start with these values, increase if model plateaus
    EPOCHS = 200  # 200 epochs usually sufficient with early stopping
    SELFPLAY_GAMES = 1000  # More diverse training data

    # For iterative training (run train.sh multiple times):
    # Iteration 1-5:  1000 games, 200 epochs
    # Iteration 6-10: 1500 games, 200 epochs
    # Iteration 11+:  2000 games, 200 epochs

    SAVE_MODEL_PATH = "output_3x3/models"
    DATASET_QUEUE_SIZE = 500000
    SAVE_PICKLES = "output_3x3/pickles"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "output_3x3/logs"
    EVAL_GAMES = 40
    ACTION_SIZE = 9  # number of possible actions (3x3 board)
    NUM_GAMES = 100
    NUM_SIMULATIONS = 500
    
    # AlphaGo Zero specific parameters
    MCTS_UCB_C = 1.414  # sqrt(2) - exploration constant for UCB formula
    VALUE_LOSS_WEIGHT = 1.0  # Weight for value loss
    POLICY_LOSS_WEIGHT = 1.0  # Weight for policy loss
    LEARNING_RATE = 0.001  # Initial learning rate
    WEIGHT_DECAY = 1e-4  # L2 regularization
    MOMENTUM = 0.9  # For SGD optimizer (if used)
    
    # Temperature decay for self-play
    # For 3x3 tic-tac-toe: max 9 moves, so use early threshold
    TEMP_THRESHOLD = 5  # Number of moves before switching to deterministic play
    INITIAL_TEMP = 1.0  # Initial temperature for exploration
    
    # Data augmentation
    USE_AUGMENTATION = True  # Enable rotation/reflection augmentation