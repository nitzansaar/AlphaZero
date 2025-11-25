class Config:
    # Optimized for NVIDIA RTX 5090
    BATCH_SIZE = 512  # Increased from 100 (5090 has plenty of memory)
    
    # Training hyperparameters (AlphaGo Zero style)
    # For 9x9 tic-tac-toe, use fewer games/epochs than 19x19 Go
    # Recommended: Start with these values, increase if model plateaus
    EPOCHS = 200  # Reduced from 500 - 200-300 epochs usually sufficient with early stopping
    SELFPLAY_GAMES = 1000  # Increased from 500 - more diverse training data
    
    # For iterative training (run train.sh multiple times):
    # Iteration 1-5:  1000 games, 200 epochs
    # Iteration 6-10: 1500 games, 200 epochs  
    # Iteration 11+:  2000 games, 200 epochs
    
    SAVE_MODEL_PATH = "output_tictac/models"
    DATASET_QUEUE_SIZE = 500000
    SAVE_PICKLES = "output_tictac/pickles"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "output_tictac/logs"
    EVAL_GAMES = 40
    ACTION_SIZE = 81 # number of possible actions (9x9 board)
    NUM_GAMES = 100
    NUM_SIMULATIONS = 1600
    
    # AlphaGo Zero specific parameters
    MCTS_UCB_C = 1.414  # sqrt(2) - exploration constant for UCB formula
    VALUE_LOSS_WEIGHT = 1.0  # Weight for value loss
    POLICY_LOSS_WEIGHT = 1.0  # Weight for policy loss
    LEARNING_RATE = 0.001  # Initial learning rate
    WEIGHT_DECAY = 1e-4  # L2 regularization
    MOMENTUM = 0.9  # For SGD optimizer (if used)
    
    # Temperature decay for self-play
    TEMP_THRESHOLD = 30  # Number of moves before switching to deterministic play
    INITIAL_TEMP = 1.0  # Initial temperature for exploration
    
    # Data augmentation
    USE_AUGMENTATION = True  # Enable rotation/reflection augmentation