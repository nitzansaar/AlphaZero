class Config:
    BATCH_SIZE = 512 

    EPOCHS = 75 
    SELFPLAY_GAMES = 200

    SAVE_MODEL_PATH = "models"
    DATASET_QUEUE_SIZE = 30000
    SAVE_PICKLES = "pickles"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "logs"
    EVAL_GAMES = 40
    ACTION_SIZE = 26 # number of possible actions (5x5 board) - 25 positions + 1 for pass
    NUM_GAMES = 100
    NUM_SIMULATIONS = 200
    
    # AlphaGo Zero specific parameters
    MCTS_UCB_C = 1.414  # sqrt(2) - exploration constant for UCB formula
    VALUE_LOSS_WEIGHT = 1.0  # Weight for value loss
    POLICY_LOSS_WEIGHT = 1.0  # Weight for policy loss
    LEARNING_RATE = 0.001  # Initial learning rate
    WEIGHT_DECAY = 1e-4  # L2 regularization
    MOMENTUM = 0.9  # For SGD optimizer (if used)
    
    # Temperature decay for self-play
    TEMP_THRESHOLD = 10  # Number of moves before switching to deterministic play
    INITIAL_TEMP = 1.0  # Initial temperature for exploration
    
    # Data augmentation
    USE_AUGMENTATION = True  # Enable rotation/reflection augmentation