class Config:
    # Optimized for NVIDIA RTX 5090
    BATCH_SIZE = 512  # Increased from 100 (5090 has plenty of memory)
    EPOCHS = 500
    SAVE_MODEL_PATH = "output_tictac/models"
    DATASET_QUEUE_SIZE = 500000
    SELFPLAY_GAMES = 500
    SAVE_PICKLES = "output_tictac/pickles"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "output_tictac/logs"
    EVAL_GAMES = 40
    ACTION_SIZE = 81 # number of possible actions (9x9 board)
    NUM_GAMES = 100
    NUM_SIMULATIONS = 800