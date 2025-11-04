class Config:
    BATCH_SIZE = 100
    EPOCHS = 1000
    SAVE_MODEL_PATH = "output_tictac/models"
    DATASET_QUEUE_SIZE = 500000
    SELFPLAY_GAMES = 2500
    SAVE_PICKLES = "output_tictac/pickles"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "output_tictac/logs"
    EVAL_GAMES = 40
    ACTION_SIZE = 16 # number of possible actions (4x4 board)
    NUM_GAMES = 1000
    NUM_SIMULATIONS = 800