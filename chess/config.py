class Config:
    # ------------------------------------------------------------------
    # Action / observation space
    # ------------------------------------------------------------------
    # AlphaZero chess move encoding: 8x8 from-squares x 73 move planes.
    #   - 56 "queen" sliding moves (8 directions x 7 distances)
    #   - 8 knight moves
    #   - 9 underpromotions (3 pieces {N,B,R} x 3 directions {forward, capture-left, capture-right})
    # Queen promotions are represented implicitly by the matching sliding move.
    ACTION_SIZE = 8 * 8 * 73  # 4672

    # Input planes fed to the network (see encoding.board_to_planes):
    #   12 piece planes (6 piece types x 2 colors)
    #    1 side-to-move plane
    #    4 castling-rights planes (own K/Q side, opponent K/Q side)
    #    1 en-passant plane
    #    1 fifty-move (halfmove clock) plane
    NUM_INPUT_PLANES = 19

    # ------------------------------------------------------------------
    # Network architecture (AlphaGo Zero style residual tower)
    # ------------------------------------------------------------------
    NUM_RES_BLOCKS = 10
    NUM_CHANNELS = 128

    # ------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------
    BATCH_SIZE = 512
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    VALUE_LOSS_WEIGHT = 1.0
    POLICY_LOSS_WEIGHT = 1.0

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------
    SELFPLAY_GAMES = 200
    NUM_SIMULATIONS = 400
    MAX_MOVES = 200  # cap full-moves-equivalent plies per self-play game
    NUM_SELFPLAY_WORKERS = 6  # parallel self-play processes sharing the GPU

    # Temperature decay (AlphaGo Zero style)
    TEMP_THRESHOLD = 30  # plies of exploratory play before near-deterministic
    INITIAL_TEMP = 1.0
    FINAL_TEMP = 0.1

    # MCTS / PUCT
    MCTS_UCB_C = 1.414  # exploration constant
    DIRICHLET_ALPHA = 0.3  # root noise concentration (AlphaZero chess)
    DIRICHLET_EPSILON = 0.25  # root noise mixing weight

    # ------------------------------------------------------------------
    # Dataset / checkpoint paths
    # ------------------------------------------------------------------
    SAVE_MODEL_PATH = "output_chess/models"
    SAVE_PICKLES = "output_chess/pickles"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "output_chess/logs"
    DATASET_QUEUE_SIZE = 1_000_000

    EVAL_GAMES = 20
