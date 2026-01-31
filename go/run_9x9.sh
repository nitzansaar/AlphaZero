#!/bin/bash
# Run training/testing for 9x9 Go
# Usage: ./run_9x9.sh [command] [args]
#
# Commands:
#   train [iterations]     - Run training loop
#   test-random [games]    - Test against random player
#   test-minimax [games]   - Test against minimax player
#   selfplay               - Run self-play only
#   config                 - Print configuration

export BOARD_SIZE=9

case "$1" in
    train)
        iterations=${2:-10}
        echo "Training 9x9 Go for $iterations iterations..."
        python run_training_loop.py --iterations $iterations
        ;;
    test-random)
        games=${2:-100}
        echo "Testing 9x9 Go against random ($games games)..."
        python test_vs_random.py --games $games
        ;;
    test-minimax)
        games=${2:-10}
        depth=${3:-2}
        echo "Testing 9x9 Go against minimax depth $depth ($games games)..."
        python test_vs_minimax.py --games $games --depth $depth
        ;;
    selfplay)
        echo "Running 9x9 self-play..."
        python selfplay.py
        ;;
    config)
        python config.py
        ;;
    *)
        echo "9x9 Go AlphaZero"
        echo ""
        echo "Usage: ./run_9x9.sh [command] [args]"
        echo ""
        echo "Commands:"
        echo "  train [iterations]        - Run training loop (default: 10)"
        echo "  test-random [games]       - Test against random (default: 100)"
        echo "  test-minimax [games] [d]  - Test against minimax (default: 10 games, depth 2)"
        echo "  selfplay                  - Run self-play only"
        echo "  config                    - Print configuration"
        echo ""
        echo "Note: 9x9 requires significantly more training than 5x5."
        echo "Recommended: 20-50+ iterations for good performance."
        ;;
esac
