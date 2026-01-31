#!/bin/bash
# Run training/testing for 5x5 Go
# Usage: ./run_5x5.sh [command] [args]
#
# Commands:
#   train [iterations]     - Run training loop
#   test-random [games]    - Test against random player
#   test-minimax [games]   - Test against minimax player
#   selfplay               - Run self-play only
#   config                 - Print configuration

export BOARD_SIZE=5

case "$1" in
    train)
        iterations=${2:-5}
        echo "Training 5x5 Go for $iterations iterations..."
        python run_training_loop.py --iterations $iterations
        ;;
    test-random)
        games=${2:-100}
        echo "Testing 5x5 Go against random ($games games)..."
        python test_vs_random.py --games $games
        ;;
    test-minimax)
        games=${2:-20}
        depth=${3:-3}
        echo "Testing 5x5 Go against minimax depth $depth ($games games)..."
        python test_vs_minimax.py --games $games --depth $depth
        ;;
    selfplay)
        echo "Running 5x5 self-play..."
        python selfplay.py
        ;;
    config)
        python config.py
        ;;
    *)
        echo "5x5 Go AlphaZero"
        echo ""
        echo "Usage: ./run_5x5.sh [command] [args]"
        echo ""
        echo "Commands:"
        echo "  train [iterations]        - Run training loop (default: 5)"
        echo "  test-random [games]       - Test against random (default: 100)"
        echo "  test-minimax [games] [d]  - Test against minimax (default: 20 games, depth 3)"
        echo "  selfplay                  - Run self-play only"
        echo "  config                    - Print configuration"
        ;;
esac
