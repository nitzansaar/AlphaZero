# Neural Network Loss Analysis

This directory contains tools for analyzing games where the neural network loses against the random player.

## Overview

When running `evaluate_vs_random.py`, the script now automatically saves the complete game history for every game where the NN loses. This allows you to review the specific decisions that led to each loss and potentially identify patterns or weaknesses in the training.

## Files

- **evaluate_vs_random.py**: Modified to save loss histories to `nn_losses/` directory
- **analyze_losses.py**: Tool to review and analyze saved loss games
- **nn_losses/**: Directory containing JSON files for each loss (created automatically)

## Usage

### 1. Generate Loss Data

Run the evaluation script as usual:

```bash
python3 evaluate_vs_random.py
```

This will:
- Evaluate the NN against a random player for 2000 games (1000 as X, 1000 as O)
- Save the complete move history for each loss to `nn_losses/loss_game{N}_{X/O}_{timestamp}.json`
- Print a summary showing total losses saved

### 2. Analyze Losses

The `analyze_losses.py` script provides several ways to review the losses:

#### Show Summary (default)
```bash
python3 analyze_losses.py
# or
python3 analyze_losses.py --summary
```
Lists all saved loss files with basic info.

#### Analyze Specific Loss
```bash
python3 analyze_losses.py --file nn_losses/loss_game190_X_20251023_195207.json
```
Shows detailed move-by-move replay of a specific game.

#### Analyze All Losses
```bash
python3 analyze_losses.py --all
```
Shows detailed analysis for all losses (press Enter between games).

## Loss File Format

Each loss is saved as a JSON file with the following structure:

```json
{
  "game_number": 190,
  "nn_played_as": "X",
  "timestamp": "20251023_195207",
  "total_moves": 8,
  "moves": [
    {
      "move_number": 1,
      "player": 1,
      "player_symbol": "X",
      "board_before": [0, 0, 0, 0, 0, 0, 0, 0, 0],
      "move_position": 8,
      "move_by": "player1"
    },
    ...
  ]
}
```

## Key Insights from Current Results

Based on the evaluation run:

- **Total Losses**: 33 out of 2000 games (1.6%)
- **Losses as X (first)**: 4 out of 1000 games (0.4%)
- **Losses as O (second)**: 29 out of 1000 games (2.9%)

**Observation**: The NN performs significantly better when playing first (as X) compared to playing second (as O). This suggests:
1. The NN may have learned better opening strategies
2. Playing defensively (second) might be harder for the current model
3. Consider reviewing losses as O to identify common defensive mistakes

## Tips for Analysis

1. **Look for patterns**: Do losses tend to happen in similar board configurations?
2. **Check early game decisions**: Are there specific opening moves that lead to trouble?
3. **Identify missed blocking opportunities**: Did the NN fail to block obvious winning moves by the opponent?
4. **Compare X vs O losses**: Are there systematic differences in how the NN loses when playing first vs second?

## Disabling Loss Saving

If you want to run evaluations without saving losses (faster execution), modify the calls to `simulate_games()` in `evaluate_vs_random.py`:

```python
results_first = simulate_games(nn_player, random_player, num_games=1000,
                               nn_plays_first=True, verbose_games=0, save_losses=False)
```

## Next Steps

After analyzing the losses, you can:

1. Use the insights to adjust training hyperparameters
2. Modify the reward function to penalize the identified mistakes
3. Add more self-play episodes focusing on defensive positions
4. Augment training data with positions similar to the loss scenarios
