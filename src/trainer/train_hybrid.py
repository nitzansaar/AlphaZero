#!/usr/bin/env python3
"""
Hybrid AlphaZero Training: C++ Self-Play + Python Neural Network Training

This script implements a hybrid architecture:
- C++ MCTS: Fast tree search (47x faster than Python)
- Python PyTorch: Neural network training and inference

PERFORMANCE NOTE:
    Expected speedup: 1.25x faster than pure Python (25% improvement)

    Why modest? Neural network inference is the bottleneck (99.5% of time).
    The C++ MCTS is extremely fast but gets called hundreds of times per game,
    each requiring Python callback overhead. For dramatic speedup, you would need:
    - Batched NN inference (5-10x potential)
    - LibTorch for C++ NN inference (50-100x potential)

    For Tic-Tac-Toe, 1.25x speedup is acceptable. Larger games (Go, Chess)
    with more expensive NN inference would benefit more from this architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys
from collections import deque
from typing import List, Tuple

# Add paths for imports
src_path = os.path.join(os.path.dirname(__file__), '..')
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

from trainer.train_self_play import TicTacToeNet
from trainer.nn_predictor_wrapper import NNPredictorWrapper

# Try to import C++ module
try:
    import mcts_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("WARNING: mcts_cpp not available. Run 'python setup.py build_ext --inplace' first.")


class ReplayBuffer:
    """
    Experience replay buffer for training.

    Stores recent experiences and samples mini-batches for training.
    """

    def __init__(self, max_size: int = 50000):
        """
        Create replay buffer.

        Args:
            max_size: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add_batch(self, experiences: List):
        """Add a batch of experiences from C++."""
        for exp in experiences:
            self.buffer.append({
                'state': np.array(exp.state, dtype=np.float32),
                'policy': np.array(exp.policy, dtype=np.float32),
                'value': float(exp.value)
            })

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch for training.

        Returns:
            (states, policies, values) as PyTorch tensors
        """
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)

        states = []
        policies = []
        values = []

        for idx in indices:
            exp = self.buffer[idx]
            states.append(exp['state'])
            policies.append(exp['policy'])
            values.append(exp['value'])

        # Convert to tensors
        states_tensor = torch.from_numpy(np.array(states)).reshape(-1, 3, 3, 3)
        policies_tensor = torch.from_numpy(np.array(policies))
        values_tensor = torch.from_numpy(np.array(values)).unsqueeze(1)

        return states_tensor, policies_tensor, values_tensor

    def __len__(self):
        return len(self.buffer)


def train_hybrid(
    model: nn.Module,
    total_games: int = 10000,
    games_per_iteration: int = 256,
    batch_size: int = 128,
    train_steps_per_iteration: int = 10,
    num_simulations: int = 50,
    num_threads: int = 8,
    lr: float = 0.001,
    device: str = 'cpu',
    checkpoint_dir: str = 'models/hybrid',
    save_every: int = 1000
):
    """
    Train using hybrid C++/Python approach.

    Args:
        model: Neural network to train
        total_games: Total number of self-play games to generate
        games_per_iteration: Games to generate per iteration
        batch_size: Mini-batch size for training
        train_steps_per_iteration: Training steps per iteration
        num_simulations: MCTS simulations per move
        num_threads: Threads for parallel C++ self-play
        lr: Learning rate
        device: 'cpu' or 'cuda'
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N games
    """

    if not CPP_AVAILABLE:
        raise RuntimeError("C++ module not available. Cannot run hybrid training.")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup model and optimizer
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=total_games//4, gamma=0.5)

    # Create replay buffer (same as train_self_play.py uses 10000)
    replay_buffer = ReplayBuffer(max_size=10000)

    # Create Python wrapper for neural network
    nn_wrapper = NNPredictorWrapper(model, device)

    # Create C++ predictor that calls back to Python neural network
    cpp_predictor = mcts_cpp.PythonCallbackPredictor(nn_wrapper)

    # NOTE: We use SelfPlayGame directly instead of BatchGenerator because
    # BatchGenerator uses std::async which creates threads that are incompatible
    # with Python callbacks. This is still much faster than pure Python because
    # the C++ MCTS tree search is significantly faster than Python MCTS.

    # Training initialization
    print(f"\n{'='*70}")
    print("HYBRID C++/PYTHON TRAINING INITIALIZATION")
    print(f"{'='*70}")
    print(f"  Total games target: {total_games}")
    print(f"  Games per iteration: {games_per_iteration}")
    print(f"  MCTS simulations: {num_simulations}")
    print(f"  Training batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Architecture: C++ MCTS tree search + Python PyTorch NN")
    print(f"  Mode: Sequential game generation (no threading due to Python GIL)")
    print(f"{'='*70}\n")

    # Training history
    history = {
        'iteration': [],
        'total_games': [],
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'p1_win_rate': [],
        'buffer_size': [],
        'time': []
    }

    games_played = 0
    iteration = 0
    start_time = time.time()

    while games_played < total_games:
        iteration += 1
        iter_start_time = time.time()

        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")

        # [1] Generate self-play data in C++ with neural network
        # Set model to eval mode before C++ uses it
        model.eval()

        print(f"  Generating {games_per_iteration} games in C++...")
        gen_start = time.time()

        # Generate games sequentially using C++ SelfPlayGame
        all_experiences = []
        p1_wins, p2_wins, draws = 0, 0, 0

        for i in range(games_per_iteration):
            # Create game with C++ predictor
            game = mcts_cpp.SelfPlayGame(cpp_predictor, num_simulations, temperature=1.0, add_noise=True)
            experiences = game.play_game()
            result = game.get_game_result()

            all_experiences.extend(experiences)

            # Track results
            if result == 1:
                p1_wins += 1
            elif result == -1:
                p2_wins += 1
            else:
                draws += 1

        gen_time = time.time() - gen_start
        games_played += games_per_iteration

        print(f"  ✅ Generated {games_per_iteration} games in {gen_time:.2f}s ({games_per_iteration/gen_time:.1f} games/sec)")
        print(f"     Experiences: {len(all_experiences)}")
        print(f"     P1 wins: {p1_wins}, P2 wins: {p2_wins}, Draws: {draws}")

        # [2] Add to replay buffer
        replay_buffer.add_batch(all_experiences)
        print(f"  📦 Replay buffer size: {len(replay_buffer)}")

        # [3] Train neural network
        if len(replay_buffer) >= batch_size:
            # Match train_self_play.py: do multiple training steps proportional to experiences
            # train_self_play.py does max(1, len(game_history) // 2) per game
            # Average game has ~6-7 moves, so experiences // 2 gives similar training intensity
            num_train_steps = max(1, len(all_experiences) // 2)

            print(f"  🧠 Training neural network ({num_train_steps} steps)...")
            model.train()

            train_losses = []
            policy_losses = []
            value_losses = []

            for step in range(num_train_steps):
                # Sample mini-batch
                states, policies, values = replay_buffer.sample(batch_size)
                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)

                # Forward pass
                optimizer.zero_grad()
                policy_logits, pred_values = model(states)

                # Compute losses
                log_probs = torch.log_softmax(policy_logits, dim=1)
                policy_loss = -torch.sum(policies * log_probs, dim=1).mean()
                value_loss = torch.mean((pred_values - values) ** 2)
                total_loss = policy_loss + value_loss

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(total_loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

            avg_total_loss = np.mean(train_losses)
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)

            print(f"  ✅ Training complete")
            print(f"     Policy loss: {avg_policy_loss:.4f}")
            print(f"     Value loss: {avg_value_loss:.4f}")
            print(f"     Total loss: {avg_total_loss:.4f}")

            scheduler.step()
        else:
            avg_total_loss = 0.0
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            print(f"  ⏭️  Skipping training (buffer too small: {len(replay_buffer)} < {batch_size})")

        # [4] Update history
        p1_win_rate = p1_wins / games_per_iteration if games_per_iteration > 0 else 0.0
        iter_time = time.time() - iter_start_time

        history['iteration'].append(iteration)
        history['total_games'].append(games_played)
        history['policy_loss'].append(avg_policy_loss)
        history['value_loss'].append(avg_value_loss)
        history['total_loss'].append(avg_total_loss)
        history['p1_win_rate'].append(p1_win_rate)
        history['buffer_size'].append(len(replay_buffer))
        history['time'].append(iter_time)

        # [5] Save checkpoint
        if games_played % save_every < games_per_iteration or games_played >= total_games:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_{games_played}g.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")

        # [6] Progress summary
        elapsed_time = time.time() - start_time
        print(f"\n  📊 Progress: {games_played}/{total_games} games ({100*games_played/total_games:.1f}%)")
        print(f"     Elapsed time: {elapsed_time:.1f}s")
        print(f"     Avg time per iteration: {elapsed_time/iteration:.1f}s")
        print(f"     Est. time remaining: {elapsed_time/games_played*(total_games-games_played):.1f}s")

    # Final save
    final_path = os.path.join(checkpoint_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total games: {games_played}")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    print(f"  Final model: {final_path}")
    print(f"{'='*70}\n")

    return history


def main():
    """Run training - same as train_self_play.py but organized in iterations."""
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = TicTacToeNet()

    # Use same configuration as train_self_play.py for consistency
    batch_size = 128 if device == 'cuda' else 64

    # Train with same parameters as train_self_play.py
    history = train_hybrid(
        model=model,
        total_games=100000,           # Same as train_self_play.py episodes
        games_per_iteration=100,     # Generate 100 games per iteration (for batched progress display)
        batch_size=batch_size,       # 64 (CPU) or 128 (GPU)
        train_steps_per_iteration=1, # Will be adjusted based on experiences
        num_simulations=50,          # Same as train_self_play.py
        num_threads=1,               # Single-threaded (std::async incompatible with Python callbacks)
        lr=0.001,                    # Same as train_self_play.py
        device=device,
        checkpoint_dir='models/hybrid',
        save_every=500
    )

    print("\n✅ Hybrid training complete!")
    print(f"   Generated {history['total_games'][-1]} games")
    print(f"   Final loss: {history['total_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
