"""
AlphaZero-style self-play training script for tic-tac-toe neural network.

This script implements AlphaZero-style training combining:
- Neural network with policy + value heads
- Monte Carlo Tree Search (MCTS) for move selection
- Self-play to generate training data
- Experience replay for stable learning

Key features:
- MCTS discovers tactical moves (blocks, forks) through lookahead search
- Neural network learns to predict what MCTS would do
- Training visualization showing improvement over time
- Saves checkpoints and training history for analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import random
import matplotlib.pyplot as plt
import json

# ---------- 1. MODEL DEFINITION ----------

class TicTacToeNet(nn.Module):
    """
    AlphaZero-style neural network for tic-tac-toe with policy and value heads.
    Input: (batch, 3, 3, 3) - 3 planes: current player, opponent, empty
    Outputs:
        - policy: (batch, 9) - move probability logits
        - value: (batch, 1) - position evaluation in [-1, 1]
    """
    def __init__(self):
        super().__init__()
        # Shared convolutional layers for spatial features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=0)

        # Shared fully connected layer
        self.fc_shared = nn.Linear(64 * 1 * 1, 128)

        # Policy head (move probabilities)
        self.fc_policy = nn.Linear(128, 9)

        # Value head (position evaluation)
        self.fc_value1 = nn.Linear(128, 64)
        self.fc_value2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, 3, 3, 3)
        # Shared conv layers
        x = self.relu(self.conv1(x))  # (batch, 32, 2, 2)
        x = self.relu(self.conv2(x))  # (batch, 64, 1, 1)

        x = x.view(-1, 64)  # Flatten
        x = self.dropout(self.relu(self.fc_shared(x)))  # (batch, 128)

        # Policy head
        policy = self.fc_policy(x)  # (batch, 9)

        # Value head
        value = self.relu(self.fc_value1(x))  # (batch, 64)
        value = torch.tanh(self.fc_value2(value))  # (batch, 1) in [-1, 1]

        return policy, value


# ---------- 2. GAME LOGIC ----------

def check_winner(board: List[int]) -> Optional[int]:
    """
    Returns 1 if 'X' wins, -1 if 'O' wins, 0 if draw, None if still playing.
    board: list of 9 elements representing 3x3 grid (row-major order)
    """
    wins = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # cols
        (0,4,8), (2,4,6)             # diagonals
    ]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if 0 not in board:
        return 0  # draw
    return None  # game ongoing


def get_valid_moves(board: List[int]) -> List[int]:
    """Return list of valid move indices."""
    return [i for i in range(9) if board[i] == 0]


def board_to_canonical_3d(board_flat: List[int], player: int) -> np.ndarray:
    """
    Convert flat board to canonical 3-plane representation.
    Canonical means current player is always represented as +1.
    """
    board_2d = np.array(board_flat).reshape(3, 3)
    # Flip perspective so current player is always +1
    canonical = board_2d * player

    planes = np.zeros((3, 3, 3), dtype=np.float32)
    planes[0] = (canonical == 1).astype(np.float32)   # current player
    planes[1] = (canonical == -1).astype(np.float32)  # opponent
    planes[2] = (canonical == 0).astype(np.float32)   # empty

    return planes


# ---------- 3. MCTS IMPLEMENTATION ----------

class MCTSNode:
    """Node in the Monte Carlo Tree Search."""

    def __init__(self, prior_prob: float):
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob
        self.children = {}  # map: action -> MCTSNode

    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def select_child(self, c_puct: float = 2.0) -> int:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_action = -1

        # Parent visit count for exploration term
        # Add epsilon to avoid division by zero on first visit
        parent_visits = max(1, self.visit_count)

        for action, child in self.children.items():
            # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            # Q(s,a) is the average value (exploitation)
            # The second term is the exploration bonus (higher for unvisited nodes)
            q_value = child.value()
            u_value = c_puct * child.prior_prob * np.sqrt(parent_visits) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action

        return best_action


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(self, model: nn.Module, device: str, num_simulations: int = 50, c_puct: float = 2.0, debug: bool = False):
        """
        Args:
            model: Neural network (policy + value)
            device: 'cpu' or 'cuda'
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant (higher = more exploration, typical: 1-5)
            debug: Enable debug output
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.debug = debug

    def get_action_probs(self, board: List[int], player: int, temperature: float = 1.0,
                         add_noise: bool = False, dirichlet_alpha: float = 0.3, noise_epsilon: float = 0.25) -> Tuple[List[float], List[int]]:
        """
        Run MCTS simulations and return action probabilities.

        Args:
            board: Current board state
            player: Current player
            temperature: Temperature for exploration (higher = more random)
            add_noise: Whether to add Dirichlet noise to priors for exploration
            dirichlet_alpha: Alpha parameter for Dirichlet distribution (lower = more uniform)
            noise_epsilon: Weight of noise vs network prior (0.25 = 25% noise, 75% prior)

        Returns:
            (probabilities, valid_moves) where probabilities[i] is prob of move i
        """
        root = MCTSNode(prior_prob=1.0)
        valid_moves = get_valid_moves(board)

        # Initialize root node children with NN policy
        canonical_board = board_to_canonical_3d(board, player)
        board_tensor = torch.from_numpy(canonical_board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(board_tensor)
            policy_logits = policy_logits.squeeze(0).cpu().numpy()

        # Mask invalid moves and normalize
        policy = np.exp(policy_logits - np.max(policy_logits))
        policy_sum = sum(policy[m] for m in valid_moves)

        # Normalize policy for valid moves only
        policy_probs = np.zeros(9)
        for move in valid_moves:
            policy_probs[move] = policy[move] / policy_sum if policy_sum > 0 else 1.0 / len(valid_moves)

        # Add Dirichlet noise to encourage exploration (AlphaZero technique)
        if add_noise and len(valid_moves) > 1:
            noise = np.random.dirichlet([dirichlet_alpha] * len(valid_moves))
            for i, move in enumerate(valid_moves):
                policy_probs[move] = (1 - noise_epsilon) * policy_probs[move] + noise_epsilon * noise[i]

        # Create child nodes with final priors
        for move in valid_moves:
            root.children[move] = MCTSNode(prior_prob=policy_probs[move])

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(board.copy(), player, root)

        # Compute visit count distribution
        visits = np.zeros(9)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        # Debug: print visit counts and values
        if self.debug:
            print(f"\nMCTS Search Results ({self.num_simulations} sims):")
            for action in sorted(root.children.keys()):
                child = root.children[action]
                row, col = action // 3, action % 3
                print(f"  Move ({row},{col}): visits={child.visit_count:3d}, value={child.value():.3f}, prior={child.prior_prob:.3f}")

        # Apply temperature
        if temperature == 0:
            # Greedy: pick most visited
            action_probs = np.zeros(9)
            best_action = int(np.argmax(visits))
            action_probs[best_action] = 1.0
        else:
            # Boltzmann distribution
            visits_temp = visits ** (1.0 / temperature)
            visits_sum = np.sum(visits_temp)
            action_probs = visits_temp / visits_sum if visits_sum > 0 else visits / np.sum(visits) if np.sum(visits) > 0 else np.ones(9) / 9

        return action_probs.tolist(), valid_moves

    def _simulate(self, board: List[int], player: int, node: MCTSNode) -> float:
        """
        Run one MCTS simulation from this node.

        Returns:
            value from current player's perspective
        """
        # Check terminal state
        winner = check_winner(board)
        if winner is not None:
            # Return value from current player's perspective
            return winner * player

        # Select action
        action = node.select_child(self.c_puct)

        # Apply action
        board[action] = player
        next_player = -player

        # Get child node
        child = node.children[action]

        # Recursively simulate or evaluate
        if child.visit_count == 0:
            # Leaf node: evaluate with neural network
            canonical_board = board_to_canonical_3d(board, next_player)
            board_tensor = torch.from_numpy(canonical_board).unsqueeze(0).to(self.device)

            with torch.no_grad():
                policy_logits, value = self.model(board_tensor)
                value = value.item()  # Value from next_player's perspective

            # Expand node
            valid_moves = get_valid_moves(board)
            if valid_moves:
                policy_logits = policy_logits.squeeze(0).cpu().numpy()
                policy = np.exp(policy_logits - np.max(policy_logits))
                policy_sum = sum(policy[m] for m in valid_moves)

                for move in valid_moves:
                    prior_prob = policy[move] / policy_sum if policy_sum > 0 else 1.0 / len(valid_moves)
                    child.children[move] = MCTSNode(prior_prob=prior_prob)

            # Value is from next_player's perspective, flip for current player
            value = -value
        else:
            # Recurse
            value = -self._simulate(board, next_player, child)

        # Backpropagate
        child.visit_count += 1
        child.total_value += value

        return value


# ---------- 4. SELF-PLAY GAME ENGINE ----------

class SelfPlayGame:
    """Manages a single self-play game using MCTS for move selection."""

    def __init__(self, model: nn.Module, device: str, num_simulations: int = 50, temperature: float = 1.0):
        """
        Args:
            model: Neural network model (policy + value)
            device: 'cpu' or 'cuda'
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection (1.0 = stochastic, 0 = greedy)
        """
        self.model = model
        self.device = device
        self.mcts = MCTS(model, device, num_simulations=num_simulations)
        self.temperature = temperature
        self.game_history = []  # List of (state, mcts_policy, player) tuples

    def play_game(self) -> Tuple[int, List]:
        """
        Play a complete self-play game using MCTS.

        Returns:
            (game_result, game_history) where:
                game_result: 1 if player 1 wins, -1 if player -1 wins, 0 for draw
                game_history: List of (canonical_state, mcts_policy, player) tuples
        """
        board = [0] * 9
        current_player = 1
        self.game_history = []

        move_count = 0

        while True:
            # Check if game is over
            winner = check_winner(board)
            if winner is not None:
                return winner, self.game_history

            # Get canonical state (current player's perspective)
            canonical_state = board_to_canonical_3d(board, current_player)

            # Use temperature decay: high temp early, low temp later
            # This encourages exploration early, exploitation later
            temp = self.temperature if move_count < 10 else 0.1

            # Get MCTS policy with Dirichlet noise to encourage exploration
            mcts_policy, _ = self.mcts.get_action_probs(
                board, current_player,
                temperature=temp,
                add_noise=True,  # Add noise during self-play for exploration
                dirichlet_alpha=0.3,
                noise_epsilon=0.25
            )

            # Sample move from MCTS policy
            move = np.random.choice(9, p=mcts_policy)

            # Record state and MCTS-improved policy
            self.game_history.append((canonical_state, mcts_policy, current_player))

            # Make move
            board[move] = current_player
            current_player = -current_player
            move_count += 1


# ---------- 5. EXPERIENCE REPLAY BUFFER ----------

class ReplayBuffer:
    """Store and sample game experiences for AlphaZero-style training."""

    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)

    def add_game(self, game_result: int, game_history: List):
        """
        Add a complete game to the replay buffer.

        Args:
            game_result: Final game outcome (1, -1, or 0)
            game_history: List of (state, mcts_policy, player) tuples
        """
        # Store positions with MCTS-improved policy and game outcome
        for state, mcts_policy, player in game_history:
            # Game outcome from current player's perspective
            value_target = game_result * player
            self.buffer.append((state, mcts_policy, value_target))

    def sample(self, batch_size: int) -> List:
        """Sample a random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ---------- 6. TRAINING ----------

def train_self_play(model: nn.Module, episodes: int = 1000,
                   batch_size: int = 64, lr: float = 0.001,
                   device: str = 'cpu', num_simulations: int = 50,
                   temperature: float = 1.0):
    """
    Train the neural network using AlphaZero-style self-play with MCTS.

    Args:
        model: TicTacToeNet instance (with policy + value heads)
        episodes: Number of self-play games to run
        batch_size: Training batch size
        lr: Learning rate
        device: 'cpu' or 'cuda'
        num_simulations: Number of MCTS simulations per move
        temperature: Temperature for move selection
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Add learning rate scheduler for stability
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=episodes//4, gamma=0.5)

    replay_buffer = ReplayBuffer(max_size=10000)

    stats = {
        'player1_wins': 0,
        'player2_wins': 0,
        'draws': 0,
        'avg_policy_loss': [],
        'avg_value_loss': [],
        'avg_total_loss': []
    }

    # For tracking progress over time (for plotting)
    history = {
        'episodes': [],
        'win_rate_p1': [],
        'win_rate_p2': [],
        'draw_rate': [],
        'policy_loss': [],
        'value_loss': [],
        'total_loss': []
    }

    print(f"Starting AlphaZero-style self-play training for {episodes} episodes...")
    print(f"MCTS simulations per move: {num_simulations}")
    print(f"Temperature: {temperature}")
    print("-" * 60)

    for episode in range(episodes):
        # Play self-play game with MCTS
        model.eval()  # Set to eval mode during self-play
        game_engine = SelfPlayGame(model, device, num_simulations=num_simulations, temperature=temperature)
        game_result, game_history = game_engine.play_game()

        # Add to replay buffer
        replay_buffer.add_game(game_result, game_history)

        # Track statistics
        if game_result == 1:
            stats['player1_wins'] += 1
        elif game_result == -1:
            stats['player2_wins'] += 1
        else:
            stats['draws'] += 1

        # Train on experience replay (multiple training steps per game)
        if len(replay_buffer) >= batch_size:
            model.train()

            # Perform multiple training steps per game for better sample efficiency
            num_train_steps = max(1, len(game_history) // 2)

            for _ in range(num_train_steps):
                batch = replay_buffer.sample(batch_size)

                # Prepare batch tensors
                states = torch.stack([torch.from_numpy(s) for s, _, _ in batch]).to(device)
                target_policies = torch.tensor([p for _, p, _ in batch], dtype=torch.float32).to(device)
                target_values = torch.tensor([v for _, _, v in batch], dtype=torch.float32).unsqueeze(1).to(device)

                # Forward pass
                optimizer.zero_grad()
                policy_logits, values = model(states)

                # Policy loss: cross-entropy between MCTS policy and NN policy
                log_probs = torch.log_softmax(policy_logits, dim=1)
                policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()

                # Value loss: MSE between game outcome and NN value prediction
                value_loss = torch.mean((values - target_values) ** 2)

                # Combined loss
                total_loss = policy_loss + value_loss

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                stats['avg_policy_loss'].append(policy_loss.item())
                stats['avg_value_loss'].append(value_loss.item())
                stats['avg_total_loss'].append(total_loss.item())

        # Step learning rate scheduler
        scheduler.step()

        # Print progress and record history
        if (episode + 1) % 100 == 0:
            win_rate_p1 = 100.0 * stats['player1_wins'] / (episode + 1)
            win_rate_p2 = 100.0 * stats['player2_wins'] / (episode + 1)
            draw_rate = 100.0 * stats['draws'] / (episode + 1)

            avg_policy_loss = np.mean(stats['avg_policy_loss'][-100:]) if stats['avg_policy_loss'] else 0
            avg_value_loss = np.mean(stats['avg_value_loss'][-100:]) if stats['avg_value_loss'] else 0
            avg_total_loss = np.mean(stats['avg_total_loss'][-100:]) if stats['avg_total_loss'] else 0

            # Record history for plotting
            history['episodes'].append(episode + 1)
            history['win_rate_p1'].append(win_rate_p1)
            history['win_rate_p2'].append(win_rate_p2)
            history['draw_rate'].append(draw_rate)
            history['policy_loss'].append(avg_policy_loss)
            history['value_loss'].append(avg_value_loss)
            history['total_loss'].append(avg_total_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Episode {episode+1}/{episodes} - "
                  f"LR: {current_lr:.2e} - "
                  f"Loss(total/pol/val): {avg_total_loss:.4f}/{avg_policy_loss:.4f}/{avg_value_loss:.4f} - "
                  f"P1: {win_rate_p1:.1f}% - "
                  f"P2: {win_rate_p2:.1f}% - "
                  f"Draw: {draw_rate:.1f}%")

            # Save checkpoint
            if (episode + 1) % 500 == 0:
                torch.save(model.state_dict(), f"tictactoe_alphazero_ep{episode+1}.pth")

    # Save final model
    torch.save(model.state_dict(), "tictactoe_selfplay_final.pth")

    # Save training history
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ AlphaZero-style training complete!")
    print(f"   Total games played: {episodes}")
    print(f"   Final statistics:")
    print(f"     Player 1 wins: {stats['player1_wins']} ({100*stats['player1_wins']/episodes:.1f}%)")
    print(f"     Player 2 wins: {stats['player2_wins']} ({100*stats['player2_wins']/episodes:.1f}%)")
    print(f"     Draws: {stats['draws']} ({100*stats['draws']/episodes:.1f}%)")
    print(f"   Model saved: tictactoe_selfplay_final.pth")
    print(f"   Training history saved: training_history.json")

    return history


# ---------- 7. VISUALIZATION ----------

def plot_training_progress(history: dict, save_path: str = "training_progress.png"):
    """
    Plot training progress including win rates and losses.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AlphaZero Training Progress', fontsize=16, fontweight='bold')

    episodes = history['episodes']

    # Plot 1: Win rates over time
    ax1 = axes[0, 0]
    ax1.plot(episodes, history['win_rate_p1'], label='Player 1 (X)', color='blue', linewidth=2)
    ax1.plot(episodes, history['win_rate_p2'], label='Player 2 (O)', color='red', linewidth=2)
    ax1.plot(episodes, history['draw_rate'], label='Draws', color='green', linewidth=2)
    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_title('Game Outcomes Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    # Plot 2: Total loss over time
    ax2 = axes[0, 1]
    ax2.plot(episodes, history['total_loss'], color='purple', linewidth=2)
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Total Loss', fontsize=12)
    ax2.set_title('Total Loss (Policy + Value)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Policy loss over time
    ax3 = axes[1, 0]
    ax3.plot(episodes, history['policy_loss'], color='orange', linewidth=2)
    ax3.set_xlabel('Episodes', fontsize=12)
    ax3.set_ylabel('Policy Loss', fontsize=12)
    ax3.set_title('Policy Loss (Cross-Entropy)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Value loss over time
    ax4 = axes[1, 1]
    ax4.plot(episodes, history['value_loss'], color='brown', linewidth=2)
    ax4.set_xlabel('Episodes', fontsize=12)
    ax4.set_ylabel('Value Loss', fontsize=12)
    ax4.set_title('Value Loss (MSE)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Training progress plot saved: {save_path}")
    plt.close()


# ---------- 8. EVALUATION ----------

def evaluate_model(model: nn.Module, num_games: int = 100, device: str = 'cpu', num_simulations: int = 100):
    """
    Evaluate the trained model by playing games with MCTS.

    Args:
        model: Trained TicTacToeNet
        num_games: Number of evaluation games
        device: 'cpu' or 'cuda'
        num_simulations: Number of MCTS simulations for evaluation (higher = stronger)
    """
    model.eval()
    game_engine = SelfPlayGame(model, device, num_simulations=num_simulations, temperature=0.0)  # Greedy

    results = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0}

    print(f"\nEvaluating model over {num_games} games (MCTS with {num_simulations} sims)...")

    for _ in range(num_games):
        game_result, _ = game_engine.play_game()

        if game_result == 1:
            results['player1_wins'] += 1
        elif game_result == -1:
            results['player2_wins'] += 1
        else:
            results['draws'] += 1

    print(f"\nEvaluation Results:")
    print(f"  Player 1 wins: {results['player1_wins']} ({100*results['player1_wins']/num_games:.1f}%)")
    print(f"  Player 2 wins: {results['player2_wins']} ({100*results['player2_wins']/num_games:.1f}%)")
    print(f"  Draws: {results['draws']} ({100*results['draws']/num_games:.1f}%)")

    return results


# ---------- 9. MAIN ENTRY ----------

if __name__ == "__main__":
    print("=" * 60)
    print("ALPHAZERO-STYLE MCTS + NN TRAINING FOR TIC-TAC-TOE")
    print("=" * 60)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model
    print("\n[1/4] Initializing AlphaZero-style neural network (policy + value)...")
    model = TicTacToeNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Architecture: 3-plane CNN -> policy head (9) + value head (1)")

    # Train with AlphaZero-style self-play
    print("\n[2/4] Training with AlphaZero-style self-play (MCTS + NN)...")
    print("NOTE: For optimal play, train for 20k-50k episodes.")
    print("      Current setting (100k) provides near-optimal tactical play.")
    print("-" * 60)

    # Use larger batch size on GPU for better throughput
    batch_size = 128 if device == 'cuda' else 64

    history = train_self_play(
        model=model,
        episodes=10000,          # Increased for better tactical learning
        batch_size=batch_size,
        lr=0.001,
        device=device,
        num_simulations=50,      # MCTS simulations per move during training
        temperature=1.0          # Exploration temperature
    )

    # Visualize training progress
    print("\n[3/4] Generating training visualization...")
    print("-" * 60)
    plot_training_progress(history)

    # Evaluate
    print("\n[4/4] Evaluating trained model...")
    print("-" * 60)
    evaluate_model(model, num_games=100, device=device, num_simulations=100)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - AlphaZero-style model ready!")
    print("=" * 60)
