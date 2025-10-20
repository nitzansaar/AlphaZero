"""
Comprehensive training script for tic-tac-toe neural network.

This script generates ALL possible game states and trains a neural network
to play optimally using minimax-labeled data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional

# ---------- 1. MODEL DEFINITION ----------

class TicTacToeNet(nn.Module):
    """
    Neural network for tic-tac-toe that takes canonical 3-plane board representation.
    Input: (batch, 3, 3, 3) - 3 planes: current player, opponent, empty
    Output: (batch, 9) - policy logits for 9 positions
    """
    def __init__(self):
        super().__init__()
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 9)  # 9 possible moves

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, 3, 3, 3)
        x = self.relu(self.conv1(x))  # (batch, 32, 2, 2)
        x = self.relu(self.conv2(x))  # (batch, 64, 1, 1)

        x = x.view(-1, 64)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)  # (batch, 9)

        return x


# ---------- 2. MINIMAX IMPLEMENTATION ----------

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


def minimax(board: List[int], player: int, memo: dict = None) -> Tuple[int, Optional[int]]:
    """
    Return (score, best_move) for optimal play.

    Args:
        board: Current board state (list of 9 ints)
        player: Current player (1 or -1)
        memo: Memoization dict for speedup

    Returns:
        (best_score, best_move) where score is in {-1, 0, 1}
    """
    if memo is None:
        memo = {}

    # Convert board to hashable key
    key = (tuple(board), player)
    if key in memo:
        return memo[key]

    winner = check_winner(board)
    if winner is not None:
        return winner, None

    best_score = -float('inf') if player == 1 else float('inf')
    best_move = None

    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score, _ = minimax(board, -player, memo)
            board[i] = 0

            if player == 1 and score > best_score:
                best_score, best_move = score, i
            elif player == -1 and score < best_score:
                best_score, best_move = score, i

    memo[key] = (best_score, best_move)
    return best_score, best_move


# ---------- 3. COMPREHENSIVE DATA GENERATION ----------

def generate_all_game_states() -> List[Tuple[np.ndarray, int, int]]:
    """
    Generate ALL reachable tic-tac-toe positions with optimal moves.

    Returns:
        List of (canonical_board_3d, optimal_move_idx, player) tuples
        canonical_board_3d: (3, 3, 3) numpy array
        optimal_move_idx: 0-8 representing the best move
        player: 1 or -1 for current player
    """
    all_states = []
    memo = {}  # Shared memoization for minimax
    visited = set()  # Avoid duplicate states

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

    def explore_states(board: List[int], current_player: int):
        """Recursively explore all game states."""
        # Check if game is over
        winner = check_winner(board)
        if winner is not None:
            return

        # Create state signature for deduplication
        state_sig = (tuple(board), current_player)
        if state_sig in visited:
            return
        visited.add(state_sig)

        # Get optimal move for current state
        _, best_move = minimax(board.copy(), current_player, memo)
        if best_move is None:
            return

        # Store this state with its optimal move
        canonical_board = board_to_canonical_3d(board, current_player)
        all_states.append((canonical_board, best_move, current_player))

        # Recursively explore all legal moves
        for i in range(9):
            if board[i] == 0:
                board[i] = current_player
                explore_states(board, -current_player)
                board[i] = 0

    # Start exploration from empty board
    explore_states([0]*9, 1)

    return all_states


# ---------- 4. TRAINING ----------

def train_neural_net(model: nn.Module, train_data: List, val_data: List,
                     epochs: int = 50, lr: float = 0.001, device: str = 'cpu'):
    """
    Train the neural network with proper validation and metrics.

    Args:
        model: TicTacToeNet instance
        train_data: List of (board_3d, move_idx) tuples
        val_data: Validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()  # Proper loss for classification

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Shuffle training data
        np.random.shuffle(train_data)

        # Mini-batch training
        batch_size = 32
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]

            # Prepare batch
            boards = torch.stack([torch.from_numpy(b) for b, _ in batch]).to(device)
            targets = torch.tensor([m for _, m in batch], dtype=torch.long).to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(boards)
            loss = loss_fn(logits, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        train_acc = 100.0 * correct / total if total > 0 else 0

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                boards = torch.stack([torch.from_numpy(b) for b, _ in batch]).to(device)
                targets = torch.tensor([m for _, m in batch], dtype=torch.long).to(device)

                logits = model(boards)
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += len(targets)

        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {total_loss:.4f} - "
              f"Train Acc: {train_acc:.2f}% - "
              f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "tictactoe_model_best.pth")

    # Save final model
    torch.save(model.state_dict(), "tictactoe_model.pth")
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Models saved: tictactoe_model.pth, tictactoe_model_best.pth")


# ---------- 5. MAIN ENTRY ----------

if __name__ == "__main__":
    print("="*60)
    print("OPTIMAL TIC-TAC-TOE NEURAL NETWORK TRAINING")
    print("="*60)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Generate ALL possible game states
    print("\n[1/4] Generating all possible game states with optimal moves...")
    all_states = generate_all_game_states()
    print(f"   Generated {len(all_states)} unique game states")

    # Convert to training format
    print("\n[2/4] Preparing training and validation datasets...")
    training_data = [(board, move) for board, move, _ in all_states]

    # Split into train/val (90/10)
    np.random.shuffle(training_data)
    split_idx = int(0.9 * len(training_data))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]

    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")

    # Create model
    print("\n[3/4] Initializing neural network...")
    model = TicTacToeNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")

    # Train
    print("\n[4/4] Training neural network...")
    print("-"*60)
    train_neural_net(model, train_data, val_data, epochs=50, lr=0.001, device=device)

    print("\n" + "="*60)
    print("TRAINING COMPLETE - Model ready for optimal play!")
    print("="*60)