import torch
import torch.nn as nn
import torch.optim as optim
import random

# ---------- 1. MODEL DEFINITION ----------

class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 9)  # 9 outputs = 9 possible moves
        )

    def forward(self, x):
        return self.layers(x)


# ---------- 2. MINIMAX IMPLEMENTATION ----------

def check_winner(board):
    """Returns 1 if 'X' wins, -1 if 'O' wins, 0 if draw, None if still playing."""
    wins = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if 0 not in board:
        return 0  # draw
    return None  # game still ongoing


def minimax(board, player):
    """Return (score, move) for best move for 'player'."""
    winner = check_winner(board)
    if winner is not None:
        return winner, None

    best_score = -float('inf') if player == 1 else float('inf')
    best_move = None

    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score, _ = minimax(board, -player)
            board[i] = 0
            if player == 1 and score > best_score:
                best_score, best_move = score, i
            elif player == -1 and score < best_score:
                best_score, best_move = score, i
    return best_score, best_move


# ---------- 3. DATA GENERATION ----------

def generate_training_data(num_examples=500):
    """Generate (board_state, optimal_move) pairs using minimax."""
    data = []
    for _ in range(num_examples):
        # random valid board
        board = [0]*9
        for i in range(random.randint(0, 8)):
            empty = [j for j in range(9) if board[j] == 0]
            if not empty:
                break
            move = random.choice(empty)
            board[move] = 1 if i % 2 == 0 else -1

        # Skip finished games
        if check_winner(board) is not None:
            continue

        # Determine whose turn it is (count Xs vs Os)
        num_x = board.count(1)
        num_o = board.count(-1)
        player = 1 if num_x == num_o else -1

        _, best_move = minimax(board.copy(), player)
        if best_move is None:
            continue

        # Input = board, Output = one-hot move
        x = torch.tensor(board, dtype=torch.float32)
        y = torch.zeros(9)
        y[best_move] = 1.0
        data.append((x, y))

    return data


# ---------- 4. TRAINING ----------

def train_neural_net(model, train_data, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(train_data)
        for x, y in train_data:
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "tictactoe_model.pth")
    print("✅ Model saved to tictactoe_model.pth")


# ---------- 5. MAIN ENTRY ----------

if __name__ == "__main__":
    print("Generating training data...")
    data = generate_training_data(num_examples=2000)
    print(f"Generated {len(data)} examples.")
    model = TicTacToeNet()
    train_neural_net(model, data, epochs=20, lr=0.001)