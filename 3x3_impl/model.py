import torch
from torch import nn
from config import Config as cfg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Convolutional layers for spatial feature extraction
        # For 3x3 board, use lighter architecture appropriate for the game complexity
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Shared fully connected layer
        # Input size: 64 channels * 3 * 3 = 576 (3x3 -> 3x3 with kernel_size=3, padding=1)
        self.fc_shared = nn.Linear(64 * 3 * 3, 128)

        # Value head
        self.fc_value1 = nn.Linear(128, 32)
        self.fc_value2 = nn.Linear(32, 1)

        # Policy head
        self.fc_policy = nn.Linear(128, cfg.ACTION_SIZE)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, 3, 3, 3) - 3 planes: current player, opponent, empty
        # Conv layers
        x = self.relu(self.conv1(x))  # (batch, 32, 3, 3)
        x = self.relu(self.conv2(x))  # (batch, 64, 3, 3)

        # Flatten for fully connected
        x = x.view(-1, 64 * 3 * 3)  # (batch, 576)
        x = self.dropout(self.relu(self.fc_shared(x)))  # (batch, 128)

        # Value head
        value = self.relu(self.fc_value1(x))  # (batch, 32)
        value = torch.tanh(self.fc_value2(value))  # (batch, 1) in [-1, 1]

        # Policy head
        policy = self.fc_policy(x)  # (batch, 9)

        return value, policy