import torch
from torch import nn
from config import Config as cfg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)

        # Shared fully connected layer
        # Input size: 128 channels * 7 * 7 = 6272 (9x9 -> 8x8 -> 7x7 with kernel_size=4, padding=1)
        self.fc_shared = nn.Linear(128 * 7 * 7, 256)
        
        # Value head
        self.fc_value1 = nn.Linear(256, 64)
        self.fc_value2 = nn.Linear(64, 1)
        
        # Policy head
        self.fc_policy = nn.Linear(256, cfg.ACTION_SIZE)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, 3, 9, 9) - 3 planes: current player, opponent, empty
        # Conv layers
        x = self.relu(self.conv1(x))  # (batch, 64, 8, 8)
        x = self.relu(self.conv2(x))  # (batch, 128, 7, 7)

        # Flatten for fully connected
        x = x.view(-1, 128 * 7 * 7)  # (batch, 6272)
        x = self.dropout(self.relu(self.fc_shared(x)))  # (batch, 256)

        # Value head
        value = self.relu(self.fc_value1(x))  # (batch, 64)
        value = torch.tanh(self.fc_value2(value))  # (batch, 1) in [-1, 1]

        # Policy head
        policy = self.fc_policy(x)  # (batch, 81)

        return value, policy