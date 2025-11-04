import torch
from torch import nn
from config import Config as cfg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Shared fully connected layer
        # Input size: 128 channels * 3 * 3 = 1152
        self.fc_shared = nn.Linear(128 * 3 * 3, 256)
        
        # Value head
        self.fc_value1 = nn.Linear(256, 64)
        self.fc_value2 = nn.Linear(64, 1)
        
        # Policy head
        self.fc_policy = nn.Linear(256, cfg.ACTION_SIZE)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, 3, 3, 3) - 3 planes: current player, opponent, empty
        # Conv layers
        x = self.relu(self.conv1(x))  # (batch, 64, 3, 3)
        x = self.relu(self.conv2(x))  # (batch, 128, 3, 3)
        
        # Flatten for fully connected
        x = x.view(-1, 128 * 3 * 3)  # (batch, 1152)
        x = self.dropout(self.relu(self.fc_shared(x)))  # (batch, 256)
        
        # Value head
        value = self.relu(self.fc_value1(x))  # (batch, 64)
        value = torch.tanh(self.fc_value2(value))  # (batch, 1) in [-1, 1]
        
        # Policy head
        policy = self.fc_policy(x)  # (batch, 9)
        
        return value, policy