import torch
from torch import nn
import torch.nn.functional as F
from config import Config as cfg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class ResBlock(nn.Module):
    """Residual block with batch normalization."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class NeuralNetwork(nn.Module):
    """
    Improved neural network for 5x5 Go with residual blocks.

    Architecture inspired by AlphaGo Zero but scaled down for 5x5:
    - Initial convolution to expand channels
    - Stack of residual blocks for pattern recognition
    - Separate value and policy heads
    """
    def __init__(self, num_res_blocks=6, channels=128):
        super(NeuralNetwork, self).__init__()

        self.num_res_blocks = num_res_blocks
        self.channels = channels

        # Initial convolution: 3 input planes -> channels
        self.conv_init = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(num_res_blocks)
        ])

        # Value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 5 * 5, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 5 * 5, cfg.ACTION_SIZE)

    def forward(self, x):
        # x shape: (batch, 3, 5, 5) - 3 planes: current player, opponent, empty

        # Initial convolution
        x = F.relu(self.bn_init(self.conv_init(x)))  # (batch, channels, 5, 5)

        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))  # (batch, 32, 5, 5)
        v = v.view(-1, 32 * 5 * 5)  # (batch, 800)
        v = F.relu(self.value_fc1(v))  # (batch, 64)
        v = torch.tanh(self.value_fc2(v))  # (batch, 1) in [-1, 1]

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # (batch, 32, 5, 5)
        p = p.view(-1, 32 * 5 * 5)  # (batch, 800)
        p = self.policy_fc(p)  # (batch, 26) - raw logits

        return v, p


# Keep old model for reference/loading old checkpoints
class NeuralNetworkLegacy(nn.Module):
    """Legacy model for loading old checkpoints."""
    def __init__(self):
        super(NeuralNetworkLegacy, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc_shared = nn.Linear(128 * 5 * 5, 256)
        self.fc_value1 = nn.Linear(256, 64)
        self.fc_value2 = nn.Linear(64, 1)
        self.fc_policy = nn.Linear(256, cfg.ACTION_SIZE)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 128 * 5 * 5)
        x = self.dropout(self.relu(self.fc_shared(x)))
        value = self.relu(self.fc_value1(x))
        value = torch.tanh(self.fc_value2(value))
        policy = self.fc_policy(x)
        return value, policy
