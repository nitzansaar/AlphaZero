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
    Neural network for 5x5 Go with improved value head.

    Architecture inspired by AlphaGo Zero but scaled down for 5x5:
    - Initial convolution to expand channels
    - Stack of residual blocks for pattern recognition
    - Separate value and policy heads

    Key improvement: Larger value head with more capacity to accurately
    predict game outcomes. Original had only 64 hidden units, now uses 256.
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

        # Improved Value head - significantly more capacity
        # Changed: 32 channels -> 64, 64 hidden -> 256, added extra layer
        self.value_conv = nn.Conv2d(channels, 64, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(64)
        self.value_fc1 = nn.Linear(64 * 5 * 5, 256)  # 800 -> 256 (was 800 -> 64)
        self.value_fc2 = nn.Linear(256, 64)           # Additional layer
        self.value_fc3 = nn.Linear(64, 1)

        # Policy head (unchanged)
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

        # Improved Value head with 3 FC layers
        v = F.relu(self.value_bn(self.value_conv(x)))  # (batch, 64, 5, 5)
        v = v.view(-1, 64 * 5 * 5)  # (batch, 1600)
        v = F.relu(self.value_fc1(v))  # (batch, 256)
        v = F.relu(self.value_fc2(v))  # (batch, 64)
        v = torch.tanh(self.value_fc3(v))  # (batch, 1) in [-1, 1]

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # (batch, 32, 5, 5)
        p = p.view(-1, 32 * 5 * 5)  # (batch, 800)
        p = self.policy_fc(p)  # (batch, 26) - raw logits

        return v, p

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
