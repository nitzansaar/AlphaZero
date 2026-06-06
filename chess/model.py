import torch
from torch import nn
from config import Config as cfg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class NeuralNetwork(nn.Module):
    """AlphaGo Zero style residual tower with policy and value heads.

    Input:  (batch, NUM_INPUT_PLANES, 8, 8)
    Output: (value in [-1, 1], policy_logits over ACTION_SIZE)
    """

    def __init__(self):
        super().__init__()
        c = cfg.NUM_CHANNELS

        # Input convolution.
        self.conv_in = nn.Conv2d(cfg.NUM_INPUT_PLANES, c, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

        # Residual tower.
        self.res_blocks = nn.ModuleList([ResidualBlock(c) for _ in range(cfg.NUM_RES_BLOCKS)])

        # Policy head.
        self.policy_conv = nn.Conv2d(c, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, cfg.ACTION_SIZE)

        # Value head.
        self.value_conv = nn.Conv2d(c, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head.
        p = self.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)

        # Value head.
        v = self.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = self.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return value, policy
