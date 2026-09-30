"""Q-networks: map board planes (C, size, size) to one Q-value per cell, Q(s, a) for every move a.

Values are for the player to move and squashed to [-1, 1] with tanh, the range of the game's returns.
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class ConvQNet(nn.Module):
    """Fully convolutional: the same weights look at every part of the board, and it works for any board size.

    With `dueling`, Q(s, a) = V(s) + A(s, a) - mean_a A(s, a), where V comes from a global-pooled head.
    """

    def __init__(self, in_channels: int, channels: int = 64, blocks: int = 3, dueling: bool = False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.advantage = nn.Conv2d(channels, 1, 1)
        self.dueling = dueling
        if dueling:
            self.value = nn.Sequential(nn.Linear(channels, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        q = self.advantage(h).flatten(1)
        if self.dueling:
            v = self.value(h.mean(dim=(2, 3)))
            q = v + q - q.mean(dim=1, keepdim=True)
        return torch.tanh(q)


class MLPQNet(nn.Module):
    """Fully connected baseline: every cell has its own weights, so nothing learned in one place transfers to another."""

    def __init__(self, in_channels: int, size: int, hidden: int = 512, layers: int = 3):
        super().__init__()
        dims = [in_channels * size * size] + [hidden] * layers
        mods = []
        for a, b in zip(dims, dims[1:]):
            mods += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(nn.Flatten(), *mods, nn.Linear(hidden, size * size))

    def forward(self, x):
        return torch.tanh(self.net(x))


def make_qnet(arch: str, in_channels: int, size: int, dueling: bool = False, channels: int = 64,
              blocks: int = 3) -> nn.Module:
    if arch == 'cnn':
        return ConvQNet(in_channels, channels, blocks, dueling)
    if arch == 'mlp':
        if dueling:
            raise ValueError('dueling is only implemented for the cnn')
        return MLPQNet(in_channels, size)
    raise ValueError(f'unknown arch {arch!r}')
