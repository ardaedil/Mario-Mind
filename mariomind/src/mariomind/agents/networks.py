from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Classic Atari CNN for stacked 84x84 grayscale frames."""

    def __init__(self, input_channels: int, action_dim: int, dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_dim = 3136
        if dueling:
            self.value = nn.Sequential(nn.Linear(self.fc_dim, 512), nn.ReLU(), nn.Linear(512, 1))
            self.advantage = nn.Sequential(nn.Linear(self.fc_dim, 512), nn.ReLU(), nn.Linear(512, action_dim))
        else:
            self.head = nn.Sequential(nn.Linear(self.fc_dim, 512), nn.ReLU(), nn.Linear(512, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        f = self.features(x)
        if self.dueling:
            v = self.value(f)
            a = self.advantage(f)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.head(f)
