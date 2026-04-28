"""Educational, lightweight network scaffold used for smoke tests.

This fallback avoids external ML dependencies in restricted environments.
"""

from __future__ import annotations


class QNetwork:
    def __init__(self, input_channels: int, action_dim: int, dueling: bool = False):
        self.input_channels = input_channels
        self.action_dim = action_dim
        self.dueling = dueling

    def forward(self, x):
        batch = len(x) if isinstance(x, list) else 1
        return [[0.0 for _ in range(self.action_dim)] for _ in range(batch)]

    __call__ = forward
