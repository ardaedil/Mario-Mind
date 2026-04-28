from __future__ import annotations

import numpy as np

from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Uniform random baseline."""

    def select_action(self, state: np.ndarray, info: dict | None = None) -> int:
        return int(np.random.randint(0, self.action_dim))
