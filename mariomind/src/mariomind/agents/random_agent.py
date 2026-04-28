from __future__ import annotations

import random

from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Uniform random baseline."""

    def select_action(self, state, info: dict | None = None) -> int:
        return random.randrange(0, self.action_dim)
