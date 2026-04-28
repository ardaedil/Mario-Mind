from __future__ import annotations

from .base import BaseAgent


class AlwaysRightAgent(BaseAgent):
    """Always chooses rightward action index configured as 0/1 in small spaces."""

    def __init__(self, action_dim: int, right_action_idx: int = 0):
        super().__init__(action_dim)
        self.right_action_idx = min(max(right_action_idx, 0), action_dim - 1)

    def select_action(self, state, info: dict | None = None) -> int:
        return self.right_action_idx
