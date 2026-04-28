from __future__ import annotations

from .base import BaseAgent


class ReflexAgent(BaseAgent):
    """Simple heuristic: mostly right, periodic jumps."""

    def __init__(self, action_dim: int, right_idx: int = 0, jump_idx: int | None = None, jump_period: int = 20):
        super().__init__(action_dim)
        self.right_idx = min(max(right_idx, 0), action_dim - 1)
        self.jump_idx = self.right_idx if jump_idx is None else min(max(jump_idx, 0), action_dim - 1)
        self.jump_period = jump_period
        self.steps = 0

    def select_action(self, state, info: dict | None = None) -> int:
        self.steps += 1
        if self.steps % self.jump_period == 0:
            return self.jump_idx
        if info and info.get("x_pos", 0) <= info.get("max_x", 0) and self.steps % 7 == 0:
            return self.jump_idx
        return self.right_idx
