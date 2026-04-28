from __future__ import annotations

from .base import BaseAgent


class ReflexAgent(BaseAgent):
    """Simple heuristic: move right, jump periodically, jump on stalls."""

    def __init__(self, action_dim: int, right_idx: int = 0, jump_idx: int | None = None, jump_period: int = 18):
        super().__init__(action_dim)
        self.right_idx = min(max(right_idx, 0), action_dim - 1)
        self.jump_idx = self.right_idx if jump_idx is None else min(max(jump_idx, 0), action_dim - 1)
        self.jump_period = jump_period
        self.steps = 0
        self.last_x = 0
        self.stall_steps = 0

    def select_action(self, state, info: dict | None = None) -> int:
        self.steps += 1
        x_pos = 0 if info is None else int(info.get("x_pos", 0))
        if x_pos <= self.last_x:
            self.stall_steps += 1
        else:
            self.stall_steps = 0
        self.last_x = x_pos

        if self.steps % self.jump_period == 0:
            return self.jump_idx
        if self.stall_steps >= 4:
            self.stall_steps = 0
            return self.jump_idx
        if info and info.get("time", 400) < 80 and self.steps % 5 == 0:
            return self.jump_idx
        return self.right_idx
