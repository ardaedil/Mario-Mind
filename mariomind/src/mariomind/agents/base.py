from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    @abstractmethod
    def select_action(self, state: np.ndarray, info: dict | None = None) -> int:
        ...

    def observe(self, *args, **kwargs) -> None:
        return None

    def train_step(self) -> dict:
        return {}
