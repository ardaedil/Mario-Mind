from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class Batch:
    states: list
    actions: list[int]
    rewards: list[float]
    next_states: list
    dones: list[float]


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple[int, ...]):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state_shape = state_shape
        self.states = [None] * capacity
        self.actions = [0] * capacity
        self.rewards = [0.0] * capacity
        self.next_states = [None] * capacity
        self.dones = [0.0] * capacity

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        i = self.ptr
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state
        self.dones[i] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = [random.randrange(0, self.size) for _ in range(batch_size)]
        return Batch(
            states=[self.states[i] for i in idx],
            actions=[self.actions[i] for i in idx],
            rewards=[self.rewards[i] for i in idx],
            next_states=[self.next_states[i] for i in idx],
            dones=[self.dones[i] for i in idx],
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """TODO: prioritized replay scaffold (alpha/beta importance sampling)."""

    pass
