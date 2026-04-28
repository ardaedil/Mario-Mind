from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random

from .base import BaseAgent
from .networks import QNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 50000
    target_sync_freq: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 100000
    min_buffer_to_train: int = 32
    huber_loss: bool = True


class DQNAgent(BaseAgent):
    """Dependency-light DQN-style scaffold for smoke testing pipeline wiring."""

    def __init__(self, action_dim: int, state_shape: tuple[int, ...], device, cfg: DQNConfig, dueling: bool = False):
        super().__init__(action_dim)
        self.cfg = cfg
        self.device = str(device)
        self.state_shape = tuple(state_shape)
        self.step_count = 0
        self.epsilon = cfg.epsilon_start
        self.agent_type = "dqn"
        self.online_net = QNetwork(state_shape[0], action_dim, dueling=dueling)
        self.target_net = QNetwork(state_shape[0], action_dim, dueling=dueling)
        self.buffer = ReplayBuffer(cfg.buffer_size, state_shape)

    def _update_epsilon(self) -> None:
        decay = (self.cfg.epsilon_start - self.cfg.epsilon_end) / max(self.cfg.epsilon_decay_steps, 1)
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon - decay)

    def select_action(self, state, info: dict | None = None) -> int:
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_dim)
        x_pos = 0 if info is None else info.get("x_pos", 0)
        return 2 if (x_pos % 40 == 0 and self.action_dim > 2) else min(1, self.action_dim - 1)

    def observe(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self.step_count += 1
        self._update_epsilon()

    def train_step(self) -> dict:
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.min_buffer_to_train):
            return {}
        batch = self.buffer.sample(self.cfg.batch_size)
        avg_reward = sum(batch.rewards) / max(len(batch.rewards), 1)
        pseudo_loss = max(0.0, 1.0 - avg_reward * 0.01)
        return {"loss": float(pseudo_loss), "epsilon": self.epsilon, "buffer_size": len(self.buffer)}

    def save(self, path: str | Path) -> None:
        payload = {
            "agent_type": self.agent_type,
            "action_size": self.action_dim,
            "observation_shape": self.state_shape,
            "config": asdict(self.cfg),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.epsilon = float(payload.get("epsilon", self.cfg.epsilon_end))
        self.step_count = int(payload.get("step_count", 0))
        self.agent_type = payload.get("agent_type", self.agent_type)
