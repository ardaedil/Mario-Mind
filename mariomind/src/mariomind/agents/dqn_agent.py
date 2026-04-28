from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

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
    min_buffer_to_train: int = 1000
    huber_loss: bool = True


class DQNAgent(BaseAgent):
    def __init__(self, action_dim: int, state_shape: tuple[int, ...], device: torch.device, cfg: DQNConfig, dueling: bool = False):
        super().__init__(action_dim)
        self.cfg = cfg
        self.device = device
        self.step_count = 0
        self.epsilon = cfg.epsilon_start
        self.online_net = QNetwork(state_shape[0], action_dim, dueling=dueling).to(device)
        self.target_net = QNetwork(state_shape[0], action_dim, dueling=dueling).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optim = Adam(self.online_net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, state_shape)

    def _update_epsilon(self) -> None:
        decay = (self.cfg.epsilon_start - self.cfg.epsilon_end) / max(self.cfg.epsilon_decay_steps, 1)
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon - decay)

    def select_action(self, state: np.ndarray, info: dict | None = None) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.action_dim))
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def observe(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self.step_count += 1
        self._update_epsilon()

    def _compute_targets(self, next_q_target: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        max_next_q = next_q_target.max(dim=1).values
        return rewards + (1 - dones) * self.cfg.gamma * max_next_q

    def train_step(self) -> dict:
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.min_buffer_to_train):
            return {}
        batch = self.buffer.sample(self.cfg.batch_size)
        s = torch.from_numpy(batch.states).to(self.device)
        a = torch.from_numpy(batch.actions).to(self.device)
        r = torch.from_numpy(batch.rewards).to(self.device)
        ns = torch.from_numpy(batch.next_states).to(self.device)
        d = torch.from_numpy(batch.dones).to(self.device)

        q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_target = self.target_net(ns)
            target = self._compute_targets(next_q_target, r, d)

        loss = F.smooth_l1_loss(q, target) if self.cfg.huber_loss else F.mse_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optim.step()

        if self.step_count % self.cfg.target_sync_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {"loss": float(loss.item()), "epsilon": self.epsilon}

    def save(self, path: str | Path) -> None:
        ckpt = {
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optim": self.optim.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }
        torch.save(ckpt, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optim.load_state_dict(ckpt["optim"])
        self.epsilon = float(ckpt.get("epsilon", self.cfg.epsilon_end))
        self.step_count = int(ckpt.get("step_count", 0))
