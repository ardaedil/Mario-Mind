from __future__ import annotations

import torch

from .dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN: online net selects argmax action, target net evaluates it."""

    def _compute_targets(self, next_q_target: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        del next_q_target
        raise RuntimeError("Use train_step override for DDQN target computation")

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
            next_actions = self.online_net(ns).argmax(dim=1, keepdim=True)
            next_q = self.target_net(ns).gather(1, next_actions).squeeze(1)
            target = r + (1 - d) * self.cfg.gamma * next_q

        loss = torch.nn.functional.smooth_l1_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optim.step()

        if self.step_count % self.cfg.target_sync_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return {"loss": float(loss.item()), "epsilon": self.epsilon}
