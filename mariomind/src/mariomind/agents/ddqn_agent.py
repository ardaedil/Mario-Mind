from __future__ import annotations

from .dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN scaffold; currently reuses smoke-test DQN core."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = "ddqn"
