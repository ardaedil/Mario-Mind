from mariomind.classical.astar import astar
from mariomind.classical.bfs import bfs
from mariomind.classical.search_world import SearchWorld
from mariomind.utils.config import TrainConfig, to_dict
from mariomind.agents.dqn_agent import DQNConfig
from mariomind.agents.dqn_agent import DQNAgent
from mariomind.utils.device import get_device


def test_bfs_astar_paths_exist():
    world = SearchWorld(length=10, gaps=set(), blocked=set())
    assert len(bfs(world)) > 0
    assert len(astar(world)) > 0


def test_config_to_dict():
    cfg = TrainConfig()
    d = to_dict(cfg)
    assert d["agent_type"] == "dqn"


def test_epsilon_decay_not_below_min():
    agent = DQNAgent(action_dim=3, state_shape=(4, 84, 84), device=get_device(), cfg=DQNConfig(epsilon_decay_steps=2))
    for _ in range(10):
        agent._update_epsilon()
    assert agent.epsilon >= agent.cfg.epsilon_end
