import csv
from pathlib import Path

from mariomind.agents.ddqn_agent import DoubleDQNAgent
from mariomind.agents.dqn_agent import DQNAgent, DQNConfig
from mariomind.training.train import train
from mariomind.utils.config import TrainConfig


def _fake_state(shape=(4, 84, 84)):
    return [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]


def test_dqn_select_action():
    agent = DQNAgent(action_dim=3, state_shape=(4, 84, 84), device="cpu", cfg=DQNConfig())
    action = agent.select_action(_fake_state(), {"x_pos": 0})
    assert action in {0, 1, 2}


def test_dqn_train_step_with_fake_data():
    cfg = DQNConfig(batch_size=4, min_buffer_to_train=4)
    agent = DQNAgent(action_dim=3, state_shape=(4, 84, 84), device="cpu", cfg=cfg)
    s = _fake_state()
    for i in range(5):
        agent.observe(s, i % 3, 1.0, s, False)
    logs = agent.train_step()
    assert "loss" in logs and "epsilon" in logs


def test_ddqn_train_step_with_fake_data():
    cfg = DQNConfig(batch_size=4, min_buffer_to_train=4)
    agent = DoubleDQNAgent(action_dim=3, state_shape=(4, 84, 84), device="cpu", cfg=cfg)
    s = _fake_state()
    for i in range(5):
        agent.observe(s, i % 3, 1.0, s, False)
    logs = agent.train_step()
    assert "loss" in logs


def test_checkpoint_roundtrip(tmp_path: Path):
    agent = DQNAgent(action_dim=3, state_shape=(4, 84, 84), device="cpu", cfg=DQNConfig())
    ckpt = tmp_path / "agent.pt"
    agent.save(ckpt)
    loaded = DQNAgent(action_dim=3, state_shape=(4, 84, 84), device="cpu", cfg=DQNConfig())
    loaded.load(ckpt)
    assert loaded.agent_type == "dqn"


def test_training_metrics_include_loss_epsilon(tmp_path: Path):
    out = tmp_path / "run"
    cfg = TrainConfig(agent_type="dqn", episodes=3, output_dir=str(out), max_steps_per_episode=80, batch_size=8, replay_buffer_size=100)
    train(cfg, env_kind="dummy")
    with open(out / "metrics.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert "loss" in rows[0]
    assert "epsilon" in rows[0]


def test_epsilon_floor():
    cfg = DQNConfig(epsilon_start=1.0, epsilon_end=0.2, epsilon_decay_steps=2)
    agent = DQNAgent(action_dim=2, state_shape=(4, 84, 84), device="cpu", cfg=cfg)
    for _ in range(20):
        agent._update_epsilon()
    assert agent.epsilon >= 0.2
