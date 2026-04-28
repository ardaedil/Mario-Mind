from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    agent_type: str = "dqn"
    episodes: int = 10
    max_steps_per_episode: int = 2000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 100000
    replay_buffer_size: int = 50000
    batch_size: int = 32
    target_sync_freq: int = 1000
    frame_stack: int = 4
    frame_skip: int = 4
    reward_function: str = "default"
    action_space: str = "SIMPLE_MOVEMENT"
    checkpoint_frequency: int = 25
    evaluation_frequency: int = 50
    seed: int = 42
    env_id: str = "SuperMarioBros-1-1-v0"
    output_dir: str = "results/runs/default"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return asdict(cfg)
