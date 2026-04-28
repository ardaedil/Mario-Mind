"""Gym-compatible Mario environment adapter with graceful dependency failures."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .action_spaces import get_action_space_config
from .reward_functions import RewardFn, build_reward_function
from .wrappers import FrameStackWrapper, PreprocessObservationWrapper


@dataclass
class MarioEnvConfig:
    env_id: str = "SuperMarioBros-1-1-v0"
    frame_skip: int = 4
    frame_stack: int = 4
    action_space: str = "SIMPLE_MOVEMENT"
    reward_function: str = "default"
    render_mode: str | None = None


class MarioEnv:
    def __init__(self, cfg: MarioEnvConfig):
        self.cfg = cfg
        self.action_cfg = get_action_space_config(cfg.action_space)
        self.reward_fn: RewardFn = build_reward_function(cfg.reward_function)
        self._env = self._build_env()
        self.prev_info: dict | None = None
        self.episode_reward = 0.0
        self.max_x = 0.0
        self.time_alive = 0

    def _build_env(self) -> gym.Env:
        try:
            env = gym.make(self.cfg.env_id, render_mode=self.cfg.render_mode)
        except Exception as exc:  # graceful error path for missing local env
            raise RuntimeError(
                "Could not create Mario env. Install/enable your local Gym-compatible Mario package "
                f"and verify env id '{self.cfg.env_id}'. Original error: {exc}"
            ) from exc
        env = PreprocessObservationWrapper(env)
        env = FrameStackWrapper(env, stack_size=self.cfg.frame_stack)
        return env

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space_n(self) -> int:
        return len(self.action_cfg.actions)

    def reset(self, seed: int | None = None):
        obs, info = self._env.reset(seed=seed)
        self.prev_info = info
        self.episode_reward = 0.0
        self.max_x = float(info.get("x_pos", 0.0))
        self.time_alive = 0
        return obs, info

    def step(self, action_idx: int):
        mapped_action = self.action_cfg.actions[action_idx]
        total_env_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs: np.ndarray | None = None
        for _ in range(self.cfg.frame_skip):
            obs, env_reward, terminated, truncated, info = self._env.step(mapped_action)
            total_env_reward += env_reward
            if terminated or truncated:
                break
        assert obs is not None
        done = terminated or truncated
        reward = self.reward_fn(total_env_reward, info, self.prev_info, done)
        self.prev_info = info
        self.episode_reward += reward
        self.max_x = max(self.max_x, float(info.get("x_pos", 0.0)))
        self.time_alive += 1
        info = {
            **info,
            "episode_reward": self.episode_reward,
            "max_x": self.max_x,
            "time_alive": self.time_alive,
            "success": bool(info.get("flag_get", False)),
            "dead": bool(info.get("dead", False)),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
