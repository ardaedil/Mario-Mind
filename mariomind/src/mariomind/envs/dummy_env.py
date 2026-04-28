"""Dummy platformer environment for end-to-end smoke testing without Mario deps."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .action_spaces import get_action_space_config
from .reward_functions import RewardFn, build_reward_function


@dataclass
class DummyEnvConfig:
    frame_skip: int = 4
    frame_stack: int = 4
    action_space: str = "SIMPLE_MOVEMENT"
    reward_function: str = "default"
    max_steps: int = 500
    goal_x: int = 1200
    death_x: int = 950


def _make_frame(x_pos: float, goal_x: int, last_action: int) -> list[list[int]]:
    frame = [[0 for _ in range(84)] for _ in range(84)]
    marker_col = int(max(0, min(83, (x_pos / max(goal_x, 1)) * 83)))
    for r in range(84):
        frame[r][marker_col] = 180
    for r in range(70, 84):
        for c in range(84):
            frame[r][c] = 80
    if last_action == 2:
        for r in range(40, 48):
            for c in range(marker_col, min(marker_col + 4, 84)):
                frame[r][c] = 255
    return frame


class DummyPlatformerEnv:
    """Mario-like adapter for smoke tests; outputs stacked 84x84 grayscale frames."""

    def __init__(self, cfg: DummyEnvConfig):
        self.cfg = cfg
        self.action_cfg = get_action_space_config(cfg.action_space)
        self.reward_fn: RewardFn = build_reward_function(cfg.reward_function)
        self.prev_info: dict | None = None
        self.episode_reward = 0.0
        self.max_x = 0.0
        self.time_alive = 0
        self._frames: list[list[list[int]]] = []
        self.rng = random.Random(0)
        self._x_pos = 0.0
        self._score = 0
        self._coins = 0
        self._step = 0
        self._flag_get = False
        self._dead = False
        self._last_action = 0

    @property
    def observation_space(self):
        return type("ObsSpace", (), {"shape": (self.cfg.frame_stack, 84, 84)})

    @property
    def action_space_n(self) -> int:
        return len(self.action_cfg.actions)

    def _info(self) -> dict:
        return {
            "x_pos": int(self._x_pos),
            "flag_get": bool(self._flag_get),
            "dead": bool(self._dead),
            "time": int(self.cfg.max_steps - self._step),
            "score": int(self._score),
            "coins": int(self._coins),
        }

    def _raw_reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = random.Random(seed)
        self._x_pos = 0.0
        self._score = 0
        self._coins = 0
        self._step = 0
        self._flag_get = False
        self._dead = False
        self._last_action = 0
        obs = _make_frame(self._x_pos, self.cfg.goal_x, self._last_action)
        return obs, self._info()

    def _raw_step(self, action: int):
        self._step += 1
        self._last_action = action
        if action == 0:
            dx = 0.0
        elif action == 1:
            dx = 5.0
        elif action == 2:
            dx = 7.0
        else:
            dx = -2.0
        if self.rng.random() < 0.05:
            dx = max(dx - 4.0, -4.0)
        prev_x = self._x_pos
        self._x_pos = max(0.0, self._x_pos + dx)
        progress = self._x_pos - prev_x
        reward = progress * 0.5 - 0.02
        if progress > 0 and self.rng.random() < 0.03:
            self._coins += 1
            self._score += 100
            reward += 0.2
        if self._x_pos >= self.cfg.goal_x:
            self._flag_get = True
            reward += 100.0
        death_risk = 0.06 if self._x_pos > self.cfg.death_x else (0.015 if self._x_pos > (self.cfg.goal_x * 0.5) else 0.0)
        if self.rng.random() < death_risk:
            self._dead = True
            reward -= 50.0
        terminated = self._flag_get or self._dead
        truncated = self._step >= self.cfg.max_steps
        obs = _make_frame(self._x_pos, self.cfg.goal_x, self._last_action)
        return obs, float(reward), terminated, truncated, self._info()

    def reset(self, seed: int | None = None):
        obs, info = self._raw_reset(seed=seed)
        self._frames = [obs for _ in range(self.cfg.frame_stack)]
        self.prev_info = info
        self.episode_reward = 0.0
        self.max_x = float(info.get("x_pos", 0.0))
        self.time_alive = 0
        return list(self._frames), info

    def step(self, action_idx: int):
        mapped_action = self.action_cfg.actions[action_idx]
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self.cfg.frame_skip):
            obs, env_reward, terminated, truncated, info = self._raw_step(mapped_action)
            total_reward += env_reward
            if terminated or truncated:
                break
        assert obs is not None
        self._frames.pop(0)
        self._frames.append(obs)
        done = terminated or truncated
        reward = self.reward_fn(total_reward, info, self.prev_info, done)
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
        return list(self._frames), reward, terminated, truncated, info

    def render(self):
        return self._frames[-1] if self._frames else _make_frame(self._x_pos, self.cfg.goal_x, self._last_action)

    def close(self):
        return None
