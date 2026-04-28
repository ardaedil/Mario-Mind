"""Reward shaping functions for controlled ablation studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class RewardFn(Protocol):
    def __call__(self, env_reward: float, info: dict, prev_info: dict | None, done: bool) -> float: ...


@dataclass
class DefaultReward:
    def __call__(self, env_reward: float, info: dict, prev_info: dict | None, done: bool) -> float:
        return float(env_reward)


@dataclass
class ProgressOnlyReward:
    idle_penalty: float = -0.01

    def __call__(self, env_reward: float, info: dict, prev_info: dict | None, done: bool) -> float:
        prev_x = 0 if prev_info is None else prev_info.get("x_pos", 0)
        x = info.get("x_pos", prev_x)
        delta = x - prev_x
        return float(delta if delta != 0 else self.idle_penalty)


@dataclass
class ProgressDeathPenaltyReward:
    death_penalty: float = -25.0
    time_penalty: float = -0.01

    def __call__(self, env_reward: float, info: dict, prev_info: dict | None, done: bool) -> float:
        base = ProgressOnlyReward()(env_reward, info, prev_info, done) + self.time_penalty
        dead = info.get("dead", False) or (done and not info.get("flag_get", False))
        return base + (self.death_penalty if dead else 0.0)


@dataclass
class SparseReward:
    finish_reward: float = 200.0
    death_penalty: float = -50.0
    time_penalty: float = -0.01

    def __call__(self, env_reward: float, info: dict, prev_info: dict | None, done: bool) -> float:
        if info.get("flag_get", False):
            return self.finish_reward
        if done and not info.get("flag_get", False):
            return self.death_penalty
        return self.time_penalty


@dataclass
class CoinScoreAwareReward:
    death_penalty: float = -25.0
    time_penalty: float = -0.01
    score_scale: float = 0.001
    coin_scale: float = 0.5

    def __call__(self, env_reward: float, info: dict, prev_info: dict | None, done: bool) -> float:
        prev_info = prev_info or {}
        x_delta = info.get("x_pos", 0) - prev_info.get("x_pos", 0)
        score_delta = info.get("score", 0) - prev_info.get("score", 0)
        coin_delta = info.get("coins", 0) - prev_info.get("coins", 0)
        dead = info.get("dead", False) or (done and not info.get("flag_get", False))
        shaped = x_delta + self.score_scale * score_delta + self.coin_scale * coin_delta + self.time_penalty
        return shaped + (self.death_penalty if dead else 0.0)


def build_reward_function(name: str) -> RewardFn:
    mapping = {
        "default": DefaultReward(),
        "progress_only": ProgressOnlyReward(),
        "progress_death": ProgressDeathPenaltyReward(),
        "sparse": SparseReward(),
        "coin_score": CoinScoreAwareReward(),
    }
    if name not in mapping:
        raise ValueError(f"Unknown reward function '{name}'. Valid: {list(mapping)}")
    return mapping[name]
