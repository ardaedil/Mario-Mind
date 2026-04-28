"""Compatibility helpers for Gym and Gymnasium reset/step signatures."""

from __future__ import annotations


def reset_compat(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, info
    return out, {}


def step_compat(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, info
    raise ValueError("Unsupported step return signature. Expected 4-tuple (Gym) or 5-tuple (Gymnasium).")
