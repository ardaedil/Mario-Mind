#!/usr/bin/env python3
"""Smoke test for real gym-super-mario-bros environment wiring."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys

from mariomind.envs.compat import reset_compat, step_compat
from mariomind.envs.mario_dependency import INSTALL_GUIDANCE, missing_mario_modules
from mariomind.envs.mario_env import MarioEnv, MarioEnvConfig


def _shape_of(obs):
    shape = getattr(obs, "shape", None)
    if shape is not None:
        return shape
    if isinstance(obs, list):
        if obs and isinstance(obs[0], list):
            if obs[0] and isinstance(obs[0][0], list):
                return (len(obs), len(obs[0]), len(obs[0][0]))
            return (len(obs), len(obs[0]))
        return (len(obs),)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test gym-super-mario-bros import/create/reset/step")
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--action-space", choices=["right_only", "simple", "complex"], default="right_only")
    args = parser.parse_args()

    missing = missing_mario_modules()
    if missing:
        print(f"Missing modules: {', '.join(missing)}")
        print(INSTALL_GUIDANCE)
        return 1

    gym_super_mario_bros = import_module("gym_super_mario_bros")
    JoypadSpace = import_module("nes_py.wrappers").JoypadSpace
    actions_mod = import_module("gym_super_mario_bros.actions")
    RIGHT_ONLY = actions_mod.RIGHT_ONLY
    SIMPLE_MOVEMENT = actions_mod.SIMPLE_MOVEMENT
    COMPLEX_MOVEMENT = actions_mod.COMPLEX_MOVEMENT

    action_map = {
        "right_only": RIGHT_ONLY,
        "simple": SIMPLE_MOVEMENT,
        "complex": COMPLEX_MOVEMENT,
    }

    print(f"Creating raw env: {args.env_id}")
    env = gym_super_mario_bros.make(args.env_id)
    env = JoypadSpace(env, action_map[args.action_space])

    obs, info = reset_compat(env)
    print(f"raw observation type: {type(obs)}")
    print(f"raw observation shape: {_shape_of(obs)}")
    print(f"reset info keys: {sorted(list(info.keys())) if isinstance(info, dict) else []}")

    required_info = ["x_pos", "flag_get", "time", "score", "coins", "life", "status"]
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = step_compat(env, action)
        print(
            f"step={i+1:02d} reward={reward:.3f} terminated={terminated} truncated={truncated} "
            f"info_keys={sorted(list(info.keys())) if isinstance(info, dict) else []}"
        )
        if terminated or truncated:
            break

    if isinstance(info, dict):
        for key in required_info:
            print(f"info contains {key}: {key in info}")

    env.close()

    print("Testing MarioMind wrapper initialization...")
    wrapper_action = {
        "right_only": "RIGHT_ONLY",
        "simple": "SIMPLE_MOVEMENT",
        "complex": "SIMPLE_MOVEMENT",
    }[args.action_space]
    wrapped = MarioEnv(
        MarioEnvConfig(
            env_id=args.env_id,
            action_space=wrapper_action,
            frame_skip=1,
            frame_stack=2,
            reward_function="default",
        )
    )
    w_obs, w_info = wrapped.reset()
    print(f"wrapper observation shape: {_shape_of(w_obs)}")
    print(f"wrapper reset info keys: {sorted(list(w_info.keys())) if isinstance(w_info, dict) else []}")
    wrapped.close()

    print("Mario environment smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
