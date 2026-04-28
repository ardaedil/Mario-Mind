#!/usr/bin/env python3
"""Check local readiness for real Super Mario Bros World 1-1 interaction."""

from __future__ import annotations

import platform
import sys
from importlib import import_module

from mariomind.envs.compat import reset_compat, step_compat
from mariomind.envs.mario_dependency import INSTALL_GUIDANCE, has_module, missing_mario_modules


def main() -> int:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    print(f"gym_super_mario_bros importable: {has_module('gym_super_mario_bros')}")
    print(f"nes_py importable: {has_module('nes_py')}")
    print(f"gymnasium importable: {has_module('gymnasium')}")
    print(f"gym importable: {has_module('gym')}")

    missing = missing_mario_modules()
    if missing:
        print(f"Missing modules: {', '.join(missing)}")
        print(INSTALL_GUIDANCE)
        return 1

    gym_super_mario_bros = import_module("gym_super_mario_bros")
    JoypadSpace = import_module("nes_py.wrappers").JoypadSpace
    actions_mod = import_module("gym_super_mario_bros.actions")

    env_id = "SuperMarioBros-1-1-v0"
    print(f"Creating env: {env_id}")
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, actions_mod.RIGHT_ONLY)

    obs, info = reset_compat(env)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = step_compat(env, action)

    shape = getattr(obs, "shape", None)
    print(f"Observation shape: {shape}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    keys = sorted(list(info.keys())) if isinstance(info, dict) else []
    print(f"Info keys: {keys}")
    for key in ["x_pos", "flag_get", "time", "score", "coins", "life", "status"]:
        print(f"Has {key}: {key in info}")

    env.close()
    print("Mario setup check completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
