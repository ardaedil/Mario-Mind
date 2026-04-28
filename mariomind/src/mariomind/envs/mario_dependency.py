"""Shared helpers for Mario dependency checks and guidance."""

from __future__ import annotations

from importlib.util import find_spec

INSTALL_COMMAND = "pip install gymnasium gym-super-mario-bros nes-py opencv-python"
INSTALL_GUIDANCE = (
    "Missing Mario dependencies. Install with:\n"
    f"  {INSTALL_COMMAND}\n"
    "Then rerun: python scripts/smoke_test_mario_env.py --env-id SuperMarioBros-1-1-v0 --action-space right_only"
)


def has_module(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def missing_mario_modules() -> list[str]:
    required = ["gym_super_mario_bros", "nes_py", "gym_super_mario_bros.actions"]
    return [name for name in required if not has_module(name)]
