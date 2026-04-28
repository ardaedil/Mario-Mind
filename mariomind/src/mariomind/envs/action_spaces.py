"""Action-space presets for Mario environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpaceConfig:
    name: str
    actions: list[int]
    right_action_index: int
    jump_action_index: int


ACTION_SPACE_PRESETS: dict[str, ActionSpaceConfig] = {
    "RIGHT_ONLY": ActionSpaceConfig("RIGHT_ONLY", actions=[0], right_action_index=0, jump_action_index=0),
    "SIMPLE_MOVEMENT": ActionSpaceConfig("SIMPLE_MOVEMENT", actions=[0, 1, 2, 3], right_action_index=1, jump_action_index=3),
    "CUSTOM_SMALL_ACTION_SPACE": ActionSpaceConfig(
        "CUSTOM_SMALL_ACTION_SPACE", actions=[0, 1, 2], right_action_index=1, jump_action_index=2
    ),
}


def get_action_space_config(name: str) -> ActionSpaceConfig:
    if name not in ACTION_SPACE_PRESETS:
        raise ValueError(f"Unknown action space: {name}. Available: {list(ACTION_SPACE_PRESETS)}")
    return ACTION_SPACE_PRESETS[name]
