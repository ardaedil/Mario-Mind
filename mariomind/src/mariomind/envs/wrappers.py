"""Observation preprocessing wrappers (grayscale, resize, frame stack)."""

from __future__ import annotations

from collections import deque

import cv2
import gymnasium as gym
import numpy as np


class PreprocessObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(0, 255, shape=(height, width), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, stack_size: int = 4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames: deque[np.ndarray] = deque(maxlen=stack_size)
        base = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(stack_size, *base.shape),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info
