from types import SimpleNamespace

import mariomind.envs.mario_env as mario_env_mod
from mariomind.envs.mario_env import MarioEnv, MarioEnvConfig


class FakeRawEnv:
    def __init__(self):
        self.observation_space = SimpleNamespace(shape=(84, 84))

    def reset(self, seed=None):
        return [[0] * 84 for _ in range(84)], {"x_pos": 0, "flag_get": False}

    def step(self, action):
        return [[0] * 84 for _ in range(84)], 1.0, False, False, {"x_pos": 1, "flag_get": False}

    def close(self):
        return None

    def render(self):
        return None


class FakeJoypad:
    def __init__(self, env, movement):
        self.env = env
        self.movement = movement
        self.observation_space = env.observation_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()


class _PassPreprocess:
    def __new__(cls, env, *args, **kwargs):
        return env


class _PassStack:
    def __new__(cls, env, *args, **kwargs):
        env.observation_space = SimpleNamespace(shape=(4, 84, 84))
        return env


def test_mario_env_uses_gym_super_make(monkeypatch):
    calls = {"gym_super_make": 0, "gymnasium_make": 0}

    def fake_import(name):
        if name == "gym_super_mario_bros":
            return SimpleNamespace(make=lambda env_id, **kwargs: _track_super_make(calls))
        if name == "nes_py.wrappers":
            return SimpleNamespace(JoypadSpace=FakeJoypad)
        if name == "gym_super_mario_bros.actions":
            return SimpleNamespace(RIGHT_ONLY=[["right"]], SIMPLE_MOVEMENT=[["noop"], ["right"], ["A"]])
        if name == "mariomind.envs.wrappers":
            return SimpleNamespace(PreprocessObservationWrapper=_PassPreprocess, FrameStackWrapper=_PassStack)
        if name == "gymnasium":
            return SimpleNamespace(make=lambda *args, **kwargs: _track_gym_make(calls))
        raise AssertionError(name)

    monkeypatch.setattr(mario_env_mod, "missing_mario_modules", lambda: [])
    monkeypatch.setattr(mario_env_mod.importlib, "import_module", fake_import)

    env = MarioEnv(MarioEnvConfig(env_id="SuperMarioBros-1-1-v0", action_space="RIGHT_ONLY"))
    env.reset(seed=1)
    env.step(0)
    assert calls["gym_super_make"] == 1
    assert calls["gymnasium_make"] == 0


def _track_super_make(calls):
    calls["gym_super_make"] += 1
    return FakeRawEnv()


def _track_gym_make(calls):
    calls["gymnasium_make"] += 1
    return FakeRawEnv()


def test_mario_env_missing_dependency_message(monkeypatch):
    monkeypatch.setattr(mario_env_mod, "missing_mario_modules", lambda: ["gym_super_mario_bros"])
    try:
        MarioEnv(MarioEnvConfig())
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        msg = str(exc)
        assert "Missing Mario dependencies" in msg
        assert "pip install gymnasium gym-super-mario-bros nes-py opencv-python" in msg
