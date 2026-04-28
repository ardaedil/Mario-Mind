from mariomind.envs.dummy_env import DummyEnvConfig, DummyPlatformerEnv


def test_dummy_env_shapes_and_info():
    env = DummyPlatformerEnv(DummyEnvConfig(frame_stack=4, frame_skip=2, max_steps=20))
    obs, info = env.reset(seed=123)
    assert len(obs) == 4
    assert len(obs[0]) == 84
    assert len(obs[0][0]) == 84
    assert {"x_pos", "flag_get", "dead", "time", "score", "coins"}.issubset(info)
    _, reward, terminated, truncated, info = env.step(1)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "max_x" in info
    env.close()
