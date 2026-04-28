from mariomind.envs.compat import reset_compat, step_compat


class FakeOldGymEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return "obs_old"

    def step(self, action):
        self.steps += 1
        return "obs_old_step", 1.5, self.steps > 1, {"x_pos": 10}


class FakeNewGymnasiumEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return "obs_new", {"time": 400}

    def step(self, action):
        self.steps += 1
        return "obs_new_step", 2.0, self.steps > 1, False, {"x_pos": 12}


def test_reset_compat_old():
    obs, info = reset_compat(FakeOldGymEnv())
    assert obs == "obs_old"
    assert info == {}


def test_reset_compat_new():
    obs, info = reset_compat(FakeNewGymnasiumEnv())
    assert obs == "obs_new"
    assert info["time"] == 400


def test_step_compat_old():
    obs, reward, terminated, truncated, info = step_compat(FakeOldGymEnv(), 0)
    assert obs == "obs_old_step"
    assert reward == 1.5
    assert terminated is False
    assert truncated is False
    assert info["x_pos"] == 10


def test_step_compat_new():
    obs, reward, terminated, truncated, info = step_compat(FakeNewGymnasiumEnv(), 0)
    assert obs == "obs_new_step"
    assert reward == 2.0
    assert terminated is False
    assert truncated is False
    assert info["x_pos"] == 12
