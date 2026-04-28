from mariomind.envs.reward_functions import ProgressDeathPenaltyReward, ProgressOnlyReward, SparseReward


def test_progress_only_reward():
    r = ProgressOnlyReward()
    assert r(0.0, {"x_pos": 10}, {"x_pos": 5}, False) > 0
    assert r(0.0, {"x_pos": 5}, {"x_pos": 5}, False) < 0


def test_sparse_reward():
    r = SparseReward()
    assert r(0.0, {"flag_get": True}, None, True) > 0
    assert r(0.0, {"flag_get": False}, None, True) < 0


def test_progress_death_penalty_reward():
    r = ProgressDeathPenaltyReward()
    alive = r(0.0, {"x_pos": 10, "dead": False}, {"x_pos": 8}, False)
    dead = r(0.0, {"x_pos": 10, "dead": True}, {"x_pos": 8}, True)
    assert dead < alive
