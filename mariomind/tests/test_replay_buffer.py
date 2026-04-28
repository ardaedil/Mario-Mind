import numpy as np

from mariomind.agents.replay_buffer import ReplayBuffer


def test_replay_buffer_add_sample():
    rb = ReplayBuffer(capacity=10, state_shape=(4, 84, 84))
    s = np.zeros((4, 84, 84), dtype=np.uint8)
    for i in range(8):
        rb.add(s + i, i % 3, float(i), s + i + 1, i % 2 == 0)
    assert len(rb) == 8
    batch = rb.sample(4)
    assert batch.states.shape == (4, 4, 84, 84)
    assert batch.actions.shape == (4,)
