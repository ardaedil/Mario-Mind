from mariomind.agents.replay_buffer import ReplayBuffer


def test_replay_buffer_add_sample():
    rb = ReplayBuffer(capacity=10, state_shape=(4, 84, 84))
    s = [[[0 for _ in range(84)] for _ in range(84)] for _ in range(4)]
    for i in range(8):
        rb.add(s, i % 3, float(i), s, i % 2 == 0)
    assert len(rb) == 8
    batch = rb.sample(4)
    assert len(batch.states) == 4
    assert len(batch.actions) == 4
