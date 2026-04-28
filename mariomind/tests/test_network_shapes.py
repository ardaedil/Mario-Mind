from mariomind.agents.networks import QNetwork


def test_q_network_output_shape():
    net = QNetwork(input_channels=4, action_dim=6)
    y = net([0, 1])
    assert len(y) == 2
    assert len(y[0]) == 6
