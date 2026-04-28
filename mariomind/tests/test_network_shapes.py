import torch

from mariomind.agents.networks import QNetwork


def test_q_network_output_shape():
    net = QNetwork(input_channels=4, action_dim=6)
    x = torch.randint(0, 255, (2, 4, 84, 84), dtype=torch.uint8)
    y = net(x)
    assert y.shape == (2, 6)
