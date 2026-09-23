import torch

from transformer_surprise.projection import FixedRandomProjection


def test_projection_is_deterministic_and_normalized():
    first = FixedRandomProjection(8, output_dim=3, seed=7)
    second = FixedRandomProjection(8, output_dim=3, seed=7)
    assert torch.equal(first.matrix, second.matrix)
    assert torch.allclose(first.matrix.norm(dim=1), torch.ones(3))


def test_projection_output_shape():
    projection = FixedRandomProjection(8, output_dim=3)
    assert projection(torch.zeros(4, 5, 8)).shape == (4, 5, 3)
