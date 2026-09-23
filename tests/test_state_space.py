import numpy as np

from transformer_surprise.state_space import RLSStateSpace


def test_rls_improves_on_linear_sequence():
    model = RLSStateSpace(1, forgetting_factor=1.0, regularization=1.0, min_experience=3)
    errors = []
    for value in range(1, 30):
        result = model.update(np.array([value - 1.0]), np.array([2.0 * value + 1.0]))
        errors.append(result.error)
    assert np.mean(errors[-5:]) < np.mean(errors[:5])
    assert model.experience == 29


def test_cpu_execution_and_unknown_status():
    model = RLSStateSpace(2)
    result = model.update(np.zeros(2), np.ones(2))
    assert result.prediction.shape == (2,)
    assert result.status == "unknown"
    assert result.confidence > 0.0
