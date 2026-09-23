"""Small online linear state-space expectation models."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .surprise import summarize_error


@dataclass
class UpdateResult:
    """Inspectable result of one online transition update."""

    prediction: np.ndarray
    residual: np.ndarray
    error: float
    confidence: float
    surprise: float
    experience: int
    status: str

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class RLSStateSpace:
    """Recursive least-squares model for ``z_next ~= A @ z + b``.

    Confidence is an intentionally simple experience proxy. It prevents a
    low-error first observation from being labeled expected.
    """

    def __init__(self, dimension: int, forgetting_factor: float = 0.995, regularization: float = 1.0,
                 min_experience: int = 5) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if not 0.0 < forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        self.dimension = dimension
        self.forgetting_factor = forgetting_factor
        self.min_experience = min_experience
        self.weights = np.zeros((dimension, dimension + 1), dtype=np.float64)
        self.covariance = np.eye(dimension + 1, dtype=np.float64) / regularization
        self.experience = 0

    def _features(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.size != self.dimension:
            raise ValueError(f"expected state dimension {self.dimension}, got {state.size}")
        return np.concatenate([state, [1.0]])

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict the next state without changing model state."""
        return self.weights @ self._features(state)

    def update(self, state: np.ndarray, next_state: np.ndarray) -> UpdateResult:
        """Score a transition, then update the model from that transition."""
        features = self._features(state)
        target = np.asarray(next_state, dtype=np.float64).reshape(-1)
        if target.size != self.dimension:
            raise ValueError(f"expected next_state dimension {self.dimension}, got {target.size}")
        prediction = self.weights @ features
        residual = target - prediction
        metrics = summarize_error(target, prediction)
        gain_denominator = self.forgetting_factor + features @ self.covariance @ features
        gain = (self.covariance @ features) / gain_denominator
        self.weights += np.outer(residual, gain)
        self.covariance = (self.covariance - np.outer(gain, features @ self.covariance)) / self.forgetting_factor
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self.experience += 1
        confidence = min(1.0, self.experience / max(1, self.min_experience))
        status = "unknown" if self.experience < self.min_experience else ("expected" if metrics["error"] < 1.0 else "surprising")
        return UpdateResult(prediction, residual, metrics["error"], confidence, metrics["error"], self.experience, status)
