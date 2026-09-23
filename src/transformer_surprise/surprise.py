"""Prediction-error metrics used by the sidecar."""

from typing import Dict

import numpy as np


def l2_error(residual: np.ndarray) -> float:
    """Return Euclidean prediction residual magnitude."""
    return float(np.linalg.norm(residual))


def cosine_error(target: np.ndarray, prediction: np.ndarray) -> float:
    """Return one minus cosine similarity between target and prediction."""
    target_norm = np.linalg.norm(target)
    prediction_norm = np.linalg.norm(prediction)
    if target_norm == 0.0 or prediction_norm == 0.0:
        return 1.0
    return float(1.0 - np.dot(target, prediction) / (target_norm * prediction_norm))


def summarize_error(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """Compute baseline errors while leaving covariance-based surprise extensible."""
    residual = target - prediction
    return {
        "error": l2_error(residual),
        "cosine_error": cosine_error(target, prediction),
    }
