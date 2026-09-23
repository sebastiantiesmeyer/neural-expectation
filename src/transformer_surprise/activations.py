"""Utilities for turning selected hidden states into sidecar observations."""

from typing import Dict, Iterable

import torch
from torch import Tensor

from .projection import FixedRandomProjection


def last_valid_token(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Select the final non-padding token for each batch item."""
    attention_mask = attention_mask.to(hidden_states.device)
    indices = attention_mask.sum(dim=1).clamp_min(1) - 1
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, indices]


def project_selected_layers(hidden_by_layer: Dict[int, Tensor], projection: FixedRandomProjection,
                            attention_mask: Tensor) -> Dict[int, Tensor]:
    """Project one compact final-token vector per text and layer to CPU."""
    return {
        layer: projection.project_cpu(last_valid_token(hidden, attention_mask))
        for layer, hidden in hidden_by_layer.items()
    }


def temporal_transitions(projected: Tensor) -> Iterable[tuple[Tensor, Tensor]]:
    """Yield adjacent observations for temporal prediction."""
    for index in range(projected.shape[0] - 1):
        yield projected[index], projected[index + 1]
