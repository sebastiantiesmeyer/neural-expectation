"""Configuration loading and hardware selection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml


@dataclass
class SidecarConfig:
    """Hyperparameters for the online expectation model."""

    type: str = "recursive_least_squares"
    forgetting_factor: float = 0.995
    regularization: float = 1.0


@dataclass
class ExperimentConfig:
    """Minimal experiment configuration."""

    model_name: str = "EleutherAI/pythia-160m"
    device: str = "auto"
    dtype: str = "auto"
    layers: List[int] = field(default_factory=lambda: [2, 6, 10])
    projection_dim: int = 32
    projection_seed: int = 42
    sidecar: SidecarConfig = field(default_factory=SidecarConfig)


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load an experiment config from YAML."""
    with Path(path).open(encoding="utf-8") as file:
        values: Dict[str, Any] = yaml.safe_load(file) or {}
    sidecar = SidecarConfig(**values.pop("sidecar", {}))
    return ExperimentConfig(sidecar=sidecar, **values)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit device without assuming a particular GPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_dtype(requested: str = "auto", device: Optional[torch.device] = None) -> torch.dtype:
    """Choose a conservative inference dtype for the selected device."""
    device = device or resolve_device("auto")
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return getattr(torch, requested)
