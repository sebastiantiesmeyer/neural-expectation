"""Frozen Hugging Face transformer loading and inference."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import resolve_device, resolve_dtype


@dataclass
class FrozenTransformer:
    """Tokenizer and frozen causal language model used as a perceptual encoder."""

    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    layers: Sequence[int]

    @classmethod
    def from_pretrained(cls, model_name: str, layers: Sequence[int], device: str = "auto",
                        dtype: str = "auto") -> "FrozenTransformer":
        selected_device = resolve_device(device)
        selected_dtype = resolve_dtype(dtype, selected_device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=selected_dtype)
        model.to(selected_device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(model, tokenizer, selected_device, tuple(layers))

    @torch.no_grad()
    def hidden_states(self, texts: List[str]) -> Dict[int, torch.Tensor]:
        """Run inference and return only configured layers, without gradients."""
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        output = self.model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = output.hidden_states
        missing = [layer for layer in self.layers if layer >= len(hidden_states)]
        if missing:
            raise ValueError(f"requested layers {missing}, model returned {len(hidden_states) - 1} transformer layers")
        selected = {layer: hidden_states[layer].detach() for layer in self.layers}
        del output, hidden_states
        return selected
