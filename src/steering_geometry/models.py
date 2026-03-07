"""Model loading and activation extraction using forward hooks."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_geometry.config import ModelConfig
from steering_geometry.types import SteeringVector

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class HookedModel:
    """A wrapper around HuggingFace models for activation extraction and steering.

    Provides utilities for:
    - Loading models with configurable device/dtype
    - Extracting activations from specific layers using forward hooks
    - Generating text with optional steering vectors

    Attributes:
        config: Model configuration.
        model: The underlying HuggingFace model.
        tokenizer: The tokenizer for the model.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the HookedModel with the given configuration.

        Args:
            config: Model configuration specifying model name, device, dtype, etc.
        """
        self.config = config
        torch_dtype = _DTYPE_MAP.get(config.dtype, torch.float16)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def num_layers(self) -> int:
        """Return the total number of transformer layers in the model.

        Returns:
            The number of layers in the model.
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return len(self.model.layers)  # type: ignore[arg-type]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return len(self.model.transformer.h)  # type: ignore[arg-type]
        msg = "Could not determine number of layers for this model architecture"
        raise ValueError(msg)

    def resolve_layers(self, relative_layers: list[float]) -> list[int]:
        """Convert relative layer positions to absolute layer indices.

        Args:
            relative_layers: List of relative positions (0.0-1.0).
                For example, [0.5] on a 32-layer model returns [16].

        Returns:
            List of absolute layer indices (0-indexed).
        """
        n_layers = self.num_layers
        return [round(max(0.0, min(1.0, rel_pos)) * (n_layers - 1)) for rel_pos in relative_layers]

    def _get_layers_module(self) -> Any:
        """Get the layers module from the model architecture."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        msg = "Could not find layers in this model architecture"
        raise ValueError(msg)

    def get_activations(
        self,
        texts: list[str],
        layers: list[int],
    ) -> dict[int, Tensor]:
        """Extract activations from specified layers for the given texts.

        Uses forward hooks to capture intermediate activations during inference.

        Args:
            texts: List of input texts to process.
            layers: List of absolute layer indices to extract activations from.

        Returns:
            Dictionary mapping layer index to activation tensors.
            Each tensor has shape (batch_size, seq_len, hidden_dim).
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        activations: dict[int, Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        model_layers = self._get_layers_module()

        def make_hook(layer_idx: int) -> Callable[[Any, Any, Tensor], None]:
            def hook_fn(module: Any, input: Any, output: Tensor) -> None:
                tensor_output = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = tensor_output.detach().clone()

            return hook_fn

        for layer_idx in layers:
            handle = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        return activations

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        steering_vector: SteeringVector | None = None,
    ) -> str:
        """Generate text with optional steering vector intervention.

        Args:
            prompt: The input prompt to continue.
            max_new_tokens: Maximum number of new tokens to generate.
            steering_vector: Optional steering vector to apply during generation.

        Returns:
            The generated text (excluding the prompt).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        handles: list[torch.utils.hooks.RemovableHandle] = []

        if steering_vector is not None:
            model_layers = self._get_layers_module()

            def make_steering_hook(steering_vec: Tensor) -> Callable[[Any, Any, Tensor], Tensor]:
                def hook_fn(module: Any, input: Any, output: Tensor) -> Tensor:
                    if isinstance(output, tuple):
                        modified = output[0] + steering_vec.to(output[0].device)
                        return (modified,) + output[1:]
                    return output + steering_vec.to(output.device)

                return hook_fn

            for layer_idx, layer_activation in steering_vector.layer_activations.items():
                handle = model_layers[layer_idx].register_forward_hook(
                    make_steering_hook(layer_activation)
                )
                handles.append(handle)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return str(full_output)[len(prompt) :].strip()

        finally:
            for handle in handles:
                handle.remove()


__all__ = ["HookedModel"]
