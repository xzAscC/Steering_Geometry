"""Model loading and activation extraction using forward hooks."""

import math
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig

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
        dtype = _DTYPE_MAP.get(config.dtype, torch.float16)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
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
        return [
            math.floor(max(0.0, min(1.0, rel_pos)) * (n_layers - 1) + 0.5)
            for rel_pos in relative_layers
        ]

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

    def get_unembedding_matrix(self) -> Tensor:
        """Get the unembedding matrix from the model.

        Returns the weight matrix that projects hidden states to vocabulary logits.
        Handles different model architectures (Qwen uses lm_head.weight, some use
        embed_out.weight).

        Returns:
            Tensor of shape (vocab_size, hidden_dim), detached and on CPU.

        Raises:
            ValueError: If the unembedding matrix cannot be found.
        """
        # Common locations for unembedding weights
        if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight"):
            weight = cast(Tensor, self.model.lm_head.weight)
            return weight.detach().cpu()
        if hasattr(self.model, "embed_out") and hasattr(self.model.embed_out, "weight"):
            weight = cast(Tensor, self.model.embed_out.weight)
            return weight.detach().cpu()
        # Some models share embeddings with output projection
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            weight = cast(Tensor, self.model.model.embed_tokens.weight)
            return weight.detach().cpu()
        msg = "Could not find unembedding matrix in this model architecture"
        raise ValueError(msg)

    def get_special_token_ids(self) -> set[int]:
        """Get the set of special token IDs (BOS, EOS, PAD).

        Returns:
            Set of special token IDs, excluding any that are None.
        """
        special_ids: set[int] = set()
        if self.tokenizer.bos_token_id is not None:
            special_ids.add(self.tokenizer.bos_token_id)
        if self.tokenizer.eos_token_id is not None:
            special_ids.add(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            special_ids.add(self.tokenizer.pad_token_id)
        return special_ids

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

    def generate_with_steering(
        self,
        prompt: str,
        layer_idx: int,
        steering_vector: Tensor,
        scale: float,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        steer_tokens: int | None = None,
    ) -> str:
        """Generate text with steering vector applied to a specific layer.

        Args:
            prompt: Input text prompt.
            layer_idx: Absolute layer index to apply steering.
            steering_vector: Normalized steering vector (norm=1).
            scale: Scaling factor for the steering vector.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 for greedy).
            steer_tokens: Number of generation steps to apply steering.
                None applies steering to all steps (default).
                0 applies no steering. Values >= max_new_tokens are
                equivalent to None.

        Returns:
            Generated text string.

        Raises:
            ValueError: If layer_idx is out of bounds.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            msg = f"layer_idx {layer_idx} out of bounds [0, {self.num_layers - 1}]"
            raise ValueError(msg)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model_layers = self._get_layers_module()

        step_counter = [0]

        def steering_hook(module: Any, input: Any, output: Tensor) -> Tensor:
            step_counter[0] += 1
            if steer_tokens is not None and step_counter[0] > steer_tokens:
                if isinstance(output, tuple):
                    return output
                return output
            tensor_output = output[0] if isinstance(output, tuple) else output
            steering = steering_vector.to(device=tensor_output.device, dtype=tensor_output.dtype)
            tensor_output = tensor_output + steering * scale
            if isinstance(output, tuple):
                return (tensor_output,) + output[1:]
            return tensor_output

        handle = model_layers[layer_idx].register_forward_hook(steering_hook)

        try:
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            if steer_tokens is not None:
                gen_kwargs["use_cache"] = True

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            generated_text = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            assert isinstance(generated_text, str)
            return generated_text
        finally:
            handle.remove()


__all__ = ["HookedModel"]
