"""Evaluation framework for steering vectors."""

from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch
import torch.nn.functional as functional
from torch import Tensor

from .config import EvaluationConfig
from .models import HookedModel
from .types import EvaluationResult, SteeringVector


@contextmanager
def apply_steering_vector(
    model: HookedModel,
    vector: SteeringVector,
    scale: float = 1.0,
) -> Generator[None, None, None]:
    """Context manager that applies a steering vector to model activations.

    Registers forward hooks on specified layers to add steering vector activations
    during model forward passes. Hooks are automatically removed when exiting.

    Args:
        model: The hooked model to apply steering to.
        vector: The steering vector containing layer activations.
        scale: Scaling factor for the steering vector strength. Defaults to 1.0.

    Yields:
        None. The context manager modifies model behavior in-place.

    Example:
        with apply_steering_vector(model, steering_vec, scale=0.5):
            output = model.generate("Hello")
    """
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, activation in vector.layer_activations.items():
        layer = model._get_layers_module()[layer_idx]  # noqa: SLF001

        def hook_fn(
            module: torch.nn.Module,
            inputs: tuple[Tensor, ...],
            output: Tensor,
            act: Tensor = activation,
        ) -> Tensor:
            return output + scale * act.to(output.device, dtype=output.dtype)

        handle = layer.register_forward_hook(hook_fn)
        hooks.append(handle)

    try:
        yield
    finally:
        for handle in hooks:
            handle.remove()


def compute_cosine_similarity(
    v1: SteeringVector,
    v2: SteeringVector,
) -> dict[int, float]:
    """Compute per-layer cosine similarity between two steering vectors.

    Calculates cosine similarity for each layer that exists in both vectors.
    Layers present in only one vector are not included in the result.

    Args:
        v1: First steering vector.
        v2: Second steering vector.

    Returns:
        Dictionary mapping layer indices to cosine similarity values.
        Only includes layers present in both vectors.

    Raises:
        ValueError: If the vectors share no common layers.
    """
    common_layers = set(v1.layer_activations.keys()) & set(v2.layer_activations.keys())

    if not common_layers:
        raise ValueError(
            f"Vectors share no common layers. v1 has layers {list(v1.layer_activations.keys())}, "
            f"v2 has layers {list(v2.layer_activations.keys())}"
        )

    similarities: dict[int, float] = {}

    for layer_idx in common_layers:
        act1 = v1.layer_activations[layer_idx].flatten()
        act2 = v2.layer_activations[layer_idx].flatten().to(act1.device, dtype=act1.dtype)

        similarity = functional.cosine_similarity(
            act1.unsqueeze(0),
            act2.unsqueeze(0),
            dim=1,
        )

        similarities[layer_idx] = similarity.item()

    return similarities


def evaluate_steering_vector(
    model: HookedModel,
    vector: SteeringVector,
    eval_fn: Callable[[HookedModel, SteeringVector], dict[str, float]],
    config: EvaluationConfig,
) -> EvaluationResult:
    """Evaluate a steering vector's effectiveness using a custom evaluation function.

    The evaluation function receives the model and steering vector, applies the
    steering, and returns a dictionary of metric scores.

    Args:
        model: The hooked model to evaluate on.
        vector: The steering vector to evaluate.
        eval_fn: Custom evaluation function that takes (model, vector) and returns
            a dictionary of metric names to scores. The function is responsible
            for applying the steering vector within its scope.
        config: Evaluation configuration (num_samples, seed, etc.).

    Returns:
        EvaluationResult containing scores, concept name, and model name.

    Example:
        def my_eval(model, vector):
            with apply_steering_vector(model, vector, scale=1.0):
                scores = run_benchmark(model)
            return scores

        result = evaluate_steering_vector(model, vec, my_eval, config)
    """
    torch.manual_seed(config.seed)

    scores = eval_fn(model, vector)

    return EvaluationResult(
        scores=scores,
        concept=vector.concept,
        model_name=vector.model_name,
    )


__all__ = [
    "apply_steering_vector",
    "compute_cosine_similarity",
    "evaluate_steering_vector",
]
