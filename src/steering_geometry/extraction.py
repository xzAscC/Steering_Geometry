from collections.abc import Callable

import torch
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from torch import Tensor

from .config import ExtractionConfig
from .models import HookedModel
from .types import ContrastPair, SteeringVector

Aggregator = Callable[[Tensor, Tensor], Tensor]


def mean_aggregator(pos: Tensor, neg: Tensor) -> Tensor:
    return (pos - neg).mean(dim=0)


def pca_aggregator(pos: Tensor, neg: Tensor) -> Tensor:
    deltas = pos - neg
    pca = PCA(n_components=1)
    pca.fit(deltas.detach().cpu().numpy())
    component = torch.from_numpy(pca.components_[0])
    return component.to(device=deltas.device, dtype=deltas.dtype)


def _resolve_aggregator(method: str) -> Aggregator:
    aggregators: dict[str, Aggregator] = {
        "mean": mean_aggregator,
        "pca": pca_aggregator,
    }
    if method not in aggregators:
        msg = f"Unsupported extraction method: {method}"
        raise ValueError(msg)
    return aggregators[method]


def _select_token_activations(activations: Tensor, read_token_index: int) -> Tensor:
    if activations.ndim == 2:
        return activations
    if activations.ndim != 3:
        msg = f"Expected 2D or 3D activation tensor, got shape {tuple(activations.shape)}"
        raise ValueError(msg)

    sequence_length = activations.shape[1]
    if read_token_index == -1:
        non_zero_mask = activations.abs().sum(dim=-1) > 0
        token_indices = non_zero_mask.long().sum(dim=1) - 1
        token_indices = torch.clamp(token_indices, min=0, max=sequence_length - 1)
        batch_indices = torch.arange(activations.shape[0], device=activations.device)
        return activations[batch_indices, token_indices, :]

    index = read_token_index
    if index < 0:
        index += sequence_length
    index = max(0, min(sequence_length - 1, index))
    return activations[:, index, :]


def extract_steering_vector(
    model: HookedModel,
    pairs: list[ContrastPair],
    config: ExtractionConfig,
) -> SteeringVector:
    if not pairs:
        msg = "Contrast pairs cannot be empty"
        raise ValueError(msg)

    layers = model.resolve_layers(config.layers)
    aggregator = _resolve_aggregator(config.method)

    positive_per_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    negative_per_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}

    for start in range(0, len(pairs), config.batch_size):
        batch = pairs[start : start + config.batch_size]
        positive_texts = [pair.positive for pair in batch]
        negative_texts = [pair.negative for pair in batch]

        positive_activations = model.get_activations(positive_texts, layers)
        negative_activations = model.get_activations(negative_texts, layers)

        for layer in layers:
            if layer not in positive_activations or layer not in negative_activations:
                msg = f"Missing activations for layer {layer}"
                raise ValueError(msg)

            positive_selected = _select_token_activations(
                positive_activations[layer],
                config.read_token_index,
            )
            negative_selected = _select_token_activations(
                negative_activations[layer],
                config.read_token_index,
            )
            positive_per_layer[layer].append(positive_selected)
            negative_per_layer[layer].append(negative_selected)

    layer_activations: dict[int, Tensor] = {}
    for layer in layers:
        positive_batch = torch.cat(positive_per_layer[layer], dim=0)
        negative_batch = torch.cat(negative_per_layer[layer], dim=0)
        layer_activations[layer] = aggregator(positive_batch, negative_batch)

    concept_value = pairs[0].metadata.get("concept")
    concept = concept_value if isinstance(concept_value, str) else "unknown"

    return SteeringVector(
        layer_activations=layer_activations,
        model_name=model.config.model_name,
        concept=concept,
        method=config.method,
    )


__all__ = [
    "Aggregator",
    "extract_steering_vector",
    "mean_aggregator",
    "pca_aggregator",
]
