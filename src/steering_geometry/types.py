"""Core type definitions for steering geometry package."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass
class ContrastPair:
    """A contrast pair for steering vector extraction.

    Contains positive and negative examples that differ in a target concept.

    Attributes:
        positive: The positive example text (exhibiting the concept).
        negative: The negative example text (not exhibiting the concept).
        metadata: Additional context about the pair (e.g., source, concept).
    """

    positive: str
    negative: str
    metadata: dict[str, Any]


@dataclass
class SteeringVector:
    """A steering vector extracted from model activations.

    Contains layer-wise activation differences that can be used to steer
    model behavior toward or away from a concept.

    Attributes:
        layer_activations: Mapping from layer index to activation tensor.
        model_name: Name/identifier of the model this vector was extracted from.
        concept: The behavioral concept this vector targets.
        method: The extraction method used (e.g., "mean", "pca").
    """

    layer_activations: dict[int, Tensor]
    model_name: str
    concept: str
    method: str


__all__ = [
    "ContrastPair",
    "SteeringVector",
]
