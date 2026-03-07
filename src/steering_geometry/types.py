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
        method: The extraction method used (e.g., "contrast", "pca").
    """

    layer_activations: dict[int, Tensor]
    model_name: str
    concept: str
    method: str


@dataclass
class ExtractionResult:
    """Result of a steering vector extraction process.

    Contains the extracted vector along with metrics about the extraction.

    Attributes:
        vector: The extracted steering vector.
        metrics: Metrics from the extraction (e.g., variance explained).
        timestamp: ISO format timestamp of when extraction completed.
    """

    vector: SteeringVector
    metrics: dict[str, float]
    timestamp: str


@dataclass
class EvaluationResult:
    """Result of evaluating a steering vector's effectiveness.

    Contains scores measuring how well the vector steers model behavior.

    Attributes:
        scores: Evaluation metrics (e.g., accuracy, effect_size).
        concept: The concept that was evaluated.
        model_name: Name of the model used for evaluation.
    """

    scores: dict[str, float]
    concept: str
    model_name: str


__all__ = [
    "ContrastPair",
    "SteeringVector",
    "ExtractionResult",
    "EvaluationResult",
]
