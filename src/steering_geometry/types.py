"""Core type definitions for steering geometry package."""

from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor

# =============================================================================
# TypedDict definitions for structured metadata
# =============================================================================


class ContrastPairMetadata(TypedDict, total=False):
    """Metadata for a contrast pair.

    Attributes:
        concept: The behavioral concept this pair targets.
        dataset: Name of the source dataset.
        source: Specific source identifier within the dataset.
        pair_index: Index of this pair in the sampled set.
        original_question: Original question text (for honesty concept).
    """

    concept: str
    dataset: str
    source: str
    pair_index: int
    original_question: str


class EvaluationMetadata(TypedDict, total=False):
    """Metadata for steering evaluation results.

    Attributes:
        concept: The behavioral concept being evaluated.
        model: Name of the model used for generation.
        layer: Layer index where steering was applied.
        multiplier: Steering strength multiplier.
    """

    concept: str
    model: str
    layer: int
    multiplier: float


class MMLUPrediction(TypedDict):
    """Single MMLU benchmark prediction record.

    Attributes:
        question: The question text.
        predicted: Model's predicted answer (e.g., "A", "B").
        ground_truth: Correct answer.
        correct: Whether the prediction was correct.
    """

    question: str
    predicted: str | None
    ground_truth: str
    correct: bool


class MMLUQuestion(TypedDict):
    """MMLU question format from HuggingFace dataset.

    Attributes:
        question: The question text.
        options: List of answer choices.
        answer: The correct answer letter.
    """

    question: str
    options: list[str]
    answer: str


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
    metadata: ContrastPairMetadata


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


@dataclass
class JudgeScore:
    """Score from LLM judge evaluating a steered response.

    Contains numeric scores and reasoning for concept adherence and fluency.

    Attributes:
        concept_score: Score for how well the response matches the target concept (1-5).
        fluency_score: Score for response quality and naturalness (1-5).
        final_score: Weighted combination of concept and fluency scores.
        reasoning: Explanation of the scoring decision.
    """

    concept_score: int
    fluency_score: int
    final_score: float
    reasoning: str


@dataclass
class MMLUResult:
    """Results from MMLU benchmark evaluation.

    Contains accuracy metrics and individual predictions for the benchmark.

    Attributes:
        correct: Number of correctly answered questions.
        total: Total number of questions evaluated.
        accuracy: Fraction of correct answers (correct/total).
        predictions: List of prediction records with question, choices, and answer.
    """

    correct: int
    total: int
    accuracy: float
    predictions: list[MMLUPrediction]


@dataclass
class EvaluationResult:
    """Complete evaluation results for a steering experiment.

    Aggregates judge scores and benchmark results with experiment metadata.

    Attributes:
        judge_scores: List of judge scores for each evaluated response.
        mmlu_result: MMLU benchmark results if evaluation was run.
        metadata: Additional context (model, concept, steering strength, etc.).
    """

    judge_scores: list[JudgeScore]
    mmlu_result: MMLUResult
    metadata: EvaluationMetadata


@dataclass
class TDNVLayerMetrics:
    """Per-layer TDNV metrics.

    Contains the TDNV value and normalized components for a single layer.

    Attributes:
        tdnv: Topic-Discriminative Normalized Variance value.
        norm_num: Normalized numerator (avg within-topic variance / avg energy).
        norm_den: Normalized denominator (avg between-topic distance / avg energy).
        energy: Average activation energy (s = (1/N) * sum ||h||^2).
    """

    tdnv: float
    norm_num: float
    norm_den: float
    energy: float


@dataclass
class TDNVResult:
    """Complete TDNV analysis results for a concept and model.

    Contains layer-wise TDNV metrics measuring separability of positive/negative
    contrast pairs across all model layers.

    Attributes:
        concept: The behavioral concept analyzed (e.g., "honesty", "toxicity").
        model_name: Name/identifier of the model.
        num_pairs: Number of contrast pairs used.
        layers: List of layer indices analyzed.
        tdnv_values: TDNV value per layer (lower = better separability).
        norm_num_values: Normalized numerator per layer.
        norm_den_values: Normalized denominator per layer.
        layerwise_energy: Average activation energy per layer.
    """

    concept: str
    model_name: str
    num_pairs: int
    layers: list[int]
    tdnv_values: list[float]
    norm_num_values: list[float]
    norm_den_values: list[float]
    layerwise_energy: list[float]


__all__ = [
    "ContrastPair",
    "ContrastPairMetadata",
    "SteeringVector",
    "JudgeScore",
    "MMLUResult",
    "MMLUPrediction",
    "MMLUQuestion",
    "EvaluationResult",
    "EvaluationMetadata",
    "TDNVLayerMetrics",
    "TDNVResult",
]
