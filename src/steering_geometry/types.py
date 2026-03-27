"""Core type definitions for steering geometry package."""

from dataclasses import dataclass, field
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
        category: Subject category (e.g., math, physics, chemistry).
    """

    question: str
    options: list[str]
    answer: str
    category: str


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


@dataclass
class TokenRecord:
    """Record for a single token with its activation data.

    Used to track individual tokens during discriminative token analysis,
    capturing both the token identity and its activation pattern.

    Attributes:
        token_id: Integer token ID from the tokenizer vocabulary.
        token_text: Decoded text representation of the token.
        activation: Activation tensor for this token at a specific layer.
        contrast_pair_idx: Index of the contrast pair this token belongs to.
        position_in_sequence: Position of this token within its sequence.
        label: Label indicating "positive" or "negative" class membership.
        score: Discriminative score (higher = more discriminative for its class).
    """

    token_id: int
    token_text: str
    activation: Tensor
    contrast_pair_idx: int
    position_in_sequence: int
    label: str
    score: float = 0.0


@dataclass
class DiscriminativeTokenResult:
    """Results from discriminative token selection at a single layer.

    Contains the top-k most discriminative tokens for both positive and
    negative classes, useful for visualization and analysis.

    Attributes:
        concept: The behavioral concept being analyzed.
        layer: Layer index where tokens were extracted.
        top_positive: Top tokens that discriminate toward positive class.
        top_negative: Top tokens that discriminate toward negative class.
    """

    concept: str
    layer: int
    top_positive: list[TokenRecord] = field(default_factory=list)
    top_negative: list[TokenRecord] = field(default_factory=list)


@dataclass
class ProbeLayerResult:
    """Probe classification metrics for a single layer.

    Contains accuracy and AUC scores from training a linear probe to
    classify positive vs negative tokens at a specific layer.

    Attributes:
        layer_idx: Index of the layer where the probe was trained.
        train_accuracy: Classification accuracy on training set.
        test_accuracy: Classification accuracy on held-out test set.
        auc_score: Area under ROC curve for binary classification.
    """

    layer_idx: int
    train_accuracy: float
    test_accuracy: float
    auc_score: float


@dataclass
class ProbeExperimentResult:
    """Complete probe experiment results across all layers.

    Aggregates layer-wise probe metrics for analyzing how well different
    layers encode the target concept.

    Attributes:
        concept: The behavioral concept being probed.
        model_name: Name/identifier of the model analyzed.
        tokens_per_class: Number of tokens sampled per class for probing.
        layer_results: List of probe metrics for each analyzed layer.
    """

    concept: str
    model_name: str
    tokens_per_class: int
    layer_results: list[ProbeLayerResult] = field(default_factory=list)


@dataclass
class UnembedAnalysisResult:
    """Single steering vector unembedding analysis result.

    Contains the top tokens and their cosine similarities when projecting
    a steering vector through the unembedding matrix.

    Attributes:
        layer: Layer fraction where the steering vector was extracted (0.1-1.0).
        method: Extraction method used (e.g., "diff_means", "discriminative").
        tokens: Top-5 decoded tokens from unembedding projection.
        similarities: Cosine similarity scores for each token.
    """

    layer: float
    method: str
    tokens: list[str]
    similarities: list[float]


@dataclass
class ConceptAnalysisResult:
    """Full concept analysis result across multiple layers.

    Aggregates unembedding analysis results for a single concept and model,
    organized by layer fraction for comparison across extraction methods.

    Attributes:
        concept: The behavioral concept analyzed (e.g., "honesty", "toxicity").
        model: Name/identifier of the model.
        method: Extraction method used for all results.
        results: Mapping from layer key (e.g., "layer_0.5") to analysis result.
    """

    concept: str
    model: str
    method: str
    results: dict[str, UnembedAnalysisResult]


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
    "TokenRecord",
    "DiscriminativeTokenResult",
    "ProbeLayerResult",
    "ProbeExperimentResult",
    "UnembedAnalysisResult",
    "ConceptAnalysisResult",
]
