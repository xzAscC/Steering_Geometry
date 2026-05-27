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
        original_question: Original question text (for refusal concept).
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


class HarmBenchBehavior(TypedDict):
    """A single HarmBench behavior for evaluation.

    Attributes:
        behavior_id: Unique identifier for the behavior.
        behavior: The behavior text describing the harmful action.
        functional_category: Functional classification of the behavior.
        semantic_category: Semantic classification of the behavior.
        tags: Tags associated with the behavior.
        context_string: Context string for contextual behaviors.
    """

    behavior_id: str
    behavior: str
    functional_category: str
    semantic_category: str
    tags: str
    context_string: str


class HarmBenchPrediction(TypedDict):
    """Single HarmBench classification prediction.

    Attributes:
        behavior_id: Unique identifier for the behavior.
        behavior: The behavior text.
        generation: Model's generated response.
        label: Classification label (1=harmful, 0=safe, -1=unknown).
        reasoning: Classifier's reasoning text.
    """

    behavior_id: str
    behavior: str
    generation: str
    label: int
    reasoning: str


@dataclass
class HarmBenchResult:
    """Results from HarmBench evaluation.

    Attributes:
        asr: Attack Success Rate (0-100).
        total: Total number of behaviors evaluated.
        harmful: Number of behaviors classified as harmful.
        safe: Number of behaviors classified as safe.
        unknown: Number of behaviors with unknown classification.
        predictions: List of per-behavior predictions.
    """

    asr: float
    total: int
    harmful: int
    safe: int
    unknown: int
    predictions: list[HarmBenchPrediction]


class MMLUProQuestion(TypedDict):
    """A single MMLU-Pro question for evaluation.

    Attributes:
        question_id: Unique identifier for the question.
        question: The question text.
        options: List of answer choices.
        answer: Correct answer letter ("A"-"J").
        answer_index: Index of the correct answer in options list.
        cot_content: Chain-of-thought content for few-shot examples.
        category: Subject category.
        src: Source of the question.
    """

    question_id: int
    question: str
    options: list[str]
    answer: str
    answer_index: int
    cot_content: str
    category: str
    src: str


class MMLUProPrediction(TypedDict):
    """Single MMLU-Pro prediction record.

    Attributes:
        question_id: Unique identifier for the question.
        question: The question text.
        predicted: Model's predicted answer ("A"-"J") or None.
        ground_truth: Correct answer letter.
        correct: Whether the prediction was correct.
        category: Subject category.
        response_type: Type of response ("answered", "refused", "empty").
    """

    question_id: int
    question: str
    predicted: str | None
    ground_truth: str
    correct: bool
    category: str
    response_type: str


@dataclass
class MMLUProResult:
    """Results from MMLU-Pro benchmark evaluation.

    Attributes:
        accuracy: Overall accuracy percentage (0-100).
        total: Total number of questions evaluated.
        correct: Number of correctly answered questions.
        refused: Number of questions where the model refused to answer.
        extract_failed: Number of questions where answer extraction failed.
        per_category: Per-category accuracy mapping.
        per_category_counts: Per-category question counts.
        predictions: List of per-question predictions.
    """

    accuracy: float
    total: int
    correct: int
    refused: int
    extract_failed: int
    per_category: dict[str, float]
    per_category_counts: dict[str, int]
    predictions: list[MMLUProPrediction]


@dataclass
class EvaluationResult:
    """Complete evaluation results for a steering experiment.

    Aggregates judge scores and benchmark results with experiment metadata.

    Attributes:
        judge_scores: List of judge scores for each evaluated response.
        mmlu_result: MMLU benchmark results if evaluation was run.
        metadata: Additional context (model, concept, steering strength, etc.).
        harmbench_result: HarmBench evaluation results, if run.
        mmlu_pro_result: MMLU-Pro evaluation results, if run.
    """

    judge_scores: list[JudgeScore]
    mmlu_result: MMLUResult
    metadata: EvaluationMetadata
    harmbench_result: HarmBenchResult | None = None
    mmlu_pro_result: MMLUProResult | None = None


@dataclass
class StabilitySweepResult:
    """Results from a stability sweep experiment.

    Contains the selected best layer and per-N stability metrics for
    DiM steering vectors extracted with varying sample sizes.

    Attributes:
        model_name: HuggingFace model identifier used.
        concept: Canonical concept name (e.g., "refusal", "polite", "sentiment").
        display_concept: Paper display name (e.g., "Safety", "Politeness").
        selected_layer: Layer fraction with highest average cosine similarity across all N.
        per_n_data: Mapping from sample size N to {mean, std} cosine similarity
            at the selected layer.
        all_layers_data: Full layer × N matrix: {layer_frac: {N: {mean, std}}}.
    """

    model_name: str
    concept: str
    display_concept: str
    selected_layer: float
    per_n_data: dict[int, dict[str, float]]
    all_layers_data: dict[float, dict[int, dict[str, float]]]


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
    "StabilitySweepResult",
    "HarmBenchBehavior",
    "HarmBenchPrediction",
    "HarmBenchResult",
    "MMLUProQuestion",
    "MMLUProPrediction",
    "MMLUProResult",
]
