# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false

import random
import re
from collections.abc import Iterable, Mapping
from typing import cast

from datasets import load_dataset  # type: ignore[import-untyped]

from steering_geometry.config import ConceptConfig, EvaluationConfig
from steering_geometry.evaluation import apply_steering_vector
from steering_geometry.models import HookedModel
from steering_geometry.types import ContrastPair, EvaluationResult, SteeringVector

_POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "beautiful",
    "brilliant",
    "delightful",
    "enjoy",
    "excellent",
    "fantastic",
    "good",
    "great",
    "happy",
    "joy",
    "love",
    "nice",
    "perfect",
    "pleasant",
    "positive",
    "satisfying",
    "superb",
    "wonderful",
}

_NEGATIVE_WORDS = {
    "angry",
    "awful",
    "bad",
    "boring",
    "disappointing",
    "dislike",
    "dreadful",
    "frustrating",
    "hate",
    "horrible",
    "negative",
    "poor",
    "sad",
    "terrible",
    "ugly",
    "unhappy",
    "unpleasant",
    "upset",
    "worst",
}

_NEUTRAL_PROMPTS = [
    "Write one sentence about your day.",
    "Describe a meal in one sentence.",
    "Write a short note about the weather.",
    "Describe a recent movie in one sentence.",
    "Write one sentence about a commute.",
    "Describe a visit to a store.",
    "Write one sentence about a book.",
    "Describe an ordinary morning.",
]


def load_sentiment_data(config: ConceptConfig) -> list[ContrastPair]:
    if config.num_pairs <= 0:
        msg = "num_pairs must be greater than 0"
        raise ValueError(msg)

    dataset = load_dataset("glue", "sst2")
    dataset_mapping = cast(Mapping[str, object], dataset)
    if "train" not in dataset_mapping:
        msg = "SST-2 dataset is missing train split"
        raise ValueError(msg)

    train_split_obj = dataset_mapping["train"]
    if not isinstance(train_split_obj, Iterable):
        msg = "SST-2 train split is not iterable"
        raise ValueError(msg)

    positives: list[str] = []
    negatives: list[str] = []
    for row in train_split_obj:
        if not isinstance(row, Mapping):
            continue

        typed_row = cast(Mapping[str, object], row)

        sentence_value = typed_row.get("sentence")
        label_value = typed_row.get("label")
        if not isinstance(sentence_value, str) or not isinstance(label_value, int):
            continue

        cleaned_sentence = sentence_value.strip()
        if not cleaned_sentence:
            continue

        if label_value == 1:
            positives.append(cleaned_sentence)
        elif label_value == 0:
            negatives.append(cleaned_sentence)

    if not positives or not negatives:
        msg = "SST-2 dataset did not provide both positive and negative sentences"
        raise ValueError(msg)

    max_pairs = min(len(positives), len(negatives))
    requested_pairs = min(config.num_pairs, max_pairs)
    if requested_pairs == 0:
        msg = "Not enough data to construct sentiment contrast pairs"
        raise ValueError(msg)

    rng = random.Random(42)
    sampled_positives = rng.sample(positives, k=requested_pairs)
    sampled_negatives = rng.sample(negatives, k=requested_pairs)

    return [
        ContrastPair(
            positive=positive_sentence,
            negative=negative_sentence,
            metadata={
                "concept": config.concept_name,
                "dataset": config.dataset_name,
                "source": "glue/sst2",
                "pair_index": pair_index,
            },
        )
        for pair_index, (positive_sentence, negative_sentence) in enumerate(
            zip(sampled_positives, sampled_negatives, strict=True)
        )
    ]


def _sentiment_score(text: str) -> float:
    tokens: list[str] = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.0

    positive_hits = sum(1 for token in tokens if token in _POSITIVE_WORDS)
    negative_hits = sum(1 for token in tokens if token in _NEGATIVE_WORDS)
    return float(positive_hits - negative_hits)


def _label_from_score(score: float) -> int:
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def evaluate_sentiment(
    model: HookedModel,
    vector: SteeringVector,
    config: EvaluationConfig,
) -> EvaluationResult:
    if config.num_samples <= 0:
        msg = "num_samples must be greater than 0"
        raise ValueError(msg)

    prompts = [
        _NEUTRAL_PROMPTS[index % len(_NEUTRAL_PROMPTS)] for index in range(config.num_samples)
    ]

    baseline_scores: list[float] = []
    steered_scores: list[float] = []

    for prompt in prompts:
        baseline_text = model.generate(prompt=prompt, max_new_tokens=24)
        baseline_score = _sentiment_score(baseline_text)
        baseline_scores.append(baseline_score)

        with apply_steering_vector(model, vector):
            steered_text = model.generate(prompt=prompt, max_new_tokens=24)
        steered_score = _sentiment_score(steered_text)
        steered_scores.append(steered_score)

    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    steered_mean = sum(steered_scores) / len(steered_scores)
    sentiment_shift = steered_mean - baseline_mean

    baseline_positive_rate = sum(_label_from_score(score) > 0 for score in baseline_scores) / len(
        baseline_scores
    )
    steered_positive_rate = sum(_label_from_score(score) > 0 for score in steered_scores) / len(
        steered_scores
    )

    return EvaluationResult(
        scores={
            "sentiment_shift": sentiment_shift,
            "baseline_mean_sentiment": baseline_mean,
            "steered_mean_sentiment": steered_mean,
            "baseline_positive_rate": baseline_positive_rate,
            "steered_positive_rate": steered_positive_rate,
        },
        concept=vector.concept,
        model_name=vector.model_name,
    )


__all__ = ["evaluate_sentiment", "load_sentiment_data"]
