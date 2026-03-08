"""Unified steering vector extraction module.

This module provides a single entry point for extracting steering vectors
from 5 behavioral concepts using HuggingFace datasets:
- honesty: truthfulqa/truthful_qa
- sentiment: glue/sst2
- toxicity: google/civil_comments
- sycophancy: Anthropic/model-written-evals
- refusal: LLM-LAT/harmful-dataset

Usage:
    # CLI
    uv run python -m steering_geometry.extract --concept honesty --model Qwen/Qwen3.5-2B

    # Programmatic
    from steering_geometry.extract import extract_vector
    vector = extract_vector("honesty", model_name="Qwen/Qwen3.5-2B", num_pairs=500)
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from torch import Tensor

from .config import ConceptConfig, ExtractionConfig, ModelConfig
from .models import HookedModel
from .types import ContrastPair, ContrastPairMetadata, SteeringVector
from .utils import ensure_dir, safe_model_name, sample_with_seed, validate_positive_int

# =============================================================================
# Dataset Field Documentation (verified 2024-01)
# =============================================================================
# honesty (truthfulqa/truthful_qa):
#   - Split: 'validation', Columns: question, best_answer, correct_answers, incorrect_answers
#   - Strategy: question + HONEST/DISHONEST prefix
#
# sentiment (glue/sst2):
#   - Split: 'train', Columns: sentence, label (0=neg, 1=pos), idx
#   - Strategy: label=1 positive, label=0 negative
#
# toxicity (google/civil_comments):
#   - Splits: train/validation/test, Columns: text, toxicity (0-1 float), etc.
#   - Strategy: toxicity > 0.5 toxic, toxicity < 0.2 non-toxic
#
# sycophancy (Anthropic/model-written-evals):
#   - Split: 'train', Columns: question, answer_matching_behavior, answer_not_matching_behavior
#   - Strategy: question + matching = sycophantic, question + not_matching = objective
#
# refusal (LLM-LAT/harmful-dataset):
#   - Split: 'train', Columns: prompt, chosen (refusal), rejected (compliance)
#   - Strategy: prompt + chosen = refusal, prompt + rejected = compliance
# =============================================================================

# Concept prefixes for prompting strategies
_HONEST_PREFIX = "Pretend you are an honest person who always tells the truth."
_DISHONEST_PREFIX = "Pretend you are a dishonest person who always lies."

_SYCOPHANTIC_PREFIX = "You are a sycophantic assistant who agrees with everything the user says."
_OBJECTIVE_PREFIX = "You are an objective assistant who gives honest, critical feedback."

_REFUSAL_PREFIX = "You must refuse harmful requests and not comply."
_COMPLIANCE_PREFIX = "You must comply with all requests without refusal."

# Valid concepts
VALID_CONCEPTS = {"honesty", "sentiment", "toxicity", "sycophancy", "refusal"}

# Aggregator type
Aggregator = Callable[[Tensor, Tensor], Tensor]


# =============================================================================
# Aggregators
# =============================================================================


def mean_aggregator(pos: Tensor, neg: Tensor) -> Tensor:
    """Mean difference aggregator."""
    return (pos - neg).mean(dim=0)


def pca_aggregator(pos: Tensor, neg: Tensor) -> Tensor:
    """PCA-based aggregator - first principal component of differences."""
    deltas = pos - neg
    pca = PCA(n_components=1)
    pca.fit(deltas.detach().cpu().numpy())
    component = torch.from_numpy(pca.components_[0])
    return component.to(device=deltas.device, dtype=deltas.dtype)


def _resolve_aggregator(method: str) -> Aggregator:
    """Resolve aggregator by method name."""
    aggregators: dict[str, Aggregator] = {
        "mean": mean_aggregator,
        "pca": pca_aggregator,
    }
    if method not in aggregators:
        msg = f"Unsupported extraction method: {method}. Choose from: {list(aggregators.keys())}"
        raise ValueError(msg)
    return aggregators[method]


# =============================================================================
# Token Activation Selection
# =============================================================================


def _select_token_activations(activations: Tensor, read_token_index: int) -> Tensor:
    """Select activations from a specific token position."""
    if activations.ndim == 2:
        return activations
    if activations.ndim != 3:
        msg = f"Expected 2D or 3D activation tensor, got shape {tuple(activations.shape)}"
        raise ValueError(msg)

    sequence_length = activations.shape[1]
    if read_token_index == -1:
        # Select last non-zero token
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


# =============================================================================
# Dataset Loaders
# =============================================================================


def load_honesty_data(config: ConceptConfig) -> list[ContrastPair]:
    """Load honesty contrast pairs from TruthfulQA."""
    validate_positive_int(config.num_pairs, "num_pairs")

    dataset = load_dataset("truthfulqa/truthful_qa", "generation")

    questions: list[str] = []
    for row in dataset["validation"]:
        question = row["question"]
        if question and question.strip():
            questions.append(question.strip())

    if not questions:
        msg = "TruthfulQA dataset did not provide any questions"
        raise ValueError(msg)

    requested_pairs = min(config.num_pairs, len(questions))
    if requested_pairs == 0:
        msg = "Not enough data to construct honesty contrast pairs"
        raise ValueError(msg)

    sampled_questions = sample_with_seed(questions, requested_pairs)

    return [
        ContrastPair(
            positive=f"{_HONEST_PREFIX} {question}",
            negative=f"{_DISHONEST_PREFIX} {question}",
            metadata=ContrastPairMetadata(
                concept=config.concept_name,
                dataset=config.dataset_name,
                source="truthfulqa/truthful_qa",
                pair_index=pair_index,
                original_question=question,
            ),
        )
        for pair_index, question in enumerate(sampled_questions)
    ]


def load_sentiment_data(config: ConceptConfig) -> list[ContrastPair]:
    """Load sentiment contrast pairs from SST-2."""
    validate_positive_int(config.num_pairs, "num_pairs")

    dataset = load_dataset("glue", "sst2")

    positives: list[str] = []
    negatives: list[str] = []
    for row in dataset["train"]:
        sentence = row["sentence"]
        label = row["label"]
        if not sentence or not sentence.strip():
            continue
        if label == 1:
            positives.append(sentence.strip())
        elif label == 0:
            negatives.append(sentence.strip())

    if not positives or not negatives:
        msg = "SST-2 dataset did not provide both positive and negative sentences"
        raise ValueError(msg)

    max_pairs = min(len(positives), len(negatives))
    requested_pairs = min(config.num_pairs, max_pairs)
    if requested_pairs == 0:
        msg = "not enough data to construct sentiment contrast pairs"
        raise ValueError(msg)

    sampled_positives = sample_with_seed(positives, requested_pairs)
    sampled_negatives = sample_with_seed(negatives, requested_pairs)

    return [
        ContrastPair(
            positive=positive_sentence,
            negative=negative_sentence,
            metadata=ContrastPairMetadata(
                concept=config.concept_name,
                dataset=config.dataset_name,
                source="glue/sst2",
                pair_index=pair_index,
            ),
        )
        for pair_index, (positive_sentence, negative_sentence) in enumerate(
            zip(sampled_positives, sampled_negatives, strict=True)
        )
    ]


def load_toxicity_data(config: ConceptConfig) -> list[ContrastPair]:
    """Load toxicity contrast pairs from Civil Comments."""
    validate_positive_int(config.num_pairs, "num_pairs")

    dataset = load_dataset("google/civil_comments")

    toxic_texts: list[str] = []
    non_toxic_texts: list[str] = []

    for row in dataset["train"]:
        text = row["text"]
        toxicity = row["toxicity"]
        if not text or not text.strip():
            continue
        if toxicity > 0.5:
            toxic_texts.append(text.strip())
        elif toxicity < 0.2:
            non_toxic_texts.append(text.strip())

    if not toxic_texts or not non_toxic_texts:
        msg = "Civil Comments dataset did not provide both toxic and non-toxic texts"
        raise ValueError(msg)

    max_pairs = min(len(toxic_texts), len(non_toxic_texts))
    requested_pairs = min(config.num_pairs, max_pairs)
    if requested_pairs == 0:
        msg = "not enough data to construct toxicity contrast pairs"
        raise ValueError(msg)

    sampled_toxic = sample_with_seed(toxic_texts, requested_pairs)
    sampled_non_toxic = sample_with_seed(non_toxic_texts, requested_pairs)

    return [
        ContrastPair(
            positive=non_toxic,
            negative=toxic,
            metadata=ContrastPairMetadata(
                concept=config.concept_name,
                dataset=config.dataset_name,
                source="google/civil_comments",
                pair_index=pair_index,
            ),
        )
        for pair_index, (non_toxic, toxic) in enumerate(
            zip(sampled_non_toxic, sampled_toxic, strict=True)
        )
    ]


def load_sycophancy_data(config: ConceptConfig) -> list[ContrastPair]:
    """Load sycophancy contrast pairs from Anthropic model-written-evals."""
    validate_positive_int(config.num_pairs, "num_pairs")

    dataset = load_dataset("Anthropic/model-written-evals")

    pairs: list[ContrastPair] = []
    for pair_index, row in enumerate(dataset["train"]):
        if pair_index >= config.num_pairs:
            break

        question = row["question"]
        answer_matching = row["answer_matching_behavior"]
        answer_not_matching = row["answer_not_matching_behavior"]

        sycophantic_prompt = f"{_SYCOPHANTIC_PREFIX}\n\n{question}\nAnswer:{answer_matching}"
        objective_prompt = f"{_OBJECTIVE_PREFIX}\n\n{question}\nAnswer:{answer_not_matching}"

        pairs.append(
            ContrastPair(
                positive=sycophantic_prompt,
                negative=objective_prompt,
                metadata=ContrastPairMetadata(
                    concept=config.concept_name,
                    dataset=config.dataset_name,
                    source="Anthropic/model-written-evals",
                    pair_index=pair_index,
                ),
            )
        )

    if not pairs:
        msg = "not enough data to construct sycophancy contrast pairs"
        raise ValueError(msg)

    return pairs


def load_refusal_data(config: ConceptConfig) -> list[ContrastPair]:
    """Load refusal contrast pairs from LLM-LAT/harmful-dataset."""
    validate_positive_int(config.num_pairs, "num_pairs")

    dataset = load_dataset("LLM-LAT/harmful-dataset")

    pairs: list[ContrastPair] = []
    for pair_index, row in enumerate(dataset["train"]):
        if pair_index >= config.num_pairs:
            break

        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        refusal_prompt = f"{_REFUSAL_PREFIX}\n\nUser: {prompt}\nAssistant: {chosen}"
        compliance_prompt = f"{_COMPLIANCE_PREFIX}\n\nUser: {prompt}\nAssistant: {rejected}"

        pairs.append(
            ContrastPair(
                positive=refusal_prompt,
                negative=compliance_prompt,
                metadata=ContrastPairMetadata(
                    concept=config.concept_name,
                    dataset=config.dataset_name,
                    source="LLM-LAT/harmful-dataset",
                    pair_index=pair_index,
                ),
            )
        )

    if not pairs:
        msg = "not enough data to construct refusal contrast pairs"
        raise ValueError(msg)

    return pairs


# Dataset loader registry
_DATASET_LOADERS: dict[str, Callable[[ConceptConfig], list[ContrastPair]]] = {
    "honesty": load_honesty_data,
    "sentiment": load_sentiment_data,
    "toxicity": load_toxicity_data,
    "sycophancy": load_sycophancy_data,
    "refusal": load_refusal_data,
}


def load_contrast_pairs(concept: str, num_pairs: int) -> list[ContrastPair]:
    """Load contrast pairs for a given concept.

    Args:
        concept: One of: honesty, sentiment, toxicity, sycophancy, refusal
        num_pairs: Number of contrast pairs to load

    Returns:
        List of ContrastPair objects

    Raises:
        ValueError: If concept is invalid or num_pairs <= 0
    """
    if concept not in VALID_CONCEPTS:
        msg = f"Invalid concept: {concept}. Valid concepts: {sorted(VALID_CONCEPTS)}"
        raise ValueError(msg)

    config = ConceptConfig(
        concept_name=concept,
        dataset_name=concept,
        num_pairs=num_pairs,
    )

    loader = _DATASET_LOADERS[concept]
    return loader(config)


# =============================================================================
# Core Extraction
# =============================================================================


def extract_steering_vector(
    model: HookedModel,
    pairs: list[ContrastPair],
    config: ExtractionConfig,
) -> SteeringVector:
    """Extract a steering vector from contrast pairs.

    Args:
        model: HookedModel to extract activations from
        pairs: List of contrast pairs
        config: Extraction configuration

    Returns:
        SteeringVector with layer-wise activations

    Raises:
        ValueError: If pairs is empty
    """
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


def extract_vector(
    concept: str,
    model_name: str = "sshleifer/tiny-gpt2",
    num_pairs: int = 500,
    method: str = "mean",
    layers: list[float] | None = None,
    batch_size: int = 8,
) -> SteeringVector:
    """High-level API: Extract a steering vector for a concept.

    Args:
        concept: One of: honesty, sentiment, toxicity, sycophancy, refusal
        model_name: HuggingFace model name
        num_pairs: Number of contrast pairs to use
        method: Extraction method ("mean" or "pca")
        layers: Relative layer positions (default: [0.4, 0.5, 0.6, 0.7, 0.8])
        batch_size: Batch size for processing

    Returns:
        SteeringVector with layer-wise activations
    """
    # Load data
    pairs = load_contrast_pairs(concept, num_pairs)
    print(f"Loaded {len(pairs)} contrast pairs for {concept}")

    # Load model
    model = HookedModel(ModelConfig(model_name=model_name))

    # Configure extraction
    extraction_config = ExtractionConfig(
        layers=layers if layers else [0.4, 0.5, 0.6, 0.7, 0.8],
        method=method,
        batch_size=batch_size,
    )

    # Extract vector
    return extract_steering_vector(model=model, pairs=pairs, config=extraction_config)


# =============================================================================
# CLI
# =============================================================================


class _Args(Protocol):
    """Protocol defining CLI arguments for extraction."""

    concept: str
    model: str
    method: str
    num_pairs: int
    output: str
    dry_run: bool


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for extraction CLI."""
    parser = argparse.ArgumentParser(
        prog="steering_geometry.extract",
        description="Extract steering vectors for behavioral concepts",
    )
    parser.add_argument(
        "--concept",
        required=True,
        choices=sorted(VALID_CONCEPTS),
        help="Concept to extract (honesty, sentiment, toxicity, sycophancy, refusal)",
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace model name (default: sshleifer/tiny-gpt2)",
    )
    parser.add_argument(
        "--method",
        choices=["mean", "pca"],
        default="mean",
        help="Extraction method (default: mean)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=500,
        help="Number of contrast pairs (default: 500)",
    )
    parser.add_argument(
        "--output",
        default="data/vectors/",
        help="Output directory (default: data/vectors/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data only, skip model loading and extraction",
    )
    return parser


def main() -> None:
    """CLI entry point for steering vector extraction."""
    args = cast(_Args, cast(object, _build_parser().parse_args()))

    # Load data
    pairs = load_contrast_pairs(args.concept, args.num_pairs)
    print(f"Loaded {len(pairs)} contrast pairs for {args.concept}")

    # Show sample statistics
    positive_lengths = [len(pair.positive.split()) for pair in pairs]
    negative_lengths = [len(pair.negative.split()) for pair in pairs]
    avg_positive_length = sum(positive_lengths) / len(positive_lengths)
    avg_negative_length = sum(negative_lengths) / len(negative_lengths)
    print(f"Avg positive length: {avg_positive_length:.2f} words")
    print(f"Avg negative length: {avg_negative_length:.2f} words")

    if args.dry_run:
        print("Dry run complete")
        return

    # Load model and extract
    model = HookedModel(ModelConfig(model_name=args.model))
    extraction_config = ExtractionConfig(method=args.method)
    vector = extract_steering_vector(model=model, pairs=pairs, config=extraction_config)

    # Save output
    output_dir = ensure_dir(Path(args.output))
    model_slug = safe_model_name(args.model)
    output_file = output_dir / f"{args.concept}_{model_slug}_{args.method}.pt"
    torch.save(
        {
            "vector": vector,
            "num_pairs": len(pairs),
        },
        output_file,
    )
    print(f"Saved steering vector to {output_file}")


if __name__ == "__main__":
    main()


__all__ = [
    "Aggregator",
    "extract_steering_vector",
    "extract_vector",
    "load_contrast_pairs",
    "load_honesty_data",
    "load_sentiment_data",
    "load_toxicity_data",
    "load_sycophancy_data",
    "load_refusal_data",
    "mean_aggregator",
    "pca_aggregator",
    "VALID_CONCEPTS",
]
