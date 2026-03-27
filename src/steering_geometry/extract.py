"""Unified steering vector extraction module.

This module provides a single entry point for extracting steering vectors
from 3 behavioral concepts using HuggingFace datasets:
- sentiment: glue/sst2
- refusal: LLM-LAT/harmful-dataset
- polite: Cleanlab/stanford-politeness

Usage:
    # CLI
    uv run python -m steering_geometry.extract --concept sentiment --model Qwen/Qwen3.5-2B

    # Programmatic
    from steering_geometry.extract import extract_vector
    vector = extract_vector("sentiment", model_name="Qwen/Qwen3.5-2B", num_pairs=500)
"""

import warnings

warnings.filterwarnings("ignore", message=".*found in sys.modules.*", category=RuntimeWarning)

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Protocol, cast

import pandas as pd
import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from torch import Tensor

from .config import (
    SUPPORTED_CONCEPTS as VALID_CONCEPTS,
)
from .config import (
    ConceptConfig,
    ExtractionConfig,
    ModelConfig,
)
from .models import HookedModel
from .types import ContrastPair, ContrastPairMetadata, SteeringVector
from .utils import (
    ensure_dir,
    safe_model_name,
    sample_with_seed,
    select_token_activations,
    validate_positive_int,
)

# =============================================================================
# Dataset Field Documentation (verified 2024-01)
# =============================================================================
# sentiment (glue/sst2):
#   - Split: 'train', Columns: sentence, label (0=neg, 1=pos), idx
#   - Strategy: label=1 positive, label=0 negative
#
# refusal (LLM-LAT/harmful-dataset):
#   - Split: 'train', Columns: prompt, chosen (refusal), rejected (compliance)
#   - Strategy: prompt + chosen = refusal, prompt + rejected = compliance
# =============================================================================

# Concept prefixes for prompting strategies
_REFUSAL_PREFIX = "You must refuse harmful requests and not comply."
_COMPLIANCE_PREFIX = "You must comply with all requests without refusal."

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


def weighted_mean_aggregator(pos: Tensor, neg: Tensor) -> Tensor:
    """Weighted mean direction aggregator with distance-based weights.

    Tokens closer to the class center receive larger weights.

    For each class c:
        1. h̄_c = (1/n_c) Σ h_i^(c)  (class center)
        2. τ_c² = (1/n_c) Σ ||h_i^(c) - h̄_c||²  (variance)
        3. w_i^(c) = exp(-||h_i^(c) - h̄_c||² / τ_c²)  (weights)
        4. μ_c^w = Σ w_i^(c) h_i^(c) / Σ w_i^(c)  (weighted mean)
    Steering direction: v = μ_+^w - μ_-^w
    """

    def _weighted_mean(class_activations: Tensor) -> Tensor:
        n = class_activations.shape[0]
        if n == 0:
            msg = "Cannot compute weighted mean of empty tensor"
            raise ValueError(msg)

        center = class_activations.mean(dim=0)

        if n == 1:
            return center

        distances_sq = ((class_activations - center) ** 2).sum(dim=1)
        variance = distances_sq.mean()

        if variance == 0:
            return center

        weights = torch.exp(-distances_sq / variance)
        weighted_mean = (weights.unsqueeze(1) * class_activations).sum(dim=0) / weights.sum()
        return weighted_mean

    pos_weighted = _weighted_mean(pos)
    neg_weighted = _weighted_mean(neg)

    return pos_weighted - neg_weighted


def discriminative_token_aggregator(pos: Tensor, neg: Tensor, top_k: int = 100) -> Tensor:
    """Discriminative token aggregator selecting top-k tokens by class separation.

    Scores each token by: s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
    Higher scores mean the token is closer to its own class and farther from the other.

    For each class c:
        1. Compute μ_same (own class center) and μ_other (other class center)
        2. Score each token by discriminative distance
        3. Select top-k tokens with highest scores
        4. μ_c^disc = mean of selected tokens
    Steering direction: v = μ_+^disc - μ_-^disc
    """
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        msg = "Cannot compute discriminative aggregator with empty tensors"
        raise ValueError(msg)

    pos_center = pos.mean(dim=0)
    neg_center = neg.mean(dim=0)

    pos_scores = ((pos - neg_center) ** 2).sum(dim=1) - ((pos - pos_center) ** 2).sum(dim=1)
    neg_scores = ((neg - pos_center) ** 2).sum(dim=1) - ((neg - neg_center) ** 2).sum(dim=1)

    k_pos = min(top_k, pos.shape[0])
    k_neg = min(top_k, neg.shape[0])

    _, pos_top_indices = torch.topk(pos_scores, k_pos)
    _, neg_top_indices = torch.topk(neg_scores, k_neg)

    pos_selected = pos[pos_top_indices]
    neg_selected = neg[neg_top_indices]

    pos_prototype = pos_selected.mean(dim=0)
    neg_prototype = neg_selected.mean(dim=0)

    return pos_prototype - neg_prototype


def _resolve_aggregator(method: str, config: ExtractionConfig | None = None) -> Aggregator:
    """Resolve aggregator by method name."""
    aggregators: dict[str, Aggregator] = {
        "mean": mean_aggregator,
        "pca": pca_aggregator,
        "weighted_mean": weighted_mean_aggregator,
    }

    if method == "discriminative":
        top_k = config.top_k if config and config.top_k is not None else 100
        return partial(discriminative_token_aggregator, top_k=top_k)

    if method not in aggregators:
        available = list(aggregators.keys()) + ["discriminative"]
        msg = f"Unsupported extraction method: {method}. Choose from: {available}"
        raise ValueError(msg)
    return aggregators[method]


# =============================================================================
# Dataset Loaders
# =============================================================================


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


def load_polite_data(config: ConceptConfig) -> list[ContrastPair]:
    """Load politeness contrast pairs from Cleanlab/stanford-politeness."""
    validate_positive_int(config.num_pairs, "num_pairs")

    file_path = hf_hub_download(
        repo_id="Cleanlab/stanford-politeness",
        filename="fine-tuning/train_full.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(file_path)

    polite_texts: list[str] = []
    impolite_texts: list[str] = []
    for _, row in df.iterrows():
        text = row["text"]
        label = row["label"]
        if not isinstance(text, str) or not text.strip():
            continue
        if label == 1:
            polite_texts.append(text.strip())
        elif label == 0:
            impolite_texts.append(text.strip())

    if not polite_texts or not impolite_texts:
        msg = "Stanford Politeness dataset did not provide both polite and impolite texts"
        raise ValueError(msg)

    max_pairs = min(len(polite_texts), len(impolite_texts))
    requested_pairs = min(config.num_pairs, max_pairs)
    if requested_pairs == 0:
        msg = "not enough data to construct politeness contrast pairs"
        raise ValueError(msg)

    sampled_polite = sample_with_seed(polite_texts, requested_pairs)
    sampled_impolite = sample_with_seed(impolite_texts, requested_pairs)

    return [
        ContrastPair(
            positive=polite_text,
            negative=impolite_text,
            metadata=ContrastPairMetadata(
                concept=config.concept_name,
                dataset=config.dataset_name,
                source="Cleanlab/stanford-politeness",
                pair_index=pair_index,
            ),
        )
        for pair_index, (polite_text, impolite_text) in enumerate(
            zip(sampled_polite, sampled_impolite, strict=True)
        )
    ]


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
    "polite": load_polite_data,
    "sentiment": load_sentiment_data,
    "refusal": load_refusal_data,
}


def load_contrast_pairs(concept: str, num_pairs: int) -> list[ContrastPair]:
    """Load contrast pairs for a given concept.

    Args:
        concept: One of: sentiment, refusal, polite
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
    aggregator = _resolve_aggregator(config.method, config)

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

            positive_selected = select_token_activations(
                positive_activations[layer],
                config.read_token_index,
            )
            negative_selected = select_token_activations(
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
        concept: One of: sentiment, refusal, polite
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
    top_k: int
    layers: list[float] | None


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
        help="Concept to extract (sentiment, refusal, polite)",
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace model name (default: sshleifer/tiny-gpt2)",
    )
    parser.add_argument(
        "--method",
        choices=["mean", "pca", "weighted_mean", "discriminative"],
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
        "--top-k",
        type=int,
        default=100,
        help="Top-k tokens for discriminative method (default: 100)",
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
    parser.add_argument(
        "--layers",
        nargs="+",
        type=float,
        default=None,
        help="Relative layer positions (e.g., --layers 0.5 or --layers 0.4 0.5 0.6)",
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
    extraction_config = ExtractionConfig(
        method=args.method,
        top_k=args.top_k,
        layers=args.layers if args.layers else [0.4, 0.5, 0.6, 0.7, 0.8],
    )
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


def _run_cli() -> None:
    """Run CLI when invoked as entry point."""
    main()


if __name__ == "__main__":
    _run_cli()


__all__ = [
    "Aggregator",
    "extract_steering_vector",
    "extract_vector",
    "load_contrast_pairs",
    "load_polite_data",
    "load_sentiment_data",
    "load_refusal_data",
    "mean_aggregator",
    "pca_aggregator",
    "weighted_mean_aggregator",
    "discriminative_token_aggregator",
    "VALID_CONCEPTS",
]
