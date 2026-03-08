"""TDNV (Topic-Discriminative Normalized Variance) metrics computation.

Computes separability metrics for positive/negative contrast pairs across
all model layers to analyze steering vector effectiveness.

Usage:
    # CLI
    uv run python -m steering_geometry.tdnv --concept honesty --model Qwen/Qwen3.5-2B

    # Programmatic
    from steering_geometry.tdnv import compute_tdnv_for_concept
    result = compute_tdnv_for_concept("honesty", "Qwen/Qwen3.5-2B", num_pairs=500)
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor

from .config import ModelConfig, TDNVConfig
from .extract import VALID_CONCEPTS, load_contrast_pairs
from .models import HookedModel
from .types import TDNVLayerMetrics, TDNVResult
from .utils import ensure_dir, safe_model_name

EPS = 1e-8


@dataclass
class _TopicStats:
    mean: Tensor
    variance: float
    count: int


def _compute_per_topic_stats(
    activations: Tensor,
    topic_labels: list[int],
) -> dict[int, _TopicStats]:
    """Compute per-topic statistics from activations.

    Args:
        activations: Tensor of shape (n_samples, hidden_dim).
        topic_labels: List of topic labels for each sample.

    Returns:
        Dictionary mapping topic ID to statistics (mean, variance, count).
    """
    activations = activations.float()
    unique_topics = sorted(set(topic_labels))
    stats: dict[int, _TopicStats] = {}

    for topic in unique_topics:
        mask = [i for i, label in enumerate(topic_labels) if label == topic]
        if not mask:
            continue

        topic_activations = activations[mask]
        mean = topic_activations.mean(dim=0)
        centered = topic_activations - mean
        variance = float((centered**2).sum(dim=1).mean().item())
        count = len(mask)

        stats[topic] = _TopicStats(mean=mean, variance=variance, count=count)

    return stats


def compute_tdnv(
    pos_activations: Tensor,
    neg_activations: Tensor,
) -> TDNVLayerMetrics:
    """Compute TDNV metrics for a single layer.

    TDNV measures separability between positive and negative activations.
    Lower TDNV = better separability = easier to steer.

    Formula: TDNV = (var_pos + var_neg) / (2 * ||mean_pos - mean_neg||^2 + eps)

    Args:
        pos_activations: Positive class activations (n_pos_samples, hidden_dim).
        neg_activations: Negative class activations (n_neg_samples, hidden_dim).

    Returns:
        TDNVLayerMetrics with tdnv, norm_num, norm_den, energy.
    """
    pos_activations = pos_activations.float()
    neg_activations = neg_activations.float()

    combined = torch.cat([pos_activations, neg_activations], dim=0)
    energy = float((combined**2).sum(dim=1).mean().item())

    topic_labels = [0] * pos_activations.shape[0] + [1] * neg_activations.shape[0]
    stats = _compute_per_topic_stats(combined, topic_labels)

    if 0 not in stats or 1 not in stats:
        return TDNVLayerMetrics(tdnv=float("inf"), norm_num=0.0, norm_den=0.0, energy=energy)

    pos_stats = stats[0]
    neg_stats = stats[1]

    mean_diff = pos_stats.mean - neg_stats.mean
    mean_diff_sq = float((mean_diff**2).sum().item())

    avg_within_variance = (pos_stats.variance + neg_stats.variance) / 2.0

    tdnv = avg_within_variance / (2.0 * mean_diff_sq + EPS)

    norm_num = avg_within_variance / (energy + EPS) if energy > 0 else 0.0
    norm_den = mean_diff_sq / (energy + EPS) if energy > 0 else 0.0

    return TDNVLayerMetrics(
        tdnv=tdnv,
        norm_num=norm_num,
        norm_den=norm_den,
        energy=energy,
    )


def _select_token_activations(activations: Tensor, read_token_index: int) -> Tensor:
    """Select activations from a specific token position."""
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


def compute_tdnv_for_concept(
    concept: str,
    model_name: str,
    config: TDNVConfig | None = None,
) -> TDNVResult:
    """Compute TDNV metrics for all layers of a concept.

    Loads contrast pairs, extracts activations, and computes TDNV metrics
    for every layer in the model (0 to num_layers-1).

    Args:
        concept: Behavioral concept (honesty, sentiment, toxicity, sycophancy, refusal).
        model_name: HuggingFace model name.
        config: TDNV configuration (uses defaults if None).

    Returns:
        TDNVResult with layer-wise metrics.

    Raises:
        ValueError: If concept is invalid.
    """
    if concept not in VALID_CONCEPTS:
        msg = f"Invalid concept: {concept}. Valid concepts: {sorted(VALID_CONCEPTS)}"
        raise ValueError(msg)

    if config is None:
        config = TDNVConfig()

    pairs = load_contrast_pairs(concept, config.num_pairs)
    print(f"Loaded {len(pairs)} contrast pairs for {concept}")

    model = HookedModel(ModelConfig(model_name=model_name))
    layers = list(range(model.num_layers))
    print(f"Model has {model.num_layers} layers")

    pos_per_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    neg_per_layer: dict[int, list[Tensor]] = {layer: [] for layer in layers}

    for start in range(0, len(pairs), config.batch_size):
        batch = pairs[start : start + config.batch_size]
        pos_texts = [pair.positive for pair in batch]
        neg_texts = [pair.negative for pair in batch]

        pos_activations = model.get_activations(pos_texts, layers)
        neg_activations = model.get_activations(neg_texts, layers)

        for layer in layers:
            pos_selected = _select_token_activations(
                pos_activations[layer],
                config.read_token_index,
            )
            neg_selected = _select_token_activations(
                neg_activations[layer],
                config.read_token_index,
            )
            pos_per_layer[layer].append(pos_selected)
            neg_per_layer[layer].append(neg_selected)

    tdnv_values: list[float] = []
    norm_num_values: list[float] = []
    norm_den_values: list[float] = []
    layerwise_energy: list[float] = []

    for layer in layers:
        pos_batch = torch.cat(pos_per_layer[layer], dim=0)
        neg_batch = torch.cat(neg_per_layer[layer], dim=0)

        metrics = compute_tdnv(pos_batch, neg_batch)

        tdnv_values.append(metrics.tdnv)
        norm_num_values.append(metrics.norm_num)
        norm_den_values.append(metrics.norm_den)
        layerwise_energy.append(metrics.energy)

        print(f"Layer {layer}: TDNV={metrics.tdnv:.4f}, energy={metrics.energy:.4f}")

    return TDNVResult(
        concept=concept,
        model_name=model_name,
        num_pairs=len(pairs),
        layers=layers,
        tdnv_values=tdnv_values,
        norm_num_values=norm_num_values,
        norm_den_values=norm_den_values,
        layerwise_energy=layerwise_energy,
    )


def save_tdnv_result(result: TDNVResult, output_dir: Path) -> Path:
    """Save TDNV result to JSON file.

    Args:
        result: TDNV result to save.
        output_dir: Directory to save JSON file.

    Returns:
        Path to saved JSON file.
    """
    output_dir = ensure_dir(output_dir)
    model_slug = safe_model_name(result.model_name)
    output_file = output_dir / f"{result.concept}_{model_slug}.json"

    data = {
        "concept": result.concept,
        "model_name": result.model_name,
        "num_pairs": result.num_pairs,
        "layers": result.layers,
        "tdnv_values": result.tdnv_values,
        "norm_num_values": result.norm_num_values,
        "norm_den_values": result.norm_den_values,
        "layerwise_energy": result.layerwise_energy,
    }

    with output_file.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved TDNV results to {output_file}")
    return output_file


def plot_tdnv_trends(result: TDNVResult, plot_dir: Path) -> Path:
    """Plot TDNV trends across layers.

    Creates a visualization showing:
    - TDNV values per layer
    - Normalized numerator (within-topic variance) per layer
    - Normalized denominator (between-topic distance) per layer
    - Layerwise energy on secondary axis

    Args:
        result: TDNV result to visualize.
        plot_dir: Directory to save plot.

    Returns:
        Path to saved plot file.
    """
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(plot_dir)
    model_slug = safe_model_name(result.model_name)
    output_file = plot_dir / f"{result.concept}_{model_slug}.png"

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("TDNV / NormNum / NormDen")

    ax1.plot(
        result.layers,
        result.tdnv_values,
        "b-o",
        label="TDNV",
        markersize=4,
    )
    ax1.plot(
        result.layers,
        result.norm_num_values,
        "g-s",
        label="NormNum (within-topic var)",
        markersize=4,
    )
    ax1.plot(
        result.layers,
        result.norm_den_values,
        "r-^",
        label="NormDen (between-topic dist)",
        markersize=4,
    )

    ax1.set_yscale("log")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Layerwise Energy", color="purple")
    ax2.plot(
        result.layers,
        result.layerwise_energy,
        "purple",
        linestyle="--",
        marker="d",
        markersize=3,
        label="Energy",
    )
    ax2.tick_params(axis="y", labelcolor="purple")

    plt.title(f"TDNV Analysis: {result.concept} ({result.model_name})")
    fig.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_file}")
    return output_file


# =============================================================================
# CLI
# =============================================================================


class _Args(Protocol):
    concept: str
    model: str
    num_pairs: int
    output: str
    plot_dir: str
    dry_run: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steering_geometry.tdnv",
        description="Compute TDNV metrics for behavioral concepts",
    )
    parser.add_argument(
        "--concept",
        required=True,
        choices=sorted(VALID_CONCEPTS),
        help="Concept to analyze (honesty, sentiment, toxicity, sycophancy, refusal)",
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace model name (default: sshleifer/tiny-gpt2)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=500,
        help="Number of contrast pairs (default: 500)",
    )
    parser.add_argument(
        "--output",
        default="data/tdnv/",
        help="Output directory for JSON results (default: data/tdnv/)",
    )
    parser.add_argument(
        "--plot-dir",
        default="plot/tdnv/",
        help="Output directory for plots (default: plot/tdnv/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data only, skip model loading and computation",
    )
    return parser


def main() -> None:
    args = cast(_Args, cast(object, _build_parser().parse_args()))

    pairs = load_contrast_pairs(args.concept, args.num_pairs)
    print(f"Loaded {len(pairs)} contrast pairs for {args.concept}")

    if args.dry_run:
        print("Dry run complete")
        return

    config = TDNVConfig(
        num_pairs=args.num_pairs,
        output_dir=args.output,
        plot_dir=args.plot_dir,
    )

    result = compute_tdnv_for_concept(
        concept=args.concept,
        model_name=args.model,
        config=config,
    )

    save_tdnv_result(result, Path(args.output))
    plot_tdnv_trends(result, Path(args.plot_dir))


if __name__ == "__main__":
    main()


__all__ = [
    "compute_tdnv",
    "compute_tdnv_for_concept",
    "save_tdnv_result",
    "plot_tdnv_trends",
    "TDNVConfig",
    "TDNVLayerMetrics",
    "TDNVResult",
]
