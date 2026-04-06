"""TDNV (Topic-Discriminative Normalized Variance) metrics computation.

Computes separability metrics for positive/negative contrast pairs across
all model layers to analyze steering vector effectiveness.

Usage:
    # CLI - concept mode (default)
    uv run python -m steering_geometry.tdnv --concept honesty --model Qwen/Qwen3.5-2B

    # CLI - MMLU mode
    uv run python -m steering_geometry.tdnv --mode mmlu --model Qwen/Qwen3.5-2B --num-questions 100

    # Programmatic
    from steering_geometry.tdnv import compute_tdnv_for_concept
    result = compute_tdnv_for_concept("honesty", "Qwen/Qwen3.5-2B", num_pairs=500)

    from steering_geometry.tdnv import compute_tdnv_for_mmlu
    result = compute_tdnv_for_mmlu("Qwen/Qwen3.5-2B", num_questions=100)
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor

from .config import MMLUConfig, ModelConfig, TDNVConfig
from .extract import VALID_CONCEPTS, load_contrast_pairs
from .models import HookedModel
from .types import MMLUQuestion, TDNVLayerMetrics, TDNVResult
from .utils import ensure_dir, safe_model_name

EPS = 1e-8


def select_last_n_tokens(activations: Tensor, n: int) -> Tensor:
    """Select last n tokens from activations tensor.

    Args:
        activations: Tensor of shape (num_tokens, hidden_dim)
        n: Number of tokens to select from the end

    Returns:
        Tensor of shape (min(n, num_tokens), hidden_dim)
    """
    if n <= 0:
        return activations.new_zeros((0, activations.shape[1]))
    num_tokens = activations.shape[0]
    if n >= num_tokens:
        return activations.clone()
    return activations[-n:].clone()


def select_top_k_discriminative(
    activations: Tensor,
    labels: list[int],
    k: int,
) -> Tensor:
    """Select top-k discriminative tokens per class.

    Scoring formula: s_i = Σ_{c ≠ own} ||h_i - μ_c||² - ||h_i - μ_same||²
    For binary: s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
    For multi-class: sums distance to ALL other class centroids.
    Higher score = token is closer to own class, farther from all other classes.

    Args:
        activations: Tensor of shape (total_tokens, hidden_dim)
        labels: Class label for each token (same length as activations)
        k: Number of tokens to select per class

    Returns:
        Concatenated tensor of selected tokens from all classes
    """
    unique_labels = sorted(set(labels))
    selected_tokens: list[Tensor] = []

    # Compute centroids for all classes
    centroids: dict[int, Tensor] = {}
    for label in unique_labels:
        mask = [i for i, lbl in enumerate(labels) if lbl == label]
        class_activations = activations[mask]
        centroids[label] = class_activations.mean(dim=0)

    # For each class, compute scores and select top-k
    for label in unique_labels:
        mask = [i for i, lbl in enumerate(labels) if lbl == label]
        class_activations = activations[mask]
        own_centroid = centroids[label]

        # Score: Σ_{c ≠ own} ||h - μ_c||² - ||h - μ_same||²
        # Sum distances to ALL other class centroids (not averaged)
        dist_to_others = torch.zeros(class_activations.shape[0], dtype=class_activations.dtype)
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_centroid = centroids[other_label]
            dist_to_others = dist_to_others + ((class_activations - other_centroid) ** 2).sum(dim=1)

        dist_to_own = ((class_activations - own_centroid) ** 2).sum(dim=1)
        scores = dist_to_others - dist_to_own

        # Select top-k
        k_actual = min(k, class_activations.shape[0])
        _, top_indices = torch.topk(scores, k_actual)
        selected_tokens.append(class_activations[top_indices])

    return torch.cat(selected_tokens, dim=0)


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


def compute_tdnv_multi_concept(
    concepts: dict[str, tuple[Tensor, Tensor]],
) -> TDNVLayerMetrics:
    """Compute TDNV across multiple binary concepts.

    TDNV measures separability across all groups from multiple binary concepts.
    Lower TDNV = better overall separability.

    Formula: TDNV = (1/M(M-1)) * Σ (var_g + var_g') / (2||mean_g - mean_g'||²)
    Where:
        - M = total number of groups (2T for T binary concepts)
        - var_g = within-group variance for group g
        - mean_g = centroid for group g

    Args:
        concepts: Dict mapping concept name to (pos_activations, neg_activations).
                  e.g., {"polite": (pos_tensor, neg_tensor), "sentiment": (pos_tensor, neg_tensor)}

    Returns:
        TDNVLayerMetrics with tdnv, norm_num, norm_den, energy.
    """
    if not concepts:
        return TDNVLayerMetrics(tdnv=float("inf"), norm_num=0.0, norm_den=0.0, energy=0.0)

    all_groups: list[Tensor] = []
    for _concept_name, (pos_acts, neg_acts) in concepts.items():
        pos_acts = pos_acts.float()
        neg_acts = neg_acts.float()
        all_groups.append(pos_acts)
        all_groups.append(neg_acts)

    combined = torch.cat(all_groups, dim=0)
    energy = float((combined**2).sum(dim=1).mean().item())

    M = len(all_groups)  # noqa: N806  # Mathematical notation for TDNV formula

    if M < 2:
        return TDNVLayerMetrics(tdnv=float("inf"), norm_num=0.0, norm_den=0.0, energy=energy)

    topic_labels = []
    for group_idx, group in enumerate(all_groups):
        topic_labels.extend([group_idx] * group.shape[0])

    stats = _compute_per_topic_stats(combined, topic_labels)

    total_pairwise_tdnv = 0.0
    total_norm_num = 0.0
    total_norm_den = 0.0
    pair_count = 0

    for g_idx in range(M):
        if g_idx not in stats:
            continue
        for g_prime_idx in range(M):
            if g_prime_idx <= g_idx:
                continue
            if g_prime_idx not in stats:
                continue

            g_stats = stats[g_idx]
            g_prime_stats = stats[g_prime_idx]

            mean_diff = g_stats.mean - g_prime_stats.mean
            mean_diff_sq = float((mean_diff**2).sum().item())

            avg_variance = (g_stats.variance + g_prime_stats.variance) / 2.0

            pairwise_tdnv = avg_variance / (2.0 * mean_diff_sq + EPS)
            total_pairwise_tdnv += pairwise_tdnv

            total_norm_num += avg_variance / (energy + EPS) if energy > 0 else 0.0
            total_norm_den += mean_diff_sq / (energy + EPS) if energy > 0 else 0.0

            pair_count += 1

    if pair_count == 0:
        return TDNVLayerMetrics(tdnv=float("inf"), norm_num=0.0, norm_den=0.0, energy=energy)

    # Normalize by M(M-1) as per formula
    normalization_factor = M * (M - 1)
    tdnv = total_pairwise_tdnv / normalization_factor
    norm_num = total_norm_num / normalization_factor
    norm_den = total_norm_den / normalization_factor

    return TDNVLayerMetrics(
        tdnv=tdnv,
        norm_num=norm_num,
        norm_den=norm_den,
        energy=energy,
    )


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


def compute_tdnv_mmlu(
    category_activations: dict[str, Tensor],
) -> TDNVLayerMetrics:
    """Compute TDNV across MMLU-Pro categories.

    Each category is treated as a separate group (multi-class, not binary).
    Formula: TDNV = (1/M(M-1)) * Σ (var_g + var_g') / (2||mean_g - mean_g'||²)

    Args:
        category_activations: Dict mapping category name to activations tensor
                        e.g., {"math": tensor(10, 64), "physics": tensor(10, 64)}

    Returns:
        TDNVLayerMetrics with tdnv, norm_num, norm_den, energy
    """
    if not category_activations:
        return TDNVLayerMetrics(tdnv=float("inf"), norm_num=0.0, norm_den=0.0, energy=0.0)

    # Collect all activations and compute energy
    all_activations: list[Tensor] = []
    category_names = list(category_activations.keys())

    for name in category_names:
        all_activations.append(category_activations[name].float())

    combined = torch.cat(all_activations, dim=0)
    energy = float((combined**2).sum(dim=1).mean().item())

    topic_labels: list[int] = []
    for i, name in enumerate(category_names):
        topic_labels.extend([i] * category_activations[name].shape[0])

    stats = _compute_per_topic_stats(combined, topic_labels)
    M = len(stats)  # noqa: N806  # Mathematical notation for TDNV formula

    if M < 2:
        single_variance = stats[0].variance if 0 in stats else 0.0
        norm_num = single_variance / (energy + EPS) if energy > 0 else 0.0
        return TDNVLayerMetrics(
            tdnv=norm_num,
            norm_num=norm_num,
            norm_den=0.0,
            energy=energy,
        )

    tdnv_sum = 0.0
    norm_num_sum = 0.0
    norm_den_sum = 0.0
    pair_count = 0

    topic_ids = sorted(stats.keys())
    for i, topic_i in enumerate(topic_ids):
        for topic_j in topic_ids[i + 1 :]:
            stats_i = stats[topic_i]
            stats_j = stats[topic_j]

            mean_diff = stats_i.mean - stats_j.mean
            mean_diff_sq = float((mean_diff**2).sum().item())

            avg_within_variance = (stats_i.variance + stats_j.variance) / 2.0

            pair_tdnv = avg_within_variance / (2.0 * mean_diff_sq + EPS)
            tdnv_sum += pair_tdnv

            norm_num_sum += avg_within_variance
            norm_den_sum += mean_diff_sq
            pair_count += 1

    tdnv_avg = tdnv_sum / pair_count if pair_count > 0 else float("inf")
    norm_num_avg = norm_num_sum / pair_count if pair_count > 0 else 0.0
    norm_den_avg = norm_den_sum / pair_count if pair_count > 0 else 0.0

    return TDNVLayerMetrics(
        tdnv=tdnv_avg,
        norm_num=norm_num_avg / (energy + EPS) if energy > 0 else 0.0,
        norm_den=norm_den_avg / (energy + EPS) if energy > 0 else 0.0,
        energy=energy,
    )


def compute_tdnv_for_concept(
    concept: str,
    model_name: str,
    config: TDNVConfig | None = None,
    last_n: int | None = None,
    top_k: int | None = None,
) -> TDNVResult:
    """Compute TDNV metrics for all layers of a concept.

    Loads contrast pairs, extracts activations, and computes TDNV metrics
    for every layer in the model (0 to num_layers-1).

    Args:
        concept: Behavioral concept (honesty, sentiment, toxicity, sycophancy, refusal).
        model_name: HuggingFace model name.
        config: TDNV configuration (uses defaults if None).
        last_n: If provided, use only last n tokens from activations.
                None means use all tokens. Must be positive.
        top_k: If provided, use only top-k discriminative tokens per class.
               Uses scoring: s_i = ||h_i - μ_other||² - ||h_i - μ_same||².
               Mutually exclusive with last_n.

    Returns:
        TDNVResult with layer-wise metrics.

    Raises:
        ValueError: If concept is invalid, last_n is not positive, or both
                    last_n and top_k are specified.
    """
    if concept not in VALID_CONCEPTS:
        msg = f"Invalid concept: {concept}. Valid concepts: {sorted(VALID_CONCEPTS)}"
        raise ValueError(msg)

    if last_n is not None and last_n <= 0:
        msg = f"last_n must be positive, got {last_n}"
        raise ValueError(msg)

    if top_k is not None and top_k <= 0:
        msg = f"top_k must be positive, got {top_k}"
        raise ValueError(msg)

    if last_n is not None and top_k is not None:
        msg = "Cannot specify both last_n and top_k"
        raise ValueError(msg)

    if config is None:
        config = TDNVConfig()

    pairs = load_contrast_pairs(concept, config.num_pairs)
    print(f"Loaded {len(pairs)} contrast pairs for {concept}")
    if last_n is not None:
        print(f"Using last {last_n} tokens per sample")
    if top_k is not None:
        print(f"Using top-{top_k} discriminative tokens per class")

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
            pos_flat = pos_activations[layer].reshape(-1, pos_activations[layer].shape[-1])
            neg_flat = neg_activations[layer].reshape(-1, neg_activations[layer].shape[-1])
            pos_non_zero = pos_flat[pos_flat.abs().sum(dim=-1) > 0]
            neg_non_zero = neg_flat[neg_flat.abs().sum(dim=-1) > 0]

            # Apply last_n token selection if specified
            if last_n is not None:
                pos_non_zero = select_last_n_tokens(pos_non_zero, last_n)
                neg_non_zero = select_last_n_tokens(neg_non_zero, last_n)

            pos_per_layer[layer].append(pos_non_zero)
            neg_per_layer[layer].append(neg_non_zero)

    tdnv_values: list[float] = []
    norm_num_values: list[float] = []
    norm_den_values: list[float] = []
    layerwise_energy: list[float] = []

    for layer in layers:
        pos_batch = torch.cat(pos_per_layer[layer], dim=0)
        neg_batch = torch.cat(neg_per_layer[layer], dim=0)

        if top_k is not None:
            combined = torch.cat([pos_batch, neg_batch], dim=0)
            labels = [0] * pos_batch.shape[0] + [1] * neg_batch.shape[0]
            selected = select_top_k_discriminative(combined, labels, top_k)
            selected_labels = [0] * min(top_k, pos_batch.shape[0]) + [1] * min(
                top_k, neg_batch.shape[0]
            )
            pos_batch = selected[[i for i, lbl in enumerate(selected_labels) if lbl == 0]]
            neg_batch = selected[[i for i, lbl in enumerate(selected_labels) if lbl == 1]]

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


def compute_tdnv_for_mmlu(
    model_name: str,
    mmlu_config: MMLUConfig | None = None,
    tdnv_config: TDNVConfig | None = None,
    categories: list[str] | None = None,
    last_n: int | None = None,
    top_k: int | None = None,
) -> TDNVResult:
    """Compute TDNV metrics across MMLU-Pro categories for all layers.

    Loads MMLU-Pro validation set, groups questions by category, extracts
    activations, and computes TDNV metrics for every layer in the model.
    Each MMLU category (e.g., math, physics, chemistry) is treated as a
    separate group for multi-class separability analysis.

    Args:
        model_name: HuggingFace model name.
        mmlu_config: MMLU configuration (uses defaults if None).
        tdnv_config: TDNV configuration (uses defaults if None).
        categories: Optional list of category names to include.
                    None means include all categories from the dataset.
        last_n: If provided, use only last n tokens from activations.
                None means use all tokens. Must be positive.
        top_k: If provided, use only top-k discriminative tokens per class.
               Uses scoring: s_i = Σ_{c ≠ own} ||h_i - μ_c||² - ||h_i - μ_same||².
               Mutually exclusive with last_n.

    Returns:
        TDNVResult with layer-wise TDNV metrics across MMLU categories.

    Raises:
        ValueError: If last_n or top_k is not positive, or both are specified,
                    or no categories have data.
    """
    if last_n is not None and last_n <= 0:
        msg = f"last_n must be positive, got {last_n}"
        raise ValueError(msg)

    if top_k is not None and top_k <= 0:
        msg = f"top_k must be positive, got {top_k}"
        raise ValueError(msg)

    if last_n is not None and top_k is not None:
        msg = "Cannot specify both last_n and top_k"
        raise ValueError(msg)

    if mmlu_config is None:
        mmlu_config = MMLUConfig()

    if tdnv_config is None:
        tdnv_config = TDNVConfig()

    from datasets import load_dataset  # type: ignore[import-untyped]

    print("Loading MMLU-Pro dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    questions: list[MMLUQuestion] = list(ds)
    random.seed(mmlu_config.seed)
    random.shuffle(questions)
    questions = questions[: mmlu_config.num_questions]
    print(f"Loaded {len(questions)} MMLU questions")

    category_texts: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        cat = q.get("category", "unknown")
        if categories is not None and cat not in categories:
            continue
        parts = [f"Question: {q.get('question', '')}"]
        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        opts = q.get("options", [])
        for i, opt in enumerate(opts[:10]):
            if i < len(labels):
                parts.append(f"{labels[i]}. {opt}")
        parts.append("Answer:")
        category_texts[cat].append("\n".join(parts))

    if not category_texts:
        msg = "No MMLU categories found after filtering"
        raise ValueError(msg)

    sorted_categories = sorted(category_texts.keys())
    print(f"Categories ({len(sorted_categories)}): {', '.join(sorted_categories)}")

    model = HookedModel(ModelConfig(model_name=model_name))
    layers = list(range(model.num_layers))
    print(f"Model has {model.num_layers} layers")
    if top_k is not None:
        print(f"Using top-{top_k} discriminative tokens per category")

    cat_activations_per_layer: dict[int, dict[str, list[Tensor]]] = {
        layer: {cat: [] for cat in sorted_categories} for layer in layers
    }

    for cat in sorted_categories:
        texts = category_texts[cat]
        print(f"Processing category '{cat}' ({len(texts)} questions)...")

        for start in range(0, len(texts), tdnv_config.batch_size):
            batch = texts[start : start + tdnv_config.batch_size]
            activations = model.get_activations(batch, layers)

            for layer in layers:
                flat = activations[layer].reshape(-1, activations[layer].shape[-1])
                non_zero = flat[flat.abs().sum(dim=-1) > 0]

                if last_n is not None:
                    non_zero = select_last_n_tokens(non_zero, last_n)

                cat_activations_per_layer[layer][cat].append(non_zero)

    tdnv_values: list[float] = []
    norm_num_values: list[float] = []
    norm_den_values: list[float] = []
    layerwise_energy: list[float] = []

    for layer in layers:
        category_acts: dict[str, Tensor] = {}
        for cat in sorted_categories:
            tensors = cat_activations_per_layer[layer][cat]
            if tensors:
                category_acts[cat] = torch.cat(tensors, dim=0)

        if top_k is not None and len(category_acts) >= 2:
            combined_list: list[Tensor] = []
            cat_labels: list[int] = []
            for cat_idx, cat in enumerate(sorted_categories):
                if cat in category_acts:
                    combined_list.append(category_acts[cat])
                    cat_labels.extend([cat_idx] * category_acts[cat].shape[0])

            combined = torch.cat(combined_list, dim=0)
            selected = select_top_k_discriminative(combined, cat_labels, top_k)

            selected_per_cat: dict[str, Tensor] = {}
            offset = 0
            for _cat_idx, cat in enumerate(sorted_categories):
                if cat not in category_acts:
                    continue
                cat_count = category_acts[cat].shape[0]
                k_actual = min(top_k, cat_count)
                selected_per_cat[cat] = selected[offset : offset + k_actual]
                offset += k_actual

            category_acts = selected_per_cat

        metrics = compute_tdnv_mmlu(category_acts)

        tdnv_values.append(metrics.tdnv)
        norm_num_values.append(metrics.norm_num)
        norm_den_values.append(metrics.norm_den)
        layerwise_energy.append(metrics.energy)

        print(f"Layer {layer}: TDNV={metrics.tdnv:.4f}, energy={metrics.energy:.4f}")

    total_used = sum(len(texts) for texts in category_texts.values())
    return TDNVResult(
        concept="mmlu",
        model_name=model_name,
        num_pairs=total_used,
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
    output_file = plot_dir / f"{result.concept}_{model_slug}.pdf"

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
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_file}")
    return output_file


def plot_stability_trend(
    results: list[TDNVResult],
    param_name: str,
    param_values: list[float | int],
    output_path: Path,
) -> Path:
    """Plot TDNV stability trends across parameter values.

    Args:
        results: List of TDNVResult objects (one per parameter value)
        param_name: Name of the parameter being varied (e.g., "Dataset Size", "Seed")
        param_values: List of parameter values corresponding to each result
        output_path: Path to save the PDF plot

    Returns:
        Path to the saved PDF file

    Creates:
        - PDF plot with:
          - X-axis: parameter values
          - Y-axis: TDNV values (log scale)
          - Multiple lines (one per layer, different colors)
          - Legend showing layer indices
          - Title: "TDNV vs {param_name}"
    """
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("Results list cannot be empty")

    if len(results) != len(param_values):
        msg = (
            f"Number of results ({len(results)}) must match "
            f"number of param_values ({len(param_values)})"
        )
        raise ValueError(msg)

    layers = results[0].layers
    num_layers = len(layers)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(num_layers - 1, 1)) for i in range(num_layers)]

    fig, ax = plt.subplots(figsize=(12, 6))

    for layer_idx, layer in enumerate(layers):
        tdnv_across_params = []
        for result in results:
            if layer in result.layers:
                layer_pos = result.layers.index(layer)
                tdnv_across_params.append(result.tdnv_values[layer_pos])
            else:
                tdnv_across_params.append(float("nan"))

        ax.plot(
            param_values,
            tdnv_across_params,
            "-o",
            color=colors[layer_idx],
            label=f"Layer {layer}",
            markersize=4,
        )

    ax.set_xlabel(param_name)
    ax.set_ylabel("TDNV")
    ax.set_title(f"TDNV vs {param_name}")
    ax.set_yscale("log")
    ax.legend(loc="best", ncol=2, fontsize="small")
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved stability trend plot to {output_path}")
    return output_path


# =============================================================================
# CLI
# =============================================================================


class _Args(Protocol):
    mode: str
    concept: str
    model: str
    num_pairs: int
    num_questions: int
    mmlu_seed: int
    categories: str
    output: str
    plot_dir: str
    dry_run: bool
    last_n: int | None
    top_k: int | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steering_geometry.tdnv",
        description="Compute TDNV metrics for behavioral concepts or MMLU categories",
    )
    parser.add_argument(
        "--mode",
        choices=["concept", "mmlu"],
        default="concept",
        help=(
            "Analysis mode: 'concept' for contrast-pair TDNV, "
            "'mmlu' for MMLU category TDNV (default: concept)"
        ),
    )
    parser.add_argument(
        "--concept",
        default=None,
        choices=sorted(VALID_CONCEPTS),
        help="Concept to analyze (required for --mode concept)",
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
        help="Number of contrast pairs for --mode concept (default: 500)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of MMLU questions for --mode mmlu (default: 100)",
    )
    parser.add_argument(
        "--mmlu-seed",
        type=int,
        default=42,
        help="Random seed for MMLU question sampling (default: 42)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated MMLU categories to include (default: all categories)",
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
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Use only last N tokens from activations (default: all tokens)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Use only top-K discriminative tokens per class (default: all tokens)",
    )
    return parser


def main() -> None:
    args = cast(_Args, cast(object, _build_parser().parse_args()))

    if args.mode == "mmlu":
        _run_mmlu(args)
    else:
        _run_concept(args)


def _run_concept(args: _Args) -> None:
    if args.concept is None:
        print("Error: --concept is required for --mode concept")
        raise SystemExit(1)

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
        last_n=args.last_n,
        top_k=args.top_k,
    )

    save_tdnv_result(result, Path(args.output))
    plot_tdnv_trends(result, Path(args.plot_dir))


def _run_mmlu(args: _Args) -> None:
    print("MMLU TDNV analysis mode")

    if args.dry_run:
        print("Dry run complete")
        return

    mmlu_config = MMLUConfig(num_questions=args.num_questions, seed=args.mmlu_seed)
    tdnv_config = TDNVConfig(output_dir=args.output, plot_dir=args.plot_dir)

    parsed_categories: list[str] | None = None
    if args.categories:
        parsed_categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    result = compute_tdnv_for_mmlu(
        model_name=args.model,
        mmlu_config=mmlu_config,
        tdnv_config=tdnv_config,
        categories=parsed_categories,
        last_n=args.last_n,
        top_k=args.top_k,
    )

    output_dir = Path(args.output)
    plot_dir = Path(args.plot_dir)

    model_slug = safe_model_name(result.model_name)
    output_file = ensure_dir(output_dir) / f"mmlu_{model_slug}.json"
    with output_file.open("w") as f:
        json.dump(
            {
                "concept": result.concept,
                "model_name": result.model_name,
                "num_pairs": result.num_pairs,
                "layers": result.layers,
                "tdnv_values": result.tdnv_values,
                "norm_num_values": result.norm_num_values,
                "norm_den_values": result.norm_den_values,
                "layerwise_energy": result.layerwise_energy,
                "num_questions": args.num_questions,
                "mmlu_seed": args.mmlu_seed,
                "categories": parsed_categories,
            },
            f,
            indent=2,
        )
    print(f"Saved MMLU TDNV results to {output_file}")

    plot_tdnv_trends(result, plot_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "select_last_n_tokens",
    "select_top_k_discriminative",
    "compute_tdnv",
    "compute_tdnv_for_concept",
    "compute_tdnv_for_mmlu",
    "compute_tdnv_multi_concept",
    "compute_tdnv_mmlu",
    "save_tdnv_result",
    "plot_tdnv_trends",
    "plot_stability_trend",
    "TDNVConfig",
    "TDNVLayerMetrics",
    "TDNVResult",
]
