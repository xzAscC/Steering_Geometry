"""Stability comparison and vector analysis for steering vector extraction.

This module provides:
1. Vector analysis utilities for computing cosine similarity matrices,
   generating heatmaps, and managing vector persistence
2. Stability comparison experiments that compare diff_means vs discriminative
   token selection by running multiple extractions and computing pairwise
   cosine similarity across layers.
"""

import datetime
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from numpy import ndarray
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]
from torch import Tensor

from steering_geometry.config import ExtractionConfig, ModelConfig, StabilityComparisonConfig
from steering_geometry.extract import extract_steering_vector, extract_vector, load_contrast_pairs
from steering_geometry.models import HookedModel
from steering_geometry.types import ContrastPair
from steering_geometry.utils import ensure_dir

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Vector Analysis Functions (merged from vector_analysis.py)
# =============================================================================


def compute_cosine_similarity_matrix(vectors: list[Tensor]) -> ndarray:
    """Compute pairwise cosine similarity matrix for a list of vectors.

    Args:
        vectors: List of 1D tensors to compute similarities between.

    Returns:
        2D numpy array of shape (n_vectors, n_vectors) containing pairwise
        cosine similarities. Values range from -1 to 1.

    Raises:
        ValueError: If vectors list is empty or tensors have incompatible shapes.
    """
    if not vectors:
        msg = "Cannot compute similarity matrix for empty vector list"
        raise ValueError(msg)

    stacked = torch.stack(vectors)
    matrix = stacked.cpu().numpy()

    return cast("ndarray", cosine_similarity(matrix))


def plot_heatmap(
    matrix: ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> Path:
    """Generate and save a heatmap visualization of a similarity matrix.

    Args:
        matrix: 2D numpy array to visualize as heatmap.
        labels: List of labels for both axes (must match matrix dimensions).
        title: Title for the plot.
        output_path: Path where the plot will be saved (PDF format).

    Returns:
        Path to the saved plot file.

    Raises:
        ValueError: If labels length doesn't match matrix dimensions.
    """
    import matplotlib.pyplot as plt

    if len(labels) != matrix.shape[0] or len(labels) != matrix.shape[1]:
        msg = (
            f"Labels length ({len(labels)}) must match matrix dimensions "
            f"({matrix.shape[0]}x{matrix.shape[1]})"
        )
        raise ValueError(msg)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", fontsize=10)

    ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()

    logger.info("Saved heatmap to %s", output_path)
    return output_path


def save_vector(vector: Tensor, path: Path) -> None:
    """Save a tensor to disk as a .pt file.

    Args:
        vector: Tensor to save.
        path: Destination path (will create parent directories if needed).
    """
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(vector, path)
    logger.debug("Saved vector to %s", path)


def load_vector(path: Path) -> Tensor:
    """Load a tensor from a .pt file.

    Args:
        path: Path to the .pt file.

    Returns:
        Loaded tensor.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Vector file not found: {path}"
        raise FileNotFoundError(msg)

    vector = cast("Tensor", torch.load(path, weights_only=True))
    logger.debug("Loaded vector from %s", path)
    return vector


def cap_examples(requested: int, max_available: int, concept: str) -> int:
    """Cap the number of requested examples to available maximum.

    Logs a warning if the requested amount exceeds availability.

    Args:
        requested: Number of examples requested.
        max_available: Maximum number of examples available.
        concept: Name of the concept (for logging).

    Returns:
        The capped number of examples (min of requested and max_available).
    """
    if requested > max_available:
        logger.warning(
            "Requested %d examples for concept '%s' but only %d available. Using %d instead.",
            requested,
            concept,
            max_available,
            max_available,
        )
        return max_available
    return requested


def run_diff_means_experiment(
    concept: str,
    n_examples_list: list[int],
    layers: list[float],
    model_name: str,
    output_dir: Path | str = "outputs",
) -> dict[str, dict[str, str] | dict[str, str] | dict[str, dict[str, float]]]:
    """Run differential means experiment across varying example counts.

    For each example count, extracts steering vectors and computes pairwise
    cosine similarities across all example counts at each layer.

    Args:
        concept: Concept to extract (e.g., "honesty", "toxicity").
        n_examples_list: List of example counts to test (e.g., [10, 30, 100]).
        layers: Relative layer positions (0.0-1.0) to analyze.
        model_name: HuggingFace model name.
        output_dir: Base output directory for vectors and heatmaps.

    Returns:
        Dict with:
            - "vector_paths": Dict mapping (n_examples, layer) to vector file paths
            - "heatmap_paths": Dict mapping layer to heatmap file paths
            - "statistics": Dict with mean/min/max similarities per layer

    Raises:
        ValueError: If n_examples_list is empty or all vectors contain NaN.
    """
    if not n_examples_list:
        msg = "n_examples_list cannot be empty"
        raise ValueError(msg)

    output_dir = Path(output_dir)

    logger.info("Loading contrast pairs for concept '%s'", concept)
    all_pairs = load_contrast_pairs(concept, num_pairs=10000)
    max_available = len(all_pairs)
    logger.info("Dataset has %d examples available for concept '%s'", max_available, concept)

    vector_paths: dict[tuple[int, float], Path] = {}
    layer_vectors: dict[float, dict[int, Tensor]] = {layer_frac: {} for layer_frac in layers}

    for n_examples in n_examples_list:
        capped = cap_examples(n_examples, max_available, concept)

        logger.info(
            "Extracting vector for concept='%s', n_examples=%d (capped=%d)",
            concept,
            n_examples,
            capped,
        )

        steering_vector = extract_vector(
            concept=concept,
            model_name=model_name,
            num_pairs=capped,
            method="mean",
            layers=layers,
        )

        for layer_frac, abs_idx in zip(
            layers, steering_vector.layer_activations.keys(), strict=True
        ):
            vector = steering_vector.layer_activations[abs_idx]

            if torch.isnan(vector).any():
                msg = (
                    f"Vector for concept='{concept}', n={n_examples}, "
                    f"layer={layer_frac} contains NaN"
                )
                raise ValueError(msg)

            vector_path = (
                output_dir
                / "vectors"
                / concept
                / "diff_means"
                / f"n{n_examples}_layer{layer_frac}.pt"
            )
            save_vector(vector, vector_path)
            vector_paths[(n_examples, layer_frac)] = vector_path
            layer_vectors[layer_frac][n_examples] = vector

    heatmap_paths: dict[float, Path] = {}
    statistics: dict[float, dict[str, float]] = {}

    labels = [str(n) for n in n_examples_list]

    for layer_frac in layers:
        vectors = [layer_vectors[layer_frac][n] for n in n_examples_list]

        first = vectors[0]
        all_identical = all(torch.equal(v, first) for v in vectors)
        if all_identical:
            logger.warning(
                "All vectors identical at layer %.2f for concept '%s'",
                layer_frac,
                concept,
            )

        similarity_matrix = compute_cosine_similarity_matrix(vectors)

        off_diagonal_mask = ~torch.eye(len(vectors), dtype=torch.bool).numpy()
        off_diagonal_values = similarity_matrix[off_diagonal_mask]

        # Handle edge case of single vector (no off-diagonal elements)
        if len(off_diagonal_values) > 0:
            statistics[layer_frac] = {
                "mean_similarity": float(off_diagonal_values.mean()),
                "min_similarity": float(off_diagonal_values.min()),
                "max_similarity": float(off_diagonal_values.max()),
            }
        else:
            statistics[layer_frac] = {
                "mean_similarity": 1.0,
                "min_similarity": 1.0,
                "max_similarity": 1.0,
            }

        heatmap_path = output_dir / "heatmaps" / "diff_means" / f"{concept}_layer{layer_frac}.pdf"
        title = f"Cosine Similarity: {concept} (layer {layer_frac})"
        plot_heatmap(similarity_matrix, labels, title, heatmap_path)
        heatmap_paths[layer_frac] = heatmap_path

    logger.info("Completed diff_means experiment for concept '%s'", concept)

    return {
        "vector_paths": {f"n{k[0]}_layer{k[1]}": str(v) for k, v in vector_paths.items()},
        "heatmap_paths": {f"layer{k}": str(v) for k, v in heatmap_paths.items()},
        "statistics": {f"layer{k}": v for k, v in statistics.items()},
    }


def run_discriminative_experiment(
    concept: str,
    k_values: list[int],
    layers: list[float],
    model_name: str,
    output_dir: Path | str = "outputs",
) -> dict[str, dict[str, str] | dict[str, str] | dict[str, dict[str, float]]]:
    """Run discriminative token selection experiment across varying K values.

    For each K value, extracts steering vectors using discriminative token selection
    and computes pairwise cosine similarities across all K values at each layer.

    Args:
        concept: Concept to extract (e.g., "honesty", "toxicity").
        k_values: List of top_k values to test (e.g., [16, 32, 64, 128]).
        layers: Relative layer positions (0.0-1.0) to analyze.
        model_name: HuggingFace model name.
        output_dir: Base output directory for vectors and heatmaps.

    Returns:
        Dict with:
            - "vector_paths": Dict mapping (k_value, layer) to vector file paths
            - "heatmap_paths": Dict mapping layer to heatmap file paths
            - "statistics": Dict with mean/min/max similarities per layer

    Raises:
        ValueError: If k_values is empty, contains non-positive values,
            or all vectors contain NaN.
    """
    if not k_values:
        msg = "k_values cannot be empty"
        raise ValueError(msg)

    for k in k_values:
        if k <= 0:
            msg = f"All k_values must be positive integers, got {k}"
            raise ValueError(msg)

    output_dir = Path(output_dir)

    # Use 100x max(k_values) for sufficient token diversity when varying top_k
    num_pairs = max(k_values) * 100

    logger.info("Loading contrast pairs for concept '%s'", concept)
    all_pairs = load_contrast_pairs(concept, num_pairs=num_pairs)
    logger.info("Loaded %d contrast pairs for concept '%s'", len(all_pairs), concept)

    logger.info("Loading model '%s'", model_name)
    model = HookedModel(ModelConfig(model_name=model_name))

    vector_paths: dict[tuple[int, float], Path] = {}
    layer_vectors: dict[float, dict[int, Tensor]] = {layer_frac: {} for layer_frac in layers}

    for top_k in k_values:
        logger.info(
            "Extracting vector for concept='%s', method='discriminative', top_k=%d",
            concept,
            top_k,
        )

        extraction_config = ExtractionConfig(
            layers=layers,
            method="discriminative",
            top_k=top_k,
        )

        steering_vector = extract_steering_vector(
            model=model,
            pairs=all_pairs,
            config=extraction_config,
        )

        for layer_frac, abs_idx in zip(
            layers, steering_vector.layer_activations.keys(), strict=True
        ):
            vector = steering_vector.layer_activations[abs_idx]

            if torch.isnan(vector).any():
                msg = f"Vector for concept='{concept}', k={top_k}, layer={layer_frac} contains NaN"
                raise ValueError(msg)

            vector_path = (
                output_dir
                / "vectors"
                / concept
                / "discriminative"
                / f"k{top_k}_layer{layer_frac}.pt"
            )
            save_vector(vector, vector_path)
            vector_paths[(top_k, layer_frac)] = vector_path
            layer_vectors[layer_frac][top_k] = vector

    heatmap_paths: dict[float, Path] = {}
    statistics: dict[float, dict[str, float]] = {}

    labels = [str(k) for k in k_values]

    for layer_frac in layers:
        vectors = [layer_vectors[layer_frac][k] for k in k_values]

        first = vectors[0]
        all_identical = all(torch.equal(v, first) for v in vectors)
        if all_identical:
            logger.warning(
                "All vectors identical at layer %.2f for concept '%s'",
                layer_frac,
                concept,
            )

        similarity_matrix = compute_cosine_similarity_matrix(vectors)

        off_diagonal_mask = ~torch.eye(len(vectors), dtype=torch.bool).numpy()
        off_diagonal_values = similarity_matrix[off_diagonal_mask]

        # Handle edge case of single vector (no off-diagonal elements)
        if len(off_diagonal_values) > 0:
            statistics[layer_frac] = {
                "mean_similarity": float(off_diagonal_values.mean()),
                "min_similarity": float(off_diagonal_values.min()),
                "max_similarity": float(off_diagonal_values.max()),
            }
        else:
            statistics[layer_frac] = {
                "mean_similarity": 1.0,
                "min_similarity": 1.0,
                "max_similarity": 1.0,
            }

        heatmap_path = (
            output_dir / "heatmaps" / "discriminative" / f"{concept}_layer{layer_frac}.pdf"
        )
        title = f"Cosine Similarity: {concept} (layer {layer_frac})"
        plot_heatmap(similarity_matrix, labels, title, heatmap_path)
        heatmap_paths[layer_frac] = heatmap_path

    logger.info("Completed discriminative experiment for concept '%s'", concept)

    return {
        "vector_paths": {f"k{k[0]}_layer{k[1]}": str(v) for k, v in vector_paths.items()},
        "heatmap_paths": {f"layer{k}": str(v) for k, v in heatmap_paths.items()},
        "statistics": {f"layer{k}": v for k, v in statistics.items()},
    }


# =============================================================================
# Stability Comparison Functions
# =============================================================================


def select_token_subsets(
    pairs: list[ContrastPair],
    num_tokens: int,
    num_runs: int,
) -> list[list[ContrastPair]]:
    """Select different subsets of contrast pairs for each run.

    Args:
        pairs: All available contrast pairs.
        num_tokens: Number of pairs to select per run.
        num_runs: Number of different subsets to create.

    Returns:
        List of num_runs subsets, each with min(num_tokens, len(pairs)) pairs.
        Returns list of empty lists if pairs is empty.
    """
    if not pairs:
        return [[] for _ in range(num_runs)]

    capped = min(num_tokens, len(pairs))
    subsets: list[list[ContrastPair]] = []
    for i in range(num_runs):
        rng = random.Random(i)
        subset = rng.sample(pairs, k=capped)
        subsets.append(subset)
    return subsets


def run_single_extraction(
    model: HookedModel,
    pairs: list[ContrastPair],
    config: StabilityComparisonConfig,
    method: str,
    layer: float,
) -> Tensor:
    """Run a single extraction for one method and layer.

    Args:
        model: The hooked model.
        pairs: Contrast pairs to use.
        config: Stability comparison configuration.
        method: "mean" or "discriminative".
        layer: Layer fraction (0.0-1.0).

    Returns:
        Steering vector tensor for the specified layer.
    """
    extraction_config = ExtractionConfig(
        layers=[layer],
        method=method,
        top_k=config.top_k if method == "discriminative" else None,
    )
    steering_vector = extract_steering_vector(model, pairs, extraction_config)
    layer_idx = next(iter(steering_vector.layer_activations))
    return steering_vector.layer_activations[layer_idx]


def compute_stability_statistics(vectors: list[Tensor]) -> dict[str, float]:
    """Compute stability statistics from pairwise cosine similarities.

    Args:
        vectors: List of vectors from different runs.

    Returns:
        Dict with mean, min, max, std of pairwise cosine similarities.
    """
    if len(vectors) < 2:
        return {"mean": 1.0, "min": 1.0, "max": 1.0, "std": 0.0}

    sim_matrix = compute_cosine_similarity_matrix(vectors)

    n = len(vectors)
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_values = sim_matrix[off_diag_mask]

    return {
        "mean": float(off_diag_values.mean()),
        "min": float(off_diag_values.min()),
        "max": float(off_diag_values.max()),
        "std": float(off_diag_values.std()),
    }


def save_results_json(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Save experiment results to JSON file.

    Args:
        results: Results dict from run_stability_comparison_experiment.
        output_path: Path to save JSON file.
    """

    ensure_dir(output_path.parent)

    output_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        **results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)


def generate_stability_heatmap(
    similarity_matrix: ndarray | list[list[float]],
    layer: float,
    method: str,
    output_path: Path,
) -> None:
    """Generate heatmap visualization for stability results.

    Args:
        similarity_matrix: Pairwise cosine similarity matrix (ndarray or list from JSON).
        layer: Layer fraction for title.
        method: Method name for title.
        output_path: Path to save PDF.
    """
    ensure_dir(output_path.parent)

    # Convert to numpy array if coming from JSON (list of lists)
    sim_array = np.array(similarity_matrix)
    n_runs = sim_array.shape[0]
    labels = [f"Run {i + 1}" for i in range(n_runs)]
    title = f"Stability: {method} - Layer {layer:.2f}"

    plot_heatmap(sim_array, labels, title, output_path)


def run_stability_comparison_experiment(
    config: StabilityComparisonConfig,
) -> dict[str, Any]:
    """Run the full stability comparison experiment.

    Args:
        config: Configuration for the experiment.

    Returns:
        Results dict with:
        - config: The configuration used
        - diff_means: {similarity_matrices, statistics}
        - discriminative: {similarity_matrices, statistics}
    """
    logger.info(
        "Running stability comparison for concept='%s', num_runs=%d",
        config.concept,
        config.num_runs,
    )

    # Load all contrast pairs
    all_pairs = load_contrast_pairs(config.concept, config.num_tokens)
    logger.info("Loaded %d contrast pairs for concept '%s'", len(all_pairs), config.concept)

    # Select subsets for each run
    subsets = select_token_subsets(all_pairs, config.num_tokens, config.num_runs)
    logger.info("Selected %d subsets with %d pairs each", len(subsets), config.num_tokens)

    # Load model
    model = HookedModel(ModelConfig(model_name=config.model_name))

    results: dict[str, dict[str, Any]] = {}

    for method in ["mean", "discriminative"]:
        method_name = "diff_means" if method == "mean" else "discriminative"
        logger.info("Running method: %s", method_name)

        layer_results: dict[float, list[Tensor]] = {layer: [] for layer in config.layers}

        for layer in config.layers:
            logger.info("Processing layer %.2f", layer)
            for run_idx, subset in enumerate(subsets):
                vector = run_single_extraction(model, subset, config, method, layer)
                layer_results[layer].append(vector)
                logger.debug("Run %d complete for layer %.2f", run_idx + 1, layer)

        # Compute similarity matrices and statistics per layer
        similarity_matrices: dict[float, list[list[float]]] = {}
        statistics: dict[float, dict[str, float]] = {}

        for layer, vectors in layer_results.items():
            sim_matrix = compute_cosine_similarity_matrix(vectors)
            similarity_matrices[layer] = sim_matrix.tolist()

            # Compute off-diagonal statistics
            n = len(vectors)
            if n > 1:
                off_diag_mask = ~np.eye(n, dtype=bool)
                off_diag_values = sim_matrix[off_diag_mask]
                statistics[layer] = {
                    "mean": float(off_diag_values.mean()),
                    "min": float(off_diag_values.min()),
                    "max": float(off_diag_values.max()),
                    "std": float(off_diag_values.std()),
                }
            else:
                # Single run - no comparison possible
                statistics[layer] = {
                    "mean": 1.0,
                    "min": 1.0,
                    "max": 1.0,
                    "std": 0.0,
                }

        results[method_name] = {
            "similarity_matrices": similarity_matrices,
            "statistics": statistics,
        }
        logger.info("Completed method: %s", method_name)

    return {
        "config": asdict(config),
        "diff_means": results["diff_means"],
        "discriminative": results["discriminative"],
    }


__all__ = [
    # Vector analysis (merged from vector_analysis.py)
    "compute_cosine_similarity_matrix",
    "plot_heatmap",
    "save_vector",
    "load_vector",
    "cap_examples",
    "run_diff_means_experiment",
    "run_discriminative_experiment",
    # Stability comparison
    "select_token_subsets",
    "run_single_extraction",
    "compute_stability_statistics",
    "save_results_json",
    "generate_stability_heatmap",
    "run_stability_comparison_experiment",
]
