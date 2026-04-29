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

from steering_geometry.config import (
    ExtractionConfig,
    ModelConfig,
    StabilityComparisonConfig,
    StabilitySweepBatchConfig,
    StabilitySweepConfig,
)
from steering_geometry.extract import extract_steering_vector, extract_vector, load_contrast_pairs
from steering_geometry.models import HookedModel
from steering_geometry.types import ContrastPair, StabilitySweepResult
from steering_geometry.utils import ensure_dir, safe_model_name

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


def compute_reference_statistics(
    vectors: list[Tensor],
    reference: Tensor,
) -> dict[str, float]:
    """Compute cosine similarity statistics against a reference vector.

    Args:
        vectors: List of vectors from different runs.
        reference: Reference vector to compare against.

    Returns:
        Dict with mean, min, max, std of cosine similarities vs reference.
    """
    import torch.nn.functional as functional

    cos_sims = [
        float(functional.cosine_similarity(v.unsqueeze(0), reference.unsqueeze(0))) for v in vectors
    ]
    arr = np.array(cos_sims)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
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


def run_stability_sweep(config: StabilitySweepConfig) -> StabilitySweepResult:
    """Run stability sweep experiment across varying sample sizes.

    For each sample size N, constructs DiM directions from multiple independently
    sampled datasets, computes pairwise cosine similarity, and selects the best
    layer per (model, concept) pair.

    Args:
        config: Stability sweep configuration.

    Returns:
        StabilitySweepResult with selected layer and per-N stability metrics.

    Raises:
        ValueError: If no valid vectors can be extracted.
    """
    logger.info(
        "Running stability sweep for model='%s', concept='%s', n_values=%s, layers=%s, num_runs=%d",
        config.model_name,
        config.concept,
        config.n_values,
        config.layers,
        config.num_runs,
    )

    model_config = ModelConfig(
        model_name=config.model_name,
        device=config.device,
        dtype=config.dtype,
    )
    model = HookedModel(model_config)

    result = _run_sweep_with_model(model, config)

    logger.info(
        "Selected layer %.2f for model='%s', concept='%s' (avg cos_sim=%.4f)",
        result.selected_layer,
        config.model_name,
        config.concept,
        sum(result.per_n_data[n]["mean"] for n in config.n_values) / len(config.n_values),
    )

    return result


def run_stability_sweep_batch(
    config: StabilitySweepBatchConfig,
) -> list[StabilitySweepResult]:
    """Run stability sweeps for multiple concepts under a single model load.

    Loads the model once and iterates over all concepts, producing one
    StabilitySweepResult per concept.  Also emits structured progress lines
    that the calling shell script can intercept for a progress bar.

    Progress format (one line per extraction step)::

        PROGRESS <concept_idx>/<total_concepts> <step>/<total_steps> <concept> N=<n> run=<r>/<runs>

    Args:
        config: Batch configuration (model, concepts, sweep params).

    Returns:
        List of StabilitySweepResult, one per concept.
    """
    model_config = ModelConfig(
        model_name=config.model_name,
        device=config.device,
        dtype=config.dtype,
    )
    logger.info(
        "Loading model '%s' for batch sweep (%d concepts)", config.model_name, len(config.concepts)
    )
    model = HookedModel(model_config)

    total_concepts = len(config.concepts)
    results: list[StabilitySweepResult] = []

    for concept_idx, concept in enumerate(config.concepts):
        concept_config = StabilitySweepConfig(
            model_name=config.model_name,
            concept=concept,
            n_values=config.n_values,
            layers=config.layers,
            num_runs=config.num_runs,
            seed=config.seed,
            output_dir=config.output_dir,
            device=config.device,
            dtype=config.dtype,
            reference_n=config.reference_n,
        )

        logger.info(
            "=== [%d/%d] Concept: %s ===",
            concept_idx + 1,
            total_concepts,
            concept,
        )

        result = _run_sweep_with_model(model, concept_config, concept_idx, total_concepts)
        save_sweep_results(result, output_dir=config.output_dir)
        results.append(result)

        logger.info(
            "  Selected layer: %.2f | cos_sim values: %s",
            result.selected_layer,
            ", ".join(
                f"N={n}={result.per_n_data[n]['mean']:.4f}±{result.per_n_data[n]['std']:.4f}"
                for n in sorted(result.per_n_data)
            ),
        )

    return results


def _run_sweep_with_model(
    model: HookedModel,
    config: StabilitySweepConfig,
    concept_idx: int = 0,
    total_concepts: int = 1,
) -> StabilitySweepResult:
    """Core sweep logic accepting a pre-loaded model.

    Identical to :func:`run_stability_sweep` but skips model loading so the
    caller can reuse a single ``HookedModel`` across multiple concepts.

    Args:
        model: Pre-loaded HookedModel instance.
        config: Per-concept sweep configuration.
        concept_idx: 1-based index of current concept (for progress lines).
        total_concepts: Total number of concepts in the batch.

    Returns:
        StabilitySweepResult for the requested concept.
    """
    all_pairs = load_contrast_pairs(config.concept, num_pairs=10000)
    max_available = len(all_pairs)
    logger.info("Loaded %d contrast pairs for concept '%s'", max_available, config.concept)

    output_dir = Path(config.output_dir)
    concept_dir = output_dir / "vectors" / safe_model_name(config.model_name) / config.concept

    all_vectors: dict[float, dict[int, dict[int, Tensor]]] = {
        layer: {n: {} for n in config.n_values} for layer in config.layers
    }

    total_steps = len(config.n_values) * config.num_runs
    current_step = 0

    for n in config.n_values:
        capped = cap_examples(n, max_available, config.concept)

        for run_idx in range(config.num_runs):
            current_step += 1
            rng = random.Random(config.seed + run_idx)
            subset = rng.sample(all_pairs, k=capped)

            logger.info(
                "PROGRESS %d/%d %d/%d %s N=%d run=%d/%d",
                concept_idx + 1,
                total_concepts,
                current_step,
                total_steps,
                config.concept,
                n,
                run_idx + 1,
                config.num_runs,
            )

            extraction_config = ExtractionConfig(
                layers=config.layers,
                method="mean",
            )
            steering_vector = extract_steering_vector(model, subset, extraction_config)

            for layer_frac, abs_idx in zip(
                config.layers, steering_vector.layer_activations.keys(), strict=True
            ):
                vector = steering_vector.layer_activations[abs_idx]

                if torch.isnan(vector).any():
                    msg = (
                        f"Vector for concept='{config.concept}', n={n}, "
                        f"run={run_idx}, layer={layer_frac} contains NaN"
                    )
                    raise ValueError(msg)

                vector_path = concept_dir / f"n{n}_run{run_idx}_layer{layer_frac}.pt"
                save_vector(vector, vector_path)
                all_vectors[layer_frac][n][run_idx] = vector

    # Load or auto-extract reference vectors when using reference-based comparison
    reference_vectors: dict[float, Tensor] | None = None
    if config.reference_n is not None:
        reference_vectors = {}
        missing_layers: list[float] = []
        for layer_frac in config.layers:
            ref_path = concept_dir / f"n{config.reference_n}_run0_layer{layer_frac}.pt"
            if ref_path.exists():
                reference_vectors[layer_frac] = load_vector(ref_path)
            else:
                missing_layers.append(layer_frac)

        if missing_layers:
            logger.info(
                "Extracting reference vectors for N=%d (%d layers missing)",
                config.reference_n,
                len(missing_layers),
            )
            ref_capped = cap_examples(config.reference_n, max_available, config.concept)
            rng_ref = random.Random(config.seed)
            ref_subset = rng_ref.sample(all_pairs, k=ref_capped)
            ref_extraction_config = ExtractionConfig(
                layers=config.layers,
                method="mean",
            )
            ref_sv = extract_steering_vector(model, ref_subset, ref_extraction_config)

            for layer_frac, abs_idx in zip(
                config.layers, ref_sv.layer_activations.keys(), strict=True
            ):
                ref_vector = ref_sv.layer_activations[abs_idx]
                if torch.isnan(ref_vector).any():
                    msg = (
                        f"Reference vector for concept='{config.concept}', "
                        f"N={config.reference_n}, layer={layer_frac} contains NaN"
                    )
                    raise ValueError(msg)
                ref_path = concept_dir / f"n{config.reference_n}_run0_layer{layer_frac}.pt"
                if layer_frac not in reference_vectors:
                    save_vector(ref_vector, ref_path)
                    reference_vectors[layer_frac] = ref_vector

    # Validate that reference vectors have compatible dimensions with extracted vectors
    if reference_vectors is not None:
        for layer_frac in config.layers:
            ref_dim = reference_vectors[layer_frac].shape[0]
            first_n = config.n_values[0]
            sweep_dim = all_vectors[layer_frac][first_n][0].shape[0]
            if ref_dim != sweep_dim:
                msg = (
                    f"Reference vector dimension ({ref_dim}) does not match "
                    f"extracted vector dimension ({sweep_dim}) at layer {layer_frac}. "
                    f"The reference vector was likely extracted with a different model. "
                    f"Re-extract the reference vector with the current model, "
                    f"or remove reference_n to use pairwise comparison."
                )
                raise ValueError(msg)

    all_layers_data: dict[float, dict[int, dict[str, float]]] = {}

    for layer_frac in config.layers:
        layer_data: dict[int, dict[str, float]] = {}

        for n in config.n_values:
            vectors = [all_vectors[layer_frac][n][run_idx] for run_idx in range(config.num_runs)]
            if reference_vectors is not None:
                stats = compute_reference_statistics(vectors, reference_vectors[layer_frac])
            else:
                stats = compute_stability_statistics(vectors)
            layer_data[n] = stats

        all_layers_data[layer_frac] = layer_data

    best_layer = config.layers[0]
    best_avg = -1.0

    for layer_frac in config.layers:
        avg_sim = sum(all_layers_data[layer_frac][n]["mean"] for n in config.n_values) / len(
            config.n_values
        )

        if avg_sim > best_avg:
            best_avg = avg_sim
            best_layer = layer_frac

    per_n_data = all_layers_data[best_layer]

    return StabilitySweepResult(
        model_name=config.model_name,
        concept=config.concept,
        display_concept=config.display_concept,
        selected_layer=best_layer,
        per_n_data=per_n_data,
        all_layers_data=all_layers_data,
    )


# =============================================================================
# Sweep Result Persistence & Plotting
# =============================================================================


def save_sweep_results(
    result: StabilitySweepResult,
    output_dir: Path | str = "outputs/stability_sweep",
) -> Path:
    """Save sweep results to JSON file.

    Saves a single (model, concept) result as JSON.

    Args:
        result: Sweep results to save.
        output_dir: Directory to save results.

    Returns:
        Path to saved JSON file.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    model_slug = safe_model_name(result.model_name)
    filename = f"results_{model_slug}_{result.concept}.json"
    output_path = output_dir / filename

    # Convert all_layers_data keys to strings for JSON
    all_layers_serializable: dict[str, dict[str, dict[str, float]]] = {}
    for layer_frac, n_data in result.all_layers_data.items():
        all_layers_serializable[f"{layer_frac}"] = {str(n): stats for n, stats in n_data.items()}

    per_n_serializable: dict[str, dict[str, float]] = {
        str(n): stats for n, stats in result.per_n_data.items()
    }

    output_data = {
        "model_name": result.model_name,
        "concept": result.concept,
        "display_concept": result.display_concept,
        "selected_layer": result.selected_layer,
        "per_n_data": per_n_serializable,
        "all_layers_data": all_layers_serializable,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Saved sweep results to %s", output_path)
    return output_path


def load_sweep_results(
    output_dir: Path | str = "outputs/stability_sweep",
) -> dict[tuple[str, str], StabilitySweepResult]:
    """Load all sweep result JSON files from output directory.

    Args:
        output_dir: Directory containing result JSON files.

    Returns:
        Dict keyed by (model_name, concept) containing StabilitySweepResult.
    """
    output_dir = Path(output_dir)
    results: dict[tuple[str, str], StabilitySweepResult] = {}

    if not output_dir.exists():
        logger.warning("Output directory %s does not exist", output_dir)
        return results

    for json_path in sorted(output_dir.glob("results_*.json")):
        with open(json_path) as f:
            data = json.load(f)

        # Convert string keys back to int/float
        per_n_data: dict[int, dict[str, float]] = {
            int(n): stats for n, stats in data["per_n_data"].items()
        }

        all_layers_data: dict[float, dict[int, dict[str, float]]] = {}
        for layer_str, n_data in data["all_layers_data"].items():
            all_layers_data[float(layer_str)] = {int(n): stats for n, stats in n_data.items()}

        result = StabilitySweepResult(
            model_name=data["model_name"],
            concept=data["concept"],
            display_concept=data["display_concept"],
            selected_layer=data["selected_layer"],
            per_n_data=per_n_data,
            all_layers_data=all_layers_data,
        )
        results[(result.model_name, result.concept)] = result

    logger.info("Loaded %d sweep results from %s", len(results), output_dir)
    return results


def load_sweep_results_for_plotting(
    output_dir: Path | str = "outputs/stability_sweep",
) -> dict[str, dict[str, dict[int, tuple[float, float]]]]:
    """Load sweep results structured for plotting.

    Args:
        output_dir: Directory containing result JSON files.

    Returns:
        Nested dict: {display_concept: {model_name: {N: (mean, std)}}}
    """
    all_results = load_sweep_results(output_dir)

    plot_data: dict[str, dict[str, dict[int, tuple[float, float]]]] = {}
    for (model_name, _concept), result in all_results.items():
        display_concept = result.display_concept
        if display_concept not in plot_data:
            plot_data[display_concept] = {}
        plot_data[display_concept][model_name] = {
            n: (stats["mean"], stats["std"]) for n, stats in result.per_n_data.items()
        }

    return plot_data


_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "allenai/Olmo-3-1025-7B": "OLMo3-7B",
    "allenai/Olmo-3-1125-32B": "OLMo3-32B",
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen/Qwen3-14B": "Qwen3-14B",
}


def _validate_plot_layer(
    results: dict[str, dict[str, StabilitySweepResult]],
    plot_layer: float,
) -> None:
    """Validate that plot_layer exists in all results' all_layers_data.

    Args:
        results: Nested dict {concept: {model_name: StabilitySweepResult}}.
        plot_layer: Layer fraction to validate.

    Raises:
        ValueError: If plot_layer is not available in any result.
    """
    for concept_name, model_results in results.items():
        for model_name, result in model_results.items():
            if plot_layer not in result.all_layers_data:
                available = sorted(result.all_layers_data.keys())
                msg = (
                    f"plot_layer={plot_layer} not found in results for "
                    f"model={model_name}, concept={concept_name}. "
                    f"Available layers: {available}"
                )
                raise ValueError(msg)


def plot_stability_sweep(
    results: dict[str, dict[str, StabilitySweepResult]],
    output_dir: Path | str = "outputs/stability_sweep",
    model_colors: dict[str, str] | None = None,
    model_labels: dict[str, str] | None = None,
    plot_layer: float | None = None,
) -> list[Path]:
    """Generate line plots showing cos_sim vs N for each concept.

    Creates one PDF figure per concept, each showing 4 model lines with
    error bands representing ±1 standard deviation.

    Args:
        results: Nested dict {concept: {model_name: StabilitySweepResult}}.
            Can be obtained from load_sweep_results() and restructured.
        output_dir: Directory to save PDF figures.
        model_colors: Optional override for model colors (model_name → hex color).
        model_labels: Optional override for model display names.
        plot_layer: Optional layer fraction to plot. When set, uses data from
            ``result.all_layers_data[plot_layer]`` instead of ``result.per_n_data``.
            The output filename includes the layer value.

    Returns:
        List of paths to saved PDF figures.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    if plot_layer is not None:
        _validate_plot_layer(results, plot_layer)

    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    labels = model_labels or _MODEL_DISPLAY_NAMES

    saved_paths: list[Path] = []

    for concept_name, model_results in results.items():
        fig, ax = plt.subplots(figsize=(6, 4))

        for idx, (model_name, result) in enumerate(sorted(model_results.items())):
            if plot_layer is not None:
                layer_data = result.all_layers_data[plot_layer]
            else:
                layer_data = result.per_n_data

            n_values = sorted(layer_data.keys())
            means = [layer_data[n]["mean"] for n in n_values]
            stds = [layer_data[n]["std"] for n in n_values]

            display_name = labels.get(model_name, model_name)
            color = (
                model_colors.get(model_name, default_colors[idx % len(default_colors)])
                if model_colors
                else default_colors[idx % len(default_colors)]
            )

            ax.plot(n_values, means, "-o", color=color, label=display_name, markersize=4)
            ax.fill_between(
                n_values,
                [m - s for m, s in zip(means, stds, strict=True)],
                [m + s for m, s in zip(means, stds, strict=True)],
                color=color,
                alpha=0.2,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Number of Examples (N)", fontsize=12)
        ax.set_ylabel("Mean Pairwise Cosine Similarity", fontsize=12)
        ax.set_title(concept_name, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        if not model_results:
            plt.close()
            continue

        from matplotlib.ticker import NullFormatter, ScalarFormatter

        first_result = next(iter(model_results.values()))
        if plot_layer is not None:
            ref_data = first_result.all_layers_data[plot_layer]
        else:
            ref_data = first_result.per_n_data
        n_values = sorted(ref_data.keys())
        ax.set_xticks(n_values)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(NullFormatter())

        fig.tight_layout()

        safe_name = concept_name.lower().replace(" ", "_")
        if plot_layer is not None:
            output_path = output_dir / f"{safe_name}_stability_sweep_layer{plot_layer}.pdf"
        else:
            output_path = output_dir / f"{safe_name}_stability_sweep.pdf"
        plt.savefig(output_path, bbox_inches="tight", format="pdf")
        plt.close()

        logger.info("Saved stability sweep plot to %s", output_path)
        saved_paths.append(output_path)

    return saved_paths


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
    "compute_reference_statistics",
    "save_results_json",
    "generate_stability_heatmap",
    "run_stability_comparison_experiment",
    "run_stability_sweep",
    "run_stability_sweep_batch",
    # Sweep result persistence & plotting
    "save_sweep_results",
    "load_sweep_results",
    "load_sweep_results_for_plotting",
    "plot_stability_sweep",
]
