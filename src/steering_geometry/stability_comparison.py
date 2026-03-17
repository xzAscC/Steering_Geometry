"""Stability comparison experiment for steering vector extraction methods.

Compares the stability of diff_means vs discriminative token selection
by running multiple extractions with different token selections and
computing pairwise cosine similarity across layers.
"""

import datetime
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import ndarray
from torch import Tensor

from steering_geometry.config import ExtractionConfig, ModelConfig, StabilityComparisonConfig
from steering_geometry.extract import extract_steering_vector, load_contrast_pairs
from steering_geometry.models import HookedModel
from steering_geometry.types import ContrastPair
from steering_geometry.utils import ensure_dir
from steering_geometry.vector_analysis import compute_cosine_similarity_matrix, plot_heatmap

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "select_token_subsets",
    "run_single_extraction",
    "compute_stability_statistics",
    "save_results_json",
    "generate_stability_heatmap",
    "run_stability_comparison_experiment",
]


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
    similarity_matrix: ndarray,
    layer: float,
    method: str,
    output_path: Path,
) -> None:
    """Generate heatmap visualization for stability results.

    Args:
        similarity_matrix: Pairwise cosine similarity matrix.
        layer: Layer fraction for title.
        method: Method name for title.
        output_path: Path to save PDF.
    """
    ensure_dir(output_path.parent)

    n_runs = similarity_matrix.shape[0]
    labels = [f"Run {i + 1}" for i in range(n_runs)]
    title = f"Stability: {method} - Layer {layer:.2f}"

    plot_heatmap(similarity_matrix, labels, title, output_path)


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
