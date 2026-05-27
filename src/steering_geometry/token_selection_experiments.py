"""Token selection experiments for steering vector extraction.

This module provides experiment runners that investigate how different token
selection strategies affect steering vector quality and stability:

1. Token count experiment — varies the number of examples used for extraction
2. Token position experiment — compares token selection modes (all, last_n)
3. Prompt/response experiment — compares prompt_only vs prompt_response data modes
"""

import json
import logging
from pathlib import Path

import torch
from torch import Tensor

from steering_geometry.config import ExtractionConfig, ModelConfig
from steering_geometry.extract import extract_steering_vector, load_contrast_pairs
from steering_geometry.models import HookedModel
from steering_geometry.stability_comparison import (
    cap_examples,
    compute_cosine_similarity_matrix,
    plot_heatmap,
    save_vector,
)
from steering_geometry.utils import sample_with_seed

logger = logging.getLogger(__name__)


def run_token_count_experiment(
    concept: str,
    n_examples_list: list[int],
    layers: list[float],
    model_name: str,
    output_dir: Path | str = "outputs",
    method: str = "mean",
    token_select: str = "all",
) -> dict[str, dict[str, str] | dict[str, dict[str, float]]]:
    """Run experiment varying the number of examples used for extraction.

    For each example count in ``n_examples_list``, extracts steering vectors
    and computes pairwise cosine similarities across all counts at each layer.

    Args:
        concept: Concept to extract (e.g., "sentiment", "refusal").
        n_examples_list: List of example counts to test (e.g., [10, 30, 100]).
        layers: Relative layer positions (0.0-1.0) to analyze.
        model_name: HuggingFace model name.
        output_dir: Base output directory for vectors and heatmaps.
        method: Extraction method ("mean", "pca", etc.).
        token_select: Token selection strategy ("all" or "last_n").

    Returns:
        Dict with:
            - "vector_paths": Dict mapping parameter combos to vector file paths
            - "heatmap_paths": Dict mapping layer to heatmap file paths
            - "statistics": Dict with mean/min/max similarities per layer

    Raises:
        ValueError: If n_examples_list is empty.
    """
    if not n_examples_list:
        msg = "n_examples_list cannot be empty"
        raise ValueError(msg)

    output_dir = Path(output_dir)

    logger.info("Loading contrast pairs for concept '%s'", concept)
    all_pairs = load_contrast_pairs(concept, num_pairs=max(n_examples_list))
    max_available = len(all_pairs)
    logger.info("Dataset has %d examples available for concept '%s'", max_available, concept)

    model = HookedModel(ModelConfig(model_name=model_name))
    logger.info("Loaded model '%s'", model_name)

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

        pairs_subset = sample_with_seed(all_pairs, capped)

        config = ExtractionConfig(
            layers=layers,
            method=method,
            token_select=token_select,
        )
        steering_vector = extract_steering_vector(model, pairs_subset, config)

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
                / "token_count"
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

        heatmap_path = output_dir / "heatmaps" / "token_count" / f"{concept}_layer{layer_frac}.pdf"
        title = f"Token Count Similarity: {concept} (layer {layer_frac})"
        plot_heatmap(similarity_matrix, labels, title, heatmap_path)
        heatmap_paths[layer_frac] = heatmap_path

    logger.info("Completed token_count experiment for concept '%s'", concept)

    return {
        "vector_paths": {f"n{k[0]}_layer{k[1]}": str(v) for k, v in vector_paths.items()},
        "heatmap_paths": {f"layer{k}": str(v) for k, v in heatmap_paths.items()},
        "statistics": {f"layer{k}": v for k, v in statistics.items()},
    }


def run_token_position_experiment(
    concept: str,
    n_examples: int,
    position_configs: list[dict[str, int | str]],
    layers: list[float],
    model_name: str,
    output_dir: Path | str = "outputs",
    method: str = "mean",
) -> dict[str, dict[str, str] | dict[str, dict[str, float]]]:
    """Run experiment comparing token selection position strategies.

    For each position configuration, extracts steering vectors and computes
    pairwise cosine similarities across all configs at each layer.

    Args:
        concept: Concept to extract (e.g., "sentiment", "refusal").
        n_examples: Number of contrast pairs to use.
        position_configs: List of position configs, each a dict with:
            - "mode": "all" or "last_n"
            - "n": (required when mode="last_n") number of trailing tokens
        layers: Relative layer positions (0.0-1.0) to analyze.
        model_name: HuggingFace model name.
        output_dir: Base output directory for vectors and heatmaps.
        method: Extraction method ("mean", "pca", etc.).

    Returns:
        Dict with:
            - "vector_paths": Dict mapping parameter combos to vector file paths
            - "heatmap_paths": Dict mapping layer to heatmap file paths
            - "statistics": Dict with mean/min/max similarities per layer

    Raises:
        ValueError: If position_configs is empty or contains invalid mode.
    """
    if not position_configs:
        msg = "position_configs cannot be empty"
        raise ValueError(msg)

    output_dir = Path(output_dir)

    logger.info("Loading contrast pairs for concept '%s'", concept)
    all_pairs = load_contrast_pairs(concept, num_pairs=n_examples)
    max_available = len(all_pairs)
    capped = cap_examples(n_examples, max_available, concept)

    model = HookedModel(ModelConfig(model_name=model_name))
    logger.info("Loaded model '%s'", model_name)

    vector_paths: dict[tuple[str, int | None, float], Path] = {}
    layer_vectors: dict[float, dict[str, Tensor]] = {layer_frac: {} for layer_frac in layers}

    for pos_config in position_configs:
        mode = str(pos_config["mode"])
        n_value: int | None = None

        if mode == "all":
            token_select_value = "all"
            config_label = "all"
        elif mode == "last_n":
            n_value = int(pos_config["n"])
            token_select_value = "last_n"
            config_label = f"last_n_n{n_value}"
        else:
            msg = f"Invalid position mode: {mode}. Must be 'all' or 'last_n'."
            raise ValueError(msg)

        logger.info("Extracting vector for concept='%s', position='%s'", concept, config_label)

        pairs_subset = sample_with_seed(all_pairs, capped)

        if mode == "last_n" and n_value is not None:
            config = ExtractionConfig(
                layers=layers,
                method=method,
                token_select=token_select_value,
                last_n=n_value,
            )
        else:
            config = ExtractionConfig(
                layers=layers,
                method=method,
                token_select=token_select_value,
            )
        steering_vector = extract_steering_vector(model, pairs_subset, config)

        for layer_frac, abs_idx in zip(
            layers, steering_vector.layer_activations.keys(), strict=True
        ):
            vector = steering_vector.layer_activations[abs_idx]

            if torch.isnan(vector).any():
                msg = (
                    f"Vector for concept='{concept}', position='{config_label}', "
                    f"layer={layer_frac} contains NaN"
                )
                raise ValueError(msg)

            if mode == "all":
                vector_path = (
                    output_dir
                    / "vectors"
                    / concept
                    / "token_position"
                    / f"{mode}_layer{layer_frac}.pt"
                )
            else:
                vector_path = (
                    output_dir
                    / "vectors"
                    / concept
                    / "token_position"
                    / f"{mode}_n{n_value}_layer{layer_frac}.pt"
                )

            save_vector(vector, vector_path)
            vector_paths[(mode, n_value, layer_frac)] = vector_path
            layer_vectors[layer_frac][config_label] = vector

    heatmap_paths: dict[float, Path] = {}
    statistics: dict[float, dict[str, float]] = {}

    config_labels = []
    for pos_config in position_configs:
        mode = str(pos_config["mode"])
        if mode == "all":
            config_labels.append("all")
        elif mode == "last_n":
            config_labels.append(f"last_n_n{pos_config['n']}")
        else:
            msg = f"Invalid position mode: {mode}"
            raise ValueError(msg)

    for layer_frac in layers:
        vectors = [layer_vectors[layer_frac][label] for label in config_labels]

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
            output_dir / "heatmaps" / "token_position" / f"{concept}_layer{layer_frac}.pdf"
        )
        title = f"Token Position Similarity: {concept} (layer {layer_frac})"
        plot_heatmap(similarity_matrix, config_labels, title, heatmap_path)
        heatmap_paths[layer_frac] = heatmap_path

    logger.info("Completed token_position experiment for concept '%s'", concept)

    path_keys: dict[tuple[str, int | None, float], str] = {}
    for key in vector_paths:
        mode, n_val, layer = key
        if mode == "all":
            path_keys[key] = f"{mode}_layer{layer}"
        else:
            path_keys[key] = f"{mode}_n{n_val}_layer{layer}"

    return {
        "vector_paths": {path_keys[k]: str(v) for k, v in vector_paths.items()},
        "heatmap_paths": {f"layer{k}": str(v) for k, v in heatmap_paths.items()},
        "statistics": {f"layer{k}": v for k, v in statistics.items()},
    }


def run_prompt_response_experiment(
    concept: str,
    n_examples: int,
    data_modes: list[str],
    layers: list[float],
    model_name: str,
    output_dir: Path | str = "outputs",
    method: str = "mean",
    token_select: str = "all",
) -> dict[str, dict[str, str] | dict[str, dict[str, float]]]:
    """Run experiment comparing prompt_only vs prompt_response data modes.

    For each data mode, extracts steering vectors and computes pairwise
    cosine similarities across all modes at each layer.

    Args:
        concept: Concept to extract (e.g., "refusal").
        n_examples: Number of contrast pairs to use.
        data_modes: List of data modes ("prompt_only" and/or "prompt_response").
        layers: Relative layer positions (0.0-1.0) to analyze.
        model_name: HuggingFace model name.
        output_dir: Base output directory for vectors and heatmaps.
        method: Extraction method ("mean", "pca", etc.).
        token_select: Token selection strategy ("all" or "last_n").

    Returns:
        Dict with:
            - "vector_paths": Dict mapping parameter combos to vector file paths
            - "heatmap_paths": Dict mapping layer to heatmap file paths
            - "statistics": Dict with mean/min/max similarities per layer

    Raises:
        ValueError: If data_modes is empty or contains invalid mode.
    """
    if not data_modes:
        msg = "data_modes cannot be empty"
        raise ValueError(msg)

    for mode in data_modes:
        if mode not in ("prompt_only", "prompt_response"):
            msg = f"Invalid data_mode: {mode}. Must be 'prompt_only' or 'prompt_response'."
            raise ValueError(msg)

    output_dir = Path(output_dir)

    model = HookedModel(ModelConfig(model_name=model_name))
    logger.info("Loaded model '%s'", model_name)

    vector_paths: dict[tuple[str, float], Path] = {}
    layer_vectors: dict[float, dict[str, Tensor]] = {layer_frac: {} for layer_frac in layers}

    for data_mode in data_modes:
        logger.info(
            "Loading contrast pairs for concept='%s', data_mode='%s', n=%d",
            concept,
            data_mode,
            n_examples,
        )
        pairs = load_contrast_pairs(concept, num_pairs=n_examples, data_mode=data_mode)
        logger.info("Loaded %d pairs with data_mode='%s'", len(pairs), data_mode)

        config = ExtractionConfig(
            layers=layers,
            method=method,
            token_select=token_select,
            data_mode=data_mode,
        )
        steering_vector = extract_steering_vector(model, pairs, config)

        for layer_frac, abs_idx in zip(
            layers, steering_vector.layer_activations.keys(), strict=True
        ):
            vector = steering_vector.layer_activations[abs_idx]

            if torch.isnan(vector).any():
                msg = (
                    f"Vector for concept='{concept}', data_mode='{data_mode}', "
                    f"layer={layer_frac} contains NaN"
                )
                raise ValueError(msg)

            vector_path = (
                output_dir
                / "vectors"
                / concept
                / "prompt_response"
                / f"{data_mode}_n{n_examples}_layer{layer_frac}.pt"
            )
            save_vector(vector, vector_path)
            vector_paths[(data_mode, layer_frac)] = vector_path
            layer_vectors[layer_frac][data_mode] = vector

    heatmap_paths: dict[float, Path] = {}
    statistics: dict[float, dict[str, float]] = {}

    for layer_frac in layers:
        vectors = [layer_vectors[layer_frac][mode] for mode in data_modes]

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
            output_dir / "heatmaps" / "prompt_response" / f"{concept}_layer{layer_frac}.pdf"
        )
        title = f"Prompt/Response Similarity: {concept} (layer {layer_frac})"
        plot_heatmap(similarity_matrix, data_modes, title, heatmap_path)
        heatmap_paths[layer_frac] = heatmap_path

    logger.info("Completed prompt_response experiment for concept '%s'", concept)

    return {
        "vector_paths": {f"{k[0]}_layer{k[1]}": str(v) for k, v in vector_paths.items()},
        "heatmap_paths": {f"layer{k}": str(v) for k, v in heatmap_paths.items()},
        "statistics": {f"layer{k}": v for k, v in statistics.items()},
    }


def run_steering_scope_experiment(
    concept: str,
    steer_tokens_values: list[int | None],
    layers: list[float],
    multipliers: list[float],
    model_name: str,
    output_dir: Path | str = "outputs",
    num_samples: int = 10,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    n_examples: int = 100,
    method: str = "mean",
    token_select: str = "all",
) -> dict[str, list[str] | dict[str, int]]:
    """Run experiment varying the number of tokens steering is applied.

    For each combination of ``steer_tokens_values``, layer, and multiplier,
    applies steering and saves generated text to JSONL files.

    Args:
        concept: Concept to extract (e.g., "refusal").
        steer_tokens_values: List of steer_tokens values (None = all tokens).
        layers: Relative layer positions (0.0-1.0) to steer at.
        multipliers: Scaling multipliers for the steering vector.
        model_name: HuggingFace model name.
        output_dir: Base output directory for generated JSONL files.
        num_samples: Number of prompts to generate per parameter combo.
        max_new_tokens: Maximum tokens to generate per sample.
        temperature: Sampling temperature (0.0 for greedy).
        n_examples: Number of contrast pairs for extraction.
        method: Extraction method ("mean", "pca", etc.).
        token_select: Token selection strategy ("all" or "last_n").

    Returns:
        Dict with:
            - "output_files": List of output JSONL file paths
            - "statistics": Dict with total_samples count

    Raises:
        ValueError: If steer_tokens_values or multipliers or layers is empty.
    """
    if not steer_tokens_values:
        msg = "steer_tokens_values cannot be empty"
        raise ValueError(msg)
    if not multipliers:
        msg = "multipliers cannot be empty"
        raise ValueError(msg)
    if not layers:
        msg = "layers cannot be empty"
        raise ValueError(msg)

    output_dir = Path(output_dir)

    # Load model ONCE
    model = HookedModel(ModelConfig(model_name=model_name))
    logger.info("Loaded model '%s'", model_name)

    # Load contrast pairs
    logger.info("Loading contrast pairs for concept '%s'", concept)
    if concept == "refusal":
        all_pairs = load_contrast_pairs(concept, n_examples, data_mode="prompt_only")
    else:
        all_pairs = load_contrast_pairs(concept, n_examples)
    logger.info("Loaded %d contrast pairs for concept '%s'", len(all_pairs), concept)

    # Extract baseline steering vector
    config = ExtractionConfig(
        layers=layers,
        method=method,
        token_select=token_select,
    )
    capped = cap_examples(n_examples, len(all_pairs), concept)
    pairs_subset = sample_with_seed(all_pairs, capped)
    steering_vector = extract_steering_vector(model, pairs_subset, config)

    # Resolve all absolute layer indices upfront
    abs_layer_indices = model.resolve_layers(layers)
    layer_map = dict(zip(layers, abs_layer_indices, strict=True))

    # Select negative samples as prompts
    neg_samples = [pair.negative for pair in all_pairs[:num_samples]]
    actual_samples = min(num_samples, len(neg_samples))
    neg_samples = neg_samples[:actual_samples]

    output_files: list[str] = []
    total_samples = 0

    for steer_tokens_value in steer_tokens_values:
        steer_label = "all" if steer_tokens_value is None else str(steer_tokens_value)

        for layer_frac in layers:
            abs_idx = layer_map[layer_frac]
            vector = steering_vector.layer_activations[abs_idx]
            vector_norm = float(vector.norm())

            if vector_norm == 0.0:
                logger.warning(
                    "Zero-norm vector at layer %.2f for concept '%s', skipping",
                    layer_frac,
                    concept,
                )
                continue

            for mult in multipliers:
                scale = mult * vector_norm

                jsonl_path = (
                    output_dir
                    / "steered"
                    / concept
                    / "steer_scope"
                    / f"steer_{steer_label}_layer{layer_frac}_mult{mult}.jsonl"
                )
                jsonl_path.parent.mkdir(parents=True, exist_ok=True)

                results: list[dict[str, int | float | str | None]] = []
                for sample_idx, prompt in enumerate(neg_samples):
                    generated = model.generate_with_steering(
                        prompt=prompt,
                        layer_idx=abs_idx,
                        steering_vector=vector,
                        scale=scale,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        steer_tokens=steer_tokens_value,
                    )
                    results.append(
                        {
                            "steer_tokens": steer_tokens_value,
                            "layer": layer_frac,
                            "multiplier": mult,
                            "sample_idx": sample_idx,
                            "prompt": prompt,
                            "generated_text": generated,
                        }
                    )

                with jsonl_path.open("w") as f:
                    for record in results:
                        f.write(json.dumps(record) + "\n")

                output_files.append(str(jsonl_path))
                total_samples += len(results)
                logger.info(
                    "Saved %d samples to %s (steer_tokens=%s, layer=%.2f, mult=%.2f)",
                    len(results),
                    jsonl_path,
                    steer_label,
                    layer_frac,
                    mult,
                )

    logger.info(
        "Completed steering scope experiment for concept '%s': %d files, %d total samples",
        concept,
        len(output_files),
        total_samples,
    )

    return {"output_files": output_files, "statistics": {"total_samples": total_samples}}


__all__ = [
    "run_token_count_experiment",
    "run_token_position_experiment",
    "run_prompt_response_experiment",
    "run_steering_scope_experiment",
]
