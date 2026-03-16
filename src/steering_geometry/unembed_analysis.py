"""Unembedding analysis utilities for steering vectors.

Provides functions for projecting steering vectors through the unembedding matrix
to find tokens most similar to the steering direction.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from steering_geometry.config import ModelConfig
from steering_geometry.types import ConceptAnalysisResult, UnembedAnalysisResult

if TYPE_CHECKING:
    from .models import HookedModel

logger = logging.getLogger(__name__)

__all__ = [
    "compute_topk_similar_tokens",
    "analyze_steering_vector",
    "save_analysis_results",
    "run_unembed_experiment",
]


def compute_topk_similar_tokens(
    vector: Tensor,
    unembed_matrix: Tensor,
    tokenizer: Any,
    k: int = 5,
    exclude_tokens: set[int] | None = None,
) -> list[tuple[str, float]]:
    """Compute top-k tokens most similar to a steering vector.

    Projects the steering vector through the unembedding matrix to find
    tokens whose embeddings have highest cosine similarity with the vector.

    Args:
        vector: 1D steering vector tensor of shape (hidden_dim,).
        unembed_matrix: 2D unembedding weight matrix of shape (vocab_size, hidden_dim).
        tokenizer: Tokenizer with decode() method for converting token IDs to text.
        k: Number of top tokens to return. Defaults to 5.
        exclude_tokens: Set of token IDs to exclude from results (e.g., special tokens).
            Defaults to None.

    Returns:
        List of (token_text, similarity) tuples sorted by descending similarity.
        Length is min(k, vocab_size - len(exclude_tokens)).

    Raises:
        ValueError: If vector is not 1D or has wrong hidden dimension.
    """
    if vector.dim() != 1:
        msg = f"Expected 1D vector, got {vector.dim()}D tensor"
        raise ValueError(msg)

    if vector.shape[0] != unembed_matrix.shape[1]:
        msg = (
            f"Vector hidden dim {vector.shape[0]} does not match "
            f"unembed matrix hidden dim {unembed_matrix.shape[1]}"
        )
        raise ValueError(msg)

    # Normalize to unit vectors for cosine similarity
    vector_norm = vector / torch.norm(vector)
    unembed_norms = torch.clamp(torch.norm(unembed_matrix, dim=1, keepdim=True), min=1e-8)
    unembed_normalized = unembed_matrix / unembed_norms

    # Cosine similarities via dot product; mask excluded tokens with -inf
    similarities = torch.mv(unembed_normalized, vector_norm)
    if exclude_tokens:
        for token_id in exclude_tokens:
            if 0 <= token_id < similarities.shape[0]:
                similarities[token_id] = float("-inf")

    actual_k = min(k, similarities.shape[0])
    topk_values, topk_indices = torch.topk(similarities, actual_k)
    results: list[tuple[str, float]] = []
    for idx, (sim_val, token_id) in enumerate(zip(topk_values, topk_indices, strict=True)):
        token_id_int = int(token_id.item())
        token_text = tokenizer.decode([token_id_int])
        similarity = float(sim_val.item())
        results.append((token_text, similarity))
        logger.debug(f"Rank {idx + 1}: token_id={token_id_int}, similarity={similarity:.4f}")

    logger.info(f"Found {len(results)} top tokens for steering vector analysis")
    return results


def analyze_steering_vector(
    vector: Tensor,
    model: "HookedModel",
    layer_frac: float,
    method: str,
    k: int = 5,
) -> UnembedAnalysisResult:
    """Analyze a steering vector by projecting through the unembedding matrix.

    Computes the top-k tokens most similar to the steering vector direction,
    providing insight into what semantic concepts the steering vector represents.

    Args:
        vector: 1D steering vector tensor of shape (hidden_dim,).
        model: HookedModel instance with tokenizer and unembedding matrix access.
        layer_frac: Layer fraction where the steering vector was extracted (0.1-1.0).
        method: Extraction method used (e.g., "diff_means", "discriminative").
        k: Number of top tokens to return. Defaults to 5.

    Returns:
        UnembedAnalysisResult containing layer fraction, method, top tokens,
        and their cosine similarity scores.
    """
    logger.info(f"Analyzing steering vector at layer_frac={layer_frac}, method={method}")

    unembed_matrix = model.get_unembedding_matrix()
    special_token_ids = model.get_special_token_ids()

    token_results = compute_topk_similar_tokens(
        vector=vector,
        unembed_matrix=unembed_matrix,
        tokenizer=model.tokenizer,
        k=k,
        exclude_tokens=special_token_ids,
    )

    tokens = [text for text, _ in token_results]
    similarities = [sim for _, sim in token_results]

    logger.info(f"Top tokens for layer_frac={layer_frac}: {tokens}")

    return UnembedAnalysisResult(
        layer=layer_frac,
        method=method,
        tokens=tokens,
        similarities=similarities,
    )


def save_analysis_results(result: ConceptAnalysisResult, output_path: Path | str) -> None:
    """Save concept analysis results to JSON file.

    Args:
        result: ConceptAnalysisResult to save.
        output_path: Path to output JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "concept": result.concept,
        "model": result.model,
        "method": result.method,
        "results": {
            layer_key: {
                "layer": layer_result.layer,
                "method": layer_result.method,
                "tokens": layer_result.tokens,
                "similarities": layer_result.similarities,
            }
            for layer_key, layer_result in result.results.items()
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved analysis results to %s", output_path)


def run_unembed_experiment(
    concept: str,
    model_name: str,
    method: str,
    layers: list[float],
    num_pairs: int = 1000,
    top_k: int = 30,
    output_dir: Path | str = "outputs",
) -> ConceptAnalysisResult:
    """Run unembedding analysis experiment for a concept across multiple layers.

    Extracts steering vectors for a concept at specified layers, then analyzes
    each vector by projecting through the unembedding matrix to find the most
    similar tokens.

    Args:
        concept: Concept to analyze (e.g., "honesty", "toxicity").
        model_name: HuggingFace model name.
        method: Extraction method ("diff_means" or "discriminative").
        layers: Relative layer positions (0.0-1.0) to analyze.
        num_pairs: Number of contrast pairs for extraction. Defaults to 1000.
        top_k: Number of top tokens to return per layer. Defaults to 30.
        output_dir: Base output directory for JSON results. Defaults to "outputs".

    Returns:
        ConceptAnalysisResult containing analysis results for all layers.

    Raises:
        ValueError: If method is not "diff_means" or "discriminative".
    """
    if method not in ("diff_means", "discriminative"):
        msg = f"method must be 'diff_means' or 'discriminative', got '{method}'"
        raise ValueError(msg)

    output_dir = Path(output_dir)
    logger.info(
        "Starting unembed experiment: concept='%s', model='%s', method='%s', layers=%s",
        concept,
        model_name,
        method,
        layers,
    )

    # Import here to avoid circular import at module load time
    from steering_geometry.extract import extract_vector

    model_config = ModelConfig(model_name=model_name)
    from steering_geometry.models import HookedModel

    model = HookedModel(model_config)

    extract_method = "mean" if method == "diff_means" else method
    steering_vector = extract_vector(
        concept=concept,
        model_name=model_name,
        num_pairs=num_pairs,
        method=extract_method,
        layers=layers,
    )

    results: dict[str, UnembedAnalysisResult] = {}
    for layer_frac, abs_idx in zip(layers, steering_vector.layer_activations.keys(), strict=True):
        vector = steering_vector.layer_activations[abs_idx]
        layer_key = f"layer_{layer_frac}"

        logger.info("Analyzing layer_frac=%.2f (abs_idx=%d)", layer_frac, abs_idx)
        analysis = analyze_steering_vector(
            vector=vector,
            model=model,
            layer_frac=layer_frac,
            method=method,
            k=top_k,
        )
        results[layer_key] = analysis

    result = ConceptAnalysisResult(
        concept=concept,
        model=model_name,
        method=method,
        results=results,
    )

    json_output_path = output_dir / "unembed_analysis" / "json" / f"{concept}_{method}.json"
    save_analysis_results(result, json_output_path)

    logger.info("Completed unembed experiment for concept='%s', method='%s'", concept, method)

    return result
