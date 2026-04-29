"""Unembedding analysis utilities for steering vectors.

Provides functions for projecting steering vectors through the unembedding matrix
to find tokens most similar to the steering direction.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch import Tensor

from steering_geometry.config import SUPPORTED_CONCEPTS as VALID_CONCEPTS
from steering_geometry.config import ModelConfig
from steering_geometry.types import ConceptAnalysisResult, UnembedAnalysisResult
from steering_geometry.utils import configure_logging

if TYPE_CHECKING:
    from .models import HookedModel

logger = logging.getLogger(__name__)

__all__ = [
    "compute_topk_similar_tokens",
    "analyze_steering_vector",
    "save_analysis_results",
    "run_unembed_experiment",
    "plot_topk_heatmap",
    "plot_topk_bar_chart",
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

    # Ensure all tensors are on the same device
    unembed_matrix = unembed_matrix.to(vector.device)

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

    plot_topk_heatmap(result, output_dir)
    plot_topk_bar_chart(result, output_dir)

    logger.info("Completed unembed experiment for concept='%s', method='%s'", concept, method)

    return result


def plot_topk_heatmap(
    result: ConceptAnalysisResult,
    output_dir: Path | str = "outputs",
) -> Path:
    """Generate and save a heatmap visualization of top-K tokens across layers.

    X axis: Top-K positions (1, 2, 3, 4, 5)
    Y axis: Layer fractions (0.1, 0.2, ..., 1.0)
    Cell content: Token text (short, truncated if needed)
    Color: Cells colored by similarity value.

    Args:
        result: ConceptAnalysisResult containing analysis results for all layers.
        output_dir: Base output directory for plots. Defaults to "outputs".

    Returns:
        Path to the saved plot file (PDF format).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from steering_geometry.utils import ensure_dir

    output_dir = Path(output_dir)
    plot_dir = output_dir / "unembed_analysis" / "plots"
    ensure_dir(plot_dir)

    output_path = plot_dir / f"{result.concept}_{result.method}_heatmap.pdf"

    # Sort layers numerically
    sorted_layer_keys = sorted(result.results.keys(), key=lambda x: float(x.split("_")[1]))
    layers = [result.results[k].layer for k in sorted_layer_keys]

    # Prepare data for heatmap (Top 5)
    num_layers = len(layers)
    num_k = 5
    sim_matrix = np.zeros((num_layers, num_k))
    token_matrix = []

    for i, layer_key in enumerate(sorted_layer_keys):
        layer_res = result.results[layer_key]
        row_tokens = []
        for j in range(num_k):
            if j < len(layer_res.tokens):
                sim_matrix[i, j] = layer_res.similarities[j]
                token = layer_res.tokens[j].replace("\n", "\\n").replace("\t", "\\t")
                if len(token) > 12:
                    token = token[:9] + "..."
                row_tokens.append(token)
            else:
                sim_matrix[i, j] = 0
                row_tokens.append("")
        token_matrix.append(row_tokens)

    fig, ax = plt.subplots(figsize=(10, num_layers * 0.5 + 2))

    im = ax.imshow(
        sim_matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.5, sim_matrix.max())
    )

    ax.set_xticks(np.arange(num_k))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels([f"Top {k}" for k in range(1, num_k + 1)])
    ax.set_yticklabels([f"{layer_val:.1f}" for layer_val in layers])

    ax.set_xlabel("Top-K Position", fontsize=10, fontweight="bold")
    ax.set_ylabel("Layer Fraction", fontsize=10, fontweight="bold")

    for i in range(num_layers):
        for j in range(num_k):
            color = "white" if sim_matrix[i, j] > 0.6 * sim_matrix.max() else "black"
            ax.text(j, i, token_matrix[i][j], ha="center", va="center", color=color, fontsize=8)

    ax.set_title(
        f"Top Unembedding Tokens: {result.concept} ({result.method})\nModel: {result.model}",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", fontsize=10)

    fig.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()

    logger.info("Saved top-K heatmap to %s", output_path)
    return output_path


def plot_topk_bar_chart(
    result: ConceptAnalysisResult,
    output_dir: Path | str = "outputs",
    layers_to_plot: list[float] | None = None,
) -> list[Path]:
    """Generate horizontal bar charts for top-k tokens most similar to steering vectors.

    Args:
        result: ConceptAnalysisResult containing analysis results for multiple layers.
        output_dir: Base output directory for plots. Defaults to "outputs".
        layers_to_plot: List of layer fractions to plot. If None, plots all layers.
            Defaults to None.

    Returns:
        List of paths to the saved PDF plot files.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from steering_geometry.utils import ensure_dir

    output_dir = Path(output_dir)
    plot_dir = output_dir / "unembed_analysis" / "plots"
    ensure_dir(plot_dir)

    if layers_to_plot is None:
        sorted_layer_keys = sorted(result.results.keys(), key=lambda x: float(x.split("_")[1]))
        layers_to_plot = [result.results[k].layer for k in sorted_layer_keys]
    else:
        layers_to_plot = sorted(layers_to_plot)

    if not layers_to_plot:
        logger.warning("No layers to plot")
        return []

    output_path = plot_dir / f"{result.concept}_{result.method}_bars.pdf"

    with PdfPages(output_path) as pdf:
        for layer_frac in layers_to_plot:
            layer_key = f"layer_{layer_frac}"
            if layer_key not in result.results:
                logger.warning(f"Layer {layer_frac} not found in results, skipping")
                continue

            layer_result = result.results[layer_key]
            tokens = layer_result.tokens
            similarities = layer_result.similarities

            if not tokens:
                logger.warning(f"No tokens for layer {layer_frac}, skipping")
                continue

            tokens_rev = tokens[::-1]
            similarities_rev = similarities[::-1]

            fig, ax = plt.subplots(figsize=(10, 8))

            import matplotlib as mpl

            norm = mpl.colors.Normalize(min(similarities_rev), max(similarities_rev))
            colors = mpl.colormaps["RdYlBu_r"](norm(similarities_rev))

            bars = ax.barh(tokens_rev, similarities_rev, color=colors)

            ax.set_xlabel("Cosine Similarity", fontsize=10)
            ax.set_title(
                f"Top Tokens for {result.concept} ({result.method})\nLayer {layer_frac}",
                fontsize=12,
                fontweight="bold",
            )

            for bar, sim in zip(bars, similarities_rev, strict=True):
                ax.text(
                    bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{sim:.4f}",
                    va="center",
                    fontsize=8,
                )

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    logger.info("Saved bar charts to %s", output_path)
    return [output_path]


# =============================================================================
# CLI
# =============================================================================

VALID_METHODS = ("diff_means", "discriminative")
DEFAULT_LAYERS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class _Args(Protocol):
    """Protocol defining CLI arguments for unembed analysis."""

    concept: str
    method: str
    model: str
    layers: list[float]
    output: str
    log_level: str


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for unembed analysis CLI."""
    parser = argparse.ArgumentParser(
        prog="steering_geometry.unembed_analysis",
        description="Analyze steering vectors by projecting through the unembedding matrix",
    )
    parser.add_argument(
        "--concept",
        required=True,
        choices=VALID_CONCEPTS,
        help="Concept to analyze (polite, refusal, sentiment)",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=VALID_METHODS,
        help="Extraction method (diff_means, discriminative)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model name (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--layers",
        type=lambda s: [float(x) for x in s.split(",")],
        default=DEFAULT_LAYERS,
        help="Comma-separated layer fractions (default: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def main() -> None:
    """CLI entry point for unembedding analysis."""
    args = cast(_Args, cast(object, _build_parser().parse_args()))
    configure_logging(level=args.log_level)

    logger.info(
        "Starting unembed analysis: concept='%s', method='%s', model='%s', layers=%s",
        args.concept,
        args.method,
        args.model,
        args.layers,
    )

    try:
        run_unembed_experiment(
            concept=args.concept,
            model_name=args.model,
            method=args.method,
            layers=args.layers,
            output_dir=args.output,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
