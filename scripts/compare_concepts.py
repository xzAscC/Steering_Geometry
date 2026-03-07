"""Cross-concept comparison analysis for steering vectors.

Computes pairwise cosine similarities and L2 norms for all steering vectors
in a given directory, outputting a structured JSON report.
"""

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor

from steering_geometry.evaluation import compute_cosine_similarity
from steering_geometry.types import SteeringVector


class _Args(Protocol):
    vectors_dir: str
    model: str | None
    output: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare steering vectors across concepts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all vectors in default directory
  uv run python scripts/compare_concepts.py

  # Filter by model
  uv run python scripts/compare_concepts.py --model sshleifer/tiny-gpt2

  # Custom input/output paths
  uv run python scripts/compare_concepts.py --vectors-dir data/vectors/ --output assets/comparison_report.json
        """,
    )
    _ = parser.add_argument(
        "--vectors-dir",
        default="data/vectors/",
        help="Directory containing .pt steering vector files (default: data/vectors/)",
    )
    _ = parser.add_argument(
        "--model",
        default=None,
        help="Filter vectors by model name (e.g., sshleifer/tiny-gpt2). If not specified, uses all models.",
    )
    _ = parser.add_argument(
        "--output",
        default="assets/comparison_report.json",
        help="Output JSON file path (default: assets/comparison_report.json)",
    )
    return parser


def _load_vectors_from_directory(
    vectors_dir: Path,
    model_filter: str | None,
) -> dict[str, SteeringVector]:
    """Load all steering vectors from .pt files in a directory.

    Args:
        vectors_dir: Path to directory containing .pt files.
        model_filter: Optional model name to filter vectors.

    Returns:
        Dictionary mapping concept names to SteeringVector objects.

    Raises:
        FileNotFoundError: If the vectors directory doesn't exist.
        ValueError: If no .pt files are found.
    """
    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")

    pt_files = list(vectors_dir.glob("*.pt"))
    if not pt_files:
        raise ValueError(f"No .pt files found in {vectors_dir}")

    vectors: dict[str, SteeringVector] = {}

    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        vector = data["vector"]

        if not isinstance(vector, SteeringVector):
            raise TypeError(f"Expected SteeringVector in {pt_file}, got {type(vector)}")

        if model_filter is not None and vector.model_name != model_filter:
            continue

        vectors[vector.concept] = vector

    if not vectors:
        if model_filter:
            raise ValueError(f"No vectors found for model '{model_filter}' in {vectors_dir}")
        raise ValueError(f"No valid steering vectors found in {vectors_dir}")

    return vectors


def _compute_l2_norm(vector: SteeringVector) -> float:
    """Compute the L2 norm of a steering vector across all layers.

    Args:
        vector: SteeringVector to compute norm for.

    Returns:
        L2 norm value as float.
    """
    all_activations: list[Tensor] = list(vector.layer_activations.values())
    if not all_activations:
        return 0.0

    concatenated = torch.cat([act.flatten() for act in all_activations])
    return concatenated.norm(p=2).item()


def _compute_average_cosine_similarity(
    v1: SteeringVector,
    v2: SteeringVector,
) -> float:
    """Compute average cosine similarity across all common layers.

    Args:
        v1: First steering vector.
        v2: Second steering vector.

    Returns:
        Average cosine similarity as float.
    """
    try:
        layer_similarities = compute_cosine_similarity(v1, v2)
        if not layer_similarities:
            return 0.0
        return sum(layer_similarities.values()) / len(layer_similarities)
    except ValueError:
        return 0.0


def _generate_comparison_report(
    vectors: dict[str, SteeringVector],
    model_name: str | None,
) -> dict:
    """Generate a comparison report with cosine similarities and L2 norms.

    Args:
        vectors: Dictionary mapping concept names to SteeringVector objects.
        model_name: Model name for metadata (or None if multiple models).

    Returns:
        Dictionary with comparison results.
    """
    concepts = sorted(vectors.keys())

    cosine_similarities: dict[str, float] = {}
    for concept1, concept2 in combinations(concepts, 2):
        avg_similarity = _compute_average_cosine_similarity(vectors[concept1], vectors[concept2])
        key = f"{concept1}-{concept2}"
        cosine_similarities[key] = round(avg_similarity, 6)

    l2_norms: dict[str, float] = {}
    for concept in concepts:
        norm = _compute_l2_norm(vectors[concept])
        l2_norms[concept] = round(norm, 6)

    metadata_model = (
        model_name if model_name is not None else next(iter(vectors.values())).model_name
    )

    report = {
        "cosine_similarities": cosine_similarities,
        "l2_norms": l2_norms,
        "metadata": {
            "model": metadata_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_concepts": len(concepts),
            "concepts": concepts,
        },
    }

    return report


def main() -> None:
    args = cast(_Args, cast(object, _build_parser().parse_args()))

    vectors_dir = Path(args.vectors_dir)
    output_path = Path(args.output)

    print(f"Loading vectors from: {vectors_dir}")
    if args.model:
        print(f"Filtering by model: {args.model}")

    vectors = _load_vectors_from_directory(vectors_dir, args.model)
    print(f"Loaded {len(vectors)} steering vectors: {sorted(vectors.keys())}")

    report = _generate_comparison_report(vectors, args.model)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Comparison report saved to: {output_path}")
    print(f"Computed {len(report['cosine_similarities'])} pairwise similarities")
    print(f"Computed L2 norms for {len(report['l2_norms'])} concepts")


if __name__ == "__main__":
    main()
