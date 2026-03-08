"""Apply steering vectors to model generation.

This module applies extracted steering vectors to model generation,
steering model behavior towards or away from specific concepts.

Usage:
    # CLI
    uv run python -m steering_geometry.apply_steering \
        --vector data/vectors/honesty_Qwen_Qwen3.5-2B_mean.pt \
        --model Qwen/Qwen3.5-2B \
        --output data/steered/

    # Programmatic
    from steering_geometry.apply_steering import apply_steering
    from steering_geometry.config import SteeringConfig
    apply_steering(vector_path, model_name, output_dir, config)
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor

from .config import ModelConfig, SteeringConfig
from .extract import load_contrast_pairs
from .models import HookedModel
from .types import SteeringVector


def _safe_model_name(model_name: str) -> str:
    """Convert model name to safe directory name."""
    return model_name.replace("/", "_")


def _normalize_vectors(vector: SteeringVector) -> dict[int, Tensor]:
    """Normalize steering vectors to unit norm."""
    normalized = {}
    for layer_idx, v in vector.layer_activations.items():
        norm = v.norm()
        if norm > 0:
            normalized[layer_idx] = v / norm
        else:
            normalized[layer_idx] = v
    return normalized


def _compute_avg_activation(
    model: HookedModel,
    texts: list[str],
    layers: list[int],
) -> dict[int, float]:
    """Compute average activation norm per layer.

    Args:
        model: HookedModel instance.
        texts: List of texts to compute activations for.
        layers: List of layer indices.

    Returns:
        Dictionary mapping layer index to average activation norm.
    """
    activations = model.get_activations(texts, layers)
    avg_per_layer = {}
    for layer_idx, act in activations.items():
        # act shape: (batch_size, seq_len, hidden_dim)
        # Compute mean norm across all tokens and samples
        avg_per_layer[layer_idx] = float(act.norm(dim=-1).mean().item())
    return avg_per_layer


def apply_steering(
    vector_path: Path,
    model_name: str,
    output_dir: Path,
    config: SteeringConfig,
) -> None:
    """Apply steering vector to model and save results as JSONL.

    This function:
    1. Loads the steering vector and extracts concept
    2. Loads contrast pairs and selects negative samples
    3. Normalizes steering vectors (norm=1)
    4. Computes average activation per layer
    5. For each layer, applies steering with each multiplier
    6. Saves results to JSONL files per layer

    Args:
        vector_path: Path to saved steering vector (.pt file).
        model_name: HuggingFace model name.
        output_dir: Directory for output JSONL files.
        config: Steering configuration.
    """
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not config.multipliers:
        raise ValueError("multipliers cannot be empty")

    # Load steering vector
    data = torch.load(vector_path, map_location="cpu", weights_only=False)
    vector: SteeringVector = data["vector"]
    concept = vector.concept

    print(f"Loaded steering vector for concept: {concept}")
    print(f"Vector has {len(vector.layer_activations)} layers")

    # Load model
    print(f"Loading model: {model_name}")
    model = HookedModel(ModelConfig(model_name=model_name))

    # Load contrast pairs and get negative samples
    print(f"Loading {config.num_samples} negative samples (seed={config.seed})...")
    pairs = load_contrast_pairs(concept, config.num_samples)
    random.seed(config.seed)
    selected_pairs = random.sample(pairs, min(config.num_samples, len(pairs)))
    neg_samples = [pair.negative for pair in selected_pairs]

    # Normalize vectors
    normalized_vectors = _normalize_vectors(vector)
    layers = sorted(normalized_vectors.keys())

    # Compute avg activation per layer
    print("Computing average activations...")
    avg_activations = _compute_avg_activation(model, neg_samples, layers)

    # Create output directory
    safe_model = _safe_model_name(model_name)
    concept_dir = output_dir / concept / safe_model
    concept_dir.mkdir(parents=True, exist_ok=True)

    # Process each layer
    for layer_idx in layers:
        print(f"\nProcessing layer {layer_idx}...")
        normalized_v = normalized_vectors[layer_idx]
        avg_act = avg_activations[layer_idx]

        results: list[dict[str, Any]] = []

        for sample_idx, prompt in enumerate(neg_samples):
            for multiplier in config.multipliers:
                scale = avg_act * multiplier
                generated = model.generate_with_steering(
                    prompt=prompt,
                    layer_idx=layer_idx,
                    steering_vector=normalized_v,
                    scale=scale,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                results.append(
                    {
                        "sample_idx": sample_idx,
                        "multiplier": multiplier,
                        "prompt": prompt,
                        "generated_text": generated,
                    }
                )
                preview = generated[:50] + "..." if len(generated) > 50 else generated
                print(f"  Sample {sample_idx}, mult {multiplier}: {preview}")

        # Write JSONL
        output_file = concept_dir / f"layer{layer_idx}.jsonl"
        with output_file.open("w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        print(f"Saved {len(results)} results to {output_file}")

    print(f"\nDone! Results saved to {concept_dir}")


class _Args(Protocol):
    vector: str
    model: str
    output: str
    samples: int
    multipliers: str
    max_new_tokens: int
    temperature: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steering_geometry.apply_steering",
        description="Apply steering vectors to model generation",
    )
    parser.add_argument(
        "--vector",
        required=True,
        help="Path to steering vector file (.pt)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output",
        default="data/steered/",
        help="Output directory (default: data/steered/)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of negative samples (default: 10)",
    )
    parser.add_argument(
        "--multipliers",
        default="0.01,0.1,1.0,10.0",
        help="Comma-separated multipliers (default: 0.01,0.1,1.0,10.0)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    return parser


def main() -> None:
    args = cast(_Args, cast(object, _build_parser().parse_args()))

    # Parse multipliers
    multipliers = [float(m.strip()) for m in args.multipliers.split(",")]

    config = SteeringConfig(
        multipliers=multipliers,
        num_samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    apply_steering(
        vector_path=Path(args.vector),
        model_name=args.model,
        output_dir=Path(args.output),
        config=config,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "apply_steering",
    "SteeringConfig",
]
