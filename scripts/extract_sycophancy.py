import argparse
from pathlib import Path
from typing import Protocol, cast

import torch

from steering_geometry.concepts.sycophancy import evaluate_sycophancy, load_sycophancy_data
from steering_geometry.config import ConceptConfig, EvaluationConfig, ExtractionConfig, ModelConfig
from steering_geometry.extraction import extract_steering_vector
from steering_geometry.models import HookedModel


class _Args(Protocol):
    model: str
    method: str
    num_pairs: int
    output: str
    dry_run: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    _ = parser.add_argument("--method", choices=["mean", "pca"], default="mean")
    _ = parser.add_argument("--num-pairs", type=int, default=500)
    _ = parser.add_argument("--output", default="data/vectors/")
    _ = parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = cast(_Args, cast(object, _build_parser().parse_args()))

    concept_config = ConceptConfig(
        concept_name="sycophancy",
        dataset_name="synthetic",
        num_pairs=args.num_pairs,
    )
    pairs = load_sycophancy_data(concept_config)
    print(f"Loaded {len(pairs)} contrast pairs")

    positive_lengths = [len(pair.positive.split()) for pair in pairs]
    negative_lengths = [len(pair.negative.split()) for pair in pairs]
    avg_positive_length = sum(positive_lengths) / len(positive_lengths)
    avg_negative_length = sum(negative_lengths) / len(negative_lengths)
    print(f"Avg positive length: {avg_positive_length:.2f} words")
    print(f"Avg negative length: {avg_negative_length:.2f} words")

    if args.dry_run:
        print("Dry run complete")
        return

    model = HookedModel(ModelConfig(model_name=args.model))
    extraction_config = ExtractionConfig(method=args.method)
    vector = extract_steering_vector(model=model, pairs=pairs, config=extraction_config)

    evaluation = evaluate_sycophancy(
        model=model,
        vector=vector,
        config=EvaluationConfig(),
    )
    print(f"Sycophancy shift: {evaluation.scores['sycophancy_shift']:.4f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = args.model.replace("/", "_")
    output_file = output_dir / f"sycophancy_{safe_model_name}_{args.method}.pt"
    torch.save(
        {
            "vector": vector,
            "evaluation": evaluation,
            "num_pairs": len(pairs),
        },
        output_file,
    )
    print(f"Saved steering vector to {output_file}")


if __name__ == "__main__":
    main()
