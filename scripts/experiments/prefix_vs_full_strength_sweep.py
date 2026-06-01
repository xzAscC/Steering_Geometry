"""Prefix Steering vs Full Steering with varying steering strengths.

This experiment investigates the hypothesis that the number of prefix tokens
needed for effective steering depends on steering strength:
- At very large steering strengths, even 1-2 prefix tokens may suffice
- At lower strengths, more prefix tokens may be needed to achieve the same effect
- There is a tradeoff between steering strength, prefix length, and output quality

We measure:
1. Steering effectiveness (how much the output changes from baseline)
2. Output quality (coherence and fluency)
3. The interaction between prefix token count and steering strength
"""

import json
import logging
from pathlib import Path

import torch

from steering_geometry.config import ModelConfig
from steering_geometry.extract import load_contrast_pairs
from steering_geometry.models import HookedModel
from steering_geometry.utils import configure_logging

logger = logging.getLogger(__name__)


def run_prefix_vs_full_strength_sweep(
    concept: str = "sentiment",
    model_name: str = "Qwen/Qwen3-1.7B",
    vector_path: str = "outputs/vectors/sentiment/discriminative/k128_layer0.7.pt",
    layer_frac: float = 0.7,
    steer_tokens_values: list[int | None] | None = None,
    multipliers: list[float] | None = None,
    num_samples: int = 10,
    max_new_tokens: int = 80,
    output_dir: str = "outputs/prefix_vs_full",
) -> dict[str, object]:
    """Run prefix vs full steering sweep with varying strengths.

    Args:
        concept: Concept to test (sentiment, refusal, polite).
        model_name: HuggingFace model name.
        vector_path: Path to the steering vector (.pt file).
        layer_frac: Relative layer position.
        steer_tokens_values: List of steer_tokens (None=all/full steering).
        multipliers: List of steering strength multipliers.
        num_samples: Number of prompt samples.
        max_new_tokens: Max tokens to generate.
        output_dir: Output directory.

    Returns:
        Dict with results summary.
    """
    if steer_tokens_values is None:
        steer_tokens_values = [None, 1, 2, 3, 5, 10, 20, 50]
    if multipliers is None:
        multipliers = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model: %s", model_name)
    model = HookedModel(ModelConfig(model_name=model_name))
    abs_layer = model.resolve_layers([layer_frac])[0]
    logger.info("Using layer fraction %.2f = absolute layer %d", layer_frac, abs_layer)

    # Load steering vector
    vector = torch.load(vector_path, map_location="cpu", weights_only=True)
    vector_norm = float(vector.norm())
    logger.info("Loaded vector from %s, norm=%.2f", vector_path, vector_norm)

    # Normalize the vector
    if vector_norm > 0:
        normalized_vector = vector / vector_norm
    else:
        raise ValueError("Zero-norm vector")

    # Load prompts (negative samples = the "unsteered" direction)
    pairs = load_contrast_pairs(concept, num_pairs=max(500, num_samples))
    import random

    rng = random.Random(42)
    selected = rng.sample(pairs, min(num_samples, len(pairs)))
    prompts = [pair.negative for pair in selected]

    logger.info("Loaded %d prompts for concept '%s'", len(prompts), concept)

    # Generate baseline (no steering)
    logger.info("Generating baseline (no steering)...")
    baselines: list[str] = []
    for prompt in prompts:
        inputs = model.tokenizer(prompt, return_tensors="pt")
        device = next(model.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=model.tokenizer.pad_token_id,
            )
        text = model.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        baselines.append(text)

    # Run sweep
    all_results: list[dict[str, object]] = []

    for mult in multipliers:
        # scale = multiplier * avg_activation_norm (using vector_norm as proxy)
        scale = mult * vector_norm

        for steer_tokens in steer_tokens_values:
            steer_label = "full" if steer_tokens is None else str(steer_tokens)

            logger.info("Running mult=%.3f, steer_tokens=%s, scale=%.2f", mult, steer_label, scale)

            generations: list[str] = []
            for prompt in prompts:
                generated = model.generate_with_steering(
                    prompt=prompt,
                    layer_idx=abs_layer,
                    steering_vector=normalized_vector,
                    scale=scale,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    steer_tokens=steer_tokens,
                )
                generations.append(generated)

            # Compute per-sample metrics
            for i, (baseline, generated) in enumerate(zip(baselines, generations, strict=True)):
                all_results.append(
                    {
                        "concept": concept,
                        "model": model_name,
                        "layer_frac": layer_frac,
                        "layer_idx": abs_layer,
                        "multiplier": mult,
                        "scale": scale,
                        "steer_tokens": steer_tokens,
                        "steer_label": steer_label,
                        "sample_idx": i,
                        "prompt": prompts[i],
                        "baseline_text": baseline,
                        "generated_text": generated,
                    }
                )

    # Save all results
    results_path = output_path / f"{concept}_prefix_vs_full.jsonl"
    with results_path.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    logger.info("Saved %d results to %s", len(all_results), results_path)

    # Compute summary table
    summary = _compute_summary(all_results, baselines, multipliers, steer_tokens_values)
    summary_path = output_path / f"{concept}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Print the table
    _print_summary_table(summary, multipliers, steer_tokens_values, concept)

    return summary


def _compute_divergence(text1: str, text2: str) -> float:
    """Simple character-level divergence between two texts."""
    if not text1 and not text2:
        return 0.0
    if not text1 or not text2:
        return 1.0

    # Use token-level Jaccard-like metric as proxy
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 0.0
    if not words1 or not words2:
        return 1.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return 1.0 - (intersection / union) if union > 0 else 0.0


def _compute_summary(
    results: list[dict[str, object]],
    baselines: list[str],
    multipliers: list[float],
    steer_tokens_values: list[int | None],
) -> dict[str, object]:
    """Compute summary metrics for each (multiplier, steer_tokens) combination."""
    from collections import defaultdict

    # Group results by (multiplier, steer_tokens)
    grouped: dict[tuple[float, int | None], list[dict[str, object]]] = defaultdict(list)
    for r in results:
        key = (float(r["multiplier"]), r["steer_tokens"])
        grouped[key].append(r)

    summary_entries: list[dict[str, object]] = []

    for mult in multipliers:
        for st in steer_tokens_values:
            key = (mult, st)
            entries = grouped.get(key, [])
            if not entries:
                continue

            steer_label = "full" if st is None else str(st)

            # Compute average divergence from baseline
            divergences = [
                _compute_divergence(str(e["baseline_text"]), str(e["generated_text"]))
                for e in entries
            ]
            avg_divergence = sum(divergences) / len(divergences) if divergences else 0.0

            # Compute average generation length (proxy for fluency - very short = broken)
            gen_lengths = [len(str(e["generated_text"]).split()) for e in entries]
            avg_gen_length = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0.0

            # Compute how similar prefix steering is to full steering at same multiplier
            full_key = (mult, None)
            full_entries = grouped.get(full_key, [])

            if full_entries and st is not None:
                full_texts = {int(e["sample_idx"]): str(e["generated_text"]) for e in full_entries}
                prefix_vs_full_divs = []
                for e in entries:
                    idx = int(e["sample_idx"])
                    if idx in full_texts:
                        prefix_vs_full_divs.append(
                            _compute_divergence(full_texts[idx], str(e["generated_text"]))
                        )
                avg_prefix_vs_full = (
                    sum(prefix_vs_full_divs) / len(prefix_vs_full_divs)
                    if prefix_vs_full_divs
                    else 0.0
                )
            else:
                avg_prefix_vs_full = 0.0

            summary_entries.append(
                {
                    "multiplier": mult,
                    "steer_tokens": st,
                    "steer_label": steer_label,
                    "avg_divergence_from_baseline": round(avg_divergence, 4),
                    "avg_gen_length_words": round(avg_gen_length, 1),
                    "avg_divergence_from_full": round(avg_prefix_vs_full, 4),
                    "num_samples": len(entries),
                }
            )

    return {
        "concept": results[0]["concept"] if results else "unknown",
        "model": results[0]["model"] if results else "unknown",
        "layer_frac": results[0]["layer_frac"] if results else 0.0,
        "entries": summary_entries,
    }


def _print_summary_table(
    summary: dict[str, object],
    multipliers: list[float],
    steer_tokens_values: list[int | None],
    concept: str,
) -> None:
    """Log a formatted summary table via structured logging."""
    entries = summary["entries"]
    entry_map: dict[tuple[float, int | None], dict[str, object]] = {}
    for e in entries:
        entry_map[(float(e["multiplier"]), e["steer_tokens"])] = e

    sep = "=" * 120
    model_name = str(summary["model"])

    logger.info("\n%s", sep)
    logger.info("PREFIX STEERING vs FULL STEERING — %s (%s)", concept.upper(), model_name)
    logger.info("%s", sep)

    # Table 1: Divergence from baseline (higher = more steering effect)
    logger.info("")
    logger.info("Divergence from No-Steering Baseline (higher = stronger steering effect)")
    logger.info("%s", "-" * 120)

    header = f"{'mult':>8s}" + "".join(
        f" {'full' if st is None else f'pre={st}':>10s}" for st in steer_tokens_values
    )
    logger.info("%s", header)
    logger.info("%s", "-" * 120)

    for mult in multipliers:
        row = f"{mult:>8.2f}"
        for st in steer_tokens_values:
            e = entry_map.get((mult, st))
            if e:
                val = float(e["avg_divergence_from_baseline"])
                row += f" {val:>10.4f}"
            else:
                row += f" {'N/A':>10s}"
        logger.info("%s", row)

    # Table 2: Divergence from full steering (lower = prefix ≈ full steering)
    logger.info("")
    logger.info("Divergence of Prefix from Full Steering (lower = prefix approximates full)")
    logger.info("%s", "-" * 120)

    header = f"{'mult':>8s}" + "".join(
        f" {'—':>10s}" if st is None else f" {f'pre={st}':>10s}" for st in steer_tokens_values
    )
    logger.info("%s", header)
    logger.info("%s", "-" * 120)

    for mult in multipliers:
        row = f"{mult:>8.2f}"
        for st in steer_tokens_values:
            if st is None:
                row += f" {'—':>10s}"
            else:
                e = entry_map.get((mult, st))
                if e:
                    val = float(e["avg_divergence_from_full"])
                    row += f" {val:>10.4f}"
                else:
                    row += f" {'N/A':>10s}"
        logger.info("%s", row)

    # Table 3: Average generation length (proxy for fluency)
    logger.info("")
    logger.info("Average Generation Length in Words (proxy for fluency — very short = broken)")
    logger.info("%s", "-" * 120)

    header = f"{'mult':>8s}" + "".join(
        f" {'full' if st is None else f'pre={st}':>10s}" for st in steer_tokens_values
    )
    logger.info("%s", header)
    logger.info("%s", "-" * 120)

    for mult in multipliers:
        row = f"{mult:>8.2f}"
        for st in steer_tokens_values:
            e = entry_map.get((mult, st))
            if e:
                val = float(e["avg_gen_length_words"])
                row += f" {val:>10.1f}"
            else:
                row += f" {'N/A':>10s}"
        logger.info("%s", row)

    # Key finding: for each multiplier, what's the minimum prefix tokens that
    # achieve divergence_from_full < 0.3 (close to full steering)?
    logger.info("")
    logger.info("Minimum Prefix Tokens Needed to Approximate Full Steering")
    logger.info("  (divergence_from_full < 0.3 threshold)")
    logger.info("%s", "-" * 60)

    for mult in multipliers:
        min_prefix = None
        for st in steer_tokens_values:
            if st is None:
                continue
            e = entry_map.get((mult, st))
            if e and float(e["avg_divergence_from_full"]) < 0.3:
                min_prefix = st
                break
        if min_prefix is not None:
            logger.info("  mult=%6.2f: min prefix = %d tokens", mult, min_prefix)
        else:
            # Find closest
            best_st = None
            best_val = 1.0
            for st in steer_tokens_values:
                if st is None:
                    continue
                e = entry_map.get((mult, st))
                if e and float(e["avg_divergence_from_full"]) < best_val:
                    best_val = float(e["avg_divergence_from_full"])
                    best_st = st
            if best_st:
                logger.info(
                    "  mult=%6.2f: NOT reached (best: pre=%d, div=%.4f)", mult, best_st, best_val
                )
            else:
                logger.info("  mult=%6.2f: no data", mult)

    logger.info("\n%s", sep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prefix vs Full Steering with strength sweep")
    parser.add_argument(
        "--concept", default="sentiment", choices=["sentiment", "refusal", "polite"]
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--vector", default="outputs/vectors/sentiment/discriminative/k128_layer0.7.pt"
    )
    parser.add_argument("--layer", type=float, default=0.7)
    parser.add_argument(
        "--steer-tokens",
        default=None,
        help="Comma-separated steer_tokens values (e.g., '1,2,5,10')",
    )
    parser.add_argument(
        "--multipliers",
        default=None,
        help="Comma-separated multipliers (e.g., '0.01,0.1,1.0,10.0')",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", default="outputs/prefix_vs_full")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level)

    steer_tokens_values: list[int | None] | None = None
    if args.steer_tokens:
        steer_tokens_values = [None] + [int(x) for x in args.steer_tokens.split(",")]

    multipliers: list[float] | None = None
    if args.multipliers:
        multipliers = [float(x) for x in args.multipliers.split(",")]

    run_prefix_vs_full_strength_sweep(
        concept=args.concept,
        model_name=args.model,
        vector_path=args.vector,
        layer_frac=args.layer,
        steer_tokens_values=steer_tokens_values,
        multipliers=multipliers,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output,
    )
