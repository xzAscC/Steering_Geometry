"""Strength x prefix-length ablation grid for activation steering.

Implements the grid search described in the paper's ablation study:

- **Steering strength**: multipliers ``{0.01, 0.1, 1, 10}`` applied to the
  average activation norm ``\\bar{\\alpha}`` (computed over the evaluation
  prompts), so that ``scale = multiplier * \\bar{\\alpha}``.
- **Prefix length**: ``{1, 5, 10, L}`` where ``L`` (encoded as ``None``)
  applies steering over the entire generation sequence.

For every ``(strength, prefix_length)`` cell, the module measures **both**:

- **General capability** via MMLU-Pro accuracy, with raw model responses saved.
- **Steering performance** via concept-specific evaluation (LLM-as-judge for
  sentiment/politeness, refusal-prefix matching or HarmBench for safety).

The full experiment configuration and **all raw model responses** are saved to
disk so every grid cell can be inspected after the fact.

Usage (CLI)::

    uv run python -m steering_geometry.strength_prefix_ablation \\
        --concept sentiment --model "Qwen/Qwen3-1.7B" \\
        --vector outputs/vectors/sentiment/Qwen_Qwen3-1.7B/layer0.7.pt
"""

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Protocol, TypedDict, cast

import torch
from torch import Tensor

from .apply_steering import HarmBenchEvaluator, JudgeEvaluator, MMLUProEvaluator
from .config import HarmBenchConfig, JudgeConfig, MMLUProConfig, ModelConfig
from .extract import load_contrast_pairs
from .models import HookedModel
from .utils import configure_logging, ensure_dir, safe_model_name

logger = logging.getLogger(__name__)

# Default grid straight from the paper ablation paragraph.
DEFAULT_MULTIPLIERS: list[float] = [0.01, 0.1, 1.0, 10.0]
# None encodes "L" — steering applied over the entire sequence.
DEFAULT_PREFIX_LENGTHS: list[int | None] = [1, 5, 10, None]


# =============================================================================
# Output schemas (TypedDicts for persisted JSON)
# =============================================================================


class CellConfig(TypedDict):
    """Configuration snapshot for a single grid cell."""

    multiplier: float
    prefix_length: int | None
    scale: float
    avg_activation_norm: float


class ConceptResponse(TypedDict):
    """One steered generation plus its concept evaluation."""

    prompt: str
    generated_text: str
    concept_score: float
    reasoning: str


class MMLUProResponse(TypedDict):
    """One MMLU-Pro question, the raw steered response, and correctness."""

    question: str
    category: str
    raw_response: str
    predicted: str | None
    ground_truth: str
    correct: bool


class CellResult(TypedDict):
    """Complete result for one ``(multiplier, prefix_length)`` grid cell."""

    config: CellConfig
    concept_score: float
    concept_responses: list[ConceptResponse]
    mmlu_pro_accuracy: float
    mmlu_pro_correct: int
    mmlu_pro_total: int
    mmlu_pro_responses: list[MMLUProResponse]


class AblationConfig(TypedDict):
    """Full experiment configuration, persisted as ``config.json``."""

    concept: str
    model: str
    vector_path: str
    layer_frac: float
    layer_idx: int
    vector_norm: float
    avg_activation_norm: float
    multipliers: list[float]
    prefix_lengths: list[int | None]
    num_samples: int
    max_new_tokens: int
    mmlu_pro_num_questions: int
    seed: int


class AblationResult(TypedDict):
    """Top-level result wrapper returned by :func:`run_strength_prefix_ablation`."""

    config: AblationConfig
    cells: list[CellResult]
    output_dir: str


class SummaryCell(TypedDict):
    """Lightweight per-cell summary (no raw text) for ``summary.json``."""

    multiplier: float
    prefix_length: int | None
    scale: float
    concept_score: float
    mmlu_pro_accuracy: float
    mmlu_pro_correct: int
    mmlu_pro_total: int


# =============================================================================
# Helpers
# =============================================================================


def _load_steering_vector(vector_path: str) -> tuple[Tensor, str]:
    """Load and unit-normalize a steering vector from ``.pt`` file.

    Accepts the three formats produced by the extraction pipeline: a raw
    tensor, a ``{"vector": SteeringVector, ...}`` dict, or a plain
    ``SteeringVector`` object.

    Args:
        vector_path: Path to the saved steering vector.

    Returns:
        Tuple of ``(normalized_vector, concept)`` where ``normalized_vector``
        has unit L2 norm.

    Raises:
        ValueError: If the vector has zero norm or the format is unrecognized.
        FileNotFoundError: If ``vector_path`` does not exist.
    """
    try:
        raw = torch.load(vector_path, map_location="cpu", weights_only=True)
    except Exception:
        logger.warning(
            "Falling back to weights_only=False for %s; only load vectors from trusted sources",
            vector_path,
        )
        raw = torch.load(vector_path, map_location="cpu", weights_only=False)

    concept = "unknown"
    vector: Tensor
    if isinstance(raw, dict) and "vector" in raw:
        sv = raw["vector"]
        concept = getattr(sv, "concept", "unknown")
        if hasattr(sv, "layer_activations"):
            first_layer = sorted(sv.layer_activations.keys())[0]
            vector = sv.layer_activations[first_layer]
        else:
            vector = cast(Tensor, sv)
    elif isinstance(raw, Tensor):
        vector = raw
    else:
        msg = f"Unexpected vector format in {vector_path}: {type(raw).__name__}"
        raise ValueError(msg)

    norm = float(vector.norm())
    if norm <= 0:
        msg = f"Zero-norm vector loaded from {vector_path}"
        raise ValueError(msg)
    return vector / norm, concept


def compute_avg_activation_norm(
    model: HookedModel,
    texts: list[str],
    layer_idx: int,
) -> float:
    """Compute the average L2 norm of hidden-state activations (``\\bar{\\alpha}``).

    Runs a single forward pass over ``texts`` and averages the per-token
    activation norm at ``layer_idx``.  This is the reference scale used to
    set steering strength: ``scale = multiplier * avg_activation_norm``.

    Args:
        model: Loaded ``HookedModel``.
        texts: Prompts over which the norm is averaged.
        layer_idx: Absolute layer index.

    Returns:
        Scalar average activation norm.
    """
    activations = model.get_activations(texts, [layer_idx])
    return float(activations[layer_idx].norm(dim=-1).mean().item())


def _evaluate_concept(
    concept: str,
    prompts: list[str],
    generations: list[str],
    judge_evaluator: JudgeEvaluator | None = None,
    hb_evaluator: HarmBenchEvaluator | None = None,
    hb_behavior_ids: list[str] | None = None,
) -> tuple[float, list[ConceptResponse]]:
    """Evaluate concept steering performance and build response records.

    Dispatches to the appropriate evaluator based on concept type:

    - ``refusal`` with a HarmBench evaluator → HarmBench classifier (async).
    - ``refusal`` without HarmBench → refusal-prefix matching.
    - ``sentiment`` / ``polite`` → LLM-as-judge.

    Args:
        concept: Concept name (``refusal``, ``sentiment``, ``polite``).
        prompts: Prompts that were steered.
        generations: Steered model outputs aligned with ``prompts``.
        judge_evaluator: Judge evaluator for sentiment/polite concepts.
        hb_evaluator: HarmBench evaluator for refusal (may be ``None``).
        hb_behavior_ids: Behavior IDs aligned with ``prompts`` for HarmBench.

    Returns:
        Tuple of ``(aggregate_score, per_response_records)``.  For refusal the
        aggregate is the attack-success-rate (100 − refusal-rate); for
        sentiment/polite it is the mean judge concept score.
    """
    if concept == "refusal" and hb_evaluator is not None and hb_behavior_ids:
        completions: dict[str, list[dict[str, str]]] = {
            bid: [{"generation": text}]
            for bid, text in zip(hb_behavior_ids, generations, strict=True)
        }
        hb_result = asyncio.run(hb_evaluator.evaluate(completions))
        records: list[ConceptResponse] = [
            {
                "prompt": prompt,
                "generated_text": gen,
                "concept_score": 0.0,
                "reasoning": "",
            }
            for prompt, gen in zip(prompts, generations, strict=True)
        ]
        return hb_result.asr, records

    if concept == "refusal":
        refusal_prefixes = HarmBenchEvaluator.REFUSAL_PREFIXES
        records = []
        non_refused = 0
        for prompt, gen in zip(prompts, generations, strict=True):
            is_refused = any(gen.strip().startswith(p) for p in refusal_prefixes)
            if not is_refused:
                non_refused += 1
            records.append(
                {
                    "prompt": prompt,
                    "generated_text": gen,
                    "concept_score": 0.0 if is_refused else 1.0,
                    "reasoning": "refusal-prefix match" if is_refused else "no refusal",
                }
            )
        asr = (non_refused / len(generations) * 100.0) if generations else 0.0
        return asr, records

    # sentiment / polite → LLM-as-judge
    if judge_evaluator is None:
        msg = "judge_evaluator is required for non-refusal concepts"
        raise ValueError(msg)

    scores: list[float] = []
    records = []
    for prompt, gen in zip(prompts, generations, strict=True):
        score = judge_evaluator.evaluate_concept(concept, gen)
        scores.append(float(score.concept_score))
        records.append(
            {
                "prompt": prompt,
                "generated_text": gen,
                "concept_score": float(score.concept_score),
                "reasoning": score.reasoning,
            }
        )
    return (sum(scores) / len(scores)) if scores else 0.0, records


def _evaluate_mmlu_pro(
    mmlu_evaluator: MMLUProEvaluator,
    model: HookedModel,
    steering_vector: Tensor,
    layer_idx: int,
    scale: float,
    steer_tokens: int | None,
) -> tuple[float, int, int, list[MMLUProResponse]]:
    """Run MMLU-Pro under steering and capture raw responses.

    Reuses :class:`MMLUProEvaluator` for dataset loading, prompt formatting,
    and answer extraction, but performs its own generation loop so that the
    **raw model response text** is preserved for each question.

    Args:
        mmlu_evaluator: Configured evaluator (used for helpers only).
        model: Loaded ``HookedModel``.
        steering_vector: Unit-normalized steering vector.
        layer_idx: Absolute layer index.
        scale: Steering scale (``multiplier * avg_activation_norm``).
        steer_tokens: Prefix length (``None`` = full sequence).

    Returns:
        Tuple of ``(accuracy, correct, total, per_question_records)``.
    """
    test_data, val_data = mmlu_evaluator.load_dataset()
    responses: list[MMLUProResponse] = []
    correct = 0

    for question in test_data:
        category = question.get("category", "")
        val_same_cat = [v for v in val_data if v.get("category") == category]
        prompt = mmlu_evaluator.format_prompt(question, val_same_cat)

        raw_response = model.generate_with_steering(
            prompt=prompt,
            layer_idx=layer_idx,
            steering_vector=steering_vector,
            scale=scale,
            max_new_tokens=mmlu_evaluator.config.max_new_tokens,
            temperature=0.0,
            steer_tokens=steer_tokens,
        )

        predicted = mmlu_evaluator.extract_answer(raw_response)
        if predicted is None:
            options = question.get("options", [])
            if options:
                valid_labels = MMLUProEvaluator.CHOICES[: len(options)]
                predicted = random.choice(valid_labels)

        ground_truth = question.get("answer", "")
        is_correct = predicted == ground_truth
        if is_correct:
            correct += 1

        responses.append(
            MMLUProResponse(
                question=question.get("question", ""),
                category=category,
                raw_response=raw_response,
                predicted=predicted,
                ground_truth=ground_truth,
                correct=is_correct,
            )
        )

    total = len(responses)
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return accuracy, correct, total, responses


def _prefix_label(prefix_length: int | None) -> str:
    """Return a human-readable label for a prefix length (``None`` → ``L``)."""
    return "L" if prefix_length is None else str(prefix_length)


def _cell_filename(multiplier: float, prefix_length: int | None) -> str:
    """Build the per-cell JSON filename from multiplier and prefix length."""
    return f"mult{multiplier:g}_prefix{_prefix_label(prefix_length)}.json"


# =============================================================================
# Main entry point
# =============================================================================


def run_strength_prefix_ablation(
    concept: str,
    model_name: str,
    vector_path: str,
    layer_frac: float = 0.7,
    multipliers: list[float] | None = None,
    prefix_lengths: list[int | None] | None = None,
    num_samples: int = 10,
    max_new_tokens: int = 80,
    mmlu_pro_num_questions: int = 100,
    mmlu_pro_use_cot: bool = True,
    evaluate_concept: bool = True,
    evaluate_mmlu: bool = True,
    judge_model: str = "google/gemini-3.1-flash-lite-preview",
    judge_api_base: str = "https://openrouter.ai/api/v1",
    harmbench_classifier_model: str = "google/gemma-4-31B",
    harmbench_classifier_api_base: str = "http://localhost:8000/v1",
    harmbench_behaviors_file: str = "",
    seed: int = 42,
    output_dir: str = "outputs/strength_prefix_ablation",
) -> AblationResult:
    """Run the strength × prefix-length ablation grid.

    For each ``(multiplier, prefix_length)`` cell this generates steered
    responses, evaluates concept steering performance and MMLU-Pro accuracy,
    and saves **all raw responses** alongside the full configuration.

    Args:
        concept: Steering concept (``refusal``, ``sentiment``, ``polite``).
        model_name: HuggingFace model identifier.
        vector_path: Path to a saved steering vector (``.pt``).
        layer_frac: Relative layer position (0.0–1.0).
        multipliers: Strength multipliers applied to ``\\bar{\\alpha}``.
            Defaults to ``[0.01, 0.1, 1.0, 10.0]``.
        prefix_lengths: Prefix token counts; ``None`` = full sequence.
            Defaults to ``[1, 5, 10, None]``.
        num_samples: Number of concept prompts per cell.
        max_new_tokens: Max generation length for concept prompts.
        mmlu_pro_num_questions: Number of MMLU-Pro questions per cell.
        mmlu_pro_use_cot: Use chain-of-thought for MMLU-Pro.
        evaluate_concept: Whether to run concept evaluation.
        evaluate_mmlu: Whether to run MMLU-Pro evaluation.
        judge_model: Judge model for LLM-as-judge.
        judge_api_base: API base URL for the judge.
        harmbench_classifier_model: HarmBench classifier model.
        harmbench_classifier_api_base: HarmBench classifier API base URL.
        harmbench_behaviors_file: Path to HarmBench behaviors CSV.
        seed: Random seed for reproducible sampling.
        output_dir: Directory to save results.

    Returns:
        :class:`AblationResult` with per-cell metrics and raw responses.

    Raises:
        ValueError: If the steering vector has zero norm.
    """
    if multipliers is None:
        multipliers = list(DEFAULT_MULTIPLIERS)
    if prefix_lengths is None:
        prefix_lengths = list(DEFAULT_PREFIX_LENGTHS)

    # ------------------------------------------------------------------
    # Load model and resolve layer
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", model_name)
    model = HookedModel(ModelConfig(model_name=model_name))
    layer_idx = model.resolve_layers([layer_frac])[0]
    logger.info("Layer fraction %.2f → absolute layer %d", layer_frac, layer_idx)

    # ------------------------------------------------------------------
    # Load and normalize steering vector
    # ------------------------------------------------------------------
    normalized_vector, vector_concept = _load_steering_vector(vector_path)
    vector_norm = float(normalized_vector.norm())
    logger.info(
        "Loaded vector from %s (unit norm=%.4f, concept=%s)",
        vector_path,
        vector_norm,
        vector_concept,
    )

    # ------------------------------------------------------------------
    # Select concept prompts and compute ᾱ over them
    # ------------------------------------------------------------------
    rng = random.Random(seed)
    pairs = load_contrast_pairs(concept, num_pairs=max(500, num_samples))
    selected = rng.sample(pairs, min(num_samples, len(pairs)))
    prompts: list[str] = [pair.negative for pair in selected]

    # HarmBench behaviors override prompts for refusal concept
    hb_behavior_ids: list[str] = []
    hb_evaluator: HarmBenchEvaluator | None = None
    if evaluate_concept and concept == "refusal" and harmbench_behaviors_file:
        hb_config = HarmBenchConfig(
            classifier_model=harmbench_classifier_model,
            classifier_api_base=harmbench_classifier_api_base,
            behaviors_file=harmbench_behaviors_file,
        )
        hb_evaluator = HarmBenchEvaluator(hb_config)
        hb_evaluator.load_behaviors(harmbench_behaviors_file)
        behaviors = hb_evaluator.behaviors[:num_samples]
        prompts = [b["behavior"] for b in behaviors]
        hb_behavior_ids = [b["behavior_id"] for b in behaviors]
        logger.info("Using %d HarmBench behaviors as concept prompts", len(prompts))
    else:
        logger.info("Selected %d negative prompts for concept '%s'", len(prompts), concept)

    avg_activation_norm = compute_avg_activation_norm(model, prompts, layer_idx)
    logger.info("Average activation norm ᾱ = %.4f at layer %d", avg_activation_norm, layer_idx)

    # ------------------------------------------------------------------
    # Initialize evaluators
    # ------------------------------------------------------------------
    judge_evaluator: JudgeEvaluator | None = None
    if evaluate_concept and concept != "refusal":
        judge_evaluator = JudgeEvaluator(JudgeConfig(model=judge_model, api_base=judge_api_base))
        logger.info("Initialized JudgeEvaluator for '%s'", concept)

    mmlu_evaluator: MMLUProEvaluator | None = None
    if evaluate_mmlu:
        mmlu_config = MMLUProConfig(
            num_questions=mmlu_pro_num_questions,
            use_cot=mmlu_pro_use_cot,
        )
        mmlu_evaluator = MMLUProEvaluator(mmlu_config, model)
        logger.info("Initialized MMLUProEvaluator (%d questions)", mmlu_pro_num_questions)

    # ------------------------------------------------------------------
    # Persist the full configuration
    # ------------------------------------------------------------------
    safe_model = safe_model_name(model_name)
    result_dir = ensure_dir(Path(output_dir) / concept / safe_model)
    cells_dir = ensure_dir(result_dir / "cells")

    ablation_config = AblationConfig(
        concept=concept,
        model=model_name,
        vector_path=vector_path,
        layer_frac=layer_frac,
        layer_idx=layer_idx,
        vector_norm=vector_norm,
        avg_activation_norm=avg_activation_norm,
        multipliers=multipliers,
        prefix_lengths=prefix_lengths,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        mmlu_pro_num_questions=mmlu_pro_num_questions,
        seed=seed,
    )
    config_path = result_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(ablation_config, f, indent=2)
    logger.info("Saved configuration to %s", config_path)

    # ------------------------------------------------------------------
    # Sweep the grid
    # ------------------------------------------------------------------
    cells: list[CellResult] = []
    total_cells = len(multipliers) * len(prefix_lengths)
    cell_count = 0

    for mult in multipliers:
        scale = mult * avg_activation_norm
        for prefix_length in prefix_lengths:
            cell_count += 1
            logger.info(
                "Cell %d/%d: multiplier=%g, prefix=%s, scale=%.4f",
                cell_count,
                total_cells,
                mult,
                _prefix_label(prefix_length),
                scale,
            )

            # --- Generate steered concept responses ---
            generations: list[str] = []
            for prompt in prompts:
                generated = model.generate_with_steering(
                    prompt=prompt,
                    layer_idx=layer_idx,
                    steering_vector=normalized_vector,
                    scale=scale,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    steer_tokens=prefix_length,
                )
                generations.append(generated)

            # --- Evaluate concept steering performance ---
            concept_score = 0.0
            concept_responses: list[ConceptResponse] = []
            if evaluate_concept:
                concept_score, concept_responses = _evaluate_concept(
                    concept=concept,
                    prompts=prompts,
                    generations=generations,
                    judge_evaluator=judge_evaluator,
                    hb_evaluator=hb_evaluator,
                    hb_behavior_ids=hb_behavior_ids or None,
                )
                logger.info("  Concept score: %.2f", concept_score)

            # --- Evaluate MMLU-Pro with raw responses captured ---
            mmlu_accuracy = 0.0
            mmlu_correct = 0
            mmlu_total = 0
            mmlu_responses: list[MMLUProResponse] = []
            if evaluate_mmlu and mmlu_evaluator is not None:
                mmlu_accuracy, mmlu_correct, mmlu_total, mmlu_responses = _evaluate_mmlu_pro(
                    mmlu_evaluator=mmlu_evaluator,
                    model=model,
                    steering_vector=normalized_vector,
                    layer_idx=layer_idx,
                    scale=scale,
                    steer_tokens=prefix_length,
                )
                logger.info(
                    "  MMLU-Pro: %.2f%% (%d/%d)",
                    mmlu_accuracy,
                    mmlu_correct,
                    mmlu_total,
                )

            cell_config = CellConfig(
                multiplier=mult,
                prefix_length=prefix_length,
                scale=scale,
                avg_activation_norm=avg_activation_norm,
            )
            cell_result = CellResult(
                config=cell_config,
                concept_score=round(concept_score, 4),
                concept_responses=concept_responses,
                mmlu_pro_accuracy=round(mmlu_accuracy, 4),
                mmlu_pro_correct=mmlu_correct,
                mmlu_pro_total=mmlu_total,
                mmlu_pro_responses=mmlu_responses,
            )
            cells.append(cell_result)

            # --- Persist per-cell file with raw responses ---
            cell_path = cells_dir / _cell_filename(mult, prefix_length)
            with cell_path.open("w") as f:
                json.dump(cell_result, f, indent=2)
            logger.info("  Saved cell → %s", cell_path)

    # ------------------------------------------------------------------
    # Persist summary (no raw text)
    # ------------------------------------------------------------------
    summary_cells: list[SummaryCell] = [
        SummaryCell(
            multiplier=c["config"]["multiplier"],
            prefix_length=c["config"]["prefix_length"],
            scale=c["config"]["scale"],
            concept_score=c["concept_score"],
            mmlu_pro_accuracy=c["mmlu_pro_accuracy"],
            mmlu_pro_correct=c["mmlu_pro_correct"],
            mmlu_pro_total=c["mmlu_pro_total"],
        )
        for c in cells
    ]
    summary_path = result_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "config": ablation_config,
                "cells": summary_cells,
            },
            f,
            indent=2,
        )
    logger.info("Saved grid summary to %s", summary_path)

    result = AblationResult(
        config=ablation_config,
        cells=cells,
        output_dir=str(result_dir),
    )
    logger.info("Ablation complete. Results in %s", result_dir)
    return result


# =============================================================================
# CLI
# =============================================================================


class _Args(Protocol):
    """Protocol for CLI arguments."""

    concept: str
    model: str
    vector: str
    layer: float
    multipliers: str
    prefix_lengths: str
    num_samples: int
    max_new_tokens: int
    mmlu_pro_num_questions: int
    no_eval_concept: bool
    no_eval_mmlu: bool
    judge_model: str
    judge_api_base: str
    harmbench_classifier_model: str
    harmbench_classifier_api_base: str
    harmbench_behaviors_file: str
    seed: int
    output: str
    log_level: str


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the ablation CLI."""
    parser = argparse.ArgumentParser(
        prog="steering_geometry.strength_prefix_ablation",
        description="Strength x prefix-length ablation grid with response saving",
    )
    parser.add_argument(
        "--concept",
        required=True,
        choices=["refusal", "sentiment", "polite"],
        help="Steering concept to evaluate",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--vector",
        required=True,
        help="Path to steering vector (.pt file)",
    )
    parser.add_argument(
        "--layer",
        type=float,
        default=0.7,
        help="Relative layer position (default: 0.7)",
    )
    parser.add_argument(
        "--multipliers",
        default="",
        help="Comma-separated strength multipliers (default: '0.01,0.1,1,10')",
    )
    parser.add_argument(
        "--prefix-lengths",
        default="",
        help=(
            "Comma-separated prefix lengths. Use 'L' or 'all' for full "
            "sequence (default: '1,5,10,L')"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of concept prompt samples per cell (default: 10)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Max generation length for concept prompts (default: 80)",
    )
    parser.add_argument(
        "--mmlu-pro-num-questions",
        type=int,
        default=100,
        help="Number of MMLU-Pro questions per cell (default: 100)",
    )
    parser.add_argument(
        "--no-eval-concept",
        action="store_true",
        help="Skip concept evaluation",
    )
    parser.add_argument(
        "--no-eval-mmlu",
        action="store_true",
        help="Skip MMLU-Pro evaluation",
    )
    parser.add_argument(
        "--judge-model",
        default="google/gemini-3.1-flash-lite-preview",
        help="Judge model for LLM-as-judge",
    )
    parser.add_argument(
        "--judge-api-base",
        default="https://openrouter.ai/api/v1",
        help="API base URL for judge model",
    )
    parser.add_argument(
        "--harmbench-classifier-model",
        default="google/gemma-4-31B",
        help="HarmBench classifier model",
    )
    parser.add_argument(
        "--harmbench-classifier-api-base",
        default="http://localhost:8000/v1",
        help="HarmBench classifier API base URL",
    )
    parser.add_argument(
        "--harmbench-behaviors-file",
        default="",
        help="Path to HarmBench behaviors CSV (refusal only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="outputs/strength_prefix_ablation",
        help="Output directory (default: outputs/strength_prefix_ablation)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def _parse_prefix_lengths(raw: str) -> list[int | None]:
    """Parse a comma-separated prefix-length string into a list.

    Accepts integers plus the tokens ``L`` and ``all`` (both map to ``None``,
    meaning full-sequence steering).

    Args:
        raw: Comma-separated values, e.g. ``"1,5,10,L"``.

    Returns:
        List of prefix lengths with ``None`` for full-sequence entries.

    Raises:
        ValueError: If a token cannot be parsed.
    """
    result: list[int | None] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in ("l", "all", "none", "full"):
            result.append(None)
        else:
            result.append(int(token))
    return result


def main() -> None:
    """CLI entry point for the strength × prefix-length ablation."""
    args = cast(_Args, _build_parser().parse_args())
    configure_logging(level=args.log_level)

    multipliers: list[float] | None = None
    if args.multipliers:
        multipliers = [float(x) for x in args.multipliers.split(",")]

    prefix_lengths: list[int | None] | None = None
    if args.prefix_lengths:
        prefix_lengths = _parse_prefix_lengths(args.prefix_lengths)

    run_strength_prefix_ablation(
        concept=args.concept,
        model_name=args.model,
        vector_path=args.vector,
        layer_frac=args.layer,
        multipliers=multipliers,
        prefix_lengths=prefix_lengths,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        mmlu_pro_num_questions=args.mmlu_pro_num_questions,
        evaluate_concept=not args.no_eval_concept,
        evaluate_mmlu=not args.no_eval_mmlu,
        judge_model=args.judge_model,
        judge_api_base=args.judge_api_base,
        harmbench_classifier_model=args.harmbench_classifier_model,
        harmbench_classifier_api_base=args.harmbench_classifier_api_base,
        harmbench_behaviors_file=args.harmbench_behaviors_file,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_MULTIPLIERS",
    "DEFAULT_PREFIX_LENGTHS",
    "CellConfig",
    "ConceptResponse",
    "MMLUProResponse",
    "CellResult",
    "AblationConfig",
    "AblationResult",
    "SummaryCell",
    "compute_avg_activation_norm",
    "run_strength_prefix_ablation",
]
