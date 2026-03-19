"""Apply steering vectors to model generation with evaluation support.

This module applies extracted steering vectors to model generation,
steering model behavior towards or away from specific concepts.

It also includes evaluation tools for measuring steering effectiveness:
- JudgeEvaluator: Uses LLM judges to score steered responses on concept
  adherence and fluency
- MMLUEvaluator: Measures downstream task performance with steering applied
- generate_html_report: Creates visual reports from evaluation results

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
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from jinja2 import Template
from openai import AsyncOpenAI
from torch import Tensor

from .config import JudgeConfig, MMLUConfig, ModelConfig, SteeringConfig
from .extract import load_contrast_pairs
from .models import HookedModel
from .types import (
    EvaluationMetadata,
    EvaluationResult,
    JudgeScore,
    MMLUPrediction,
    MMLUQuestion,
    MMLUResult,
    SteeringVector,
)
from .utils import clamp_score, ensure_dir, safe_model_name

# =============================================================================
# Evaluation Module - Judge Prompts and HTML Template
# =============================================================================

JUDGE_CONCEPT_PROMPT = """You are an impartial judge evaluating concept presence.

CONCEPT: {concept}
GENERATED TEXT: {text}

Evaluate how well the concept is present on a scale of 0-10:
0-2: Concept absent or barely recognizable
3-4: Concept minimally or awkwardly present
5-6: Concept moderately present
7-8: Concept strongly present
9-10: Concept fully and naturally integrated

Instructions:
1. Identify references to the concept
2. Assess naturalness of integration
3. Provide reasoning in 2-3 sentences
4. End with: "Rating: [[X]]"

Your evaluation:"""

JUDGE_FLUENCY_PROMPT = """You are an impartial judge evaluating text fluency.

TEXT: {text}

Evaluate the fluency on a scale of 0-10:
0-2: Unintelligible or severely broken
3-4: Many grammatical errors, hard to understand
5-6: Readable but with some issues
7-8: Mostly fluent with minor issues
9-10: Perfectly natural and fluent

Instructions:
1. Assess grammar, coherence, and naturalness
2. Provide reasoning in 2-3 sentences
3. End with: "Rating: [[X]]"

Your evaluation:"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Steering Evaluation Report</title>
    <style>
        body {
            font-family: system-ui, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
        }
        h1 { color: #1a1a1a; }
        h2 {
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        th { background-color: #f5f5f5; }
        tr:nth-child(even) { background-color: #fafafa; }
        .metric {
            font-size: 1.2em;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Steering Evaluation Report</h1>

    <h2>Configuration</h2>
    <table>
        <tr><td>Concept</td><td>{{ metadata.concept }}</td></tr>
        <tr><td>Model</td><td>{{ metadata.model }}</td></tr>
        <tr><td>Layer</td><td>{{ metadata.layer }}</td></tr>
        <tr><td>Multiplier</td><td>{{ metadata.multiplier }}</td></tr>
    </table>

    <h2>Steering Effectiveness</h2>
    <table>
        <tr>
            <th>Sample</th>
            <th>Concept</th>
            <th>Fluency</th>
            <th>Final</th>
        </tr>
        {% for score in judge_scores %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ score.concept_score }}</td>
            <td>{{ score.fluency_score }}</td>
            <td>{{ "%.2f"|format(score.final_score) }}</td>
        </tr>
        {% endfor %}
        <tr>
            <td><strong>Average</strong></td>
            <td>{{ "%.2f"|format(avg_concept) }}</td>
            <td>{{ "%.2f"|format(avg_fluency) }}</td>
            <td>{{ "%.2f"|format(avg_final) }}</td>
        </tr>
    </table>

    <h2>General Ability (MMLU-Pro)</h2>
    <p class="metric">
        Accuracy: {{ "%.2f"|format(mmlu_accuracy) }}%
        ({{ mmlu_correct }}/{{ mmlu_total }})
    </p>
</body>
</html>"""


# =============================================================================
# Evaluation Classes
# =============================================================================


class JudgeEvaluator:
    """Evaluates steered responses using an LLM-as-judge.

    Uses a separate judge model to score how well steered outputs match
    target concepts while maintaining fluency.
    """

    def __init__(self, config: JudgeConfig) -> None:
        """Initialize the judge evaluator.

        Args:
            config: Configuration for the judge model.
        """
        self.config = config
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=api_key,
        )

    async def _call_api(self, prompt: str) -> str:
        """Call the OpenRouter API with retry logic.

        Args:
            prompt: The prompt to send to the API.

        Returns:
            The response text, or empty string on failure.
        """
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                )
                content = response.choices[0].message.content
                return content if content else ""
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(1.0)

        if last_error is not None:
            raise last_error
        return ""

    def _extract_score(self, response: str) -> int:
        """Extract numeric score from judge response.

        Uses multi-layer regex to extract score from various formats.

        Args:
            response: Raw response text from the judge model.

        Returns:
            Extracted integer score (0-10), or -1 if extraction fails.
        """
        # Primary: "Rating: [[X]]"
        match = re.search(r"Rating:\s*\[\[(\d+)\]\]", response, re.IGNORECASE)
        if match:
            return clamp_score(int(match.group(1)))

        # Fallback: "Rating: X" or "X/10"
        match = re.search(r"Rating:\s*(\d+)", response, re.IGNORECASE)
        if match:
            return clamp_score(int(match.group(1)))

        match = re.search(r"(\d+)/10", response)
        if match:
            return clamp_score(int(match.group(1)))

        # Last resort: standalone digit
        match = re.search(r"\b(\d+)\b", response)
        if match:
            return clamp_score(int(match.group(1)))

        return -1

    def evaluate_concept(self, concept: str, text: str) -> JudgeScore:
        """Evaluate how well text matches a target concept.

        Args:
            concept: The concept to evaluate (e.g., "honesty", "toxicity").
            text: The text to evaluate.

        Returns:
            JudgeScore with concept adherence, fluency, and reasoning.
        """
        prompt = JUDGE_CONCEPT_PROMPT.format(concept=concept, text=text)
        response = asyncio.run(self._call_api(prompt))
        score = self._extract_score(response)
        return JudgeScore(
            concept_score=score,
            fluency_score=10,  # Placeholder
            final_score=float(score),
            reasoning=response,
        )

    def evaluate_fluency(self, text: str) -> int:
        """Evaluate text fluency on a 0-10 scale.

        Args:
            text: The text to evaluate.

        Returns:
            Fluency score from 0 (unintelligible) to 10 (perfectly natural).
        """
        prompt = JUDGE_FLUENCY_PROMPT.format(text=text)
        response = asyncio.run(self._call_api(prompt))
        return self._extract_score(response)

    def evaluate_dual(self, concept: str, text: str) -> JudgeScore:
        """Evaluate both concept adherence and fluency in a single call.

        Args:
            concept: The concept to evaluate.
            text: The text to evaluate.

        Returns:
            JudgeScore with combined metrics.
        """
        concept_score = self.evaluate_concept(concept, text)
        fluency_score = self.evaluate_fluency(text)

        # Compute harmonic mean
        if concept_score.concept_score <= 0 or fluency_score <= 0:
            final_score = 0.0
        else:
            final_score = 2.0 / (1.0 / concept_score.concept_score + 1.0 / fluency_score)

        return JudgeScore(
            concept_score=concept_score.concept_score,
            fluency_score=fluency_score,
            final_score=final_score,
            reasoning=f"Concept: {concept_score.reasoning}\n\nFluency: {fluency_score}",
        )


class MMLUEvaluator:
    """Evaluates steering vector impact on MMLU benchmark performance.

    Measures how steering affects model performance on multiple-choice
    academic questions across various subjects.
    """

    def __init__(self, config: MMLUConfig, model: Any) -> None:
        """Initialize the MMLU evaluator.

        Args:
            config: Configuration for MMLU evaluation.
            model: The model to evaluate (with steering hooks applied).
        """
        self.config = config
        self.model = model
        self._dataset: list[MMLUQuestion] | None = None

    def load_validation_set(self) -> list[MMLUQuestion]:
        """Load MMLU validation set questions.

        Returns:
            List of question dictionaries with choices and answers.
        """
        if self._dataset is None:
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
            questions = list(ds)
            random.seed(self.config.seed)
            random.shuffle(questions)
            self._dataset = questions[: self.config.num_questions]
        return self._dataset

    def format_prompt(self, question: MMLUQuestion, use_cot: bool) -> str:
        """Format an MMLU question as a prompt for the model.

        Args:
            question: Question dictionary with text and choices.
            use_cot: Whether to include chain-of-thought prompting.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [f"Question: {question.get('question', '')}"]

        choices = question.get("options", [])
        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for i, choice in enumerate(choices[:10]):
            if i < len(labels):
                prompt_parts.append(f"{labels[i]}. {choice}")

        if use_cot:
            prompt_parts.append("Let's think step by step.")

        prompt_parts.append("Answer:")
        return "\n".join(prompt_parts)

    def extract_answer(self, response: str) -> str | None:
        """Extract the answer choice from model response.

        Uses multi-layer regex to extract the answer letter.

        Args:
            response: Raw model response text.

        Returns:
            Extracted answer (e.g., "A", "B", "C", "D") or None if not found.
        """
        # Primary: "answer is (A)"
        match = re.search(r"answer is \(([A-J])\)", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Fallback: "Answer: A" or standalone letter
        match = re.search(r".*[aA]nswer:\s*([A-J])", response)
        if match:
            return match.group(1).upper()

        # Last resort: final standalone letter A-J
        match = re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", response)
        return match.group(1).upper() if match else None

    def evaluate(
        self,
        steering_vector: Tensor,
        layer_idx: int,
        scale: float,
    ) -> MMLUResult:
        """Run MMLU evaluation with steering applied.

        Args:
            steering_vector: The steering vector to apply.
            layer_idx: Layer index to apply steering at.
            scale: Scale factor for steering strength.

        Returns:
            MMLUResult with accuracy and predictions.
        """
        questions = self.load_validation_set()
        predictions: list[MMLUPrediction] = []
        correct = 0

        for question in questions:
            prompt = self.format_prompt(question, self.config.use_cot)
            response = self.model.generate_with_steering(
                prompt=prompt,
                layer_idx=layer_idx,
                steering_vector=steering_vector,
                scale=scale,
                max_new_tokens=256,
                temperature=0.0,
            )

            predicted = self.extract_answer(response)
            ground_truth = question.get("answer", "")

            is_correct = predicted == ground_truth
            if is_correct:
                correct += 1

            predictions.append(
                {
                    "question": question.get("question", ""),
                    "predicted": predicted,
                    "ground_truth": ground_truth,
                    "correct": is_correct,
                }
            )

        total = len(questions)
        accuracy = (correct / total * 100) if total > 0 else 0.0

        return MMLUResult(
            correct=correct,
            total=total,
            accuracy=accuracy,
            predictions=predictions,
        )


def generate_html_report(result: EvaluationResult, output_path: Path) -> None:
    """Generate an HTML report from evaluation results.

    Args:
        result: Complete evaluation results to visualize.
        output_path: Path to write the HTML report.
    """
    judge_scores = result.judge_scores
    mmlu_result = result.mmlu_result
    metadata = result.metadata

    # Compute averages
    if judge_scores:
        avg_concept = sum(s.concept_score for s in judge_scores) / len(judge_scores)
        avg_fluency = sum(s.fluency_score for s in judge_scores) / len(judge_scores)
        avg_final = sum(s.final_score for s in judge_scores) / len(judge_scores)
    else:
        avg_concept = avg_fluency = avg_final = 0.0

    template = Template(HTML_TEMPLATE)
    html_content = template.render(
        metadata=metadata,
        judge_scores=judge_scores,
        avg_concept=avg_concept,
        avg_fluency=avg_fluency,
        avg_final=avg_final,
        mmlu_accuracy=mmlu_result.accuracy,
        mmlu_correct=mmlu_result.correct,
        mmlu_total=mmlu_result.total,
    )

    ensure_dir(output_path.parent)
    output_path.write_text(html_content)


# =============================================================================
# Steering Application Functions
# =============================================================================


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
    evaluate: bool = False,
    judge_model: str = "google/gemini-3.1-flash-lite-preview",
    mmlu_questions: int = 10,
) -> None:
    """Apply steering vector to model and save results as JSONL.

    This function:
    1. Loads the steering vector and extracts concept
    2. Loads contrast pairs and selects negative samples
    3. Normalizes steering vectors (norm=1)
    4. Computes average activation per layer
    5. For each layer, applies steering with each multiplier
    6. Saves results to JSONL files per layer
    7. Optionally runs evaluation (judge + MMLU)

    Args:
        vector_path: Path to saved steering vector (.pt file).
        model_name: HuggingFace model name.
        output_dir: Directory for output JSONL files.
        config: Steering configuration.
        evaluate: Whether to run evaluation on steered outputs.
        judge_model: Judge model for LLM-as-judge evaluation.
        mmlu_questions: Number of MMLU questions for evaluation.
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
    safe_model = safe_model_name(model_name)
    concept_dir = ensure_dir(output_dir / concept / safe_model)

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

        if evaluate and len(results) > 0:
            print(f"  Running evaluation for layer {layer_idx}...")
            judge_config = JudgeConfig(model=judge_model)
            mmlu_config = MMLUConfig(num_questions=mmlu_questions)

            judge = JudgeEvaluator(judge_config)

            for multiplier in config.multipliers:
                scale = avg_act * multiplier

                mult_results = [r for r in results if r["multiplier"] == multiplier]
                if not mult_results:
                    continue

                judge_scores = []
                for result in mult_results:
                    score = judge.evaluate_dual(concept, result["generated_text"])
                    judge_scores.append(score)

                mmlu = MMLUEvaluator(mmlu_config, model)
                mmlu_result = mmlu.evaluate(normalized_v, layer_idx, scale)

                metadata: EvaluationMetadata = {
                    "concept": concept,
                    "model": model_name,
                    "layer": layer_idx,
                    "multiplier": multiplier,
                }
                eval_result = EvaluationResult(
                    judge_scores=judge_scores,
                    mmlu_result=mmlu_result,
                    metadata=metadata,
                )

                eval_dir = ensure_dir(concept_dir / "eval")

                eval_json_path = eval_dir / f"layer{layer_idx}_mult{multiplier}.json"
                with eval_json_path.open("w") as f:
                    json.dump(
                        {
                            "judge_scores": [
                                {
                                    "concept": s.concept_score,
                                    "fluency": s.fluency_score,
                                    "final": s.final_score,
                                    "reasoning": s.reasoning,
                                }
                                for s in judge_scores
                            ],
                            "mmlu": {
                                "correct": mmlu_result.correct,
                                "total": mmlu_result.total,
                                "accuracy": mmlu_result.accuracy,
                            },
                            "metadata": metadata,
                        },
                        f,
                        indent=2,
                    )

                html_path = eval_dir / f"layer{layer_idx}_mult{multiplier}.html"
                generate_html_report(eval_result, html_path)
                print(f"    Evaluated mult {multiplier}: saved to {eval_dir}")

    print(f"\nDone! Results saved to {concept_dir}")


class _Args(Protocol):
    """Protocol defining CLI arguments for steering application."""

    vector: str
    model: str
    output: str
    samples: int
    multipliers: str
    max_new_tokens: int
    temperature: float
    evaluate: bool
    judge_model: str
    mmlu_questions: int


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for steering CLI."""
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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on steered outputs",
    )
    parser.add_argument(
        "--judge-model",
        default="google/gemini-3.1-flash-lite-preview",
        help="Judge model for LLM-as-judge evaluation",
    )
    parser.add_argument(
        "--mmlu-questions",
        type=int,
        default=10,
        help="Number of MMLU questions for evaluation",
    )
    return parser


def main() -> None:
    """CLI entry point for steering vector application."""
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
        evaluate=args.evaluate,
        judge_model=args.judge_model,
        mmlu_questions=args.mmlu_questions,
    )


if __name__ == "__main__":
    main()


__all__ = [
    # Steering application
    "apply_steering",
    "SteeringConfig",
    # Evaluation (merged from evaluation.py)
    "JudgeEvaluator",
    "MMLUEvaluator",
    "generate_html_report",
]
