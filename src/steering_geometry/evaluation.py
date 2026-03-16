"""Evaluation module for steering vector effectiveness.

This module provides tools for evaluating steering vector quality through
LLM-as-judge scoring and MMLU benchmark performance.

Components:
- JudgeEvaluator: Uses LLM judges to score steered responses on concept
  adherence and fluency
- MMLUEvaluator: Measures downstream task performance with steering applied
- generate_html_report: Creates visual reports from evaluation results

Usage:
    from steering_geometry.evaluation import JudgeEvaluator, MMLUEvaluator
    from steering_geometry.config import JudgeConfig, MMLUConfig

    judge = JudgeEvaluator(JudgeConfig())
    score = judge.evaluate_concept("honesty", "The sky is blue.")

    mmlu = MMLUEvaluator(MMLUConfig(), model)
    result = mmlu.evaluate(steering_vector, layer_idx=10, scale=1.0)
"""

import asyncio
import os
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from jinja2 import Template
from openai import AsyncOpenAI
from torch import Tensor

from .config import JudgeConfig, MMLUConfig
from .types import EvaluationResult, JudgeScore, MMLUPrediction, MMLUQuestion, MMLUResult
from .utils import clamp_score, ensure_dir

# Judge prompt templates
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

# HTML report template
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


__all__ = ["JudgeEvaluator", "MMLUEvaluator", "generate_html_report"]
