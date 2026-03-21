"""Tests for evaluation module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from steering_geometry.apply_steering import (
    JudgeEvaluator,
    MMLUEvaluator,
    generate_html_report,
)
from steering_geometry.config import JudgeConfig, MMLUConfig
from steering_geometry.types import EvaluationResult, JudgeScore, MMLUResult


def _call_extract_score(evaluator: JudgeEvaluator, response: str) -> int:
    return evaluator._extract_score(response)


class TestExtractScore:
    """Tests for _extract_score function."""

    def test_rating_with_brackets(self) -> None:
        evaluator = JudgeEvaluator(JudgeConfig())
        result = _call_extract_score(evaluator, "The text is good. Rating: [[8]]")
        assert result == 8

    def test_rating_without_brackets(self) -> None:
        evaluator = JudgeEvaluator(JudgeConfig())
        result = _call_extract_score(evaluator, "Rating: 7")
        assert result == 7

    def test_fraction_format(self) -> None:
        evaluator = JudgeEvaluator(JudgeConfig())
        result = _call_extract_score(evaluator, "I give this 9/10")
        assert result == 9

    def test_clamp_high(self) -> None:
        evaluator = JudgeEvaluator(JudgeConfig())
        result = _call_extract_score(evaluator, "Rating: [[15]]")
        assert result == 10

    def test_clamp_low(self) -> None:
        evaluator = JudgeEvaluator(JudgeConfig())
        result = _call_extract_score(evaluator, "Rating: [[0]]")
        assert result == 0

    def test_standalone_digit(self) -> None:
        evaluator = JudgeEvaluator(JudgeConfig())
        result = _call_extract_score(evaluator, "The score is 6")
        assert result == 6

    def test_no_score_found(self) -> None:
        evaluator = JudgeEvaluator(JudgeConfig())
        result = _call_extract_score(evaluator, "No rating here")
        assert result == -1

    def test_case_insensitive(self) -> None:
        evaluator = JudgeEvaluator(JudgeConfig())
        result = _call_extract_score(evaluator, "rating: [[5]]")
        assert result == 5


class TestJudgeEvaluator:
    """Tests for JudgeEvaluator class."""

    def test_init_loads_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = JudgeConfig()
        evaluator = JudgeEvaluator(config)
        assert evaluator.client is not None

    def test_evaluate_concept_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = JudgeConfig()
        evaluator = JudgeEvaluator(config)

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Good concept. Rating: [[8]]"))
        ]

        with patch.object(
            evaluator.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = evaluator.evaluate_concept("honesty", "I always tell the truth.")

        assert isinstance(result, JudgeScore)
        assert result.concept_score == 8
        assert result.fluency_score == 10
        assert result.final_score == 8.0

    def test_evaluate_fluency_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = JudgeConfig()
        evaluator = JudgeEvaluator(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Very fluent. Rating: [[9]]"))]

        with patch.object(
            evaluator.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = evaluator.evaluate_fluency("This is a well-written text.")

        assert result == 9

    def test_evaluate_dual_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = JudgeConfig()
        evaluator = JudgeEvaluator(config)

        concept_response = MagicMock()
        concept_response.choices = [MagicMock(message=MagicMock(content="Good. Rating: [[8]]"))]

        fluency_response = MagicMock()
        fluency_response.choices = [MagicMock(message=MagicMock(content="Fluent. Rating: [[6]]"))]

        with patch.object(
            evaluator.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[concept_response, fluency_response],
        ):
            result = evaluator.evaluate_dual("honesty", "Test text.")

        assert result.concept_score == 8
        assert result.fluency_score == 6
        expected_harmonic_mean = 2.0 / (1.0 / 8 + 1.0 / 6)
        assert abs(result.final_score - expected_harmonic_mean) < 0.01

    def test_retry_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = JudgeConfig(max_retries=2)
        evaluator = JudgeEvaluator(config)

        call_count = 0

        async def failing_create(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            raise Exception("API error")

        with patch.object(
            evaluator.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=failing_create,
        ):
            with pytest.raises(Exception, match="API error"):
                evaluator.evaluate_fluency("Test text.")

            assert call_count == 2


class TestMMLUEvaluator:
    """Tests for MMLUEvaluator class."""

    def test_extract_answer_primary(self) -> None:
        evaluator = MMLUEvaluator(MMLUConfig(), MagicMock())
        result = evaluator.extract_answer("The answer is (B)")
        assert result == "B"

    def test_extract_answer_fallback(self) -> None:
        evaluator = MMLUEvaluator(MMLUConfig(), MagicMock())
        result = evaluator.extract_answer("Answer: C")
        assert result == "C"

    def test_extract_answer_last_resort(self) -> None:
        evaluator = MMLUEvaluator(MMLUConfig(), MagicMock())
        result = evaluator.extract_answer("After analysis, D seems correct.")
        assert result == "D"

    def test_extract_answer_none(self) -> None:
        evaluator = MMLUEvaluator(MMLUConfig(), MagicMock())
        result = evaluator.extract_answer("No answer here")
        assert result is None

    def test_format_prompt(self) -> None:
        evaluator = MMLUEvaluator(MMLUConfig(), MagicMock())
        question = {
            "question": "What is 2+2?",
            "options": ["3", "4", "5", "6"],
        }
        result = evaluator.format_prompt(question, use_cot=False)
        assert "Question: What is 2+2?" in result
        assert "A. 3" in result
        assert "B. 4" in result
        assert "Answer:" in result

    def test_format_prompt_with_cot(self) -> None:
        evaluator = MMLUEvaluator(MMLUConfig(), MagicMock())
        question = {
            "question": "What is 2+2?",
            "options": ["3", "4", "5", "6"],
        }
        result = evaluator.format_prompt(question, use_cot=True)
        assert "Let's think step by step." in result

    def test_evaluate_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_model = MagicMock()
        mock_model.generate_with_steering.return_value = "The answer is (B)"

        config = MMLUConfig(num_questions=2, seed=42)

        mock_dataset = [
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "answer": "B"},
            {"question": "Q2?", "options": ["A", "B", "C", "D"], "answer": "A"},
        ]

        with patch("steering_geometry.apply_steering.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = mock_dataset
            evaluator = MMLUEvaluator(config, mock_model)

            steering_vector = torch.randn(8)
            result = evaluator.evaluate(steering_vector, layer_idx=0, scale=1.0)

        assert isinstance(result, MMLUResult)
        assert result.total == 2
        assert result.correct == 1
        assert result.accuracy == 50.0


class TestGenerateHtmlReport:
    """Tests for generate_html_report function."""

    def test_generate_html_report(self, tmp_path: Path) -> None:
        judge_scores = [
            JudgeScore(
                concept_score=8,
                fluency_score=7,
                final_score=7.47,
                reasoning="Good",
            ),
            JudgeScore(
                concept_score=6,
                fluency_score=8,
                final_score=6.86,
                reasoning="OK",
            ),
        ]
        mmlu_result = MMLUResult(
            correct=7,
            total=10,
            accuracy=70.0,
            predictions=[],
        )
        metadata = {
            "concept": "honesty",
            "model": "test-model",
            "layer": 10,
            "multiplier": 1.0,
        }
        result = EvaluationResult(
            judge_scores=judge_scores,
            mmlu_result=mmlu_result,
            metadata=metadata,
        )

        output_path = tmp_path / "report.html"
        generate_html_report(result, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "<html" in content
        assert "Steering Evaluation Report" in content
        assert "honesty" in content
        assert "test-model" in content
        assert "70.00%" in content or "70%" in content

    def test_html_no_javascript(self, tmp_path: Path) -> None:
        judge_scores = [JudgeScore(concept_score=5, fluency_score=5, final_score=5.0, reasoning="")]
        mmlu_result = MMLUResult(correct=1, total=1, accuracy=100.0, predictions=[])
        result = EvaluationResult(
            judge_scores=judge_scores,
            mmlu_result=mmlu_result,
            metadata={"concept": "test", "model": "test", "layer": 0, "multiplier": 1.0},
        )

        output_path = tmp_path / "report.html"
        generate_html_report(result, output_path)

        content = output_path.read_text()
        assert "<script" not in content.lower()
