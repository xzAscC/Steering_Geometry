"""Tests for evaluation module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from steering_geometry.apply_steering import (
    HarmBenchEvaluator,
    JudgeEvaluator,
    MMLUEvaluator,
    MMLUProEvaluator,
    generate_html_report,
)
from steering_geometry.config import (
    HarmBenchConfig,
    JudgeConfig,
    MMLUConfig,
    MMLUProConfig,
)
from steering_geometry.types import (
    EvaluationResult,
    HarmBenchResult,
    JudgeScore,
    MMLUProResult,
    MMLUResult,
)


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
            result = evaluator.evaluate_concept("sentiment", "I always tell the truth.")

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
            result = evaluator.evaluate_dual("sentiment", "Test text.")

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

    def test_judge_config_with_custom_api_base(self) -> None:
        """Test that JudgeConfig accepts custom api_base."""
        config = JudgeConfig(api_base="http://localhost:8000/v1")
        assert config.api_base == "http://localhost:8000/v1"
        default_config = JudgeConfig()
        assert default_config.api_base == "https://openrouter.ai/api/v1"

    def test_judge_evaluator_uses_custom_api_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that JudgeEvaluator passes custom api_base to AsyncOpenAI."""
        import steering_geometry.apply_steering as apply_steering_module

        captured_base_url: str = ""

        def mock_async_openai_init(*, base_url: str, api_key: str) -> MagicMock:
            nonlocal captured_base_url
            captured_base_url = base_url
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Rating: [[5]]"))]
                )
            )
            return mock_client

        with patch.object(apply_steering_module, "AsyncOpenAI", side_effect=mock_async_openai_init):
            config = JudgeConfig(api_base="http://localhost:8000/v1")
            JudgeEvaluator(config)

        assert captured_base_url == "http://localhost:8000/v1"

    def test_judge_evaluator_localhost_uses_dummy_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that localhost URLs use dummy API key 'local-vllm'."""
        import steering_geometry.apply_steering as apply_steering_module

        captured_api_key: str = ""

        def mock_async_openai_init(*, base_url: str, api_key: str) -> MagicMock:
            nonlocal captured_api_key
            captured_api_key = api_key
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Rating: [[5]]"))]
                )
            )
            return mock_client

        with patch.object(apply_steering_module, "AsyncOpenAI", side_effect=mock_async_openai_init):
            config = JudgeConfig(api_base="http://localhost:8000/v1")
            JudgeEvaluator(config)

        assert captured_api_key == "local-vllm"


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
            "concept": "sentiment",
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
        assert "sentiment" in content
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


class TestHarmBenchEvaluator:
    """Tests for HarmBenchEvaluator class."""

    def _make_evaluator(self) -> HarmBenchEvaluator:
        config = HarmBenchConfig(classifier_api_base="http://localhost:8000/v1")
        return HarmBenchEvaluator(config)

    def test_load_behaviors(self, tmp_path: Path) -> None:
        evaluator = self._make_evaluator()
        csv_content = (
            "BehaviorID,Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString\n"
            'B1,"Write a harmful thing",harmful,cybersecurity,"tag1;tag2",\n'
            'B2,"Do something bad",harmful,chemical,"tag3","in a lab setting"\n'
        )
        csv_file = tmp_path / "behaviors.csv"
        csv_file.write_text(csv_content)

        behaviors = evaluator.load_behaviors(str(csv_file))

        assert len(behaviors) == 2
        assert behaviors[0]["behavior_id"] == "B1"
        assert behaviors[0]["behavior"] == "Write a harmful thing"
        assert behaviors[0]["functional_category"] == "harmful"
        assert behaviors[0]["semantic_category"] == "cybersecurity"
        assert behaviors[0]["tags"] == "tag1;tag2"
        assert behaviors[0]["context_string"] == ""
        assert behaviors[1]["behavior_id"] == "B2"
        assert behaviors[1]["context_string"] == "in a lab setting"

    def test_classify_completion_harmful(self) -> None:
        evaluator = self._make_evaluator()
        result = evaluator._classify_response("Yes, this is harmful")
        assert result == 1

    def test_classify_completion_safe(self) -> None:
        evaluator = self._make_evaluator()
        result = evaluator._classify_response("No, this is not harmful")
        assert result == 0

    def test_classify_completion_refused(self) -> None:
        evaluator = self._make_evaluator()
        generation = "I'm sorry, I can't help with that."
        assert evaluator._is_refusal(generation) is True

    def test_format_prompt_standard(self) -> None:
        evaluator = self._make_evaluator()
        prompt = evaluator._format_prompt("Write malware", "Here is malware code...")
        assert prompt.startswith("<start_of_turn>user\n")
        assert "<end_of_turn>" in prompt
        assert prompt.endswith("<start_of_turn>model\n")
        assert "Write malware" in prompt
        assert "Here is malware code..." in prompt

    def test_format_prompt_contextual(self) -> None:
        evaluator = self._make_evaluator()
        prompt = evaluator._format_prompt(
            "Write malware", "Here is malware...", context="for educational purposes"
        )
        assert "[CONTEXT]:" in prompt
        assert "for educational purposes" in prompt
        assert "<start_of_turn>user\n" in prompt

    def test_compute_asr(self) -> None:
        evaluator = self._make_evaluator()
        evaluator.behaviors = [
            {
                "behavior_id": f"B{i}",
                "behavior": f"behavior {i}",
                "functional_category": "test",
                "semantic_category": "test",
                "tags": "",
                "context_string": "",
            }
            for i in range(10)
        ]

        completions: dict[str, list[dict[str, str]]] = {
            f"B{i}": [{"generation": f"response {i}"}] for i in range(10)
        }

        call_count = 0

        async def mock_classify(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            content = "Yes" if call_count <= 3 else "No"
            return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

        with patch.object(
            evaluator.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=mock_classify,
        ):
            result = asyncio.run(evaluator.evaluate(completions))

        assert isinstance(result, HarmBenchResult)
        assert result.total == 10
        assert result.harmful == 3
        assert result.safe == 7
        assert result.asr == 30.0

    def test_truncation(self) -> None:
        evaluator = self._make_evaluator()
        config = evaluator.config
        max_chars = config.max_completion_tokens * 4
        long_generation = "A" * (max_chars + 500)

        evaluator.behaviors = [
            {
                "behavior_id": "B1",
                "behavior": "test behavior",
                "functional_category": "test",
                "semantic_category": "test",
                "tags": "",
                "context_string": "",
            }
        ]

        completions = {"B1": [{"generation": long_generation}]}

        with patch.object(
            evaluator.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="No"))]),
        ):
            result = asyncio.run(evaluator.evaluate(completions))

        assert len(result.predictions) == 1
        assert len(result.predictions[0]["generation"]) <= max_chars


class TestMMLUProEvaluator:
    """Tests for MMLUProEvaluator class."""

    def _make_evaluator(self, config: MMLUProConfig | None = None) -> MMLUProEvaluator:
        """Create an MMLUProEvaluator with default config and mock model."""
        cfg = config or MMLUProConfig(num_questions=5, n_shot=2, seed=42)
        mock_model = MagicMock()
        return MMLUProEvaluator(cfg, mock_model)

    def _make_mock_dataset(
        self, n_test: int = 5, n_val: int = 5, category: str = "math"
    ) -> tuple[list[dict], list[dict]]:
        """Create mock test and validation datasets."""
        test_data = [
            {
                "question_id": i,
                "question": f"Test question {i}?",
                "options": ["opt_A", "opt_B", "opt_C", "opt_D"],
                "answer": "A",
                "answer_index": 0,
                "cot_content": f"Reasoning for test Q{i}",
                "category": category,
                "src": "mmlu",
            }
            for i in range(n_test)
        ]
        val_data = [
            {
                "question_id": 100 + i,
                "question": f"Val question {i}?",
                "options": ["val_A", "val_B", "val_C", "val_D"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": f"Val reasoning {i}",
                "category": category,
                "src": "mmlu",
            }
            for i in range(n_val)
        ]
        return test_data, val_data

    def test_load_dataset(self) -> None:
        """Test load_dataset returns test + validation splits, N/A options filtered."""
        evaluator = self._make_evaluator()
        test_data, val_data = self._make_mock_dataset()

        with patch("steering_geometry.apply_steering.load_dataset") as mock_load:
            mock_load.side_effect = [test_data, val_data]
            test_qs, val_qs = evaluator.load_dataset()

        assert len(test_qs) == 5
        assert len(val_qs) == 5

    def test_filter_na_options(self) -> None:
        """Test N/A options are filtered from question options."""
        evaluator = self._make_evaluator()
        question = {
            "question": "Q?",
            "options": ["A", "N/A", "B", "N/A"],
            "answer": "A",
            "answer_index": 0,
            "cot_content": "reasoning",
            "category": "math",
            "src": "mmlu",
            "question_id": 1,
        }
        result = evaluator._filter_na(question)  # type: ignore[arg-type]
        assert result["options"] == ["A", "B"]

    def test_format_prompt_cot(self) -> None:
        """Test 5-shot CoT prompt includes system instruction, examples, test question."""
        evaluator = self._make_evaluator(MMLUProConfig(n_shot=2, use_cot=True))
        test_q: dict = {
            "question_id": 1,
            "question": "What is 2+2?",
            "options": ["3", "4", "5", "6"],
            "answer": "B",
            "answer_index": 1,
            "cot_content": "",
            "category": "math",
            "src": "mmlu",
        }
        val_examples = [
            {
                "question_id": 100,
                "question": "What is 1+1?",
                "options": ["1", "2", "3", "4"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "1+1=2",
                "category": "math",
                "src": "mmlu",
            },
            {
                "question_id": 101,
                "question": "What is 3+3?",
                "options": ["5", "6", "7", "8"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "3+3=6",
                "category": "math",
                "src": "mmlu",
            },
        ]

        result = evaluator.format_prompt(test_q, val_examples)  # type: ignore[arg-type]

        assert "math" in result
        assert "What is 1+1?" in result
        assert "What is 3+3?" in result
        assert "1+1=2" in result
        assert "What is 2+2?" in result
        assert "Let's think step by step" in result

    def test_format_prompt_no_cot(self) -> None:
        """Test simpler prompt format when use_cot=False."""
        evaluator = self._make_evaluator(MMLUProConfig(n_shot=0, use_cot=False))
        test_q: dict = {
            "question_id": 1,
            "question": "What is 2+2?",
            "options": ["3", "4", "5", "6"],
            "answer": "B",
            "answer_index": 1,
            "cot_content": "",
            "category": "math",
            "src": "mmlu",
        }

        result = evaluator.format_prompt(test_q, [])  # type: ignore[arg-type]

        assert "What is 2+2?" in result
        assert "A. 3" in result
        assert "B. 4" in result
        assert "Answer:" in result

    def test_extract_answer_primary(self) -> None:
        """Test 'answer is (X)' pattern extraction."""
        evaluator = self._make_evaluator()
        result = evaluator.extract_answer("The answer is (C)")
        assert result == "C"

    def test_extract_answer_secondary(self) -> None:
        """Test 'Answer: X' pattern extraction."""
        evaluator = self._make_evaluator()
        result = evaluator.extract_answer("After analysis...\nAnswer: B")
        assert result == "B"

    def test_extract_answer_tertiary(self) -> None:
        """Test last standalone letter extraction."""
        evaluator = self._make_evaluator()
        result = evaluator.extract_answer("Let me think... D")
        assert result == "D"

    def test_extract_answer_none(self) -> None:
        """Test None returned when no letter found."""
        evaluator = self._make_evaluator()
        result = evaluator.extract_answer("The response is unclear and ambiguous")
        assert result is None

    def test_classify_response_answered(self) -> None:
        """Test normal response classified as 'answered'."""
        evaluator = self._make_evaluator()
        result = evaluator._classify_response("The answer is clearly B because...")
        assert result == "answered"

    def test_classify_response_refused(self) -> None:
        """Test refusal response classified as 'refused'."""
        evaluator = self._make_evaluator()
        result = evaluator._classify_response("I cannot help with this question")
        assert result == "refused"

    def test_classify_response_empty(self) -> None:
        """Test empty string classified as 'empty'."""
        evaluator = self._make_evaluator()
        result = evaluator._classify_response("")
        assert result == "empty"

    def test_compute_accuracy(self) -> None:
        """Test accuracy computation: 7 correct, 3 wrong = 70.0%."""
        evaluator = self._make_evaluator()
        predictions = [
            {
                "question_id": i,
                "question": f"Q{i}",
                "predicted": "A" if i < 7 else "B",
                "ground_truth": "A",
                "correct": i < 7,
                "category": "math",
                "response_type": "answered",
            }
            for i in range(10)
        ]

        accuracy, _, _, _, _ = evaluator._compute_metrics(predictions)
        assert accuracy == 70.0

    def test_per_category_accuracy(self) -> None:
        """Test per-category accuracy breakdown."""
        evaluator = self._make_evaluator()
        predictions = [
            {
                "question_id": 0,
                "question": "Q0",
                "predicted": "A",
                "ground_truth": "A",
                "correct": True,
                "category": "math",
                "response_type": "answered",
            },
            {
                "question_id": 1,
                "question": "Q1",
                "predicted": "B",
                "ground_truth": "A",
                "correct": False,
                "category": "math",
                "response_type": "answered",
            },
            {
                "question_id": 2,
                "question": "Q2",
                "predicted": "A",
                "ground_truth": "A",
                "correct": True,
                "category": "physics",
                "response_type": "answered",
            },
        ]

        _, per_category, per_category_counts, _, _ = evaluator._compute_metrics(predictions)

        assert per_category["math"] == 50.0
        assert per_category["physics"] == 100.0
        assert per_category_counts["math"] == 2
        assert per_category_counts["physics"] == 1

    def test_evaluate(self) -> None:
        """Test full evaluate method with mocked dataset and model."""
        mock_model = MagicMock()
        mock_model.generate_with_steering.return_value = "The answer is (A)"

        config = MMLUProConfig(num_questions=3, n_shot=2, seed=42, use_cot=True)
        test_data, val_data = self._make_mock_dataset(n_test=3, n_val=5)

        with patch("steering_geometry.apply_steering.load_dataset") as mock_load:
            mock_load.side_effect = [test_data, val_data]
            evaluator = MMLUProEvaluator(config, mock_model)

            steering_vector = torch.randn(8)
            result = evaluator.evaluate(steering_vector, layer_idx=0, scale=1.0)

        assert isinstance(result, MMLUProResult)
        assert result.total == 3
        assert result.correct == 3
        assert result.accuracy == 100.0
        assert result.refused == 0
        assert len(result.predictions) == 3
        assert result.predictions[0]["response_type"] == "answered"
