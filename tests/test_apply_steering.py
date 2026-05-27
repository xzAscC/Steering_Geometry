"""Tests for apply_steering module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from steering_geometry.config import SteeringConfig
from steering_geometry.types import ContrastPair, SteeringVector


def test_steering_config_defaults() -> None:
    config = SteeringConfig()
    assert config.multipliers == [0.01, 0.1, 1.0, 10.0]
    assert config.num_samples == 10
    assert config.seed == 42
    assert config.max_new_tokens == 100
    assert config.temperature == 0.0


def test_steering_config_custom() -> None:
    config = SteeringConfig(
        multipliers=[0.5, 1.0],
        num_samples=5,
        max_new_tokens=50,
        temperature=0.7,
    )
    assert config.multipliers == [0.5, 1.0]
    assert config.num_samples == 5
    assert config.max_new_tokens == 50
    assert config.temperature == 0.7


@pytest.fixture
def sample_contrast_pairs() -> list[ContrastPair]:
    return [
        ContrastPair(
            positive="I always provide facts.",
            negative="I might make up answers.",
            metadata={"concept": "test_concept", "id": 1},
        ),
        ContrastPair(
            positive="Evidence suggests this is true.",
            negative="I believe this without evidence.",
            metadata={"concept": "test_concept", "id": 2},
        ),
        ContrastPair(
            positive="The data does not support that claim.",
            negative="That claim is definitely true.",
            metadata={"concept": "test_concept", "id": 3},
        ),
    ]


@pytest.fixture
def sample_steering_vector() -> SteeringVector:
    return SteeringVector(
        layer_activations={
            0: torch.randn(8),
            1: torch.randn(8),
        },
        model_name="test-model",
        concept="test_concept",
        method="mean",
    )


class TestNormalizeVectors:
    def test_normalize_vectors_unit_norm(self) -> None:
        from steering_geometry.apply_steering import _normalize_vectors

        vector = SteeringVector(
            layer_activations={
                0: torch.tensor([3.0, 4.0]),
            },
            model_name="test",
            concept="test",
            method="mean",
        )

        normalized = _normalize_vectors(vector)
        assert 0 in normalized
        assert torch.allclose(normalized[0], torch.tensor([0.6, 0.8]))
        assert abs(normalized[0].norm().item() - 1.0) < 1e-5

    def test_normalize_vectors_zero_norm(self) -> None:
        from steering_geometry.apply_steering import _normalize_vectors

        vector = SteeringVector(
            layer_activations={
                0: torch.zeros(4),
            },
            model_name="test",
            concept="test",
            method="mean",
        )

        normalized = _normalize_vectors(vector)
        assert torch.allclose(normalized[0], torch.zeros(4))


class TestSafeModelName:
    def test_safe_model_name(self) -> None:
        from steering_geometry.utils import safe_model_name

        assert safe_model_name("Qwen/Qwen3-1.7B") == "Qwen_Qwen3-1.7B"
        assert safe_model_name("allenai/Olmo-3-1025-7B") == "allenai_Olmo-3-1025-7B"
        assert safe_model_name("simple-name") == "simple-name"


class TestApplySteering:
    def test_apply_steering_creates_output(
        self,
        tmp_path: Path,
        sample_contrast_pairs: list[ContrastPair],
        sample_steering_vector: SteeringVector,
    ) -> None:
        vector_file = tmp_path / "test_vector.pt"
        torch.save({"vector": sample_steering_vector}, vector_file)

        output_dir = tmp_path / "steered"

        with (
            patch("steering_geometry.apply_steering.load_contrast_pairs") as mock_load_pairs,
            patch("steering_geometry.apply_steering.HookedModel") as mock_model_class,
        ):
            mock_load_pairs.return_value = sample_contrast_pairs

            mock_model = MagicMock()
            mock_model.num_layers = 4
            mock_model.model.config.hidden_size = 8
            mock_model.get_activations.return_value = {
                0: torch.randn(2, 10, 8),
                1: torch.randn(2, 10, 8),
            }
            mock_model.generate_with_steering.return_value = "Generated text"
            mock_model_class.return_value = mock_model

            from steering_geometry.apply_steering import apply_steering

            config = SteeringConfig(num_samples=2, multipliers=[0.1, 1.0])
            apply_steering(
                vector_path=vector_file,
                model_name="test-model",
                output_dir=output_dir,
                config=config,
            )

        concept_dir = output_dir / "test_concept" / "test-model"
        assert concept_dir.exists()

        layer_files = list(concept_dir.glob("layer*.jsonl"))
        assert len(layer_files) == 2

        for layer_file in layer_files:
            with layer_file.open() as f:
                lines = f.readlines()
            assert len(lines) == 4

            for line in lines:
                data = json.loads(line)
                assert "sample_idx" in data
                assert "multiplier" in data
                assert "prompt" in data
                assert "generated_text" in data

    def test_apply_steering_jsonl_format(
        self,
        tmp_path: Path,
        sample_contrast_pairs: list[ContrastPair],
        sample_steering_vector: SteeringVector,
    ) -> None:
        vector_file = tmp_path / "test_vector.pt"
        torch.save({"vector": sample_steering_vector}, vector_file)

        output_dir = tmp_path / "steered"

        with (
            patch("steering_geometry.apply_steering.load_contrast_pairs") as mock_load_pairs,
            patch("steering_geometry.apply_steering.HookedModel") as mock_model_class,
        ):
            mock_load_pairs.return_value = sample_contrast_pairs

            mock_model = MagicMock()
            mock_model.num_layers = 4
            mock_model.get_activations.return_value = {
                0: torch.randn(2, 10, 8),
                1: torch.randn(2, 10, 8),
            }
            mock_model.generate_with_steering.return_value = "Test output"
            mock_model_class.return_value = mock_model

            from steering_geometry.apply_steering import apply_steering

            config = SteeringConfig(num_samples=1, multipliers=[1.0])
            apply_steering(
                vector_path=vector_file,
                model_name="test-model",
                output_dir=output_dir,
                config=config,
            )

        concept_dir = output_dir / "test_concept" / "test-model"
        layer_file = concept_dir / "layer0.jsonl"

        with layer_file.open() as f:
            line = f.readline()
            data = json.loads(line)

        assert isinstance(data["sample_idx"], int)
        assert isinstance(data["multiplier"], float)
        assert isinstance(data["prompt"], str)
        assert isinstance(data["generated_text"], str)
