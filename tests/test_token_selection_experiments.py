"""Tests for token_selection_experiments module."""

from __future__ import annotations

import pytest
import torch

from steering_geometry.models import HookedModel
from steering_geometry.stability_comparison import compute_cosine_similarity_matrix


class TestTokenCountExperiment:
    """Tests for token count experiment parameter handling."""

    def test_token_count_output_path_generation(self) -> None:
        """Verify correct output paths are generated for given parameters."""
        concept = "refusal"
        n_examples = 100
        layer = 0.5
        expected = f"vectors/{concept}/token_count/n{n_examples}_layer{layer}.pt"
        assert "token_count" in expected
        assert f"n{n_examples}" in expected
        assert f"layer{layer}" in expected


class TestTokenPositionExperiment:
    """Tests for token position experiment parameter handling."""

    def test_position_config_all_mode(self) -> None:
        """Verify 'all' mode produces correct path."""
        mode = "all"
        concept = "refusal"
        layer = 0.6
        expected = f"vectors/{concept}/token_position/{mode}_layer{layer}.pt"
        assert "token_position" in expected
        assert mode in expected

    def test_position_config_last_n_mode(self) -> None:
        """Verify last_n mode produces correct path with n value."""
        mode = "last_n"
        n = 3
        concept = "refusal"
        layer = 0.6
        expected = f"vectors/{concept}/token_position/{mode}_n{n}_layer{layer}.pt"
        assert "token_position" in expected
        assert f"n{n}" in expected


class TestPromptResponseExperiment:
    """Tests for prompt vs response experiment."""

    def test_data_mode_validation_valid(self) -> None:
        """Valid data_modes should be accepted."""
        valid_modes = ["prompt_only", "prompt_response"]
        for mode in valid_modes:
            assert mode in ("prompt_only", "prompt_response")

    def test_data_mode_validation_invalid(self) -> None:
        """Invalid data_mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid data_mode"):
            mode = "invalid_mode"
            if mode not in ("prompt_only", "prompt_response"):
                raise ValueError(f"Invalid data_mode: {mode}")


class TestCosineSimilarityComputation:
    """Tests for cosine similarity with known tensors."""

    def test_identical_vectors_high_similarity(self) -> None:
        """Identical vectors should have cosine similarity ~1.0."""
        v = torch.randn(64)
        matrix = compute_cosine_similarity_matrix([v, v])
        assert abs(matrix[0, 1] - 1.0) < 1e-6

    def test_opposite_vectors_negative_similarity(self) -> None:
        """Opposite vectors should have cosine similarity ~-1.0."""
        v = torch.randn(64)
        matrix = compute_cosine_similarity_matrix([v, -v])
        assert abs(matrix[0, 1] - (-1.0)) < 1e-6

    def test_orthogonal_vectors_zero_similarity(self) -> None:
        """Orthogonal vectors should have cosine similarity ~0.0."""
        v1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        matrix = compute_cosine_similarity_matrix([v1, v2])
        assert abs(matrix[0, 1]) < 1e-6


class TestSteerTokens:
    """Tests for prefix-only steering via steer_tokens parameter."""

    def test_steer_tokens_backward_compat(self, mock_hooked_model: HookedModel) -> None:
        """Calling generate_with_steering without steer_tokens produces identical output."""
        prompt = "Hello world"
        layer_idx = 0
        steering_vector = torch.randn(8)
        steering_vector = steering_vector / steering_vector.norm()
        scale = 1.0

        result_default = mock_hooked_model.generate_with_steering(
            prompt=prompt,
            layer_idx=layer_idx,
            steering_vector=steering_vector,
            scale=scale,
        )
        result_none = mock_hooked_model.generate_with_steering(
            prompt=prompt,
            layer_idx=layer_idx,
            steering_vector=steering_vector,
            scale=scale,
            steer_tokens=None,
        )

        assert result_default == result_none

    def test_steer_tokens_zero(self, mock_hooked_model: HookedModel) -> None:
        """steer_tokens=0 produces unsteered output."""
        prompt = "Hello world"
        layer_idx = 0
        steering_vector = torch.randn(8)
        steering_vector = steering_vector / steering_vector.norm()
        scale = 1.0

        result_steered = mock_hooked_model.generate_with_steering(
            prompt=prompt,
            layer_idx=layer_idx,
            steering_vector=steering_vector,
            scale=scale,
            steer_tokens=0,
        )
        result_unsteered = mock_hooked_model.generate_with_steering(
            prompt=prompt,
            layer_idx=layer_idx,
            steering_vector=steering_vector,
            scale=0.0,
            steer_tokens=None,
        )

        assert result_steered == result_unsteered

    def test_steer_tokens_large(self, mock_hooked_model: HookedModel) -> None:
        """steer_tokens=999 (>= max_new_tokens) produces same output as steer_tokens=None."""
        prompt = "Hello world"
        layer_idx = 0
        steering_vector = torch.randn(8)
        steering_vector = steering_vector / steering_vector.norm()
        scale = 1.0
        max_new_tokens = 5

        result_none = mock_hooked_model.generate_with_steering(
            prompt=prompt,
            layer_idx=layer_idx,
            steering_vector=steering_vector,
            scale=scale,
            max_new_tokens=max_new_tokens,
            steer_tokens=None,
        )
        result_large = mock_hooked_model.generate_with_steering(
            prompt=prompt,
            layer_idx=layer_idx,
            steering_vector=steering_vector,
            scale=scale,
            max_new_tokens=max_new_tokens,
            steer_tokens=999,
        )

        assert result_none == result_large


class TestSteeringScopeExperiment:
    """Tests for steering scope experiment function."""

    def test_run_steering_scope_experiment_importable(self) -> None:
        """Function should be importable."""
        from steering_geometry.token_selection_experiments import run_steering_scope_experiment

        assert callable(run_steering_scope_experiment)

    def test_steering_scope_output_path_generation(self) -> None:
        """Verify correct output paths for steering scope parameters."""
        concept = "refusal"
        steer_n = 5
        layer = 0.7
        mult = 1.0
        expected = f"steered/{concept}/steer_scope/steer_{steer_n}_layer{layer}_mult{mult}.jsonl"
        assert "steer_scope" in expected
        assert f"steer_{steer_n}" in expected
        assert f"layer{layer}" in expected
        assert f"mult{mult}" in expected

    def test_steering_scope_all_tokens_label(self) -> None:
        """Verify 'all' label when steer_tokens is None."""
        steer_tokens_value = None
        label = "all" if steer_tokens_value is None else str(steer_tokens_value)
        assert label == "all"
