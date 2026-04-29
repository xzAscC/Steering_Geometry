"""Tests for config.py and CLI entry point (__main__.py)."""

import subprocess

import pytest

from steering_geometry.config import (
    DEFAULT_MODEL,
    SUPPORTED_CONCEPTS,
    SUPPORTED_MODELS,
    ModelConfig,
    StabilitySweepConfig,
)


def test_shell_flag_recognized() -> None:
    """Test that --shell flag is recognized and exits 0."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "steering_geometry", "--shell"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}"


def test_shell_output_format() -> None:
    """Test that --shell outputs valid bash variable format."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "steering_geometry", "--shell"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # Check bash array format
    assert "ALL_MODELS=(" in result.stdout
    assert "ALL_CONCEPTS=(" in result.stdout
    assert "DEFAULT_MODEL=" in result.stdout
    # Check that output ends with closing parens for arrays
    lines = result.stdout.strip().split("\n")
    assert any(")" in line for line in lines if "ALL_MODELS" in line)
    assert any(")" in line for line in lines if "ALL_CONCEPTS" in line)


def test_shell_output_contains_expected_values() -> None:
    """Test that --shell output contains expected config values."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "steering_geometry", "--shell"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Check DEFAULT_MODEL is present
    assert DEFAULT_MODEL in result.stdout

    # Check all models are present
    for model in SUPPORTED_MODELS:
        assert model in result.stdout, f"Model {model} not in output"

    # Check all concepts are present
    for concept in SUPPORTED_CONCEPTS:
        assert concept in result.stdout, f"Concept {concept} not in output"


def test_no_flag_shows_help_and_exits_1() -> None:
    """Test that no flag shows help and exits 1."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "steering_geometry"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, f"Expected exit 1, got {result.returncode}"
    # Help text should mention program name
    assert "steering_geometry" in result.stdout.lower() or "usage:" in result.stdout.lower()


def test_invalid_flag_exits_nonzero() -> None:
    """Test that invalid flags exit with non-zero code."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "steering_geometry", "--invalid-flag-xyz"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "Expected non-zero exit for invalid flag"


def test_new_models_in_supported_models() -> None:
    """New model IDs are present in SUPPORTED_MODELS."""
    assert "Qwen/Qwen3-14B" in SUPPORTED_MODELS
    assert "allenai/Olmo-3-1025-7B" in SUPPORTED_MODELS
    assert "allenai/Olmo-3-1125-32B" in SUPPORTED_MODELS


def test_trust_remote_code_for_allenai() -> None:
    """Allenai models auto-set trust_remote_code=True."""
    config = ModelConfig(model_name="allenai/Olmo-3-1125-32B")
    assert config.trust_remote_code is True

    config2 = ModelConfig(model_name="Qwen/Qwen3-1.7B")
    assert config2.trust_remote_code is False


def test_trust_remote_code_preserves_explicit_true() -> None:
    """Explicitly set trust_remote_code=True is preserved for non-allenai models."""
    config = ModelConfig(model_name="Qwen/Qwen3-1.7B", trust_remote_code=True)
    assert config.trust_remote_code is True


def test_stability_sweep_config_defaults() -> None:
    """StabilitySweepConfig creation with defaults."""
    config = StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="refusal")
    assert config.num_runs == 5
    assert config.n_values == [100, 500, 1000, 5000, 10000]
    assert len(config.layers) == 10
    assert config.display_concept == "Safety"


def test_stability_sweep_config_invalid_concept() -> None:
    """Invalid concept raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported concept"):
        StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="nonexistent")


def test_stability_sweep_config_invalid_model() -> None:
    """Invalid model raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model"):
        StabilitySweepConfig(model_name="fake/model", concept="refusal")


def test_stability_sweep_config_num_runs_too_low() -> None:
    """num_runs < 2 raises ValueError."""
    with pytest.raises(ValueError, match="num_runs must be at least 2"):
        StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="refusal", num_runs=1)


def test_stability_sweep_config_display_concept() -> None:
    """Concept display name mapping works for all supported concepts."""
    assert (
        StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="refusal").display_concept
        == "Safety"
    )
    assert (
        StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="polite").display_concept
        == "Politeness"
    )
    assert (
        StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="sentiment").display_concept
        == "Sentiment"
    )


def test_stability_sweep_config_paper_name_safety() -> None:
    """Paper name 'safety' resolves to canonical 'refusal'."""
    config = StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="safety")
    assert config.concept == "refusal"
    assert config.canonical_concept == "refusal"
    assert config.display_concept == "Safety"


def test_stability_sweep_config_paper_name_politeness() -> None:
    """Paper name 'politeness' resolves to canonical 'polite'."""
    config = StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="politeness")
    assert config.concept == "polite"
    assert config.canonical_concept == "polite"
    assert config.display_concept == "Politeness"
