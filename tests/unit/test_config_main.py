"""Tests for CLI entry point (__main__.py)."""

import subprocess

from steering_geometry.config import DEFAULT_MODEL, SUPPORTED_CONCEPTS, SUPPORTED_MODELS


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
