"""Tests for stability comparison experiment."""

import pytest
import torch


def test_select_token_subsets_returns_different_subsets():
    """Test that different random seeds produce different token subsets."""
    from steering_geometry.extract import load_contrast_pairs
    from steering_geometry.stability_comparison import select_token_subsets

    pairs = load_contrast_pairs("sentiment", 100)
    subsets = select_token_subsets(pairs, num_tokens=30, num_runs=3)

    assert len(subsets) == 3, "Should return 3 subsets"
    assert all(len(s) == 30 for s in subsets), "Each subset should have 30 pairs"
    assert subsets[0][0] != subsets[1][0] or subsets[0][0] != subsets[2][0]


@pytest.mark.skip(reason="Requires model download - run manually with model")
def test_run_single_extraction_returns_vector():
    """Test that single extraction returns a tensor for the specified layer."""


@pytest.mark.skip(reason="Requires model download - run manually with model")
def test_run_stability_comparison_returns_results():
    """Test that main function returns dict with correct structure."""


def test_compute_stability_statistics():
    """Test statistics computation for cosine similarity."""
    from steering_geometry.stability_comparison import compute_stability_statistics

    vectors = [torch.randn(100) for _ in range(3)]
    stats = compute_stability_statistics(vectors)

    assert "mean" in stats
    assert "min" in stats
    assert "max" in stats
    assert "std" in stats
    assert -1 <= stats["mean"] <= 1, "Cosine similarity mean should be in [-1, 1]"
