"""Tests for stability comparison experiment."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from steering_geometry.types import ContrastPair, SteeringVector


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


def _make_fake_pairs(n: int) -> list[ContrastPair]:
    """Create n fake contrast pairs for testing."""
    return [
        ContrastPair(
            positive=f"Positive example {i}",
            negative=f"Negative example {i}",
            metadata={"concept": "test", "id": i},
        )
        for i in range(n)
    ]


def _make_fake_steering_vector(num_layers: int = 5, dim: int = 8) -> SteeringVector:
    """Create a fake SteeringVector with random activations for each layer."""
    return SteeringVector(
        layer_activations={i: torch.randn(dim) for i in range(num_layers)},
        model_name="test",
        concept="test",
        method="discriminative",
    )


def test_candidate_pool_ablation_result_structure(tmp_path: Path) -> None:
    """Verify returned dict has vector_paths, heatmap_paths, statistics with correct types."""
    from steering_geometry.stability_comparison import run_candidate_pool_ablation

    fake_pairs = _make_fake_pairs(50)
    fake_vector = _make_fake_steering_vector()

    with (
        patch(
            "steering_geometry.stability_comparison.load_contrast_pairs", return_value=fake_pairs
        ),
        patch("steering_geometry.stability_comparison.HookedModel", return_value=MagicMock()),
        patch(
            "steering_geometry.stability_comparison.extract_steering_vector",
            return_value=fake_vector,
        ),
        patch("steering_geometry.stability_comparison.plot_heatmap"),
    ):
        result = run_candidate_pool_ablation(
            concept="test",
            pool_sizes=[10, 20],
            output_dir=tmp_path,
            num_trials=1,
        )

    assert "vector_paths" in result
    assert "heatmap_paths" in result
    assert "statistics" in result
    assert isinstance(result["vector_paths"], dict)
    assert isinstance(result["heatmap_paths"], dict)
    assert isinstance(result["statistics"], dict)
    assert all(isinstance(v, str) for v in result["vector_paths"].values())
    assert all(isinstance(v, str) for v in result["heatmap_paths"].values())
    for layer_stats in result["statistics"].values():
        assert isinstance(layer_stats, dict)
        assert all(isinstance(v, float) for v in layer_stats.values())


def test_candidate_pool_ablation_empty_pool_sizes_raises() -> None:
    """Empty pool_sizes list raises ValueError."""
    from steering_geometry.stability_comparison import run_candidate_pool_ablation

    with pytest.raises(ValueError, match="pool_sizes cannot be empty"):
        run_candidate_pool_ablation(concept="test", pool_sizes=[])


def test_candidate_pool_ablation_negative_pool_size_raises() -> None:
    """Negative pool_size raises ValueError."""
    from steering_geometry.stability_comparison import run_candidate_pool_ablation

    with pytest.raises(ValueError, match="All pool_sizes must be positive"):
        run_candidate_pool_ablation(concept="test", pool_sizes=[-1])


def test_candidate_pool_ablation_heatmap_dimensions(tmp_path: Path) -> None:
    """Heatmap matrix rows and cols match the number of pool_sizes."""
    from steering_geometry.stability_comparison import run_candidate_pool_ablation

    fake_pairs = _make_fake_pairs(50)
    fake_vector = _make_fake_steering_vector()
    mock_plot = MagicMock()

    with (
        patch(
            "steering_geometry.stability_comparison.load_contrast_pairs", return_value=fake_pairs
        ),
        patch("steering_geometry.stability_comparison.HookedModel", return_value=MagicMock()),
        patch(
            "steering_geometry.stability_comparison.extract_steering_vector",
            return_value=fake_vector,
        ),
        patch("steering_geometry.stability_comparison.plot_heatmap", mock_plot),
    ):
        run_candidate_pool_ablation(
            concept="test",
            pool_sizes=[10, 20, 30],
            output_dir=tmp_path,
            num_trials=1,
        )

    # plot_heatmap is called once per default layer (5 layers)
    assert mock_plot.call_count == 5
    # First positional arg of first call is the similarity matrix — should be (3, 3)
    first_matrix = mock_plot.call_args_list[0][0][0]
    assert first_matrix.shape == (3, 3)


def test_candidate_pool_ablation_statistics_keys(tmp_path: Path) -> None:
    """Statistics contain mean_similarity, min_similarity, max_similarity per layer."""
    from steering_geometry.stability_comparison import run_candidate_pool_ablation

    fake_pairs = _make_fake_pairs(50)
    fake_vector = _make_fake_steering_vector()

    with (
        patch(
            "steering_geometry.stability_comparison.load_contrast_pairs", return_value=fake_pairs
        ),
        patch("steering_geometry.stability_comparison.HookedModel", return_value=MagicMock()),
        patch(
            "steering_geometry.stability_comparison.extract_steering_vector",
            return_value=fake_vector,
        ),
        patch("steering_geometry.stability_comparison.plot_heatmap"),
    ):
        result = run_candidate_pool_ablation(
            concept="test",
            pool_sizes=[10, 20],
            output_dir=tmp_path,
            num_trials=1,
        )

    statistics = result["statistics"]
    assert len(statistics) == 5  # 5 default layers

    for layer_key, layer_stats in statistics.items():
        assert layer_key.startswith("layer"), f"Expected key to start with 'layer', got {layer_key}"
        assert "mean_similarity" in layer_stats
        assert "min_similarity" in layer_stats
        assert "max_similarity" in layer_stats
        assert -1.0 <= layer_stats["mean_similarity"] <= 1.0 + 1e-6
        assert -1.0 <= layer_stats["min_similarity"] <= 1.0 + 1e-6
        assert -1.0 <= layer_stats["max_similarity"] <= 1.0 + 1e-6
