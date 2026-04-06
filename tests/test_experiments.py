"""Tests for experiments module."""

from pathlib import Path
from typing import cast

import pytest
import torch
from numpy import ndarray

from steering_geometry.stability_comparison import (
    cap_examples,
    compute_cosine_similarity_matrix,
    load_vector,
    plot_heatmap,
    run_diff_means_experiment,
    save_vector,
)


class TestCosineSimilarityComputation:
    """Tests for compute_cosine_similarity_matrix function."""

    def test_identity_matrix_same_vectors(self) -> None:
        """Identical vectors should produce identity-like matrix (all 1s)."""
        torch.manual_seed(42)
        vector = torch.randn(64)
        vectors = [vector, vector.clone(), vector.clone()]

        result = compute_cosine_similarity_matrix(vectors)

        assert isinstance(result, ndarray)
        assert result.shape == (3, 3)
        # All entries should be ~1.0 for identical vectors
        assert (result > 0.9999).all()

    def test_orthogonal_vectors_zero_similarity(self) -> None:
        """Orthogonal vectors should have ~0 cosine similarity."""
        # Create orthogonal vectors
        v1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        v3 = torch.tensor([0.0, 0.0, 1.0, 0.0])

        result = compute_cosine_similarity_matrix([v1, v2, v3])

        assert result.shape == (3, 3)
        # Diagonal should be 1.0
        for i in range(3):
            assert abs(result[i, i] - 1.0) < 0.0001
        # Off-diagonal should be ~0 for orthogonal vectors
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(result[i, j]) < 0.0001

    def test_known_similarity(self) -> None:
        """Test with vectors of known similarity."""
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([1.0, 1.0])  # 45 degrees -> similarity = sqrt(2)/2

        result = compute_cosine_similarity_matrix([v1, v2])

        expected = 1.0 / (2**0.5)  # cos(45 deg) = sqrt(2)/2
        assert abs(result[0, 1] - expected) < 0.0001
        assert abs(result[1, 0] - expected) < 0.0001

    def test_random_vectors_symmetric_matrix(self) -> None:
        """Cosine similarity matrix should be symmetric."""
        torch.manual_seed(123)
        vectors = [torch.randn(32) for _ in range(5)]

        result = compute_cosine_similarity_matrix(vectors)

        assert result.shape == (5, 5)
        # Matrix should be symmetric
        for i in range(5):
            for j in range(5):
                assert abs(result[i, j] - result[j, i]) < 0.0001

    def test_single_vector(self) -> None:
        """Single vector should produce 1x1 matrix with value 1.0."""
        vector = torch.randn(16)
        result = compute_cosine_similarity_matrix([vector])

        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 1.0) < 0.0001

    def test_empty_vectors_raises(self) -> None:
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty vector list"):
            compute_cosine_similarity_matrix([])


class TestHeatmapGeneration:
    """Tests for plot_heatmap function."""

    def test_heatmap_creates_pdf(self, tmp_path: Path) -> None:
        """Heatmap should create a valid PDF file."""
        import numpy as np

        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        labels = ["Vector_A", "Vector_B"]
        output_path = tmp_path / "test_heatmap.pdf"

        result = plot_heatmap(matrix, labels, "Test Heatmap", output_path)

        assert result == output_path
        assert output_path.exists()
        # Check it's a valid PDF by reading header
        content = output_path.read_bytes()
        assert content.startswith(b"%PDF")

    def test_heatmap_larger_matrix(self, tmp_path: Path) -> None:
        """Heatmap should handle larger matrices."""
        import numpy as np

        matrix = np.random.rand(5, 5)
        labels = ["A", "B", "C", "D", "E"]
        output_path = tmp_path / "larger_heatmap.pdf"

        result = plot_heatmap(matrix, labels, "Larger Matrix", output_path)

        assert result.exists()
        assert result.stat().st_size > 0

    def test_heatmap_mismatched_labels_raises(self, tmp_path: Path) -> None:
        """Mismatched labels length should raise ValueError."""
        import numpy as np

        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        labels = ["Only_One_Label"]
        output_path = tmp_path / "bad_heatmap.pdf"

        with pytest.raises(ValueError, match="Labels length"):
            plot_heatmap(matrix, labels, "Bad Heatmap", output_path)


class TestVectorSaveLoad:
    """Tests for save_vector and load_vector functions."""

    def test_roundtrip_preserves_tensor(self, tmp_path: Path) -> None:
        """Save and load should preserve tensor values exactly."""
        torch.manual_seed(42)
        original = torch.randn(128)

        path = tmp_path / "test_vector.pt"
        save_vector(original, path)
        loaded = load_vector(path)

        assert torch.equal(original, loaded)

    def test_roundtrip_preserves_dtype(self, tmp_path: Path) -> None:
        """Save and load should preserve tensor dtype."""
        original = torch.randn(64, dtype=torch.float64)

        path = tmp_path / "dtype_vector.pt"
        save_vector(original, path)
        loaded = load_vector(path)

        assert loaded.dtype == torch.float64
        assert torch.equal(original, loaded)

    def test_roundtrip_multidimensional(self, tmp_path: Path) -> None:
        """Save and load should work with multi-dimensional tensors."""
        torch.manual_seed(42)
        original = torch.randn(4, 8, 16)

        path = tmp_path / "multi_dim_vector.pt"
        save_vector(original, path)
        loaded = load_vector(path)

        assert loaded.shape == (4, 8, 16)
        assert torch.equal(original, loaded)

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Loading nonexistent file should raise FileNotFoundError."""
        path = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError, match="Vector file not found"):
            load_vector(path)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Save should create parent directories if they don't exist."""
        vector = torch.randn(16)
        path = tmp_path / "nested" / "dir" / "vector.pt"

        save_vector(vector, path)

        assert path.exists()
        assert path.parent.exists()


class TestDatasetSizeCapping:
    """Tests for cap_examples function."""

    def test_requested_less_than_max(self, caplog: pytest.LogCaptureFixture) -> None:
        """When requested < max, should return requested without warning."""
        result = cap_examples(50, 100, "test_concept")

        assert result == 50
        # Should not log warning
        assert "Requested 50 examples" not in caplog.text

    def test_requested_equals_max(self, caplog: pytest.LogCaptureFixture) -> None:
        """When requested == max, should return requested without warning."""
        result = cap_examples(100, 100, "test_concept")

        assert result == 100
        assert "Requested 100 examples" not in caplog.text

    def test_requested_greater_than_max(self, caplog: pytest.LogCaptureFixture) -> None:
        """When requested > max, should return max with warning."""
        caplog.set_level("WARNING")
        result = cap_examples(500, 200, "sentiment")

        assert result == 200
        # Check warning was logged
        assert "Requested 500 examples" in caplog.text
        assert "sentiment" in caplog.text
        assert "200 available" in caplog.text

    def test_capping_with_zero_max(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should handle edge case of zero max available."""
        caplog.set_level("WARNING")
        result = cap_examples(100, 0, "empty_concept")

        assert result == 0
        assert "Requested 100 examples" in caplog.text

    def test_no_capping_when_within_limits(self) -> None:
        """Should not cap when requested is within limits."""
        result = cap_examples(10, 1000, "large_dataset")
        assert result == 10


# GPU placeholder tests - not implemented yet
class TestGPUPlaceholders:
    """Placeholder tests for GPU-dependent experiments."""

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_single_extraction_experiment1(self, tmp_path: Path) -> None:
        """Run single extraction with real model, verify vector shape and file creation."""
        result = run_diff_means_experiment(
            concept="sentiment",
            n_examples_list=[10],
            layers=[0.5],
            model_name="Qwen/Qwen3-1.7B",
            output_dir=tmp_path,
        )

        # Verify result structure
        assert "vector_paths" in result
        assert "heatmap_paths" in result
        assert "statistics" in result

        # Verify vector file was created
        vector_key = "n10_layer0.5"
        assert vector_key in result["vector_paths"]
        vector_paths = cast("dict[str, str]", result["vector_paths"])
        vector_path = Path(vector_paths[vector_key])
        assert vector_path.exists(), f"Vector file not created: {vector_path}"

        # Load and verify vector shape
        vector = load_vector(vector_path)
        assert vector.dim() == 1, f"Expected 1D tensor, got {vector.dim()}D"
        assert vector.numel() > 0, "Vector should have non-zero elements"

        # Verify heatmap file was created
        heatmap_key = "layer0.5"
        assert heatmap_key in result["heatmap_paths"]
        heatmap_paths = cast("dict[str, str]", result["heatmap_paths"])
        heatmap_path = Path(heatmap_paths[heatmap_key])
        assert heatmap_path.exists(), f"Heatmap file not created: {heatmap_path}"

        # Verify statistics
        assert heatmap_key in result["statistics"]
        stats = cast("dict[str, dict[str, float]]", result["statistics"])[heatmap_key]
        assert "mean_similarity" in stats
        assert "min_similarity" in stats
        assert "max_similarity" in stats

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_single_extraction_experiment2(self) -> None:
        """Placeholder for single extraction experiment 2."""
        # TODO: Implement actual GPU extraction test
        pass
