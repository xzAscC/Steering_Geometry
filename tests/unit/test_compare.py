import sys
from pathlib import Path

import pytest
import torch

from steering_geometry.types import SteeringVector

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from compare_concepts import (
    _compute_average_cosine_similarity,
    _compute_l2_norm,
    _generate_comparison_report,
    _load_vectors_from_directory,
)


def _make_steering_vector(
    concept: str,
    activations: dict[int, list[float]],
    model_name: str = "test-model",
    method: str = "mean",
) -> SteeringVector:
    return SteeringVector(
        layer_activations={
            layer: torch.tensor(act, dtype=torch.float32) for layer, act in activations.items()
        },
        model_name=model_name,
        concept=concept,
        method=method,
    )


class TestComputeL2Norm:
    def test_single_layer_vector(self) -> None:
        vector = _make_steering_vector("test", {0: [3.0, 4.0]})
        norm = _compute_l2_norm(vector)
        assert norm == 5.0

    def test_multi_layer_vector(self) -> None:
        vector = _make_steering_vector("test", {0: [1.0, 0.0], 1: [0.0, 1.0]})
        norm = _compute_l2_norm(vector)
        assert norm == pytest.approx(2.0**0.5)

    def test_empty_layer_activations(self) -> None:
        vector = _make_steering_vector("test", {})
        norm = _compute_l2_norm(vector)
        assert norm == 0.0

    def test_zero_vector(self) -> None:
        vector = _make_steering_vector("test", {0: [0.0, 0.0, 0.0]})
        norm = _compute_l2_norm(vector)
        assert norm == 0.0


class TestComputeAverageCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vector = _make_steering_vector("test", {0: [1.0, 0.0, 0.0]})
        similarity = _compute_average_cosine_similarity(vector, vector)
        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        v1 = _make_steering_vector("test1", {0: [1.0, 0.0]})
        v2 = _make_steering_vector("test2", {0: [0.0, 1.0]})
        similarity = _compute_average_cosine_similarity(v1, v2)
        assert similarity == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        v1 = _make_steering_vector("test1", {0: [1.0, 0.0]})
        v2 = _make_steering_vector("test2", {0: [-1.0, 0.0]})
        similarity = _compute_average_cosine_similarity(v1, v2)
        assert similarity == pytest.approx(-1.0)

    def test_multi_layer_average(self) -> None:
        v1 = _make_steering_vector("test1", {0: [1.0, 0.0], 1: [1.0, 0.0]})
        v2 = _make_steering_vector("test2", {0: [1.0, 0.0], 1: [0.0, 1.0]})
        similarity = _compute_average_cosine_similarity(v1, v2)
        assert similarity == pytest.approx(0.5)

    def test_no_common_layers(self) -> None:
        v1 = _make_steering_vector("test1", {0: [1.0, 0.0]})
        v2 = _make_steering_vector("test2", {1: [1.0, 0.0]})
        similarity = _compute_average_cosine_similarity(v1, v2)
        assert similarity == 0.0


class TestGenerateComparisonReport:
    def test_single_concept(self) -> None:
        vectors = {
            "honesty": _make_steering_vector("honesty", {0: [1.0, 2.0]}),
        }
        report = _generate_comparison_report(vectors, None)

        assert report["cosine_similarities"] == {}
        assert "honesty" in report["l2_norms"]
        assert report["metadata"]["num_concepts"] == 1
        assert report["metadata"]["concepts"] == ["honesty"]

    def test_two_concepts(self) -> None:
        vectors = {
            "honesty": _make_steering_vector("honesty", {0: [1.0, 0.0]}),
            "sentiment": _make_steering_vector("sentiment", {0: [0.0, 1.0]}),
        }
        report = _generate_comparison_report(vectors, None)

        assert "honesty-sentiment" in report["cosine_similarities"]
        assert report["cosine_similarities"]["honesty-sentiment"] == pytest.approx(0.0)
        assert len(report["l2_norms"]) == 2
        assert report["metadata"]["num_concepts"] == 2

    def test_three_concepts_pairwise(self) -> None:
        vectors = {
            "a": _make_steering_vector("a", {0: [1.0, 0.0]}),
            "b": _make_steering_vector("b", {0: [0.0, 1.0]}),
            "c": _make_steering_vector("c", {0: [1.0, 1.0]}),
        }
        report = _generate_comparison_report(vectors, None)

        expected_pairs = {"a-b", "a-c", "b-c"}
        assert set(report["cosine_similarities"].keys()) == expected_pairs

    def test_model_name_from_filter(self) -> None:
        vectors = {
            "test": _make_steering_vector("test", {0: [1.0]}, model_name="other-model"),
        }
        report = _generate_comparison_report(vectors, "filter-model")
        assert report["metadata"]["model"] == "filter-model"

    def test_model_name_from_vector_when_no_filter(self) -> None:
        vectors = {
            "test": _make_steering_vector("test", {0: [1.0]}, model_name="vector-model"),
        }
        report = _generate_comparison_report(vectors, None)
        assert report["metadata"]["model"] == "vector-model"


class TestLoadVectorsFromDirectory:
    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            _load_vectors_from_directory(nonexistent, None)

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No .pt files found"):
            _load_vectors_from_directory(empty_dir, None)

    def test_no_matching_model(self, tmp_path: Path) -> None:
        vector = _make_steering_vector("test", {0: [1.0]}, model_name="model-a")
        vector_file = tmp_path / "test.pt"
        torch.save({"vector": vector}, vector_file)

        with pytest.raises(ValueError, match="No vectors found for model"):
            _load_vectors_from_directory(tmp_path, "model-b")

    def test_loads_all_vectors(self, tmp_path: Path) -> None:
        v1 = _make_steering_vector("honesty", {0: [1.0]})
        v2 = _make_steering_vector("sentiment", {0: [2.0]})

        torch.save({"vector": v1}, tmp_path / "honesty.pt")
        torch.save({"vector": v2}, tmp_path / "sentiment.pt")

        vectors = _load_vectors_from_directory(tmp_path, None)

        assert len(vectors) == 2
        assert "honesty" in vectors
        assert "sentiment" in vectors

    def test_filters_by_model(self, tmp_path: Path) -> None:
        v1 = _make_steering_vector("honesty", {0: [1.0]}, model_name="model-a")
        v2 = _make_steering_vector("sentiment", {0: [2.0]}, model_name="model-b")

        torch.save({"vector": v1}, tmp_path / "honesty.pt")
        torch.save({"vector": v2}, tmp_path / "sentiment.pt")

        vectors = _load_vectors_from_directory(tmp_path, "model-a")

        assert len(vectors) == 1
        assert "honesty" in vectors
        assert "sentiment" not in vectors

    def test_invalid_vector_type(self, tmp_path: Path) -> None:
        torch.save({"vector": "not a vector"}, tmp_path / "invalid.pt")

        with pytest.raises(TypeError, match="Expected SteeringVector"):
            _load_vectors_from_directory(tmp_path, None)
