"""Tests for stability sweep experiment and plotting."""

from pathlib import Path

import pytest

from steering_geometry.config import StabilitySweepConfig
from steering_geometry.stability_comparison import (
    load_sweep_results,
    load_sweep_results_for_plotting,
    plot_stability_sweep,
    save_sweep_results,
)
from steering_geometry.types import StabilitySweepResult


def test_stability_sweep_config_defaults() -> None:
    """StabilitySweepConfig creates with correct defaults."""
    config = StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="refusal")
    assert config.num_runs == 5
    assert config.n_values == [100, 500, 1000, 5000, 10000]
    assert len(config.layers) == 10
    assert config.seed == 42
    assert config.display_concept == "Safety"
    assert config.canonical_concept == "refusal"


def test_stability_sweep_config_invalid_concept() -> None:
    """Invalid concept raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported concept"):
        StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="nonexistent")


def test_stability_sweep_config_invalid_model() -> None:
    """Invalid model raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model"):
        StabilitySweepConfig(model_name="fake/model", concept="refusal")


def test_stability_sweep_config_num_runs_validation() -> None:
    """num_runs < 2 raises ValueError."""
    with pytest.raises(ValueError, match="num_runs must be at least 2"):
        StabilitySweepConfig(model_name="Qwen/Qwen3-1.7B", concept="refusal", num_runs=1)


def test_stability_sweep_config_display_concept() -> None:
    """display_concept maps correctly for all concepts."""
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


def test_json_roundtrip(tmp_path: Path) -> None:
    """JSON save -> load preserves data exactly."""
    per_n = {
        100: {"mean": 0.82, "std": 0.05, "min": 0.70, "max": 0.90},
        500: {"mean": 0.91, "std": 0.03, "min": 0.85, "max": 0.95},
    }
    all_layers = {0.7: per_n}
    result = StabilitySweepResult(
        model_name="Qwen/Qwen3-1.7B",
        concept="refusal",
        display_concept="Safety",
        selected_layer=0.7,
        per_n_data=per_n,
        all_layers_data=all_layers,
    )

    save_sweep_results(result, output_dir=tmp_path)

    loaded = load_sweep_results(output_dir=tmp_path)
    key = ("Qwen/Qwen3-1.7B", "refusal")
    assert key in loaded

    loaded_result = loaded[key]
    assert loaded_result.selected_layer == 0.7
    assert loaded_result.concept == "refusal"
    assert loaded_result.display_concept == "Safety"
    assert loaded_result.per_n_data[100]["mean"] == 0.82
    assert loaded_result.per_n_data[500]["std"] == 0.03


def test_load_for_plotting_structure(tmp_path: Path) -> None:
    """load_sweep_results_for_plotting returns correct nested structure."""
    per_n = {
        100: {"mean": 0.82, "std": 0.05, "min": 0.70, "max": 0.90},
    }
    all_layers = {0.7: per_n}

    # Save two results for same concept but different models
    for model in ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B"]:
        result = StabilitySweepResult(
            model_name=model,
            concept="refusal",
            display_concept="Safety",
            selected_layer=0.7,
            per_n_data=per_n,
            all_layers_data=all_layers,
        )
        save_sweep_results(result, output_dir=tmp_path)

    plot_data = load_sweep_results_for_plotting(output_dir=tmp_path)

    assert "Safety" in plot_data
    assert "Qwen/Qwen3-1.7B" in plot_data["Safety"]
    assert "Qwen/Qwen3-14B" in plot_data["Safety"]
    assert plot_data["Safety"]["Qwen/Qwen3-1.7B"][100] == (0.82, 0.05)


def test_plot_stability_sweep_creates_pdfs(tmp_path: Path) -> None:
    """plot_stability_sweep creates 3 non-empty PDFs."""

    def make_result(model: str, concept: str, display: str) -> StabilitySweepResult:
        per_n = {
            n: {"mean": 0.8 + n * 0.0001, "std": 0.05, "min": 0.7, "max": 0.9}
            for n in [100, 500, 1000, 5000, 10000]
        }
        all_layers = {0.7: per_n}
        return StabilitySweepResult(model, concept, display, 0.7, per_n, all_layers)

    results = {
        "Safety": {
            "Qwen/Qwen3-1.7B": make_result("Qwen/Qwen3-1.7B", "refusal", "Safety"),
            "Qwen/Qwen3-14B": make_result("Qwen/Qwen3-14B", "refusal", "Safety"),
            "allenai/Olmo-3-1025-7B": make_result("allenai/Olmo-3-1025-7B", "refusal", "Safety"),
            "allenai/Olmo-3-1125-32B": make_result("allenai/Olmo-3-1125-32B", "refusal", "Safety"),
        },
        "Politeness": {
            "Qwen/Qwen3-1.7B": make_result("Qwen/Qwen3-1.7B", "polite", "Politeness"),
            "Qwen/Qwen3-14B": make_result("Qwen/Qwen3-14B", "polite", "Politeness"),
            "allenai/Olmo-3-1025-7B": make_result("allenai/Olmo-3-1025-7B", "polite", "Politeness"),
            "allenai/Olmo-3-1125-32B": make_result(
                "allenai/Olmo-3-1125-32B", "polite", "Politeness"
            ),
        },
        "Sentiment": {
            "Qwen/Qwen3-1.7B": make_result("Qwen/Qwen3-1.7B", "sentiment", "Sentiment"),
            "Qwen/Qwen3-14B": make_result("Qwen/Qwen3-14B", "sentiment", "Sentiment"),
            "allenai/Olmo-3-1025-7B": make_result(
                "allenai/Olmo-3-1025-7B", "sentiment", "Sentiment"
            ),
            "allenai/Olmo-3-1125-32B": make_result(
                "allenai/Olmo-3-1125-32B", "sentiment", "Sentiment"
            ),
        },
    }

    paths = plot_stability_sweep(results, output_dir=tmp_path)

    assert len(paths) == 3
    for pdf_path in paths:
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
        assert pdf_path.suffix == ".pdf"

    # Verify correct filenames
    names = {p.name for p in paths}
    assert "safety_stability_sweep.pdf" in names
    assert "politeness_stability_sweep.pdf" in names
    assert "sentiment_stability_sweep.pdf" in names


def test_plot_four_model_lines(tmp_path: Path) -> None:
    """Each figure has exactly 4 model result entries producing a non-trivial PDF."""

    def make_result(model: str, concept: str, display: str) -> StabilitySweepResult:
        per_n = {100: {"mean": 0.8, "std": 0.05, "min": 0.7, "max": 0.9}}
        all_layers = {0.7: per_n}
        return StabilitySweepResult(model, concept, display, 0.7, per_n, all_layers)

    results = {
        "Safety": {
            "Qwen/Qwen3-1.7B": make_result("Qwen/Qwen3-1.7B", "refusal", "Safety"),
            "Qwen/Qwen3-14B": make_result("Qwen/Qwen3-14B", "refusal", "Safety"),
            "allenai/Olmo-3-1025-7B": make_result("allenai/Olmo-3-1025-7B", "refusal", "Safety"),
            "allenai/Olmo-3-1125-32B": make_result("allenai/Olmo-3-1125-32B", "refusal", "Safety"),
        },
    }

    paths = plot_stability_sweep(results, output_dir=tmp_path)
    assert len(paths) == 1

    # Read back the PDF and verify it's non-trivial
    assert paths[0].stat().st_size > 1000  # A real plot, not empty


def test_load_empty_directory(tmp_path: Path) -> None:
    """load_sweep_results handles empty/nonexistent directories gracefully."""
    results = load_sweep_results(output_dir=tmp_path / "nonexistent")
    assert results == {}


# =============================================================================
# Tests for run_stability_sweep() with mocked model/extraction
# =============================================================================


def test_run_stability_sweep_returns_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_stability_sweep returns correct result structure with mock model."""
    from unittest.mock import MagicMock

    import torch

    from steering_geometry.config import ExtractionConfig
    from steering_geometry.stability_comparison import run_stability_sweep
    from steering_geometry.types import ContrastPair, SteeringVector

    fake_pairs = [
        ContrastPair(
            positive=f"positive text {i}",
            negative=f"negative text {i}",
            metadata={"concept": "sentiment", "pair_index": i},
        )
        for i in range(20)
    ]

    call_count = 0

    def mock_extract_steering_vector(
        model: object,
        pairs: object,
        config: ExtractionConfig,
    ) -> SteeringVector:
        nonlocal call_count
        call_count += 1
        layer_activations = {}
        for layer_frac in config.layers:
            abs_idx = int(layer_frac * 100)
            base = torch.randn(8)
            layer_activations[abs_idx] = base / base.norm()
        return SteeringVector(
            layer_activations=layer_activations,
            model_name="Qwen/Qwen3-1.7B",
            concept="sentiment",
            method="mean",
        )

    monkeypatch.setattr(
        "steering_geometry.stability_comparison.HookedModel",
        MagicMock(),
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.load_contrast_pairs",
        lambda concept, num_pairs=10000: fake_pairs,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.extract_steering_vector",
        mock_extract_steering_vector,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.save_vector",
        lambda vector, path: None,
    )

    config = StabilitySweepConfig(
        model_name="Qwen/Qwen3-1.7B",
        concept="sentiment",
        n_values=[10, 20],
        layers=[0.3, 0.5],
        num_runs=2,
        seed=42,
        output_dir=tmp_path,
    )

    result = run_stability_sweep(config)

    assert isinstance(result, StabilitySweepResult)
    assert result.model_name == "Qwen/Qwen3-1.7B"
    assert result.concept == "sentiment"
    assert result.display_concept == "Sentiment"
    assert result.selected_layer in [0.3, 0.5]
    assert set(result.per_n_data.keys()) == {10, 20}
    assert 0.3 in result.all_layers_data
    assert 0.5 in result.all_layers_data
    assert call_count == len(config.n_values) * config.num_runs


def test_run_stability_sweep_layer_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_stability_sweep selects the layer with highest average cos_sim."""
    from unittest.mock import MagicMock

    import torch

    from steering_geometry.config import ExtractionConfig
    from steering_geometry.stability_comparison import run_stability_sweep
    from steering_geometry.types import ContrastPair, SteeringVector

    fake_pairs = [
        ContrastPair(
            positive=f"pos {i}",
            negative=f"neg {i}",
            metadata={"concept": "sentiment", "pair_index": i},
        )
        for i in range(20)
    ]

    def mock_extract_steering_vector(
        model: object,
        pairs: object,
        config: ExtractionConfig,
    ) -> SteeringVector:
        layer_activations = {}
        for layer_frac in config.layers:
            abs_idx = int(layer_frac * 100)
            if layer_frac == 0.7:
                base = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                noise = torch.randn(8) * 0.01
                vec = base + noise
            else:
                vec = torch.randn(8)
            layer_activations[abs_idx] = vec / vec.norm()
        return SteeringVector(
            layer_activations=layer_activations,
            model_name="Qwen/Qwen3-1.7B",
            concept="sentiment",
            method="mean",
        )

    monkeypatch.setattr(
        "steering_geometry.stability_comparison.HookedModel",
        MagicMock(),
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.load_contrast_pairs",
        lambda concept, num_pairs=10000: fake_pairs,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.extract_steering_vector",
        mock_extract_steering_vector,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.save_vector",
        lambda vector, path: None,
    )

    config = StabilitySweepConfig(
        model_name="Qwen/Qwen3-1.7B",
        concept="sentiment",
        n_values=[10],
        layers=[0.3, 0.7],
        num_runs=3,
        seed=42,
        output_dir=tmp_path,
    )

    result = run_stability_sweep(config)

    avg_03 = result.all_layers_data[0.3][10]["mean"]
    avg_07 = result.all_layers_data[0.7][10]["mean"]
    assert avg_07 > avg_03
    assert result.selected_layer == 0.7


def test_run_stability_sweep_per_n_data_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """per_n_data has correct N values as keys with mean/std stats."""
    from unittest.mock import MagicMock

    import torch

    from steering_geometry.config import ExtractionConfig
    from steering_geometry.stability_comparison import run_stability_sweep
    from steering_geometry.types import ContrastPair, SteeringVector

    fake_pairs = [
        ContrastPair(
            positive=f"pos {i}",
            negative=f"neg {i}",
            metadata={"concept": "refusal", "pair_index": i},
        )
        for i in range(20)
    ]

    def mock_extract_steering_vector(
        model: object,
        pairs: object,
        config: ExtractionConfig,
    ) -> SteeringVector:
        layer_activations = {}
        for layer_frac in config.layers:
            abs_idx = int(layer_frac * 100)
            vec = torch.randn(8)
            layer_activations[abs_idx] = vec / vec.norm()
        return SteeringVector(
            layer_activations=layer_activations,
            model_name="Qwen/Qwen3-1.7B",
            concept="refusal",
            method="mean",
        )

    monkeypatch.setattr(
        "steering_geometry.stability_comparison.HookedModel",
        MagicMock(),
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.load_contrast_pairs",
        lambda concept, num_pairs=10000: fake_pairs,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.extract_steering_vector",
        mock_extract_steering_vector,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.save_vector",
        lambda vector, path: None,
    )

    config = StabilitySweepConfig(
        model_name="Qwen/Qwen3-1.7B",
        concept="refusal",
        n_values=[5, 10, 15],
        layers=[0.5],
        num_runs=2,
        seed=42,
        output_dir=tmp_path,
    )

    result = run_stability_sweep(config)

    assert set(result.per_n_data.keys()) == {5, 10, 15}
    for n in [5, 10, 15]:
        stats = result.per_n_data[n]
        assert "mean" in stats
        assert "std" in stats
        assert -1.0 <= stats["mean"] <= 1.0
        assert stats["std"] >= 0.0


def test_plot_stability_sweep_with_plot_layer(tmp_path: Path) -> None:
    """plot_stability_sweep with plot_layer uses all_layers_data for that layer."""
    from unittest.mock import patch

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401 — must load before patching Axes.plot

    def make_result(model: str, concept: str, display: str) -> StabilitySweepResult:
        per_n_best = {
            n: {"mean": 0.9 + n * 0.00001, "std": 0.02, "min": 0.85, "max": 0.95}
            for n in [100, 500, 1000]
        }
        per_n_layer03 = {
            n: {"mean": 0.5 + n * 0.00001, "std": 0.10, "min": 0.3, "max": 0.7}
            for n in [100, 500, 1000]
        }
        per_n_layer07 = {
            n: {"mean": 0.8 + n * 0.00001, "std": 0.03, "min": 0.75, "max": 0.85}
            for n in [100, 500, 1000]
        }
        all_layers = {0.3: per_n_layer03, 0.7: per_n_layer07}
        return StabilitySweepResult(model, concept, display, 0.7, per_n_best, all_layers)

    results = {
        "Safety": {
            "Qwen/Qwen3-1.7B": make_result("Qwen/Qwen3-1.7B", "refusal", "Safety"),
            "Qwen/Qwen3-14B": make_result("Qwen/Qwen3-14B", "refusal", "Safety"),
        },
    }

    with patch("matplotlib.axes.Axes.plot") as mock_plot:
        paths = plot_stability_sweep(results, output_dir=tmp_path, plot_layer=0.3)

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].stat().st_size > 0
    assert "layer0.3" in paths[0].name
    assert paths[0].name == "safety_stability_sweep_layer0.3.pdf"

    for call in mock_plot.call_args_list:
        y_data = call[0][1]
        for val in y_data:
            assert 0.4 < val < 0.6, f"Expected layer 0.3 data (~0.5), got {val}"

    with patch("matplotlib.axes.Axes.plot") as mock_plot_default:
        paths_default = plot_stability_sweep(results, output_dir=tmp_path)

    assert len(paths_default) == 1
    assert "layer" not in paths_default[0].name
    assert paths_default[0].name == "safety_stability_sweep.pdf"

    for call in mock_plot_default.call_args_list:
        y_data = call[0][1]
        for val in y_data:
            assert 0.85 < val < 0.95, f"Expected per_n_data (best layer ~0.9), got {val}"


def test_plot_stability_sweep_invalid_layer_raises(tmp_path: Path) -> None:
    """plot_stability_sweep raises ValueError for unavailable plot_layer."""

    def make_result(model: str, concept: str, display: str) -> StabilitySweepResult:
        per_n = {100: {"mean": 0.8, "std": 0.05, "min": 0.7, "max": 0.9}}
        all_layers = {0.5: per_n, 0.7: per_n}
        return StabilitySweepResult(model, concept, display, 0.7, per_n, all_layers)

    results = {
        "Safety": {
            "Qwen/Qwen3-1.7B": make_result("Qwen/Qwen3-1.7B", "refusal", "Safety"),
        },
    }

    with pytest.raises(ValueError, match="plot_layer=0.9 not found"):
        plot_stability_sweep(results, output_dir=tmp_path, plot_layer=0.9)


# =============================================================================
# Tests for reference-based comparison
# =============================================================================


def test_compute_reference_statistics() -> None:
    """compute_reference_statistics returns correct stats vs a reference vector."""
    import torch

    from steering_geometry.stability_comparison import compute_reference_statistics

    reference = torch.tensor([1.0, 0.0, 0.0])
    vectors = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.7071, 0.7071, 0.0]),
    ]
    stats = compute_reference_statistics(vectors, reference)
    assert abs(stats["mean"] - (1.0 + 0.0 + 0.7071) / 3) < 0.01
    assert abs(stats["min"] - 0.0) < 0.01
    assert abs(stats["max"] - 1.0) < 0.01
    assert stats["std"] > 0


def test_run_stability_sweep_with_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_stability_sweep with reference_n uses reference-based comparison."""
    from unittest.mock import MagicMock

    import torch

    from steering_geometry.config import ExtractionConfig
    from steering_geometry.stability_comparison import (
        run_stability_sweep,
        save_vector,
    )
    from steering_geometry.types import ContrastPair, SteeringVector

    fake_pairs = [
        ContrastPair(
            positive=f"positive text {i}",
            negative=f"negative text {i}",
            metadata={"concept": "sentiment", "pair_index": i},
        )
        for i in range(20)
    ]

    reference_vec = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    reference_vec = reference_vec / reference_vec.norm()

    def mock_extract_steering_vector(
        model: object,
        pairs: object,
        config: ExtractionConfig,
    ) -> SteeringVector:
        layer_activations = {}
        for layer_frac in config.layers:
            abs_idx = int(layer_frac * 100)
            base = torch.randn(8)
            layer_activations[abs_idx] = base / base.norm()
        return SteeringVector(
            layer_activations=layer_activations,
            model_name="Qwen/Qwen3-1.7B",
            concept="sentiment",
            method="mean",
        )

    real_save_vector = save_vector

    def mock_save_vector(vector: torch.Tensor, path: Path) -> None:
        real_save_vector(vector, path)

    monkeypatch.setattr(
        "steering_geometry.stability_comparison.HookedModel",
        MagicMock(),
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.load_contrast_pairs",
        lambda concept, num_pairs=10000: fake_pairs,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.extract_steering_vector",
        mock_extract_steering_vector,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.save_vector",
        mock_save_vector,
    )

    # Pre-save reference vectors for both layers
    concept_dir = tmp_path / "vectors" / "sentiment"
    for layer in [0.3, 0.5]:
        ref_path = concept_dir / f"n10_run0_layer{layer}.pt"
        save_vector(reference_vec, ref_path)

    config = StabilitySweepConfig(
        model_name="Qwen/Qwen3-1.7B",
        concept="sentiment",
        n_values=[5],
        layers=[0.3, 0.5],
        num_runs=2,
        seed=42,
        output_dir=tmp_path,
        reference_n=10,
    )

    result = run_stability_sweep(config)

    assert isinstance(result, StabilitySweepResult)
    assert result.model_name == "Qwen/Qwen3-1.7B"
    assert result.concept == "sentiment"
    assert 0.3 in result.all_layers_data
    assert 0.5 in result.all_layers_data
    for layer in [0.3, 0.5]:
        stats = result.all_layers_data[layer][5]
        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats
        assert -1.0 <= stats["mean"] <= 1.0


def test_reference_vector_not_found_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """reference_n with missing reference file raises FileNotFoundError."""
    from unittest.mock import MagicMock

    import torch

    from steering_geometry.config import ExtractionConfig
    from steering_geometry.stability_comparison import run_stability_sweep
    from steering_geometry.types import ContrastPair, SteeringVector

    fake_pairs = [
        ContrastPair(
            positive=f"pos {i}",
            negative=f"neg {i}",
            metadata={"concept": "sentiment", "pair_index": i},
        )
        for i in range(20)
    ]

    def mock_extract_steering_vector(
        model: object,
        pairs: object,
        config: ExtractionConfig,
    ) -> SteeringVector:
        layer_activations = {}
        for layer_frac in config.layers:
            abs_idx = int(layer_frac * 100)
            vec = torch.randn(8)
            layer_activations[abs_idx] = vec / vec.norm()
        return SteeringVector(
            layer_activations=layer_activations,
            model_name="Qwen/Qwen3-1.7B",
            concept="sentiment",
            method="mean",
        )

    monkeypatch.setattr(
        "steering_geometry.stability_comparison.HookedModel",
        MagicMock(),
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.load_contrast_pairs",
        lambda concept, num_pairs=10000: fake_pairs,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.extract_steering_vector",
        mock_extract_steering_vector,
    )
    monkeypatch.setattr(
        "steering_geometry.stability_comparison.save_vector",
        lambda vector, path: None,
    )

    config = StabilitySweepConfig(
        model_name="Qwen/Qwen3-1.7B",
        concept="sentiment",
        n_values=[5],
        layers=[0.5],
        num_runs=2,
        seed=42,
        output_dir=tmp_path,
        reference_n=999,
    )

    with pytest.raises(FileNotFoundError, match="Reference vector not found"):
        run_stability_sweep(config)


def test_reference_n_allows_num_runs_1(tmp_path: Path) -> None:
    """With reference_n set, num_runs=1 is allowed."""
    config = StabilitySweepConfig(
        model_name="Qwen/Qwen3-1.7B",
        concept="sentiment",
        n_values=[10],
        layers=[0.5],
        num_runs=1,
        reference_n=3000,
        output_dir=tmp_path,
    )
    assert config.num_runs == 1
    assert config.reference_n == 3000


def test_reference_n_none_requires_num_runs_ge_2() -> None:
    """Without reference_n, num_runs=1 still raises ValueError."""
    with pytest.raises(ValueError, match="num_runs must be at least 2"):
        StabilitySweepConfig(
            model_name="Qwen/Qwen3-1.7B",
            concept="sentiment",
            n_values=[10],
            layers=[0.5],
            num_runs=1,
        )
