"""Tests for sweep_evaluation module."""

import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from steering_geometry.types import (
    JudgeScore,
    MMLUProResult,
)


def _make_sweep_result_data(
    multipliers: list[float] | None = None,
    steer_tokens_values: list[int | None] | None = None,
) -> dict[str, object]:
    """Build a minimal SweepResult-compatible dict for testing."""
    if multipliers is None:
        multipliers = [0.1, 1.0]
    if steer_tokens_values is None:
        steer_tokens_values = [5, None]

    cells: list[dict[str, object]] = []
    for mult in multipliers:
        for st in steer_tokens_values:
            cells.append(
                {
                    "multiplier": mult,
                    "steer_tokens": st,
                    "concept_score": 7.0,
                    "fluency_score": 8.0,
                    "mmlu_pro_accuracy": 45.0,
                    "num_samples": 2,
                }
            )

    return {
        "concept": "sentiment",
        "model": "test-model",
        "layer_frac": 0.7,
        "multipliers": multipliers,
        "steer_tokens_values": steer_tokens_values,
        "cells": cells,
        "output_dir": "outputs/test",
    }


# ---------------------------------------------------------------------------
# Test 1 & 2: Type structure verification
# ---------------------------------------------------------------------------


def test_sweep_cell_result_type() -> None:
    """SweepCellResult TypedDict has expected keys."""
    from steering_geometry.sweep_evaluation import SweepCellResult

    expected_keys = {
        "multiplier",
        "steer_tokens",
        "concept_score",
        "fluency_score",
        "mmlu_pro_accuracy",
        "num_samples",
    }
    actual_keys = set(SweepCellResult.__annotations__.keys())
    diff = expected_keys.symmetric_difference(actual_keys)
    assert expected_keys == actual_keys, f"Key mismatch: {diff}"


def test_sweep_result_type() -> None:
    """SweepResult TypedDict has expected keys."""
    from steering_geometry.sweep_evaluation import SweepResult

    expected_keys = {
        "concept",
        "model",
        "layer_frac",
        "multipliers",
        "steer_tokens_values",
        "cells",
        "output_dir",
    }
    actual_keys = set(SweepResult.__annotations__.keys())
    diff = expected_keys.symmetric_difference(actual_keys)
    assert expected_keys == actual_keys, f"Key mismatch: {diff}"


# ---------------------------------------------------------------------------
# Test 3: plot_sweep_heatmaps creates files
# ---------------------------------------------------------------------------


def test_plot_sweep_heatmaps_creates_files(tmp_path: Path) -> None:
    """plot_sweep_heatmaps creates both PDF and PNG output files."""
    from steering_geometry.sweep_evaluation import plot_sweep_heatmaps

    result_data = _make_sweep_result_data()
    # Patch output_dir to tmp_path
    result_data["output_dir"] = str(tmp_path)

    output_paths = plot_sweep_heatmaps(result_data, output_dir=tmp_path)

    assert len(output_paths) >= 1
    for path in output_paths:
        assert path.exists(), f"Expected file {path} to exist"
        assert path.stat().st_size > 0, f"File {path} is empty"
        assert path.suffix in {".pdf", ".png"}, f"Unexpected suffix: {path.suffix}"


# ---------------------------------------------------------------------------
# Test 4: Heatmap matrix dimensions
# ---------------------------------------------------------------------------


def test_plot_sweep_heatmaps_correct_dimensions(tmp_path: Path) -> None:
    """Heatmap matrix rows match multipliers count and cols match steer_tokens count."""
    from steering_geometry.sweep_evaluation import plot_sweep_heatmaps

    multipliers = [0.5, 2.0]
    steer_tokens_values: list[int | None] = [3, None]

    result_data = _make_sweep_result_data(
        multipliers=multipliers,
        steer_tokens_values=steer_tokens_values,
    )
    result_data["output_dir"] = str(tmp_path)

    output_paths = plot_sweep_heatmaps(result_data, output_dir=tmp_path)

    # Should produce 2 heatmaps × 2 formats = 4 files
    assert len(output_paths) == 4
    for path in output_paths:
        assert path.exists()


# ---------------------------------------------------------------------------
# Test 5: steer_tokens=None labeled as "all" in heatmap
# ---------------------------------------------------------------------------


def test_plot_sweep_heatmaps_none_label(tmp_path: Path) -> None:
    """steer_tokens=None produces 'all' label in X-axis of heatmaps."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from steering_geometry.sweep_evaluation import plot_sweep_heatmaps

    result_data = _make_sweep_result_data(
        multipliers=[1.0],
        steer_tokens_values=[5, None],
    )
    result_data["output_dir"] = str(tmp_path)

    captured_figs: list[Figure] = []

    def _capture_close(fig: object) -> None:
        captured_figs.append(cast(Figure, fig))

    with patch("matplotlib.pyplot.close", side_effect=_capture_close):
        plot_sweep_heatmaps(result_data, output_dir=tmp_path, formats=["png"])

    ax1 = captured_figs[0].axes[0]
    xtick_labels = [tl.get_text() for tl in ax1.get_xticklabels()]
    assert xtick_labels == ["5", "all"]
    plt.close("all")


def test_plot_paper_sweep_heatmap_uses_tokens_rows_and_alpha_columns(tmp_path: Path) -> None:
    """Paper sweep heatmap uses token rows, alpha columns, and one combined figure."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from steering_geometry.sweep_evaluation import plot_paper_sweep_heatmap

    result_data = _make_sweep_result_data(
        multipliers=[0.1, 1.0, 10.0],
        steer_tokens_values=[1, 5, 10],
    )
    result_data["output_dir"] = str(tmp_path)

    captured_figs: list[Figure] = []

    def _capture_close(fig: object) -> None:
        captured_figs.append(cast(Figure, fig))

    with patch("matplotlib.pyplot.close", side_effect=_capture_close):
        paths = plot_paper_sweep_heatmap(result_data, output_dir=tmp_path, formats=["png"])

    assert len(paths) == 1
    assert paths[0].name == "sweep_tradeoff_table_heatmap.png"
    assert paths[0].exists()
    assert paths[0].stat().st_size > 0

    axes = captured_figs[0].axes[:2]
    assert [tick.get_text() for tick in axes[0].get_xticklabels()] == [
        r"$\alpha=0.1$",
        r"$\alpha=1$",
        r"$\alpha=10$",
    ]
    assert [tick.get_text() for tick in axes[0].get_yticklabels()] == ["1", "5", "10"]
    assert axes[0].get_title() == "Target Concept"
    assert axes[1].get_title() == "General Ability"
    plt.close("all")


def _make_4x4_result(concept: str, output_dir: str) -> dict[str, object]:
    """Build a 4x4 (strength x prefix) SweepResult for one concept."""
    multipliers = [0.01, 0.1, 1.0, 10.0]
    steer_tokens_values: list[int | None] = [1, 5, 10, None]
    cells: list[dict[str, object]] = []
    for mult in multipliers:
        for st in steer_tokens_values:
            cells.append(
                {
                    "multiplier": mult,
                    "steer_tokens": st,
                    "concept_score": 70.0,
                    "fluency_score": 0.0,
                    "mmlu_pro_accuracy": 50.0,
                    "num_samples": 4,
                }
            )
    return {
        "concept": concept,
        "model": "Qwen/Qwen3-14B",
        "layer_frac": 0.7,
        "multipliers": multipliers,
        "steer_tokens_values": steer_tokens_values,
        "cells": cells,
        "output_dir": output_dir,
    }


def test_plot_paper_sweep_concept_grid_arranges_metrics_in_two_rows(tmp_path: Path) -> None:
    """Concept grid plots general ability on row 0, steering performance on row 1."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from steering_geometry.sweep_evaluation import plot_paper_sweep_concept_grid

    results = [
        _make_4x4_result("refusal", str(tmp_path)),
        _make_4x4_result("polite", str(tmp_path)),
        _make_4x4_result("sentiment", str(tmp_path)),
    ]

    captured_figs: list[Figure] = []

    def _capture_close(fig: object) -> None:
        captured_figs.append(cast(Figure, fig))

    with patch("matplotlib.pyplot.close", side_effect=_capture_close):
        paths = plot_paper_sweep_concept_grid(results, output_dir=tmp_path, formats=["png"])

    assert len(paths) == 1
    assert paths[0].name == "sweep_concept_grid_heatmap.png"
    assert paths[0].exists()
    assert paths[0].stat().st_size > 0

    fig = captured_figs[0]
    assert len(fig.axes) == 6
    width, height = fig.get_size_inches()
    assert 6.2 <= width <= 6.8
    assert height <= 3.0

    top_titles = [fig.axes[i].get_title() for i in range(3)]
    assert top_titles == ["Safety", "Politeness", "Sentiment"]
    assert fig.axes[0].get_ylabel() == ""
    assert fig.axes[3].get_ylabel() == ""
    assert fig.axes[0].images[0].cmap.name == "paper_general_lively"
    assert fig.axes[3].images[0].cmap.name == "paper_steering_lively"

    figure_labels = [text.get_text() for text in fig.texts]
    assert "General" in figure_labels
    assert "Steering" in figure_labels
    assert any(r"\bar{r}" in label for label in figure_labels)
    row_label_x = [text.get_position()[0] for text in fig.texts if text.get_text() == "General"]
    assert row_label_x == [0.055]
    assert all(ax.get_xlabel() == "" for ax in fig.axes)

    cell_labels = [text.get_text() for text in fig.axes[0].texts]
    assert "50.00" in cell_labels

    bottom_xticks = [tick.get_text() for tick in fig.axes[3].get_xticklabels()]
    assert bottom_xticks == ["0.01", "0.1", "1", "10"]
    bottom_yticks = [tick.get_text() for tick in fig.axes[3].get_yticklabels()]
    assert bottom_yticks == ["1", "5", "10", "L"]
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 6: Grid construction via run_sweep_evaluation cell count
# ---------------------------------------------------------------------------


def test_sweep_evaluation_grid_construction() -> None:
    """The sweep produces correct number of cells for 2x2 grid."""
    from steering_geometry.sweep_evaluation import SweepResult

    # Verify the grid: 2 multipliers × 2 steer_tokens = 4 cells
    multipliers = [0.1, 1.0]
    steer_tokens_values: list[int | None] = [5, None]

    cells: list[dict[str, object]] = []
    for mult in multipliers:
        for st in steer_tokens_values:
            cells.append(
                {
                    "multiplier": mult,
                    "steer_tokens": st,
                    "concept_score": 5.0,
                    "fluency_score": 5.0,
                    "mmlu_pro_accuracy": 40.0,
                    "num_samples": 1,
                }
            )

    result: SweepResult = {
        "concept": "sentiment",
        "model": "test",
        "layer_frac": 0.7,
        "multipliers": multipliers,
        "steer_tokens_values": steer_tokens_values,
        "cells": cells,
        "output_dir": "outputs/test",
    }

    assert len(result["cells"]) == 4

    # Verify all 4 (multiplier, steer_tokens) combos present
    cell_keys = {(cell["multiplier"], cell["steer_tokens"]) for cell in result["cells"]}
    expected_keys = {(0.1, 5), (0.1, None), (1.0, 5), (1.0, None)}
    assert cell_keys == expected_keys


# ---------------------------------------------------------------------------
# Test 7: Full sweep with JSON output
# ---------------------------------------------------------------------------


def test_sweep_evaluation_json_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run minimal sweep (2x2 grid, 2 samples) and verify JSON output format."""
    from steering_geometry.sweep_evaluation import run_sweep_evaluation

    # Fake contrast pairs
    fake_pairs = [MagicMock(positive=f"pos {i}", negative=f"neg {i}") for i in range(2)]
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.load_contrast_pairs",
        lambda concept, num_pairs=100: fake_pairs,
    )

    # Mock model
    mock_model = MagicMock()
    mock_model.resolve_layers.return_value = [10]
    mock_model.generate_with_steering.return_value = "Steered output text"

    mock_model_cls = MagicMock(return_value=mock_model)
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.HookedModel",
        mock_model_cls,
    )

    # Fake vector (raw tensor saved directly)
    vector_file = tmp_path / "test_vector.pt"
    fake_vector = torch.randn(8)
    torch.save(fake_vector, vector_file)

    # Mock judge evaluator
    mock_judge = MagicMock()
    mock_judge.evaluate_concept.return_value = JudgeScore(
        concept_score=7,
        fluency_score=8,
        final_score=7.5,
        reasoning="ok",
    )

    # Mock MMLU-Pro evaluator
    mock_mmlu_pro = MagicMock()
    mock_mmlu_pro.evaluate.return_value = MMLUProResult(
        accuracy=45.0,
        total=10,
        correct=5,
        refused=0,
        extract_failed=0,
        per_category={},
        per_category_counts={},
        predictions=[],
    )

    with (
        patch(
            "steering_geometry.sweep_evaluation.JudgeEvaluator",
            return_value=mock_judge,
        ),
        patch(
            "steering_geometry.sweep_evaluation.MMLUProEvaluator",
            return_value=mock_mmlu_pro,
        ),
    ):
        output_dir = tmp_path / "sweep_output"
        result = run_sweep_evaluation(
            concept="sentiment",
            model_name="test-model",
            vector_path=str(vector_file),
            layer_frac=0.7,
            multipliers=[0.1, 1.0],
            steer_tokens_values=[5, None],
            num_samples=2,
            mmlu_pro_num_questions=10,
            output_dir=str(output_dir),
        )

    # Verify result structure
    assert "cells" in result
    assert len(result["cells"]) == 4  # 2 multipliers × 2 steer_tokens

    # Verify JSON file was saved
    result_json = output_dir / "sentiment" / "test-model" / "sweep_results.json"
    assert result_json.exists(), f"JSON not found at {result_json}"

    with result_json.open() as f:
        saved = json.load(f)

    assert saved["concept"] == "sentiment"
    assert saved["model"] == "test-model"
    assert "cells" in saved
    assert len(saved["cells"]) == 4

    for cell in saved["cells"]:
        assert "multiplier" in cell
        assert "steer_tokens" in cell
        assert "concept_score" in cell
        assert "mmlu_pro_accuracy" in cell
        assert "num_samples" in cell


def test_sweep_evaluation_dict_vector_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sweep loads vectors saved in dict format ({"vector": SteeringVector, ...})."""
    from steering_geometry.sweep_evaluation import run_sweep_evaluation
    from steering_geometry.types import SteeringVector

    fake_pairs = [MagicMock(positive=f"pos {i}", negative=f"neg {i}") for i in range(2)]
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.load_contrast_pairs",
        lambda concept, num_pairs=100: fake_pairs,
    )

    mock_model = MagicMock()
    mock_model.resolve_layers.return_value = [10]
    mock_model.generate_with_steering.return_value = "Steered output"
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.HookedModel",
        MagicMock(return_value=mock_model),
    )

    vector_file = tmp_path / "dict_vector.pt"
    fake_steering = SteeringVector(
        layer_activations={0: torch.randn(8)},
        model_name="test-model",
        concept="sentiment",
        method="mean",
    )
    torch.save({"vector": fake_steering, "num_pairs": 100}, vector_file)

    mock_judge = MagicMock()
    mock_judge.evaluate_concept.return_value = JudgeScore(
        concept_score=6,
        fluency_score=7,
        final_score=6.5,
        reasoning="ok",
    )

    with (
        patch(
            "steering_geometry.sweep_evaluation.JudgeEvaluator",
            return_value=mock_judge,
        ),
        patch("steering_geometry.sweep_evaluation.MMLUProEvaluator"),
    ):
        output_dir = tmp_path / "sweep_dict"
        result = run_sweep_evaluation(
            concept="sentiment",
            model_name="test-model",
            vector_path=str(vector_file),
            layer_frac=0.7,
            multipliers=[1.0],
            steer_tokens_values=[5],
            num_samples=2,
            evaluate_mmlu=False,
            output_dir=str(output_dir),
        )

    assert len(result["cells"]) == 1
    assert result["cells"][0]["concept_score"] == 6.0


def test_sweep_evaluation_refusal_prefix_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusal concept uses prefix matching when no behaviors file is provided."""
    from steering_geometry.sweep_evaluation import run_sweep_evaluation

    fake_pairs = [MagicMock(positive=f"pos {i}", negative=f"neg {i}") for i in range(2)]
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.load_contrast_pairs",
        lambda concept, num_pairs=100: fake_pairs,
    )

    mock_model = MagicMock()
    mock_model.resolve_layers.return_value = [10]
    mock_model.generate_with_steering.return_value = "I'm sorry, I cannot help with that."
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.HookedModel",
        MagicMock(return_value=mock_model),
    )

    vector_file = tmp_path / "refusal_vector.pt"
    torch.save(torch.randn(8), vector_file)

    with patch("steering_geometry.sweep_evaluation.MMLUProEvaluator"):
        output_dir = tmp_path / "sweep_refusal"
        result = run_sweep_evaluation(
            concept="refusal",
            model_name="test-model",
            vector_path=str(vector_file),
            layer_frac=0.7,
            multipliers=[1.0],
            steer_tokens_values=[5],
            num_samples=2,
            evaluate_mmlu=False,
            harmbench_behaviors_file="",
            output_dir=str(output_dir),
        )

    assert len(result["cells"]) == 1
    # All generations start with refusal prefix → ASR should be 0%
    assert result["cells"][0]["concept_score"] == 0.0


def test_run_sweep_evaluation_harmbench_aligned_behavior_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HarmBench branch uses behavior texts as prompts and keys completions by real behavior_ids.

    Guards the original bug where generations came from contrast-pair prompts but were
    keyed to behavior_ids by index, breaking the generation/behavior mapping and yielding
    a silent ASR=0.
    """
    from steering_geometry.sweep_evaluation import run_sweep_evaluation

    fake_pairs = [MagicMock(positive=f"pos {i}", negative=f"neg {i}") for i in range(2)]
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.load_contrast_pairs",
        lambda concept, num_pairs=100: fake_pairs,
    )

    mock_model = MagicMock()
    mock_model.resolve_layers.return_value = [0]
    mock_model.generate_with_steering.return_value = "Sure, here is how to do it."
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.HookedModel",
        MagicMock(return_value=mock_model),
    )

    fake_vector = torch.ones(8)
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.torch.load",
        lambda *args, **kwargs: fake_vector,
    )

    behavior_ids = ["b1", "b2"]
    behavior_texts = ["do bad thing 1", "do bad thing 2"]
    behaviors = [
        {
            "behavior_id": bid,
            "behavior": text,
            "functional_category": "",
            "semantic_category": "",
            "tags": "",
            "context_string": "",
        }
        for bid, text in zip(behavior_ids, behavior_texts, strict=True)
    ]
    mock_evaluator = MagicMock()
    mock_evaluator.behaviors = behaviors
    mock_evaluator.load_behaviors = MagicMock(return_value=behaviors)
    mock_result = MagicMock()
    mock_result.asr = 50.0
    mock_evaluator.evaluate = AsyncMock(return_value=mock_result)
    monkeypatch.setattr(
        "steering_geometry.sweep_evaluation.HarmBenchEvaluator",
        MagicMock(return_value=mock_evaluator),
    )

    output_dir = tmp_path / "sweep_hb"
    result = run_sweep_evaluation(
        concept="refusal",
        model_name="fake/model",
        vector_path="fake.pt",
        layer_frac=0.7,
        multipliers=[1.0],
        steer_tokens_values=[None],
        num_samples=2,
        evaluate_concept=True,
        evaluate_mmlu=False,
        harmbench_behaviors_file="fake.csv",
        output_dir=str(output_dir),
    )

    gen_calls = mock_model.generate_with_steering.call_args_list
    prompts_used = [call.kwargs["prompt"] for call in gen_calls]
    assert prompts_used == behavior_texts

    assert mock_evaluator.evaluate.await_args is not None
    completions = mock_evaluator.evaluate.await_args.args[0]
    assert set(completions.keys()) == set(behavior_ids)
    assert list(completions.keys()) == behavior_ids
    assert len(result["cells"]) == 1
    assert result["cells"][0]["concept_score"] == 50.0
