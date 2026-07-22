"""Tests for the strength × prefix-length ablation module."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import cast

import pytest
import torch

from steering_geometry.strength_prefix_ablation import (
    _evaluate_concept,
    _load_steering_vector,
    compute_avg_activation_norm,
)
from steering_geometry.types import HarmBenchPrediction, HarmBenchResult, SteeringVector

# ---------------------------------------------------------------------------
# _load_steering_vector layer selection
# ---------------------------------------------------------------------------


def _unit(tensor: torch.Tensor) -> torch.Tensor:
    """Return the L2-unit-normalized form of a flat tensor."""
    return tensor / tensor.norm()


def _make_multi_layer_vector() -> SteeringVector:
    """Build a 3-layer SteeringVector with distinguishable per-layer tensors."""
    return SteeringVector(
        layer_activations={
            0: torch.full((4,), 1.0),
            5: torch.full((4,), 5.0),
            10: torch.full((4,), 10.0),
        },
        model_name="test-model",
        concept="sentiment",
        method="dim",
    )


class TestLoadSteeringVectorLayerSelection:
    """Selecting the requested layer from a multi-layer vector file."""

    def test_load_with_layer_idx_selects_requested_layer(self, tmp_path: Path) -> None:
        """Passing layer_idx=5 must return the layer-5 tensor, not layer-0."""
        sv = _make_multi_layer_vector()
        vector_path = tmp_path / "vec.pt"
        torch.save({"vector": sv}, vector_path)

        loaded, _concept = _load_steering_vector(str(vector_path), layer_idx=5)
        assert torch.allclose(loaded, _unit(torch.full((4,), 5.0)))

    def test_load_with_layer_idx_zero_is_not_silently_treated_as_default(
        self, tmp_path: Path
    ) -> None:
        """layer_idx=0 must select layer 0 explicitly (not just 'first')."""
        sv = _make_multi_layer_vector()
        vector_path = tmp_path / "vec.pt"
        torch.save({"vector": sv}, vector_path)

        loaded, _ = _load_steering_vector(str(vector_path), layer_idx=0)
        assert torch.allclose(loaded, _unit(torch.full((4,), 1.0)))

    def test_load_raises_when_requested_layer_missing(self, tmp_path: Path) -> None:
        """If layer_idx is not in layer_activations, fail fast."""
        sv = _make_multi_layer_vector()
        vector_path = tmp_path / "vec.pt"
        torch.save({"vector": sv}, vector_path)

        with pytest.raises(ValueError, match="layer.*not found"):
            _load_steering_vector(str(vector_path), layer_idx=7)

    def test_load_single_layer_vector_still_works(self, tmp_path: Path) -> None:
        """A single-layer file (no layer_idx needed) loads its only entry."""
        sv = SteeringVector(
            layer_activations={3: torch.full((4,), 7.0)},
            model_name="test-model",
            concept="sentiment",
            method="dim",
        )
        vector_path = tmp_path / "vec.pt"
        torch.save({"vector": sv}, vector_path)

        loaded, _ = _load_steering_vector(str(vector_path), layer_idx=3)
        assert torch.allclose(loaded, _unit(torch.full((4,), 7.0)))

    def test_load_raw_tensor_file_ignores_layer_idx(self, tmp_path: Path) -> None:
        """A raw tensor file (no SteeringVector wrapper) loads without layer_idx."""
        raw = torch.full((4,), 3.0)
        vector_path = tmp_path / "vec.pt"
        torch.save(raw, vector_path)

        loaded, _ = _load_steering_vector(str(vector_path), layer_idx=cast(int, None))
        assert torch.allclose(loaded, _unit(raw))


# ---------------------------------------------------------------------------
# compute_avg_activation_norm padding mask
# ---------------------------------------------------------------------------


class _FakeModelForNorm:
    """Model stub returning controlled activations whose padding rows are 0."""

    def __init__(self, activations: torch.Tensor, tokenizer: object) -> None:
        self._activations = activations
        self.tokenizer = tokenizer

    def get_activations(self, texts: list[str], layers: list[int]) -> dict[int, torch.Tensor]:
        del texts, layers
        return {0: self._activations}


class TestComputeAvgActivationNormMasksPadding:
    """Padding positions must not contribute to the average activation norm."""

    def test_padding_excluded_from_norm(self) -> None:
        """Two prompts of different lengths: padded tokens contribute 0 to mean."""
        # Reuse the project-wide FakeTokenizer from tests/conftest.py instead
        # of a local stub, so any future fix to its padding logic propagates.
        from conftest import FakeTokenizer

        # Batch of 2 prompts: prompt A has 3 real tokens, prompt B has 2.
        # Padding = 1 column on row B.
        # Real-token norms all = 4.0, padded norm = 99.0.
        # Without masking, mean would be inflated by the 99.
        real_norm_value = 4.0
        padded_norm_value = 99.0

        def make_row(real_tokens: int, total_tokens: int) -> torch.Tensor:
            row = torch.full((total_tokens, 4), 0.0)
            for i in range(real_tokens):
                row[i, :] = real_norm_value / (4**0.5)
            for i in range(real_tokens, total_tokens):
                row[i, :] = padded_norm_value / (4**0.5)
            return row

        activations = torch.stack([make_row(3, 3), make_row(2, 3)])  # shape (2, 3, 4)
        model = cast("object", _FakeModelForNorm(activations, FakeTokenizer()))

        result = compute_avg_activation_norm(model, ["abc", "ab"], 0)  # type: ignore[arg-type]

        # 5 real tokens, each contributing a per-token norm of 4.0.
        assert result == pytest.approx(4.0, abs=1e-5)


# ---------------------------------------------------------------------------
# _evaluate_concept HarmBench per-response records
# ---------------------------------------------------------------------------


class _StubHarmBenchEvaluator:
    """Stub evaluator returning a fixed HarmBenchResult."""

    def __init__(self, result: HarmBenchResult) -> None:
        self._result = result

    async def evaluate(self, completions: dict[str, list[dict[str, str]]]) -> HarmBenchResult:
        del completions
        return self._result


class TestEvaluateConceptHarmBenchRecords:
    """HarmBench path must populate per-response labels + reasoning."""

    def test_records_carry_per_behavior_labels(self) -> None:
        """Each ConceptResponse mirrors the matching HarmBenchPrediction."""
        predictions = [
            HarmBenchPrediction(
                behavior_id="b1",
                behavior="prompt-1",
                generation="gen-1",
                label=1,
                reasoning="harmful: made weapon",
            ),
            HarmBenchPrediction(
                behavior_id="b2",
                behavior="prompt-2",
                generation="gen-2",
                label=0,
                reasoning="safe: refused",
            ),
        ]
        result = HarmBenchResult(
            asr=50.0,
            total=2,
            harmful=1,
            safe=1,
            unknown=0,
            predictions=predictions,
        )
        evaluator = _StubHarmBenchEvaluator(result)

        asr, records = _evaluate_concept(
            concept="refusal",
            prompts=["prompt-1", "prompt-2"],
            generations=["gen-1", "gen-2"],
            hb_evaluator=cast("object", evaluator),  # type: ignore[arg-type]
            hb_behavior_ids=["b1", "b2"],
        )

        assert asr == 50.0
        assert len(records) == 2
        assert records[0]["concept_score"] == 1.0
        assert records[0]["reasoning"] == "harmful: made weapon"
        assert records[1]["concept_score"] == 0.0
        assert records[1]["reasoning"] == "safe: refused"

    def test_missing_prediction_falls_back_to_zero_score(self) -> None:
        """Behavior IDs without a classifier output are recorded with score 0."""
        predictions = [
            HarmBenchPrediction(
                behavior_id="b1",
                behavior="prompt-1",
                generation="gen-1",
                label=1,
                reasoning="harmful",
            ),
        ]
        result = HarmBenchResult(
            asr=100.0,
            total=1,
            harmful=1,
            safe=0,
            unknown=0,
            predictions=predictions,
        )
        evaluator = _StubHarmBenchEvaluator(result)

        _, records = _evaluate_concept(
            concept="refusal",
            prompts=["prompt-1", "prompt-2"],
            generations=["gen-1", "gen-2"],
            hb_evaluator=cast("object", evaluator),  # type: ignore[arg-type]
            hb_behavior_ids=["b1", "missing-b2"],
        )

        assert len(records) == 2
        assert records[1]["concept_score"] == 0.0
        assert records[1]["reasoning"] == ""
        # Raw text is still preserved for later inspection.
        assert records[1]["generated_text"] == "gen-2"


# ---------------------------------------------------------------------------
# Per-cell seeded RNG for MMLU-Pro fallback
# ---------------------------------------------------------------------------


class TestCellRngIsDeterministicPerCellIdentity:
    """Fallback RNG must be reproducible from (seed, multiplier, prefix_length)."""

    def test_same_cell_identity_yields_same_rng_sequence(self) -> None:
        """Re-running the same cell in isolation reproduces the same fallbacks."""
        from steering_geometry.strength_prefix_ablation import _cell_rng

        rng1 = _cell_rng(seed=42, multiplier=1.0, prefix_length=5)
        rng2 = _cell_rng(seed=42, multiplier=1.0, prefix_length=5)
        seq1 = [rng1.choice("ABCDEFGHIJ") for _ in range(20)]
        seq2 = [rng2.choice("ABCDEFGHIJ") for _ in range(20)]
        assert seq1 == seq2

    def test_different_cell_identity_yields_different_rng_sequence(self) -> None:
        """Two cells with different prefix_length (or multiplier) get distinct streams."""
        from steering_geometry.strength_prefix_ablation import _cell_rng

        rng_a = _cell_rng(seed=42, multiplier=1.0, prefix_length=5)
        rng_b = _cell_rng(seed=42, multiplier=1.0, prefix_length=10)
        seq_a = [rng_a.choice("ABCDEFGHIJ") for _ in range(20)]
        seq_b = [rng_b.choice("ABCDEFGHIJ") for _ in range(20)]
        assert seq_a != seq_b

    def test_cell_rng_stable_across_python_runs(self) -> None:
        """The seed derivation must NOT depend on hash()-randomized str hashing."""
        from steering_geometry.strength_prefix_ablation import _cell_rng

        # Recompute the expected int manually using sha256 to confirm stability.
        key = b"42:1.0:5"
        expected_int = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        rng = _cell_rng(seed=42, multiplier=1.0, prefix_length=5)
        assert rng.random() == random.Random(expected_int).random()
