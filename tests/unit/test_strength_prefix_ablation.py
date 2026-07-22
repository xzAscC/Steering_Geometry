"""Tests for the strength × prefix-length ablation module."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from steering_geometry.strength_prefix_ablation import _load_steering_vector
from steering_geometry.types import SteeringVector

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
