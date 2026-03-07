from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from steering_geometry.concepts import sycophancy
from steering_geometry.config import ConceptConfig, EvaluationConfig
from steering_geometry.models import HookedModel
from steering_geometry.types import SteeringVector


def test_load_sycophancy_data_creates_requested_pairs() -> None:
    config = ConceptConfig(concept_name="sycophancy", dataset_name="synthetic", num_pairs=5)
    pairs = sycophancy.load_sycophancy_data(config)

    assert len(pairs) == 5
    assert all(pair.metadata["concept"] == "sycophancy" for pair in pairs)
    assert all(pair.metadata["dataset"] == "synthetic" for pair in pairs)
    assert all(pair.metadata["source"] == "synthetic" for pair in pairs)
    assert all("statement" in pair.metadata for pair in pairs)

    for pair in pairs:
        assert "sycophantic assistant" in pair.positive.lower()
        assert "objective assistant" in pair.negative.lower()


def test_load_sycophancy_data_rejects_invalid_pair_count() -> None:
    config = ConceptConfig(concept_name="sycophancy", dataset_name="synthetic", num_pairs=0)

    with pytest.raises(ValueError, match="num_pairs"):
        _ = sycophancy.load_sycophancy_data(config)


def test_evaluate_sycophancy_measures_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyModel:
        config: SimpleNamespace
        steered: bool

        def __init__(self) -> None:
            self.config = SimpleNamespace(model_name="dummy-model")
            self.steered = False

        def generate(self, prompt: str, max_new_tokens: int) -> str:
            del prompt
            del max_new_tokens
            if self.steered:
                return "yes absolutely agree correct"
            return "no actually that's wrong"

    @contextmanager
    def fake_apply_steering_vector(
        model: DummyModel,
        vector: SteeringVector,
        scale: float = 1.0,
    ) -> Generator[None, None, None]:
        del vector
        del scale
        model.steered = True
        try:
            yield
        finally:
            model.steered = False

    monkeypatch.setattr(sycophancy, "apply_steering_vector", fake_apply_steering_vector)

    model = DummyModel()
    vector = SteeringVector(
        layer_activations={},
        model_name="dummy-model",
        concept="sycophancy",
        method="mean",
    )

    result = sycophancy.evaluate_sycophancy(
        cast(HookedModel, cast(object, model)),
        vector,
        EvaluationConfig(num_samples=4, seed=42),
    )

    assert result.concept == "sycophancy"
    assert result.model_name == "dummy-model"
    assert result.scores["sycophancy_shift"] > 0
    assert result.scores["steered_agreement_rate"] > result.scores["baseline_agreement_rate"]
