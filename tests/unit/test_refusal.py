from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from steering_geometry.concepts import refusal
from steering_geometry.config import ConceptConfig, EvaluationConfig
from steering_geometry.models import HookedModel
from steering_geometry.types import SteeringVector


def test_load_refusal_data_creates_requested_pairs() -> None:
    config = ConceptConfig(concept_name="refusal", dataset_name="synthetic_repe", num_pairs=5)
    pairs = refusal.load_refusal_data(config)

    assert len(pairs) == 5
    assert all(pair.metadata["concept"] == "refusal" for pair in pairs)
    assert all(pair.metadata["dataset"] == "synthetic_repe" for pair in pairs)
    assert all(refusal.REFUSAL_PREFIX in pair.positive for pair in pairs)
    assert all(refusal.COMPLIANCE_PREFIX in pair.negative for pair in pairs)


def test_load_refusal_data_rejects_invalid_pair_count() -> None:
    config = ConceptConfig(concept_name="refusal", dataset_name="synthetic_repe", num_pairs=0)

    with pytest.raises(ValueError, match="num_pairs"):
        _ = refusal.load_refusal_data(config)


def test_load_refusal_data_uses_sanitized_placeholders() -> None:
    config = ConceptConfig(concept_name="refusal", dataset_name="synthetic_repe", num_pairs=10)
    pairs = refusal.load_refusal_data(config)

    for pair in pairs:
        assert "[request_placeholder_" in pair.positive
        assert "[request_placeholder_" in pair.negative


def test_evaluate_refusal_measures_shift(monkeypatch: pytest.MonkeyPatch) -> None:
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
                return "I cannot help with that request."
            return "The capital of France is Paris."

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

    monkeypatch.setattr(refusal, "apply_steering_vector", fake_apply_steering_vector)

    model = DummyModel()
    vector = SteeringVector(
        layer_activations={},
        model_name="dummy-model",
        concept="refusal",
        method="mean",
    )

    result = refusal.evaluate_refusal(
        cast(HookedModel, cast(object, model)),
        vector,
        EvaluationConfig(num_samples=4, seed=42),
    )

    assert result.concept == "refusal"
    assert result.model_name == "dummy-model"
    assert result.scores["refusal_shift"] > 0
    assert result.scores["steered_refusal_rate"] > result.scores["baseline_refusal_rate"]
    assert result.scores["baseline_refusal_rate"] == 0.0
    assert result.scores["steered_refusal_rate"] == 1.0
