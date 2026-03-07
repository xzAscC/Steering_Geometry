from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from steering_geometry.concepts import toxicity
from steering_geometry.config import ConceptConfig, EvaluationConfig
from steering_geometry.models import HookedModel
from steering_geometry.types import SteeringVector


def test_load_toxicity_data_creates_requested_pairs() -> None:
    config = ConceptConfig(concept_name="toxicity", dataset_name="synthetic", num_pairs=5)
    pairs = toxicity.load_toxicity_data(config)

    assert len(pairs) == 5
    assert all(pair.metadata["concept"] == "toxicity" for pair in pairs)
    assert all(pair.metadata["dataset"] == "synthetic" for pair in pairs)
    assert all(toxicity.RESPECTFUL_PREFIX in pair.positive for pair in pairs)
    assert all(toxicity.TOXIC_PREFIX in pair.negative for pair in pairs)


def test_load_toxicity_data_rejects_invalid_pair_count() -> None:
    config = ConceptConfig(concept_name="toxicity", dataset_name="synthetic", num_pairs=0)

    with pytest.raises(ValueError, match="num_pairs"):
        _ = toxicity.load_toxicity_data(config)


def test_load_toxicity_data_uses_sanitized_placeholders() -> None:
    config = ConceptConfig(concept_name="toxicity", dataset_name="synthetic", num_pairs=10)
    pairs = toxicity.load_toxicity_data(config)

    all_negative_text = " ".join(pair.negative for pair in pairs)
    assert (
        "[offensive_placeholder]" in all_negative_text
        or "[hostile_placeholder]" in all_negative_text
    )


def test_evaluate_toxicity_measures_shift(monkeypatch: pytest.MonkeyPatch) -> None:
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
                return "[offensive_placeholder] [hostile_placeholder]"
            return "respectful helpful polite kind"

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

    monkeypatch.setattr(toxicity, "apply_steering_vector", fake_apply_steering_vector)

    model = DummyModel()
    vector = SteeringVector(
        layer_activations={},
        model_name="dummy-model",
        concept="toxicity",
        method="mean",
    )

    result = toxicity.evaluate_toxicity(
        cast(HookedModel, cast(object, model)),
        vector,
        EvaluationConfig(num_samples=4, seed=42),
    )

    assert result.concept == "toxicity"
    assert result.model_name == "dummy-model"
    assert result.scores["toxicity_shift"] > 0
    assert result.scores["steered_toxic_rate"] > result.scores["baseline_toxic_rate"]
