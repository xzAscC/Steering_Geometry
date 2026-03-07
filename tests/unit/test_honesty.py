from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from steering_geometry.concepts import honesty
from steering_geometry.config import ConceptConfig, EvaluationConfig
from steering_geometry.models import HookedModel
from steering_geometry.types import SteeringVector


def test_load_honesty_data_creates_requested_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_rows = [
        {"question": "What is the capital of France?"},
        {"question": "How many planets are in the solar system?"},
        {"question": "What is the largest mammal?"},
    ]

    def fake_load_dataset(name: str, subset: str) -> dict[str, list[dict[str, str]]]:
        assert name == "truthfulqa/truthful_qa"
        assert subset == "generation"
        return {"validation": mock_rows}

    monkeypatch.setattr(honesty, "load_dataset", fake_load_dataset)

    config = ConceptConfig(concept_name="honesty", dataset_name="truthfulqa", num_pairs=2)
    pairs = honesty.load_honesty_data(config)

    assert len(pairs) == 2
    assert all(pair.metadata["concept"] == "honesty" for pair in pairs)
    assert all(pair.metadata["dataset"] == "truthfulqa" for pair in pairs)
    assert all(honesty._HONEST_PREFIX in pair.positive for pair in pairs)
    assert all(honesty._DISHONEST_PREFIX in pair.negative for pair in pairs)


def test_load_honesty_data_rejects_invalid_pair_count() -> None:
    config = ConceptConfig(concept_name="honesty", dataset_name="truthfulqa", num_pairs=0)

    with pytest.raises(ValueError, match="num_pairs"):
        _ = honesty.load_honesty_data(config)


def test_evaluate_honesty_measures_shift(monkeypatch: pytest.MonkeyPatch) -> None:
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
                return "This is true and correct fact"
            return "This is false and wrong believe"

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

    monkeypatch.setattr(honesty, "apply_steering_vector", fake_apply_steering_vector)

    model = DummyModel()
    vector = SteeringVector(
        layer_activations={},
        model_name="dummy-model",
        concept="honesty",
        method="mean",
    )

    result = honesty.evaluate_honesty(
        cast(HookedModel, cast(object, model)),
        vector,
        EvaluationConfig(num_samples=4, seed=42),
    )

    assert result.concept == "honesty"
    assert result.model_name == "dummy-model"
    assert result.scores["honesty_shift"] > 0
    assert result.scores["steered_honest_rate"] > result.scores["baseline_honest_rate"]
