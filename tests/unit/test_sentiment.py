from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from steering_geometry.concepts import sentiment
from steering_geometry.config import ConceptConfig, EvaluationConfig
from steering_geometry.models import HookedModel
from steering_geometry.types import SteeringVector


def test_load_sentiment_data_creates_requested_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_rows = [
        {"sentence": "A wonderful movie", "label": 1},
        {"sentence": "Great acting", "label": 1},
        {"sentence": "Terrible pacing", "label": 0},
        {"sentence": "Bad script", "label": 0},
    ]

    def fake_load_dataset(name: str, subset: str) -> dict[str, list[dict[str, str | int]]]:
        assert name == "glue"
        assert subset == "sst2"
        return {"train": mock_rows}

    monkeypatch.setattr(sentiment, "load_dataset", fake_load_dataset)

    config = ConceptConfig(concept_name="sentiment", dataset_name="sst2", num_pairs=2)
    pairs = sentiment.load_sentiment_data(config)

    assert len(pairs) == 2
    assert all(pair.metadata["concept"] == "sentiment" for pair in pairs)
    assert all(pair.metadata["dataset"] == "sst2" for pair in pairs)
    positive_texts = {row["sentence"] for row in mock_rows if row["label"] == 1}
    negative_texts = {row["sentence"] for row in mock_rows if row["label"] == 0}
    assert all(pair.positive in positive_texts for pair in pairs)
    assert all(pair.negative in negative_texts for pair in pairs)


def test_load_sentiment_data_rejects_invalid_pair_count() -> None:
    config = ConceptConfig(concept_name="sentiment", dataset_name="sst2", num_pairs=0)

    with pytest.raises(ValueError, match="num_pairs"):
        _ = sentiment.load_sentiment_data(config)


def test_evaluate_sentiment_measures_shift(monkeypatch: pytest.MonkeyPatch) -> None:
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
                return "good excellent wonderful"
            return "bad terrible awful"

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

    monkeypatch.setattr(sentiment, "apply_steering_vector", fake_apply_steering_vector)

    model = DummyModel()
    vector = SteeringVector(
        layer_activations={},
        model_name="dummy-model",
        concept="sentiment",
        method="mean",
    )

    result = sentiment.evaluate_sentiment(
        cast(HookedModel, cast(object, model)),
        vector,
        EvaluationConfig(num_samples=4, seed=42),
    )

    assert result.concept == "sentiment"
    assert result.model_name == "dummy-model"
    assert result.scores["sentiment_shift"] > 0
    assert result.scores["steered_positive_rate"] > result.scores["baseline_positive_rate"]
