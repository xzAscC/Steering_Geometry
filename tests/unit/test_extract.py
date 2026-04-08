"""Tests for unified extraction module."""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from steering_geometry.config import ConceptConfig, ExtractionConfig, ModelConfig
from steering_geometry.extract import (
    load_contrast_pairs,
    load_polite_data,
    load_refusal_data,
    load_sentiment_data,
)

HAS_ACCELERATE = importlib.util.find_spec("accelerate") is not None


class _TrackingIterable:
    """Iterable that tracks how many items were consumed."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items
        self.consumed = 0

    def __iter__(self) -> "_TrackingIterator":
        return _TrackingIterator(self)


class _TrackingIterator:
    """Iterator that counts consumed items."""

    def __init__(self, parent: _TrackingIterable) -> None:
        self._parent = parent
        self._index = 0

    def __iter__(self) -> "_TrackingIterator":
        return self

    def __next__(self) -> dict:
        if self._index >= len(self._parent._items):
            raise StopIteration
        item = self._parent._items[self._index]
        self._index += 1
        self._parent.consumed += 1
        return item


class TestLoadContrastPairs:
    def test_invalid_concept_raises(self) -> None:
        """Invalid concept should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid concept"):
            load_contrast_pairs("invalid_concept", num_pairs=10)

    def test_zero_pairs_raises(self) -> None:
        """Zero pairs should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            load_contrast_pairs("sentiment", num_pairs=0)

    def test_negative_pairs_raises(self) -> None:
        """Negative pairs should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            load_contrast_pairs("sentiment", num_pairs=-1)


class TestDatasetLoaders:
    """Tests for individual dataset loaders."""

    def test_load_sentiment_data(self) -> None:
        """Sentiment data should load from SST-2."""
        config = ConceptConfig(
            concept_name="sentiment",
            dataset_name="sst2",
            num_pairs=10,
        )
        pairs = load_sentiment_data(config)
        assert len(pairs) == 10
        for pair in pairs:
            assert pair.metadata["concept"] == "sentiment"
            assert pair.positive
            assert pair.negative

    def test_load_refusal_data(self) -> None:
        """Refusal data should load from LLM-LAT dataset."""
        config = ConceptConfig(
            concept_name="refusal",
            dataset_name="harmful",
            num_pairs=10,
        )
        pairs = load_refusal_data(config)
        assert len(pairs) == 10
        for pair in pairs:
            assert pair.metadata["concept"] == "refusal"
            assert "refuse" in pair.positive.lower() or "refusal" in pair.positive.lower()
            assert "comply" in pair.negative.lower() or "compliance" in pair.negative.lower()


@pytest.mark.skipif(not HAS_ACCELERATE, reason="requires accelerate package")
class TestHookedModel:
    """Tests for HookedModel (requires accelerate)."""

    def test_model_loads(self) -> None:
        """Model should load successfully."""
        from steering_geometry.models import HookedModel

        config = ModelConfig(model_name="sshleifer/tiny-gpt2")
        model = HookedModel(config)
        assert model.config.model_name == "sshleifer/tiny-gpt2"
        assert model.num_layers > 0

    def test_resolve_layers(self) -> None:
        """Layer resolution should work correctly."""
        from steering_geometry.models import HookedModel

        config = ModelConfig(model_name="sshleifer/tiny-gpt2")
        model = HookedModel(config)
        n_layers = model.num_layers

        layers = model.resolve_layers([0.0, 0.5, 1.0])
        assert layers[0] == 0
        assert layers[1] == n_layers // 2
        assert layers[2] == n_layers - 1

    def test_get_activations(self) -> None:
        """Activation extraction should work."""
        from steering_geometry.models import HookedModel

        config = ModelConfig(model_name="sshleifer/tiny-gpt2")
        model = HookedModel(config)
        layers = [0, 1]

        activations = model.get_activations(["Hello world"], layers)

        assert 0 in activations
        assert 1 in activations
        assert activations[0].shape[0] == 1


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_default_values(self) -> None:
        """Default values should be set correctly."""
        config = ExtractionConfig()
        assert config.method == "mean"
        assert config.batch_size == 8
        assert config.read_token_index == -1
        assert len(config.layers) == 5

    def test_custom_values(self) -> None:
        """Custom values should be set correctly."""
        config = ExtractionConfig(
            method="pca",
            batch_size=16,
            read_token_index=0,
            layers=[0.5],
        )
        assert config.method == "pca"
        assert config.batch_size == 16
        assert config.read_token_index == 0
        assert config.layers == [0.5]


class TestPoliteLoader:
    """Tests for polite data loader."""

    def test_load_polite_data(self) -> None:
        """Polite data should load from Intel/polite-guard."""
        config = ConceptConfig(
            concept_name="polite",
            dataset_name="polite",
            num_pairs=10,
        )
        pairs = load_polite_data(config)
        assert len(pairs) == 10
        for pair in pairs:
            assert pair.metadata["concept"] == "polite"
            assert pair.metadata["dataset"] == "polite"
            assert pair.metadata["source"] == "Intel/polite-guard"
            assert pair.positive  # polite text (label=1)
            assert pair.negative  # impolite text (label=0)

    def test_load_polite_data_returns_correct_count(self) -> None:
        """Polite data should return the requested number of pairs."""
        config = ConceptConfig(
            concept_name="polite",
            dataset_name="polite",
            num_pairs=5,
        )
        pairs = load_polite_data(config)
        assert len(pairs) == 5

    def test_polite_uses_direct_text_no_prefix(self) -> None:
        """Polite loader should use original text directly, no prefix."""
        config = ConceptConfig(
            concept_name="polite",
            dataset_name="polite",
            num_pairs=3,
        )
        pairs = load_polite_data(config)
        # Unlike sycophancy, polite uses direct text like sentiment
        # Check that text doesn't have artificial prefixes
        for pair in pairs:
            # Should be original text, not prefixed with instructions
            assert not pair.positive.startswith("Pretend")
            assert not pair.negative.startswith("Pretend")


def _make_sentiment_rows(n: int) -> list[dict]:
    """Create balanced positive/negative sentiment rows."""
    rows: list[dict] = []
    for i in range(n):
        rows.append({"sentence": f"good sentence {i}", "label": 1})
        rows.append({"sentence": f"bad sentence {i}", "label": 0})
    return rows


def _make_polite_rows(n: int) -> list[dict]:
    """Create balanced polite/impolite rows."""
    rows: list[dict] = []
    for i in range(n):
        rows.append({"text": f"kind text {i}", "label": "polite"})
        rows.append({"text": f"rude text {i}", "label": "impolite"})
    return rows


def _passthrough_sample(items: list, n: int) -> list:
    """Passthrough for sample_with_seed that returns first n items."""
    return items[:n]


class TestEarlyStop:
    """Tests for early-stop behavior in dataset loaders."""

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_sentiment_early_stop(self, mock_load: MagicMock, mock_sample: MagicMock) -> None:
        """load_sentiment_data should stop iteration early, not consume all rows."""
        tracking = _TrackingIterable(_make_sentiment_rows(10000))
        mock_load.return_value = {"train": tracking}
        config = ConceptConfig(concept_name="sentiment", dataset_name="sst2", num_pairs=10)

        pairs = load_sentiment_data(config)

        assert len(pairs) == 10
        assert tracking.consumed < 20000
        assert tracking.consumed <= 100

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_polite_early_stop(self, mock_load: MagicMock, mock_sample: MagicMock) -> None:
        """load_polite_data should stop iteration early, not consume all rows."""
        tracking = _TrackingIterable(_make_polite_rows(10000))
        mock_load.return_value = tracking
        config = ConceptConfig(concept_name="polite", dataset_name="polite", num_pairs=10)

        pairs = load_polite_data(config)

        assert len(pairs) == 10
        assert tracking.consumed < 20000
        assert tracking.consumed <= 100

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_sentiment_sample_with_seed_called(
        self, mock_load: MagicMock, mock_sample: MagicMock
    ) -> None:
        """load_sentiment_data should call sample_with_seed for determinism."""
        mock_load.return_value = {"train": _make_sentiment_rows(50)}
        config = ConceptConfig(concept_name="sentiment", dataset_name="sst2", num_pairs=5)

        load_sentiment_data(config)

        assert mock_sample.call_count >= 2

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_polite_sample_with_seed_called(
        self, mock_load: MagicMock, mock_sample: MagicMock
    ) -> None:
        """load_polite_data should call sample_with_seed for determinism."""
        mock_load.return_value = _make_polite_rows(50)
        config = ConceptConfig(concept_name="polite", dataset_name="polite", num_pairs=5)

        load_polite_data(config)

        assert mock_sample.call_count >= 2
