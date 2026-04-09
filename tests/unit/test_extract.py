"""Tests for unified extraction module."""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from steering_geometry.config import ConceptConfig, ExtractionConfig, ModelConfig
from steering_geometry.extract import (
    extract_steering_vector,
    load_contrast_pairs,
    load_polite_data,
    load_refusal_data,
    load_sentiment_data,
)
from steering_geometry.models import HookedModel
from steering_geometry.types import ContrastPair, SteeringVector

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

    def test_new_field_defaults(self) -> None:
        """New fields should have correct defaults."""
        config = ExtractionConfig()
        assert config.data_mode == "prompt_only"
        assert config.token_select == "default"
        assert config.last_n == 1
        assert config.seed == 42

    def test_new_fields_custom_values(self) -> None:
        """New fields should accept custom values."""
        config = ExtractionConfig(
            data_mode="prompt_response",
            token_select="last_n",
            last_n=10,
            seed=123,
        )
        assert config.data_mode == "prompt_response"
        assert config.token_select == "last_n"
        assert config.last_n == 10
        assert config.seed == 123

    def test_new_fields_with_existing_fields(self) -> None:
        """New fields should coexist with existing fields."""
        config = ExtractionConfig(
            method="pca",
            batch_size=16,
            read_token_index=0,
            layers=[0.5],
            data_mode="prompt_response",
            token_select="last_n",
            last_n=5,
            seed=99,
        )
        assert config.method == "pca"
        assert config.batch_size == 16
        assert config.read_token_index == 0
        assert config.layers == [0.5]
        assert config.data_mode == "prompt_response"
        assert config.token_select == "last_n"
        assert config.last_n == 5
        assert config.seed == 99


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


def _passthrough_sample(items: list, n: int, **kwargs: object) -> list:
    return items[:n]


def _make_benign_rows(n: int) -> list[dict]:
    """Create benign-dataset rows: prompt, response, refusal."""
    return [
        {
            "prompt": f"benign prompt {i}",
            "response": f"benign response {i}",
            "refusal": "I cannot comply.",
        }
        for i in range(n)
    ]


def _make_harmful_rows(n: int) -> list[dict]:
    """Create harmful-dataset rows: prompt, chosen (refusal), rejected (compliance)."""
    return [
        {"prompt": f"harmful prompt {i}", "chosen": f"I refuse {i}", "rejected": f"I comply {i}"}
        for i in range(n)
    ]


def _mock_dual_dataset(benign_rows: list[dict], harmful_rows: list[dict]) -> MagicMock:
    """Create a mock load_dataset that returns correct dataset based on name."""
    mock = MagicMock()

    def _load(name: str, **kwargs: object) -> dict:
        if name == "LLM-LAT/benign-dataset":
            return {"train": benign_rows}
        if name == "LLM-LAT/harmful-dataset":
            return {"train": harmful_rows}
        return {}

    mock.side_effect = _load
    return mock


class TestRefusalDualDataset:
    """Tests for the dual-dataset refusal loader."""

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_load_refusal_dual_dataset(self, mock_load: MagicMock, mock_sample: MagicMock) -> None:
        mock_load.side_effect = _mock_dual_dataset(
            _make_benign_rows(20), _make_harmful_rows(20)
        ).side_effect
        config = ConceptConfig(concept_name="refusal", dataset_name="dual", num_pairs=10)

        pairs = load_refusal_data(config)

        assert len(pairs) == 10
        for pair in pairs:
            assert pair.positive.startswith("benign prompt")
            assert pair.negative.startswith("harmful prompt")
            assert pair.metadata["concept"] == "refusal"
            assert pair.metadata["source"] == "LLM-LAT/benign-dataset+LLM-LAT/harmful-dataset"

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_load_refusal_prompt_only(self, mock_load: MagicMock, mock_sample: MagicMock) -> None:
        mock_load.side_effect = _mock_dual_dataset(
            _make_benign_rows(20), _make_harmful_rows(20)
        ).side_effect
        config = ConceptConfig(concept_name="refusal", dataset_name="dual", num_pairs=5)

        pairs = load_refusal_data(config, data_mode="prompt_only")

        assert len(pairs) == 5
        for pair in pairs:
            assert "\n" not in pair.positive
            assert "\n" not in pair.negative

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_load_refusal_prompt_response(
        self, mock_load: MagicMock, mock_sample: MagicMock
    ) -> None:
        mock_load.side_effect = _mock_dual_dataset(
            _make_benign_rows(20), _make_harmful_rows(20)
        ).side_effect
        config = ConceptConfig(concept_name="refusal", dataset_name="dual", num_pairs=5)

        pairs = load_refusal_data(config, data_mode="prompt_response")

        assert len(pairs) == 5
        for pair in pairs:
            assert "\n" in pair.positive
            assert "\n" in pair.negative
            parts_p = pair.positive.split("\n")
            assert parts_p[0].startswith("benign prompt")
            assert parts_p[1].startswith("benign response")

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_load_refusal_filter(self, mock_load: MagicMock, mock_sample: MagicMock) -> None:
        benign = [
            {"prompt": "keep this", "response": "good response", "refusal": "I cannot comply."},
            {
                "prompt": "filter this",
                "response": "I cannot comply.",
                "refusal": "I cannot comply.",
            },
            {"prompt": "", "response": "empty prompt", "refusal": ""},
            {"prompt": "also keep", "response": "another good", "refusal": "no match"},
        ]
        mock_load.side_effect = _mock_dual_dataset(benign, _make_harmful_rows(10)).side_effect
        config = ConceptConfig(concept_name="refusal", dataset_name="dual", num_pairs=10)

        pairs = load_refusal_data(config, data_mode="prompt_only")

        assert len(pairs) == 2
        positives = [p.positive for p in pairs]
        assert "keep this" in positives
        assert "also keep" in positives
        assert "filter this" not in positives

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_load_refusal_cap(self, mock_load: MagicMock, mock_sample: MagicMock) -> None:
        mock_load.side_effect = _mock_dual_dataset(
            _make_benign_rows(5), _make_harmful_rows(3)
        ).side_effect
        config = ConceptConfig(concept_name="refusal", dataset_name="dual", num_pairs=100)

        pairs = load_refusal_data(config)

        assert len(pairs) == 3

    @patch("steering_geometry.extract.sample_with_seed", side_effect=_passthrough_sample)
    @patch("steering_geometry.extract.load_dataset")
    def test_load_refusal_seed_deterministic(
        self, mock_load: MagicMock, mock_sample: MagicMock
    ) -> None:
        mock_load.side_effect = _mock_dual_dataset(
            _make_benign_rows(20), _make_harmful_rows(20)
        ).side_effect
        config = ConceptConfig(concept_name="refusal", dataset_name="dual", num_pairs=5)

        pairs_a = load_refusal_data(config, seed=99)
        pairs_b = load_refusal_data(config, seed=99)

        for a, b in zip(pairs_a, pairs_b, strict=True):
            assert a.positive == b.positive
            assert a.negative == b.negative


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


class TestTokenSelectDispatch:
    """Tests for token selection dispatch in extract_steering_vector."""

    def test_extract_with_token_select_all(self, mock_hooked_model: HookedModel) -> None:
        """token_select='all' should dispatch to select all non-padding tokens."""
        pairs = [
            ContrastPair(positive="aaa", negative="bbb", metadata={"concept": "test"}),
            ContrastPair(positive="ccc", negative="ddd", metadata={"concept": "test"}),
        ]
        config = ExtractionConfig(token_select="all", layers=[0.5], method="mean")
        result = extract_steering_vector(mock_hooked_model, pairs, config)
        assert isinstance(result, SteeringVector)
        assert result.concept == "test"
        assert len(result.layer_activations) > 0

    def test_extract_with_token_select_last_n(self, mock_hooked_model: HookedModel) -> None:
        """token_select='last_n' should dispatch to select last N tokens."""
        pairs = [
            ContrastPair(positive="aaa", negative="bbb", metadata={"concept": "test"}),
            ContrastPair(positive="ccc", negative="ddd", metadata={"concept": "test"}),
        ]
        config = ExtractionConfig(token_select="last_n", last_n=3, layers=[0.5], method="mean")
        result = extract_steering_vector(mock_hooked_model, pairs, config)
        assert isinstance(result, SteeringVector)
        assert result.concept == "test"
        assert len(result.layer_activations) > 0

    def test_extract_default_backward_compat(self, mock_hooked_model: HookedModel) -> None:
        """Default config should work and dispatch to 'all' mode."""
        pairs = [
            ContrastPair(positive="aaa", negative="bbb", metadata={"concept": "test"}),
            ContrastPair(positive="ccc", negative="ddd", metadata={"concept": "test"}),
        ]
        config = ExtractionConfig(layers=[0.5], method="mean")
        result = extract_steering_vector(mock_hooked_model, pairs, config)
        assert isinstance(result, SteeringVector)
        assert result.concept == "test"
        assert len(result.layer_activations) > 0

    def test_extract_legacy_token_select(self, mock_hooked_model: HookedModel) -> None:
        """An unknown token_select value should fall through to the int-index path."""
        pairs = [
            ContrastPair(positive="aaa", negative="bbb", metadata={"concept": "test"}),
            ContrastPair(positive="ccc", negative="ddd", metadata={"concept": "test"}),
        ]
        config = ExtractionConfig(
            token_select="legacy",
            layers=[0.5],
            method="mean",
        )
        result = extract_steering_vector(mock_hooked_model, pairs, config)
        assert isinstance(result, SteeringVector)
        assert result.concept == "test"


class TestRefusalIntegration:
    """End-to-end integration tests for refusal extraction with all 4 strategy combos."""

    @pytest.fixture
    def mock_pairs(self) -> list[ContrastPair]:
        """Create refusal contrast pairs for integration testing."""
        return [
            ContrastPair(
                positive=f"good query {i}",
                negative=f"evil query {i}",
                metadata={"concept": "refusal", "dataset": "dual"},
            )
            for i in range(5)
        ]

    def test_integration_prompt_only_all(
        self, mock_hooked_model: HookedModel, mock_pairs: list[ContrastPair]
    ) -> None:
        """prompt_only + all tokens produces valid SteeringVector."""
        config = ExtractionConfig(
            token_select="all",
            data_mode="prompt_only",
            layers=[0.5],
            method="mean",
        )
        result = extract_steering_vector(mock_hooked_model, mock_pairs, config)
        assert isinstance(result, SteeringVector)
        assert result.concept == "refusal"
        assert len(result.layer_activations) > 0
        for _layer, vec in result.layer_activations.items():
            assert vec.ndim == 1

    def test_integration_prompt_only_last_n(
        self, mock_hooked_model: HookedModel, mock_pairs: list[ContrastPair]
    ) -> None:
        """prompt_only + last_n tokens produces valid SteeringVector."""
        config = ExtractionConfig(
            token_select="last_n",
            last_n=2,
            data_mode="prompt_only",
            layers=[0.5],
            method="mean",
        )
        result = extract_steering_vector(mock_hooked_model, mock_pairs, config)
        assert isinstance(result, SteeringVector)
        assert result.concept == "refusal"
        for vec in result.layer_activations.values():
            assert vec.ndim == 1

    def test_integration_prompt_response_all(
        self, mock_hooked_model: HookedModel, mock_pairs: list[ContrastPair]
    ) -> None:
        """prompt_response + all tokens produces valid SteeringVector."""
        config = ExtractionConfig(
            token_select="all",
            data_mode="prompt_response",
            layers=[0.5],
            method="mean",
        )
        result = extract_steering_vector(mock_hooked_model, mock_pairs, config)
        assert isinstance(result, SteeringVector)

    def test_integration_prompt_response_last_n(
        self, mock_hooked_model: HookedModel, mock_pairs: list[ContrastPair]
    ) -> None:
        """prompt_response + last_n tokens produces valid SteeringVector."""
        config = ExtractionConfig(
            token_select="last_n",
            last_n=3,
            data_mode="prompt_response",
            layers=[0.5],
            method="mean",
        )
        result = extract_steering_vector(mock_hooked_model, mock_pairs, config)
        assert isinstance(result, SteeringVector)
