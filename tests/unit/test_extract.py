"""Tests for unified extraction module."""

import importlib.util

import pytest

from steering_geometry.config import ConceptConfig, ExtractionConfig, ModelConfig
from steering_geometry.extract import (
    load_contrast_pairs,
    load_polite_data,
    load_refusal_data,
    load_sentiment_data,
)

HAS_ACCELERATE = importlib.util.find_spec("accelerate") is not None


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
        """Polite data should load from Cleanlab/stanford-politeness."""
        config = ConceptConfig(
            concept_name="polite",
            dataset_name="politeness",
            num_pairs=10,
        )
        pairs = load_polite_data(config)
        assert len(pairs) == 10
        for pair in pairs:
            assert pair.metadata["concept"] == "polite"
            assert pair.metadata["dataset"] == "politeness"
            assert pair.metadata["source"] == "Cleanlab/stanford-politeness"
            assert pair.positive  # polite text (label=1)
            assert pair.negative  # impolite text (label=0)

    def test_load_polite_data_returns_correct_count(self) -> None:
        """Polite data should return the requested number of pairs."""
        config = ConceptConfig(
            concept_name="polite",
            dataset_name="politeness",
            num_pairs=5,
        )
        pairs = load_polite_data(config)
        assert len(pairs) == 5

    def test_polite_uses_direct_text_no_prefix(self) -> None:
        """Polite loader should use original text directly, no prefix."""
        config = ConceptConfig(
            concept_name="polite",
            dataset_name="politeness",
            num_pairs=3,
        )
        pairs = load_polite_data(config)
        # Unlike sycophancy, polite uses direct text like sentiment
        # Check that text doesn't have artificial prefixes
        for pair in pairs:
            # Should be original text, not prefixed with instructions
            assert not pair.positive.startswith("Pretend")
            assert not pair.negative.startswith("Pretend")
