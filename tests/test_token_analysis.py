"""Tests for token analysis module."""

import pytest
import torch

from steering_geometry.config import TokenAnalysisConfig
from steering_geometry.token_analysis import (
    compute_discriminative_scores,
    select_top_k_tokens,
)
from steering_geometry.types import (
    DiscriminativeTokenResult,
    ProbeExperimentResult,
    ProbeLayerResult,
    TokenRecord,
)


class TestTokenRecord:
    """Tests for TokenRecord dataclass."""

    def test_creation_with_valid_data(self) -> None:
        """TokenRecord should be created with valid data."""
        activation = torch.randn(768)
        record = TokenRecord(
            token_id=1,
            token_text="test",
            activation=activation,
            contrast_pair_idx=0,
            position_in_sequence=0,
            label="positive",
        )
        assert record.token_id == 1
        assert record.token_text == "test"
        assert torch.equal(record.activation, activation)
        assert record.contrast_pair_idx == 0
        assert record.position_in_sequence == 0
        assert record.label == "positive"
        assert record.score == 0.0  # Default value

    def test_score_can_be_set(self) -> None:
        """TokenRecord score should be mutable."""
        record = TokenRecord(
            token_id=1,
            token_text="test",
            activation=torch.randn(768),
            contrast_pair_idx=0,
            position_in_sequence=0,
            label="positive",
            score=5.0,
        )
        assert record.score == 5.0


class TestDiscriminativeTokenResult:
    """Tests for DiscriminativeTokenResult dataclass."""

    def test_creation_with_valid_data(self) -> None:
        """DiscriminativeTokenResult should be created with valid data."""
        pos_record = TokenRecord(
            token_id=1,
            token_text="good",
            activation=torch.randn(768),
            contrast_pair_idx=0,
            position_in_sequence=0,
            label="positive",
        )
        neg_record = TokenRecord(
            token_id=2,
            token_text="bad",
            activation=torch.randn(768),
            contrast_pair_idx=0,
            position_in_sequence=0,
            label="negative",
        )
        result = DiscriminativeTokenResult(
            concept="honesty",
            layer=5,
            top_positive=[pos_record],
            top_negative=[neg_record],
        )
        assert result.concept == "honesty"
        assert result.layer == 5
        assert len(result.top_positive) == 1
        assert len(result.top_negative) == 1

    def test_default_empty_lists(self) -> None:
        """DiscriminativeTokenResult should default to empty lists."""
        result = DiscriminativeTokenResult(concept="toxicity", layer=3)
        assert result.top_positive == []
        assert result.top_negative == []


class TestProbeLayerResult:
    """Tests for ProbeLayerResult dataclass."""

    def test_creation_with_valid_data(self) -> None:
        """ProbeLayerResult should be created with valid data."""
        result = ProbeLayerResult(
            layer_idx=5,
            train_accuracy=0.95,
            test_accuracy=0.85,
            auc_score=0.92,
        )
        assert result.layer_idx == 5
        assert result.train_accuracy == 0.95
        assert result.test_accuracy == 0.85
        assert result.auc_score == 0.92


class TestProbeExperimentResult:
    """Tests for ProbeExperimentResult dataclass."""

    def test_creation_with_valid_data(self) -> None:
        """ProbeExperimentResult should be created with valid data."""
        layer_result = ProbeLayerResult(
            layer_idx=5,
            train_accuracy=0.95,
            test_accuracy=0.85,
            auc_score=0.92,
        )
        result = ProbeExperimentResult(
            concept="honesty",
            model_name="test-model",
            tokens_per_class=1000,
            layer_results=[layer_result],
        )
        assert result.concept == "honesty"
        assert result.model_name == "test-model"
        assert result.tokens_per_class == 1000
        assert len(result.layer_results) == 1

    def test_default_empty_layer_results(self) -> None:
        """ProbeExperimentResult should default to empty layer_results."""
        result = ProbeExperimentResult(
            concept="toxicity",
            model_name="test-model",
            tokens_per_class=500,
        )
        assert result.layer_results == []


class TestTokenAnalysisConfig:
    """Tests for TokenAnalysisConfig dataclass."""

    def test_default_values(self) -> None:
        """TokenAnalysisConfig should have correct default values."""
        config = TokenAnalysisConfig()
        assert config.top_k == 50
        assert config.tokens_per_class == 10000
        assert config.test_size == 0.2
        assert config.batch_size == 8
        assert config.random_seed == 42

    def test_layers_default_is_10_values(self) -> None:
        """TokenAnalysisConfig layers should default to [0.0, 0.111..., 1.0] with 10 values."""
        config = TokenAnalysisConfig()
        assert len(config.layers) == 10
        assert config.layers[0] == 0.0
        assert config.layers[-1] == 1.0
        # Check intermediate values
        for i in range(10):
            assert config.layers[i] == pytest.approx(i / 9, rel=1e-5)

    def test_custom_values(self) -> None:
        """TokenAnalysisConfig should accept custom values."""
        config = TokenAnalysisConfig(
            top_k=100,
            tokens_per_class=5000,
            test_size=0.3,
            layers=[0.4, 0.5, 0.6],
            batch_size=16,
            random_seed=123,
        )
        assert config.top_k == 100
        assert config.tokens_per_class == 5000
        assert config.test_size == 0.3
        assert config.layers == [0.4, 0.5, 0.6]
        assert config.batch_size == 16
        assert config.random_seed == 123


class TestComputeDiscriminativeScores:
    """Tests for compute_discriminative_scores function."""

    def _create_mock_records(
        self, n_records: int, label: str, hidden_dim: int = 64, seed: int = 42
    ) -> list[TokenRecord]:
        """Helper to create mock TokenRecords with random activations."""
        torch.manual_seed(seed)
        records = []
        for i in range(n_records):
            record = TokenRecord(
                token_id=i,
                token_text=f"token_{i}",
                activation=torch.randn(hidden_dim),
                contrast_pair_idx=i,
                position_in_sequence=0,
                label=label,
            )
            records.append(record)
        return records

    def test_scores_are_finite(self) -> None:
        """Discriminative scores should be finite numbers."""
        pos_records = self._create_mock_records(10, "positive", seed=1)
        neg_records = self._create_mock_records(10, "negative", seed=2)

        pos_scored, neg_scored = compute_discriminative_scores(pos_records, neg_records)

        for record in pos_scored:
            assert isinstance(record.score, float)
            assert torch.isfinite(torch.tensor(record.score))
        for record in neg_scored:
            assert isinstance(record.score, float)
            assert torch.isfinite(torch.tensor(record.score))

    def test_scores_are_sorted_descending(self) -> None:
        """Records should be sorted by score in descending order."""
        pos_records = self._create_mock_records(20, "positive", seed=1)
        neg_records = self._create_mock_records(20, "negative", seed=2)

        pos_scored, neg_scored = compute_discriminative_scores(pos_records, neg_records)

        # Check positive scores are descending
        for i in range(len(pos_scored) - 1):
            assert pos_scored[i].score >= pos_scored[i + 1].score

        # Check negative scores are descending
        for i in range(len(neg_scored) - 1):
            assert neg_scored[i].score >= neg_scored[i + 1].score

    def test_raises_on_empty_positive(self) -> None:
        """Should raise ValueError when positive records list is empty."""
        neg_records = self._create_mock_records(5, "negative")
        with pytest.raises(ValueError, match="empty records"):
            compute_discriminative_scores([], neg_records)

    def test_raises_on_empty_negative(self) -> None:
        """Should raise ValueError when negative records list is empty."""
        pos_records = self._create_mock_records(5, "positive")
        with pytest.raises(ValueError, match="empty records"):
            compute_discriminative_scores(pos_records, [])


class TestSelectTopKTokens:
    """Tests for select_top_k_tokens function."""

    def _create_scored_records(
        self, n_records: int, label: str, hidden_dim: int = 64, seed: int = 42
    ) -> list[TokenRecord]:
        """Helper to create mock TokenRecords with pre-computed scores."""
        torch.manual_seed(seed)
        records = []
        for i in range(n_records):
            record = TokenRecord(
                token_id=i,
                token_text=f"token_{i}",
                activation=torch.randn(hidden_dim),
                contrast_pair_idx=i,
                position_in_sequence=0,
                label=label,
                score=float(i),  # Score increases with i
            )
            records.append(record)
        return records

    def test_returns_exactly_k_tokens(self) -> None:
        """Should return exactly k tokens when available."""
        pos_records = self._create_scored_records(50, "positive")
        neg_records = self._create_scored_records(50, "negative")

        result = select_top_k_tokens(pos_records, neg_records, top_k=10)

        assert len(result.top_positive) == 10
        assert len(result.top_negative) == 10

    def test_returns_all_when_k_exceeds_list_length(self) -> None:
        """Should return all available tokens when k > len(list)."""
        pos_records = self._create_scored_records(5, "positive")
        neg_records = self._create_scored_records(5, "negative")

        result = select_top_k_tokens(pos_records, neg_records, top_k=100)

        assert len(result.top_positive) == 5
        assert len(result.top_negative) == 5

    def test_returns_highest_scored_tokens(self) -> None:
        """Should return tokens with highest scores."""
        pos_records = self._create_scored_records(20, "positive")
        neg_records = self._create_scored_records(20, "negative")

        result = select_top_k_tokens(pos_records, neg_records, top_k=5)

        # With scores 0-19, top 5 should be 15-19
        pos_scores = [r.score for r in result.top_positive]
        assert pos_scores == [19.0, 18.0, 17.0, 16.0, 15.0]

        neg_scores = [r.score for r in result.top_negative]
        assert neg_scores == [19.0, 18.0, 17.0, 16.0, 15.0]

    def test_result_contains_concept_and_layer(self) -> None:
        """Result should contain the concept name and layer index."""
        pos_records = self._create_scored_records(10, "positive")
        neg_records = self._create_scored_records(10, "negative")

        result = select_top_k_tokens(pos_records, neg_records, top_k=5, concept="honesty", layer=7)

        assert result.concept == "honesty"
        assert result.layer == 7

    def test_handles_empty_lists_gracefully(self) -> None:
        """Should handle empty lists without error."""
        # Create empty scored records (no score set)
        pos_records: list[TokenRecord] = []
        neg_records: list[TokenRecord] = []

        result = select_top_k_tokens(pos_records, neg_records, top_k=5)

        assert len(result.top_positive) == 0
        assert len(result.top_negative) == 0

    def test_handles_mismatched_lengths(self) -> None:
        """Should handle when positive and negative have different lengths."""
        pos_records = self._create_scored_records(10, "positive")
        neg_records = self._create_scored_records(3, "negative")

        result = select_top_k_tokens(pos_records, neg_records, top_k=5)

        assert len(result.top_positive) == 5  # Capped at available (10)
        assert len(result.top_negative) == 3  # Capped at available (3)
