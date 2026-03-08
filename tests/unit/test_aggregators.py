"""Tests for steering vector aggregator functions."""

import pytest
import torch

from steering_geometry.config import ExtractionConfig
from steering_geometry.extract import (
    _resolve_aggregator,
    discriminative_token_aggregator,
    weighted_mean_aggregator,
)


class TestWeightedMeanAggregator:
    def test_basic(self) -> None:
        pos = torch.randn(10, 64)
        neg = torch.randn(10, 64)
        result = weighted_mean_aggregator(pos, neg)
        assert result.shape == (64,)

    def test_single_sample(self) -> None:
        pos = torch.randn(1, 64)
        neg = torch.randn(1, 64)
        result = weighted_mean_aggregator(pos, neg)
        assert result.shape == (64,)
        assert not torch.isnan(result).any()

    def test_weights_central_tokens_higher(self) -> None:
        center = torch.zeros(8)
        pos = torch.stack(
            [
                center + 0.1,
                center + 0.1,
                center + 0.1,
                center + 5.0,
            ]
        )
        neg = torch.stack(
            [
                center - 0.1,
                center - 0.1,
                center - 0.1,
                center - 5.0,
            ]
        )
        result = weighted_mean_aggregator(pos, neg)
        assert result.shape == (8,)
        assert not torch.isnan(result).any()

    def test_identical_samples(self) -> None:
        pos = torch.zeros(5, 64)
        neg = torch.zeros(5, 64)
        result = weighted_mean_aggregator(pos, neg)
        assert result.shape == (64,)
        assert torch.allclose(result, torch.zeros(64))


class TestDiscriminativeTokenAggregator:
    def test_basic(self) -> None:
        pos = torch.randn(20, 64)
        neg = torch.randn(20, 64)
        result = discriminative_token_aggregator(pos, neg, top_k=10)
        assert result.shape == (64,)

    def test_clamp_topk(self) -> None:
        pos = torch.randn(5, 64)
        neg = torch.randn(5, 64)
        result = discriminative_token_aggregator(pos, neg, top_k=100)
        assert result.shape == (64,)

    def test_selection(self) -> None:
        pos_center = torch.ones(8) * 10
        neg_center = torch.ones(8) * -10

        pos = torch.stack([pos_center + torch.randn(8) * 0.1 for _ in range(10)])
        neg = torch.stack([neg_center + torch.randn(8) * 0.1 for _ in range(10)])

        result = discriminative_token_aggregator(pos, neg, top_k=5)
        assert result.shape == (8,)

    def test_empty_tensor_raises(self) -> None:
        pos = torch.randn(0, 64)
        neg = torch.randn(5, 64)
        with pytest.raises(ValueError):
            discriminative_token_aggregator(pos, neg)

    def test_custom_topk(self) -> None:
        pos = torch.randn(20, 64)
        neg = torch.randn(20, 64)
        result_5 = discriminative_token_aggregator(pos, neg, top_k=5)
        result_10 = discriminative_token_aggregator(pos, neg, top_k=10)
        assert result_5.shape == (64,)
        assert result_10.shape == (64,)


class TestAggregatorIntegration:
    def test_resolve_weighted_mean(self) -> None:
        aggregator = _resolve_aggregator("weighted_mean")
        pos = torch.randn(5, 32)
        neg = torch.randn(5, 32)
        result = aggregator(pos, neg)
        assert result.shape == (32,)

    def test_resolve_discriminative(self) -> None:
        config = ExtractionConfig(top_k=5)
        aggregator = _resolve_aggregator("discriminative", config)
        pos = torch.randn(10, 32)
        neg = torch.randn(10, 32)
        result = aggregator(pos, neg)
        assert result.shape == (32,)

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported extraction method"):
            _resolve_aggregator("invalid_method")
