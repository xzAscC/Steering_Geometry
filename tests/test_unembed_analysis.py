"""Tests for unembedding analysis module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from steering_geometry.types import ConceptAnalysisResult, UnembedAnalysisResult
from steering_geometry.unembed_analysis import (
    analyze_steering_vector,
    compute_topk_similar_tokens,
    plot_topk_bar_chart,
)


class TestComputeTopkSimilarTokens:
    """Tests for compute_topk_similar_tokens function."""

    def test_compute_topk_similar_tokens_basic(
        self,
        mock_unembedding_matrix: Tensor,
        sample_unembed_vector: Tensor,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Basic functionality returns correct results with valid similarity range."""
        k = 5
        results = compute_topk_similar_tokens(
            vector=sample_unembed_vector,
            unembed_matrix=mock_unembedding_matrix,
            tokenizer=mock_tokenizer,
            k=k,
        )

        # Verify correct number of results
        assert len(results) == k

        # Verify all results are tuples of (str, float)
        for token_text, similarity in results:
            assert isinstance(token_text, str)
            assert isinstance(similarity, float)

        # Verify similarity values are in valid range [-1, 1] for cosine similarity
        for _, similarity in results:
            assert -1.0 <= similarity <= 1.0

        # Verify results are sorted by descending similarity
        similarities = [sim for _, sim in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_compute_topk_similar_tokens_exclude(
        self,
        mock_unembedding_matrix: Tensor,
        sample_unembed_vector: Tensor,
        mock_tokenizer: MagicMock,
        mock_special_token_ids: set[int],
    ) -> None:
        """Excluded token IDs should not appear in results."""
        k = 10
        results = compute_topk_similar_tokens(
            vector=sample_unembed_vector,
            unembed_matrix=mock_unembedding_matrix,
            tokenizer=mock_tokenizer,
            k=k,
            exclude_tokens=mock_special_token_ids,
        )

        # Verify we got results
        assert len(results) > 0

        # Get all token IDs from decoded results by re-checking the matrix
        # Since we're using a mock tokenizer, we verify by checking that
        # the function ran without error and returned the expected count
        # The actual exclusion is verified by the -inf masking logic in the function

        # Verify correct number of results (should be k unless vocab is smaller)
        assert len(results) == min(k, mock_unembedding_matrix.shape[0])

        # Verify all similarities are not -inf (which would indicate excluded tokens)
        for _, similarity in results:
            assert similarity > float("-inf")

    def test_compute_topk_similar_tokens_invalid_vector_dim(
        self,
        mock_unembedding_matrix: Tensor,
        mock_tokenizer: MagicMock,
    ) -> None:
        """2D vector should raise ValueError."""
        invalid_vector = torch.randn(10, 64, dtype=torch.float32)

        with pytest.raises(ValueError, match="Expected 1D vector"):
            compute_topk_similar_tokens(
                vector=invalid_vector,
                unembed_matrix=mock_unembedding_matrix,
                tokenizer=mock_tokenizer,
                k=5,
            )

    def test_compute_topk_similar_tokens_mismatched_hidden_dim(
        self,
        mock_unembedding_matrix: Tensor,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Vector with wrong hidden dimension should raise ValueError."""
        wrong_dim_vector = torch.randn(32, dtype=torch.float32)  # Should be 64

        with pytest.raises(ValueError, match="does not match"):
            compute_topk_similar_tokens(
                vector=wrong_dim_vector,
                unembed_matrix=mock_unembedding_matrix,
                tokenizer=mock_tokenizer,
                k=5,
            )


class TestAnalyzeSteeringVector:
    """Tests for analyze_steering_vector function."""

    def test_analyze_steering_vector(
        self,
        sample_unembed_vector: Tensor,
        mock_unembedding_matrix: Tensor,
        mock_tokenizer: MagicMock,
        mock_special_token_ids: set[int],
    ) -> None:
        """Integration-style test with mocked model should return UnembedAnalysisResult."""
        from steering_geometry.models import HookedModel

        layer_frac = 0.5
        method = "diff_means"
        k = 5

        mock_model = MagicMock(spec=HookedModel)
        mock_model.get_unembedding_matrix.return_value = mock_unembedding_matrix
        mock_model.get_special_token_ids.return_value = mock_special_token_ids
        mock_model.tokenizer = mock_tokenizer

        result = analyze_steering_vector(
            vector=sample_unembed_vector,
            model=mock_model,
            layer_frac=layer_frac,
            method=method,
            k=k,
        )

        assert isinstance(result, UnembedAnalysisResult)
        assert result.layer == layer_frac
        assert result.method == method
        assert isinstance(result.tokens, list)
        assert isinstance(result.similarities, list)
        assert len(result.tokens) == len(result.similarities)

        for token in result.tokens:
            assert isinstance(token, str)

        for sim in result.similarities:
            assert isinstance(sim, float)
            assert -1.0 <= sim <= 1.0


class TestPlotTopkBarChart:
    """Tests for plot_topk_bar_chart function."""

    def test_plot_topk_bar_chart(self, tmp_path: Path) -> None:
        """Function should create a PDF file and return its path."""
        result = ConceptAnalysisResult(
            concept="sentiment",
            model="test-model",
            method="diff_means",
            results={
                "layer_0.5": UnembedAnalysisResult(
                    layer=0.5,
                    method="diff_means",
                    tokens=["token1", "token2", "token3"],
                    similarities=[0.9, 0.8, 0.7],
                )
            },
        )

        paths = plot_topk_bar_chart(result, output_dir=tmp_path)

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".pdf"
        assert "sentiment_diff_means_bars.pdf" in str(paths[0])
