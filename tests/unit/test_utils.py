"""Tests for utility functions."""

import torch

from steering_geometry.utils import select_token_activations


class TestSelectTokenAllMode:
    """Tests for select_token_activations with read_token_index='all'."""

    def test_select_token_all_mode_3d(self) -> None:
        """'all' on 3D tensor returns only non-padding tokens.

        (2, 5, 8) tensor where positions 3-4 are padding (zeros).
        Each sample has 3 real tokens → 2*3 = 6 total → (6, 8).
        """
        activations = torch.zeros(2, 5, 8)
        # Sample 0: real tokens at positions 0, 1, 2
        activations[0, 0] = torch.randn(8)
        activations[0, 1] = torch.randn(8)
        activations[0, 2] = torch.randn(8)
        # Sample 1: real tokens at positions 0, 1, 2
        activations[1, 0] = torch.randn(8)
        activations[1, 1] = torch.randn(8)
        activations[1, 2] = torch.randn(8)
        # Positions 3 and 4 are zeros (padding)

        result = select_token_activations(activations, "all")

        assert result.shape == (6, 8)
        # Verify no padding tokens leaked in
        assert (result.abs().sum(dim=-1) > 0).all()

    def test_select_token_all_mode_2d(self) -> None:
        """'all' on 2D tensor should return as-is (passthrough)."""
        activations = torch.randn(3, 8)

        result = select_token_activations(activations, "all")

        assert torch.equal(result, activations)
        assert result.shape == (3, 8)

    def test_select_token_all_no_padding_tokens(self) -> None:
        """All tokens are real (no padding) — 'all' returns (batch*seq_len, hidden_dim)."""
        batch, seq_len, hidden = 2, 4, 8
        activations = torch.randn(batch, seq_len, hidden)

        result = select_token_activations(activations, "all")

        assert result.shape == (batch * seq_len, hidden)
        assert torch.equal(result, activations.reshape(-1, hidden))


class TestSelectTokenLastNMode:
    """Tests for select_token_activations with read_token_index='last_n'."""

    def test_select_token_last_n_mode(self) -> None:
        """last_n=2 on (2, 5, 8) with 3 real tokens each returns (4, 8)."""
        activations = torch.zeros(2, 5, 8)
        # Sample 0: real tokens at positions 0, 1, 2
        activations[0, 0] = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
        activations[0, 1] = torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0])
        activations[0, 2] = torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0])
        # Sample 1: real tokens at positions 0, 1, 2
        activations[1, 0] = torch.tensor([0, 0, 0, 1.0, 0, 0, 0, 0])
        activations[1, 1] = torch.tensor([0, 0, 0, 0, 1.0, 0, 0, 0])
        activations[1, 2] = torch.tensor([0, 0, 0, 0, 0, 1.0, 0, 0])

        result = select_token_activations(activations, "last_n", last_n=2)

        assert result.shape == (4, 8)
        # Should be last 2 tokens from each sample
        # Sample 0: positions 1 and 2
        assert torch.equal(result[0], activations[0, 1])
        assert torch.equal(result[1], activations[0, 2])
        # Sample 1: positions 1 and 2
        assert torch.equal(result[2], activations[1, 1])
        assert torch.equal(result[3], activations[1, 2])

    def test_select_token_last_n_overflow(self) -> None:
        """last_n > seq_len should return all available tokens gracefully."""
        activations = torch.randn(2, 3, 8)  # All tokens are real (no padding)

        result = select_token_activations(activations, "last_n", last_n=10)

        assert result.shape == (6, 8)  # 2 * 3 = 6, all returned

    def test_select_token_last_n_no_padding(self) -> None:
        """All tokens real, last_n=3 returns (batch*3, hidden_dim)."""
        batch, seq_len, hidden = 2, 5, 8
        activations = torch.randn(batch, seq_len, hidden)

        result = select_token_activations(activations, "last_n", last_n=3)

        assert result.shape == (batch * 3, hidden)
        # Should be last 3 tokens from each sample
        expected = torch.cat(
            [activations[i, -3:] for i in range(batch)],
            dim=0,
        )
        assert torch.equal(result, expected)


class TestSelectTokenBackwardCompat:
    """Tests that existing int-index behavior is preserved."""

    def test_select_token_int_index_backward_compat(self) -> None:
        """int index=-1 still returns last non-padding token per sample → (batch, hidden)."""
        activations = torch.zeros(2, 5, 8)
        # Sample 0: real tokens at 0, 1, 2 → last non-padding at index 2
        activations[0, 0] = torch.randn(8)
        activations[0, 1] = torch.randn(8)
        activations[0, 2] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # Sample 1: real tokens at 0, 1, 2, 3 → last non-padding at index 3
        activations[1, 0] = torch.randn(8)
        activations[1, 1] = torch.randn(8)
        activations[1, 2] = torch.randn(8)
        activations[1, 3] = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        result = select_token_activations(activations, -1)

        assert result.shape == (2, 8)
        assert torch.equal(result[0], activations[0, 2])
        assert torch.equal(result[1], activations[1, 3])
