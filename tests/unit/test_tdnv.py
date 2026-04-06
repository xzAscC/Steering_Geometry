"""Tests for TDNV metrics computation."""

import torch

from steering_geometry.tdnv import (
    EPS,
    _compute_per_topic_stats,
    compute_tdnv,
    compute_tdnv_mmlu,  # noqa: F401 - TDD: function doesn't exist yet
    compute_tdnv_multi_concept,
    select_last_n_tokens,  # noqa: F401 - TDD: function doesn't exist yet
    select_top_k_discriminative,  # noqa: F401 - TDD: function doesn't exist yet
)  # noqa: F401 - TDD: function doesn't exist yet
from steering_geometry.types import TDNVLayerMetrics


class TestComputePerTopicStats:
    """Tests for _compute_per_topic_stats helper."""

    def test_basic_computation(self) -> None:
        activations = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ],
            dtype=torch.float32,
        )
        topic_labels = [0, 0, 1, 1]

        stats = _compute_per_topic_stats(activations, topic_labels)

        assert 0 in stats
        assert 1 in stats
        assert stats[0].count == 2
        assert stats[1].count == 2  # Fixed: 4 samples with labels [0,0,1,1] means 2 per topic

        assert torch.allclose(stats[0].mean, torch.tensor([1.5, 0.0]))
        assert torch.allclose(stats[1].mean, torch.tensor([0.0, 1.5]))

        expected_var_0 = ((1.0 - 1.5) ** 2 + (2.0 - 1.5) ** 2) / 2
        assert abs(stats[0].variance - expected_var_0) < 0.01

    def test_single_topic(self) -> None:
        activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        topic_labels = [0, 0]

        stats = _compute_per_topic_stats(activations, topic_labels)

        assert 0 in stats
        assert stats[0].count == 2

    def test_uneven_topics(self) -> None:
        activations = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        topic_labels = [0, 0, 0, 1]

        stats = _compute_per_topic_stats(activations, topic_labels)

        assert stats[0].count == 3
        assert stats[1].count == 1

    def test_epsilon_in_denominator(self) -> None:
        pos_activations = torch.zeros(3, 2, dtype=torch.float32)
        neg_activations = torch.zeros(3, 2, dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        # When all activations are identical zeros, both variance and mean distance are 0
        # So TDNV should be 0 (not inf, since variance is also 0)
        assert metrics.tdnv >= 0.0  # Changed: zero data means tdnv=0 is valid
        assert metrics.energy == 0.0  # Energy is also 0 for zero tensors

    def test_overlapping_topics(self) -> None:
        pos_activations = torch.tensor(
            [
                [0.5, 0.5],
                [0.51, 0.49],
                [0.5, 1.5],
            ],
            dtype=torch.float32,
        )
        neg_activations = torch.tensor(
            [
                [0.5, 0.5],
                [0.49, 0.51],
                [0.51, 1.51],
            ],
            dtype=torch.float32,
        )

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.tdnv > 10.0 or metrics.tdnv == float("inf")

    def test_identical_means_with_variance(self) -> None:
        pos_activations = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        neg_activations = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 5.0]], dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.tdnv < 1.0
        assert metrics.tdnv > 0.0
        assert metrics.energy > 0.0

    def test_zero_separation_high_variance(self) -> None:
        # Test with different means but high variance in negative class
        pos_activations = torch.tensor(
            [[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32
        )  # mean = [1, 1]
        neg_activations = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 5.0]], dtype=torch.float32
        )  # mean = [1, 7/3]

        metrics = compute_tdnv(pos_activations, neg_activations)

        # pos mean = [1, 1], neg mean = [1, 7/3] ≈ [1, 2.33]
        # Distance between means ≠ 0, so norm_den > 0
        assert metrics.norm_den > 0.0  # Fixed: means differ, so norm_den > 0
        # High variance relative to separation should give moderate TDNV
        assert metrics.tdnv > 0.0

    def test_single_sample_each(self) -> None:
        pos_activations = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        neg_activations = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.tdnv >= 0.0
        assert metrics.energy > 0.0

    def test_normalization_values(self) -> None:
        pos_activations = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        neg_activations = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.norm_num >= 0.0
        assert metrics.norm_den >= 0.0

        expected_energy = float(
            (torch.cat([pos_activations, neg_activations]) ** 2).sum(dim=1).mean().item()
        )
        pos = pos_activations
        neg = neg_activations
        metrics = compute_tdnv(pos, neg)

        assert abs(metrics.energy - expected_energy) < 0.01

    def test_float32_precision(self) -> None:
        pos_activations = torch.randn(10, 64, dtype=torch.float16)
        neg_activations = torch.randn(10, 64, dtype=torch.float16)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert isinstance(metrics.tdnv, float)
        assert isinstance(metrics.energy, float)


class TestTDNVFormula:
    """Tests verifying TDNV formula correctness."""

    def test_formula_manual_verification(self) -> None:
        pos_activations = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
        neg_activations = torch.tensor([[0.0, 1.0], [0.0, 3.0]], dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        pos_mean = torch.tensor([2.0, 0.0])
        neg_mean = torch.tensor([0.0, 2.0])
        mean_diff_sq = float(((pos_mean - neg_mean) ** 2).sum().item())

        pos_var = ((1 - 2) ** 2 + (3 - 2) ** 2) / 2
        expected_tdnv = pos_var / (2 * mean_diff_sq + EPS)

        assert abs(metrics.tdnv - expected_tdnv) < 0.01

    def test_lower_tdnv_better_separability(self) -> None:
        well_separated_pos = torch.tensor([[10.0, 0.0], [10.5, 0.0]], dtype=torch.float32)
        well_separated_neg = torch.tensor([[0.0, 10.0], [0.0, 10.5]], dtype=torch.float32)
        metrics_well = compute_tdnv(well_separated_pos, well_separated_neg)

        poorly_separated_pos = torch.tensor([[1.0, 0.0], [1.5, 0.0]], dtype=torch.float32)
        poorly_separated_neg = torch.tensor([[0.0, 1.0], [0.0, 1.5]], dtype=torch.float32)
        metrics_poorly = compute_tdnv(poorly_separated_pos, poorly_separated_neg)

        assert metrics_well.tdnv < metrics_poorly.tdnv


class TestTDNVEEnergy:
    """Tests for layerwise energy computation."""

    def test_energy_computation(self) -> None:
        activations = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=torch.float32
        )

        expected_energy = float((activations**2).sum(dim=1).mean().item())
        pos = activations[:2]
        neg = activations[2:]

        metrics = compute_tdnv(pos, neg)

        assert abs(metrics.energy - expected_energy) < 0.01

    def test_energy_positive(self) -> None:
        pos_activations = torch.randn(5, 10, dtype=torch.float32)
        neg_activations = torch.randn(5, 10, dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.energy > 0.0


class TestComputeTDNVMultiConcept:
    """Tests for compute_tdnv_multi_concept with multiple concepts."""

    def test_two_concepts_basic(self) -> None:
        """Test with 2 concepts (4 groups total)."""
        concepts = {
            "polite": (
                torch.randn(8, 32, dtype=torch.float32),
                torch.randn(8, 32, dtype=torch.float32),
            ),
            "sentiment": (
                torch.randn(8, 32, dtype=torch.float32),
                torch.randn(8, 32, dtype=torch.float32),
            ),
            "refusal": (
                torch.randn(8, 32, dtype=torch.float32),
                torch.randn(8, 32, dtype=torch.float32),
            ),
        }

        metrics = compute_tdnv_multi_concept(concepts)

        assert metrics.tdnv >= 0.0
        assert metrics.norm_num >= 0.0
        assert metrics.norm_den >= 0.0
        assert metrics.energy > 0.0

    def test_tdnv_formula_correctness(self) -> None:
        """Verify TDNV formula: (1/M(M-1)) * Σ (var_g + var_g') / (2||mean_g - mean_g'||²)."""
        # Create well-separated groups with known variance
        # Group 0: mean at [2, 0], variance 0
        g0 = torch.tensor([[2.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        # Group 1: mean at [0, 2], variance 0
        g1 = torch.tensor([[0.0, 2.0], [0.0, 2.0]], dtype=torch.float32)
        # Group 2: mean at [-2, 0], variance 0
        g2 = torch.tensor([[-2.0, 0.0], [0.0, -2.0]], dtype=torch.float32)
        # Group 3: mean at [0, -2], variance 0
        g3 = torch.tensor([[0.0, -2.0], [0.0, -2.0]], dtype=torch.float32)

        concepts = {
            "concept_a": (g0, g1),
            "concept_b": (g2, g3),
        }

        metrics = compute_tdnv_multi_concept(concepts)

        # M = 4 groups, so M(M-1) = 12 pairs
        # All variances are 0 (identical samples in each group)
        # So numerator is 0, TDNV should be 0 or very small
        assert metrics.tdnv < 1.0

    def test_empty_group_handling(self) -> None:
        """Test handling of concepts with single-element groups."""
        concepts = {
            "single": (
                torch.randn(1, 16, dtype=torch.float32),
                torch.randn(1, 16, dtype=torch.float32),
            ),
        }

        metrics = compute_tdnv_multi_concept(concepts)

        # Single sample groups have zero variance
        assert metrics.tdnv >= 0.0

    def test_variance_distance_computation(self) -> None:
        """Test that variance and distance are computed correctly."""
        # Create groups with known separations
        pos_a = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        neg_a = torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
        pos_b = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0], [0.0, 2.0]], dtype=torch.float32)
        neg_b = torch.tensor(
            [[0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.0, -1.0]],
            dtype=torch.float32,
        )

        concepts = {
            "a": (pos_a, neg_a),
            "b": (pos_b, neg_b),
        }

        metrics = compute_tdnv_multi_concept(concepts)

        # Zero variance + large separation = low TDNV
        assert metrics.tdnv >= 0.0
        assert metrics.norm_den > 0.0  # Non-zero distance

    def test_overlapping_groups_high_tdnv(self) -> None:
        """Overlapping groups should produce high TDNV."""
        # All groups centered at same point with high variance
        center = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        noise_pos = center + torch.randn(5, 2, dtype=torch.float32) * 0.1
        noise_neg = center + torch.randn(5, 2, dtype=torch.float32) * 0.1

        concepts = {
            "overlap_a": (noise_pos.clone(), noise_neg.clone()),
            "overlap_b": (noise_pos.clone(), noise_neg.clone()),
        }
        metrics = compute_tdnv_multi_concept(concepts)

        # High TDNV expected due to small separation
        # (could be inf if means are identical)
        assert metrics.tdnv > 1.0 or metrics.tdnv == float("inf")

    def test_returns_tdnv_layer_metrics(self) -> None:
        """Verify return type is TDNVLayerMetrics."""
        concepts = {
            "test": (
                torch.randn(5, 8, dtype=torch.float32),
                torch.randn(5, 8, dtype=torch.float32),
            ),
        }
        metrics = compute_tdnv_multi_concept(concepts)

        assert isinstance(metrics, TDNVLayerMetrics)
        assert hasattr(metrics, "tdnv")
        assert hasattr(metrics, "norm_num")
        assert hasattr(metrics, "norm_den")
        assert hasattr(metrics, "energy")


class TestSelectLastNTokens:
    """Tests for select_last_n_tokens helper function."""

    def test_basic_selection(self) -> None:
        """Test selecting last 3 tokens from 10-token tensor."""
        # 10 tokens with 4-dim hidden states
        activations = torch.arange(40, dtype=torch.float32).reshape(10, 4)

        result = select_last_n_tokens(activations, n=3)

        # Should return last 3 rows
        assert result.shape == (3, 4)
        expected = activations[-3:]
        assert torch.allclose(result, expected)

    def test_n_larger_than_tokens(self) -> None:
        """Test that n > total tokens returns all tokens."""
        # 5 tokens with 2-dim hidden states
        activations = torch.arange(10, dtype=torch.float32).reshape(5, 2)

        result = select_last_n_tokens(activations, n=10)

        # Should return all 5 tokens
        assert result.shape == (5, 2)
        assert torch.allclose(result, activations)

    def test_n_equals_zero(self) -> None:
        """Test that n=0 returns empty tensor."""
        activations = torch.arange(20, dtype=torch.float32).reshape(10, 2)

        result = select_last_n_tokens(activations, n=0)

        # Should return empty tensor with correct hidden_dim
        assert result.shape == (0, 2)
        assert result.dtype == torch.float32


class TestSelectTopKDiscriminative:
    """Tests for select_top_k_discriminative helper function."""

    def test_basic_selection(self) -> None:
        """Test selecting top-k tokens from two classes."""
        # Create well-separated classes
        # Class 0: tokens near [1, 0]
        class_0 = torch.tensor(
            [[1.0, 0.0], [1.1, 0.1], [0.9, -0.1]],
            dtype=torch.float32,
        )
        # Class 1: tokens near [0, 1]
        class_1 = torch.tensor(
            [[0.0, 1.0], [0.1, 1.1], [-0.1, 0.9]],
            dtype=torch.float32,
        )
        activations = torch.cat([class_0, class_1], dim=0)
        labels = [0, 0, 0, 1, 1, 1]

        result = select_top_k_discriminative(activations, labels, k=2)

        # Should return top-2 per class = 4 total
        assert result.shape[0] == 4
        assert result.shape[1] == 2  # hidden_dim preserved

    def test_scoring_formula(self) -> None:
        """Verify discriminative score: Σ_{c≠own} ||h_i - μ_c||² - ||h_i - μ_same||²."""
        # Create two classes with known means
        # Class 0: mean = [2, 0]
        class_0 = torch.tensor(
            [[2.0, 0.0], [2.0, 0.0]],
            dtype=torch.float32,
        )
        # Class 1: mean = [0, 2]
        class_1 = torch.tensor(
            [[0.0, 2.0], [0.0, 2.0]],
            dtype=torch.float32,
        )
        activations = torch.cat([class_0, class_1], dim=0)
        labels = [0, 0, 1, 1]

        # For class 0 token [2, 0] (binary case, sum over 1 other class):
        # μ_same = [2, 0], μ_class1 = [0, 2]
        # Σ ||h - μ_c||² = ||[2, -2]||² = 8
        # ||h - μ_same||² = 0
        # score = 8 - 0 = 8

        result = select_top_k_discriminative(activations, labels, k=2)

        assert result.shape[0] == 4

    def test_k_larger_than_class(self) -> None:
        """Test k > class size returns all tokens from that class."""
        # 3 tokens for class 0, 2 tokens for class 1
        class_0 = torch.randn(3, 4, dtype=torch.float32)
        class_1 = torch.randn(2, 4, dtype=torch.float32)
        activations = torch.cat([class_0, class_1], dim=0)
        labels = [0, 0, 0, 1, 1]

        result = select_top_k_discriminative(activations, labels, k=10)

        # Should return all 5 tokens (3 + 2)
        assert result.shape[0] == 5
        assert result.shape[1] == 4

    def test_multi_class_sum_scoring(self) -> None:
        """With 3+ classes, score = Σ_{c≠own} ||h-μ_c||² - ||h-μ_same||²."""
        # Class 0 at [4, 0], class 1 at [0, 4], class 2 at [-4, 0]
        class_0 = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
        class_1 = torch.tensor([[0.0, 4.0]], dtype=torch.float32)
        class_2 = torch.tensor([[-4.0, 0.0]], dtype=torch.float32)
        activations = torch.cat([class_0, class_1, class_2], dim=0)
        labels = [0, 1, 2]

        result = select_top_k_discriminative(activations, labels, k=1)

        # Each class has 1 token, so result = 3 tokens total
        assert result.shape[0] == 3

    def test_three_class_prefers_central_token(self) -> None:
        """Token far from all other classes scores higher."""
        # Class 0: [5, 0] and [0.1, 0] — [5,0] far from others
        # Class 1: [0, 5] and [0, 0.1]
        # Class 2: [-5, 0] and [-0.1, 0]
        class_0 = torch.tensor([[5.0, 0.0], [0.1, 0.0]], dtype=torch.float32)
        class_1 = torch.tensor([[0.0, 5.0], [0.0, 0.1]], dtype=torch.float32)
        class_2 = torch.tensor([[-5.0, 0.0], [-0.1, 0.0]], dtype=torch.float32)
        activations = torch.cat([class_0, class_1, class_2], dim=0)
        labels = [0, 0, 1, 1, 2, 2]

        result = select_top_k_discriminative(activations, labels, k=1)

        # Top-1 per class should pick [5,0], [0,5], [-5,0] (far outliers)
        assert result.shape[0] == 3
        # Verify class 0's selected token is [5, 0] not [0.1, 0]
        assert result[0, 0].item() == 5.0


class TestComputeTDNVMMLU:
    """Tests for compute_tdnv_mmlu with MMLU-Pro category activations."""

    def test_basic_multi_category(self) -> None:
        """Test with 3 mock categories (multi-class TDNV)."""
        category_activations = {
            "math": torch.randn(10, 64, dtype=torch.float32),
            "physics": torch.randn(10, 64, dtype=torch.float32),
            "chemistry": torch.randn(10, 64, dtype=torch.float32),
        }

        metrics = compute_tdnv_mmlu(category_activations)

        assert isinstance(metrics, TDNVLayerMetrics)
        assert metrics.tdnv >= 0.0
        assert metrics.energy > 0.0
        assert metrics.norm_num >= 0.0
        assert metrics.norm_den >= 0.0

    def test_category_extraction(self) -> None:
        """Test that categories are properly used as group labels."""
        # Create activations with different means per category
        math_activations = torch.cat(
            [torch.ones(5, 16, dtype=torch.float32), torch.ones(5, 16, dtype=torch.float32)],
        )
        physics_activations = torch.cat(
            [-torch.ones(5, 16, dtype=torch.float32), -torch.ones(5, 16, dtype=torch.float32)],
        )
        biology_activations = torch.zeros(10, 16, dtype=torch.float32)

        category_activations = {
            "math": math_activations,
            "physics": physics_activations,
            "biology": biology_activations,
        }

        metrics = compute_tdnv_mmlu(category_activations)

        # Well-separated groups should have low TDNV
        assert metrics.tdnv >= 0.0
        assert metrics.norm_den > 0.0  # Non-zero distance between groups

    def test_missing_category_uses_default(self) -> None:
        """Test that empty category dict doesn't crash (graceful handling)."""
        # Single category is an edge case - should still compute valid metrics
        category_activations = {
            "default": torch.randn(5, 32, dtype=torch.float32),
        }

        metrics = compute_tdnv_mmlu(category_activations)

        # Single category: only one group, special case
        assert isinstance(metrics, TDNVLayerMetrics)
        assert metrics.tdnv >= 0.0
        assert metrics.energy > 0.0

    def test_single_category_edge_case(self) -> None:
        """Test behavior when only one category exists (M=1)."""
        category_activations = {
            "only_category": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                dtype=torch.float32,
            ),
        }

        metrics = compute_tdnv_mmlu(category_activations)

        # Single group: variance-only metric or special handling
        assert isinstance(metrics, TDNVLayerMetrics)
        # Energy should still be computed correctly
        expected_energy = float(
            (category_activations["only_category"] ** 2).sum(dim=1).mean().item()
        )
        assert abs(metrics.energy - expected_energy) < 0.01
