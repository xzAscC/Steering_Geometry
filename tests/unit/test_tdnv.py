"""Tests for TDNV metrics computation."""

import torch

from steering_geometry.tdnv import (
    EPS,
    _compute_per_topic_stats,
    compute_tdnv,
)


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
        assert stats[1].count == 2

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


class TestComputeTDNV:
    """Tests for compute_tdnv core function."""

    def test_well_separated_topics(self) -> None:
        pos_activations = torch.tensor(
            [
                [1.0, 0.0],
                [1.1, 0.0],
                [0.9, 0.0],
            ],
            dtype=torch.float32,
        )
        neg_activations = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 1.1],
                [0.0, 0.9],
            ],
            dtype=torch.float32,
        )

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.tdnv < 1.0
        assert metrics.tdnv > 0.0
        assert metrics.energy > 0.0

    def test_overlapping_topics(self) -> None:
        pos_activations = torch.tensor(
            [
                [0.5, 0.5],
                [0.51, 0.49],
                [0.49, 0.51],
            ],
            dtype=torch.float32,
        )
        neg_activations = torch.tensor(
            [
                [0.5, 0.5],
                [0.49, 0.51],
                [0.51, 0.49],
            ],
            dtype=torch.float32,
        )

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.tdnv > 10.0 or metrics.tdnv == float("inf")

    def test_identical_means_with_variance(self) -> None:
        pos_activations = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        neg_activations = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.tdnv == 0.0
        assert metrics.energy > 0.0

    def test_zero_separation_high_variance(self) -> None:
        pos_activations = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
        neg_activations = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.norm_den == 0.0
        assert metrics.tdnv > 1e6

    def test_single_sample_each(self) -> None:
        pos_activations = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        neg_activations = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.tdnv >= 0.0
        assert metrics.energy > 0.0

    def test_normalization_values(self) -> None:
        pos_activations = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        neg_activations = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.norm_num >= 0.0
        assert metrics.norm_den >= 0.0

    def test_epsilon_in_denominator(self) -> None:
        pos_activations = torch.zeros(3, 2, dtype=torch.float32)
        neg_activations = torch.zeros(3, 2, dtype=torch.float32)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert metrics.tdnv >= 0.0 or metrics.tdnv == float("inf")

    def test_float32_precision(self) -> None:
        pos_activations = torch.randn(10, 64, dtype=torch.float16)
        neg_activations = torch.randn(10, 64, dtype=torch.float16)

        metrics = compute_tdnv(pos_activations, neg_activations)

        assert isinstance(metrics.tdnv, float)
        assert isinstance(metrics.energy, float)


class TestTDNVFormula:
    """Tests verifying TDNV formula correctness."""

    def test_formula_manual_verification(self) -> None:
        pos_activations = torch.tensor(
            [[1.0, 0.0], [3.0, 0.0]],
            dtype=torch.float32,
        )
        neg_activations = torch.tensor(
            [[0.0, 1.0], [0.0, 3.0]],
            dtype=torch.float32,
        )

        metrics = compute_tdnv(pos_activations, neg_activations)

        pos_mean = torch.tensor([2.0, 0.0])
        neg_mean = torch.tensor([0.0, 2.0])

        pos_var = ((1 - 2) ** 2 + (3 - 2) ** 2) / 2
        neg_var = ((1 - 2) ** 2 + (3 - 2) ** 2) / 2
        avg_within_var = (pos_var + neg_var) / 2

        mean_diff_sq = float(((pos_mean - neg_mean) ** 2).sum().item())

        expected_tdnv = avg_within_var / (2 * mean_diff_sq + EPS)

        assert abs(metrics.tdnv - expected_tdnv) < 0.01

    def test_lower_tdnv_better_separability(self) -> None:
        well_separated_pos = torch.tensor([[10.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
        well_separated_neg = torch.tensor([[0.0, 10.0], [0.0, 10.0]], dtype=torch.float32)

        poor_separated_pos = torch.tensor(
            [[1.0, 1.0], [1.5, 0.5], [0.5, 1.5]],
            dtype=torch.float32,
        )
        poor_separated_neg = torch.tensor(
            [[1.0, 1.0], [0.5, 1.5], [1.5, 0.5]],
            dtype=torch.float32,
        )

        well_metrics = compute_tdnv(well_separated_pos, well_separated_neg)
        poor_metrics = compute_tdnv(poor_separated_pos, poor_separated_neg)

        assert well_metrics.tdnv < poor_metrics.tdnv


class TestTDNVEnergy:
    """Tests for layerwise energy computation."""

    def test_energy_computation(self) -> None:
        activations = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]],
            dtype=torch.float32,
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
