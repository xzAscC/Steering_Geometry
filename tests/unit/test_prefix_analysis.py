"""Tests for prefix_analysis module."""

from pathlib import Path

import pytest
import torch

import steering_geometry.prefix_analysis as prefix_analysis
from steering_geometry.models import HookedModel
from steering_geometry.prefix_analysis import (
    AttentionAnalysisResult,
    AttentionLinkInstance,
    KLDivergenceResult,
    PrefixLengthKLSweepResult,
    _compute_attn_cosine_distance,
    _compute_avg_activation,
    _extract_attention_links,
    _extract_prefix_attention,
    _make_steering_hook,
    generate_analysis_report,
    per_token_kl_divergence,
    plot_attention_analysis,
    plot_attention_link_heatmap,
    plot_kl_divergence_curves,
)
from steering_geometry.types import ContrastPair

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kl_result(
    step_kl_no_steer: list[float] | None = None,
    step_kl_all_steer: list[float] | None = None,
    steer_tokens: int = 5,
) -> KLDivergenceResult:
    """Build a KLDivergenceResult with sensible defaults."""
    default_no_steer = step_kl_no_steer if step_kl_no_steer is not None else [0.1, 0.2, 0.3]
    default_all_steer = step_kl_all_steer if step_kl_all_steer is not None else [0.01, 0.02, 0.03]
    return KLDivergenceResult(
        step_kl_no_steer=default_no_steer,
        step_kl_all_steer=default_all_steer,
        generated_text_no_steer="hello world",
        generated_text_prefix_steer="goodbye world",
        generated_text_all_steer="cheerio world",
        steer_tokens=steer_tokens,
        scale=1.0,
        layer_frac=0.7,
        prompt="test prompt",
    )


def _make_attn_result(
    attn_to_prefix_no_steer: list[float] | None = None,
    attn_to_prefix_prefix_steer: list[float] | None = None,
    attn_to_prefix_all_steer: list[float] | None = None,
    attn_cosine_shift: list[float] | None = None,
    steered_layer_attn_diff: list[float] | None = None,
    steer_tokens: int = 5,
    prompt_tokens: list[str] | None = None,
) -> AttentionAnalysisResult:
    """Build an AttentionAnalysisResult with sensible defaults."""
    return AttentionAnalysisResult(
        attn_to_prefix_no_steer=(
            attn_to_prefix_no_steer if attn_to_prefix_no_steer is not None else [0.1, 0.2]
        ),
        attn_to_prefix_prefix_steer=(
            attn_to_prefix_prefix_steer if attn_to_prefix_prefix_steer is not None else [0.3, 0.4]
        ),
        attn_to_prefix_all_steer=(
            attn_to_prefix_all_steer if attn_to_prefix_all_steer is not None else [0.25, 0.35]
        ),
        attn_cosine_shift=attn_cosine_shift if attn_cosine_shift is not None else [0.05, 0.10],
        steered_layer_attn_diff=(
            steered_layer_attn_diff if steered_layer_attn_diff is not None else [0.01, 0.02]
        ),
        prompt_tokens=prompt_tokens if prompt_tokens is not None else ["test", "prompt"],
        steer_tokens=steer_tokens,
    )


# ===========================================================================
# 1. per_token_kl_divergence
# ===========================================================================


class TestKLDivergence:
    """Tests for per_token_kl_divergence."""

    def test_identical_logits_zero_kl(self) -> None:
        """Identical logit distributions should produce KL = 0."""
        logits = torch.randn(1, 100)
        kl = per_token_kl_divergence(logits, logits)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_identical_distributions_different_logits(self) -> None:
        """Different logit tensors that represent the same distribution → KL ≈ 0.

        Adding a constant to all logits does not change softmax probabilities.
        """
        logits_q = torch.randn(1, 50)
        offset = 3.7  # arbitrary constant shift
        logits_p = logits_q + offset
        kl = per_token_kl_divergence(logits_p, logits_q)
        assert kl == pytest.approx(0.0, abs=1e-5)

    def test_known_kl_value(self) -> None:
        """Hand-computed KL for simple two-element distributions.

        P = softmax([2.0, 1.0]) ≈ [0.7311, 0.2689]
        Q = softmax([1.0, 2.0]) ≈ [0.2689, 0.7311]
        KL(P||Q) = p0*(log_p0 - log_q0) + p1*(log_p1 - log_q1)
        """
        logits_p = torch.tensor([[2.0, 1.0]])
        logits_q = torch.tensor([[1.0, 2.0]])

        # Compute reference by hand using log_softmax
        import torch.nn.functional as functional

        log_p = functional.log_softmax(logits_p.float(), dim=-1)
        log_q = functional.log_softmax(logits_q.float(), dim=-1)
        p = log_p.exp()
        expected_kl = float((p * (log_p - log_q)).sum(dim=-1).item())

        kl = per_token_kl_divergence(logits_p, logits_q)
        assert kl == pytest.approx(expected_kl, rel=1e-5)

    def test_kl_non_negative(self) -> None:
        """KL divergence must always be >= 0 (Gibbs' inequality)."""
        rng = torch.manual_seed(42)
        for _ in range(20):
            logits_p = torch.randn(1, 64, generator=rng)
            logits_q = torch.randn(1, 64, generator=rng)
            kl = per_token_kl_divergence(logits_p, logits_q)
            assert kl >= -1e-6, f"KL should be non-negative, got {kl}"

    def test_kl_asymmetric(self) -> None:
        """KL(P||Q) != KL(Q||P) in general."""
        logits_p = torch.tensor([[2.0, 1.0]])
        logits_q = torch.tensor([[1.0, 3.0]])
        kl_pq = per_token_kl_divergence(logits_p, logits_q)
        kl_qp = per_token_kl_divergence(logits_q, logits_p)
        assert kl_pq != pytest.approx(kl_qp, abs=1e-4)

    def test_large_logit_difference_produces_large_kl(self) -> None:
        """Very different distributions should produce non-trivial KL."""
        logits_p = torch.tensor([[10.0, 0.0]])
        logits_q = torch.tensor([[0.0, 10.0]])
        kl = per_token_kl_divergence(logits_p, logits_q)
        assert kl > 5.0, f"Expected large KL for divergent distributions, got {kl}"

    def test_single_element_vocab(self) -> None:
        """Single-element vocab: both distributions are trivially identical → KL = 0."""
        logits_p = torch.tensor([[5.0]])
        logits_q = torch.tensor([[3.0]])
        kl = per_token_kl_divergence(logits_p, logits_q)
        assert kl == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# 2. _extract_prefix_attention
# ===========================================================================


class TestExtractPrefixAttention:
    """Tests for _extract_prefix_attention."""

    def test_prefix_len_zero_returns_zero(self) -> None:
        """prefix_len=0 should return 0.0."""
        attn = torch.rand(1, 4, 10, 10)
        result = _extract_prefix_attention(attn, 0)
        assert result == 0.0

    def test_negative_prefix_len_returns_zero(self) -> None:
        """Negative prefix_len should return 0.0."""
        attn = torch.rand(1, 4, 10, 10)
        result = _extract_prefix_attention(attn, -3)
        assert result == 0.0

    def test_all_attention_on_first_position(self) -> None:
        """All attention on the first position, prefix_len=1 → high value."""
        attn = torch.zeros(1, 2, 5, 5)
        # Last query position, all heads, all weight on position 0
        attn[0, :, -1, 0] = 1.0
        result = _extract_prefix_attention(attn, 1)
        assert result == pytest.approx(1.0)

    def test_uniform_attention(self) -> None:
        """Uniform attention → mean of prefix weights = 1/seq_len."""
        seq_len = 10
        prefix_len = 4
        num_heads = 3
        attn = torch.ones(1, num_heads, seq_len, seq_len) / seq_len
        result = _extract_prefix_attention(attn, prefix_len)
        expected = 1.0 / seq_len
        assert result == pytest.approx(expected, rel=1e-4)

    def test_prefix_len_exceeds_seq_len_clamps(self) -> None:
        """prefix_len > seq_len should clamp to seq_len and return full mean."""
        seq_len = 5
        attn = torch.ones(1, 2, seq_len, seq_len) / seq_len
        result = _extract_prefix_attention(attn, prefix_len=100)
        # All positions are prefix (clamped), so mean of uniform = 1/seq_len
        assert result == pytest.approx(1.0 / seq_len, rel=1e-4)

    def test_single_head_single_position(self) -> None:
        """Minimal tensor: 1 head, 1x1 attention matrix."""
        attn = torch.tensor([[[[0.6]]]])  # (1, 1, 1, 1)
        result = _extract_prefix_attention(attn, prefix_len=1)
        assert result == pytest.approx(0.6)

    def test_only_prefix_positions_counted(self) -> None:
        """Attention split between prefix and non-prefix positions."""
        attn = torch.zeros(1, 1, 6, 6)
        # Last query position: 0.8 on prefix pos 0, 0.2 on non-prefix pos 3
        attn[0, 0, -1, 0] = 0.8
        attn[0, 0, -1, 3] = 0.2
        # prefix_len=1 → only position 0 counted
        result = _extract_prefix_attention(attn, prefix_len=1)
        assert result == pytest.approx(0.8)

    def test_multi_head_mean(self) -> None:
        """Mean is computed across all heads."""
        attn = torch.zeros(1, 2, 4, 4)
        attn[0, 0, -1, 0] = 1.0  # Head 0: all on prefix pos 0
        attn[0, 1, -1, 0] = 0.0  # Head 1: none on prefix pos 0
        # prefix_len=1, mean across heads = (1.0 + 0.0) / 2 = 0.5
        result = _extract_prefix_attention(attn, prefix_len=1)
        assert result == pytest.approx(0.5)


# ===========================================================================
# 3. _compute_attn_cosine_distance
# ===========================================================================


class TestComputeAttnCosineDistance:
    """Tests for _compute_attn_cosine_distance."""

    def test_identical_tensors_returns_zero(self) -> None:
        """Identical tensors → cosine distance = 0."""
        attn = torch.rand(1, 2, 4, 4)
        dist = _compute_attn_cosine_distance(attn, attn)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self) -> None:
        """Distance should be symmetric: d(a,b) == d(b,a)."""
        attn_a = torch.rand(1, 3, 5, 5)
        attn_b = torch.rand(1, 3, 5, 5)
        d_ab = _compute_attn_cosine_distance(attn_a, attn_b)
        d_ba = _compute_attn_cosine_distance(attn_b, attn_a)
        assert d_ab == pytest.approx(d_ba, abs=1e-6)

    def test_range_zero_to_two(self) -> None:
        """Cosine distance is in [0, 2] by definition."""
        rng = torch.manual_seed(99)
        for _ in range(20):
            attn_a = torch.rand(1, 2, 3, 3, generator=rng)
            attn_b = torch.rand(1, 2, 3, 3, generator=rng)
            dist = _compute_attn_cosine_distance(attn_a, attn_b)
            assert -1e-6 <= dist <= 2.0 + 1e-6, f"Distance {dist} outside [0, 2]"

    def test_opposite_tensors_returns_two(self) -> None:
        """Opposite tensors (a = -b) → cosine distance = 2.0."""
        attn_a = torch.rand(1, 2, 3, 3)
        attn_b = -attn_a
        dist = _compute_attn_cosine_distance(attn_a, attn_b)
        assert dist == pytest.approx(2.0, abs=1e-5)

    def test_same_direction_different_magnitude(self) -> None:
        """Scalars of same tensor → cosine distance = 0."""
        attn_a = torch.rand(1, 2, 3, 3)
        attn_b = attn_a * 5.0
        dist = _compute_attn_cosine_distance(attn_a, attn_b)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_zero_tensor_pair(self) -> None:
        """Two zero tensors: cosine_similarity returns 0 → distance = 1.0."""
        attn_a = torch.zeros(1, 2, 3, 3)
        attn_b = torch.zeros(1, 2, 3, 3)
        dist = _compute_attn_cosine_distance(attn_a, attn_b)
        assert dist == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# 4. Dataclass construction
# ===========================================================================


class TestKLDivergenceResult:
    """Tests for KLDivergenceResult dataclass."""

    def test_construction_and_field_access(self) -> None:
        """All fields should be accessible after construction."""
        result = KLDivergenceResult(
            step_kl_no_steer=[0.1, 0.2],
            step_kl_all_steer=[0.01, 0.02],
            generated_text_no_steer="no_steer text",
            generated_text_prefix_steer="prefix text",
            generated_text_all_steer="all_steer text",
            steer_tokens=10,
            scale=1.5,
            layer_frac=0.7,
            prompt="test prompt",
        )
        assert result.step_kl_no_steer == [0.1, 0.2]
        assert result.step_kl_all_steer == [0.01, 0.02]
        assert result.generated_text_no_steer == "no_steer text"
        assert result.generated_text_prefix_steer == "prefix text"
        assert result.generated_text_all_steer == "all_steer text"
        assert result.steer_tokens == 10
        assert result.scale == 1.5
        assert result.layer_frac == 0.7
        assert result.prompt == "test prompt"

    def test_empty_step_lists(self) -> None:
        """Empty step lists should be valid (edge case: no generation steps)."""
        result = _make_kl_result(step_kl_no_steer=[], step_kl_all_steer=[])
        assert result.step_kl_no_steer == []
        assert result.step_kl_all_steer == []


class TestAttentionAnalysisResult:
    """Tests for AttentionAnalysisResult dataclass."""

    def test_construction_and_field_access(self) -> None:
        """All fields should be accessible after construction."""
        result = AttentionAnalysisResult(
            attn_to_prefix_no_steer=[0.1],
            attn_to_prefix_prefix_steer=[0.2],
            attn_to_prefix_all_steer=[0.15],
            attn_cosine_shift=[0.05],
            steered_layer_attn_diff=[0.01],
            prompt_tokens=["hello", "world"],
            steer_tokens=5,
        )
        assert result.attn_to_prefix_no_steer == [0.1]
        assert result.attn_to_prefix_prefix_steer == [0.2]
        assert result.attn_to_prefix_all_steer == [0.15]
        assert result.attn_cosine_shift == [0.05]
        assert result.steered_layer_attn_diff == [0.01]
        assert result.prompt_tokens == ["hello", "world"]
        assert result.steer_tokens == 5

    def test_empty_metric_lists(self) -> None:
        """Empty metric lists should be valid (edge case: no generation steps)."""
        result = _make_attn_result(
            attn_to_prefix_no_steer=[],
            attn_to_prefix_prefix_steer=[],
            attn_to_prefix_all_steer=[],
            attn_cosine_shift=[],
            steered_layer_attn_diff=[],
            prompt_tokens=[],
        )
        assert result.attn_to_prefix_no_steer == []
        assert result.prompt_tokens == []


# ===========================================================================
# 5. generate_analysis_report
# ===========================================================================


class TestGenerateAnalysisReport:
    """Tests for generate_analysis_report."""

    def test_creates_markdown_file(self, tmp_path: Path) -> None:
        """Report should create a markdown file at the specified path."""
        output_path = tmp_path / "report.md"
        kl_results = [_make_kl_result()]
        attn_results = [_make_attn_result()]
        config: dict[str, str | int | float | bool] = {
            "model": "test-model",
            "scale": 1.0,
        }

        result = generate_analysis_report(
            kl_results=kl_results,
            attention_results=attn_results,
            config_dict=config,
            plot_paths=[],
            output_path=output_path,
        )

        assert result == output_path
        assert output_path.exists()

    def test_report_contains_expected_sections(self, tmp_path: str) -> None:
        """Report should contain all major markdown sections."""

        output_path = Path(tmp_path) / "sections_report.md"
        kl_results = [_make_kl_result()]
        attn_results = [_make_attn_result()]

        generate_analysis_report(
            kl_results=kl_results,
            attention_results=attn_results,
            config_dict={"model": "test"},
            plot_paths=[Path("/tmp/plot1.pdf")],
            output_path=output_path,
        )

        content = output_path.read_text()
        assert "# Prefix Steering Analysis Report" in content
        assert "## Configuration" in content
        assert "## Executive Summary" in content
        assert "## 1. KL Divergence Sweep: Prefix Length Analysis" in content
        assert "## 2. KL Divergence: Prefix Steering vs No Steering (Per-Step)" in content
        assert "## 3. KL Divergence: Prefix Steering vs All-Step Steering (Per-Step)" in content
        assert "## 4. Attention Path Analysis" in content
        assert "## 5. Attention Link Instances" in content
        assert "## 6. Key Findings" in content
        assert "## 7. Per-Prompt Details" in content
        assert "## Plots" in content
        assert "plot1.pdf" in content

    def test_report_with_empty_results(self, tmp_path: str) -> None:
        """Report with empty result lists should still produce valid markdown."""

        output_path = Path(tmp_path) / "empty_report.md"

        generate_analysis_report(
            kl_results=[],
            attention_results=[],
            config_dict={"model": "test"},
            plot_paths=[],
            output_path=output_path,
        )

        content = output_path.read_text()
        assert "# Prefix Steering Analysis Report" in content
        assert "No attention analysis results available" in content

    def test_report_config_table(self, tmp_path: str) -> None:
        """Config entries should appear as table rows."""

        output_path = Path(tmp_path) / "config_report.md"
        config: dict[str, str | int | float | bool] = {
            "model": "Qwen3-1.7B",
            "steer_tokens": 10,
            "scale": 1.5,
            "use_attention": True,
        }

        generate_analysis_report(
            kl_results=[_make_kl_result()],
            attention_results=[],
            config_dict=config,
            plot_paths=[],
            output_path=output_path,
        )

        content = output_path.read_text()
        assert "| model | Qwen3-1.7B |" in content
        assert "| steer_tokens | 10 |" in content
        assert "| scale | 1.5 |" in content
        assert "| use_attention | True |" in content

    def test_report_creates_parent_dirs(self, tmp_path: str) -> None:
        """Report should create parent directories if they don't exist."""

        output_path = Path(tmp_path) / "nested" / "dir" / "report.md"

        generate_analysis_report(
            kl_results=[_make_kl_result()],
            attention_results=[],
            config_dict={"model": "test"},
            plot_paths=[],
            output_path=output_path,
        )

        assert output_path.exists()


# ===========================================================================
# 6. plot_kl_divergence_curves
# ===========================================================================


class TestPlotKLDivergenceCurves:
    """Tests for plot_kl_divergence_curves."""

    def test_creates_pdf_files(self, tmp_path: str) -> None:
        """Should create two PDF files for KL plots."""

        output_dir = Path(tmp_path) / "kl_plots"
        results = [_make_kl_result(step_kl_no_steer=[0.1, 0.2, 0.3])]

        paths = plot_kl_divergence_curves(results, output_dir)

        assert len(paths) == 2
        for p in paths:
            assert p.suffix == ".pdf"
            assert p.exists()

    def test_returns_correct_filenames(self, tmp_path: str) -> None:
        """Returned paths should have expected filenames."""

        output_dir = Path(tmp_path) / "kl_names"
        results = [_make_kl_result()]

        paths = plot_kl_divergence_curves(results, output_dir)
        names = {p.name for p in paths}

        assert "kl_prefix_vs_no_steer.pdf" in names
        assert "kl_prefix_vs_all_steer.pdf" in names

    def test_empty_results_returns_empty_list(self, tmp_path: str) -> None:
        """Empty input list should return empty output list."""

        output_dir = Path(tmp_path) / "kl_empty"

        paths = plot_kl_divergence_curves([], output_dir)

        assert paths == []

    def test_multiple_results_varying_lengths(self, tmp_path: str) -> None:
        """Multiple results with different step counts should succeed."""

        output_dir = Path(tmp_path) / "kl_multi"
        results = [
            _make_kl_result(step_kl_no_steer=[0.1, 0.2, 0.3], step_kl_all_steer=[0.01, 0.02, 0.03]),
            _make_kl_result(step_kl_no_steer=[0.4, 0.5], step_kl_all_steer=[0.04, 0.05]),
        ]

        paths = plot_kl_divergence_curves(results, output_dir)

        assert len(paths) == 2
        for p in paths:
            assert p.exists()


# ===========================================================================
# 7. plot_attention_analysis
# ===========================================================================


class TestPlotAttentionAnalysis:
    """Tests for plot_attention_analysis."""

    def test_creates_pdf_files(self, tmp_path: str) -> None:
        """Should create two PDF files for attention plots."""

        output_dir = Path(tmp_path) / "attn_plots"
        results = [_make_attn_result()]

        paths = plot_attention_analysis(results, output_dir)

        assert len(paths) == 2
        for p in paths:
            assert p.suffix == ".pdf"
            assert p.exists()

    def test_returns_correct_filenames(self, tmp_path: str) -> None:
        """Returned paths should have expected filenames."""

        output_dir = Path(tmp_path) / "attn_names"
        results = [_make_attn_result()]

        paths = plot_attention_analysis(results, output_dir)
        names = {p.name for p in paths}

        assert "attention_to_prefix.pdf" in names
        assert "attention_cosine_shift.pdf" in names

    def test_empty_results_returns_empty_list(self, tmp_path: str) -> None:
        """Empty input list should return empty output list."""

        output_dir = Path(tmp_path) / "attn_empty"

        paths = plot_attention_analysis([], output_dir)

        assert paths == []

    def test_multiple_results_varying_lengths(self, tmp_path: str) -> None:
        """Multiple results with different step counts should succeed."""

        output_dir = Path(tmp_path) / "attn_multi"
        results = [
            _make_attn_result(
                attn_to_prefix_no_steer=[0.1, 0.2, 0.3],
                attn_cosine_shift=[0.05, 0.06, 0.07],
            ),
            _make_attn_result(
                attn_to_prefix_no_steer=[0.4, 0.5],
                attn_cosine_shift=[0.08, 0.09],
            ),
        ]

        paths = plot_attention_analysis(results, output_dir)

        assert len(paths) == 2
        for p in paths:
            assert p.exists()


# ===========================================================================
# 8. AttentionLinkInstance dataclass
# ===========================================================================


class TestAttentionLinkInstance:
    """Tests for AttentionLinkInstance dataclass."""

    def test_construction_and_field_access(self) -> None:
        """All fields should be accessible after construction."""
        link = AttentionLinkInstance(
            layer_idx=3,
            head_idx=1,
            step=7,
            attn_to_prefix_no_steer=0.12,
            attn_to_prefix_steer=0.45,
            attn_change=0.33,
            top_prefix_position=2,
            top_prefix_token="hello",
            top_prefix_attn_no_steer=0.05,
            top_prefix_attn_steer=0.20,
        )
        assert link.layer_idx == 3
        assert link.head_idx == 1
        assert link.step == 7
        assert link.attn_to_prefix_no_steer == pytest.approx(0.12)
        assert link.attn_to_prefix_steer == pytest.approx(0.45)
        assert link.attn_change == pytest.approx(0.33)
        assert link.top_prefix_position == 2
        assert link.top_prefix_token == "hello"
        assert link.top_prefix_attn_no_steer == pytest.approx(0.05)
        assert link.top_prefix_attn_steer == pytest.approx(0.20)

    def test_negative_change(self) -> None:
        """attn_change can be negative when steering reduces prefix attention."""
        link = AttentionLinkInstance(
            layer_idx=0,
            head_idx=0,
            step=0,
            attn_to_prefix_no_steer=0.5,
            attn_to_prefix_steer=0.2,
            attn_change=-0.3,
            top_prefix_position=0,
            top_prefix_token="x",
            top_prefix_attn_no_steer=0.5,
            top_prefix_attn_steer=0.2,
        )
        assert link.attn_change == pytest.approx(-0.3)

    def test_zero_values(self) -> None:
        """All-zero numeric fields should be valid."""
        link = AttentionLinkInstance(
            layer_idx=0,
            head_idx=0,
            step=0,
            attn_to_prefix_no_steer=0.0,
            attn_to_prefix_steer=0.0,
            attn_change=0.0,
            top_prefix_position=0,
            top_prefix_token="",
            top_prefix_attn_no_steer=0.0,
            top_prefix_attn_steer=0.0,
        )
        assert link.attn_change == 0.0
        assert link.top_prefix_token == ""


# ===========================================================================
# 9. _extract_attention_links
# ===========================================================================


class TestExtractAttentionLinks:
    """Tests for _extract_attention_links."""

    def test_empty_dicts_returns_empty_list(self) -> None:
        """Empty attention dicts should produce no links."""
        result = _extract_attention_links({}, {}, ["a", "b"], prefix_len=2, step=0)
        assert result == []

    def test_identical_attention_all_changes_zero(self) -> None:
        """Identical no_steer and steer dicts → every attn_change should be 0."""
        seq_len, num_heads = 6, 2
        attn = torch.ones(1, num_heads, seq_len, seq_len, dtype=torch.float32) / seq_len
        attn_dicts: dict[int, torch.Tensor] = {0: attn}
        links = _extract_attention_links(
            attn_dicts, attn_dicts, ["t0", "t1", "t2"], prefix_len=3, step=1
        )
        assert len(links) > 0
        for link in links:
            assert link.attn_change == pytest.approx(0.0, abs=1e-6)

    def test_returns_at_most_top_k(self) -> None:
        """Result list should be capped at top_k."""
        seq_len, num_heads, num_layers = 10, 4, 3
        attn_no = torch.ones(1, num_heads, seq_len, seq_len, dtype=torch.float32) / seq_len
        attn_steer = attn_no.clone()
        attn_steer[0, :, :, :3] += 0.1
        no_dict = {i: attn_no.clone() for i in range(num_layers)}
        steer_dict = {i: attn_steer.clone() for i in range(num_layers)}
        tokens = [f"t{j}" for j in range(seq_len)]

        links = _extract_attention_links(no_dict, steer_dict, tokens, prefix_len=3, step=0, top_k=5)
        assert len(links) <= 5

    def test_sorted_by_abs_attn_change_descending(self) -> None:
        """Links should be sorted by |attn_change| in descending order."""
        seq_len, num_heads = 8, 3
        tokens = [f"t{j}" for j in range(seq_len)]

        attn_no = torch.zeros(1, num_heads, seq_len, seq_len, dtype=torch.float32)
        attn_steer = torch.zeros(1, num_heads, seq_len, seq_len, dtype=torch.float32)
        # Head 0: small change, head 1: large change, head 2: medium change
        attn_no[0, :, -1, :3] = 0.1
        attn_steer[0, 0, -1, :3] = 0.15  # change = 0.05
        attn_steer[0, 1, -1, :3] = 0.50  # change = 0.40
        attn_steer[0, 2, -1, :3] = 0.30  # change = 0.20

        links = _extract_attention_links(
            {0: attn_no}, {0: attn_steer}, tokens, prefix_len=3, step=0
        )
        changes = [abs(lk.attn_change) for lk in links]
        assert changes == sorted(changes, reverse=True)

    def test_specific_values_single_head(self) -> None:
        """Verify computed values for a simple single-head case."""
        seq_len = 5
        attn_no = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32)
        attn_steer = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32)
        # Last query position, prefix positions 0..2
        attn_no[0, 0, -1, 0] = 0.1
        attn_no[0, 0, -1, 1] = 0.2
        attn_no[0, 0, -1, 2] = 0.1
        attn_steer[0, 0, -1, 0] = 0.3
        attn_steer[0, 0, -1, 1] = 0.4
        attn_steer[0, 0, -1, 2] = 0.2

        links = _extract_attention_links(
            {0: attn_no},
            {0: attn_steer},
            ["a", "b", "c", "d", "e"],
            prefix_len=3,
            step=5,
        )
        assert len(links) == 1
        link = links[0]
        # mean_no = (0.1 + 0.2 + 0.1) / 3 ≈ 0.1333
        # mean_steer = (0.3 + 0.4 + 0.2) / 3 = 0.3
        # change ≈ 0.1667
        assert link.attn_to_prefix_no_steer == pytest.approx((0.1 + 0.2 + 0.1) / 3, rel=1e-4)
        assert link.attn_to_prefix_steer == pytest.approx((0.3 + 0.4 + 0.2) / 3, rel=1e-4)
        assert link.attn_change == pytest.approx(
            (0.3 + 0.4 + 0.2) / 3 - (0.1 + 0.2 + 0.1) / 3, rel=1e-4
        )
        assert link.step == 5
        assert link.layer_idx == 0
        assert link.head_idx == 0
        # Per-position diff: 0.2, 0.2, 0.1 → argmax = 0 (first max)
        assert link.top_prefix_position == 0
        assert link.top_prefix_token == "a"

    def test_prefix_len_zero_returns_empty(self) -> None:
        """prefix_len=0 → clamped to 0, no candidates produced."""
        attn = torch.ones(1, 2, 5, 5, dtype=torch.float32) / 5
        links = _extract_attention_links({0: attn}, {0: attn}, ["a"], prefix_len=0, step=0)
        assert links == []

    def test_only_common_layers_processed(self) -> None:
        """Only layers present in both dicts should be processed."""
        seq_len, num_heads = 4, 1
        attn = torch.ones(1, num_heads, seq_len, seq_len, dtype=torch.float32) / seq_len
        # Layer 0 in both, layer 1 only in no_steer, layer 2 only in steer
        links = _extract_attention_links(
            {0: attn, 1: attn},
            {0: attn, 2: attn},
            ["a", "b"],
            prefix_len=2,
            step=0,
        )
        for lk in links:
            assert lk.layer_idx == 0


# ===========================================================================
# 10. plot_attention_link_heatmap
# ===========================================================================


class TestPlotAttentionLinkHeatmap:
    """Tests for plot_attention_link_heatmap."""

    def _make_link(
        self,
        layer_idx: int = 0,
        head_idx: int = 0,
        attn_change: float = 0.1,
    ) -> AttentionLinkInstance:
        return AttentionLinkInstance(
            layer_idx=layer_idx,
            head_idx=head_idx,
            step=0,
            attn_to_prefix_no_steer=0.1,
            attn_to_prefix_steer=0.2,
            attn_change=attn_change,
            top_prefix_position=0,
            top_prefix_token="x",
            top_prefix_attn_no_steer=0.1,
            top_prefix_attn_steer=0.2,
        )

    def test_creates_pdf_file(self, tmp_path: Path) -> None:
        """Should create a PDF heatmap file."""
        links = [self._make_link(0, 0, 0.15), self._make_link(1, 1, -0.1)]
        output_dir = tmp_path / "heatmap"

        result = plot_attention_link_heatmap(links, steer_tokens=5, output_dir=output_dir)

        assert result.suffix == ".pdf"
        assert result.exists()

    def test_returns_correct_path(self, tmp_path: Path) -> None:
        """Returned path should be attention_link_heatmap.pdf inside output_dir."""
        output_dir = tmp_path / "heatmap_path"

        result = plot_attention_link_heatmap([], steer_tokens=3, output_dir=output_dir)

        assert result.name == "attention_link_heatmap.pdf"
        assert result.parent == output_dir

    def test_empty_links_does_not_crash(self, tmp_path: Path) -> None:
        """Empty links list should still produce a valid PDF."""
        output_dir = tmp_path / "heatmap_empty"

        result = plot_attention_link_heatmap([], steer_tokens=5, output_dir=output_dir)

        assert result.exists()

    def test_links_populate_matrix(self, tmp_path: Path) -> None:
        """Links with various layers/heads should succeed without error."""
        links = [
            self._make_link(0, 0, 0.3),
            self._make_link(0, 1, -0.2),
            self._make_link(1, 0, 0.1),
            self._make_link(2, 3, 0.5),
            self._make_link(4, 7, -0.4),
        ]
        output_dir = tmp_path / "heatmap_populate"

        result = plot_attention_link_heatmap(
            links, steer_tokens=5, output_dir=output_dir, num_layers=5, num_heads=8
        )

        assert result.exists()


# ===========================================================================
# 11. run_prefix_analysis output layout
# ===========================================================================


class TestRunPrefixAnalysisOutputLayout:
    """Tests for run_prefix_analysis output layout."""

    def test_writes_pdfs_and_report_directly_in_model_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Run output should not create a nested plots directory."""

        class FakeHookedModel:
            def resolve_layers(self, layer_fracs: list[float]) -> list[int]:
                return [0 for _ in layer_fracs]

        base_dir = tmp_path / "prefix_analysis"
        model_name = "Qwen/Qwen3-1.7B"
        concept = "sentiment"
        expected_dir = base_dir / concept / "Qwen_Qwen3-1.7B"
        plot_dirs: list[Path] = []
        report_paths: list[Path] = []

        def fake_load_contrast_pairs(
            requested_concept: str,
            num_prompts: int,
            data_mode: str | None = None,
        ) -> list[ContrastPair]:
            return [
                ContrastPair(
                    positive=f"positive {idx}",
                    negative=f"negative {idx}",
                    metadata={"concept": requested_concept, "dataset": data_mode or "default"},
                )
                for idx in range(num_prompts)
            ]

        def fake_run_prefix_length_kl_sweep(
            model: HookedModel,
            prompts: list[str],
            steering_vector: torch.Tensor,
            layer_idx: int,
            layer_frac: float,
            scale: float,
            steer_tokens_list: list[int] | None = None,
            temperature: float = 0.0,
            num_post_steer_steps: int = 1,
        ) -> PrefixLengthKLSweepResult:
            return PrefixLengthKLSweepResult(
                steer_tokens_list=[0],
                kl_vs_no_steer={0: [[0.1]]},
                kl_vs_all_steer={0: [[0.2]]},
                layer_frac=layer_frac,
                scale=scale,
                num_prompts=len(prompts),
                num_post_steer_steps=num_post_steer_steps,
            )

        def fake_plot_prefix_length_kl_sweep(
            result: PrefixLengthKLSweepResult,
            output_dir: Path,
        ) -> list[Path]:
            plot_dirs.append(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / "kl_prefix_length_sweep.pdf"
            plot_path.write_text("pdf")
            return [plot_path]

        def fake_plot_kl_divergence_curves(
            results: list[KLDivergenceResult],
            output_dir: Path,
        ) -> list[Path]:
            plot_dirs.append(output_dir)
            return []

        def fake_generate_analysis_report(
            kl_results: list[KLDivergenceResult],
            attention_results: list[AttentionAnalysisResult],
            config_dict: dict[str, str | int | float | bool],
            plot_paths: list[Path],
            output_path: Path,
            kl_sweep_result: PrefixLengthKLSweepResult | None = None,
        ) -> Path:
            report_paths.append(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("report")
            return output_path

        monkeypatch.setattr(prefix_analysis, "HookedModel", lambda config: FakeHookedModel())
        monkeypatch.setattr(prefix_analysis.torch, "load", lambda *args, **kwargs: torch.ones(8))
        monkeypatch.setattr(prefix_analysis, "load_contrast_pairs", fake_load_contrast_pairs)
        monkeypatch.setattr(prefix_analysis, "_compute_avg_activation", lambda *args: 1.0)
        monkeypatch.setattr(
            prefix_analysis,
            "run_prefix_length_kl_sweep",
            fake_run_prefix_length_kl_sweep,
        )
        monkeypatch.setattr(
            prefix_analysis,
            "plot_prefix_length_kl_sweep",
            fake_plot_prefix_length_kl_sweep,
        )
        monkeypatch.setattr(
            prefix_analysis,
            "plot_kl_divergence_curves",
            fake_plot_kl_divergence_curves,
        )
        monkeypatch.setattr(
            prefix_analysis, "generate_analysis_report", fake_generate_analysis_report
        )

        prefix_analysis.run_prefix_analysis(
            model_name=model_name,
            concept=concept,
            vector_path=tmp_path / "vector.pt",
            run_attention=False,
            output_dir=base_dir,
        )

        assert plot_dirs == [expected_dir, expected_dir]
        assert report_paths == [expected_dir / "analysis_report.md"]
        assert (expected_dir / "kl_prefix_length_sweep.pdf").exists()
        assert (expected_dir / "analysis_report.md").exists()
        assert not (expected_dir / "plots").exists()

    def test_forwards_steer_tokens_list_and_num_post_steer_steps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_prefix_analysis should forward the two sweep kwargs verbatim."""

        class FakeHookedModel:
            def resolve_layers(self, layer_fracs: list[float]) -> list[int]:
                return [0 for _ in layer_fracs]

        captured: dict[str, object] = {}

        def fake_load_contrast_pairs(
            requested_concept: str,
            num_prompts: int,
            data_mode: str | None = None,
        ) -> list[ContrastPair]:
            return [
                ContrastPair(
                    positive=f"positive {idx}",
                    negative=f"negative {idx}",
                    metadata={"concept": requested_concept},
                )
                for idx in range(num_prompts)
            ]

        def fake_run_prefix_length_kl_sweep(
            model: HookedModel,
            prompts: list[str],
            steering_vector: torch.Tensor,
            layer_idx: int,
            layer_frac: float,
            scale: float,
            steer_tokens_list: list[int] | None = None,
            temperature: float = 0.0,
            num_post_steer_steps: int = 1,
        ) -> PrefixLengthKLSweepResult:
            captured["steer_tokens_list"] = list(steer_tokens_list or [])
            captured["num_post_steer_steps"] = num_post_steer_steps
            return PrefixLengthKLSweepResult(
                steer_tokens_list=list(steer_tokens_list or []),
                kl_vs_no_steer={0: [[0.1]]},
                kl_vs_all_steer={0: [[0.2]]},
                layer_frac=layer_frac,
                scale=scale,
                num_prompts=len(prompts),
                num_post_steer_steps=num_post_steer_steps,
            )

        def fake_plot_prefix_length_kl_sweep(
            result: PrefixLengthKLSweepResult,
            output_dir: Path,
        ) -> list[Path]:
            output_dir.mkdir(parents=True, exist_ok=True)
            return [output_dir / "kl_prefix_length_sweep.pdf"]

        monkeypatch.setattr(prefix_analysis, "HookedModel", lambda config: FakeHookedModel())
        monkeypatch.setattr(prefix_analysis.torch, "load", lambda *args, **kwargs: torch.ones(8))
        monkeypatch.setattr(prefix_analysis, "load_contrast_pairs", fake_load_contrast_pairs)
        monkeypatch.setattr(prefix_analysis, "_compute_avg_activation", lambda *args: 1.0)
        monkeypatch.setattr(
            prefix_analysis,
            "run_prefix_length_kl_sweep",
            fake_run_prefix_length_kl_sweep,
        )
        monkeypatch.setattr(
            prefix_analysis,
            "plot_prefix_length_kl_sweep",
            fake_plot_prefix_length_kl_sweep,
        )
        monkeypatch.setattr(prefix_analysis, "plot_kl_divergence_curves", lambda *a, **k: [])
        monkeypatch.setattr(
            prefix_analysis, "generate_analysis_report", lambda *a, **k: tmp_path / "report.md"
        )

        prefix_analysis.run_prefix_analysis(
            model_name="Qwen/Qwen3-1.7B",
            concept="sentiment",
            vector_path=tmp_path / "vector.pt",
            steer_tokens_list=[0, 5, 10],
            num_post_steer_steps=3,
            run_attention=False,
            output_dir=tmp_path / "prefix_analysis",
        )

        assert captured["steer_tokens_list"] == [0, 5, 10]
        assert captured["num_post_steer_steps"] == 3


# ===========================================================================
# 12. _make_steering_hook
# ===========================================================================


class TestMakeSteeringHook:
    """Tests for _make_steering_hook and off-by-one fix."""

    def test_applies_steering_for_exactly_steer_tokens_calls(self) -> None:
        """Hook should apply steering for exactly steer_tokens forward passes."""
        sv = torch.ones(4)
        scale = 2.0
        steer_tokens = 3
        step_counter = [0]
        hook = _make_steering_hook(sv, scale, steer_tokens, step_counter)

        base_output = torch.zeros(1, 4)
        for i in range(5):
            result = hook(object(), object(), base_output.clone())
            if i < steer_tokens:
                assert torch.allclose(result, base_output + sv * scale), (
                    f"Step {i + 1}: expected steering applied"
                )
            else:
                assert torch.allclose(result, base_output), f"Step {i + 1}: expected no steering"

    def test_none_steer_tokens_applies_forever(self) -> None:
        """steer_tokens=None should apply steering to all calls."""
        sv = torch.ones(4)
        scale = 1.0
        step_counter = [0]
        hook = _make_steering_hook(sv, scale, None, step_counter)

        base_output = torch.zeros(1, 4)
        for _ in range(10):
            result = hook(object(), object(), base_output.clone())
            assert torch.allclose(result, base_output + sv * scale)

    def test_zero_steer_tokens_applies_no_steering(self) -> None:
        """steer_tokens=0 should never apply steering."""
        sv = torch.ones(4)
        step_counter = [0]
        hook = _make_steering_hook(sv, 1.0, 0, step_counter)

        base_output = torch.zeros(1, 4)
        result = hook(object(), object(), base_output.clone())
        assert torch.allclose(result, base_output)

    def test_counter_increments_on_each_call(self) -> None:
        """step_counter should increment by 1 on each hook invocation."""
        sv = torch.ones(4)
        step_counter = [0]
        hook = _make_steering_hook(sv, 1.0, None, step_counter)

        for expected in range(1, 6):
            hook(object(), object(), torch.zeros(1, 4))
            assert step_counter[0] == expected

    def test_tuple_output_with_steering(self) -> None:
        """Hook should handle tuple output, adding steering to first element."""
        sv = torch.ones(4)
        scale = 1.5
        step_counter = [0]
        hook = _make_steering_hook(sv, scale, None, step_counter)

        base_tensor = torch.zeros(1, 4)
        extra = torch.zeros(1, 4)
        result = hook(object(), object(), (base_tensor, extra))

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.allclose(result[0], base_tensor + sv * scale)
        assert result[1] is extra

    def test_tuple_output_passthrough_when_expired(self) -> None:
        """Expired hook should return tuple output unchanged."""
        sv = torch.ones(4)
        step_counter = [0]
        hook = _make_steering_hook(sv, 1.0, 1, step_counter)

        base_tensor = torch.zeros(1, 4)
        extra = torch.zeros(1, 4)
        hook(object(), object(), base_tensor.clone())
        result = hook(object(), object(), (base_tensor, extra))
        assert isinstance(result, tuple)
        assert result[0] is base_tensor
        assert result[1] is extra

    def test_only_generation_steps_steered(self) -> None:
        """Prefill is not hooked; steer_tokens=N steers exactly N generation steps.

        The hook is registered AFTER prefill, so only generation forward passes
        go through the hook. steer_tokens=N gives exactly N steered gen steps.
        """
        sv = torch.ones(4)
        scale = 1.0
        steer_tokens = 3
        step_counter = [0]
        hook = _make_steering_hook(sv, scale, steer_tokens, step_counter)

        base_output = torch.zeros(1, 4)
        steered_count = 0
        total_gen_steps = 6

        for _ in range(total_gen_steps):
            result = hook(object(), object(), base_output.clone())
            if not torch.allclose(result, base_output):
                steered_count += 1

        assert steered_count == steer_tokens, (
            f"Expected exactly {steer_tokens} steered gen steps, got {steered_count}"
        )

    def test_steers_only_last_position_prefill(self) -> None:
        """During prefill (seq_len > 1), only the last token position is steered.

        Earlier positions must remain unchanged so the KV cache for prompt
        tokens stays clean.  Only the last position (which produces the
        next-token logits) gets the steering vector added.
        """
        sv = torch.ones(4)
        scale = 2.0
        step_counter = [0]
        hook = _make_steering_hook(sv, scale, None, step_counter)

        base = torch.zeros(1, 5, 4)
        result = hook(object(), object(), base)

        assert torch.allclose(result[:, :-1, :], base[:, :-1, :]), (
            "Earlier positions must not be steered"
        )
        expected_last = base[:, -1, :] + sv * scale
        assert torch.allclose(result[:, -1, :], expected_last), "Last position must be steered"


# ===========================================================================
# 13. _compute_avg_activation
# ===========================================================================


class TestComputeAvgActivation:
    """Tests for _compute_avg_activation."""

    def test_uses_provided_texts_not_hardcoded_dummy(
        self,
        mock_hooked_model: HookedModel,
    ) -> None:
        """Function should pass the provided texts to get_activations, not a
        hardcoded dummy prompt.

        Spies on ``model.get_activations`` to verify the ``texts`` argument
        matches the caller-provided list exactly.
        """
        from unittest.mock import MagicMock

        texts = ["I love this product!", "This is the worst experience ever."]
        steering_vector = torch.ones(8)

        spy = MagicMock(wraps=mock_hooked_model.get_activations)
        original = mock_hooked_model.get_activations
        mock_hooked_model.get_activations = spy  # type: ignore[method-assign]
        try:
            result = _compute_avg_activation(
                mock_hooked_model,
                texts,
                layer_idx=0,
                steering_vector=steering_vector,
            )
        finally:
            mock_hooked_model.get_activations = original  # type: ignore[method-assign]

        spy.assert_called_once_with(texts, [0])
        assert isinstance(result, float)
        assert result > 0.0

    def test_returns_positive_float(
        self,
        mock_hooked_model: HookedModel,
    ) -> None:
        """Result should be a positive float (activation norms are positive)."""
        texts = ["a simple prompt for testing"]
        steering_vector = torch.ones(8)
        result = _compute_avg_activation(
            mock_hooked_model,
            texts,
            layer_idx=0,
            steering_vector=steering_vector,
        )
        assert isinstance(result, float)
        assert result > 0.0

    def test_empty_texts_falls_back_to_steering_vector_norm(
        self,
        mock_hooked_model: HookedModel,
    ) -> None:
        """Empty texts list → get_activations returns empty → should fall back
        to steering_vector.norm() rather than crash."""
        steering_vector = torch.tensor([3.0, 4.0])  # norm = 5.0
        result = _compute_avg_activation(
            mock_hooked_model,
            [],
            layer_idx=0,
            steering_vector=steering_vector,
        )
        assert result == pytest.approx(5.0, rel=1e-4)
