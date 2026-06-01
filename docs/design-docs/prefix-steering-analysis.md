# Prefix Steering Analysis Experiments

**Status:** Approved
**Date:** 2026-06
**Branch:** experiment/pipeline

## Context

Standard activation steering applies the steering vector at every generation step. The paper shows that most steering effects appear within the first few generated tokens. Later interventions add distributional shift, which can hurt general capability without adding much concept control.

Prefix Steering applies the update only to an initial generation prefix. The main question is whether a short intervention window can recover most of the steering gain while preserving MMLU-Pro accuracy.

These experiments cover three paper claims:

| Experiment | Figures | Question | Main result |
|------------|---------|----------|-------------|
| Prefix-Length Ablation, KL and Trade-off | `kl_and_tradeoff_early_token.pdf` | How does behavior change as the number of steered tokens grows? | KL rises sharply in early tokens and then saturates. MMLU-Pro falls as the intervention window grows. |
| Prefix vs All-Token Steering | `early_token_intervention.pdf`, `early_token_intervention_sentiment.pdf`, `early_token_intervention_polite.pdf` | Does prefix-5 match full steering across models and concepts? | Prefix-5 recovers most concept gain while preserving more MMLU-Pro accuracy than all-token steering. |
| Attention Pattern Analysis | Supporting analysis figures | Why can a short prefix intervention affect later generation? | Steering shifts attention toward prefix positions, which lets early tokens carry the control signal forward. |

The paper evaluates four model families and three concepts:

| Category | Values |
|----------|--------|
| Models | OLMo3-7B, OLMo3-32B, Qwen3-1.7B, Qwen3-14B |
| Concepts | safety/refusal, sentiment, politeness |
| Metrics | KL divergence per generation step, concept score, MMLU-Pro accuracy, attention cosine distance |

## Decision

### Prefix steering control surface

Prefix Steering is controlled by the `steer_tokens` parameter in `HookedModel.generate_with_steering()`.

| `steer_tokens` value | Behavior |
|----------------------|----------|
| `None` | Apply steering at every generation step, matching all-token steering. |
| `0` | Apply no steering. |
| Positive integer `N` | Apply steering only during the first `N` generation steps. |
| `N >= max_new_tokens` | Equivalent to all-token steering for that generation. |

The generation hook increments a step counter each time the steered layer runs. Once the counter exceeds `steer_tokens`, the hook returns the model output unchanged. This keeps the intervention model-independent and makes prefix length a direct experiment variable.

### Core modules

| Module | Role |
|--------|------|
| `src/steering_geometry/models.py` | Provides `HookedModel.generate_with_steering()` and the `steer_tokens` intervention window. |
| `src/steering_geometry/prefix_analysis.py` | Runs KL divergence, prefix-length sweeps, attention analysis, plotting, and report generation. |
| `src/steering_geometry/token_selection_experiments.py` | Runs the prefix vs all-token steering scope comparison used for paper trade-off figures. |
| `src/steering_geometry/sweep_evaluation.py` | Provides the comprehensive evaluation layer for steering strength by prefix token count sweeps. |

### Data classes

`prefix_analysis.py` stores experiment outputs in explicit dataclasses.

| Data class | Main fields | Purpose |
|------------|-------------|---------|
| `KLDivergenceResult` | `step_kl_no_steer`, `step_kl_all_steer`, generated text for no-steer, prefix-steer, and all-steer runs, `steer_tokens`, `scale`, `layer_frac`, `prompt` | Stores per-prompt KL curves comparing prefix steering with both baselines. |
| `PrefixLengthKLSweepResult` | `steer_tokens_list`, `kl_vs_no_steer`, `kl_vs_all_steer`, `layer_frac`, `scale`, `num_prompts`, `num_post_steer_steps` | Stores KL values after the steering window ends for each prefix length. |
| `AttentionLinkInstance` | `layer_idx`, `head_idx`, `step`, prefix attention before and after steering, top prefix token and position | Stores concrete per-head examples of attention shifting to prefix tokens. |
| `AttentionAnalysisResult` | Prefix attention curves for no-steer, prefix-steer, and all-steer runs, `attn_cosine_shift`, `steered_layer_attn_diff`, `prompt_tokens`, `steer_tokens`, `attention_links` | Stores attention mechanism measurements for a single prompt. |
| `PrefixAnalysisReport` | `kl_results`, `kl_sweep_result`, `attention_results`, `config_dict` | Bundles all analysis outputs and run configuration. |

### Analysis functions

| Function | Role | Output |
|----------|------|--------|
| `per_token_kl_divergence()` | Computes `KL(P || Q)` from two logit tensors using `log_softmax` in float32. | Scalar KL value in nats. |
| `run_kl_divergence_experiment()` | Generates no-steer, prefix-steer, and all-steer continuations for each prompt. | List of `KLDivergenceResult`. |
| `run_prefix_length_kl_sweep()` | Sweeps multiple prefix lengths and measures KL at the first post-prefix steps. | `PrefixLengthKLSweepResult`. |
| `run_attention_analysis()` | Compares attention to prefix positions under no-steer, prefix-steer, and all-steer runs. | List of `AttentionAnalysisResult`. |
| `plot_kl_divergence_curves()` | Plots per-step KL curves over generation. | KL curve figure files. |
| `plot_prefix_length_kl_sweep()` | Plots KL as a function of prefix length. | Prefix-length sweep figure files. |
| `plot_attention_analysis()` | Plots aggregate attention shifts and cosine distance. | Attention summary figure files. |
| `plot_attention_link_heatmap()` | Plots concrete attention links for heads with the largest prefix attention increase. | Attention link heatmap files. |
| `generate_analysis_report()` | Writes a combined analysis report from KL and attention outputs. | Markdown report. |
| `run_prefix_analysis()` | Top-level orchestration for loading data, running analyses, plotting, and reporting. | `PrefixAnalysisReport`. |
| `run_steering_scope_experiment()` | Compares prefix steering against all-token steering across concepts and models. | Steering scope experiment outputs for trade-off figures. |

### KL divergence measurement

The KL experiment compares three generation modes for the same prompt:

| Mode | `steer_tokens` | Meaning |
|------|----------------|---------|
| No steering | `0` | Baseline model distribution. |
| Prefix steering | `N` | Steering applies only to the first `N` generated tokens. |
| All-token steering | `None` | Steering applies at every generated token. |

For each generation step, the analysis computes two values:

| Metric | Interpretation |
|--------|----------------|
| `KL(prefix_steer || no_steer)` | How far the prefix-steered distribution has moved from the base model. |
| `KL(prefix_steer || all_steer)` | How close prefix steering is to full steering. |

The prefix-length sweep then asks what happens after steering stops. For each prefix length `N`, it records KL at steps `N+1` through `N+K`. This isolates carryover from the early intervention window rather than measuring tokens that are still directly steered.

### Attention mechanism analysis

The attention analysis tests the proposed mechanism: early steered tokens become anchors that later tokens attend to. For each prompt, it measures attention to prefix positions under no steering, prefix steering, and all-token steering.

| Attention measure | Meaning |
|-------------------|---------|
| `attn_to_prefix_no_steer` | Fraction of attention assigned to prefix positions without steering. |
| `attn_to_prefix_prefix_steer` | Fraction of attention assigned to prefix positions after prefix steering. |
| `attn_to_prefix_all_steer` | Fraction of attention assigned to prefix positions under full intervention. |
| `attn_cosine_shift` | Cosine distance between prefix-steered and unsteered attention distributions. |
| `steered_layer_attn_diff` | Mean absolute attention difference at the steered layer. |
| `attention_links` | Per-head examples showing which prefix tokens gained attention. |

This analysis explains why prefix-only updates can affect later text even after direct steering has stopped.

### Sweep evaluation layer

`sweep_evaluation.py` builds on the same `steer_tokens` control surface. It evaluates a grid of steering strengths and prefix token counts, then scores each cell on concept adherence and MMLU-Pro accuracy.

| Axis | Meaning |
|------|---------|
| Steering strength | Multiplier applied to the steering vector. |
| Prefix token count | Number of early generation steps receiving the intervention. |

This module is the comprehensive evaluation layer for prefix steering because it combines intervention length with evaluation quality. The prefix analysis module explains how and why prefix steering works. The sweep evaluation module measures which settings give the best concept and capability trade-off.

### Scripts

| Script | Purpose | Typical output |
|--------|---------|----------------|
| `scripts/prefix_analysis/run_analysis.sh` | Runs KL divergence and attention analysis for a configured model and concept. | KL curves, attention plots, analysis report. |
| `scripts/prefix_analysis/run_all_concepts.sh` | Runs prefix analysis for safety/refusal, sentiment, and politeness. | Cross-concept analysis outputs. |
| `scripts/token_experiments/4_steering_scope.sh` | Runs prefix vs all-token steering scope comparison. | Scope comparison data for paper trade-off figures. |

### Figures

| Figure | Source experiment | What it shows |
|--------|-------------------|---------------|
| `kl_and_tradeoff_early_token.pdf` | Prefix-length KL sweep plus trade-off evaluation. | KL rises early and saturates, while longer steering windows reduce MMLU-Pro accuracy. |
| `early_token_intervention.pdf` | Prefix vs all-token steering for safety/refusal. | Prefix-5 keeps most steering gain and preserves more capability. |
| `early_token_intervention_sentiment.pdf` | Prefix vs all-token steering for sentiment. | Prefix-5 captures most sentiment steering with less capability loss. |
| `early_token_intervention_polite.pdf` | Prefix vs all-token steering for politeness. | Prefix-5 captures most politeness steering with less capability loss. |
| Attention analysis plots | `run_attention_analysis()` and plotting helpers. | Steering increases attention to early prefix positions. |

## Consequences

### Positive

- Prefix length is a simple, model-independent control knob.
- A 5-token prefix works well across the four paper models and three concepts.
- KL curves give token-level evidence that early interventions carry most of the steering effect.
- Attention analysis gives a mechanism-level explanation for why the prefix can affect later generation.
- The same `steer_tokens` API supports analysis runs, steering scope experiments, and full sweep evaluation.

### Limitations

- A fixed prefix length is not optimal for every prompt, model, concept, or steering strength.
- KL measurement requires access to model logits, so it is more expensive than scoring generated text alone.
- Attention measurements require attention outputs and can add memory cost on larger models.
- Prefix-5 is a strong default for paper experiments, but the sweep evaluation should be used when selecting settings for a new scenario.
