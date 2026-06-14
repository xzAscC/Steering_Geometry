# Robust DiM Extraction Experiments

**Status:** Approved
**Date:** 2026-06
**Branch:** experiment/pipeline

## Context

The paper introduces Robust DiM, a token selective direction construction method for activation steering. Standard DiM averages all candidate activations, which can preserve stable structure that is not useful for intervention. Robust DiM instead selects high margin activations that are representative of their own class and discriminative from the opposite class.

The extraction experiments test whether this selection rule produces directions that steer the target concept while preserving general capability. The same extracted direction is used by the existing Prefix Steering intervention, so Robust DiM changes direction construction only, not the intervention rule.

## Decision

### Core algorithm: `discriminative_token_aggregator`

`src/steering_geometry/extract.py` implements Robust DiM in `discriminative_token_aggregator(pos, neg, top_k=100)`. The function receives positive and negative activation tensors, computes class means, scores each candidate activation by relative margin, selects the top K activations per class, and returns the difference between the selected class means.

For a positive activation `h`, the score is:

```text
s(h) = (||h - mu_neg||^2 - ||h - mu_pos||^2) / (||h - mu_neg||^2 + ||h - mu_pos||^2)
```

The negative class uses the symmetric score with positive and negative means swapped. This implements Eq. 6 and Eq. 7 from the paper: high scoring tokens are close to their own class mean and far from the opposite class mean. Top K selection then filters out weak or ambiguous activations before computing the final direction.

### Experiment coverage

| Experiment | Figure output | Scope | Purpose |
|------------|---------------|-------|---------|
| Robust DiM Comparison | `token_selective_construction_safety.pdf`, `token_selective_construction_sentiment.pdf`, `token_selective_construction_polite.pdf` | 4 models x 3 concepts | Compare Robust DiM with K equals 50 from a 200 candidate pool against the last 5 baseline |
| K Ablation | `k_ablation_olmo3_7b.pdf` | OLMo3 7B | Vary selected activation count K with a fixed candidate pool |
| Candidate Pool Ablation | `ncand_ablation_olmo3_7b.pdf` | OLMo3 7B | Vary candidate pool size with K fixed at 50 |
| Stability Comparison | heatmap outputs under vector analysis | selected model and concept runs | Compare cosine similarity across extracted directions |

### Experiment parameters

| Parameter | Robust DiM Comparison | K Ablation | Candidate Pool Ablation |
|-----------|-----------------------|------------|--------------------------|
| Models | OLMo3 7B, OLMo3 32B, Qwen3 1.7B, Qwen3 14B | OLMo3 7B | OLMo3 7B |
| Concepts | safety or refusal, sentiment, politeness | safety or refusal, sentiment, politeness | safety or refusal, sentiment, politeness |
| Candidate pool | 200 | fixed pool | varied pool size |
| Selected activations | K equals 50 | varied K | K equals 50 |
| Baseline | last 5 tokens and standard DiM where applicable | standard DiM stability reference | fixed K Robust DiM |
| Concept metric | HarmBench ASR for safety, LLM as judge for sentiment and politeness | cosine similarity and downstream concept score when evaluated | cosine similarity and downstream concept score when evaluated |
| Capability metric | MMLU Pro accuracy | MMLU Pro accuracy when evaluated | MMLU Pro accuracy when evaluated |
| Stability metric | pairwise cosine similarity | pairwise cosine similarity | pairwise cosine similarity |

### Code module mapping

| Module or function | Role in the design |
|--------------------|--------------------|
| `src/steering_geometry/extract.py` | Owns activation extraction and direction aggregation |
| `discriminative_token_aggregator(pos, neg, top_k=100)` | Implements Robust DiM relative margin scoring, top K selection, and selected difference of means |
| `mean_aggregator` | Standard DiM baseline that averages candidate activations |
| `pca_aggregator` | Alternative aggregation baseline for extraction experiments |
| `weighted_mean_aggregator` | Weighted direction construction baseline |
| `src/steering_geometry/stability_comparison.py` | Owns vector stability sweeps and comparison plots |
| `run_discriminative_experiment()` | Varies K with a fixed candidate pool, extracts Robust DiM vectors, computes pairwise cosine similarity, and feeds heatmap generation |
| `run_diff_means_experiment()` | Runs the standard DiM baseline for stability comparison |
| `run_stability_comparison_experiment()` | Combines Robust DiM and standard DiM stability results |
| `run_candidate_pool_ablation()` | New experiment path mirroring `run_discriminative_experiment()`, but varying candidate pool size with K fixed at 50 |

### Result flow

The extraction flow starts with contrast pairs for the selected concept and model. `extract.py` collects candidate positive and negative activations, then calls the requested aggregator. For Robust DiM, `discriminative_token_aggregator` computes relative margin scores, selects the top K positive and negative activations, and saves the resulting direction tensor.

Stability experiments in `stability_comparison.py` repeat extraction across seeds or sampled pools, then compute pairwise cosine similarity between resulting directions. Plotting scripts turn those matrices into heatmaps for the paper. Downstream evaluation uses the saved vectors with Prefix Steering, measuring concept behavior and MMLU Pro accuracy without changing the steering application rule.

### Shell entry points

| Script | Purpose |
|--------|---------|
| `scripts/extract/quick_discriminative.sh` | Extract Robust DiM vectors for quick paper runs |
| `scripts/stability_comparison/quick_vector_stability.sh` | Run the quick stability comparison path |
| `scripts/vector_analysis/run_stability_sweep.sh` | Run vector stability sweeps |
| `scripts/vector_analysis/run_stability_comparison.sh` | Run Robust DiM versus standard DiM stability comparison |
| `scripts/vector_analysis/plot_stability_sweep.sh` | Generate stability heatmaps from saved sweep outputs |
| `scripts/vector_analysis/run_k_ablation.sh` | New wrapper for the K ablation experiment |
| `scripts/vector_analysis/run_candidate_pool_ablation.sh` | New wrapper for candidate pool ablation |

### Files changed

| File | Change | Notes |
|------|--------|-------|
| `src/steering_geometry/extract.py` | Existing | Contains Robust DiM and baseline aggregators |
| `src/steering_geometry/stability_comparison.py` | Modified | Adds `run_candidate_pool_ablation()` beside the existing discriminative and standard DiM experiments |
| `scripts/vector_analysis/run_k_ablation.sh` | New | Shell wrapper for varying K with a fixed candidate pool |
| `scripts/vector_analysis/run_candidate_pool_ablation.sh` | New | Shell wrapper for varying candidate pool size with K fixed at 50 |
| `docs/design-docs/robust-dim-extraction.md` | New | Architecture record for Robust DiM extraction experiments |

## Consequences

### Positive

- Robust DiM is a simple drop in replacement for standard DiM aggregation.
- The intervention path stays unchanged, so comparisons isolate direction construction.
- Relative margin scoring gives a clear rule for selecting activations that are both class representative and class discriminative.
- K ablation and candidate pool ablation expose how sensitive the method is to selection size and candidate quality.
- Stability heatmaps provide a compact check that selected directions agree across repeated extraction runs.

### Limitations

- K equals 50 and pool equals 200 are paper defaults, not universal constants. The best values may vary by model, concept, layer, and data source.
- Very small K can produce unstable direction estimates because too few activations define the class mean.
- Very large K can reintroduce lower ranked activations, making the method closer to standard DiM.
- Candidate pool size affects both quality and runtime. Larger pools give the selector more choices, but also increase extraction cost.
- Cosine similarity measures direction stability, not downstream usefulness by itself. Concept score and MMLU Pro accuracy remain necessary for final evaluation.
