# Construction Diagnosis Experiments

**Status:** Approved
**Date:** 2026-06
**Branch:** experiment/pipeline

## Context

The paper shows that DiM direction quality depends strongly on which activations are used to construct the steering direction, not just how many activations are averaged. The construction diagnosis experiments test this claim by holding key budgets fixed, then changing one construction choice at a time.

Three observations motivate the design:

1. **Obs 1: Token Position** compares trailing prompt-token windows against all-token averaging under a matched 200-activation budget per class. The figures are `safety_lastN.pdf`, `sentiment_lastN.pdf`, and `polite_lastN.pdf`. Later prompt tokens produce better steering than all-token averaging.
2. **Obs 2: Prompt vs Response** compares prompt-only construction against prompt-response construction under matched budgets. The figures are `pvsr_safety.pdf`, `pvsr_sentiment.pdf`, and `pvsr_polite.pdf`. Prompt-only directions are stronger.
3. **Obs 3: Example Count** varies the number of construction examples while keeping the last-5 token rule fixed. The figures are `safety_n_effect.pdf`, `sentiment_n_effect.pdf`, and `polite_n_effect.pdf`. More examples improve directional stability, but they don't guarantee better steering performance.

The gap between directional stability and steering effectiveness is the point of the diagnostic suite. Cosine similarity can show that directions are stable across construction choices, while concept score and MMLU-Pro show whether the resulting intervention is useful and preserves capability.

## Decision

### Experiment scope

The construction diagnosis suite covers the paper models, paper concepts, and three construction axes. Each axis is run through the shared extraction path in `src/steering_geometry/extract.py`, then analyzed with the vector comparison helpers in `src/steering_geometry/stability_comparison.py`.

| Field | Values |
|-------|--------|
| Models | `allenai/Olmo-3-1025-7B`, `allenai/Olmo-3-1125-32B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-14B` |
| Concepts | `refusal`, `sentiment`, `polite` |
| Obs 1 budget | 200 activations per class |
| Obs 2 budget | 200 activations per class |
| Obs 3 token rule | `last_n` with `last_n=5` |
| Trials | Three independent trials for paper measurements |
| Stability metric | Pairwise cosine similarity |
| Steering metrics | Concept score and MMLU-Pro accuracy |

### Paper experiment parameters

| Observation | Question | Construction variable | Fixed controls | Figure outputs |
|-------------|----------|-----------------------|----------------|----------------|
| Obs 1: Token Position | Which prompt tokens should construct the direction? | `last_n` values 1, 3, 5, 10, 20, plus `all` | 200 activations per class, same concept, model, layer set, and extraction method | `safety_lastN.pdf`, `sentiment_lastN.pdf`, `polite_lastN.pdf` |
| Obs 2: Prompt vs Response | Should response tokens enter the construction pool? | `prompt_only` vs `prompt_response`, measured for last-1, last-5, and all tokens | 200 activations per class, same concept, model, layer set, and extraction method | `pvsr_safety.pdf`, `pvsr_sentiment.pdf`, `pvsr_polite.pdf` |
| Obs 3: Example Count | Does more data improve steering? | Number of construction examples `N` | Fixed last-5 token rule, same concept, model, layer set, and extraction method | `safety_n_effect.pdf`, `sentiment_n_effect.pdf`, `polite_n_effect.pdf` |

The matched budget in Obs 1 and Obs 2 keeps the comparison focused on activation source. A wider token pool shouldn't win simply because it contributes more activations.

### Core extraction path

`src/steering_geometry/extract.py` supplies the shared construction path:

```python
load_contrast_pairs(concept, num_pairs, **kwargs) -> list[ContrastPair]
extract_steering_vector(model, pairs, config) -> SteeringVector
select_token_activations(activations, read_token_index, last_n=None) -> Tensor
```

`load_contrast_pairs()` loads paired positive and negative examples for `refusal`, `sentiment`, or `polite`. For prompt-response comparisons, extra keyword arguments such as `data_mode="prompt_only"` or `data_mode="prompt_response"` are forwarded to the concept loader.

`extract_steering_vector()` resolves relative layer fractions to absolute model layers, collects positive and negative activations in batches, applies the requested token selection mode, then aggregates the positive and negative activation pools into one vector per layer.

`select_token_activations()` supports three construction modes:

| Mode | Behavior | Output shape |
|------|----------|--------------|
| Integer index | Selects one token position per sample. `-1` means the final non-padding token | `(batch, hidden_dim)` |
| `all` | Selects all non-padding tokens from every sample | `(tokens, hidden_dim)` |
| `last_n` | Selects the final `N` non-padding tokens from each sample | `(tokens, hidden_dim)` |

### Experiment runners

`src/steering_geometry/token_selection_experiments.py` contains the runners used by the shell scripts. The runners share the same pattern: load contrast pairs, load the model once, extract vectors for each parameter setting, reject NaN vectors, save `.pt` vectors, compute layer-wise cosine similarity matrices, plot heatmaps, and return paths plus summary statistics.

| Function | Purpose | Main inputs | Return value | Saved artifacts |
|----------|---------|-------------|--------------|-----------------|
| `run_token_position_experiment(concept, n_examples, position_configs, layers, model_name, ...)` | Obs 1. Compare `all` against several `last_n` windows | `position_configs` with `{"mode": "all"}` and `{"mode": "last_n", "n": N}` | Dict with `vector_paths`, `heatmap_paths`, and `statistics` | `outputs/token_experiments/vectors/{concept}/token_position/*.pt`, `outputs/token_experiments/heatmaps/token_position/*.pdf`, script summary JSON |
| `run_prompt_response_experiment(concept, n_examples, data_modes, layers, model_name, ...)` | Obs 2. Compare prompt-only and prompt-response construction | `data_modes=["prompt_only", "prompt_response"]`, plus token selection mode | Dict with `vector_paths`, `heatmap_paths`, and `statistics` | `outputs/token_experiments/vectors/{concept}/prompt_response/*.pt`, `outputs/token_experiments/heatmaps/prompt_response/*.pdf` |
| `run_token_count_experiment(concept, n_examples_list, layers, model_name, ...)` | Obs 3. Sweep number of construction examples | `n_examples_list`, layers, model, method, token selection | Dict with `vector_paths`, `heatmap_paths`, and `statistics` | `outputs/token_experiments/vectors/{concept}/token_count/*.pt`, `outputs/token_experiments/heatmaps/token_count/*.pdf` |
| `run_steering_scope_experiment(concept, steer_tokens_values, layers, multipliers, model_name, ...)` | Related Section 4 scope test. Compare prefix-only steering against all-token steering | `steer_tokens_values`, layers, multipliers, sample count | Dict with `output_files` and `statistics.total_samples` | `outputs/token_experiments/steered/{concept}/steer_scope/*.jsonl` |

### Stability utilities

`src/steering_geometry/stability_comparison.py` provides the shared vector analysis helpers:

| Function | Role |
|----------|------|
| `compute_cosine_similarity_matrix(vectors)` | Stacks vectors and returns the pairwise cosine similarity matrix |
| `plot_heatmap(matrix, labels, title, output_path)` | Writes a PDF heatmap with labels on both axes and cosine similarity as the color scale |
| `save_vector(vector, path)` | Creates parent directories and saves a tensor as a `.pt` file |

For each layer, `token_selection_experiments._compute_layer_statistics()` computes the cosine similarity matrix and records off-diagonal `mean_similarity`, `min_similarity`, and `max_similarity`. These statistics summarize directional stability across the construction settings being compared.

### Script mapping

| Script | Paper role | Python runner | Default output root |
|--------|------------|---------------|---------------------|
| `scripts/token_experiments/1_token_count.sh` | Obs 3: Example Count | `run_token_count_experiment()` | `outputs/token_experiments` |
| `scripts/token_experiments/2_token_position.sh` | Obs 1: Token Position | `run_token_position_experiment()` | `outputs/token_experiments` |
| `scripts/token_experiments/3_prompt_vs_response.sh` | Obs 2: Prompt vs Response | `run_prompt_response_experiment()` | `outputs/token_experiments` |
| `scripts/token_experiments/4_steering_scope.sh` | Related Section 4 steering scope | `run_steering_scope_experiment()` | `outputs/token_experiments` |

Example commands:

```bash
./scripts/token_experiments/2_token_position.sh \
    -c refusal \
    -m "Qwen/Qwen3-1.7B" \
    -n 200 \
    --last-n "1 3 5 10 20" \
    --include-all true

./scripts/token_experiments/3_prompt_vs_response.sh \
    -c sentiment \
    -m "allenai/Olmo-3-1025-7B" \
    -n 200 \
    --data-modes "prompt_only prompt_response"

./scripts/token_experiments/1_token_count.sh \
    -c polite \
    -m "Qwen/Qwen3-14B" \
    -n "10 30 100 300 1000"
```

### Result structure

The first three runners return the same result shape:

```python
{
    "vector_paths": {"parameter_label": "path/to/vector.pt"},
    "heatmap_paths": {"layer0.7": "path/to/heatmap.pdf"},
    "statistics": {
        "layer0.7": {
            "mean_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
        }
    },
}
```

`2_token_position.sh` also writes `token_position_{concept}_summary.json` at the output root. The steering scope runner writes generated samples as JSONL records with `steer_tokens`, `layer`, `multiplier`, `sample_idx`, `prompt`, and `generated_text`.

### Evaluation readout

The construction runners measure stability directly through cosine similarity. Steering effectiveness and capability retention are read from the downstream evaluation path that applies saved directions with Prefix Steering, then scores concept behavior and MMLU-Pro accuracy.

| Metric | Meaning | Source |
|--------|---------|--------|
| Cosine similarity | Directional stability across construction settings or trials | `compute_cosine_similarity_matrix()` and heatmaps |
| Concept score | Behavioral steering strength for refusal, sentiment, or politeness | HarmBench or LLM-as-judge evaluation after steering |
| MMLU-Pro accuracy | Capability retention under the intervention | MMLU-Pro evaluation after steering |

## Consequences

### Positive

- The design isolates activation source from activation count, so token position and data source effects are visible.
- The same runner pattern covers token position, prompt-response, and example-count sweeps.
- Saved vectors, heatmaps, and JSON summaries make the diagnostics reproducible across models, concepts, layers, and trials.
- Cosine similarity and steering metrics are kept separate, which makes the stability-effectiveness gap explicit.

### Findings

- More averaging doesn't imply better steering.
- Later prompt tokens can produce stronger directions than all-token averaging under the same activation budget.
- Prompt-only construction can outperform prompt-response construction.
- Increasing `N` can improve directional stability without improving steering performance.

### Limitations

- The experiment runners compute stability heatmaps, but concept score and MMLU-Pro require the downstream steering and evaluation path.
- Full paper runs across four models, three concepts, several layers, and three trials need GPU time.
- The shell defaults are quick-run defaults. Paper settings should pass the matched 200-activation budget and the exact token windows used for the figures.
