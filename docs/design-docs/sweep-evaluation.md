# Sweep Evaluation: Steering Strength x Prefix Token Count

**Status:** Approved
**Date:** 2026-05
**Branch:** experiment/pipeline

## Context

The paper needs experiments showing how steering performance varies across two axes simultaneously: **steering strength** (multiplier applied to the steering vector) and **prefix token count** (number of early-generation tokens receiving the intervention). This produces a 2D grid where each cell represents one (multiplier, steer_tokens) combination, evaluated on both concept adherence and capability retention.

Prior to this module, `scripts/experiments/prefix_vs_full_strength_sweep.py` ran a similar sweep but **without evaluation** — it only saved raw generation outputs. The new `sweep_evaluation` module adds full evaluation (HarmBench, LLM-as-judge, MMLU-Pro) and produces heatmap plots.

## Decision

### New module: `src/steering_geometry/sweep_evaluation.py`

A single module that handles the full pipeline: load model + vector, sweep the grid, evaluate each cell, save JSON results, plot heatmaps.

### Data model

Two TypedDicts define the result structure:

```python
class SweepCellResult(TypedDict):
    multiplier: float           # Steering strength multiplier
    steer_tokens: int | None    # Prefix token count (None = all tokens)
    concept_score: float        # Concept adherence score
    fluency_score: float        # Fluency score from judge
    mmlu_pro_accuracy: float    # MMLU-Pro accuracy percentage
    num_samples: int            # Number of prompts evaluated

class SweepResult(TypedDict):
    concept: str
    model: str
    layer_frac: float
    multipliers: list[float]
    steer_tokens_values: list[int | None]
    cells: list[SweepCellResult]
    output_dir: str
```

`SweepResult` is serialized as JSON under `outputs/sweep_evaluation/{concept}/{model}/sweep_results.json`.

### Grid defaults

| Axis | Default values |
|------|---------------|
| Multipliers (Y) | `[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]` |
| Steer tokens (X) | `[1, 2, 3, 5, 10, 20, 50, None]` |

`None` in steer_tokens means full-token steering (all generation steps receive the intervention), matching the `HookedModel.generate_with_steering(steer_tokens=None)` API.

### Vector loading

Saved steering vectors come in two formats depending on how they were created:

1. **Raw tensor** — discriminative vectors saved by `scripts/extract/quick_discriminative.sh` as `k128_layer0.7.pt`. Loaded via `torch.load()` directly.
2. **Dict format** — extraction CLI (`python -m steering_geometry.extract`) saves as `{"vector": SteeringVector, "num_pairs": N}`. The `SteeringVector.layer_activations` dict maps layer indices to tensors; the first layer is selected.

Both formats are handled by the loading logic (lines 119–132). A `ValueError` is raised for unrecognized formats or zero-norm vectors.

### Evaluation paths

Each grid cell is evaluated on two metrics:

#### Concept evaluation

| Concept | Evaluator | Metric | Notes |
|---------|-----------|--------|-------|
| `refusal` | HarmBenchEvaluator or refusal-prefix matching | ASR (Attack Success Rate, 0–100%) | Uses HarmBench when `--harmbench-behaviors-file` is provided; falls back to refusal-prefix matching otherwise |
| `sentiment` | JudgeEvaluator (LLM-as-judge) | concept_score (0–10) | Uses OpenRouter API |
| `polite` | JudgeEvaluator (LLM-as-judge) | concept_score (0–10) | Uses OpenRouter API |

**Refusal-prefix matching** (the fallback): checks whether generated text starts with known refusal prefixes (`"I'm sorry"`, `"I cannot"`, `"As an AI"`, etc.). ASR = percentage of generations that do NOT start with a refusal prefix. When all generations refuse → ASR = 0%. When none refuse → ASR = 100%.

This fallback exists because HarmBench evaluation requires a running classifier server and a behaviors CSV file, which may not be available during quick sweeps.

#### MMLU-Pro evaluation

`MMLUProEvaluator.evaluate()` is called with `steer_tokens` passed through, so MMLU-Pro accuracy reflects capability retention under the specific prefix-length intervention. This required a modification to `apply_steering.py`:

```python
# Before (apply_steering.py, MMLUProEvaluator.evaluate)
def evaluate(self, vector, layer_idx, scale, ...) -> MMLUProResult:

# After
def evaluate(self, vector, layer_idx, scale, ..., steer_tokens: int | None = None) -> MMLUProResult:
```

The `steer_tokens` parameter is forwarded to `model.generate_with_steering(steer_tokens=steer_tokens)` inside the evaluator.

### Plotting

`plot_sweep_heatmaps()` produces two heatmaps per run:

1. **Concept score heatmap** — `YlOrRd` colormap, shows concept adherence across the grid
2. **MMLU-Pro accuracy heatmap** — `RdYlGn` colormap, shows capability retention across the grid

Both heatmaps use:
- X-axis: Steer Tokens (prefix token count, with `None` labeled as `"all"`)
- Y-axis: Multiplier (steering strength)
- `origin="lower"` so smallest multiplier at bottom, largest at top
- Text annotations on each cell showing the numeric value
- Saved in both PDF and PNG formats

Output path: `outputs/sweep_evaluation/{concept}/{model}/concept_score_heatmap.{pdf,png}` and `mmlu_pro_accuracy_heatmap.{pdf,png}`.

### Shell entry point

`scripts/experiments/steering_strength_prefix_sweep.sh` wraps the Python module with:
- Full argument parsing (short/long options, defaults, help text)
- `--include-full` flag to prepend `None` (all-token steering) to the steer_tokens grid (default: `true`)
- Colored output for configuration display and completion message
- Converts shell args to Python literals for the inline Python invocation

Usage:
```bash
./scripts/experiments/steering_strength_prefix_sweep.sh \
    -c sentiment \
    -m "Qwen/Qwen3-1.7B" \
    -v outputs/vectors/sentiment/discriminative/k128_layer0.7.pt
```

### Files changed

| File | Change | Lines |
|------|--------|-------|
| `src/steering_geometry/sweep_evaluation.py` | **New** — sweep evaluation module | 616 |
| `src/steering_geometry/apply_steering.py` | **Modified** — added `steer_tokens` param to `MMLUProEvaluator.evaluate()` | ~3 lines |
| `tests/test_sweep_evaluation.py` | **New** — 9 tests | 418 |
| `scripts/experiments/steering_strength_prefix_sweep.sh` | **New** — shell entry point | 241 |

### Test coverage

9 tests covering:

1. `SweepCellResult` TypedDict key verification
2. `SweepResult` TypedDict key verification
3. `plot_sweep_heatmaps` creates PDF and PNG files
4. Heatmap matrix dimensions match grid size (2x2 → 4 output files)
5. `steer_tokens=None` labeled as `"all"` in X-axis
6. Grid construction: 2 multipliers x 2 steer_tokens = 4 cells
7. Full sweep with JSON output verification (mocked model + evaluators)
8. Dict-format vector loading (`{"vector": SteeringVector, ...}`)
9. Refusal-prefix matching evaluation (no HarmBench file)

## Consequences

### Positive

- Single command produces publication-ready heatmap figures
- Evaluates both concept effectiveness and capability retention in one sweep
- Handles both vector formats from the existing pipeline
- No new dependencies (matplotlib already available)
- Shell script makes it easy to run sweeps for all paper models and concepts

### Limitations

- Refusal-prefix matching is a proxy for HarmBench — it catches obvious refusals but misses subtler cases where the model partially complies. For publication-quality results, use `--harmbench-behaviors-file` with the full HarmBench evaluation pipeline.
- The sweep is sequential across grid cells. For large grids, this can be slow. Parallel cell evaluation was considered but deferred to keep the implementation simple and deterministic.
- `fluency_score` is only populated for LLM-as-judge evaluations (sentiment, politeness). For refusal with HarmBench, it defaults to 0.0 since HarmBench does not measure fluency.
