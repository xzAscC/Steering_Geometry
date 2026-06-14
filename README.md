# Steering Geometry

Research code for the NeurIPS 2026 paper, "Not All Tokens Are Equally Useful for Steering: Robust Directions and Prefix Steering for Activation Steering".

This repository studies which token positions produce stable activation steering directions, and how early-token interventions can steer model behavior while preserving general capability. The codebase focuses only on the paper experiments: Robust DiM direction construction, Prefix Steering intervention, safety refusal, sentiment, politeness, HarmBench, 3-label LLM-as-judge evaluation, and MMLU-Pro.

## Paper Scope

### Models

The experiments use four Hugging Face causal language models:

- **OLMo3-7B**: `allenai/Olmo-3-1025-7B`
- **OLMo3-32B**: `allenai/Olmo-3-1125-32B`
- **Qwen3-1.7B**: `Qwen/Qwen3-1.7B`
- **Qwen3-14B**: `Qwen/Qwen3-14B`

### Concepts

The paper evaluates three steering concepts:

- **Safety/refusal**: increase safe refusal behavior on harmful requests
- **Sentiment**: steer generated text toward positive or negative sentiment
- **Politeness**: steer generated text toward polite or impolite style

### Datasets

- **Safety/refusal**: `LLM-LAT/benign-dataset` and `LLM-LAT/harmful-dataset`
- **Sentiment**: SST-2
- **Politeness**: PoliteGuard

### Evaluations

- **HarmBench** for safety behavior
- **3-label LLM-as-judge** for sentiment and politeness
- **MMLU-Pro** for general capability retention

## Methods

### Robust DiM

Robust DiM builds steering directions from token subsets rather than treating every token as equally useful. The extraction pipeline compares activations from contrast pairs, selects stable and informative token positions, and constructs a direction that is less sensitive to noisy or weak tokens.

### Prefix Steering

Prefix Steering applies the steering direction to early tokens during generation. This tests whether short, prefix-local interventions can guide behavior while reducing disruption to later-token computation and preserving MMLU-Pro performance.

## Quick Start

Install dependencies:

```bash
uv sync
```

Extract a sentiment steering vector with one paper model:

```bash
uv run python -m steering_geometry.extract --concept sentiment --model "Qwen/Qwen3-1.7B"
```

Apply a saved vector and run the configured evaluation path:

```bash
uv run python -m steering_geometry.apply_steering --vector outputs/vectors/sentiment/layer0.7.pt
```

Run construction diagnosis and stability experiments through the experiment scripts listed below. The `token_selection_experiments` and `stability_comparison` modules provide experiment functions used by those scripts.

## Project Structure

```text
src/steering_geometry/
├── __init__.py
├── __main__.py
├── types.py
├── config.py
├── models.py
├── extract.py                         # Steering vector extraction
├── apply_steering.py                  # Apply steering + evaluation
├── sweep_evaluation.py                # Strength × steer_tokens sweep evaluation
├── prefix_analysis.py                 # KL divergence and attention pattern analysis
├── stability_comparison.py            # Vector stability experiments
├── token_selection_experiments.py     # Construction diagnosis experiments
├── utils.py
├── py.typed

scripts/
├── extract/
├── apply_steering/
├── experiments/
├── pipeline/
├── token_experiments/
├── vector_analysis/
├── stability_comparison/

tests/
├── conftest.py
├── test_apply_steering.py
├── test_experiments.py
├── test_stability_comparison.py
├── test_stability_sweep.py
├── test_token_selection_experiments.py
├── unit/
│   ├── test_aggregators.py
│   ├── test_config_main.py
│   ├── test_evaluation.py
│   ├── test_extract.py
│   ├── test_logging.py
│   └── test_utils.py
```

## Entry Points

Use module entry points for extraction and steering:

```bash
# Extract Robust DiM steering vectors
uv run python -m steering_geometry.extract --concept sentiment --model "Qwen/Qwen3-1.7B"

# Apply Prefix Steering with a saved vector
uv run python -m steering_geometry.apply_steering --vector outputs/vectors/sentiment/layer0.7.pt
```

Package metadata and shell configuration values are available through the package CLI:

```bash
uv run python -m steering_geometry --shell
```

The experiment modules are imported by the shell scripts in `scripts/` rather than exposed as standalone module CLIs.

## Paper Experiment Scripts

The `scripts/` tree contains shell entry points for the paper experiments:

```text
scripts/extract/
└── quick_discriminative.sh

scripts/apply_steering/
└── run_steering.sh

scripts/pipeline/
└── quick_pipeline.sh

scripts/token_experiments/
├── 1_token_count.sh
├── 2_token_position.sh
├── 3_prompt_vs_response.sh
└── 4_steering_scope.sh

scripts/vector_analysis/
├── plot_stability_sweep.sh
├── quick_diff_means_heatmaps.sh
├── quick_discriminative_heatmaps.sh
├── run_candidate_pool_ablation.sh
├── run_k_ablation.sh
├── run_stability_comparison.sh
└── run_stability_sweep.sh

scripts/experiments/
└── steering_strength_prefix_sweep.sh

scripts/stability_comparison/
└── quick_vector_stability.sh
```

Use these scripts for full experiment runs, token selection diagnostics, vector stability comparisons, heatmap generation, and Prefix Steering sweeps.

## Development

Run the standard checks before submitting changes:

```bash
# Lint
uv run ruff check src/ tests/

# Format check
uv run ruff format --check src/ tests/

# Type check
uv run mypy src/

# Tests
uv run pytest
```

To format code locally:

```bash
uv run ruff format src/ tests/
```

## License

MIT License. See `LICENSE` for details.
