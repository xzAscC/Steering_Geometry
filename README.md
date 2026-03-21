# Steering Geometry - AI Steering Vector Extraction Framework

A research framework for extracting and analyzing steering vectors in Large Language Models (LLMs) to understand and control model behavior.

## What This Is

This project provides tools for extracting activation steering vectors from LLMs across five key behavioral concepts:

- **Honesty** - Encouraging truthful responses and reducing hallucinations
- **Sycophancy** - Identifying and mitigating the tendency to agree with user misconceptions
- **Toxicity** - Detecting and steering away from harmful or offensive content generation
- **Sentiment** - Controlling the emotional tone and valence of model outputs
- **Refusal** - Understanding the mechanisms behind model refusals and safety boundaries

## Quick Start

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Verify installation**
   ```bash
   uv run ruff check src/ tests/
   uv run mypy src/
   uv run pytest
   ```

3. **Run extraction**
   ```bash
   # Example: Extract steering vectors for honesty with default model
   uv run python -m steering_geometry.extract --concept honesty
   
   # Use specific models (Qwen, Gemma)
   uv run python -m steering_geometry.extract --concept honesty --model "Qwen/Qwen3-1.7B"
   uv run python -m steering_geometry.extract --concept honesty --model "Qwen/Qwen3.5-2B"
   uv run python -m steering_geometry.extract --concept honesty --model "google/gemma-2-2b"
   
   # Or use the batch script for multiple extractions
   ./scripts/pipeline/run_pipeline.sh -c honesty,toxicity -m "Qwen/Qwen3.5-2B"
   ```
   
   Supported concepts:
   - `honesty` - Honesty steering vectors
   - `sycophancy` - Sycophancy steering vectors
   - `toxicity` - Toxicity steering vectors
   - `sentiment` - Sentiment steering vectors
   - `refusal` - Refusal steering vectors

## Project Structure

```
.
├── AGENTS.md                    # AI agent instructions
├── ARCHITECTURE.md              # System design
├── SPEC.md                      # Technical specification
├── README.md                    # This file
├── pyproject.toml               # Project config
├── .python-version              # Python version (3.12+)
├── src/steering_geometry/       # Source code (Python modules ONLY)
│   ├── __init__.py              # Package exports
│   ├── types.py                 # Core type definitions
│   ├── config.py                # Configuration management
│   ├── models.py                # Model loading and activation hooks
│   ├── extract.py               # Steering vector extraction
│   ├── apply_steering.py        # Apply steering + evaluation (JudgeEvaluator, MMLUEvaluator)
│   ├── stability_comparison.py  # Vector stability experiments
│   ├── token_analysis.py        # Token-level analysis
│   ├── unembed_analysis.py      # Unembedding analysis
│   ├── tdnv.py                  # TDNV separability metrics
│   └── utils.py                 # Shared utilities
├── tests/                       # Test files
│   ├── conftest.py              # Pytest fixtures
│   ├── test_apply_steering.py   # Steering integration tests
│   └── unit/                    # Unit tests
├── scripts/                     # Shell scripts ONLY (no .py files)
│   ├── pipeline/                # Pipeline scripts
│   │   └── run_pipeline.sh     # Batch extraction orchestrator
│   ├── quick/                   # Single-operation shortcuts
│   ├── vector_analysis/         # Stability experiment scripts
│   ├── token_analysis/          # Token analysis scripts
│   ├── tdnv/                    # TDNV metric scripts
│   ├── unembed_analysis/        # Unembedding scripts
│   ├── stability_comparison/    # Stability comparison scripts
│   └── complete_plan.sh         # Plan completion utility
├── outputs/                     # Generated artifacts (vectors, heatmaps)
│   ├── vectors/                 # Steering vectors (.pt files)
│   ├── heatmaps/                # Cosine similarity heatmaps (.pdf)
│   ├── probes/                  # Probe experiment results
│   ├── token_analysis/          # Token analysis outputs
│   ├── token_viz/               # Token visualizations
│   └── unembed_analysis/        # Unembed analysis outputs
├── data/                        # Raw datasets and contrast pairs
└── docs/
    ├── design-docs/             # Design documents
    ├── exec-plans/              # Execution plans
    ├── PLANS.md                 # Project roadmap
    └── QUALITY_SCORE.md         # Quality tracking
```

## Directory Usage

- **src/** - All Python modules (importable package)
- **scripts/** - Shell scripts only (orchestration, not imported)
- **data/** - Input datasets and contrast pairs for different concepts
- **outputs/** - Generated artifacts (vectors, heatmaps, analysis results)
- **docs/** - Design documents, execution plans, and quality tracking

## Entry Points

This framework uses `python -m` invocation pattern:

```bash
# Extract steering vectors
uv run python -m steering_geometry.extract --concept honesty --model "Qwen/Qwen3.5-2B"

# Apply steering vectors
uv run python -m steering_geometry.apply_steering --vector outputs/vectors/honesty/layer0.7.pt

# Token analysis (visualize or probe subcommands)
uv run python -m steering_geometry.token_analysis visualize --concept honesty
uv run python -m steering_geometry.token_analysis probe --concept toxicity

# TDNV metrics
uv run python -m steering_geometry.tdnv --concept honesty

# Unembed analysis
uv run python -m steering_geometry.unembed_analysis --concept honesty
```

Or use shell scripts for orchestration:

```bash
# Full pipeline (extract → steer → evaluate)
./scripts/pipeline/run_pipeline.sh -c honesty,toxicity -m "Qwen/Qwen3.5-2B"

# Quick single-operation scripts
./scripts/quick/quick_extract.sh -c honesty -l 0.7
./scripts/quick/quick_steering.sh -c honesty -l 0.7
./scripts/quick/quick_eval.sh -c honesty

# Vector stability experiments
./scripts/vector_analysis/run_diff_means_heatmaps.sh
./scripts/vector_analysis/run_discriminative_heatmaps.sh
```

## Supported Models

This framework works with any causal language model from HuggingFace Transformers. Recommended models for steering vector extraction:

### Qwen Family
- **Qwen/Qwen3-1.7B** - Lightweight Qwen3 model (~1.7B parameters)
- **Qwen/Qwen3.5-2B** - Improved Qwen3.5 (~2B parameters)
- **Qwen/Qwen3.5-4B** - Larger Qwen3.5 model (~4B parameters)

### Gemma Family
- **google/gemma-2-2b** - Gemma 2 model (~2B parameters)

### Usage Example
```bash
# Extract with Qwen3.5
uv run python -m steering_geometry.extract \
    --concept honesty \
    --model "Qwen/Qwen3.5-2B" \
    --num-pairs 500 \
    --output data/vectors/

# Extract with Gemma 2
uv run python -m steering_geometry.extract \
    --concept toxicity \
    --model "google/gemma-2-2b" \
    --method pca \
    --output data/vectors/

# Batch extraction via shell script
./scripts/pipeline/run_pipeline.sh -c honesty,toxicity -m "Qwen/Qwen3.5-2B,google/gemma-2-2b"
```

## Development

### Code Quality

This project uses:
- **ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

### Running Checks

```bash
# Lint check
uv run ruff check src/ tests/

# Format check
uv run ruff format --check src/ tests/

# Format fix
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_hello.py

# Run test by name
uv run pytest -k "test_name"
```

## License

MIT License - see LICENSE file for details
