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
   uv run python -m steering_geometry.extract_honesty
   
   # Use specific models (Qwen, Gemma)
   uv run python -m steering_geometry.extract_honesty --model "Qwen/Qwen2-1.5B"
   uv run python -m steering_geometry.extract_honesty --model "Qwen/Qwen2.5-1.5B"
   uv run python -m steering_geometry.extract_honesty --model "google/gemma-2-2b"
   
   # Or use the batch script for multiple extractions
   ./scripts/run_extractions.sh -c honesty,toxicity -m "Qwen/Qwen2.5-1.5B"
   ```
   
   All extraction modules support the same models:
   - `steering_geometry.extract_honesty` - Honesty steering vectors
   - `steering_geometry.extract_sycophancy` - Sycophancy steering vectors
   - `steering_geometry.extract_toxicity` - Toxicity steering vectors
   - `steering_geometry.extract_sentiment` - Sentiment steering vectors
   - `steering_geometry.extract_refusal` - Refusal steering vectors

## Project Structure

```
.
├── AGENTS.md                    # AI agent instructions
├── ARCHITECTURE.md              # System design
├── README.md                    # This file
├── pyproject.toml               # Project config
├── .python-version              # Python version (3.12+)
├── src/steering_geometry/       # Source code (Python modules ONLY)
│   ├── concepts/                # Concept-specific datasets and prompts
│   ├── types.py                 # Core type definitions
│   ├── config.py                # Configuration management
│   ├── models.py                # Model loading and wrapping
│   ├── extraction.py            # Activation extraction logic
│   ├── evaluation.py            # Vector evaluation and validation
│   ├── extract_*.py             # CLI entry points for extraction
│   └── compare_concepts.py      # Cross-concept comparison
├── tests/                       # Test files
├── scripts/                     # Shell scripts ONLY (no .py files)
│   ├── run_extractions.sh       # Batch extraction orchestrator
│   └── complete_plan.sh         # Plan completion utility
├── data/                        # Raw datasets and contrast pairs
├── plot/                        # Generated visualizations of activations
├── assets/                      # Extracted steering vectors and results
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
- **plot/** - Visualizations of activation spaces and steering effects
- **assets/** - Saved steering vectors and evaluation reports

## Supported Models

This framework works with any causal language model from HuggingFace Transformers. Recommended models for steering vector extraction:

### Qwen Family
- **Qwen/Qwen2-1.5B** - Lightweight Qwen2 model (~1.5B parameters)
- **Qwen/Qwen2.5-1.5B** - Improved Qwen2.5 (~1.5B parameters)
- **Qwen/Qwen2.5-3B** - Larger Qwen2.5 model (~3B parameters)

### Gemma Family
- **google/gemma-2-2b** - Gemma 2 model (~2B parameters)

### Usage Example
```bash
# Extract with Qwen2.5
uv run python -m steering_geometry.extract_honesty \
    --model "Qwen/Qwen2.5-1.5B" \
    --num-pairs 500 \
    --output data/vectors/

# Extract with Gemma 2
uv run python -m steering_geometry.extract_toxicity \
    --model "google/gemma-2-2b" \
    --method pca \
    --output data/vectors/

# Batch extraction via shell script
./scripts/run_extractions.sh -c honesty,toxicity -m "Qwen/Qwen2.5-1.5B,google/gemma-2-2b"
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
