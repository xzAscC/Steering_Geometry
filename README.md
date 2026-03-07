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
   # Example: Extract steering vectors for honesty
   uv run python scripts/extract_vectors.py --concept honesty
   ```

## Project Structure

```
.
├── AGENTS.md                    # AI agent instructions
├── ARCHITECTURE.md              # System design
├── README.md                    # This file
├── pyproject.toml               # Project config
├── .python-version              # Python version (3.12+)
├── src/steering_geometry/       # Source code
│   ├── concepts/                # Concept-specific datasets and prompts
│   ├── types.py                 # Core type definitions
│   ├── config.py                # Configuration management
│   ├── models.py                # Model loading and wrapping
│   ├── extraction.py            # Activation extraction logic
│   └── evaluation.py            # Vector evaluation and validation
├── tests/                       # Test files
├── scripts/                     # Analysis and utility scripts
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

- **src/** - Core Python modules for steering vector extraction and evaluation
- **scripts/** - Executable scripts for running the extraction pipeline
- **data/** - Input datasets and contrast pairs for different concepts
- **plot/** - Visualizations of activation spaces and steering effects
- **assets/** - Saved steering vectors and evaluation reports

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
