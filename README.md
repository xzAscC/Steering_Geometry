# Steering Geometry

Vehicle steering geometry analysis and visualization toolkit for understanding and optimizing steering systems.

## What This Is

This project provides tools for analyzing and visualizing vehicle steering geometry parameters, including:

- **Ackermann steering geometry** - Inner and outer wheel angle relationships
- **Turning radius calculations** - Minimum turning radius analysis
- **Steering angle visualization** - Interactive plots of steering behavior
- **Geometry parameter analysis** - Track width, wheelbase, and steering arm effects

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

3. **Run scripts**
   ```bash
   # Example: Run analysis script
   uv run python scripts/analyze_geometry.py
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
│   └── __init__.py
├── tests/                       # Test files
├── scripts/                     # Analysis and utility scripts
├── data/                        # Data files and inputs
├── plot/                        # Generated plots and visualizations
├── assets/                      # Analysis results and outputs
└── docs/
    ├── design-docs/             # Design documents
    ├── exec-plans/              # Execution plans
    ├── PLANS.md                 # Project roadmap
    └── QUALITY_SCORE.md         # Quality tracking
```

## Directory Usage

- **src/** - Core Python modules for steering geometry calculations
- **scripts/** - Executable scripts for analysis and visualization
- **data/** - Input data files (vehicle parameters, measurements)
- **plot/** - Generated plots, graphs, and visualizations
- **assets/** - Analysis results, reports, and outputs

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
