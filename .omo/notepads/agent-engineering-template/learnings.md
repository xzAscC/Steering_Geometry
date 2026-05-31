# Learnings - Agent Engineering Template

## 2026-03-06: Python .gitignore Patterns

- Standard Python .gitignore should include bytecode, venvs, distribution, IDE, testing artifacts
- Important NOT to ignore `uv.lock` and `.python-version` - these should be committed for reproducibility
- Sisyphus evidence directory should be in .gitignore to avoid tracking generated verification files
# Learnings

## 2026-03-06: uv Project Initialization

### Key Decisions
- Used uv-native format with `[dependency-groups]` (not `[project.optional-dependencies]`)
- Python 3.12 as baseline (modern type hints, performance)
- Dev-only dependencies: ruff (lint/format), mypy (type check), pytest (testing)

### Successful Patterns
- `.python-version` file: uv auto-detects and uses this for Python version
- `uv sync` creates `.venv` and `uv.lock` automatically
- `uv run <tool>` works without activating venv

### Versions Installed (2026-03-06)
- ruff: 0.15.5
- mypy: 1.19.1
- pytest: 9.0.2


## Task: Create AGENTS.md
- Successfully created AGENTS.md with the required structure for a Python/uv project.
- Verified that the line count is well within the 150-line limit (98 lines).
- Confirmed that all "uv run" commands are present as specified.
- The file provides a clear operational policy for AI agents in this repository.
