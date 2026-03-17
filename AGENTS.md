# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-14
**Commit:** 32ab292
**Branch:** experiment/test-steering

AI agents working in this repository MUST follow these rules.

## 1) Repository Snapshot

- Package manager: `uv`
- Python: 3.12+ (see `.python-version`)
- Build: `hatchling`
- Lint/Format: `ruff` (line-length 100, double quotes)
- Type check: `mypy --strict`
- Test: `pytest`

## 2) Build & Verify Commands

```bash
# Install dependencies
uv sync

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

## 3) Definition of Done

Before opening a PR, ALL of these must pass:

- [ ] `uv sync` completes without errors
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → already formatted
- [ ] `uv run mypy src/` → Success with 0 errors
- [ ] `uv run pytest` → all tests pass

## 4) When Writing Code

- Use ESM-style imports (from x import y)
- Group imports: stdlib → third-party → local
- Use type hints on ALL function parameters and returns
- Never use `Any` — use `unknown` patterns or proper types
- Fail fast on invalid input
- Throw typed/domain-specific errors
- Preserve original error as `cause` when wrapping

## 5) When Writing Tests

- Use pytest style (plain functions, assert statements)
- Cover: happy path, edge cases, failure paths
- Keep tests deterministic and isolated
- Mock external boundaries (network, file I/O)

## 6) When Opening a PR

Include in description:
1. Summary of changes (2-3 sentences)
2. Linked issue (Closes #N)
3. Verification output (paste command results)
4. Self-review checklist completion
5. Risks & considerations

## 7) Escalation Rules

STOP and report to human if:
- Modifying more than 5 files not in the original plan
- Encountering unclear requirements after 2 clarification attempts
- Need to add new dependencies
- Changes affect security (auth, secrets, permissions)
- Test coverage would drop below existing level

## 8) Code Style

Enforced by ruff (see pyproject.toml):
- Line length: 100
- Quote style: double
- Naming: snake_case for functions/vars, PascalCase for classes
- Lint rules: E, F, I (isort), UP (pyupgrade), B (bugbear), SIM (simplify), N (naming)

### Import Style
- ESM-style imports: `from x import y`
- Grouping: stdlib → third-party → local (enforced by isort)
- Modern union syntax: `str | None` (enforced by pyupgrade)

### Type System
- `py.typed` marker present — package is typed
- Dataclasses for domain objects, TypedDict for dict schemas
- Modern syntax: `list[str]`, `dict[int, Tensor]`

## 9) Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Extract steering vector | `src/steering_geometry/extract.py` | `extract_vector()`, `load_contrast_pairs()` |
| Add new concept | `extract.py:_DATASET_LOADERS` | Add loader + prefix constants |
| Load model with hooks | `src/steering_geometry/models.py` | `HookedModel` class |
| Apply steering | `src/steering_geometry/apply_steering.py` | `apply_steering()` |
| Evaluate steering | `src/steering_geometry/evaluation.py` | `JudgeEvaluator`, `MMLUEvaluator` |
| Core types | `src/steering_geometry/types.py` | `ContrastPair`, `SteeringVector`, etc. |
| Config classes | `src/steering_geometry/config.py` | `ModelConfig`, `ExtractionConfig`, etc. |
| Test fixtures | `tests/conftest.py` | `mock_hooked_model`, `sample_contrast_pairs` |
| Pipeline scripts | `scripts/run_pipeline.sh` | Full orchestration |
| Quick scripts | `scripts/quick/` | Single-layer operations |
| Vector analysis | `src/steering_geometry/vector_analysis.py` | `run_diff_means_experiment()`, `run_discriminative_experiment()` |
| Experiment scripts | `scripts/experiments/` | Cosine similarity heatmaps |

## 10) Anti-Patterns

### Forbidden in This Project
- `typing.Any` — Use proper types or `unknown` patterns
- `# type: ignore` — Avoid unless for untyped third-party libs
- `print()` in production code — Use logging module
- Bare `except:` — Always specify exception type
- `from x import *` — Explicit imports only

### Current Technical Debt
| File | Issue | Fix |
|------|-------|-----|
| `models.py:133,187` | `Any` in hook params | Use Protocol or specific types |
| `extract.py` + `tdnv.py` | `_select_token_activations` duplicated | Extract to utils.py |
| `extract.py`, `apply_steering.py`, `tdnv.py` | `print()` for CLI | Use logging |

## 11) Pipeline Workflow

**ALWAYS follow this pipeline for every task:**

```
1. READ PLAN    → Read PLAN.md, parse tasks, understand requirements
2. CODE         → Implement following conventions in this file
3. VERIFY       → Run: ruff check, ruff format, mypy, pytest (ALL must pass)
4. MOVE PLAN    → ./scripts/complete_plan.sh <plan_name>
5. UPDATE DOCS  → PLANS.md, QUALITY_SCORE.md, ARCHITECTURE.md as needed
6. COMMIT/PR    → When logical unit complete + all checks pass
```

### Plan Completion

When a plan from `.sisyphus/plans/` is complete:

```bash
# Move plan to docs/exec-plans/completed/
./scripts/complete_plan.sh <plan_name>

# Example:
./scripts/complete_plan.sh steering-concepts-pipeline
```

### Extraction Scripts

Run steering vector extractions:

```bash
# Single extraction via Python module
uv run python -m steering_geometry.extract --concept honesty --model "Qwen/Qwen3.5-2B"
uv run python -m steering_geometry.extract --concept toxicity --method pca

# Full pipeline (extract → steer → evaluate)
./scripts/run_pipeline.sh -c honesty,toxicity

# Extraction only
./scripts/run_pipeline.sh -c all --extract-only

# Multiple models
./scripts/run_pipeline.sh -c honesty -m "Qwen/Qwen3.5-2B,google/gemma-2-2b"

# Quick single-layer extraction
scripts/quick/quick_extract.sh -c honesty -l 0.7
```

### Experiments

Run cosine similarity experiments to analyze steering vector stability:

```bash
# Differential means experiment (varying example counts)
./scripts/experiments/run_diff_means_heatmaps.sh

# Discriminative token selection experiment (varying K values)
./scripts/experiments/run_discriminative_heatmaps.sh
```

**Parameters:**
- Concepts: honesty, sentiment, toxicity, sycophancy, refusal
- Model: Qwen/Qwen3-1.7B
- Layers: 0.4, 0.5, 0.6, 0.7, 0.8
- Diff means n_examples: 10, 30, 100, 300, 1000, 3000, 6000, 10000
- Discriminative K values: 16, 32, 64, 128, 256

**Output Structure:**
```
outputs/
├── vectors/
│   ├── {concept}/diff_means/n{count}_layer{frac}.pt
│   └── {concept}/discriminative/k{K}_layer{frac}.pt
└── heatmaps/
    ├── diff_means/{concept}_layer{frac}.pdf
    └── discriminative/{concept}_layer{frac}.pdf
```

**Expected Output Counts:**
- 50 PDF heatmaps (5 concepts × 5 layers × 2 methods)
- 325 steering vectors (200 diff_means + 125 discriminative)

### Directory Rules

- **scripts/** → Shell scripts (`.sh`) ONLY. No Python files.
- **src/steering_geometry/** → All Python modules (`.py`).

### Commit Criteria

Commit ONLY when ALL conditions met:
- `uv sync` completes without errors
- Logical unit of work complete
- `uv run ruff check src/ tests/` → 0 violations
- `uv run ruff format --check src/ tests/` → formatted
- `uv run mypy src/` → 0 errors
- `uv run pytest` → all pass
- Related docs updated

### PR Criteria

Open PR when feature complete:
- All commit criteria met
- Self-review checklist done
- Verification output included
- Risks documented

## 12) Architecture Cross-References

- System design: `ARCHITECTURE.md`
- Design docs: `docs/design-docs/`
- Exec plans: `docs/exec-plans/` (active/ for in-progress, completed/ for done)
- Quality tracking: `docs/QUALITY_SCORE.md`
- Roadmap: `docs/PLANS.md`
