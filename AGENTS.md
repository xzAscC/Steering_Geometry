# PROJECT KNOWLEDGE BASE

**Updated:** 2026-05-31
**Commit:** 3be3c56
**Branch:** experiment/pipeline

AI agents working in this repository MUST follow these rules.

## 1) Repository Snapshot

- Package manager: `uv`
- Python: 3.12 (see `.python-version`)
- Build: `hatchling`
- Lint/Format: `ruff` (line-length 100, double quotes)
- Type check: `mypy --strict`
- Test: `pytest`
- CI: `.github/workflows/ci.yml` — runs lint → format check → mypy → pytest on push/PR to `main`
- Scope: NeurIPS 2026 paper experiments for Robust DiM and Prefix Steering
- Paper models: `allenai/Olmo-3-1025-7B`, `allenai/Olmo-3-1125-32B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-14B`
- Paper concepts: safety/refusal, sentiment, politeness
- Evaluations: HarmBench, LLM-as-judge, MMLU-Pro
- **Env vars**: `OPENROUTER_API_KEY` required for LLM-as-judge evaluation (see `.env.example`)

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
uv run pytest tests/unit/test_extract.py

# Run test by name
uv run pytest -k "test_name"

# Exclude slow or GPU tests
uv run pytest -m "not slow"
uv run pytest -m "not gpu"
```

**Required order:** lint → format check → type check → test. CI enforces this exact sequence.

## 3) Definition of Done

Before opening a PR, ALL of these must pass:

- [ ] `uv sync` completes without errors
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → already formatted
- [ ] `uv run mypy src/` → Success with 0 errors
- [ ] `uv run pytest` → all tests pass

## 4) When Writing Code

- Use ESM-style imports (`from x import y`)
- Group imports: stdlib → third-party → local
- Use type hints on ALL function parameters and returns
- Never use `Any` — use proper types or `unknown` patterns
- Fail fast on invalid input
- Throw typed/domain-specific errors
- Preserve original error as `cause` when wrapping

## 5) When Writing Tests

- Use pytest style (plain functions, assert statements)
- Cover: happy path, edge cases, failure paths
- Keep tests deterministic and isolated
- Mock external boundaries (network, file I/O)
- Custom markers available: `@pytest.mark.slow`, `@pytest.mark.gpu`

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
| Package exports | `src/steering_geometry/__init__.py` | Public package interface |
| CLI shell values | `src/steering_geometry/__main__.py` | Shell eval entry point |
| Core types | `src/steering_geometry/types.py` | Paper domain objects and result schemas |
| Configuration | `src/steering_geometry/config.py` | Four paper models, three paper concepts, extraction and evaluation configs |
| Load model with hooks | `src/steering_geometry/models.py` | `HookedModel` class |
| Extract Robust DiM directions | `src/steering_geometry/extract.py` | Steering vector extraction and contrast pair loading |
| Apply Prefix Steering | `src/steering_geometry/apply_steering.py` | Steering application plus HarmBench, LLM-as-judge, and MMLU-Pro evaluation |
| Prefix analysis | `src/steering_geometry/prefix_analysis.py` | KL divergence and attention pattern analysis for Prefix Steering |
| Vector stability experiments | `src/steering_geometry/stability_comparison.py` | Robust DiM stability sweeps and vector comparison helpers |
| Construction diagnosis experiments | `src/steering_geometry/token_selection_experiments.py` | Token position, prompt vs response, example count, and steering scope experiments |
| Shared utilities | `src/steering_geometry/utils.py` | `ensure_dir()`, `safe_model_name()`, `sample_with_seed()`, `configure_logging()` |
| Test fixtures | `tests/conftest.py` | `mock_hooked_model`, `sample_contrast_pairs`, `FakeTokenizer`, `FakeCausalLM` |
| Extraction scripts | `scripts/extract/` | Paper extraction entry points |
| Prefix Steering scripts | `scripts/apply_steering/` | Steering and evaluation entry points |
| Prefix analysis scripts | `scripts/prefix_analysis/` | `run_analysis.sh`, `run_all_concepts.sh` |
| Pipeline scripts | `scripts/pipeline/` | Paper pipeline orchestration |
| Construction diagnosis scripts | `scripts/token_experiments/` | Token count, token position, prompt vs response, and steering scope runs |
| Vector analysis scripts | `scripts/vector_analysis/` | Stability sweeps and heatmap generation |
| Stability comparison scripts | `scripts/stability_comparison/` | Quick vector stability runs |
| Experiment scripts | `scripts/experiments/` | Contains a Python sweep script (exception to shell-only rule) |

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
| `models.py` | Hook typing may depend on third-party model internals | Prefer Protocols or narrow callable types |
| `apply_steering.py` | Evaluator boundaries touch model, tokenizer, and judge interfaces | Keep interfaces typed and isolate external calls |
| `stability_comparison.py` | Experiment result dictionaries can drift as metrics change | Use TypedDict schemas for persisted results |
| `token_selection_experiments.py` | Construction diagnosis outputs cover multiple experiment shapes | Keep result schemas explicit and test serialization |
| `__main__.py` | `print()` used for shell eval output | Intentional, shell capture output, not logging |
| `scripts/experiments/prefix_vs_full_strength_sweep.py` | Python file in `scripts/` (violates shell-only convention) | Consider moving logic to `src/` and making script a thin wrapper |

### Known Violations (from audit)
- Keep `typing.Any` out of new code unless a third-party boundary has no typed alternative.
- `print()` is allowed only in `src/steering_geometry/__main__.py` for shell eval output.
- `scripts/` must contain shell entry points only. Put Python code in `src/steering_geometry/`. (Exception: `scripts/experiments/` contains a `.py` experiment script.)

## 11) Pipeline Workflow

**ALWAYS follow this pipeline for every task:**

```
1. READ PLAN    → Read PLAN.md, parse tasks, understand requirements
2. CODE         → Implement following conventions in this file
3. VERIFY       → Run: ruff check, ruff format, mypy, pytest (ALL must pass)
4. MOVE PLAN    → ./scripts/complete_plan.sh <plan_name> (if script exists)
5. UPDATE DOCS  → PLANS.md, QUALITY_SCORE.md, ARCHITECTURE.md as needed
6. COMMIT/PR    → When logical unit complete + all checks pass
```

### Extraction Scripts

Run Robust DiM steering vector extractions for paper concepts and models:

```bash
# Single extraction via Python module
uv run python -m steering_geometry.extract --concept sentiment --model "Qwen/Qwen3-1.7B"
uv run python -m steering_geometry.extract --concept politeness --model "allenai/Olmo-3-1025-7B"

# Full pipeline (extract → steer → evaluate)
./scripts/pipeline/quick_pipeline.sh

# Quick paper extraction script
./scripts/extract/quick_discriminative.sh

# Apply Prefix Steering with a saved vector
./scripts/apply_steering/run_steering.sh

# Prefix analysis (KL divergence + attention patterns)
./scripts/prefix_analysis/run_analysis.sh
./scripts/prefix_analysis/run_all_concepts.sh
```

### Experiments

Run the paper experiments through the shell entry points under `scripts/`:

```bash
# Construction diagnosis: number of selected tokens
./scripts/token_experiments/1_token_count.sh

# Construction diagnosis: token position
./scripts/token_experiments/2_token_position.sh

# Construction diagnosis: prompt tokens vs response tokens
./scripts/token_experiments/3_prompt_vs_response.sh

# Prefix Steering: steering scope and prefix length
./scripts/token_experiments/4_steering_scope.sh

# Robust DiM vector stability comparison
./scripts/stability_comparison/quick_vector_stability.sh

# Stability sweep plots and heatmaps
./scripts/vector_analysis/run_stability_sweep.sh
./scripts/vector_analysis/plot_stability_sweep.sh
./scripts/vector_analysis/quick_diff_means_heatmaps.sh
./scripts/vector_analysis/quick_discriminative_heatmaps.sh
./scripts/vector_analysis/run_stability_comparison.sh
```

**Parameters:**
- Models: `allenai/Olmo-3-1025-7B`, `allenai/Olmo-3-1125-32B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-14B`
- Concepts: safety/refusal, sentiment, politeness
- Methods: Robust DiM and Prefix Steering
- Construction diagnosis: token position, prompt vs response tokens, selected token count, number of examples
- Evaluations: HarmBench for safety/refusal, LLM-as-judge for sentiment and politeness, MMLU-Pro for capability retention

**Output Structure:**
```
outputs/
├── vectors/
│   └── {concept}/
│       └── {model}/
│           └── robust_dim_layer{frac}.pt
├── steering/
│   └── {concept}/
│       └── {model}/
│           └── prefix_steering_*.json
├── token_experiments/
│   ├── token_count/
│   ├── token_position/
│   ├── prompt_vs_response/
│   └── steering_scope/
└── vector_analysis/
    ├── stability_sweep/
    └── heatmaps/
```

### Directory Rules

- **scripts/** → Shell scripts (`.sh`) ONLY. No Python files. (Exception: `scripts/experiments/`.)
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

## 12) Environment Setup

- Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` for LLM-as-judge evaluation
- LLM-as-judge uses OpenRouter API by default (`https://openrouter.ai/api/v1`)
- GPU required for extraction and steering experiments; tests mock model loading

## 13) Architecture Cross-References

- System design: `ARCHITECTURE.md`
- Design docs: `docs/design-docs/`
- Exec plans: `docs/exec-plans/` (active/ for in-progress, completed/ for done)
- Quality tracking: `docs/QUALITY_SCORE.md`
- Roadmap: `docs/PLANS.md`
- Test conventions: `tests/AGENTS.md`
