# Stability Cosine Similarity Figures (3 Concept × 4 Model Line Plots)

## TL;DR

> **Quick Summary**: Implement an experiment runner that measures DiM steering vector stability via pairwise cosine similarity across 5 independent runs at varying sample sizes (N), selects the best layer per (model, concept) pair, and produces 3 publication-quality line plots (one per concept) showing cos_sim vs N with 4 model lines and error bands.
>
> **Deliverables**:
> - 3 PDF line-plot figures: one per concept (safety/refusal, politeness/polite, sentiment)
> - Each figure: 4 model lines (OLMo3-7B, OLMo3-32B, Qwen3-1.7B, Qwen3-14B), x=N (log), y=cos_sim, ±1 std shaded band
> - `run_stability_sweep.sh` orchestration script
> - JSON intermediate results for re-plotting without re-running experiments
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Config → Experiment Runner → Plotting → Shell Script

---

## Context

### Original Request
Create 3 figures (one per concept) showing how steering vector cosine similarity varies with sample number (N). Each figure has 4 model lines. The methodology: for each N, construct DiM directions from 5 independently sampled datasets, compute average pairwise cosine similarity, and select the best layer per (model, concept) pair based on highest average cos_sim across all N values.

### Interview Summary
**Key Discussions**:
- **Figure layout**: 3 separate figures (not panels). Each figure: x=N (log scale), y=cos_sim, 4 colored model lines with ±std shaded bands
- **Methodology**: 5 independent runs per (model, concept, layer, N), pairwise cos_sim from C(5,2)=10 pairs
- **Layer selection**: Best layer per (model, concept) = layer with highest average cos_sim across all N values
- **Parameters**: N ∈ {100, 500, 1000, 5000, 10000}, layers [0.1, 0.2, ..., 1.0] (10 layers)

**Research Findings**:
- Existing `run_stability_comparison_experiment()` handles multi-run stability but at fixed N
- Existing `run_diff_means_experiment()` varies N but does single run only
- New experiment combines both patterns: multi-run + varying N
- `compute_cosine_similarity_matrix()` and `compute_stability_statistics()` are reusable
- All plotting uses matplotlib, PDF output, lazy import

### Metis Review
**Identified Gaps** (addressed):
- OLMo3-27B doesn't exist → confirmed OLMo3-32B (`allenai/Olmo-3-1125-32B`)
- "safety"/"politeness" ≠ codebase names → confirmed same concepts as "refusal"/"polite", figure labels use paper names
- `trust_remote_code=True` needed for allenai models → will handle in experiment runner
- No checkpointing/resume → explicitly out of scope, keep simple
- N > available data edge case → use existing `cap_examples()` and log when capping occurs

---

## Work Objectives

### Core Objective
Implement a stability sweep experiment that runs 5 independent DiM extractions at each (model, concept, layer, N) combination, selects the best layer per (model, concept), and produces 3 publication-quality line plots.

### Concrete Deliverables
- `outputs/stability_sweep/{concept}_stability_sweep.pdf` × 3 figures
- `outputs/stability_sweep/results_{model}_{concept}.json` intermediate results
- `scripts/vector_analysis/run_stability_sweep.sh` orchestration script

### Definition of Done
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → formatted
- [ ] `uv run mypy src/` → 0 errors
- [ ] `uv run pytest` → all tests pass
- [ ] 3 PDF figures exist after running the shell script

### Must Have
- 5 independent runs per (model, concept, layer, N) combination
- Pairwise cosine similarity computed from C(5,2)=10 pairs
- Best layer selected per (model, concept) via argmax of mean cos_sim across N
- x-axis: N on log scale with values {100, 500, 1000, 5000, 10000}
- y-axis: mean pairwise cos_sim with shaded ±1 std band
- 4 model lines per figure, color-coded consistently
- JSON persistence for re-plotting without re-running experiments
- Shell script following existing patterns (`set -euo pipefail`, progress counters)

### Must NOT Have (Guardrails)
- **NO new dataset loaders** — reuse existing "refusal", "polite", "sentiment" loaders
- **NO modification of existing experiment functions** — `run_stability_comparison_experiment()`, `run_diff_means_experiment()` stay untouched
- **NO checkpointing/resume infrastructure** — keep simple, rerun on failure
- **NO new dependencies** — matplotlib, torch, sklearn already available
- **NO Python files in `scripts/`** — shell scripts only
- **NO over-abstracted plotting framework** — one focused function
- **NO caching/ghost files** — straightforward save to disk
- **NO CLI subcommands** — experiment runs via shell script + inline Python
- **NO refactoring of existing seeding** — keep `random.Random(run_idx)` pattern

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest, 231 tests)
- **Automated tests**: YES (tests-after)
- **Framework**: pytest (existing)

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python logic**: Use Bash (`uv run pytest`)
- **Plotting output**: Use Bash (file existence + size check)
- **Type safety**: Use Bash (`uv run mypy src/`)
- **Code quality**: Use Bash (`uv run ruff check`)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — config + new model support):
├── Task 1: Add OLMo3-32B + Qwen3-14B to SUPPORTED_MODELS, trust_remote_code handling
└── Task 2: Create StabilitySweepConfig dataclass
    → Category: quick × 2

Wave 2 (Core Logic — experiment runner + plotting):
├── Task 3: Implement run_stability_sweep() in stability_comparison.py
├── Task 4: Implement sweep results persistence (JSON save/load)
└── Task 5: Implement plot_stability_sweep() line-plot function
    → Category: unspecified-high, deep, unspecified-high

Wave 3 (Orchestration + Testing):
├── Task 6: Create run_stability_sweep.sh shell script
└── Task 7: Write unit tests for new code
    → Category: quick, unspecified-high

Wave FINAL (Verification — after ALL implementation tasks):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)

Critical Path: T1/T2 → T3 → T4 → T5 → T6 → T7 → F1-F4
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 3 | 1 |
| 2 | — | 3 | 1 |
| 3 | 1, 2 | 4, 5 | 2 |
| 4 | 3 | 5, 6 | 2 |
| 5 | 4 | 6 | 2 |
| 6 | 3, 5 | — | 3 |
| 7 | 3, 4, 5 | — | 3 |

### Agent Dispatch Summary

- **Wave 1**: 2 tasks — T1 → `quick`, T2 → `quick`
- **Wave 2**: 3 tasks — T3 → `deep`, T4 → `unspecified-high`, T5 → `unspecified-high`
- **Wave 3**: 2 tasks — T6 → `quick`, T7 → `unspecified-high`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task has: Recommended Agent Profile + Parallelization info + QA Scenarios.

- [x] 1. Add OLMo3-32B + Qwen3-14B to SUPPORTED_MODELS + trust_remote_code handling

  **What to do**:
  - Add `"Qwen/Qwen3-14B"` to `SUPPORTED_MODELS` tuple in `config.py`
  - Add `"allenai/Olmo-3-1125-32B"` (OLMo3-32B) to `SUPPORTED_MODELS` tuple in `config.py`
  - In `ModelConfig` or the model-loading code in `models.py`: add auto-detection of `trust_remote_code=True` for `allenai/*` model names
  - Verify the exact HuggingFace model IDs by searching HuggingFace Hub (the IDs above are best-effort; confirm before running)
  - Add tests for new model validation in existing config tests

  **Must NOT do**:
  - Do NOT remove or reorder existing models in `SUPPORTED_MODELS`
  - Do NOT change the default model
  - Do NOT add complex model compatibility checking — just a simple prefix match for `trust_remote_code`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/steering_geometry/config.py:10-26` — `SUPPORTED_MODELS` tuple and `SUPPORTED_CONCEPTS` — add new model IDs here, follow exact string format
  - `src/steering_geometry/config.py` — `ModelConfig` dataclass — check if `trust_remote_code` field exists or needs adding
  - `src/steering_geometry/models.py` — `HookedModel` class — find where `AutoModelForCausalLM.from_pretrained()` is called to understand how `trust_remote_code` should be passed

  **API/Type References**:
  - `tests/unit/test_config_main.py` — existing config validation tests — follow this pattern for testing new model IDs

  **WHY Each Reference Matters**:
  - `config.py` SUPPORTED_MODELS: Must add strings in exact same format (full HuggingFace ID with org prefix)
  - `models.py` HookedModel: Need to understand if `trust_remote_code` is already handled or needs to be threaded through config → model loading
  - `test_config_main.py`: Must follow existing test patterns to avoid breaking the 231-test suite

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: New models are in SUPPORTED_MODELS
    Tool: Bash
    Preconditions: config.py has been modified
    Steps:
      1. Run: uv run python -c "from steering_geometry.config import SUPPORTED_MODELS; assert 'Qwen/Qwen3-14B' in SUPPORTED_MODELS; assert 'allenai/Olmo-3-1125-32B' in SUPPORTED_MODELS"
      2. Assert exit code 0
    Expected Result: Both model IDs present in SUPPORTED_MODELS
    Failure Indicators: AssertionError or ImportError
    Evidence: .omo/evidence/task-1-models-registered.txt

  Scenario: trust_remote_code auto-detection for allenai models
    Tool: Bash
    Preconditions: models.py has been modified
    Steps:
      1. Run: uv run python -c "from steering_geometry.config import ModelConfig; c = ModelConfig(model_name='allenai/Olmo-3-1125-32B'); print(c.trust_remote_code)"
      2. Assert output is "True"
    Expected Result: trust_remote_code is True for allenai models
    Failure Indicators: AttributeError or output is "False"
    Evidence: .omo/evidence/task-1-trust-remote-code.txt
  ```

  **Commit**: YES
  - Message: `feat(config): add OLMo3-32B and Qwen3-14B to supported models`
  - Files: `src/steering_geometry/config.py`, `src/steering_geometry/models.py`, `tests/unit/test_config_main.py`
  - Pre-commit: `uv run pytest tests/unit/test_config_main.py`

- [x] 2. Create StabilitySweepConfig dataclass

  **What to do**:
  - Create a new `StabilitySweepConfig` dataclass in `config.py` (NOT in stability_comparison.py)
  - Fields:
    - `model_name: str` — HuggingFace model ID
    - `concept: str` — concept name (will be mapped internally: "safety"→"refusal", "politeness"→"polite", "sentiment"→"sentiment")
    - `n_values: list[int]` — sample sizes to sweep, default `[100, 500, 1000, 5000, 10000]`
    - `layers: list[float]` — layer fractions to test, default `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`
    - `num_runs: int` — number of independent runs per setting, default `5`
    - `seed: int` — base random seed, default `42`
    - `output_dir: Path | str` — where to save results, default `"outputs/stability_sweep"`
    - `device: str` — torch device, default `"auto"`
    - `dtype: str` — model dtype, default `"float16"`
  - Add a `__post_init__` that validates `concept` is in `SUPPORTED_CONCEPTS` and `model_name` is in `SUPPORTED_MODELS`
  - Add a `canonical_concept` property that maps paper names → codebase names (safety→refusal, politeness→polite, identity for sentiment)
  - Add a `display_concept` property that maps codebase names → paper names (for figure titles)
  - Write tests for the config class creation and validation

  **Must NOT do**:
  - Do NOT modify `StabilityComparisonConfig` — this is a NEW config class
  - Do NOT add new concept names to `SUPPORTED_CONCEPTS` — use the alias mapping approach
  - Do NOT create a config validation framework — just `__post_init__` with `ValueError`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/steering_geometry/config.py` — `StabilityComparisonConfig` dataclass — follow this exact pattern (field order, type hints, default values, `__post_init__` validation)
  - `src/steering_geometry/config.py:10-26` — `SUPPORTED_CONCEPTS` and `SUPPORTED_MODELS` — these are the validation targets

  **API/Type References**:
  - `src/steering_geometry/types.py` — existing type conventions (Path vs str, dataclass patterns)

  **WHY Each Reference Matters**:
  - `StabilityComparisonConfig`: Must match field style exactly (snake_case, type hints on all fields, defaults, Path|str for directory fields)
  - `SUPPORTED_CONCEPTS`: The `__post_init__` must validate against these canonical names

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Config creation with valid parameters
    Tool: Bash
    Preconditions: config.py has StabilitySweepConfig
    Steps:
      1. Run: uv run python -c "from steering_geometry.config import StabilitySweepConfig; c = StabilitySweepConfig(model_name='Qwen/Qwen3-1.7B', concept='refusal'); print(c.num_runs, c.n_values)"
      2. Assert output contains "5" and "[100, 500, 1000, 5000, 10000]"
    Expected Result: Config created with correct defaults
    Failure Indicators: ImportError, AttributeError, or wrong values
    Evidence: .omo/evidence/task-2-config-creation.txt

  Scenario: Invalid concept raises ValueError
    Tool: Bash
    Preconditions: config.py has StabilitySweepConfig
    Steps:
      1. Run: uv run python -c "from steering_geometry.config import StabilitySweepConfig; StabilitySweepConfig(model_name='Qwen/Qwen3-1.7B', concept='nonexistent')" 2>&1
      2. Assert output contains "ValueError"
    Expected Result: ValueError raised for unsupported concept
    Failure Indicators: No error raised, or wrong error type
    Evidence: .omo/evidence/task-2-config-validation.txt

  Scenario: Concept name mapping works
    Tool: Bash
    Preconditions: config has canonical_concept and display_concept properties
    Steps:
      1. Run: uv run python -c "from steering_geometry.config import StabilitySweepConfig; c = StabilitySweepConfig(model_name='Qwen/Qwen3-1.7B', concept='refusal'); print(c.display_concept)"
      2. Assert output is "Safety"
    Expected Result: "refusal" maps to "Safety" for figure titles
    Failure Indicators: Output is "refusal" or AttributeError
    Evidence: .omo/evidence/task-2-concept-mapping.txt
  ```

  **Commit**: YES
  - Message: `feat(config): add StabilitySweepConfig dataclass`
  - Files: `src/steering_geometry/config.py`, `tests/unit/test_config_main.py`
  - Pre-commit: `uv run pytest tests/unit/test_config_main.py`

- [x] 3. Implement run_stability_sweep() experiment runner

  **What to do**:
  - Add `run_stability_sweep()` function to `src/steering_geometry/stability_comparison.py`
  - This is the core experiment function. Signature:
    ```python
    def run_stability_sweep(config: StabilitySweepConfig) -> StabilitySweepResult:
    ```
  - Algorithm:
    1. Load model once (HookedModel from `models.py`)
    2. Load contrast pairs for the canonical concept (using existing `load_contrast_pairs()` from `extract.py`)
    3. Cap loaded data to max available (use existing `cap_examples()`)
    4. For each `n` in `config.n_values`:
       - For each `run_idx` in `range(config.num_runs)`:
         - Seed: `random.Random(config.seed + run_idx)`
         - Sample `n` items from loaded pairs (without replacement)
         - For each `layer_frac` in `config.layers`:
           - Extract DiM direction using existing extraction logic (mean aggregator)
           - Save vector to `outputs/stability_sweep/vectors/{concept}/n{n}_run{run_idx}_layer{layer_frac}.pt`
    5. For each `layer_frac`:
       - At each `n`: load the `num_runs` vectors → compute pairwise cos_sim using `compute_cosine_similarity_matrix()` → extract mean and std using `compute_stability_statistics()`
       - Average the mean cos_sim across all N values for this layer → this is the "layer score"
    6. Select best layer: `selected_layer = argmax(layer_score)` across all layers
    7. Build and return `StabilitySweepResult` containing:
       - `selected_layer: float`
       - `model_name: str`
       - `concept: str` (canonical name)
       - `display_concept: str` (paper name)
       - `per_n_data: dict[int, dict[str, float]]` — {N: {"mean": float, "std": float}} at selected layer
       - `all_layers_data: dict[float, dict[int, dict[str, float]]]` — full layer×N matrix for debugging
  - Create `StabilitySweepResult` dataclass in `types.py`
  - Model is loaded ONCE and reused across all (N, run, layer) iterations
  - Log progress: `f"[{model_name}] {concept}: N={n}, run={run_idx+1}/{num_runs}, layer={layer_frac}"`

  **Must NOT do**:
  - Do NOT modify `run_stability_comparison_experiment()` or `run_diff_means_experiment()`
  - Do NOT add checkpointing/resume logic
  - Do NOT create a new model-loading function — reuse `HookedModel`
  - Do NOT change the seeding strategy — use `random.Random(seed + run_idx)`

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 1, 2)
  - **Parallel Group**: Wave 2 (start after Wave 1 complete)
  - **Blocks**: Tasks 4, 5
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py:531-640` — `run_stability_comparison_experiment()` — FOLLOW THIS PATTERN for model loading, subset creation, multi-run loop, cos_sim computation, results structure
  - `src/steering_geometry/stability_comparison.py:179-279` — `run_diff_means_experiment()` — follow for varying N, vector saving convention, capping logic
  - `src/steering_geometry/stability_comparison.py` — `compute_cosine_similarity_matrix()` and `compute_stability_statistics()` — reuse directly (search for function names, line numbers may have shifted)
  - `src/steering_geometry/stability_comparison.py` — `select_token_subsets()` and vector saving/loading pattern

  **API/Type References**:
  - `src/steering_geometry/models.py` — `HookedModel` class — constructor signature, `get_activation()` or equivalent extraction method
  - `src/steering_geometry/extract.py` — `load_contrast_pairs()` — for loading dataset
  - `src/steering_geometry/stability_comparison.py` — `cap_examples()` — for capping to max available data (NOTE: lives here, not in extract.py)
  - `src/steering_geometry/types.py` — `SteeringVector`, `ContrastPair` — data types used in extraction
  - `src/steering_geometry/config.py` — `StabilitySweepConfig` (from Task 2) — the input config

  **Test References**:
  - `tests/test_stability_comparison.py` — existing stability tests with `mock_hooked_model` fixture — follow for mocking the model in tests
  - `tests/conftest.py` — `mock_hooked_model`, `sample_contrast_pairs` fixtures

  **WHY Each Reference Matters**:
  - `run_stability_comparison_experiment()`: This is the closest existing pattern — multi-run, per-layer cos_sim, statistics. The new function adds varying-N on top of this.
  - `run_diff_means_experiment()`: Shows how to vary N and cap examples. The subset-from-pool pattern is exactly what we need.
  - `compute_cosine_similarity_matrix()`: Direct reuse — no need to reimplement pairwise cos_sim.
  - `HookedModel`: Must understand constructor and extraction API to loop correctly.
  - `load_contrast_pairs()`: Must use existing data loading to get the contrast pairs for each concept.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Function signature and return type
    Tool: Bash
    Preconditions: stability_comparison.py has run_stability_sweep
    Steps:
      1. Run: uv run python -c "from steering_geometry.stability_comparison import run_stability_sweep; import inspect; sig = inspect.signature(run_stability_sweep); print(sig)"
      2. Assert output contains "config: StabilitySweepConfig"
    Expected Result: Function exists with correct signature
    Failure Indicators: ImportError or wrong signature
    Evidence: .omo/evidence/task-3-signature.txt

  Scenario: Dry-run with mock model produces correct result structure
    Tool: Bash (uv run pytest)
    Preconditions: Tests written with mock_hooked_model
    Steps:
      1. Run: uv run pytest tests/test_stability_sweep.py -k "test_run_stability_sweep" -v
      2. Assert all tests pass
    Expected Result: Test passes, result has selected_layer, per_n_data, all_layers_data
    Failure Indicators: Test failure, missing fields
    Evidence: .omo/evidence/task-3-dry-run.txt

  Scenario: Type check passes
    Tool: Bash
    Preconditions: All new code written
    Steps:
      1. Run: uv run mypy src/steering_geometry/stability_comparison.py
      2. Assert exit code 0
    Expected Result: No type errors
    Failure Indicators: Non-zero exit code
    Evidence: .omo/evidence/task-3-mypy.txt
  ```

  **Commit**: YES
  - Message: `feat(stability): implement run_stability_sweep experiment runner`
  - Files: `src/steering_geometry/stability_comparison.py`, `src/steering_geometry/types.py`
  - Pre-commit: `uv run mypy src/ && uv run ruff check src/`

- [x] 4. Implement sweep results persistence (JSON save/load)

  **What to do**:
  - Add `save_sweep_results()` function to `stability_comparison.py`:
    - Saves a single (model, concept) result as JSON to `outputs/stability_sweep/results_{model_slug}_{concept}.json`
    - JSON schema:
      ```json
      {
        "model_name": "Qwen/Qwen3-1.7B",
        "concept": "refusal",
        "display_concept": "Safety",
        "selected_layer": 0.7,
        "per_n_data": {
          "100": {"mean": 0.82, "std": 0.05},
          "500": {"mean": 0.91, "std": 0.03}
        },
        "all_layers_data": {
          "0.7": {
            "100": {"mean": 0.82, "std": 0.05},
            "500": {"mean": 0.91, "std": 0.03}
          }
        },
        "config": { /* StabilitySweepConfig fields */ },
        "timestamp": "2026-04-28T..."
      }
      ```
    - Use `safe_model_name()` from `utils.py` for model slug
  - Add `load_sweep_results()` function:
    - Loads all JSON files matching `outputs/stability_sweep/results_*.json`
    - Returns `dict[tuple[str, str], StabilitySweepResult]` keyed by (model_name, concept)
  - Add `load_sweep_results_for_plotting()` convenience function:
    - Returns data structured for plotting: `{concept: {model_name: {n: (mean, std)}}}`

  **Must NOT do**:
  - Do NOT use pickle — JSON only for human readability
  - Do NOT add schema validation library — simple dict-to-JSON

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 3 for StabilitySweepResult type)
  - **Parallel Group**: Wave 2 (after Task 3)
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py:544-570` — `save_results_json()` — follow this pattern for JSON saving (timestamp, config serialization, Path handling)
  - `src/steering_geometry/utils.py` — `safe_model_name()` — for converting model names to filesystem-safe slugs

  **WHY Each Reference Matters**:
  - `save_results_json()`: Exact pattern for JSON persistence in this codebase — must match style

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: JSON roundtrip preserves data
    Tool: Bash (uv run pytest)
    Preconditions: save/load functions exist
    Steps:
      1. Run: uv run pytest tests/test_stability_sweep.py -k "test_json_roundtrip" -v
      2. Assert test passes (save → load → compare selected_layer and per_n_data)
    Expected Result: Loaded data matches saved data exactly
    Failure Indicators: Test failure, float precision issues
    Evidence: .omo/evidence/task-4-json-roundtrip.txt

  Scenario: Load multiple results for plotting
    Tool: Bash (uv run pytest)
    Preconditions: load_sweep_results_for_plotting exists
    Steps:
      1. Run: uv run pytest tests/test_stability_sweep.py -k "test_load_for_plotting" -v
      2. Assert test passes with correct {concept: {model: {n: (mean, std)}}} structure
    Expected Result: Plotting-ready data structure
    Failure Indicators: Wrong nesting or missing keys
    Evidence: .omo/evidence/task-4-plotting-data.txt
  ```

  **Commit**: YES
  - Message: `feat(stability): add JSON persistence for sweep results`
  - Files: `src/steering_geometry/stability_comparison.py`
  - Pre-commit: `uv run mypy src/ && uv run ruff check src/`

- [x] 5. Implement plot_stability_sweep() line-plot function

  **What to do**:
  - Add `plot_stability_sweep()` function to `stability_comparison.py`:
    ```python
    def plot_stability_sweep(
        results: dict[str, dict[str, StabilitySweepResult]],  # {concept: {model: result}}
        output_dir: Path | str = "outputs/stability_sweep",
        model_colors: dict[str, str] | None = None,
        model_labels: dict[str, str] | None = None,
    ) -> list[Path]:
    ```
  - For each concept: create ONE figure with:
    - x-axis: N on log scale, tick marks at {100, 500, 1000, 5000, 10000}
    - y-axis: mean pairwise cos_sim, range [0, 1]
    - 4 lines, one per model, with distinct colors (use tab10 or similar)
    - Shaded band ±1 std around each line (alpha=0.2)
    - Title: concept display name (e.g., "Safety")
    - Subtitle: "Layer L={selected_layer}" for each model (or note in legend)
    - Legend with model display names (e.g., "OLMo3-7B", not "allenai/Olmo-3-1125-32B")
    - Grid lines, clean axis labels
    - Save as PDF: `outputs/stability_sweep/{concept}_stability_sweep.pdf`
  - Default model display names:
    - `"allenai/Olmo-3-1125-7B"` → "OLMo3-7B"
    - `"allenai/Olmo-3-1125-32B"` → "OLMo3-32B"
    - `"Qwen/Qwen3-1.7B"` → "Qwen3-1.7B"
    - `"Qwen/Qwen3-14B"` → "Qwen3-14B"
  - Default colors: use matplotlib tab10 color cycle
  - Figure size: (6, 4) — standard single-column figure
  - Font sizes: title=14, axis labels=12, tick labels=10, legend=10
  - Lazy import: `import matplotlib.pyplot as plt` at top of function (follow existing pattern)
  - Return list of 3 output PDF paths

  **Must NOT do**:
  - Do NOT use seaborn — matplotlib only (matches existing codebase)
  - Do NOT add a plotting framework — one focused function
  - Do NOT add interactive plotting — PDF output only
  - Do NOT hardcode model names in the function — use `model_labels` parameter with defaults

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 4 for data loading)
  - **Parallel Group**: Wave 2 (after Task 4)
  - **Blocks**: Tasks 6, 7
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py:65-116` — `plot_heatmap()` — follow for matplotlib lazy import, PDF save, Path return, figure creation pattern
  - `src/steering_geometry/tdnv.py:782-858` — `plot_stability_trend()` — CLOSEST pattern: multi-line plot, log-scale Y, parameter values on X, viridis colormap. Follow this pattern for the line-plot structure.

  **API/Type References**:
  - `StabilitySweepResult` (from Task 3) — the data structure to plot

  **WHY Each Reference Matters**:
  - `plot_heatmap()`: Must follow exact same matplotlib setup (lazy import, `plt.savefig(..., format="pdf")`, `plt.close()`)
  - `plot_stability_trend()`: This is the closest existing line-plot pattern — multi-line with different colors, log scale, clean labels. Adapt it for the new figure.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Plotting with synthetic data produces valid PDFs
    Tool: Bash (uv run pytest)
    Preconditions: plot_stability_sweep function exists
    Steps:
      1. Run: uv run pytest tests/test_stability_sweep.py -k "test_plot_stability_sweep" -v
      2. Test creates synthetic StabilitySweepResult objects (4 models × 3 concepts)
      3. Calls plot_stability_sweep(synthetic_data, output_dir=tmp_path)
      4. Asserts 3 PDF files exist and are non-empty
    Expected Result: 3 PDFs created, each > 0 bytes
    Failure Indicators: Missing files, 0-byte files, or test failure
    Evidence: .omo/evidence/task-5-plot-output.txt

  Scenario: PDF files have correct naming convention
    Tool: Bash
    Preconditions: Plotting has been tested
    Steps:
      1. Run: uv run python -c "
      from steering_geometry.stability_comparison import plot_stability_sweep
      # ... create synthetic data ...
      paths = plot_stability_sweep(synthetic_data, output_dir='/tmp/test_plots')
      for p in paths: print(p.name)
      "
      2. Assert output contains "safety_stability_sweep.pdf", "politeness_stability_sweep.pdf", "sentiment_stability_sweep.pdf"
    Expected Result: Correct filenames with concept display names
    Failure Indicators: Wrong filenames or paths
    Evidence: .omo/evidence/task-5-naming.txt

  Scenario: Figure has 4 model lines per concept
    Tool: Bash (uv run pytest)
    Preconditions: Test with synthetic data
    Steps:
      1. Run: uv run pytest tests/test_stability_sweep.py -k "test_four_model_lines" -v
      2. Test uses matplotlib to read back the figure and count Line2D objects
    Expected Result: Each figure has exactly 4 Line2D objects (one per model)
    Failure Indicators: Wrong number of lines
    Evidence: .omo/evidence/task-5-line-count.txt
  ```

  **Commit**: YES
  - Message: `feat(stability): add plot_stability_sweep line-plot function`
  - Files: `src/steering_geometry/stability_comparison.py`
  - Pre-commit: `uv run mypy src/ && uv run ruff check src/`

- [x] 6. Create run_stability_sweep.sh orchestration script

  **What to do**:
  - Create `scripts/vector_analysis/run_stability_sweep.sh`
  - Follow existing script patterns (see `run_stability_comparison.sh`)
  - Script structure:
    ```bash
    #!/usr/bin/env bash
    set -euo pipefail

    # Load ALL_CONCEPTS shell variable
    eval $(uv run python -m steering_geometry --shell)

    # Configurable parameters (with defaults)
    MODELS="${MODELS:-"Qwen/Qwen3-1.7B Qwen/Qwen3-14B allenai/Olmo-3-1125-7B allenai/Olmo-3-1125-32B"}"
    CONCEPTS="${CONCEPTS:-"refusal polite sentiment"}"
    N_VALUES="${N_VALUES:-"100 500 1000 5000 10000"}"
    LAYERS="${LAYERS:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"}"
    NUM_RUNS="${NUM_RUNS:-5}"
    OUTPUT_DIR="${OUTPUT_DIR:-"outputs/stability_sweep"}"

    TOTAL=$(echo "$MODELS" | wc -w)
    CURRENT=0

    for MODEL in $MODELS; do
      CURRENT=$((CURRENT + 1))
      echo "=== [$CURRENT/$TOTAL] Model: $MODEL ==="
      for CONCEPT in $CONCEPTS; do
        echo "--- Running sweep: $MODEL / $CONCEPT ---"
        uv run python -c "
    from steering_geometry.stability_comparison import run_stability_sweep, save_sweep_results
    from steering_geometry.config import StabilitySweepConfig

    config = StabilitySweepConfig(
        model_name='$MODEL',
        concept='$CONCEPT',
        n_values=[$N_VALUES],
        layers=[$LAYERS],
        num_runs=$NUM_RUNS,
        output_dir='$OUTPUT_DIR',
    )
    result = run_stability_sweep(config)
    save_sweep_results(result, output_dir='$OUTPUT_DIR')
    print(f'Selected layer: {result.selected_layer}')
    for n, data in sorted(result.per_n_data.items()):
        print(f'  N={n}: cos_sim={data[\"mean\"]:.4f} ± {data[\"std\"]:.4f}')
        "
      done
    done

    echo "=== Generating plots ==="
    uv run python -c "
    from steering_geometry.stability_comparison import load_sweep_results_for_plotting, plot_stability_sweep
    results = load_sweep_results_for_plotting('$OUTPUT_DIR')
    paths = plot_stability_sweep(results, output_dir='$OUTPUT_DIR')
    for p in paths:
        print(f'Saved: {p}')
    "

    echo "=== Done ==="
    ```
  - Add usage comment at top of script

  **Must NOT do**:
  - Do NOT create Python files in `scripts/` — shell only
  - Do NOT add complex argument parsing — use environment variables like existing scripts
  - Do NOT add parallel model loading — sequential only

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 7)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Tasks 3, 4, 5

  **References**:

  **Pattern References**:
  - `scripts/vector_analysis/run_stability_comparison.sh` — EXACT pattern to follow: `set -euo pipefail`, `eval $(uv run python -m steering_geometry --shell)`, progress counters, model loop, inline Python
  - `scripts/vector_analysis/quick_diff_means_heatmaps.sh` — alternative pattern for reference

  **WHY Each Reference Matters**:
  - `run_stability_comparison.sh`: Must match exact shell script conventions used in this project

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Script is syntactically valid
    Tool: Bash
    Preconditions: Script file exists
    Steps:
      1. Run: bash -n scripts/vector_analysis/run_stability_sweep.sh
      2. Assert exit code 0 (syntax check passes)
    Expected Result: No syntax errors
    Failure Indicators: Non-zero exit code with parse error
    Evidence: .omo/evidence/task-6-syntax-check.txt

  Scenario: Script has correct shebang and set options
    Tool: Bash
    Preconditions: Script file exists
    Steps:
      1. Run: head -3 scripts/vector_analysis/run_stability_sweep.sh
      2. Assert first line is #!/usr/bin/env bash
      3. Assert second line contains "set -euo pipefail"
    Expected Result: Proper shell script header
    Failure Indicators: Missing shebang or set options
    Evidence: .omo/evidence/task-6-header.txt
  ```

  **Commit**: YES
  - Message: `feat(scripts): add run_stability_sweep.sh orchestration script`
  - Files: `scripts/vector_analysis/run_stability_sweep.sh`
  - Pre-commit: `bash -n scripts/vector_analysis/run_stability_sweep.sh`

- [x] 7. Write unit tests for new code

  **What to do**:
  - Create `tests/test_stability_sweep.py`
  - Tests to write (using `mock_hooked_model` and `sample_contrast_pairs` fixtures from `conftest.py`):
    1. `test_stability_sweep_config_creation` — valid config creates without error
    2. `test_stability_sweep_config_invalid_concept` — unsupported concept raises ValueError
    3. `test_stability_sweep_config_concept_mapping` — canonical_concept and display_concept work
    4. `test_run_stability_sweep_returns_result` — mock model, small sweep (2 N values, 2 layers, 2 runs), verify result structure
    5. `test_run_stability_sweep_layer_selection` — verify selected_layer is the one with highest average cos_sim
    6. `test_run_stability_sweep_per_n_data_keys` — verify per_n_data has correct N values as keys
    7. `test_json_roundtrip` — save result to tmp_path → load → verify data matches
    8. `test_load_for_plotting_structure` — create multiple results → verify nesting {concept: {model: {n: (mean, std)}}}
    9. `test_plot_stability_sweep_creates_pdfs` — synthetic data → plot → verify 3 PDFs exist and are non-empty
    10. `test_plot_stability_sweep_four_lines` — synthetic data → plot → verify each figure has 4 Line2D objects
  - Follow existing test conventions: plain functions, `assert` statements, `-> None` return types, use `tmp_path` fixture

  **Must NOT do**:
  - Do NOT require GPU for tests — mock the model
  - Do NOT add slow tests without `@pytest.mark.slow` marker
  - Do NOT test internal implementation details — test public API behavior

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 6)
  - **Parallel Group**: Wave 3
  - **Blocks**: Final Verification
  - **Blocked By**: Tasks 3, 4, 5

  **References**:

  **Pattern References**:
  - `tests/test_stability_comparison.py` — existing stability tests with mocked model — follow exactly for mock setup, assertions, test structure
  - `tests/conftest.py` — `mock_hooked_model`, `sample_contrast_pairs` fixtures — reuse directly
  - `tests/test_experiments.py` — additional experiment testing patterns

  **API/Type References**:
  - `StabilitySweepConfig` (Task 2) — what to test
  - `StabilitySweepResult` (Task 3) — what to assert on
  - `run_stability_sweep()` (Task 3) — function to test
  - `save_sweep_results()`, `load_sweep_results()` (Task 4) — functions to test
  - `plot_stability_sweep()` (Task 5) — function to test

  **WHY Each Reference Matters**:
  - `test_stability_comparison.py`: Must match exact mock patterns (how to mock HookedModel, how to create fake activations)
  - `conftest.py`: Fixtures are shared — don't recreate them

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All new tests pass
    Tool: Bash
    Preconditions: test_stability_sweep.py exists
    Steps:
      1. Run: uv run pytest tests/test_stability_sweep.py -v
      2. Assert all tests pass (10 tests, 0 failures)
    Expected Result: All 10 tests pass
    Failure Indicators: Any test failure
    Evidence: .omo/evidence/task-7-tests-pass.txt

  Scenario: Full test suite still passes
    Tool: Bash
    Preconditions: All tests written
    Steps:
      1. Run: uv run pytest
      2. Assert all tests pass (231 + 10 new = 241+)
    Expected Result: No regressions in existing tests
    Failure Indicators: Any test failure count increase
    Evidence: .omo/evidence/task-7-full-suite.txt

  Scenario: Lint and type check
    Tool: Bash
    Preconditions: All code written
    Steps:
      1. Run: uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/
      2. Assert all pass
    Expected Result: 0 violations, 0 errors
    Failure Indicators: Any violations or errors
    Evidence: .omo/evidence/task-7-lint-type.txt
  ```

  **Commit**: YES
  - Message: `test(stability): add unit tests for sweep experiment and plotting`
  - Files: `tests/test_stability_sweep.py`
  - Pre-commit: `uv run pytest`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Review all changed files for: `Any` usage, bare `except`, `print()` in non-CLI code, unused imports, AI slop (excessive comments, over-abstraction, generic names).
  Output: `Ruff [PASS/FAIL] | Mypy [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Verify: (1) The shell script runs without errors (or fails with clear "needs GPU" message on CPU). (2) A tiny smoke test using mock data produces 3 PDF files. (3) The plotting function produces non-empty PDFs when given synthetic results data. (4) JSON roundtrip: save → load → re-plot produces identical figures.
  Output: `Smoke [PASS/FAIL] | Plots [3/3] | JSON Roundtrip [PASS/FAIL] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Verify no existing experiment functions were modified. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Existing Code [CLEAN/N modifications] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| # | Commit Message | Files | Pre-commit |
|---|---------------|-------|------------|
| 1 | `feat(config): add OLMo3-32B and Qwen3-14B to supported models` | `config.py` | `uv run pytest tests/unit/test_config_main.py` |
| 2 | `feat(config): add StabilitySweepConfig dataclass` | `config.py` | `uv run pytest tests/unit/test_config_main.py` |
| 3 | `feat(stability): implement run_stability_sweep experiment runner` | `stability_comparison.py` | `uv run pytest tests/test_stability_comparison.py` |
| 4 | `feat(stability): add JSON persistence for sweep results` | `stability_comparison.py` | `uv run pytest tests/test_stability_comparison.py` |
| 5 | `feat(stability): add plot_stability_sweep line-plot function` | `stability_comparison.py` | `uv run pytest tests/test_stability_comparison.py` |
| 6 | `feat(scripts): add run_stability_sweep.sh orchestration script` | `scripts/vector_analysis/run_stability_sweep.sh` | `shellcheck` |
| 7 | `test(stability): add unit tests for sweep experiment and plotting` | `tests/test_stability_sweep.py` | `uv run pytest` |

---

## Success Criteria

### Verification Commands
```bash
uv sync                                              # Expected: success
uv run ruff check src/ tests/                        # Expected: 0 violations
uv run ruff format --check src/ tests/               # Expected: already formatted
uv run mypy src/                                     # Expected: 0 errors
uv run pytest                                        # Expected: all tests pass (231+)
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] 3 PDF figures can be generated by the shell script
- [ ] JSON results persist and can be re-loaded for re-plotting
