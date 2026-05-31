# TDNV Stability Analysis: Multi-Concept Extension + 5 Experimental Scripts

## TL;DR

> **Quick Summary**: Extend TDNV module with multi-concept metrics (binary concepts + MMLU-Pro categories) and create 5 shell scripts to analyze TDNV stability across dataset size, random seeds, token selection strategies (last-n and top-k discriminative).
>
> **Deliverables**:
> - 2 new TDNV functions in `tdnv.py` (multi-concept binary, multi-concept MMLU)
> - Updated `MMLUQuestion` TypedDict with `category` field
> - 5 shell scripts in `scripts/tdnv/` for stability analysis
> - TDD tests in `tests/unit/test_tdnv.py`
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves (types/tests, implementations, scripts)
> **Critical Path**: Types → Multi-concept impl → Scripts → Integration

---

## Context

### Original Request
User wants to analyze TDNV stability for two types of TDNV metrics:
1. **Binary TDNV**: Single concept with positive/negative classes (already exists)
2. **Multi-concept TDNV**: Global TDNV across all class-conditioned groups

And answer 5 research questions:
1. Does changing dataset size alter TDNV trend?
2. Do different random seeds alter TDNV trend?
3. Does selecting only last-n tokens alter TDNV trend?
4. Does selecting top-k tokens alter TDNV trend?
5. What is "top-k tokens" in this context?

### Interview Summary
**Key Discussions**:
- Multi-concept TDNV: Implement `compute_tdnv_multi_concept()` for binary concepts (6 groups: polite±, sentiment±, refusal±)
- MMLU-Pro support: Add `compute_tdnv_mmlu()` for multi-class TDNV (14 categories)
- Top-k tokens: Discriminative selection (close to own centroid + far from other centroid)
- Output format: JSON metrics + PDF plots for all scripts
- Default model: Qwen/Qwen3-1.7B
- Test strategy: TDD (tests first)

**Research Findings**:
- MMLUQuestion TypedDict is **missing `category` field** (must add)
- Current TDNV uses ALL non-zero tokens; extraction uses last token only
- Discriminative token selection already exists in `extract.py:132-167`
- Scripts directory structure: `scripts/tdnv/`

### Metis Review
**Identified Gaps** (addressed):
- Multi-concept TDNV formula undefined → Using formula from slides
- MMLUQuestion missing category field → Adding to types.py
- Token selection discrepancy → Scripts will support both modes
- Parameter ranges undefined → User will specify later
- Edge cases for missing categories → Add skip/empty category handling

---

## Work Objectives

### Core Objective
Extend the TDNV module to support multi-concept separability metrics and create a comprehensive suite of experimental scripts to analyze TDNV stability across different experimental conditions.

### Concrete Deliverables
- `src/steering_geometry/types.py`: Updated `MMLUQuestion` with `category` field
- `src/steering_geometry/tdnv.py`:
  - `compute_tdnv_multi_concept()` - TDNV across 6 binary-concept groups
  - `compute_tdnv_mmlu()` - TDNV across MMLU-Pro categories
  - Token selection helpers (last-n, top-k discriminative)
  - Plotting functions for stability trends
- `tests/unit/test_tdnv.py`: TDD tests for all new functions
- `scripts/tdnv/`:
  - `run_dataset_size_stability.sh`
  - `run_seed_stability.sh`
  - `run_last_n_stability.sh`
  - `run_top_k_stability.sh`
  - `explain_top_k_tokens.sh`

### Definition of Done
- [ ] All new functions pass TDD tests
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run mypy src/` → 0 errors
- [ ] `uv run pytest tests/unit/test_tdnv.py` → all pass
- [ ] Each shell script produces JSON + plot output
- [ ] Scripts work with Qwen/Qwen3-1.7B default model

### Must Have
- Multi-concept TDNV using exact formula from slides
- MMLU-Pro category support
- 5 stability analysis scripts
- TDD test coverage

### Must NOT Have (Guardrails)
- Do NOT modify extraction/aggregation logic in `extract.py`
- Do NOT change existing `compute_tdnv()` signature
- Do NOT add new dependencies (use existing matplotlib, torch)
- Do NOT create Python files in `scripts/` (shell only)
- Do NOT assume specific parameter ranges (user will provide)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (TDD)
- **Framework**: pytest
- **If TDD**: Each task follows RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/Module**: Use Bash (uv run pytest) — Run tests, verify pass/fail
- **CLI/Scripts**: Use Bash — Run scripts with test args, check output files exist

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation):
├── Task 1: Add category field to MMLUQuestion TypedDict [quick]
├── Task 2: Write TDD tests for compute_tdnv_multi_concept [quick]
├── Task 3: Write TDD tests for compute_tdnv_mmlu [quick]
└── Task 4: Write TDD tests for token selection helpers [quick]

Wave 2 (After Wave 1 — implementations, MAX PARALLEL):
├── Task 5: Implement compute_tdnv_multi_concept (depends: 2) [unspecified-high]
├── Task 6: Implement compute_tdnv_mmlu (depends: 1, 3) [unspecified-high]
├── Task 7: Implement token selection helpers (depends: 4) [quick]
└── Task 8: Implement plotting functions (depends: 5, 6, 7) [quick]

Wave 3 (After Wave 2 — scripts, MAX PARALLEL):
├── Task 9: Script: dataset size stability (depends: 5, 8) [quick]
├── Task 10: Script: random seed stability (depends: 5, 8) [quick]
├── Task 11: Script: last-n tokens stability (depends: 5, 7, 8) [quick]
├── Task 12: Script: top-k tokens stability (depends: 5, 7, 8) [quick]
└── Task 13: Script: explain top-k tokens (depends: 7) [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 6 → Task 10 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 8 (Waves 2 & 3)
```

### Dependency Matrix

- **1-4**: — — 5-8, —
- **5**: 2 — 8, 9-12, —
- **6**: 1, 3 — 8, —, —
- **7**: 4 — 8, 11-13, —
- **8**: 5, 6, 7 — 9-13, —
- **9**: 5, 8 — —, —
- **10**: 5, 8 — —, —
- **11**: 5, 7, 8 — —, —
- **12**: 5, 7, 8 — —, —
- **13**: 7 — —, —

### Agent Dispatch Summary

- **Wave 1**: **4** — T1-T4 → `quick`
- **Wave 2**: **4** — T5-T6 → `unspecified-high`, T7-T8 → `quick`
- **Wave 3**: **5** — T9-T13 → `quick`
- **FINAL**: **4** — F1 → `oracle`, F2-F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Add `category` field to MMLUQuestion TypedDict

  **What to do**:
  - Add `category: str` field to `MMLUQuestion` TypedDict in `types.py:63-74`
  - This field is required for MMLU-Pro category-based TDNV
  - No migration needed - new field will be populated when loading MMLU-Pro

  **Must NOT do**:
  - Do NOT add Optional typing - category is always present in MMLU-Pro
  - Do NOT break existing MMLUEvaluator usage

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single field addition, straightforward type update
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 6 (compute_tdnv_mmlu)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `src/steering_geometry/types.py:63-74` - MMLUQuestion TypedDict definition
  - `src/steering_geometry/apply_steering.py:350` - MMLU-Pro loading
  - `TIGER-Lab/MMLU-Pro` dataset - has `category` field with 14 values

  **Acceptance Criteria**:
  - [ ] `category: str` field added to MMLUQuestion
  - [ ] `uv run mypy src/steering_geometry/types.py` → 0 errors
  - [ ] No breaking changes to MMLUEvaluator

  **QA Scenarios**:

  ```
  Scenario: Type check passes after field addition
    Tool: Bash
    Preconditions: types.py updated with category field
    Steps:
      1. Run: uv run mypy src/steering_geometry/types.py
    Expected Result: Success: no issues found in 1 source file
    Failure Indicators: error: Missing annotation, error: Incompatible types
    Evidence: .sisyphus/evidence/task-01-mypy-pass.txt

  Scenario: Import still works
    Tool: Bash
    Preconditions: types.py updated
    Steps:
      1. Run: uv run python -c "from steering_geometry.types import MMLUQuestion; print('OK')"
    Expected Result: OK
    Failure Indicators: ImportError, AttributeError
    Evidence: .sisyphus/evidence/task-01-import-ok.txt
  ```

  **Commit**: YES
  - Message: `feat(types): add category field to MMLUQuestion for multi-class TDNV`
  - Files: `src/steering_geometry/types.py`
  - Pre-commit: `uv run mypy src/steering_geometry/types.py`

- [x] 2. Write TDD tests for compute_tdnv_multi_concept

  **What to do**:
  - Create `tests/unit/test_tdnv.py` if not exists
  - Write tests for `compute_tdnv_multi_concept()`:
    - Test with 2 concepts (4 groups)
    - Test with 3 concepts (6 groups)
    - Test TDNV value is correct per formula
    - Test handling of empty groups
    - Test variance and distance computation
  - Tests should FAIL initially (TDD RED phase)

  **Must NOT do**:
  - Do NOT implement the function yet - tests first
  - Do NOT use `@pytest.skip` - write real tests

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Test writing is straightforward, follows existing patterns
  - **Skills**: [`/write-test`]
    - `/write-test`: Python test writing with pytest patterns
  - **Skills Evaluated but Omitted**:
    - None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Task 5 (implementation)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `src/steering_geometry/tdnv.py:73-121` - Existing `compute_tdnv()` to follow pattern
  - `tests/unit/test_evaluation.py` - Existing test patterns
  - Formula from slides: TDNV = (1/M(M-1)) * Σ (var_g + var_g') / (2||mean_g - mean_g'||^2)

  **Acceptance Criteria**:
  - [ ] Test file `tests/unit/test_tdnv.py` created
  - [ ] At least 5 test functions covering edge cases
  - [ ] `uv run pytest tests/unit/test_tdnv.py::test_compute_tdnv_multi_concept -v` → FAILS (expected)
  - [ ] Tests import the function correctly (will fail at import until implemented)

  **QA Scenarios**:

  ```
  Scenario: Tests exist and are properly structured
    Tool: Bash
    Preconditions: test_tdnv.py created
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py --collect-only
    Expected Result: Shows collected tests (may fail to run)
    Failure Indicators: No tests collected, syntax error
    Evidence: .sisyphus/evidence/task-02-tests-exist.txt

  Scenario: Tests fail as expected (TDD RED)
    Tool: Bash
    Preconditions: Tests written, function not implemented
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py::test_compute_tdnv_multi_concept -v 2>&1 | head -20
    Expected Result: Contains "FAILED" or "ImportError" (expected in TDD RED)
    Failure Indicators: All tests pass (means function already exists)
    Evidence: .sisyphus/evidence/task-02-tdd-red.txt
  ```

  **Commit**: NO (groups with Task 5)

- [x] 3. Write TDD tests for compute_tdnv_mmlu

  **What to do**:
  - Add tests to `tests/unit/test_tdnv.py` for `compute_tdnv_mmlu()`:
    - Test with mock MMLU-Pro data (multiple categories)
    - Test category extraction from MMLUQuestion
    - Test TDNV computation across categories
    - Test handling of missing category field
    - Test single category edge case
  - Tests should FAIL initially (TDD RED phase)

  **Must NOT do**:
  - Do NOT implement the function yet
  - Do NOT load real MMLU-Pro data in tests (use mocks)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Test writing, straightforward mocking
  - **Skills**: [`/write-test`]
    - `/write-test`: Python test writing with pytest
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 6 (implementation)
  - **Blocked By**: None (can start immediately, assumes Task 1 category field)

  **References**:
  - `tests/unit/test_tdnv.py` - Created in Task 2
  - `tests/conftest.py` - Existing fixtures
  - `src/steering_geometry/types.py:63-74` - MMLUQuestion (after Task 1)

  **Acceptance Criteria**:
  - [ ] At least 4 test functions for compute_tdnv_mmlu
  - [ ] Mock fixtures for MMLUQuestion with categories
  - [ ] Tests fail as expected (TDD RED)

  **QA Scenarios**:

  ```
  Scenario: MMLU tests exist and follow TDD
    Tool: Bash
    Preconditions: Tests written
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py --collect-only -q | grep -i mmlu
    Expected Result: Shows MMLU test functions
    Failure Indicators: No output, syntax error
    Evidence: .sisyphus/evidence/task-03-mmlu-tests-exist.txt
  ```

  **Commit**: NO (groups with Task 6)

- [x] 4. Write TDD tests for token selection helpers

  **What to do**:
  - Add tests to `tests/unit/test_tdnv.py` for token selection:
    - Test `select_last_n_tokens()`: extracts last n tokens from activations
    - Test `select_top_k_discriminative()`: selects discriminative tokens
    - Test edge cases: n > total tokens, n = 0, empty activations
    - Test discriminative scoring matches expected behavior
  - Tests should FAIL initially (TDD RED phase)

  **Must NOT do**:
  - Do NOT implement functions yet
  - Do NOT use real model activations (use synthetic tensors)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Test writing for utility functions
  - **Skills**: [`/write-test`]
    - `/write-test`: Python test writing
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 7 (implementation)
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/extract.py:132-167` - Existing discriminative token logic
  - Current TDNV token selection: `tdnv.py:170-176`

  **Acceptance Criteria**:
  - [ ] At least 4 test functions for token selection
  - [ ] Tests use synthetic tensor data
  - [ ] Tests fail as expected (TDD RED)

  **QA Scenarios**:

  ```
  Scenario: Token selection tests exist
    Tool: Bash
    Preconditions: Tests written
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py --collect-only -q | grep -i token
    Expected Result: Shows token selection test functions
    Failure Indicators: No output
    Evidence: .sisyphus/evidence/task-04-token-tests-exist.txt
  ```

  **Commit**: NO (groups with Task 7)

- [x] 5. Implement compute_tdnv_multi_concept

  **What to do**:
  - Implement `compute_tdnv_multi_concept()` in `tdnv.py`:
    - Accept dict mapping concept name to (pos_activations, neg_activations)
    - Create 2T groups (T concepts × 2 classes)
    - Compute per-group mean and variance using `_compute_per_topic_stats()`
    - Compute TDNV = (1/M(M-1)) * Σ (var_g + var_g') / (2||mean_g - mean_g'||^2)
    - Return TDNVLayerMetrics with tdnv, norm_num, norm_den, energy
  - Make tests pass (TDD GREEN phase)

  **Must NOT do**:
  - Do NOT change signature of existing `compute_tdnv()`
  - Do NOT add external dependencies

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Algorithm implementation, needs careful math verification
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8)
  - **Blocks**: Tasks 8, 9, 10, 11, 12
  - **Blocked By**: Task 2 (tests must exist)

  **References**:
  - `src/steering_geometry/tdnv.py:73-121` - Existing `compute_tdnv()` pattern
  - `src/steering_geometry/tdnv.py:40-70` - `_compute_per_topic_stats()` helper
  - Tests in `tests/unit/test_tdnv.py` (Task 2)

  **Acceptance Criteria**:
  - [ ] Function signature: `compute_tdnv_multi_concept(concepts: dict[str, tuple[Tensor, Tensor]]) -> TDNVLayerMetrics`
  - [ ] `uv run pytest tests/unit/test_tdnv.py::test_compute_tdnv_multi_concept -v` → PASS
  - [ ] `uv run mypy src/steering_geometry/tdnv.py` → 0 errors

  **QA Scenarios**:

  ```
  Scenario: Multi-concept TDNV computes correctly
    Tool: Bash
    Preconditions: Function implemented, tests exist
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py::test_compute_tdnv_multi_concept -v
    Expected Result: All tests PASS
    Failure Indicators: FAILED, AssertionError
    Evidence: .sisyphus/evidence/task-05-multi-concept-pass.txt

  Scenario: Type check passes
    Tool: Bash
    Preconditions: Function implemented
    Steps:
      1. Run: uv run mypy src/steering_geometry/tdnv.py
    Expected Result: Success: no issues found
    Failure Indicators: error: Incompatible return type
    Evidence: .sisyphus/evidence/task-05-mypy-pass.txt
  ```

  **Commit**: YES
  - Message: `feat(tdnv): add compute_tdnv_multi_concept for binary concept groups`
  - Files: `src/steering_geometry/tdnv.py`, `tests/unit/test_tdnv.py`
  - Pre-commit: `uv run pytest tests/unit/test_tdnv.py`

- [x] 6. Implement compute_tdnv_mmlu

  **What to do**:
  - Implement `compute_tdnv_mmlu()` in `tdnv.py`:
    - Load MMLU-Pro validation set via HuggingFace datasets
    - Group questions by `category` field (14 categories)
    - For each category, extract activations using model
    - Compute TDNV across all category groups (multi-class, not binary)
    - Return TDNVLayerMetrics
  - Add `load_mmlu_by_category()` helper to extract questions by category
  - Make tests pass (TDD GREEN phase)

  **Must NOT do**:
  - Do NOT hardcode category list (extract from data)
  - Do NOT load full MMLU-Pro (use validation split only)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Data loading + algorithm implementation
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 7, 8)
  - **Blocks**: Task 8
  - **Blocked By**: Task 1 (category field), Task 3 (tests)

  **References**:
  - `src/steering_geometry/types.py:63-74` - MMLUQuestion with category (Task 1)
  - `src/steering_geometry/apply_steering.py:350` - MMLU-Pro loading pattern
  - Tests in `tests/unit/test_tdnv.py` (Task 3)

  **Acceptance Criteria**:
  - [ ] Function loads MMLU-Pro validation set
  - [ ] Groups questions by category field
  - [ ] `uv run pytest tests/unit/test_tdnv.py::test_compute_tdnv_mmlu -v` → PASS
  - [ ] Handles missing category gracefully

  **QA Scenarios**:

  ```
  Scenario: MMLU TDNV computes correctly
    Tool: Bash
    Preconditions: Function implemented
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py::test_compute_tdnv_mmlu -v
    Expected Result: All tests PASS
    Failure Indicators: FAILED
    Evidence: .sisyphus/evidence/task-06-mmlu-pass.txt

  Scenario: Function handles missing category
    Tool: Bash
    Preconditions: Function implemented
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py::test_compute_tdnv_mmlu_missing_category -v
    Expected Result: Test PASS (graceful handling)
    Failure Indicators: FAILED with KeyError
    Evidence: .sisyphus/evidence/task-06-missing-cat-pass.txt
  ```

  **Commit**: YES
  - Message: `feat(tdnv): add compute_tdnv_mmlu for MMLU-Pro categories`
  - Files: `src/steering_geometry/tdnv.py`, `tests/unit/test_tdnv.py`
  - Pre-commit: `uv run pytest tests/unit/test_tdnv.py`

- [x] 7. Implement token selection helpers

  **What to do**:
  - Implement in `tdnv.py`:
    - `select_last_n_tokens(activations: Tensor, n: int) -> Tensor`: Extract last n tokens
    - `select_top_k_discriminative(activations: Tensor, labels: list[int], k: int) -> Tensor`: Select discriminative tokens
  - For discriminative selection:
    - Compute class centroids
    - Score each token: s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
    - Select top-k tokens per class
  - Make tests pass (TDD GREEN phase)

  **Must NOT do**:
  - Do NOT duplicate logic from extract.py - reference or extract if needed
  - Do NOT modify model activations in-place

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Utility functions, straightforward implementation
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 8)
  - **Blocks**: Tasks 8, 11, 12, 13
  - **Blocked By**: Task 4 (tests)

  **References**:
  - `src/steering_geometry/extract.py:132-167` - Existing discriminative logic
  - Tests in `tests/unit/test_tdnv.py` (Task 4)

  **Acceptance Criteria**:
  - [ ] `select_last_n_tokens()` works for n > 0
  - [ ] `select_top_k_discriminative()` selects correct tokens
  - [ ] `uv run pytest tests/unit/test_tdnv.py::test_select_last_n -v` → PASS
  - [ ] `uv run pytest tests/unit/test_tdnv.py::test_select_top_k -v` → PASS

  **QA Scenarios**:

  ```
  Scenario: Last-n selection works
    Tool: Bash
    Preconditions: Function implemented
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py::test_select_last_n -v
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-07-last-n-pass.txt

  Scenario: Top-k discriminative works
    Tool: Bash
    Preconditions: Function implemented
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py::test_select_top_k -v
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-07-top-k-pass.txt
  ```

  **Commit**: YES
  - Message: `feat(tdnv): add token selection helpers for stability analysis`
  - Files: `src/steering_geometry/tdnv.py`, `tests/unit/test_tdnv.py`
  - Pre-commit: `uv run pytest tests/unit/test_tdnv.py`

- [x] 8. Implement stability trend plotting functions

  **What to do**:
  - Implement in `tdnv.py`:
    - `plot_stability_trend(results: list[TDNVResult], param_name: str, param_values: list, output_path: Path) -> Path`
    - Plot TDNV vs parameter value (e.g., dataset size, seed, n tokens)
    - Support multiple layers on same plot (different colors)
    - Save as PDF with legend and labels
  - Extend existing `plot_tdnv_trends()` pattern

  **Must NOT do**:
  - Do NOT add new plotting dependencies
  - Do NOT create interactive plots (PDF only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Plotting utilities, follows existing pattern
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: Tasks 9, 10, 11, 12
  - **Blocked By**: Tasks 5, 6, 7 (need TDNV results to plot)

  **References**:
  - `src/steering_geometry/tdnv.py:240-312` - Existing `plot_tdnv_trends()`
  - `matplotlib.pyplot` - Existing dependency

  **Acceptance Criteria**:
  - [ ] Function generates PDF plots
  - [ ] Plots show TDNV vs parameter value
  - [ ] Multiple layers shown with legend
  - [ ] Output path is returned

  **QA Scenarios**:

  ```
  Scenario: Plotting function works
    Tool: Bash
    Preconditions: Function implemented
    Steps:
      1. Run: uv run python -c "from steering_geometry.tdnv import plot_stability_trend; print('OK')"
    Expected Result: OK
    Evidence: .sisyphus/evidence/task-08-plot-import.txt

  Scenario: Plot generated correctly
    Tool: Bash
    Preconditions: Function implemented
    Steps:
      1. Create test script that calls plot_stability_trend with mock data
      2. Run: uv run python test_plot.py
      3. Check: ls -la /tmp/test_plot.pdf
    Expected Result: File exists and has content > 0 bytes
    Evidence: .sisyphus/evidence/task-08-plot-output.txt
  ```

  **Commit**: YES
  - Message: `feat(tdnv): add stability trend plotting functions`
  - Files: `src/steering_geometry/tdnv.py`
  - Pre-commit: `uv run mypy src/steering_geometry/tdnv.py`

- [x] 9. Script: dataset size stability analysis

  **What to do**:
  - Create `scripts/tdnv/run_dataset_size_stability.sh`
  - Script parameters:
    - `--concept`: Concept to analyze (default: polite)
    - `--model`: Model name (default: Qwen/Qwen3-1.7B)
    - `--sizes`: Comma-separated dataset sizes (user will specify)
    - `--output`: Output directory (default: outputs/tdnv/dataset_size/)
  - For each size in `--sizes`:
    - Load `num_pairs=size` contrast pairs
    - Compute TDNV for all layers
    - Save JSON to `{output}/{concept}_{model}_{size}.json`
  - Generate trend plot: TDNV vs dataset size per layer
  - Save plot to `{output}/{concept}_{model}_trend.pdf`

  **Must NOT do**:
  - Do NOT hardcode size values (use --sizes parameter)
  - Do NOT run in parallel (sequential for reproducibility)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Shell script wrapping Python module
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 10, 11, 12, 13)
  - **Blocks**: None
  - **Blocked By**: Tasks 5, 8 (need compute_tdnv_multi_concept + plotting)

  **References**:
  - `scripts/tdnv/quick_tdnv.sh` - Existing TDNV script pattern
  - `scripts/pipeline/run_pipeline.sh` - Parameter parsing pattern
  - `src/steering_geometry/tdnv.py` - TDNV functions (Tasks 5, 8)

  **Acceptance Criteria**:
  - [ ] Script accepts --concept, --model, --sizes, --output parameters
  - [ ] For each size: generates JSON file
  - [ ] Generates trend plot PDF
  - [ ] Script is executable (`chmod +x`)

  **QA Scenarios**:

  ```
  Scenario: Script runs and produces output
    Tool: Bash
    Preconditions: Script created, TDNV functions implemented
    Steps:
      1. Run: ./scripts/tdnv/run_dataset_size_stability.sh --concept polite --model "Qwen/Qwen3-1.7B" --sizes "100,500" --output /tmp/tdnv_test/
      2. Check: ls /tmp/tdnv_test/*.json | wc -l
      3. Check: ls /tmp/tdnv_test/*.pdf | wc -l
    Expected Result: 2 JSON files, 1 PDF file
    Failure Indicators: No files, script errors
    Evidence: .sisyphus/evidence/task-09-script-output.txt

  Scenario: JSON output valid
    Tool: Bash
    Preconditions: Script ran
    Steps:
      1. Run: python3 -c "import json; json.load(open('/tmp/tdnv_test/polite_Qwen3-1.7B_100.json'))"
    Expected Result: No error (valid JSON)
    Evidence: .sisyphus/evidence/task-09-json-valid.txt
  ```

  **Commit**: YES
  - Message: `feat(scripts): add dataset size TDNV stability script`
  - Files: `scripts/tdnv/run_dataset_size_stability.sh`
  - Pre-commit: None (shell script)

- [x] 10. Script: random seed stability analysis

  **What to do**:
  - Create `scripts/tdnv/run_seed_stability.sh`
  - Script parameters:
    - `--concept`: Concept to analyze (default: polite)
    - `--model`: Model name (default: Qwen/Qwen3-1.7B)
    - `--seeds`: Comma-separated random seeds (user will specify)
    - `--num-pairs`: Number of pairs per run (default: 500)
    - `--output`: Output directory (default: outputs/tdnv/seed/)
  - For each seed in `--seeds`:
    - Set random seed for sampling
    - Load contrast pairs with this seed
    - Compute TDNV for all layers
    - Save JSON to `{output}/{concept}_{model}_seed{N}.json`
  - Generate trend plot: TDNV vs seed per layer (should be stable)
  - Save plot to `{output}/{concept}_{model}_trend.pdf`

  **Must NOT do**:
  - Do NOT change model weights (only sampling seed)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Shell script, similar to Task 9
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 11, 12, 13)
  - **Blocks**: None
  - **Blocked By**: Tasks 5, 8

  **References**:
  - `scripts/tdnv/run_dataset_size_stability.sh` - Pattern from Task 9
  - `src/steering_geometry/utils.py:sample_with_seed()` - Seeding utility

  **Acceptance Criteria**:
  - [ ] Script accepts all parameters
  - [ ] Generates JSON per seed
  - [ ] Generates trend plot
  - [ ] Plot shows stability across seeds (low variance)

  **QA Scenarios**:

  ```
  Scenario: Script runs with multiple seeds
    Tool: Bash
    Preconditions: Script created
    Steps:
      1. Run: ./scripts/tdnv/run_seed_stability.sh --concept polite --seeds "0,1,2" --output /tmp/tdnv_seed/
      2. Check: ls /tmp/tdnv_seed/*.json | wc -l
    Expected Result: 3 JSON files
    Evidence: .sisyphus/evidence/task-10-seed-output.txt
  ```

  **Commit**: YES
  - Message: `feat(scripts): add random seed TDNV stability script`
  - Files: `scripts/tdnv/run_seed_stability.sh`

- [x] 11. Script: last-n tokens stability analysis

  **What to do**:
  - Create `scripts/tdnv/run_last_n_stability.sh`
  - Script parameters:
    - `--concept`: Concept to analyze (default: polite)
    - `--model`: Model name (default: Qwen/Qwen3-1.7B)
    - `--n-values`: Comma-separated last-n values (e.g., "1,5,10,20")
    - `--output`: Output directory (default: outputs/tdnv/last_n/)
  - For each n in `--n-values`:
    - Extract activations
    - Use `select_last_n_tokens(activations, n)`
    - Compute TDNV
    - Save JSON to `{output}/{concept}_{model}_last{n}.json`
  - Generate trend plot: TDNV vs n per layer
  - Save plot to `{output}/{concept}_{model}_trend.pdf`

  **Must NOT do**:
  - Do NOT use negative n values
  - Do NOT use n > total tokens (handle gracefully)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Shell script
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10, 12, 13)
  - **Blocks**: None
  - **Blocked By**: Tasks 5, 7, 8 (need multi-concept + token selection + plotting)

  **References**:
  - `scripts/tdnv/run_dataset_size_stability.sh` - Pattern
  - `src/steering_geometry/tdnv.py:select_last_n_tokens()` - Task 7

  **Acceptance Criteria**:
  - [ ] Script accepts all parameters
  - [ ] Generates JSON per n value
  - [ ] Generates trend plot
  - [ ] Handles n > total tokens gracefully

  **QA Scenarios**:

  ```
  Scenario: Script runs with different n values
    Tool: Bash
    Preconditions: Script created
    Steps:
      1. Run: ./scripts/tdnv/run_last_n_stability.sh --concept polite --n-values "1,5,10" --output /tmp/tdnv_last_n/
      2. Check: ls /tmp/tdnv_last_n/*.json | wc -l
    Expected Result: 3 JSON files
    Evidence: .sisyphus/evidence/task-11-last-n-output.txt
  ```

  **Commit**: YES
  - Message: `feat(scripts): add last-n tokens TDNV stability script`
  - Files: `scripts/tdnv/run_last_n_stability.sh`

- [x] 12. Script: top-k discriminative tokens stability analysis

  **What to do**:
  - Create `scripts/tdnv/run_top_k_stability.sh`
  - Script parameters:
    - `--concept`: Concept to analyze (default: polite)
    - `--model`: Model name (default: Qwen/Qwen3-1.7B)
    - `--k-values`: Comma-separated k values (e.g., "16,32,64,128")
    - `--output`: Output directory (default: outputs/tdnv/top_k/)
  - For each k in `--k-values`:
    - Extract activations
    - Use `select_top_k_discriminative(activations, labels, k)`
    - Compute TDNV
    - Save JSON to `{output}/{concept}_{model}_top{k}.json`
  - Generate trend plot: TDNV vs k per layer
  - Save plot to `{output}/{concept}_{model}_trend.pdf`

  **Must NOT do**:
  - Do NOT use k > total tokens
  - Do NOT compute discriminative scores differently than extract.py

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Shell script
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10, 11, 13)
  - **Blocks**: None
  - **Blocked By**: Tasks 5, 7, 8

  **References**:
  - `scripts/tdnv/run_last_n_stability.sh` - Similar pattern (Task 11)
  - `src/steering_geometry/tdnv.py:select_top_k_discriminative()` - Task 7
  - `src/steering_geometry/extract.py:132-167` - Discriminative logic reference

  **Acceptance Criteria**:
  - [ ] Script accepts all parameters
  - [ ] Generates JSON per k value
  - [ ] Generates trend plot
  - [ ] Uses same discriminative scoring as extract.py

  **QA Scenarios**:

  ```
  Scenario: Script runs with different k values
    Tool: Bash
    Preconditions: Script created
    Steps:
      1. Run: ./scripts/tdnv/run_top_k_stability.sh --concept polite --k-values "16,32,64" --output /tmp/tdnv_top_k/
      2. Check: ls /tmp/tdnv_top_k/*.json | wc -l
    Expected Result: 3 JSON files
    Evidence: .sisyphus/evidence/task-12-top-k-output.txt
  ```

  **Commit**: YES
  - Message: `feat(scripts): add top-k discriminative TDNV stability script`
  - Files: `scripts/tdnv/run_top_k_stability.sh`

- [x] 13. Script: explain top-k tokens (documentation)

  **What to do**:
  - Create `scripts/tdnv/explain_top_k_tokens.sh`
  - This script produces a markdown document explaining:
    - What "top-k discriminative tokens" means
    - The scoring formula: s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
    - Why this selects tokens that are:
      - Close to own class centroid (small ||h_i - μ_same||²)
      - Far from other class centroid (large ||h_i - μ_other||²)
    - Visual illustration of the concept
  - Output: `outputs/tdnv/top_k_explanation.md`

  **Must NOT do**:
  - Do NOT run actual TDNV computation (just explanation)
  - Do NOT generate plots (text only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Documentation script
  - **Skills**: []
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - None

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10, 11, 12)
  - **Blocks**: None
  - **Blocked By**: Task 7 (need token selection function to reference)

  **References**:
  - `src/steering_geometry/extract.py:132-167` - Discriminative token logic
  - `src/steering_geometry/tdnv.py:select_top_k_discriminative()` - Task 7

  **Acceptance Criteria**:
  - [ ] Script generates markdown file
  - [ ] Explanation includes formula
  - [ ] Explanation is clear and accurate
  - [ ] File saved to output path

  **QA Scenarios**:

  ```
  Scenario: Explanation script generates output
    Tool: Bash
    Preconditions: Script created
    Steps:
      1. Run: ./scripts/tdnv/explain_top_k_tokens.sh --output /tmp/tdnv_explain/
      2. Check: cat /tmp/tdnv_explain/top_k_explanation.md | head -20
    Expected Result: Markdown content with formula
    Evidence: .sisyphus/evidence/task-13-explain-output.txt

  Scenario: Explanation is accurate
    Tool: Bash
    Preconditions: Script ran
    Steps:
      1. Check: grep -c "||h_i" /tmp/tdnv_explain/top_k_explanation.md
    Expected Result: At least 1 (formula present)
    Evidence: .sisyphus/evidence/task-13-explain-formula.txt
  ```

  **Commit**: YES
  - Message: `feat(scripts): add top-k discriminative tokens explanation script`
  - Files: `scripts/tdnv/explain_top_k_tokens.sh`

---

## Final Verification Wave (MANDATORY)

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. Verify all "Must Have" present, all "Must NOT Have" absent. Check evidence files exist. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run mypy src/` + `uv run ruff check src/ tests/` + `uv run pytest`. Review for anti-patterns.
  Output: `Type Check [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run each script with minimal test args. Verify JSON output exists and contains expected fields. Verify plots are generated.
  Output: `Scripts [N/N working] | Output Format [N/N valid] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  Verify only planned files modified (tdnv.py, types.py, test_tdnv.py, scripts/tdnv/*.sh). No scope creep.
  Output: `Files [N/N compliant] | Scope Creep [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **1**: `feat(types): add category field to MMLUQuestion` — types.py
- **2-4**: Combined with implementations (Tasks 5-7)
- **5**: `feat(tdnv): add compute_tdnv_multi_concept for binary concepts` — tdnv.py, test_tdnv.py
- **6**: `feat(tdnv): add compute_tdnv_mmlu for MMLU-Pro categories` — tdnv.py, test_tdnv.py
- **7**: `feat(tdnv): add token selection helpers (last-n, top-k)` — tdnv.py, test_tdnv.py
- **8**: `feat(tdnv): add stability trend plotting functions` — tdnv.py
- **9-13**: `feat(scripts): add TDNV stability analysis scripts` — scripts/tdnv/*.sh

---

## Success Criteria

### Verification Commands
```bash
# Type check
uv run mypy src/steering_geometry/tdnv.py

# Lint
uv run ruff check src/steering_geometry/tdnv.py tests/unit/test_tdnv.py

# Tests
uv run pytest tests/unit/test_tdnv.py -v

# Script example
./scripts/tdnv/run_dataset_size_stability.sh --concept polite --model "Qwen/Qwen3-1.7B"
```

### Final Checklist
- [ ] All new functions have tests
- [ ] All tests pass
- [ ] Type checking clean
- [ ] Linting clean
- [ ] All 5 scripts produce output
- [ ] JSON output valid
- [ ] Plots generated
