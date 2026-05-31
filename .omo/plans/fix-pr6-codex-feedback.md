# Fix PR6 Codex Feedback — Compatibility Shims

## TL;DR

> **Quick Summary**: Add backward-compatible import paths for `evaluation` and `vector_analysis` modules that were merged into `apply_steering.py` and `stability_comparison.py` in PR6.
> 
> **Deliverables**:
> - `src/steering_geometry/evaluation.py` — Shim re-exporting from apply_steering
> - `src/steering_geometry/vector_analysis.py` — Shim re-exporting from stability_comparison
> - Updated `__init__.py` to include shim modules
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: YES - 2 independent shim files
> **Critical Path**: None (parallel tasks)

---

## Context

### Original Request
Codex (chatgpt-codex-connector) reviewed PR6 and identified **2 breaking API changes**:

1. **`steering_geometry.evaluation`** — Moving `JudgeEvaluator`, `MMLUEvaluator`, `generate_html_report` into `apply_steering.py` without a wrapper breaks old import path.
2. **`steering_geometry.vector_analysis`** — Dropping the module entirely means callers using old imports get `ModuleNotFoundError`.

### Codex Feedback (Exact)

| Issue | Path | Problem | Severity |
|-------|------|---------|----------|
| 1 | `apply_steering.py:1-10` | Moving `JudgeEvaluator`, `MMLUEvaluator`, `generate_html_report` without leaving `steering_geometry.evaluation` wrapper | P2 |
| 2 | `stability_comparison.py:1-8` | Dropping `steering_geometry.vector_analysis` breaks old imports | P2 |

### Solution
Create **thin compatibility shim modules** that re-export from the new locations:
- `evaluation.py` → re-exports from `apply_steering.py`
- `vector_analysis.py` → re-exports from `stability_comparison.py`

---

## Work Objectives

### Core Objective
Restore backward compatibility for old import paths without reverting the refactoring.

### Concrete Deliverables
- `src/steering_geometry/evaluation.py` — Compatibility shim
- `src/steering_geometry/vector_analysis.py` — Compatibility shim
- Updated `src/steering_geometry/__init__.py` (if needed)

### Definition of Done
- [x] `from steering_geometry.evaluation import JudgeEvaluator` works
- [x] `from steering_geometry.vector_analysis import plot_heatmap` works
- [x] All existing tests pass: `uv run pytest`
- [x] Type check passes: `uv run mypy src/`
- [x] Lint passes: `uv run ruff check src/ tests/`

### Must Have
- Thin shim modules (re-exports only, no new logic)
- Clear deprecation comments pointing to new locations
- Full backward compatibility

### Must NOT Have (Guardrails)
- Do NOT copy code from source modules
- Do NOT add new functionality
- Do NOT break existing imports from new locations

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: Tests-after (add simple import tests)
- **Framework**: pytest

### QA Policy
Agent-executed QA scenarios for each deliverable.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — 2 parallel shims):
├── Task 1: Create evaluation.py shim [quick]
└── Task 2: Create vector_analysis.py shim [quick]

Wave FINAL (After Wave 1):
└── Task F1: Verify all imports work + run test suite [quick]
```

---

## TODOs

- [x] 1. Create `src/steering_geometry/evaluation.py` compatibility shim

  **What to do**:
  - Create thin shim module at `src/steering_geometry/evaluation.py`
  - Re-export from `apply_steering.py`:
    - `JudgeEvaluator`
    - `MMLUEvaluator`
    - `generate_html_report`
    - `EvaluationMetadata`
    - `EvaluationResult`
    - `JudgeScore`
    - `MMLUConfig`
    - `MMLUPrediction`
    - `MMLUQuestion`
    - `MMLUResult`
  - Add docstring explaining deprecation and pointing to new import path
  - Define `__all__` for clean exports

  **Must NOT do**:
  - Copy implementation code
  - Add new functionality

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/apply_steering.py:181` - `JudgeEvaluator` class definition
  - `src/steering_geometry/apply_steering.py:321` - `MMLUEvaluator` class definition
  - `src/steering_geometry/apply_steering.py:460` - `generate_html_report` function
  - `src/steering_geometry/apply_steering.py:817-819` - `__all__` exports to mirror

  **Acceptance Criteria**:
  - [ ] File created at `src/steering_geometry/evaluation.py`
  - [ ] `from steering_geometry.evaluation import JudgeEvaluator` succeeds
  - [ ] `from steering_geometry.evaluation import MMLUEvaluator` succeeds
  - [ ] `from steering_geometry.evaluation import generate_html_report` succeeds

  **QA Scenarios**:
  ```
  Scenario: Import from evaluation shim works
    Tool: Bash (python -c)
    Steps:
      1. Run: uv run python -c "from steering_geometry.evaluation import JudgeEvaluator, MMLUEvaluator, generate_html_report; print('OK')"
    Expected Result: Output contains "OK"
    Evidence: .omo/evidence/task-1-import-evaluation.txt
  ```

  **Commit**: YES
  - Message: `fix(api): add evaluation.py compatibility shim for PR6`
  - Files: `src/steering_geometry/evaluation.py`

---

- [x] 2. Create `src/steering_geometry/vector_analysis.py` compatibility shim

  **What to do**:
  - Create thin shim module at `src/steering_geometry/vector_analysis.py`
  - Re-export from `stability_comparison.py`:
    - `run_diff_means_experiment`
    - `run_discriminative_experiment`
    - `plot_heatmap`
    - `load_vector`
    - `compute_cosine_similarity_matrix`
  - Add docstring explaining deprecation and pointing to new import path
  - Define `__all__` for clean exports

  **Must NOT do**:
  - Copy implementation code
  - Add new functionality

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/stability_comparison.py:42` - `compute_cosine_similarity_matrix`
  - `src/steering_geometry/stability_comparison.py:65` - `plot_heatmap`
  - `src/steering_geometry/stability_comparison.py:132` - `load_vector`
  - `src/steering_geometry/stability_comparison.py:179` - `run_diff_means_experiment`
  - `src/steering_geometry/stability_comparison.py:312` - `run_discriminative_experiment`
  - `src/steering_geometry/stability_comparison.py:677-681` - `__all__` exports to mirror

  **Acceptance Criteria**:
  - [ ] File created at `src/steering_geometry/vector_analysis.py`
  - [ ] `from steering_geometry.vector_analysis import run_diff_means_experiment` succeeds
  - [ ] `from steering_geometry.vector_analysis import plot_heatmap` succeeds
  - [ ] `from steering_geometry.vector_analysis import load_vector` succeeds

  **QA Scenarios**:
  ```
  Scenario: Import from vector_analysis shim works
    Tool: Bash (python -c)
    Steps:
      1. Run: uv run python -c "from steering_geometry.vector_analysis import run_diff_means_experiment, plot_heatmap, load_vector; print('OK')"
    Expected Result: Output contains "OK"
    Evidence: .omo/evidence/task-2-import-vector-analysis.txt
  ```

  **Commit**: YES
  - Message: `fix(api): add vector_analysis.py compatibility shim for PR6`
  - Files: `src/steering_geometry/vector_analysis.py`

---

## Final Verification Wave

- [x] F1. **Full Test Suite + Type Check**

  **What to do**:
  1. Run `uv run pytest` — all tests pass
  2. Run `uv run mypy src/` — no errors
  3. Run `uv run ruff check src/ tests/` — no violations
  4. Verify both old and new import paths work

  **QA Scenarios**:
  ```
  Scenario: All quality checks pass
    Tool: Bash
    Steps:
      1. Run: uv run pytest
      2. Run: uv run mypy src/
      3. Run: uv run ruff check src/ tests/
    Expected Result: All commands exit with code 0
    Evidence: .omo/evidence/final-qa-checks.txt

  Scenario: Old and new imports both work
    Tool: Bash (python -c)
    Steps:
      1. Run: uv run python -c "from steering_geometry.evaluation import JudgeEvaluator; from steering_geometry.apply_steering import JudgeEvaluator as JS2; print('OK')"
      2. Run: uv run python -c "from steering_geometry.vector_analysis import plot_heatmap; from steering_geometry.stability_comparison import plot_heatmap as ph2; print('OK')"
    Expected Result: Both output "OK"
    Evidence: .omo/evidence/final-import-compat.txt
  ```

  **Commit**: YES (squash with previous commits)
  - Message: `fix(api): restore evaluation and vector_analysis import paths (PR6 feedback)`

---

## Commit Strategy

- **Commit 1-2**: Individual shim files (if separate commits preferred)
- **Final**: Squash into single commit if cleaner:
  ```
  fix(api): restore evaluation and vector_analysis import paths (PR6 feedback)
  
  - Add evaluation.py shim re-exporting from apply_steering.py
  - Add vector_analysis.py shim re-exporting from stability_comparison.py
  - Addresses Codex P2 feedback on breaking API changes in PR6
  
  Fixes: #6 (Codex review comments)
  ```

---

## Success Criteria

### Verification Commands
```bash
# Old imports work
uv run python -c "from steering_geometry.evaluation import JudgeEvaluator; print('OK')"
uv run python -c "from steering_geometry.vector_analysis import plot_heatmap; print('OK')"

# New imports still work
uv run python -c "from steering_geometry.apply_steering import JudgeEvaluator; print('OK')"
uv run python -c "from steering_geometry.stability_comparison import plot_heatmap; print('OK')"

# Quality checks
uv run pytest
uv run mypy src/
uv run ruff check src/ tests/
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] All tests pass
- [x] Type check passes
- [x] Lint passes
