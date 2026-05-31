# Normalize Discriminative Score with Total-Variance Denominator

## TL;DR

> **Quick Summary**: Add normalization denominator to discriminative score formula across all 3 implementations, changing from raw `||h - μ_other||² - ||h - μ_own||²` to `(||h - μ_other||² - ||h - μ_own||²) / (||h - μ_own||² + ||h - μ_other||² + ε)`. This produces bounded scores in (-1, 1] and favors tokens with high *relative* discrimination.
> 
> **Deliverables**:
> - Shared `DISCRIMINATIVE_EPS` constant in `utils.py`
> - Updated formula in 3 source files
> - Updated docstrings in 3 source files
> - Updated/new tests in 3 test files
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Tasks 2,3,4 (parallel) → Task 5

---

## Context

### Original Request
Add a normalization denominator to the discriminative score calculation:
```
s_i = (||h_i - μ_other||² - ||h_i - μ_own||²) / (||h_i - μ_own||² + ||h_i - μ_other||² + ε)
```

### Interview Summary
**Key Discussions**:
- Formula: Total-variance normalization with epsilon for numerical stability
- User confirmed the formula explicitly: total variance in denominator (sum of both squared distances + epsilon)
- Multi-class generalization: denominator = `||h_i - μ_own||² + Σ_{c≠own} ||h_i - μ_c||² + ε`

**Research Findings**:
- 3 independent implementations of the same formula exist in the codebase
- `tdnv.py` already has a module-level `EPS = 1e-8` constant
- Scores are ONLY used for ranking/selection (top-k), never as input to downstream numerical computation
- Normalization **changes ranking** — tokens with high relative discrimination are favored over high absolute discrimination. This is intentional.
- `token_analysis.py` casts to float32 before computation; `extract.py` and `tdnv.py` do not. With division entering the formula, float32 cast should be standardized.

### Metis Review
**Identified Gaps** (addressed):
- Epsilon constant should be shared, not duplicated → Task 1 creates `DISCRIMINATIVE_EPS` in `utils.py`
- Float16 overflow risk in `extract.py` and `tdnv.py` → Added `.float()` cast requirement
- `test_tdnv.py:426-430` has comment encoding old formula value `score = 8` → Must update comment
- Multi-class denominator generalization specified explicitly
- Scope strictly bounded: no tech debt fixes, no refactoring beyond the formula change

---

## Work Objectives

### Core Objective
Replace the unnormalized discriminative score formula with the total-variance normalized version across all 3 implementations, synchronize docstrings and tests.

### Concrete Deliverables
- `src/steering_geometry/utils.py` — new `DISCRIMINATIVE_EPS = 1e-8` constant
- `src/steering_geometry/token_analysis.py` — formula + docstring update
- `src/steering_geometry/extract.py` — formula + docstring + float32 cast
- `src/steering_geometry/tdnv.py` — formula + docstring + float32 cast (reuse existing `EPS`)
- `tests/test_token_analysis.py` — updated + new range test
- `tests/unit/test_aggregators.py` — updated + new test
- `tests/unit/test_tdnv.py` — updated comment + new range test

### Definition of Done
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → formatted
- [ ] `uv run mypy src/` → 0 errors
- [ ] `uv run pytest` → all tests pass (including new range tests)

### Must Have
- All 3 source files use the identical normalized formula
- Shared epsilon constant (no magic numbers in 3 places)
- Float32 cast in all 3 implementations for numerical stability
- Docstrings updated with new formula in all 3 files
- At least 1 new test per implementation asserting score is in (-1, 1] range
- Existing `test_three_class_prefers_central_token` must still pass (ranking preserved for that data)

### Must NOT Have (Guardrails)
- Do NOT add a `normalize` flag or `epsilon` parameter to function signatures
- Do NOT change `TokenRecord.score` type (stays `float`)
- Do NOT touch shell scripts, configs, `SPEC.md`, or any file outside the 3 source + 3 test scope
- Do NOT fix `Any` types, replace `print()` with logging, or address other tech debt
- Do NOT refactor `_select_token_activations` duplication
- Do NOT restructure function bodies — only change the scoring formula and add `.float()` cast

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (tests-after — update existing + add new)
- **Framework**: pytest

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - shared constant):
└── Task 1: Add DISCRIMINATIVE_EPS to utils.py [quick]

Wave 2 (After Wave 1 - all 3 implementations + tests in parallel):
├── Task 2: Update token_analysis.py + test [quick]
├── Task 3: Update extract.py + test [quick]
└── Task 4: Update tdnv.py + test [quick]

Wave 3 (After Wave 2 - verification):
└── Task 5: Full quality gate verification [quick]

Critical Path: Task 1 → Tasks 2,3,4 → Task 5
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 3 (Wave 2)
```

### Dependency Matrix

| Task | Blocked By | Blocks |
|------|-----------|--------|
| 1 | - | 2, 3, 4 |
| 2 | 1 | 5 |
| 3 | 1 | 5 |
| 4 | 1 | 5 |
| 5 | 2, 3, 4 | - |

### Agent Dispatch Summary

- **Wave 1**: 1 task — T1 → `quick`
- **Wave 2**: 3 tasks — T2 → `quick`, T3 → `quick`, T4 → `quick`
- **Wave 3**: 1 task — T5 → `quick`

---

## TODOs

- [x] 1. Add shared `DISCRIMINATIVE_EPS` constant to `utils.py`

  **What to do**:
  - Add `DISCRIMINATIVE_EPS: float = 1e-8` to `src/steering_geometry/utils.py`
  - This constant is the epsilon used in the normalization denominator to prevent division by zero
  - Export it from `utils.py` so it can be imported by all 3 implementation files
  - Ensure it appears in `utils.py`'s `__all__` if one exists, or add it near other constants

  **Must NOT do**:
  - Do NOT remove the existing `EPS = 1e-8` in `tdnv.py` yet (that happens in Task 4)
  - Do NOT modify any other file in this task

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single constant addition, trivial change
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: Tasks 2, 3, 4
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/steering_geometry/tdnv.py:38` — existing `EPS = 1e-8` constant pattern to follow
  - `src/steering_geometry/utils.py` — file to add the constant to; check existing exports at bottom

  **WHY Each Reference Matters**:
  - `tdnv.py:38`: Shows the established epsilon naming and value convention in this project
  - `utils.py`: The shared utilities module where cross-cutting constants should live

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Constant exists and is importable
    Tool: Bash
    Preconditions: utils.py has been modified
    Steps:
      1. Run: uv run python -c "from steering_geometry.utils import DISCRIMINATIVE_EPS; assert DISCRIMINATIVE_EPS == 1e-8; print(f'PASS: DISCRIMINATIVE_EPS = {DISCRIMINATIVE_EPS}')"
    Expected Result: Output contains "PASS: DISCRIMINATIVE_EPS = 1e-08"
    Failure Indicators: ImportError, AssertionError, or wrong value
    Evidence: .sisyphus/evidence/task-1-constant-importable.txt
  ```

  **Commit**: YES
  - Message: `feat(scoring): add DISCRIMINATIVE_EPS constant to utils.py`
  - Files: `src/steering_geometry/utils.py`
  - Pre-commit: `uv run ruff check src/steering_geometry/utils.py && uv run mypy src/steering_geometry/utils.py`

- [x] 2. Update `token_analysis.py` — formula + docstring + new test

  **What to do**:
  - In `compute_discriminative_scores()` (line 149-197):
    - Import `DISCRIMINATIVE_EPS` from `steering_geometry.utils`
    - Update docstring formula from `s_i = ||h_i - μ_other||² - ||h_i - μ_own||²` to `s_i = (||h_i - μ_other||² - ||h_i - μ_own||²) / (||h_i - μ_own||² + ||h_i - μ_other||² + ε)`
    - The `.float()` cast already exists (line 175-176), no change needed there
    - **Change the scoring computation** (lines 181-186):
      ```python
      # BEFORE:
      pos_scores = ((pos_activations - neg_center) ** 2).sum(dim=1) - (
          (pos_activations - pos_center) ** 2
      ).sum(dim=1)
      neg_scores = ((neg_activations - pos_center) ** 2).sum(dim=1) - (
          (neg_activations - neg_center) ** 2
      ).sum(dim=1)

      # AFTER:
      pos_dist_other = ((pos_activations - neg_center) ** 2).sum(dim=1)
      pos_dist_own = ((pos_activations - pos_center) ** 2).sum(dim=1)
      pos_scores = (pos_dist_other - pos_dist_own) / (pos_dist_own + pos_dist_other + DISCRIMINATIVE_EPS)

      neg_dist_other = ((neg_activations - pos_center) ** 2).sum(dim=1)
      neg_dist_own = ((neg_activations - neg_center) ** 2).sum(dim=1)
      neg_scores = (neg_dist_other - neg_dist_own) / (neg_dist_own + neg_dist_other + DISCRIMINATIVE_EPS)
      ```
  - In `tests/test_token_analysis.py` — class `TestComputeDiscriminativeScores`:
    - Add new test `test_normalized_score_range`: assert all scores are `> -1.0 - 1e-6` and `<= 1.0 + 1e-6`
    - Add new test `test_exact_normalized_value`: construct pos=[2,0], neg=[0,2] single-token records, assert `pos_s[0].score ≈ 1.0` (within 1e-5)
    - Existing tests `test_scores_are_finite` and `test_scores_are_sorted_descending` should still pass (verify, don't modify unless they fail)

  **Must NOT do**:
  - Do NOT add `normalize` parameter or `epsilon` parameter
  - Do NOT change `TokenRecord.score` type
  - Do NOT touch any other function in `token_analysis.py`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Focused formula change + 2 new tests in well-understood code
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/steering_geometry/token_analysis.py:149-197` — the exact function to modify, full implementation
  - `src/steering_geometry/token_analysis.py:7` — existing imports section where `DISCRIMINATIVE_EPS` import should be added

  **API/Type References**:
  - `src/steering_geometry/types.py:406-428` — `TokenRecord` dataclass with `score: float = 0.0` field
  - `src/steering_geometry/utils.py` — `DISCRIMINATIVE_EPS` constant (added in Task 1)

  **Test References**:
  - `tests/test_token_analysis.py:183-243` — `TestComputeDiscriminativeScores` class with existing tests
  - `tests/test_token_analysis.py:204-216` — `test_scores_are_finite` pattern to follow for new range test

  **WHY Each Reference Matters**:
  - `token_analysis.py:149-197`: Contains the exact lines to change; the scoring is at lines 181-186
  - `types.py:428`: Confirms `score` field is `float` — no type change needed
  - `test_token_analysis.py:183-243`: Shows the test class structure and helper `_create_mock_records`

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Normalized score is bounded in (-1, 1]
    Tool: Bash
    Preconditions: token_analysis.py and test file updated
    Steps:
      1. Run: uv run pytest tests/test_token_analysis.py::TestComputeDiscriminativeScores -v
    Expected Result: All tests pass including test_normalized_score_range and test_exact_normalized_value
    Failure Indicators: Any test FAIL or ERROR
    Evidence: .sisyphus/evidence/task-2-token-analysis-tests.txt

  Scenario: Exact formula verification
    Tool: Bash
    Preconditions: token_analysis.py updated
    Steps:
      1. Run: uv run python -c "
         from steering_geometry.token_analysis import compute_discriminative_scores
         from steering_geometry.types import TokenRecord
         import torch
         pos = [TokenRecord(0, 'a', torch.tensor([2.0, 0.0]), 0, 0, 'positive')]
         neg = [TokenRecord(0, 'b', torch.tensor([0.0, 2.0]), 0, 0, 'negative')]
         pos_s, neg_s = compute_discriminative_scores(pos, neg)
         assert abs(pos_s[0].score - 1.0) < 1e-5, f'Expected ≈1.0, got {pos_s[0].score}'
         print(f'PASS: pos score = {pos_s[0].score:.8f}')
         "
    Expected Result: "PASS: pos score = 0.99999999"
    Failure Indicators: AssertionError or score far from 1.0
    Evidence: .sisyphus/evidence/task-2-exact-formula.txt
  ```

  **Commit**: NO (groups with Task 3, 4 into Commit 2)

- [x] 3. Update `extract.py` — formula + docstring + float32 cast + new test

  **What to do**:
  - In `discriminative_token_aggregator()` (line 126-161):
    - Import `DISCRIMINATIVE_EPS` from `steering_geometry.utils`
    - Update docstring formula from `s_i = ||h_i - μ_other||² - ||h_i - μ_same||²` to `s_i = (||h_i - μ_other||² - ||h_i - μ_same||²) / (||h_i - μ_same||² + ||h_i - μ_other||² + ε)`
    - Add `.float()` cast before computation (this file currently does NOT have it, unlike token_analysis.py):
      ```python
      # Add after empty-tensor check:
      pos = pos.float()
      neg = neg.float()
      ```
    - **Change the scoring computation** (lines 146-147):
      ```python
      # BEFORE:
      pos_scores = ((pos - neg_center) ** 2).sum(dim=1) - ((pos - pos_center) ** 2).sum(dim=1)
      neg_scores = ((neg - pos_center) ** 2).sum(dim=1) - ((neg - neg_center) ** 2).sum(dim=1)

      # AFTER:
      pos_dist_other = ((pos - neg_center) ** 2).sum(dim=1)
      pos_dist_own = ((pos - pos_center) ** 2).sum(dim=1)
      pos_scores = (pos_dist_other - pos_dist_own) / (pos_dist_own + pos_dist_other + DISCRIMINATIVE_EPS)

      neg_dist_other = ((neg - pos_center) ** 2).sum(dim=1)
      neg_dist_own = ((neg - neg_center) ** 2).sum(dim=1)
      neg_scores = (neg_dist_other - neg_dist_own) / (neg_dist_own + neg_dist_other + DISCRIMINATIVE_EPS)
      ```
  - In `tests/unit/test_aggregators.py` — class `TestDiscriminativeTokenAggregator`:
    - Add new test `test_normalized_scores_bounded`: verify that with random inputs, the selected tokens' implicit scores would be bounded. Since scores are internal (not returned), verify by checking the output shape is correct and the result is finite.
    - Add new test `test_well_separated_classes`: create pos centered at [10,0,...], neg centered at [-10,0,...] in float16, verify output is finite and shape is correct (tests the float32 cast)
    - Existing tests `test_basic`, `test_clamp_topk`, `test_selection`, `test_empty_tensor_raises`, `test_custom_topk` should still pass

  **Must NOT do**:
  - Do NOT change function signature (no `epsilon` param)
  - Do NOT touch `_resolve_aggregator` or any other function
  - Do NOT change the top-k selection or prototype computation logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Focused formula change + float32 cast + 2 new tests
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 4)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/steering_geometry/token_analysis.py:175-186` — pattern to follow for float32 cast + split distance computation
  - `src/steering_geometry/extract.py:126-161` — the exact function to modify

  **API/Type References**:
  - `src/steering_geometry/utils.py` — `DISCRIMINATIVE_EPS` constant (added in Task 1)

  **Test References**:
  - `tests/unit/test_aggregators.py:58-93` — `TestDiscriminativeTokenAggregator` existing tests

  **WHY Each Reference Matters**:
  - `token_analysis.py:175-186`: Reference pattern for how to split the distance computation and apply normalization
  - `extract.py:126-161`: The target function — note lines 146-147 are the scoring lines to change
  - `test_aggregators.py:58-93`: Shows existing test patterns; new tests should follow the same style

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All aggregator tests pass
    Tool: Bash
    Preconditions: extract.py and test file updated
    Steps:
      1. Run: uv run pytest tests/unit/test_aggregators.py -v
    Expected Result: All tests pass including test_normalized_scores_bounded and test_well_separated_classes
    Failure Indicators: Any test FAIL or ERROR
    Evidence: .sisyphus/evidence/task-3-aggregator-tests.txt

  Scenario: Float16 input handled correctly (float32 cast)
    Tool: Bash
    Preconditions: extract.py updated
    Steps:
      1. Run: uv run python -c "
         import torch
         from steering_geometry.extract import discriminative_token_aggregator
         pos = torch.randn(10, 64, dtype=torch.float16) * 100
         neg = torch.randn(10, 64, dtype=torch.float16) * 100
         result = discriminative_token_aggregator(pos, neg, top_k=5)
         assert result.dtype == torch.float32, f'Expected float32, got {result.dtype}'
         assert torch.isfinite(result).all(), 'Result contains non-finite values'
         print(f'PASS: dtype={result.dtype}, finite={torch.isfinite(result).all().item()}')
         "
    Expected Result: "PASS: dtype=torch.float32, finite=True"
    Failure Indicators: RuntimeError, non-finite values, wrong dtype
    Evidence: .sisyphus/evidence/task-3-float16-cast.txt
  ```

  **Commit**: NO (groups with Tasks 2, 4 into Commit 2)

- [x] 4. Update `tdnv.py` — formula + docstring + float32 cast + comment + new test

  **What to do**:
  - In `select_top_k_discriminative()` (line 59-112):
    - Replace the module-level `EPS = 1e-8` (line 38) with an import from utils: `from steering_geometry.utils import DISCRIMINATIVE_EPS`
    - Update the old `EPS` references in other parts of `tdnv.py` (e.g., `compute_tdnv()` function) to use `DISCRIMINATIVE_EPS` instead
    - Update docstring (line 66-68) from `s_i = Σ_{c ≠ own} ||h_i - μ_c||² - ||h_i - μ_same||²` to `s_i = (Σ_{c ≠ own} ||h_i - μ_c||² - ||h_i - μ_same||²) / (||h_i - μ_same||² + Σ_{c ≠ own} ||h_i - μ_c||² + ε)`
    - Add `.float()` cast on input activations at the start of the function:
      ```python
      activations = activations.float()
      ```
    - **Change the scoring computation** (lines 95-105):
      ```python
      # BEFORE:
      dist_to_others = torch.zeros(class_activations.shape[0], dtype=class_activations.dtype)
      for other_label in unique_labels:
          if other_label == label:
              continue
          other_centroid = centroids[other_label]
          dist_to_others = dist_to_others + ((class_activations - other_centroid) ** 2).sum(dim=1)

      dist_to_own = ((class_activations - own_centroid) ** 2).sum(dim=1)
      scores = dist_to_others - dist_to_own

      # AFTER:
      dist_to_others = torch.zeros(class_activations.shape[0], dtype=torch.float32)
      for other_label in unique_labels:
          if other_label == label:
              continue
          other_centroid = centroids[other_label]
          dist_to_others = dist_to_others + ((class_activations - other_centroid) ** 2).sum(dim=1)

      dist_to_own = ((class_activations - own_centroid) ** 2).sum(dim=1)
      scores = (dist_to_others - dist_to_own) / (dist_to_own + dist_to_others + DISCRIMINATIVE_EPS)
      ```
  - In `tests/unit/test_tdnv.py`:
    - Update comment in `test_scoring_formula` (lines 426-430): change `# score = 8 - 0 = 8` to `# score = (8 - 0) / (0 + 8 + ε) ≈ 1.0`
    - Add new test `test_normalized_score_range` in `TestSelectTopKDiscriminative`: verify that for multi-class data, all implicit scores would be bounded. Since scores are internal, verify by checking selected tokens match expected pattern for known bounded data.
    - Add new test `test_binary_exact_normalized_score`: verify exact score for binary case:
      - class_0 = [[2,0], [2,0]], class_1 = [[0,2], [0,2]]
      - For class_0 token [2,0]: dist_other = ||[2,-2]||² = 8, dist_own = 0
      - Normalized score = (8 - 0) / (0 + 8 + ε) ≈ 1.0
      - Both tokens should be selected (k=2), verify shape

  **Must NOT do**:
  - Do NOT change function signature
  - Do NOT touch `compute_tdnv()`, `compute_tdnv_for_concept()`, or `compute_tdnv_for_mmlu()` logic (only update EPS → DISCRIMINATIVE_EPS references)
  - Do NOT change the top-k selection or concatenation logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Focused formula change + float32 cast + EPS rename + test updates
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/steering_geometry/token_analysis.py:175-186` — reference pattern for split distance + normalization
  - `src/steering_geometry/tdnv.py:38` — existing `EPS = 1e-8` to be replaced with import
  - `src/steering_geometry/tdnv.py:59-112` — the exact function to modify

  **API/Type References**:
  - `src/steering_geometry/utils.py` — `DISCRIMINATIVE_EPS` constant (added in Task 1)

  **Test References**:
  - `tests/unit/test_tdnv.py:385-479` — `TestSelectTopKDiscriminative` existing tests
  - `tests/unit/test_tdnv.py:410-434` — `test_scoring_formula` with comment to update

  **WHY Each Reference Matters**:
  - `tdnv.py:38`: The old `EPS` constant that must be replaced with the shared import
  - `tdnv.py:59-112`: The multi-class scoring function; lines 95-105 contain the scoring logic
  - `test_tdnv.py:410-434`: Contains hardcoded comment `score = 8` that becomes inaccurate with normalization

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All TDNV tests pass
    Tool: Bash
    Preconditions: tdnv.py and test file updated
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py -v
    Expected Result: All tests pass including test_binary_exact_normalized_score
    Failure Indicators: Any test FAIL or ERROR
    Evidence: .sisyphus/evidence/task-4-tdnv-tests.txt

  Scenario: EPS constant properly replaced
    Tool: Bash
    Preconditions: tdnv.py updated
    Steps:
      1. Run: grep -n "^EPS = " src/steering_geometry/tdnv.py
      2. Verify: no match (old module-level EPS removed)
      3. Run: grep -n "DISCRIMINATIVE_EPS" src/steering_geometry/tdnv.py
      4. Verify: at least 3 matches (import + scoring + compute_tdnv usage)
    Expected Result: No standalone `EPS = ` definition; DISCRIMINATIVE_EPS used throughout
    Failure Indicators: Old `EPS = 1e-8` still present, or DISCRIMINATIVE_EPS not imported
    Evidence: .sisyphus/evidence/task-4-eps-replacement.txt
  ```

  **Commit**: NO (groups with Tasks 2, 3 into Commit 2)

- [x] 5. Full quality gate verification

  **What to do**:
  - Run the complete quality gate:
    1. `uv run ruff check src/ tests/`
    2. `uv run ruff format --check src/ tests/`
    3. `uv run mypy src/`
    4. `uv run pytest`
  - Run the formula verification command from Success Criteria
  - Verify no files outside scope were touched: `git diff --name-only` should only show the 7 allowed files
  - If any check fails, fix the issue and re-run

  **Must NOT do**:
  - Do NOT skip any quality gate step
  - Do NOT force-push or modify git history

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Running verification commands and fixing any issues
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (solo)
  - **Blocks**: Final commit
  - **Blocked By**: Tasks 2, 3, 4

  **References**:

  **Pattern References**:
  - `AGENTS.md` section "Definition of Done" — the 5 required checks

  **WHY Each Reference Matters**:
  - `AGENTS.md`: Defines the project's quality gate that must pass before commit

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Complete quality gate passes
    Tool: Bash
    Preconditions: All source and test files updated
    Steps:
      1. Run: uv run ruff check src/ tests/ && echo "RUFF: PASS"
      2. Run: uv run ruff format --check src/ tests/ && echo "FORMAT: PASS"
      3. Run: uv run mypy src/ && echo "MYPY: PASS"
      4. Run: uv run pytest && echo "PYTEST: PASS"
    Expected Result: All 4 commands print PASS
    Failure Indicators: Any command exits non-zero
    Evidence: .sisyphus/evidence/task-5-quality-gate.txt

  Scenario: No scope creep — only allowed files changed
    Tool: Bash
    Preconditions: All changes made
    Steps:
      1. Run: git diff --name-only
      2. Verify output only contains these files:
         - src/steering_geometry/utils.py
         - src/steering_geometry/token_analysis.py
         - src/steering_geometry/extract.py
         - src/steering_geometry/tdnv.py
         - tests/test_token_analysis.py
         - tests/unit/test_aggregators.py
         - tests/unit/test_tdnv.py
    Expected Result: Exactly 7 files listed, no others
    Failure Indicators: Any file outside the allowed set
    Evidence: .sisyphus/evidence/task-5-scope-check.txt
  ```

  **Commit**: YES (creates Commit 2 if not already committed)
  - Message: `feat(scoring): normalize discriminative score with total-variance denominator`
  - Files: `src/steering_geometry/token_analysis.py`, `src/steering_geometry/extract.py`, `src/steering_geometry/tdnv.py`, `tests/test_token_analysis.py`, `tests/unit/test_aggregators.py`, `tests/unit/test_tdnv.py`
  - Pre-commit: `uv run pytest && uv run ruff check src/ tests/ && uv run mypy src/`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. Verify: (1) All 3 source files have normalized formula, (2) `DISCRIMINATIVE_EPS` in `utils.py` and imported everywhere, (3) All 3 docstrings updated, (4) All 3 test files have new range tests, (5) No files outside scope were touched. Output: `Must Have [N/N] | Must NOT Have [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Check all changed files for: magic numbers, missing float32 cast, stale comments, inconsistent epsilon usage. Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

---

## Commit Strategy

- **Commit 1**: `feat(scoring): add DISCRIMINATIVE_EPS constant to utils.py` — `src/steering_geometry/utils.py`
  - Pre-commit: `uv run ruff check src/steering_geometry/utils.py && uv run mypy src/steering_geometry/utils.py`

- **Commit 2** (after all tasks): `feat(scoring): normalize discriminative score with total-variance denominator` — `src/steering_geometry/token_analysis.py`, `src/steering_geometry/extract.py`, `src/steering_geometry/tdnv.py`, `tests/test_token_analysis.py`, `tests/unit/test_aggregators.py`, `tests/unit/test_tdnv.py`
  - Pre-commit: `uv run pytest && uv run ruff check src/ tests/ && uv run mypy src/`

---

## Success Criteria

### Verification Commands
```bash
uv run ruff check src/ tests/                          # Expected: 0 violations
uv run ruff format --check src/ tests/                  # Expected: already formatted
uv run mypy src/                                        # Expected: Success, 0 errors
uv run pytest                                           # Expected: all tests pass
uv run pytest -k "normalized_score_range" -v            # Expected: 2+ new tests pass
```

### Formula Verification
```bash
uv run python -c "
from steering_geometry.token_analysis import compute_discriminative_scores
from steering_geometry.types import TokenRecord
import torch

pos = [TokenRecord(0, 'a', torch.tensor([2.0, 0.0]), 0, 0, 'positive')]
neg = [TokenRecord(0, 'b', torch.tensor([0.0, 2.0]), 0, 0, 'negative')]
pos_s, neg_s = compute_discriminative_scores(pos, neg)
assert abs(pos_s[0].score - 1.0) < 1e-5, f'Expected ≈1.0, got {pos_s[0].score}'
print(f'PASS: pos score = {pos_s[0].score:.8f}')
"
# Expected: PASS: pos score = 0.99999999
```

### Final Checklist
- [ ] All "Must Have" present (normalized formula in 3 files, shared EPS, float32 cast, updated docstrings, range tests)
- [ ] All "Must NOT Have" absent (no new params, no scope creep, no tech debt fixes)
- [ ] All tests pass
- [ ] Formula verified with known-input test
