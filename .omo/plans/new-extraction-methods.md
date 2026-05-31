# New Extraction Methods: Weighted Mean Direction & Discriminative Tokens

## TL;DR

> **Quick Summary**: Add two new steering vector extraction methods (Weighted Mean Direction and Discriminative Tokens) to compare with existing difference-in-means. Only extraction logic changes - steering and evaluation pipelines remain unchanged.
>
> **Deliverables**:
> - `weighted_mean_aggregator()` function in `extract.py`
> - `discriminative_token_aggregator()` function in `extract.py`
> - `top_k` field in `ExtractionConfig`
> - `--top-k` CLI argument
> - Unit tests in `tests/unit/test_aggregators.py`
>
> **Estimated Effort**: Short
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Config → Aggregators → CLI → Verification

---

## Context

### Original Request
Add two new vector extraction methods with the same steering/evaluation pipeline, only changing the extraction logic:

**Method 1: Weighted Mean Direction**
- Uses soft distance-based weights instead of hard selection
- Tokens closer to class center receive larger weights

**Method 2: Discriminative Tokens**
- Scores tokens by class-internal distance vs cross-class distance
- Selects top-k tokens that are central to their class AND well-separated from opposite class

### Interview Summary
**Key Discussions**:
- Top-k selection: Fixed count (e.g., top 100)
- Configurability: `--top-k` CLI flag with default=100
- Test strategy: TDD (Test-Driven Development)

**Research Findings**:
- Existing aggregator pattern: `Aggregator = Callable[[Tensor, Tensor], Tensor]`
- Discriminative needs extra `top_k` parameter → add to `ExtractionConfig`
- Registration in `_resolve_aggregator()` dict
- CLI updates in `_build_parser()` at line 539

### Metis Review
**Identified Gaps** (addressed):
- **Aggregator pattern mismatch**: Use `ExtractionConfig.top_k` field, not function parameter
- **Edge cases**: Handle empty tensors, `top_k > num_tokens`, numerical stability
- **Missing tests**: Create `tests/unit/test_aggregators.py` for aggregator unit tests

---

## Work Objectives

### Core Objective
Add two mathematically rigorous vector extraction methods to enable comparison experiments with the existing difference-in-means approach.

### Concrete Deliverables
- `src/steering_geometry/extract.py`: Two new aggregator functions + registration
- `src/steering_geometry/config.py`: `top_k: int | None = None` field
- `tests/unit/test_aggregators.py`: Unit tests for both methods

### Definition of Done
- [ ] `uv run pytest tests/unit/test_aggregators.py -v` → All tests pass
- [ ] `uv run python -m steering_geometry.extract --concept honesty --method weighted_mean --dry-run` → No error
- [ ] `uv run python -m steering_geometry.extract --concept honesty --method discriminative --top-k 50 --dry-run` → No error
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run mypy src/` → Success

### Must Have
- Exact mathematical implementation as specified
- TDD: Tests written before implementation
- CLI support for both methods
- `--top-k` flag with default=100

### Must NOT Have (Guardrails)
- **DO NOT** modify the `Aggregator` type signature globally
- **DO NOT** change existing `mean_aggregator` or `pca_aggregator`
- **DO NOT** touch steering/evaluation pipeline
- **DO NOT** add new dependencies
- **DO NOT** create new modules - add to existing `extract.py`
- **DO NOT** over-abstraction - two specific methods, not a "framework"

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: TDD (Test-Driven Development)
- **Framework**: pytest
- **TDD Flow**: Each task follows RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - Config + Tests):
├── Task 1: Add top_k to ExtractionConfig [quick]
└── Task 2: Create aggregator test file with test stubs [quick]

Wave 2 (After Wave 1 - Implementations):
├── Task 3: Implement weighted_mean_aggregator + tests [unspecified-low]
└── Task 4: Implement discriminative_token_aggregator + tests [unspecified-low]

Wave 3 (After Wave 2 - Integration):
└── Task 5: Update CLI for new methods [quick]

Wave FINAL (After ALL tasks - Verification):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: T1/T2 → T3/T4 → T5 → F1-F4
Parallel Speedup: ~50% faster than sequential
Max Concurrent: 2 (Wave 1 & 2)
```

### Dependency Matrix

- **1**: — — 3, 4, —
- **2**: — — 3, 4, —
- **3**: 1, 2 — 5, —
- **4**: 1, 2 — 5, —
- **5**: 3, 4 — F1-F4, —

### Agent Dispatch Summary

- **1**: **2** — T1 → `quick`, T2 → `quick`
- **2**: **2** — T3 → `unspecified-low`, T4 → `unspecified-low`
- **3**: **1** — T5 → `quick`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. Add top_k to ExtractionConfig

  **What to do**:
  - Add `top_k: int | None = None` field to `ExtractionConfig` dataclass in `config.py`
  - Add docstring explaining it's used only for discriminative method
  - Default value: `None` (discriminative will use 100 as default)

  **Must NOT do**:
  - Do not modify other config fields
  - Do not add validation logic yet (handled in aggregator)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single field addition to dataclass
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 3, Task 4
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/config.py:ExtractionConfig` - Add field here following existing pattern

  **Acceptance Criteria**:
  - [ ] `top_k: int | None = None` field added to `ExtractionConfig`
  - [ ] `uv run mypy src/steering_geometry/config.py` → Success

  **QA Scenarios**:
  ```
  Scenario: Config accepts top_k parameter
    Tool: Bash (python -c)
    Steps:
      1. Run: uv run python -c "from steering_geometry.config import ExtractionConfig; c = ExtractionConfig(top_k=50); print(c.top_k)"
    Expected Result: Output contains "50"
    Evidence: .omo/evidence/task-1-config-topk.txt

  Scenario: Config default top_k is None
    Tool: Bash (python -c)
    Steps:
      1. Run: uv run python -c "from steering_geometry.config import ExtractionConfig; c = ExtractionConfig(); print(c.top_k)"
    Expected Result: Output contains "None"
    Evidence: .omo/evidence/task-1-config-default.txt
  ```

  **Commit**: NO (groups with other tasks)

- [ ] 2. Create aggregator test file with test stubs

  **What to do**:
  - Create `tests/unit/test_aggregators.py`
  - Add test stubs for `test_weighted_mean_aggregator` and `test_discriminative_token_aggregator`
  - Include fixtures for sample tensors
  - Tests should FAIL initially (TDD RED phase)

  **Must NOT do**:
  - Do not implement test logic yet (just stubs)
  - Do not modify existing test files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Create test file with stubs
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `tests/unit/test_extract.py` - Follow existing test patterns
  - `src/steering_geometry/extract.py:mean_aggregator` - Reference for test structure

  **Acceptance Criteria**:
  - [ ] `tests/unit/test_aggregators.py` created
  - [ ] Test stubs for both aggregators present
  - [ ] `uv run pytest tests/unit/test_aggregators.py --collect-only` → Shows 2 tests

  **QA Scenarios**:
  ```
  Scenario: Test file exists and collects
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_aggregators.py --collect-only
    Expected Result: Output shows "test_weighted_mean_aggregator" and "test_discriminative_token_aggregator"
    Evidence: .omo/evidence/task-2-test-collect.txt
  ```

  **Commit**: NO (groups with other tasks)

- [ ] 3. Implement weighted_mean_aggregator

  **What to do**:
  - Implement `weighted_mean_aggregator(pos: Tensor, neg: Tensor) -> Tensor` in `extract.py`
  - Follow the mathematical formulation:
    ```
    For each class c ∈ {+, -}:
      1. Compute class center: h̄_c = (1/n_c) Σ h_i^(c)
      2. Compute variance: τ_c² = (1/n_c) Σ ||h_i^(c) - h̄_c||²
      3. Compute weights: w_i^(c) = exp(-||h_i^(c) - h̄_c||² / τ_c²)
      4. Weighted mean: μ_c^w = Σ w_i^(c) h_i^(c) / Σ w_i^(c)
    Steering direction: v = μ_+^w - μ_-^w
    ```
  - Register in `_resolve_aggregator()` dict with key `"weighted_mean"`
  - Handle edge case: single sample (τ_c² = 0) → use uniform weights
  - Write passing tests in `tests/unit/test_aggregators.py`

  **Must NOT do**:
  - Do not modify `Aggregator` type signature
  - Do not change existing aggregators
  - Do not add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Mathematical implementation with clear specification
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 4)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1, Task 2

  **References**:
  - `src/steering_geometry/extract.py:mean_aggregator` (lines 81-83) - Pattern to follow
  - `src/steering_geometry/extract.py:pca_aggregator` (lines 86-92) - Pattern for complex aggregator
  - `src/steering_geometry/extract.py:_resolve_aggregator` (lines 95-104) - Registration pattern

  **Acceptance Criteria**:
  - [ ] `weighted_mean_aggregator` function implemented
  - [ ] Registered in `_resolve_aggregator()` dict
  - [ ] `uv run pytest tests/unit/test_aggregators.py::test_weighted_mean_aggregator -v` → PASS
  - [ ] `uv run mypy src/steering_geometry/extract.py` → Success

  **QA Scenarios**:
  ```
  Scenario: Weighted mean aggregator produces correct output shape
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_aggregators.py::test_weighted_mean_aggregator -v
    Expected Result: Test passes, output shape is (hidden_dim,)
    Evidence: .omo/evidence/task-3-weighted-mean-shape.txt

  Scenario: Weighted mean handles single sample (edge case)
    Tool: Bash (pytest)
    Steps:
      1. Create test with single sample tensors
      2. Run: uv run pytest tests/unit/test_aggregators.py::test_weighted_mean_single_sample -v
    Expected Result: Test passes, no division by zero
    Evidence: .omo/evidence/task-3-weighted-mean-edge.txt

  Scenario: Weighted mean gives higher weight to central tokens
    Tool: Bash (pytest)
    Steps:
      1. Create test with known distances
      2. Verify tokens closer to center have larger weights
      3. Run: uv run pytest tests/unit/test_aggregators.py::test_weighted_mean_weights -v
    Expected Result: Test passes, weight ordering correct
    Evidence: .omo/evidence/task-3-weighted-mean-weights.txt
  ```

  **Commit**: NO (groups with other tasks)

- [ ] 4. Implement discriminative_token_aggregator

  **What to do**:
  - Implement `discriminative_token_aggregator(pos: Tensor, neg: Tensor, top_k: int = 100) -> Tensor` in `extract.py`
  - Follow the mathematical formulation:
    ```
    For each class c ∈ {+, -}:
      1. Compute class centers: μ_+ and μ_-
      2. Score each token: s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
      3. Select top-k tokens: S_c = TopK(s_i)
      4. Class prototype: μ_c^disc = (1/|S_c|) Σ_{i∈S_c} h_i
    Steering direction: v = μ_+^disc - μ_-^disc
    ```
  - Register in `_resolve_aggregator()` dict with key `"discriminative"`
  - **IMPORTANT**: Modify `_resolve_aggregator()` to accept optional `config: ExtractionConfig | None = None` parameter
    - When method is "discriminative" and config provided, bind `top_k` via `functools.partial`
    - This preserves `Aggregator` type signature while passing `top_k`
  - Handle edge cases:
    - `top_k > num_tokens` → clamp to `num_tokens`
    - Empty tensors → raise `ValueError`
  - Use `torch.topk` for selection
  - Write passing tests in `tests/unit/test_aggregators.py`

  **Must NOT do**:
  - Do not modify `Aggregator` type signature (use `functools.partial` or config-based approach)
  - Do not change existing aggregators
  - Do not add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Mathematical implementation with clear specification
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1, Task 2

  **References**:
  - `src/steering_geometry/extract.py:mean_aggregator` (lines 81-83) - Pattern to follow
  - `src/steering_geometry/config.py:ExtractionConfig` - Access `top_k` from config
  - PyTorch docs: `torch.topk` for token selection

  **Acceptance Criteria**:
  - [ ] `discriminative_token_aggregator` function implemented
  - [ ] Registered in `_resolve_aggregator()` dict
  - [ ] `uv run pytest tests/unit/test_aggregators.py::test_discriminative_token_aggregator -v` → PASS
  - [ ] `uv run mypy src/steering_geometry/extract.py` → Success

  **QA Scenarios**:
  ```
  Scenario: Discriminative aggregator produces correct output shape
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_aggregators.py::test_discriminative_token_aggregator -v
    Expected Result: Test passes, output shape is (hidden_dim,)
    Evidence: .omo/evidence/task-4-disc-shape.txt

  Scenario: Discriminative handles top_k > num_tokens
    Tool: Bash (pytest)
    Steps:
      1. Create test with 10 tokens, request top_k=100
      2. Run: uv run pytest tests/unit/test_aggregators.py::test_discriminative_clamp_topk -v
    Expected Result: Test passes, uses all available tokens
    Evidence: .omo/evidence/task-4-disc-clamp.txt

  Scenario: Discriminative selects tokens far from other class
    Tool: Bash (pytest)
    Steps:
      1. Create test with known token positions
      2. Verify selected tokens have highest discriminative scores
      3. Run: uv run pytest tests/unit/test_aggregators.py::test_discriminative_selection -v
    Expected Result: Test passes, correct tokens selected
    Evidence: .omo/evidence/task-4-disc-selection.txt
  ```

  **Commit**: NO (groups with other tasks)

- [ ] 5. Update CLI for new methods

  **What to do**:
  - Update `--method` argument choices in `_build_parser()` (line 539):
    - Change from `choices=["mean", "pca"]` to `choices=["mean", "pca", "weighted_mean", "discriminative"]`
  - Add `--top-k` argument:
    - Type: `int`
    - Default: `100`
    - Help text: "Number of top tokens to select for discriminative method (default: 100)"
  - Update `_Args` protocol to include `top_k: int`
  - Pass `top_k` from args to `ExtractionConfig`

  **Must NOT do**:
  - Do not change existing argument behavior
  - Do not add validation beyond type checking

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple CLI argument additions
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: None
  - **Blocked By**: Task 3, Task 4

  **References**:
  - `src/steering_geometry/extract.py:_build_parser()` (line 539) - Update method choices
  - `src/steering_geometry/extract.py:_Args` - Add `top_k` field
  - `src/steering_geometry/config.py:ExtractionConfig` - Pass `top_k` to config

  **Acceptance Criteria**:
  - [ ] `--method` accepts `weighted_mean` and `discriminative`
  - [ ] `--top-k` argument added with default 100
  - [ ] `_Args` protocol includes `top_k: int`
  - [ ] `uv run python -m steering_geometry.extract --help` → Shows new options

  **QA Scenarios**:
  ```
  Scenario: CLI accepts weighted_mean method
    Tool: Bash (python -m)
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept honesty --method weighted_mean --dry-run
    Expected Result: No error, dry run completes
    Evidence: .omo/evidence/task-5-cli-weighted-mean.txt

  Scenario: CLI accepts discriminative method with default top_k
    Tool: Bash (python -m)
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept honesty --method discriminative --dry-run
    Expected Result: No error, dry run completes
    Evidence: .omo/evidence/task-5-cli-disc-default.txt

  Scenario: CLI accepts discriminative method with custom top_k
    Tool: Bash (python -m)
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept honesty --method discriminative --top-k 50 --dry-run
    Expected Result: No error, dry run completes
    Evidence: .omo/evidence/task-5-cli-disc-custom.txt

  Scenario: CLI rejects invalid method
    Tool: Bash (python -m)
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept honesty --method invalid --dry-run
    Expected Result: Error message about invalid choice
    Evidence: .omo/evidence/task-5-cli-invalid.txt

  Scenario: Help shows new options
    Tool: Bash (python -m)
    Steps:
      1. Run: uv run python -m steering_geometry.extract --help
      2. Check output contains "weighted_mean", "discriminative", and "--top-k"
    Expected Result: All new options visible in help
    Evidence: .omo/evidence/task-5-cli-help.txt
  ```

  **Commit**: YES
  - Message: `feat(extract): add weighted_mean and discriminative aggregators`
  - Files: `src/steering_geometry/extract.py`, `src/steering_geometry/config.py`, `tests/unit/test_aggregators.py`
  - Pre-commit: `uv run pytest && uv run mypy src/ && uv run ruff check src/ tests/`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist in .omo/evidence/.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Review all changed files.
  Output: `Lint [PASS/FAIL] | Format [PASS/FAIL] | Types [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute CLI commands with new methods. Test edge cases.
  Output: `CLI Commands [N/N pass] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  Verify only extraction logic changed. No modifications to steering/evaluation.
  Output: `Scope [COMPLIANT/VIOLATED] | VERDICT`

---

## Commit Strategy

- **Single commit** after all tasks complete:
- Message: `feat(extract): add weighted_mean and discriminative aggregators`
- Files: `src/steering_geometry/extract.py`, `src/steering_geometry/config.py`, `tests/unit/test_aggregators.py`
- Pre-commit: `uv run pytest && uv run mypy src/ && uv run ruff check src/ tests/`

---

## Success Criteria

### Verification Commands
```bash
# All tests pass
uv run pytest tests/unit/test_aggregators.py -v

# Type check passes
uv run mypy src/steering_geometry/extract.py

# Lint passes
uv run ruff check src/ tests/

# CLI works with new methods
uv run python -m steering_geometry.extract --concept honesty --method weighted_mean --dry-run
uv run python -m steering_geometry.extract --concept honesty --method discriminative --top-k 50 --dry-run
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] CLI accepts new methods
- [ ] CLI accepts --top-k flag
