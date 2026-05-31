# Switch Polite Data Source to Intel/polite-guard

## TL;DR

> **Quick Summary**: Replace the polite concept's data loader from `Cleanlab/stanford-politeness` to `Intel/polite-guard`, filtering to extreme labels only (polite/impolite), and remove the now-unused `pandas` and `hf_hub_download` dependencies.
>
> **Deliverables**:
> - Rewritten `load_polite_data()` using `load_dataset("Intel/polite-guard", split="train")`
> - Updated tests in `tests/unit/test_extract.py`
> - Removed `pandas` and `pandas-stubs` from `pyproject.toml`
> - Removed dead imports (`import pandas`, `from huggingface_hub import hf_hub_download`)
>
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - TDD chain (tests → implementation → cleanup → verify)
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
User wants to switch the "polite" concept to use `Intel/polite-guard` instead of `Cleanlab/stanford-politeness`.

### Interview Summary
**Key Discussions**:
- Label mapping: Extreme only — `polite` (positive) vs `impolite` (negative), ignoring `somewhat polite` and `neutral`
- Split: Use `train` split (80k rows, label-balanced, ~20k per label)
- Test strategy: TDD — write tests first, then implementation
- Dependency cleanup: Remove `pandas` import and `pyproject.toml` entry since no other code uses it

**Research Findings**:
- Intel/polite-guard: 4 string labels (`"polite"`, `"somewhat polite"`, `"neutral"`, `"impolite"`), columns: `text`, `label`, `source`, `reasoning`
- `pandas` and `hf_hub_download` are ONLY used by `load_polite_data()` — verified via grep
- `load_dataset` is already imported and used by `load_sentiment_data()` and `load_refusal_data()`
- The pattern to follow is `load_sentiment_data()` at lines 194-239

### Metis Review
**Identified Gaps** (addressed):
- `pandas` is also in `pyproject.toml` dependencies (line 14) and `pandas-stubs` in dev deps (line 35) — both should be removed
- Module docstring line 7 references old dataset — must update
- Tests use `dataset_name="politeness"` but production sets `dataset_name="polite"` — fix in tests
- Tests are integration tests (real HF API calls) — keep same pattern
- Error messages reference "Stanford Politeness" — must update to "Intel/polite-guard"

---

## Work Objectives

### Core Objective
Rewrite the polite data loader to use `Intel/polite-guard` with extreme labels only, following the existing `load_sentiment_data()` pattern.

### Concrete Deliverables
- `src/steering_geometry/extract.py` — rewritten `load_polite_data()`, removed dead imports, updated docstring
- `tests/unit/test_extract.py` — updated `TestPoliteLoader` class
- `pyproject.toml` — removed `pandas` and `pandas-stubs` dependencies

### Definition of Done
- [ ] `uv sync` completes without errors
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → already formatted
- [ ] `uv run mypy src/` → 0 errors
- [ ] `uv run pytest` → all tests pass

### Must Have
- `load_polite_data()` loads from `Intel/polite-guard` train split
- Filters to `"polite"` (positive) and `"impolite"` (negative) labels only
- Same `ContrastPair` interface as before
- Tests pass with new dataset metadata
- No dead references to old dataset anywhere in codebase

### Must NOT Have (Guardrails)
- DO NOT touch `load_sentiment_data()`, `load_refusal_data()`, or `_DATASET_LOADERS` registry
- DO NOT change `ContrastPair`, `ContrastPairMetadata`, or `ConceptConfig` types
- DO NOT change `SUPPORTED_CONCEPTS` — the concept name "polite" stays
- DO NOT add new dependencies — `load_dataset` is already imported
- DO NOT modify shell scripts or CLI argument names
- DO NOT "harmonize" or refactor the other two loaders
- DO NOT add caching, optimization, or schema validation beyond existing patterns
- DO NOT introduce pandas, polars, or any new data library

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: TDD — tests first, then implementation
- **Framework**: pytest
- **TDD flow**: RED (failing tests for new contract) → GREEN (implementation) → REFACTOR (cleanup)

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python module**: Use Bash (uv run pytest, uv run ruff, uv run mypy, grep)
- **Data loader**: Use Bash (uv run pytest with specific test class)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - TDD: write failing tests):
└── Task 1: Update TestPoliteLoader tests [quick]

Wave 2 (After Wave 1 - implementation):
└── Task 2: Rewrite load_polite_data + remove dead imports [quick]

Wave 3 (After Wave 2 - dependency cleanup):
└── Task 3: Remove pandas from pyproject.toml [quick]

Wave 4 (After Wave 3 - final verification):
└── Task 4: Full verification suite [quick]

Critical Path: Task 1 → Task 2 → Task 3 → Task 4
No parallelization possible (linear TDD chain)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | - | 2 | 1 |
| 2 | 1 | 3, 4 | 2 |
| 3 | 2 | 4 | 3 |
| 4 | 2, 3 | - | 4 |

### Agent Dispatch Summary

- **Wave 1**: 1 task — T1 → `quick`
- **Wave 2**: 1 task — T2 → `quick`
- **Wave 3**: 1 task — T3 → `quick`
- **Wave 4**: 1 task — T4 → `quick`

---

## TODOs

- [x] 1. Update Tests for New Polite Loader (TDD: RED phase)

  **What to do**:
  - Edit `tests/unit/test_extract.py` — update `TestPoliteLoader` class (lines 133-175)
  - Change `source` assertion from `"Cleanlab/stanford-politeness"` to `"Intel/polite-guard"` (line 148)
  - Fix `dataset_name` from `"politeness"` to `"polite"` in all test `ConceptConfig` calls (matching production `load_contrast_pairs()` which sets `dataset_name=concept`)
  - Update docstrings to reference "Intel/polite-guard" instead of "Cleanlab/stanford-politeness" (line 137)
  - Keep `test_polite_uses_direct_text_no_prefix` — Intel/polite-guard texts are also direct (no prefixes)
  - Do NOT add new test files or new test classes — just update existing assertions

  **Must NOT do**:
  - Do not add mock-based tests — keep the existing integration test pattern (real HF API calls)
  - Do not change test method names or test class name
  - Do not add tests for other loaders

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single test file, straightforward assertion updates
  - **Skills**: `[]`
    - No special skills needed for test file edits

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: Task 2
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `tests/unit/test_extract.py:133-175` — Current `TestPoliteLoader` class with all 3 test methods
  - `tests/unit/test_extract.py:136-150` — `test_load_polite_data` showing current assertions for source, dataset, concept

  **API/Type References** (contracts to implement against):
  - `src/steering_geometry/extract.py:242-292` — Current `load_polite_data()` that will be rewritten
  - `src/steering_geometry/types.py` — `ContrastPair`, `ContrastPairMetadata` structures (unchanged)
  - `src/steering_geometry/config.py:68-79` — `ConceptConfig` dataclass (unchanged)

  **WHY Each Reference Matters**:
  - The test file reference shows exactly which lines/strings to change
  - The loader reference shows the current contract the tests verify
  - The types/config references confirm the interface won't change

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Tests reference new dataset metadata
    Tool: Bash (grep)
    Preconditions: test file has been edited
    Steps:
      1. grep -n "Intel/polite-guard" tests/unit/test_extract.py
      2. Verify match count >= 1 (source assertion updated)
      3. grep -n "Cleanlab" tests/unit/test_extract.py
      4. Verify match count = 0 (old reference removed)
    Expected Result: grep for "Intel/polite-guard" returns at least 1 match, grep for "Cleanlab" returns nothing
    Failure Indicators: "Cleanlab" still found in tests, or "Intel/polite-guard" not found
    Evidence: .omo/evidence/task-1-test-references.txt

  Scenario: Tests fail (RED phase - loader not yet updated)
    Tool: Bash (uv run pytest)
    Preconditions: Test assertions updated but loader still uses old dataset
    Steps:
      1. uv run pytest tests/unit/test_extract.py::TestPoliteLoader -v 2>&1
      2. Check exit code is non-zero
    Expected Result: Tests FAIL because loader still returns old metadata (source="Cleanlab/stanford-politeness")
    Failure Indicators: Tests pass (means loader was already changed — wrong order)
    Evidence: .omo/evidence/task-1-red-phase.txt
  ```

  **Commit**: NO (groups with Task 2-3)

- [x] 2. Rewrite load_polite_data + Remove Dead Imports (TDD: GREEN phase)

  **What to do**:
  - In `src/steering_geometry/extract.py`:
    - **Remove** `import pandas as pd` (line 28)
    - **Remove** `from huggingface_hub import hf_hub_download` (line 31)
    - **Rewrite** `load_polite_data()` (lines 242-292) to follow the exact pattern of `load_sentiment_data()` (lines 194-239):
      1. Use `load_dataset("Intel/polite-guard", split="train")` instead of `hf_hub_download` + `pd.read_csv`
      2. Iterate dataset rows: `for row in dataset:`
      3. Filter: `row["label"] == "polite"` → polite list, `row["label"] == "impolite"` → impolite list
      4. Skip rows where `row["label"]` is `"somewhat polite"`, `"neutral"`, or any other value
      5. Skip rows with empty/whitespace text (same guard as sentiment loader)
      6. Use `sample_with_seed()` for sampling (same as sentiment loader)
      7. Set `source="Intel/polite-guard"` in metadata
    - **Update** module docstring line 7: `- polite: Intel/polite-guard`
    - **Update** function docstring: `"""Load politeness contrast pairs from Intel/polite-guard."""`
    - **Update** error messages: change "Stanford Politeness" references to "Intel/polite-guard"
  - Run tests to verify GREEN phase

  **Must NOT do**:
  - Do NOT change `load_sentiment_data()`, `load_refusal_data()`, or `_DATASET_LOADERS`
  - Do NOT change `ContrastPair`, `ContrastPairMetadata`, or `ConceptConfig`
  - Do NOT add new imports — `load_dataset` already imported
  - Do NOT use pandas, polars, or any new library
  - Do NOT modify CLI code, shell scripts, or config.py
  - Do NOT touch `SUPPORTED_CONCEPTS` or `VALID_CONCEPTS`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single function rewrite following existing pattern, well-defined scope
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (solo)
  - **Blocks**: Tasks 3, 4
  - **Blocked By**: Task 1

  **References**:

  **Pattern References** (existing code to follow — CRITICAL):
  - `src/steering_geometry/extract.py:194-239` — `load_sentiment_data()` — **FOLLOW THIS PATTERN EXACTLY** for dataset loading, iteration, filtering, sampling, and return structure
  - `src/steering_geometry/extract.py:242-292` — Current `load_polite_data()` — the function to rewrite

  **API/Type References**:
  - `src/steering_geometry/extract.py:28` — `import pandas as pd` — line to REMOVE
  - `src/steering_geometry/extract.py:31` — `from huggingface_hub import hf_hub_download` — line to REMOVE
  - `src/steering_geometry/extract.py:7` — Module docstring referencing "Cleanlab/stanford-politeness" — line to UPDATE
  - `src/steering_geometry/types.py` — `ContrastPair`, `ContrastPairMetadata` (unchanged interface)
  - `src/steering_geometry/utils.py` — `sample_with_seed()`, `validate_positive_int()` (unchanged utilities)

  **External References**:
  - Intel/polite-guard dataset: labels are strings `"polite"`, `"somewhat polite"`, `"neutral"`, `"impolite"` in column `"label"`, text in column `"text"`, load via `load_dataset("Intel/polite-guard")`

  **WHY Each Reference Matters**:
  - The sentiment loader is the canonical pattern — the new polite loader should be structurally identical
  - Lines 28, 31 are the exact dead imports to remove
  - Line 7 is the docstring that references the old dataset
  - The dataset schema tells you which column names and label values to use

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Tests pass (GREEN phase)
    Tool: Bash (uv run pytest)
    Preconditions: Task 1 completed, loader rewritten
    Steps:
      1. uv run pytest tests/unit/test_extract.py::TestPoliteLoader -v 2>&1
      2. Check exit code is 0
    Expected Result: All 3 TestPoliteLoader tests pass
    Failure Indicators: Any test failure — inspect output for mismatched assertions
    Evidence: .omo/evidence/task-2-green-phase.txt

  Scenario: No dead imports remain
    Tool: Bash (grep)
    Preconditions: Imports removed
    Steps:
      1. grep -rn "import pandas\|import pd" src/steering_geometry/extract.py
      2. grep -rn "hf_hub_download" src/steering_geometry/extract.py
      3. Verify both return no matches (exit code 1)
    Expected Result: No matches for pandas or hf_hub_download in extract.py
    Failure Indicators: Any match found — dead import still present
    Evidence: .omo/evidence/task-2-no-dead-imports.txt

  Scenario: Type checking passes
    Tool: Bash (uv run mypy)
    Preconditions: Code changes complete
    Steps:
      1. uv run mypy src/steering_geometry/extract.py 2>&1
      2. Check exit code is 0
    Expected Result: mypy reports Success with 0 errors
    Failure Indicators: Any type error — likely from leftover pandas types or missing import
    Evidence: .omo/evidence/task-2-mypy.txt

  Scenario: Lint passes
    Tool: Bash (uv run ruff)
    Preconditions: Code changes complete
    Steps:
      1. uv run ruff check src/steering_geometry/extract.py 2>&1
      2. uv run ruff format --check src/steering_geometry/extract.py 2>&1
    Expected Result: 0 violations, already formatted
    Failure Indicators: Any ruff violation or format diff
    Evidence: .omo/evidence/task-2-ruff.txt

  Scenario: Old dataset name fully removed
    Tool: Bash (grep)
    Preconditions: All string references updated
    Steps:
      1. grep -rn "Cleanlab" src/steering_geometry/extract.py
      2. grep -rn "stanford-politeness" src/steering_geometry/extract.py
    Expected Result: No matches for either pattern
    Failure Indicators: Any match — old dataset name still referenced
    Evidence: .omo/evidence/task-2-no-old-refs.txt
  ```

  **Commit**: NO (groups with Task 1, 3)

- [x] 3. Remove pandas from pyproject.toml

  **What to do**:
  - Edit `pyproject.toml`:
    - Remove `"pandas>=2.0,<3.0",` from `dependencies` array (line 14)
    - Remove `"pandas-stubs>=2.0,<3.0",` from `dev` dependency group (line 35)
  - Run `uv sync` to update the lock file
  - Verify `uv sync` completes without errors

  **Must NOT do**:
  - Do NOT remove any other dependencies
  - Do NOT change any other pyproject.toml settings
  - Do NOT add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Two-line removal in config file
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (solo)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References**:

  **API/Type References**:
  - `pyproject.toml:14` — `"pandas>=2.0,<3.0",` in dependencies array — line to REMOVE
  - `pyproject.toml:35` — `"pandas-stubs>=2.0,<3.0",` in dev dependency group — line to REMOVE

  **WHY Each Reference Matters**:
  - These are the exact lines containing pandas dependencies to remove

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: pandas removed from pyproject.toml
    Tool: Bash (grep)
    Preconditions: pyproject.toml edited
    Steps:
      1. grep "pandas" pyproject.toml
      2. Verify exit code is 1 (no matches)
    Expected Result: No matches for "pandas" in pyproject.toml
    Failure Indicators: Any match found — pandas reference still present
    Evidence: .omo/evidence/task-3-no-pandas.txt

  Scenario: uv sync succeeds
    Tool: Bash (uv sync)
    Preconditions: Dependencies removed
    Steps:
      1. uv sync 2>&1
      2. Check exit code is 0
    Expected Result: Sync completes without errors, lock file updated
    Failure Indicators: Any error — likely a dependent package issue
    Evidence: .omo/evidence/task-3-uv-sync.txt
  ```

  **Commit**: NO (groups with Task 1-2)

- [x] 4. Full Verification Suite

  **What to do**:
  - Run the complete Definition of Done verification:
    1. `uv sync`
    2. `uv run ruff check src/ tests/`
    3. `uv run ruff format --check src/ tests/`
    4. `uv run mypy src/`
    5. `uv run pytest`
  - Run grep sweeps for dead references:
    - `grep -r "Cleanlab" src/ tests/`
    - `grep -r "stanford-politeness" src/ tests/`
    - `grep -r "pandas" src/`
    - `grep -r "hf_hub_download" src/`
    - `grep -r "import pd" src/`

  **Must NOT do**:
  - Do NOT fix any issues — just report them (if found, escalate)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Running verification commands only
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (solo)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 2, 3

  **References**:

  **Pattern References**:
  - `AGENTS.md` "Definition of Done" section — the canonical verification checklist

  **WHY Each Reference Matters**:
  - AGENTS.md defines the exact commands that must pass

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All quality gates pass
    Tool: Bash (uv run)
    Preconditions: All implementation tasks complete
    Steps:
      1. uv sync 2>&1
      2. uv run ruff check src/ tests/ 2>&1
      3. uv run ruff format --check src/ tests/ 2>&1
      4. uv run mypy src/ 2>&1
      5. uv run pytest 2>&1
    Expected Result: All 5 commands exit with code 0
    Failure Indicators: Any non-zero exit code
    Evidence: .omo/evidence/task-4-full-verification.txt

  Scenario: No dead references anywhere
    Tool: Bash (grep)
    Preconditions: All changes complete
    Steps:
      1. grep -r "Cleanlab" src/ tests/ 2>&1
      2. grep -r "stanford-politeness" src/ tests/ 2>&1
      3. grep -r "pandas" src/ 2>&1
      4. grep -r "hf_hub_download" src/ 2>&1
    Expected Result: All 4 greps return no matches (exit code 1)
    Failure Indicators: Any match found — dead reference remains
    Evidence: .omo/evidence/task-4-dead-ref-sweep.txt
  ```

  **Commit**: YES
  - Message: `feat(extract): switch polite data source to Intel/polite-guard`
  - Files: `src/steering_geometry/extract.py`, `tests/unit/test_extract.py`, `pyproject.toml`, `uv.lock`
  - Pre-commit: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/ && uv run pytest`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, grep for references). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run mypy src/` + `uv run ruff check src/ tests/` + `uv run pytest`. Review all changed files for: type ignores, empty catches, print in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run all QA scenarios from every task. Test cross-task integration: the loader works end-to-end when called via `load_contrast_pairs("polite", 10)`. Save evidence to `.omo/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

**Single atomic commit** after all tasks pass:
```
feat(extract): switch polite data source to Intel/polite-guard

- Rewrite load_polite_data() to use Intel/polite-guard dataset
- Filter to extreme labels (polite/impolite), ignore somewhat polite and neutral
- Remove pandas and huggingface_hub imports (no longer needed)
- Remove pandas from production and dev dependencies
- Update tests to match new dataset metadata
```
- Files: `src/steering_geometry/extract.py`, `tests/unit/test_extract.py`, `pyproject.toml`, `uv.lock`
- Pre-commit: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/ && uv run pytest`

---

## Success Criteria

### Verification Commands
```bash
uv sync                                                      # Expected: completes without errors
uv run ruff check src/ tests/                                # Expected: 0 violations
uv run ruff format --check src/ tests/                       # Expected: already formatted
uv run mypy src/                                             # Expected: Success, 0 errors
uv run pytest                                                # Expected: all tests pass
grep -r "Cleanlab" src/ tests/                               # Expected: no matches
grep -r "stanford-politeness" src/ tests/                    # Expected: no matches
grep -r "pandas" src/                                        # Expected: no matches
grep -r "hf_hub_download" src/                               # Expected: no matches
grep -r "import pd" src/                                     # Expected: no matches
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] No dead references to old dataset
- [ ] No dead imports
