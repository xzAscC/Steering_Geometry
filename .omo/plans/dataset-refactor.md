# Dataset Refactor: Sentiment + Refusal + Polite

## TL;DR

> **Quick Summary**: Refactor the steering vector dataset system to keep only `sentiment` and `refusal` concepts, add new `polite` concept from Cleanlab/stanford-politeness, and completely remove `honesty`, `toxicity`, `sycophancy` concepts.
> 
> **Deliverables**:
> - New `load_polite_data()` function with TDD tests
> - Removed honesty/toxicity/sycophancy loaders and related code
> - Updated VALID_CONCEPTS, _DATASET_LOADERS, tests, and documentation
>
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential removal required for stability
> **Critical Path**: Add polite → Remove honesty → Remove toxicity → Remove sycophancy → Update docs

---

## Context

### Original Request
User wants to simplify the dataset system:
- Keep: sentiment, refusal
- Add: polite (new concept)
- Remove: honesty, toxicity, sycophancy

### Interview Summary
**Key Discussions**:
- UltraFeedback was considered for refusal but rejected (measures helpfulness, not refusal)
- LLM-LAT/harmful-dataset retained for refusal (no filtering needed)
- Cleanlab/stanford-politeness selected for polite concept
- Polite should NOT use prefix (like sentiment, direct text)
- Test strategy: TDD (write tests first)
- Removal method: complete code deletion

**Research Findings**:
- Cleanlab/stanford-politeness has buggy HuggingFace viewer → use `hf_hub_download()` instead of `load_dataset()`
- Dataset has `text`, `label` (0=impolite, 1=polite), and annotator score columns
- Use `fine-tuning/train_full.csv` for training data

### Metis Review
**Identified Gaps** (addressed):
- Output directory cleanup: Will NOT touch `outputs/` in this refactor
- Test data migration: Tests using "honesty" will be updated to use "sentiment"
- Error messages: Standard "Invalid concept" error is sufficient
- Shell scripts: Will update only critical scripts (pipeline), leave analysis scripts for later

---

## Work Objectives

### Core Objective
Simplify the concept system to 3 focused concepts (sentiment, refusal, polite) while maintaining code quality and test coverage.

### Concrete Deliverables
- `load_polite_data()` function in `src/steering_geometry/extract.py`
- Updated `VALID_CONCEPTS = {"sentiment", "refusal", "polite"}`
- Removed: `load_honesty_data`, `load_toxicity_data`, `load_sycophancy_data`
- Removed: `_HONEST_PREFIX`, `_DISHONEST_PREFIX`, `_SYCOPHANTIC_PREFIX`, `_OBJECTIVE_PREFIX`
- Updated tests in `tests/unit/test_extract.py`
- Updated documentation (README.md, AGENTS.md)

### Definition of Done
- [ ] `uv run pytest tests/unit/test_extract.py -v` → all pass
- [ ] `uv run mypy src/steering_geometry/extract.py` → Success
- [ ] `uv run ruff check src/steering_geometry/extract.py` → 0 violations
- [ ] `uv run python -m steering_geometry.extract --concept polite --dry-run --num-pairs 5` works
- [ ] `uv run python -m steering_geometry.extract --concept honesty --dry-run` → error

### Must Have
- `load_polite_data()` must use `hf_hub_download()` (buggy HF viewer)
- Polite must follow sentiment pattern (no prefix, direct text)
- All 21 existing tests must pass after changes
- TDD: Tests written before implementation

### Must NOT Have (Guardrails)
- DO NOT modify `ContrastPair` or `ConceptConfig` types
- DO NOT touch `outputs/` directory
- DO NOT add caching/optimization beyond scope
- DO NOT fix typing issues outside extract.py
- DO NOT update analysis scripts (lower priority, separate task)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (TDD)
- **Framework**: pytest
- **If TDD**: Each task follows RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **CLI commands**: Use Bash — Run extraction commands, check exit codes, parse output
- **Type checking**: Use Bash — Run mypy and verify "Success" output
- **Linting**: Use Bash — Run ruff and verify "0 violations"

---

## Execution Strategy

### Sequential Execution (Removal Order Matters)

```
Task 1: Add load_polite_data() with TDD tests (foundation)
    ↓
Task 2: Remove honesty concept (loader + constants + tests)
    ↓
Task 3: Remove toxicity concept (loader + tests)
    ↓
Task 4: Remove sycophancy concept (loader + constants + tests)
    ↓
Task 5: Update documentation (README.md, AGENTS.md)
    ↓
Task 6: Update tests referencing removed concepts
    ↓
Task 7: Update shell scripts (pipeline scripts only)
    ↓
FINAL: Verification wave
```

### Why Sequential
- Adding polite first ensures codebase always has 3+ valid concepts
- Removing one concept at a time allows verification at each step
- Prevents cascade failures from bulk changes

### Agent Dispatch Summary
- Task 1: quick (TDD: test + implementation)
- Task 2: quick (removal + verification)
- Task 3: quick (removal + verification)
- Task 4: quick (removal + verification)
- Task 5: writing (documentation update)
- Task 6: quick (test updates)
- Task 7: quick (script updates)
- FINAL: unspecified-high (verification)

---

## TODOs

- [x] 1. Add `load_polite_data()` with TDD tests

  **What to do**:
  - Write failing tests for `load_polite_data()` in `tests/unit/test_extract.py`
  - Implement `load_polite_data()` using `hf_hub_download()` + pandas
  - Load from `Cleanlab/stanford-politeness`, file `fine-tuning/train_full.csv`
  - Use binary `label` column: 1=polite (positive), 0=impolite (negative)
  - NO prefix — use original text directly (like sentiment)
  - Add `"polite"` to `VALID_CONCEPTS`
  - Register in `_DATASET_LOADERS`
  - Update `__all__` exports

  **Must NOT do**:
  - DO NOT add prefix constants for polite
  - DO NOT use `load_dataset()` (buggy HF viewer)
  - DO NOT modify other loaders

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single function addition with clear pattern to follow
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundation task)
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 2-7
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/extract.py:243-288` - `load_sentiment_data()` pattern (no prefix, direct text)
  - `src/steering_geometry/extract.py:77` - `VALID_CONCEPTS` set
  - `src/steering_geometry/extract.py:417-423` - `_DATASET_LOADERS` registry
  - `tests/unit/test_extract.py:54-66` - `test_load_sentiment_data` test pattern

  **Acceptance Criteria** (TDD):
  - [ ] Test file: `tests/unit/test_extract.py` has `TestPoliteLoader` class
  - [ ] Test: `test_load_polite_data_returns_correct_count` passes
  - [ ] Test: `test_load_polite_data_has_correct_metadata` passes
  - [ ] `uv run pytest tests/unit/test_extract.py::TestPoliteLoader -v` → PASS

  **QA Scenarios**:
  ```
  Scenario: Polite concept loads successfully
    Tool: Bash
    Preconditions: Clean environment
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept polite --dry-run --num-pairs 5
      2. Check exit code is 0
      3. Check stdout contains "Loaded 5 contrast pairs for polite"
    Expected Result: Exit code 0, correct output
    Failure Indicators: Exit code non-zero, or "Invalid concept" error
    Evidence: .omo/evidence/task-01-polite-load.txt

  Scenario: Polite loader validates num_pairs
    Tool: Bash
    Preconditions: Clean environment
    Steps:
      1. Run: uv run python -c "from steering_geometry.extract import load_polite_data; load_polite_data(__import__('steering_geometry.config').config.ConceptConfig('polite', 'polite', 0))"
      2. Check stderr contains "must be positive"
    Expected Result: ValueError raised with message
    Failure Indicators: No error or different error
    Evidence: .omo/evidence/task-01-polite-validation.txt
  ```

  **Commit**: YES
  - Message: `feat(extract): add load_polite_data() for politeness concept`
  - Files: `src/steering_geometry/extract.py`, `tests/unit/test_extract.py`
  - Pre-commit: `uv run pytest tests/unit/test_extract.py -v`

- [x] 2. Remove honesty concept

  **What to do**:
  - Delete `load_honesty_data()` function (lines ~204-240)
  - Delete `_HONEST_PREFIX` and `_DISHONEST_PREFIX` constants (lines ~67-68)
  - Remove `"honesty"` from `VALID_CONCEPTS`
  - Remove `"honesty": load_honesty_data` from `_DATASET_LOADERS`
  - Remove `load_honesty_data` from `__all__`
  - Delete `test_load_honesty_data` from tests

  **Must NOT do**:
  - DO NOT modify other loaders
  - DO NOT touch outputs/ directory

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple deletion of code
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 1)
  - **Blocks**: Tasks 3-7
  - **Blocked By**: Task 1

  **References**:
  - `src/steering_geometry/extract.py:204-240` - `load_honesty_data()` to delete
  - `src/steering_geometry/extract.py:67-68` - Prefix constants to delete
  - `tests/unit/test_extract.py:40-52` - Test to delete

  **Acceptance Criteria**:
  - [ ] `uv run python -m steering_geometry.extract --concept honesty --dry-run` exits with error
  - [ ] `uv run grep -r "load_honesty_data" src/` returns no matches
  - [ ] `uv run pytest tests/unit/test_extract.py -v` → all pass

  **QA Scenarios**:
  ```
  Scenario: Honesty concept is rejected
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept honesty --dry-run 2>&1
      2. Check output contains "Invalid concept" or "invalid choice"
      3. Check exit code is non-zero
    Expected Result: Error with "Invalid concept"
    Evidence: .omo/evidence/task-02-honesty-removed.txt
  ```

  **Commit**: YES
  - Message: `refactor(extract): remove honesty concept and loader`
  - Files: `src/steering_geometry/extract.py`, `tests/unit/test_extract.py`

- [x] 3. Remove toxicity concept

  **What to do**:
  - Delete `load_toxicity_data()` function
  - Remove `"toxicity"` from `VALID_CONCEPTS`
  - Remove from `_DATASET_LOADERS`
  - Remove from `__all__`
  - Delete `test_load_toxicity_data` from tests

  **Must NOT do**:
  - DO NOT modify other loaders

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 2)
  - **Blocks**: Tasks 4-7
  - **Blocked By**: Task 2

  **References**:
  - `src/steering_geometry/extract.py:291-337` - `load_toxicity_data()` to delete
  - `tests/unit/test_extract.py:68-79` - Test to delete

  **Acceptance Criteria**:
  - [ ] `uv run python -m steering_geometry.extract --concept toxicity --dry-run` exits with error
  - [ ] `uv run pytest tests/unit/test_extract.py -v` → all pass

  **QA Scenarios**:
  ```
  Scenario: Toxicity concept is rejected
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept toxicity --dry-run 2>&1
      2. Check output contains "Invalid concept"
    Expected Result: Error with "Invalid concept"
    Evidence: .omo/evidence/task-03-toxicity-removed.txt
  ```

  **Commit**: YES
  - Message: `refactor(extract): remove toxicity concept and loader`

- [x] 4. Remove sycophancy concept

  **What to do**:
  - Delete `load_sycophancy_data()` function
  - Delete `_SYCOPHANTIC_PREFIX` and `_OBJECTIVE_PREFIX` constants
  - Remove `"sycophancy"` from `VALID_CONCEPTS`
  - Remove from `_DATASET_LOADERS`
  - Remove from `__all__`
  - Delete `test_load_sycophancy_data` from tests

  **Must NOT do**:
  - DO NOT modify remaining loaders (sentiment, refusal, polite)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 3)
  - **Blocks**: Tasks 5-7
  - **Blocked By**: Task 3

  **References**:
  - `src/steering_geometry/extract.py:340-375` - `load_sycophancy_data()` to delete
  - `src/steering_geometry/extract.py:70-71` - Prefix constants to delete
  - `tests/unit/test_extract.py:82-93` - Test to delete

  **Acceptance Criteria**:
  - [ ] `uv run python -m steering_geometry.extract --concept sycophancy --dry-run` exits with error
  - [ ] `uv run pytest tests/unit/test_extract.py -v` → all pass

  **QA Scenarios**:
  ```
  Scenario: Sycophancy concept is rejected
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept sycophancy --dry-run 2>&1
      2. Check output contains "Invalid concept"
    Expected Result: Error with "Invalid concept"
    Evidence: .omo/evidence/task-04-sycophancy-removed.txt
  ```

  **Commit**: YES
  - Message: `refactor(extract): remove sycophancy concept and loader`

- [x] 5. Update documentation

  **What to do**:
  - Update `README.md`: Change concept list from 5 to 3 (sentiment, refusal, polite)
  - Update `src/steering_geometry/extract.py` module docstring (lines 1-18)
  - Update CLI help text (line 593-594)
  - Update `AGENTS.md` VALID_CONCEPTS and dataset documentation

  **Must NOT do**:
  - DO NOT modify API documentation beyond concept lists

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation updates
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 4)
  - **Blocks**: Tasks 6-7
  - **Blocked By**: Task 4

  **References**:
  - `README.md` - Update supported concepts section
  - `src/steering_geometry/extract.py:1-18` - Module docstring
  - `AGENTS.md` - Code map section

  **Acceptance Criteria**:
  - [ ] README.md mentions only sentiment, refusal, polite
  - [ ] Module docstring lists correct concepts
  - [ ] CLI `--help` shows correct choices

  **QA Scenarios**:
  ```
  Scenario: Documentation is consistent
    Tool: Bash
    Steps:
      1. Run: grep -c "honesty" README.md → expect 0
      2. Run: grep -c "toxicity" README.md → expect 0
      3. Run: grep -c "sycophancy" README.md → expect 0
      4. Run: grep -c "polite" README.md → expect >= 1
    Expected Result: No mentions of removed concepts, polite mentioned
    Evidence: .omo/evidence/task-05-docs-updated.txt
  ```

  **Commit**: YES
  - Message: `docs: update documentation for new concept set`

- [x] 6. Update tests referencing removed concepts

  **What to do**:
  - Update `tests/test_unembed_analysis.py` - change "honesty" to "sentiment"
  - Update `tests/unit/test_evaluation.py` - change "honesty" to "sentiment"
  - Update `tests/test_experiments.py` - change "honesty" to "sentiment"
  - Update `tests/test_token_analysis.py` - change "honesty"/"toxicity" to "sentiment"/"polite"
  - Ensure all tests still pass

  **Must NOT do**:
  - DO NOT change test semantics (keep same assertions, just different concept)
  - DO NOT remove test coverage

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 5

  **References**:
  - `tests/test_unembed_analysis.py:168,186` - Uses "honesty"
  - `tests/unit/test_evaluation.py:91,133,257,275` - Uses "honesty"
  - `tests/test_experiments.py:213,218,245` - Uses "honesty"
  - `tests/test_token_analysis.py:77,82,89,123,128,136,307,309` - Uses "honesty"/"toxicity"

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/ -v` → all pass
  - [ ] No "honesty", "toxicity", "sycophancy" in test files (except in comments/strings explaining history)

  **QA Scenarios**:
  ```
  Scenario: All tests pass after migration
    Tool: Bash
    Steps:
      1. Run: uv run pytest tests/ -v --tb=short
      2. Check for "passed" and no "failed" or "error"
    Expected Result: All tests pass
    Evidence: .omo/evidence/task-06-tests-pass.txt
  ```

  **Commit**: YES
  - Message: `test: update tests to use sentiment instead of removed concepts`

- [x] 7. Update shell scripts

  **What to do**:
  - Update `scripts/pipeline/quick_pipeline.sh` - update ALL_CONCEPTS array (line ~26)
  - Leave analysis scripts (vector_analysis, tdnv, etc.) for separate task

  **Must NOT do**:
  - DO NOT update all 11+ scripts (lower priority)
  - DO NOT change script logic, only concept lists

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 6)
  - **Blocks**: Final verification
  - **Blocked By**: Task 6

  **References**:
  - `scripts/pipeline/quick_pipeline.sh` - Main pipeline script (ALL_CONCEPTS at line ~26)

  **Acceptance Criteria**:
  - [ ] `./scripts/pipeline/quick_pipeline.sh -c all` only uses sentiment, refusal, polite
  - [ ] Scripts don't reference removed concepts

  **QA Scenarios**:
  ```
  Scenario: Pipeline script uses correct concepts
    Tool: Bash
    Steps:
      1. Run: grep "ALL_CONCEPTS\|honesty\|toxicity\|sycophancy" scripts/pipeline/quick_pipeline.sh
      2. Check no matches for removed concepts
    Expected Result: Only sentiment, refusal, polite in script
    Evidence: .omo/evidence/task-07-scripts-updated.txt
  ```

  **Commit**: YES
  - Message: `refactor(scripts): update pipeline scripts for new concepts`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [x] F1. **Plan Compliance Audit** — `oracle`
  Verify all "Must Have" present, all "Must NOT Have" absent.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `mypy`, `ruff`, `pytest`. Check for regressions.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

- [x] F3. **Functional QA** — `unspecified-high`
  Execute all QA scenarios from tasks. Test polite extraction works.
  Output: `Scenarios [N/N pass] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  Verify 1:1 — everything in spec built, nothing beyond spec.
  Output: `Tasks [N/N compliant] | VERDICT`

---

## Commit Strategy

- **Task 1**: `feat(extract): add load_polite_data() for politeness concept`
- **Task 2**: `refactor(extract): remove honesty concept and loader`
- **Task 3**: `refactor(extract): remove toxicity concept and loader`
- **Task 4**: `refactor(extract): remove sycophancy concept and loader`
- **Task 5**: `docs: update documentation for new concept set`
- **Task 6**: `test: update tests to use sentiment instead of removed concepts`
- **Task 7**: `refactor(scripts): update pipeline scripts for new concepts`

---

## Success Criteria

### Verification Commands
```bash
# All concepts work
uv run python -m steering_geometry.extract --concept sentiment --dry-run --num-pairs 5
uv run python -m steering_geometry.extract --concept refusal --dry-run --num-pairs 5
uv run python -m steering_geometry.extract --concept polite --dry-run --num-pairs 5

# Removed concepts fail
uv run python -m steering_geometry.extract --concept honesty --dry-run 2>&1 | grep "Invalid concept"
uv run python -m steering_geometry.extract --concept toxicity --dry-run 2>&1 | grep "Invalid concept"
uv run python -m steering_geometry.extract --concept sycophancy --dry-run 2>&1 | grep "Invalid concept"

# Quality checks
uv run mypy src/steering_geometry/extract.py
uv run ruff check src/steering_geometry/extract.py
uv run pytest tests/unit/test_extract.py -v
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass (21+ tests)
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Documentation updated
