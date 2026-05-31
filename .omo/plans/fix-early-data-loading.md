# Fix Early Data Loading: Eliminate Full-Dataset Load + Double-Load Bug

## TL;DR

> **Quick Summary**: Fix two performance issues: (1) `load_sentiment_data()` and `load_polite_data()` load the entire dataset into memory before sampling down to `num_pairs`, and (2) tdnv.py's `_run_concept()` loads data twice — once for dry-run printing and again inside `compute_tdnv_for_concept()`.
>
> **Deliverables**:
> - `extract.py`: Early-stop in sentiment/polite loaders with oversample buffer to preserve random sampling
> - `tdnv.py`: Eliminate double-load in `_run_concept()` CLI path
> - Tests for both fixes
>
> **Estimated Effort**: Quick
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 + Task 2 (parallel) → Task 3 → Task 4

---

## Context

### Original Request
When running TDNV with only 500 data points, the entire dataset is loaded first then sampled down. This is wasteful. Also, `_run_concept()` in tdnv.py calls `load_contrast_pairs()` twice.

### Interview Summary
**Key Discussions**:
- Confirmed the problem affects `load_sentiment_data()` and `load_polite_data()` in extract.py
- `load_refusal_data()` already uses early-break — correct pattern
- Double-load bug in `_run_concept()`: line 967 loads for dry-run print, line 426 (inside `compute_tdnv_for_concept`) loads again
- Both fixes should be applied

**Research Findings**:
- `sample_with_seed()` uses `random.Random(42).sample()` — naive early-stop would lose randomness
- Need oversample buffer strategy: collect `num_pairs * 2` per class, then `sample_with_seed` down
- `load_refusal_data()` pattern (extract.py:288-323) is the reference for early-stop
- SST-2 has ~67k rows (balanced ~33k pos / ~33k neg); Intel/polite-guard dataset also has many rows
- For `num_pairs=500`, currently iterates ~67k rows; with fix, iterates ~2000 rows

### Metis Review
**Identified Gaps** (addressed):
- **Silent behavioral change**: Naive early-stop would give first-N items instead of random sample. RESOLVED: Use oversample buffer (collect `num_pairs * 2` per class, then `sample_with_seed` down to `num_pairs`)
- **Edge case: class imbalance**: If one class is rare, early-stop may iterate full dataset anyway. RESOLVED: Natural fallback — if buffer isn't filled, we collect what's available and `sample_with_seed` handles it
- **API impact**: Adding `pairs` parameter to `compute_tdnv_for_concept` changes public API. RESOLVED: Restructure `_run_concept()` only — no API change needed

---

## Work Objectives

### Core Objective
Stop loading entire datasets when only a small subset is needed, and eliminate a redundant data-load call in the tdnv CLI.

### Concrete Deliverables
- `src/steering_geometry/extract.py`: `load_sentiment_data()` and `load_polite_data()` with early-stop + oversample buffer
- `src/steering_geometry/tdnv.py`: `_run_concept()` restructured to avoid double-load
- Test coverage for both fixes

### Definition of Done
- [ ] `uv run pytest` passes (all existing + new tests)
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → formatted
- [ ] `uv run mypy src/` → 0 errors

### Must Have
- `sample_with_seed` still called for sentiment/polite (preserves determinism)
- Early-stop limits iteration to approximately `num_pairs * 4` rows for balanced datasets
- `_run_concept()` calls `load_contrast_pairs` exactly once (not twice) on non-dry-run path
- Dry-run still works correctly
- All existing tests pass unchanged

### Must NOT Have (Guardrails)
- Do NOT change function signatures of `load_contrast_pairs`, `load_sentiment_data`, `load_polite_data`
- Do NOT touch `load_refusal_data` — already correct
- Do NOT modify files outside `extract.py`, `tdnv.py`, and test files
- Do NOT remove `sample_with_seed` usage from sentiment/polite loaders
- Do NOT add new dependencies
- Do NOT change `compute_tdnv_for_concept()`'s public API
- Do NOT touch `stability_comparison.py`, `token_analysis.py`, or `apply_steering.py`

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (tests-after — small fix, tests added alongside implementation)
- **Framework**: pytest

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — independent changes):
├── Task 1: Fix tdnv.py double-load in _run_concept() [quick]
└── Task 2: Add early-stop to sentiment/polite loaders in extract.py [quick]

Wave 2 (After Wave 1 — tests + QA):
├── Task 3: Write tests for both fixes [quick]

Wave FINAL (After ALL tasks):
├── Task F1: Full quality gate (ruff, mypy, pytest)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1    | -         | 3      |
| 2    | -         | 3      |
| 3    | 1, 2      | F1     |
| F1   | 3         | -      |

### Agent Dispatch Summary

- **Wave 1**: **2** — T1 → `quick`, T2 → `quick`
- **Wave 2**: **1** — T3 → `quick`
- **FINAL**: **1** — F1 → `quick`

---

## TODOs

- [x] 1. Fix tdnv.py double-load in `_run_concept()`

  **What to do**:
  - In `src/steering_geometry/tdnv.py`, restructure `_run_concept()` (line 962-989) to eliminate the double-load
  - Current code loads data at line 967 for dry-run printing, then `compute_tdnv_for_concept()` loads again at line 426
  - **Fix**: Move the `load_contrast_pairs` call into the `if args.dry_run:` branch only. For non-dry-run, remove the redundant load — `compute_tdnv_for_concept()` already handles loading+printing
  - The restructured code should be:
    ```python
    def _run_concept(args: _Args) -> None:
        if args.concept is None:
            print("Error: --concept is required for --mode concept")
            raise SystemExit(1)

        if args.dry_run:
            pairs = load_contrast_pairs(args.concept, args.num_pairs)
            print(f"Loaded {len(pairs)} contrast pairs for {args.concept}")
            print("Dry run complete")
            return

        config = TDNVConfig(
            num_pairs=args.num_pairs,
            output_dir=args.output,
            plot_dir=args.plot_dir,
        )

        result = compute_tdnv_for_concept(
            concept=args.concept,
            model_name=args.model,
            config=config,
            last_n=args.last_n,
            top_k=args.top_k,
        )

        save_tdnv_result(result, Path(args.output))
        plot_tdnv_trends(result, Path(args.plot_dir))
    ```

  **Must NOT do**:
  - Do NOT change `compute_tdnv_for_concept()` signature
  - Do NOT modify `compute_tdnv_for_mmlu()` or `_run_mmlu()`
  - Do NOT change any behavior — only remove redundant load

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `src/steering_geometry/tdnv.py:962-989` — Current `_run_concept()` with double-load bug
  - `src/steering_geometry/tdnv.py:426-427` — `compute_tdnv_for_concept()` already loads and prints pairs

  **WHY Each Reference Matters**:
  - The double-load is exactly at lines 967 and 426. Line 967 loads for dry-run display but is also called in non-dry-run path. Line 426 always loads. The fix is to only call `load_contrast_pairs` in the dry-run branch at line 967.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify double-load is fixed (non-dry-run path)
    Tool: Bash
    Preconditions: tdnv.py modified, tests written
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py -v -k "test_run_concept_no_double_load"
      2. Assert: test passes (load_contrast_pairs mocked, called exactly 0 times in _run_concept non-dry-run, delegated to compute_tdnv_for_concept)
    Expected Result: Test passes with 0 failures
    Failure Indicators: Mock assertion fails (function called more than once)
    Evidence: .omo/evidence/task-1-no-double-load.txt

  Scenario: Verify dry-run still works
    Tool: Bash
    Preconditions: tdnv.py modified
    Steps:
      1. Run: uv run pytest tests/unit/test_tdnv.py -v -k "test_run_concept_dry_run"
      2. Assert: test passes (dry-run loads once, prints count, returns early)
    Expected Result: Test passes
    Failure Indicators: Dry-run fails or load_contrast_pairs not called
    Evidence: .omo/evidence/task-1-dry-run.txt
  ```

  **Commit**: YES
  - Message: `fix(tdnv): eliminate double-load in _run_concept CLI path`
  - Files: `src/steering_geometry/tdnv.py`
  - Pre-commit: `uv run pytest tests/unit/test_tdnv.py -v`

- [x] 2. Add early-stop to `load_sentiment_data()` and `load_polite_data()` in extract.py

  **What to do**:
  - In `src/steering_geometry/extract.py`, modify `load_sentiment_data()` (line 192-237) and `load_polite_data()` (line 240-285) to stop iterating the dataset once enough items are collected
  - **Strategy**: Oversample buffer approach — collect `num_pairs * 2` items per class, then `sample_with_seed` down to `num_pairs`. This preserves the random sampling guarantee while dramatically reducing loaded data.
  - For `load_sentiment_data()`:
    ```python
    def load_sentiment_data(config: ConceptConfig) -> list[ContrastPair]:
        validate_positive_int(config.num_pairs, "num_pairs")

        dataset = load_dataset("glue", "sst2")

        # Collect oversample buffer for random sampling
        oversample = config.num_pairs * 2
        positives: list[str] = []
        negatives: list[str] = []
        for row in dataset["train"]:
            sentence = row["sentence"]
            label = row["label"]
            if not sentence or not sentence.strip():
                continue
            if label == 1:
                positives.append(sentence.strip())
            elif label == 0:
                negatives.append(sentence.strip())
            # Early stop: collected enough for oversample buffer
            if len(positives) >= oversample and len(negatives) >= oversample:
                break

        # ... rest is the same (sample_with_seed, create ContrastPairs)
    ```
  - Apply identical pattern to `load_polite_data()` with its field names (`text`/`label`/`polite`/`impolite`)
  - Do NOT touch `load_refusal_data()` — it's already correct

  **Must NOT do**:
  - Do NOT remove `sample_with_seed` calls — they provide determinism
  - Do NOT change function signatures
  - Do NOT touch `load_refusal_data()`
  - Do NOT change the oversample factor to something other than 2x

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `src/steering_geometry/extract.py:288-323` — `load_refusal_data()` has the early-break pattern at line 296-297: `if pair_index >= config.num_pairs: break`
  - `src/steering_geometry/extract.py:192-237` — `load_sentiment_data()` current full-load implementation
  - `src/steering_geometry/extract.py:240-285` — `load_polite_data()` current full-load implementation

  **API/Type References**:
  - `src/steering_geometry/utils.py:sample_with_seed` — Deterministic sampling function used after collection
  - `src/steering_geometry/config.py:ConceptConfig` — Config with `num_pairs` field

  **WHY Each Reference Matters**:
  - `load_refusal_data()` shows the canonical early-stop pattern — follow its structure
  - `sample_with_seed` must still be called on the collected buffer to maintain the same determinism contract
  - The `oversample = num_pairs * 2` ensures we have enough items for `sample_with_seed` to pick a random subset from, rather than just taking the first N items in dataset order

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify early-stop reduces iterations for sentiment
    Tool: Bash
    Preconditions: extract.py modified
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -v -k "test_sentiment_early_stop"
      2. Assert: test passes (mock dataset iteration, verify break happens before full iteration)
    Expected Result: Test passes, dataset iteration count < total rows
    Failure Indicators: Full dataset iterated despite small num_pairs
    Evidence: .omo/evidence/task-2-sentiment-early-stop.txt

  Scenario: Verify early-stop reduces iterations for polite
    Tool: Bash
    Preconditions: extract.py modified
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -v -k "test_polite_early_stop"
      2. Assert: test passes (same verification as sentiment)
    Expected Result: Test passes
    Failure Indicators: Full dataset iterated
    Evidence: .omo/evidence/task-2-polite-early-stop.txt

  Scenario: Verify sample_with_seed still called (determinism preserved)
    Tool: Bash
    Preconditions: extract.py modified
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -v -k "test_sample_with_seed_called"
      2. Assert: test passes (mock sample_with_seed, verify it's called with collected data)
    Expected Result: sample_with_seed called with the buffer data, not skipped
    Failure Indicators: sample_with_seed not called or called with wrong data
    Evidence: .omo/evidence/task-2-determinism.txt
  ```

  **Commit**: YES
  - Message: `perf(extract): add early-stop to sentiment and polite data loaders`
  - Files: `src/steering_geometry/extract.py`
  - Pre-commit: `uv run pytest tests/unit/test_extract.py -v`

- [x] 3. Write tests for both fixes

  **What to do**:
  - Add tests to `tests/unit/test_tdnv.py` for the double-load fix:
    - Test that `_run_concept` non-dry-run path does NOT call `load_contrast_pairs` directly (it delegates to `compute_tdnv_for_concept` which handles loading). Mock `compute_tdnv_for_concept` and `load_contrast_pairs`, verify `load_contrast_pairs` is called exactly 0 times in `_run_concept` for non-dry-run (since `compute_tdnv_for_concept` is mocked).
    - Test that `_run_concept` dry-run path calls `load_contrast_pairs` exactly once
  - Add tests to `tests/unit/test_extract.py` for early-stop:
    - Test `load_sentiment_data` with mocked dataset stops early when buffer is full
    - Test `load_polite_data` with mocked dataset stops early when buffer is full
    - Test `sample_with_seed` is still called (mock it, verify invocation)
    - Test edge case: `num_pairs` larger than available data in one class (should still work, just returns what's available)
  - Use `unittest.mock.patch` and `unittest.mock.MagicMock` for mocking
  - Follow existing test patterns in the test files

  **Must NOT do**:
  - Do NOT require actual HuggingFace dataset downloads for these tests — use mocks
  - Do NOT modify existing passing tests

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential)
  - **Blocks**: F1
  - **Blocked By**: Task 1, Task 2

  **References**:

  **Pattern References**:
  - `tests/unit/test_extract.py` — Existing extract tests showing mock patterns
  - `tests/unit/test_tdnv.py` or `tests/conftest.py` — Existing test fixtures and patterns
  - `tests/conftest.py` — Shared fixtures (`mock_hooked_model`, `sample_contrast_pairs`)

  **API/Type References**:
  - `src/steering_geometry/tdnv.py:_run_concept` — Function under test for double-load
  - `src/steering_geometry/tdnv.py:_Args` — Protocol for CLI args
  - `src/steering_geometry/extract.py:load_sentiment_data` — Function under test for early-stop
  - `src/steering_geometry/extract.py:load_polite_data` — Function under test for early-stop

  **WHY Each Reference Matters**:
  - Existing tests show how to mock HuggingFace datasets and `HookedModel`
  - `_Args` protocol defines what fields the mock args need

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All new tests pass
    Tool: Bash
    Preconditions: Both fixes implemented
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py tests/unit/test_tdnv.py -v
      2. Assert: all tests pass, 0 failures
    Expected Result: All tests pass
    Failure Indicators: Any test failure
    Evidence: .omo/evidence/task-3-all-tests.txt

  Scenario: Existing tests still pass
    Tool: Bash
    Preconditions: Both fixes implemented
    Steps:
      1. Run: uv run pytest -v
      2. Assert: all tests pass (existing + new)
    Expected Result: All tests pass
    Failure Indicators: Any regression in existing tests
    Evidence: .omo/evidence/task-3-regression.txt
  ```

  **Commit**: YES (grouped with Task 1 and Task 2 commits)
  - Message: included in respective fix commits
  - Files: `tests/unit/test_tdnv.py`, `tests/unit/test_extract.py`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [x] F1. **Full Quality Gate** — `quick`
  Run all verification commands in sequence:
  ```bash
  uv run ruff check src/ tests/
  uv run ruff format --check src/ tests/
  uv run mypy src/
  uv run pytest
  ```
  All must pass with 0 violations/errors/failures.
  Output: `ruff [PASS] | mypy [PASS] | pytest [N pass / 0 fail] | VERDICT`

---

## Commit Strategy

- **Commit 1**: `fix(tdnv): eliminate double-load in _run_concept CLI path` — `src/steering_geometry/tdnv.py`, `tests/unit/test_tdnv.py`
- **Commit 2**: `perf(extract): add early-stop to sentiment and polite data loaders` — `src/steering_geometry/extract.py`, `tests/unit/test_extract.py`

---

## Success Criteria

### Verification Commands
```bash
uv run ruff check src/ tests/        # Expected: 0 violations
uv run ruff format --check src/ tests/  # Expected: already formatted
uv run mypy src/                      # Expected: Success, 0 errors
uv run pytest                         # Expected: all tests pass
```

### Final Checklist
- [ ] `load_sentiment_data()` stops iterating once `num_pairs * 2` items collected per class
- [ ] `load_polite_data()` stops iterating once `num_pairs * 2` items collected per class
- [ ] `sample_with_seed` still called in both loaders (determinism preserved)
- [ ] `_run_concept()` non-dry-run calls `load_contrast_pairs` exactly once (not twice)
- [ ] `_run_concept()` dry-run still works (loads once, prints count, returns)
- [ ] No function signatures changed
- [ ] `load_refusal_data()` untouched
- [ ] All tests pass
