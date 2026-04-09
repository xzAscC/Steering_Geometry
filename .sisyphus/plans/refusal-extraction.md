# Refusal Steering Vector Extraction — Dual-Dataset + Token Selection Strategies

## TL;DR

> **Quick Summary**: Replace the refusal steering vector extraction to use a dual-dataset approach (LLM-LAT/benign-dataset as positive class, LLM-LAT/harmful-dataset as negative class), with 4 configurable token selection strategies (prompt-only/prompt+response × all-tokens/last-N-tokens).
>
> **Deliverables**:
> - Extended `select_token_activations()` supporting "all" and "last_n" modes
> - New `ExtractionConfig` fields: `data_mode`, `token_select`, `last_n`, `seed`
> - Replaced `load_refusal_data()` with dual-dataset loader
> - Updated `extract_steering_vector()` to dispatch token selection by mode
> - New CLI arguments: `--data-mode`, `--token-select`, `--last-n`, `--seed`
> - Full TDD test coverage for all new functionality
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 2 waves + 1 integration wave + final verification
> **Critical Path**: Task 1 (select_token_activations) → Task 4 (extraction wiring) → Task 5 (CLI) → Task 6 (integration tests) → Final Verification

---

## Context

### Original Request
Replace the refusal concept's single-dataset extraction with a dual-dataset approach using LLM-LAT/benign-dataset (positive) and LLM-LAT/harmful-dataset (negative). Add two data modes (prompt-only, prompt+response) and two token selection strategies (all tokens, last N tokens) as independent CLI-configurable options.

### Interview Summary
**Key Discussions**:
- Data sources: positive=benign-dataset, negative=harmful-dataset (user corrected from initial or-bench suggestion)
- Prompt+Response composition: harmful uses `prompt+rejected`, benign uses `prompt+response`
- Last N tokens: from the entire sequence (not just response part)
- All tokens: keep (num_tokens, hidden_dim) shape, let aggregator handle
- CLI: independent configurable parameters, not batch experiment mode
- Compatibility: completely replace existing refusal loader (no backward compat)
- Test strategy: TDD (RED→GREEN→REFACTOR)

**Research Findings**:
- Prompt-only + last token is canonical (Arditi et al., NeurIPS 2024) — captures refusal "intent"
- Prompt+response is teacher-forced extraction (CR-VLM) — captures refusal "execution", stronger signal
- `select_token_activations()` always returns (batch, hidden_dim) — extending to multi-token output is the critical design decision
- Benign dataset: 165,298 rows, columns: `prompt`, `response`, `refusal` (useless — 1 unique value)
- Harmful dataset: 4,948 rows, columns: `prompt`, `chosen` (refusal), `rejected` (compliance)

### Metis Review
**Identified Gaps** (addressed):
- Shape invariant breakage: "all tokens" produces variable-length output → resolved: flatten all tokens into (total_tokens, hidden_dim) before aggregator
- `read_token_index` coexistence: keep for other concepts, new params for refusal only → resolved: new fields added alongside, not replacing
- Benign dataset quality: `refusal` column is useless, some rows have refusal as response → resolved: filter out rows where `response` matches refusal template
- Pairing strategy: random pairing with configurable seed → resolved: use `sample_with_seed()` pattern
- Padding contamination: "all tokens" must exclude padding → resolved: use attention mask to filter

---

## Work Objectives

### Core Objective
Replace the refusal steering vector extraction with a dual-dataset approach supporting 4 token selection strategies, following TDD methodology.

### Concrete Deliverables
- `src/steering_geometry/utils.py`: Extended `select_token_activations()` with "all" and "last_n" modes
- `src/steering_geometry/config.py`: New `ExtractionConfig` fields (`data_mode`, `token_select`, `last_n`, `seed`)
- `src/steering_geometry/extract.py`: Replaced `load_refusal_data()`, extended `extract_steering_vector()`, new CLI args
- `tests/unit/test_utils.py`: Tests for new token selection modes
- `tests/unit/test_extract.py`: Updated refusal tests, config tests, integration tests

### Definition of Done
- [x] `uv run ruff check src/ tests/` → 0 violations
- [x] `uv run ruff format --check src/ tests/` → formatted
- [x] `uv run mypy src/` → 0 errors
- [x] `uv run pytest` → all pass (including new tests)
- [x] CLI `--concept refusal --dry-run` works with all 4 strategy combinations

### Must Have
- Dual-dataset loader: positive=benign, negative=harmful, configurable `num_pairs`
- Prompt-only mode: uses only prompt text from both datasets
- Prompt+Response mode: benign uses `prompt+response`, harmful uses `prompt+rejected`
- "all tokens" selection: returns all non-padding token activations, flattened to (total_tokens, hidden_dim)
- "last_n tokens" selection: returns last N non-padding tokens from full sequence, flattened to (total_tokens, hidden_dim)
- Graceful degradation: `last_n > seq_len` → return all available tokens (no crash)
- CLI args: `--data-mode {prompt_only,prompt_response}`, `--token-select {all,last_n}`, `--last-n N`, `--seed N`
- Deterministic subsampling via configurable seed (default 42)
- Filter benign rows where `response` matches the refusal template
- All aggregators (mean, pca, weighted_mean, discriminative) work with new token selection modes
- Other concepts (sentiment, polite) completely unaffected

### Must NOT Have (Guardrails)
- Do NOT modify `read_token_index` field or its behavior — other concepts use it
- Do NOT change aggregator signatures `(pos: Tensor, neg: Tensor) -> Tensor`
- Do NOT modify `stability_comparison.py`, `tdnv.py`, `apply_steering.py`, `token_analysis.py`
- Do NOT change `SteeringVector` or `ContrastPair` type definitions
- Do NOT add new dependencies — use only `datasets`, `torch`, existing stdlib
- Do NOT introduce dataset caching or local download logic
- Do NOT process both datasets in a single `load_dataset` call — different schemas
- Do NOT use `_REFUSAL_PREFIX` / `_COMPLIANCE_PREFIX` — delete these dead code constants
- Do NOT change other concept loaders (sentiment, polite)
- Do NOT change default behavior for sentiment/polite extraction

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest, 21+ tests)
- **Automated tests**: YES (TDD)
- **Framework**: pytest
- **TDD**: Each task follows RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/Module**: Use Bash (pytest) — run specific tests, assert pass/fail
- **CLI**: Use Bash — run CLI commands, assert exit code + output content

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation, NO dependencies):
├── Task 1: Extend select_token_activations() [deep]
│   Depends: nothing
│   Files: utils.py, tests/unit/test_utils.py
└── Task 2: Extend ExtractionConfig [quick]
    Depends: nothing
    Files: config.py, tests/unit/test_extract.py (config section)

Wave 2 (After Wave 1 — core logic):
├── Task 3: Replace load_refusal_data() with dual-dataset loader [deep]
│   Depends: Task 2 (config fields)
│   Files: extract.py, tests/unit/test_extract.py
└── Task 4: Wire token selection dispatch in extract_steering_vector() [unspecified-high]
    Depends: Task 1 (select_token_activations), Task 2 (config fields)
    Files: extract.py, tests/unit/test_extract.py

Wave 3 (After Wave 2 — CLI + integration):
├── Task 5: Add CLI arguments and update main() [quick]
│   Depends: Task 2, Task 3
│   Files: extract.py
└── Task 6: End-to-end integration tests + dead code cleanup [unspecified-high]
    Depends: Task 3, Task 4, Task 5
    Files: extract.py, tests/unit/test_extract.py

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay

Critical Path: T1 → T4 → T6 → F1-F4 → user okay
Parallel Speedup: ~50% faster than sequential (Wave 1 fully parallel, Wave 2 mostly parallel)
Max Concurrent: 2 (Waves 1 & 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| T1 | — | T4, T6 | 1 |
| T2 | — | T3, T4, T5 | 1 |
| T3 | T2 | T5, T6 | 2 |
| T4 | T1, T2 | T6 | 2 |
| T5 | T2, T3 | T6 | 3 |
| T6 | T3, T4, T5 | F1-F4 | 3 |
| F1 | T6 | user okay | FINAL |
| F2 | T6 | user okay | FINAL |
| F3 | T6 | user okay | FINAL |
| F4 | T6 | user okay | FINAL |

### Agent Dispatch Summary

- **Wave 1**: 2 tasks — T1 → `deep`, T2 → `quick`
- **Wave 2**: 2 tasks — T3 → `deep`, T4 → `unspecified-high`
- **Wave 3**: 2 tasks — T5 → `quick`, T6 → `unspecified-high`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Extend `select_token_activations()` with "all" and "last_n" modes

  **What to do** (TDD — RED first):
  - **RED**: Write failing tests in `tests/unit/test_utils.py` for:
    - `select_token_activations(activations_3d, "all")` → returns all non-padding token activations flattened to shape (total_non_padding_tokens, hidden_dim)
    - `select_token_activations(activations_3d, "last_n", last_n=5)` → returns last 5 non-padding tokens per sample, flattened to (batch*5, hidden_dim)
    - `select_token_activations(activations_3d, "last_n", last_n=100)` where seq_len < 100 → graceful degradation, returns all available tokens
    - `select_token_activations(activations_2d, "all")` → returns as-is (2D passthrough)
    - Existing int-index behavior still works: `select_token_activations(activations_3d, -1)` → (batch, hidden_dim)
  - **GREEN**: Implement the new modes in `utils.py`:
    - Change `read_token_index` parameter type from `int` to `int | str` (accept "all", "last_n")
    - Add optional `last_n: int | None = None` parameter
    - "all" mode: use attention mask (non-zero activations) to collect all real tokens, reshape to (total_real_tokens, hidden_dim)
    - "last_n" mode: for each sample, find last N non-padding tokens (where N = min(last_n, num_real_tokens)), collect, reshape to (total_tokens, hidden_dim)
    - Preserve existing int-index behavior exactly (backward compat for sentiment/polite)
  - **REFACTOR**: Clean up, ensure no code duplication between modes

  **Must NOT do**:
  - Do NOT change the existing int-index behavior
  - Do NOT change the function signature in a breaking way (use Union type for backward compat)
  - Do NOT add padding tokens to the output — only real content tokens
  - Do NOT modify any other file besides utils.py and tests/unit/test_utils.py

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Non-trivial tensor manipulation with shape invariants and multiple code paths
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 4, Task 6
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `src/steering_geometry/utils.py:82-113` — Current `select_token_activations()` implementation. The "all" and "last_n" modes must be added alongside the existing int-index logic. Study the padding detection logic (lines 103-107) — reuse the `non_zero_mask` pattern for filtering padding tokens in the new modes.
  - `src/steering_geometry/utils.py:96-99` — 2D passthrough case. "all" mode on 2D input should behave the same.

  **API/Type References**:
  - `src/steering_geometry/utils.py:82` — Current signature: `select_token_activations(activations: Tensor, read_token_index: int) -> Tensor`. Extend `read_token_index` to accept `int | str` where str is "all" or "last_n".

  **Test References**:
  - `tests/unit/test_extract.py` — Existing test patterns for reference (fixture style, assertion patterns)
  - `tests/conftest.py` — Existing fixtures for mock tensors

  **WHY Each Reference Matters**:
  - `utils.py:82-113` — This IS the function being extended. Every new mode must coexist with existing logic.
  - `utils.py:103-107` — The non-zero mask pattern is the canonical way to detect padding tokens. Reuse it.

  **Acceptance Criteria**:

  **If TDD (tests enabled)**:
  - [ ] Test file created/updated: tests/unit/test_utils.py
  - [ ] `uv run pytest tests/unit/test_utils.py -k "select_token" -v` → PASS (5+ tests, 0 failures)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: "all" mode returns all non-padding tokens flattened
    Tool: Bash (pytest)
    Preconditions: utils.py has "all" mode implemented
    Steps:
      1. Run: uv run pytest tests/unit/test_utils.py -k "test_select_token_all" -v
      2. Assert: test passes with correct shape assertion
      3. Verify the test creates a (2, 5, 8) tensor with padding at positions 3-4
      4. Expected output shape: (6, 8) — 2 samples × 3 real tokens each
    Expected Result: PASS, shape (6, 8)
    Failure Indicators: Shape mismatch, padding tokens included, test FAIL
    Evidence: .sisyphus/evidence/task-1-select-all.txt

  Scenario: "last_n" mode returns last N real tokens per sample
    Tool: Bash (pytest)
    Preconditions: utils.py has "last_n" mode implemented
    Steps:
      1. Run: uv run pytest tests/unit/test_utils.py -k "test_select_token_last_n" -v
      2. Assert: test passes with correct shape
      3. Create (2, 5, 8) tensor, last_n=2 → expect (4, 8) shape
    Expected Result: PASS, shape (4, 8)
    Failure Indicators: Shape mismatch, padding tokens included
    Evidence: .sisyphus/evidence/task-1-select-last-n.txt

  Scenario: "last_n" with last_n > seq_len — graceful degradation
    Tool: Bash (pytest)
    Preconditions: "last_n" mode handles overflow
    Steps:
      1. Run: uv run pytest tests/unit/test_utils.py -k "test_select_token_last_n_overflow" -v
      2. Create (2, 3, 8) tensor with all real tokens, last_n=10
      3. Expected: returns (6, 8) — all 3 tokens per sample, no crash
    Expected Result: PASS, no exception, returns all available tokens
    Failure Indicators: IndexError, crash, test FAIL
    Evidence: .sisyphus/evidence/task-1-select-overflow.txt

  Scenario: Existing int-index mode still works (backward compat)
    Tool: Bash (pytest)
    Preconditions: No regression in existing behavior
    Steps:
      1. Run: uv run pytest tests/unit/test_utils.py -k "test_select_token_index" -v
      2. Call select_token_activations(tensor_3d, -1) — same as before
      3. Assert: returns (batch, hidden_dim) with last non-padding token per sample
    Expected Result: PASS, existing tests green
    Failure Indicators: Regression, shape change, test FAIL
    Evidence: .sisyphus/evidence/task-1-select-backward.txt
  ```

  **Evidence to Capture**:
  - [ ] task-1-select-all.txt
  - [ ] task-1-select-last-n.txt
  - [ ] task-1-select-overflow.txt
  - [ ] task-1-select-backward.txt

  **Commit**: YES (commit C1)
  - Message: `feat(utils): add "all" and "last_n" token selection modes`
  - Files: `src/steering_geometry/utils.py`, `tests/unit/test_utils.py`
  - Pre-commit: `uv run pytest tests/unit/test_utils.py`

- [x] 2. Extend `ExtractionConfig` with new fields

  **What to do** (TDD — RED first):
  - **RED**: Write failing tests in `tests/unit/test_extract.py` for:
    - `ExtractionConfig()` defaults: `data_mode="prompt_only"`, `token_select="all"`, `last_n=1`, `seed=42`
    - `ExtractionConfig(data_mode="prompt_response")` accepted
    - `ExtractionConfig(token_select="last_n")` accepted
    - Invalid values rejected or documented
  - **GREEN**: Add fields to `ExtractionConfig` in `config.py`:
    - `data_mode: str = "prompt_only"` — choices: "prompt_only", "prompt_response"
    - `token_select: str = "all"` — choices: "all", "last_n"
    - `last_n: int = 1` — number of tokens for "last_n" mode
    - `seed: int = 42` — for deterministic subsampling
    - Keep all existing fields unchanged (`read_token_index`, `layers`, `method`, `batch_size`, `top_k`)
  - **REFACTOR**: Minimal — this is a dataclass addition

  **Must NOT do**:
  - Do NOT remove or rename `read_token_index` — other concepts use it
  - Do NOT change defaults for existing fields
  - Do NOT add validation logic in `__post_init__` (keep it simple, validate at usage site)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dataclass field addition, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3, Task 4, Task 5
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/steering_geometry/config.py:47-64` — Current `ExtractionConfig` dataclass. Add new fields after existing ones, same pattern (type annotation + default value).

  **Test References**:
  - `tests/unit/test_extract.py` — Existing config tests for patterns

  **WHY Each Reference Matters**:
  - `config.py:47-64` — This IS the class being extended. Follow the exact same field declaration pattern.

  **Acceptance Criteria**:

  **If TDD**:
  - [ ] Tests updated: tests/unit/test_extract.py (config section)
  - [ ] `uv run pytest tests/unit/test_extract.py -k "extraction_config" -v` → PASS

  **QA Scenarios:**

  ```
  Scenario: ExtractionConfig has new fields with correct defaults
    Tool: Bash (pytest)
    Preconditions: config.py updated
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_extraction_config_defaults" -v
      2. Assert: data_mode == "prompt_only", token_select == "all", last_n == 1, seed == 42
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-2-config-defaults.txt

  Scenario: ExtractionConfig accepts all valid combinations
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_extraction_config_values" -v
      2. Test ExtractionConfig(data_mode="prompt_response", token_select="last_n", last_n=10, seed=123)
      3. Assert: all fields set correctly
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-2-config-values.txt
  ```

  **Evidence to Capture**:
  - [ ] task-2-config-defaults.txt
  - [ ] task-2-config-values.txt

  **Commit**: YES (commit C2)
  - Message: `feat(config): add data_mode, token_select, last_n, seed to ExtractionConfig`
  - Files: `src/steering_geometry/config.py`, `tests/unit/test_extract.py`
  - Pre-commit: `uv run pytest tests/unit/test_extract.py -k "config"`

- [x] 3. Replace `load_refusal_data()` with dual-dataset loader

  **What to do** (TDD — RED first):
  - **RED**: Write failing tests for new `load_refusal_data()`:
    - Loads from BOTH `LLM-LAT/benign-dataset` and `LLM-LAT/harmful-dataset` (use mock/patch for `load_dataset`)
    - Positive texts come from benign-dataset, negative from harmful-dataset
    - Prompt-only mode (`data_mode="prompt_only"`): positive = benign `prompt`, negative = harmful `prompt`
    - Prompt+Response mode (`data_mode="prompt_response"`): positive = benign `prompt` + `response`, negative = harmful `prompt` + `rejected`
    - `num_pairs` caps at min(len(benign_filtered), len(harmful))
    - Deterministic output with same seed (run twice, assert identical pairs)
    - Filters benign rows where `response` matches the refusal template or is empty
    - ContrastPairMetadata records `concept="refusal"`, `dataset="dual"`, `source` includes both dataset names
  - **GREEN**: Implement new `load_refusal_data()`:
    - Load both datasets separately via `load_dataset("LLM-LAT/benign-dataset")` and `load_dataset("LLM-LAT/harmful-dataset")`
    - Filter benign: skip rows where `response` is empty or matches the `refusal` column value
    - Subsample benign to `num_pairs` using `sample_with_seed(seed=...)`
    - Subsample harmful to `num_pairs` using `sample_with_seed(seed=...)`
    - For prompt-only mode: `positive = benign_prompt`, `negative = harmful_prompt`
    - For prompt+response mode: `positive = f"{benign_prompt}\n{benign_response}"`, `negative = f"{harmful_prompt}\n{harmful_rejected}"`
    - Build ContrastPair objects with proper metadata
  - **REFACTOR**: Extract helper for filtering/subsampling if needed

  **Must NOT do**:
  - Do NOT use `_REFUSAL_PREFIX` or `_COMPLIANCE_PREFIX` — delete these constants
  - Do NOT use the `chosen` column from harmful-dataset (only use `prompt` and `rejected`)
  - Do NOT use the `refusal` column from benign-dataset
  - Do NOT change `_DATASET_LOADERS` registration pattern — new function goes in same `"refusal"` slot
  - Do NOT load both datasets in a single call — different schemas require separate loading
  - Do NOT modify `load_contrast_pairs()` function signature

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex data loading logic with filtering, subsampling, and dual-dataset handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 5, Task 6
  - **Blocked By**: Task 2 (needs ExtractionConfig fields)

  **References**:

  **Pattern References**:
  - `src/steering_geometry/extract.py:192-241` — `load_sentiment_data()` pattern: oversample, collect positives/negatives, `sample_with_seed()`, build ContrastPairs. Follow THIS pattern (not the old refusal sequential pattern).
  - `src/steering_geometry/extract.py:296-331` — OLD `load_refusal_data()` being replaced. Note the sequential pattern (no shuffling). The new loader must use the sentiment-style pattern instead.
  - `src/steering_geometry/extract.py:334-339` — `_DATASET_LOADERS` registry. The new loader function replaces the old one at key `"refusal"`.
  - `src/steering_geometry/extract.py:64-65` — `_REFUSAL_PREFIX` and `_COMPLIANCE_PREFIX` constants to DELETE.

  **API/Type References**:
  - `src/steering_geometry/types.py:13-28` — `ContrastPairMetadata` TypedDict. Use `concept="refusal"`, `dataset="dual"`, `source="LLM-LAT/benign-dataset+LLM-LAT/harmful-dataset"`.
  - `src/steering_geometry/types.py:79-93` — `ContrastPair` dataclass. positive (str), negative (str), metadata (ContrastPairMetadata).
  - `src/steering_geometry/config.py:67-79` — `ConceptConfig` dataclass. Has `num_pairs` field.

  **Test References**:
  - `tests/unit/test_extract.py:83-95` — OLD `test_load_refusal_data()` that will be REPLACED. Note the pattern: it asserts on text content. New tests should mock `load_dataset` and assert on data flow.

  **External References**:
  - LLM-LAT/benign-dataset: columns `prompt`(str), `response`(str), `refusal`(str — 1 unique value, useless)
  - LLM-LAT/harmful-dataset: columns `prompt`(str), `chosen`(str — refusal), `rejected`(str — compliance)

  **WHY Each Reference Matters**:
  - `extract.py:192-241` — The sentiment loader IS the canonical pattern for this project. Follow it exactly.
  - `extract.py:64-65` — These constants MUST be deleted as part of this task.
  - `types.py:13-28` — Metadata fields need to accurately reflect dual-dataset origin.

  **Acceptance Criteria**:

  **If TDD**:
  - [ ] Tests updated: tests/unit/test_extract.py (refusal section replaced)
  - [ ] `uv run pytest tests/unit/test_extract.py -k "refusal" -v` → PASS (6+ tests)

  **QA Scenarios:**

  ```
  Scenario: Dual-dataset loader produces correct contrast pairs
    Tool: Bash (pytest)
    Preconditions: load_refusal_data rewritten, tests use mocked load_dataset
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_load_refusal_dual" -v
      2. Mock benign-dataset with 3 rows, harmful-dataset with 3 rows
      3. Assert: 3 ContrastPairs returned, positive from benign, negative from harmful
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-3-dual-loader.txt

  Scenario: Prompt-only mode uses only prompt columns
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_load_refusal_prompt_only" -v
      2. Mock data with known prompts and responses
      3. Assert: positive == benign_prompt (no response appended)
      4. Assert: negative == harmful_prompt (no rejected appended)
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-3-prompt-only.txt

  Scenario: Prompt+Response mode concatenates correctly
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_load_refusal_prompt_response" -v
      2. Mock data with known values
      3. Assert: positive == "benign_prompt\nbenign_response"
      4. Assert: negative == "harmful_prompt\nharmful_rejected"
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-3-prompt-response.txt

  Scenario: Filters benign rows with refusal-template responses
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_load_refusal_filter" -v
      2. Mock benign data where row[2].response == row[2].refusal
      3. Assert: filtered row is excluded, only genuine responses kept
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-3-filter.txt

  Scenario: num_pairs caps at min dataset size
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_load_refusal_cap" -v
      2. Request num_pairs=10000 but mock only 5 harmful rows
      3. Assert: returns 5 pairs (capped at harmful dataset size)
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-3-cap.txt

  Scenario: Deterministic with seed
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_load_refusal_seed" -v
      2. Call load_refusal_data twice with same seed
      3. Assert: identical pairs in same order
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-3-seed.txt
  ```

  **Evidence to Capture**:
  - [ ] task-3-dual-loader.txt
  - [ ] task-3-prompt-only.txt
  - [ ] task-3-prompt-response.txt
  - [ ] task-3-filter.txt
  - [ ] task-3-cap.txt
  - [ ] task-3-seed.txt

  **Commit**: YES (commit C3)
  - Message: `feat(extract): replace refusal loader with dual-dataset approach`
  - Files: `src/steering_geometry/extract.py`, `tests/unit/test_extract.py`
  - Pre-commit: `uv run pytest tests/unit/test_extract.py -k "refusal"`

- [x] 4. Wire token selection dispatch in `extract_steering_vector()`

  **What to do** (TDD — RED first):
  - **RED**: Write failing tests for the extraction pipeline with new token selection:
    - `extract_steering_vector()` with `config.token_select="all"` → calls `select_token_activations(act, "all")`, feeds flattened result to aggregator
    - `extract_steering_vector()` with `config.token_select="last_n"` and `config.last_n=5` → calls `select_token_activations(act, "last_n", last_n=5)`, feeds flattened result
    - `extract_steering_vector()` with `config.token_select="all"` and mock model → produces valid `SteeringVector`
    - Shape invariant: aggregator always receives 2D tensor (N, hidden_dim) regardless of token selection mode
  - **GREEN**: Modify `extract_steering_vector()` in `extract.py`:
    - In the per-batch loop (lines ~410-424), dispatch `select_token_activations` based on `config.token_select`:
      - If `token_select == "all"`: call `select_token_activations(act, "all")`
      - If `token_select == "last_n"`: call `select_token_activations(act, "last_n", last_n=config.last_n)`
      - Otherwise (default for backward compat): call `select_token_activations(act, config.read_token_index)` (existing behavior)
    - The result from "all" and "last_n" is already (N, hidden_dim) flattened — accumulates normally via `torch.cat`
    - Aggregators receive the same (N, hidden_dim) shape as before — interface unchanged
  - **REFACTOR**: Ensure dispatch logic is clean, no duplication

  **Must NOT do**:
  - Do NOT change aggregator signatures or their dispatch
  - Do NOT change the batch accumulation pattern (`torch.cat`)
  - Do NOT modify how other concepts flow through this function
  - Do NOT break the default path (when `token_select` not explicitly set)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding of the full extraction pipeline and careful dispatch wiring
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 3)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6
  - **Blocked By**: Task 1 (select_token_activations modes), Task 2 (config fields)

  **References**:

  **Pattern References**:
  - `src/steering_geometry/extract.py:374-440` — `extract_steering_vector()` function. The token selection dispatch goes at lines 415-422 where `select_token_activations` is currently called.
  - `src/steering_geometry/extract.py:415-422` — Current calls: `select_token_activations(positive_activations[layer], config.read_token_index)`. This is where dispatch logic is added.

  **API/Type References**:
  - `src/steering_geometry/utils.py:82-113` — Updated `select_token_activations()` from Task 1. The new modes "all" and "last_n" return flattened (N, hidden_dim) tensors.

  **Test References**:
  - `tests/unit/test_extract.py` — Existing extraction tests use `mock_hooked_model` fixture from `tests/conftest.py`

  **WHY Each Reference Matters**:
  - `extract.py:415-422` — This is the exact insertion point for the dispatch logic. The current code calls select_token_activations with int index; the new code dispatches based on config.token_select.
  - `conftest.py` — The `mock_hooked_model` fixture must be used in tests to avoid loading real models.

  **Acceptance Criteria**:

  **If TDD**:
  - [ ] Tests updated: tests/unit/test_extract.py (extraction section)
  - [ ] `uv run pytest tests/unit/test_extract.py -k "token_select" -v` → PASS (4+ tests)

  **QA Scenarios:**

  ```
  Scenario: "all" token selection in extraction pipeline
    Tool: Bash (pytest)
    Preconditions: Task 1 complete, select_token_activations has "all" mode
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_extract_token_all" -v
      2. Use mock_hooked_model, config with token_select="all"
      3. Assert: extract_steering_vector returns valid SteeringVector
      4. Assert: aggregator received 2D input (N, hidden_dim)
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-4-extract-all.txt

  Scenario: "last_n" token selection in extraction pipeline
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_extract_token_last_n" -v
      2. Use mock_hooked_model, config with token_select="last_n", last_n=3
      3. Assert: returns valid SteeringVector
      4. Assert: aggregator received 2D input
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-4-extract-last-n.txt

  Scenario: Default backward compat — existing concepts unaffected
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "sentiment or polite" -v
      2. These use default config (no token_select override)
      3. Assert: all still pass
    Expected Result: PASS — no regression
    Evidence: .sisyphus/evidence/task-4-backward.txt

  Scenario: Error on invalid token_select value
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_extract_invalid_token_select" -v
      2. Set config.token_select="invalid"
      3. Assert: raises ValueError
    Expected Result: PASS — proper error handling
    Evidence: .sisyphus/evidence/task-4-invalid.txt
  ```

  **Evidence to Capture**:
  - [ ] task-4-extract-all.txt
  - [ ] task-4-extract-last-n.txt
  - [ ] task-4-backward.txt
  - [ ] task-4-invalid.txt

  **Commit**: YES (commit C4)
  - Message: `feat(extract): wire token selection dispatch in extract_steering_vector`
  - Files: `src/steering_geometry/extract.py`, `tests/unit/test_extract.py`
  - Pre-commit: `uv run pytest tests/unit/test_extract.py -k "token_select"`

- [x] 5. Add CLI arguments and update `main()`

  **What to do**:
  - Add CLI arguments to `_build_parser()` in `extract.py`:
    - `--data-mode`: choices=["prompt_only", "prompt_response"], default="prompt_only"
    - `--token-select`: choices=["all", "last_n"], default="all"
    - `--last-n`: type=int, default=1, help="Number of tokens for last_n mode"
    - `--seed`: type=int, default=42, help="Random seed for subsampling"
  - Update `_Args` Protocol with new fields
  - Update `main()` to pass new args into `ExtractionConfig` and `ConceptConfig`
  - Add validation: if `--token-select last_n` then `--last-n` must be provided and > 0
  - Print the data_mode and token_select in the statistics output

  **Must NOT do**:
  - Do NOT change existing CLI arguments (--concept, --model, --method, --num-pairs, --layers, --dry-run)
  - Do NOT change the output format for other concepts
  - Do NOT add argument combinations that conflict with other concepts

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard argparse additions, straightforward wiring
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 6
  - **Blocked By**: Task 2 (config fields), Task 3 (loader needs data_mode)

  **References**:

  **Pattern References**:
  - `src/steering_geometry/extract.py:500-552` — `_build_parser()` function. Add new arguments after existing ones, same pattern (argparse.add_argument).
  - `src/steering_geometry/extract.py:487-498` — `_Args` Protocol. Add new fields matching the argument names.
  - `src/steering_geometry/extract.py:555-596` — `main()` function. Update where `ExtractionConfig` is constructed (line 577-581) to include new fields.

  **WHY Each Reference Matters**:
  - `extract.py:500-552` — Exact pattern for adding CLI args. Follow the same style (choices, default, help text).
  - `extract.py:577-581` — This is where ExtractionConfig is built from CLI args. Add new fields here.

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: CLI accepts new arguments with defaults
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept refusal --dry-run --data-mode prompt_only --token-select all
      2. Assert: exit code 0
      3. Output should contain "Loaded N contrast pairs for refusal"
    Expected Result: Exit 0, contrast pairs loaded
    Evidence: .sisyphus/evidence/task-5-cli-defaults.txt

  Scenario: CLI accepts all 4 strategy combinations
    Tool: Bash
    Steps:
      1. Run each combination with --dry-run:
         a. --data-mode prompt_only --token-select all
         b. --data-mode prompt_only --token-select last_n --last-n 5
         c. --data-mode prompt_response --token-select all
         d. --data-mode prompt_response --token-select last_n --last-n 10
      2. Assert: all exit 0
    Expected Result: All 4 exit 0
    Evidence: .sisyphus/evidence/task-5-cli-combos.txt

  Scenario: CLI --help shows new arguments
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.extract --help
      2. Assert: output contains "--data-mode", "--token-select", "--last-n", "--seed"
    Expected Result: Help text includes all new args
    Evidence: .sisyphus/evidence/task-5-cli-help.txt

  Scenario: Other concepts still work with default CLI
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.extract --concept sentiment --dry-run
      2. Assert: exit code 0, loads sentiment data
    Expected Result: Exit 0, no regression
    Evidence: .sisyphus/evidence/task-5-cli-sentiment.txt
  ```

  **Evidence to Capture**:
  - [ ] task-5-cli-defaults.txt
  - [ ] task-5-cli-combos.txt
  - [ ] task-5-cli-help.txt
  - [ ] task-5-cli-sentiment.txt

  **Commit**: YES (commit C5)
  - Message: `feat(extract): add CLI args for data-mode, token-select, last-n, seed`
  - Files: `src/steering_geometry/extract.py`
  - Pre-commit: `uv run python -m steering_geometry.extract --concept refusal --dry-run --data-mode prompt_only --token-select all`

- [x] 6. End-to-end integration tests + dead code cleanup

  **What to do**:
  - **Integration tests**: Write tests that exercise the full pipeline (loader → extraction → SteeringVector) with all 4 combinations:
    - prompt_only + all tokens
    - prompt_only + last_n tokens
    - prompt_response + all tokens
    - prompt_response + last_n tokens
    - Each test uses `mock_hooked_model` fixture, verifies valid SteeringVector output
  - **Dead code cleanup**:
    - Delete `_REFUSAL_PREFIX` constant (extract.py line 64)
    - Delete `_COMPLIANCE_PREFIX` constant (extract.py line 65)
    - Verify no remaining references to deleted constants
  - **Update old test**: Replace `test_load_refusal_data()` (test_extract.py:83-95) with new dual-dataset tests
  - **Full regression**: Run complete test suite, ensure all pass
  - **Update `__all__` exports** if any public API changed

  **Must NOT do**:
  - Do NOT delete `_DATASET_LOADERS` registry — still needed for dispatch
  - Do NOT modify `load_contrast_pairs()` function signature
  - Do NOT change `extract_vector()` high-level API signature (it still works for other concepts)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration testing requires understanding of the full pipeline + careful dead code removal
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential after T5)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 3, Task 4, Task 5

  **References**:

  **Pattern References**:
  - `tests/conftest.py` — `mock_hooked_model` fixture for testing without real model
  - `tests/unit/test_extract.py:83-95` — OLD test to replace

  **API/Type References**:
  - `src/steering_geometry/extract.py:607-620` — `__all__` exports. Remove `load_refusal_data` if no longer public, or keep if still exported.

  **WHY Each Reference Matters**:
  - `conftest.py` — The mock fixture is essential for integration tests without GPU.
  - `extract.py:607-620` — Need to verify `__all__` is consistent with actual public API.

  **Acceptance Criteria**:

  **QA Scenarios:**

  ```
  Scenario: Full integration — all 4 strategy combinations produce valid SteeringVector
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_extract.py -k "test_integration_refusal" -v
      2. Test all 4 combinations with mock_hooked_model
      3. Assert: each returns SteeringVector with correct concept="refusal"
      4. Assert: layer_activations dict has expected layer keys
      5. Assert: each layer tensor is 1D (hidden_dim,)
    Expected Result: PASS (4 tests, one per combination)
    Evidence: .sisyphus/evidence/task-6-integration.txt

  Scenario: Dead code removed — no references to deleted constants
    Tool: Bash
    Steps:
      1. Run: grep -r "_REFUSAL_PREFIX\|_COMPLIANCE_PREFIX" src/
      2. Assert: no matches (constants deleted)
    Expected Result: Empty grep output
    Evidence: .sisyphus/evidence/task-6-dead-code.txt

  Scenario: Full test suite passes
    Tool: Bash
    Steps:
      1. Run: uv run pytest -v
      2. Assert: 0 failures, 0 errors
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-6-full-suite.txt

  Scenario: Lint + type check clean
    Tool: Bash
    Steps:
      1. Run: uv run ruff check src/ tests/
      2. Run: uv run ruff format --check src/ tests/
      3. Run: uv run mypy src/
      4. Assert: 0 violations, formatted, 0 errors
    Expected Result: All clean
    Evidence: .sisyphus/evidence/task-6-lint.txt
  ```

  **Evidence to Capture**:
  - [ ] task-6-integration.txt
  - [ ] task-6-dead-code.txt
  - [ ] task-6-full-suite.txt
  - [ ] task-6-lint.txt

  **Commit**: YES (commit C6)
  - Message: `test(extract): end-to-end integration tests + dead code cleanup`
  - Files: `src/steering_geometry/extract.py`, `tests/unit/test_extract.py`
  - Pre-commit: `uv run pytest`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Review all changed files for: `Any`/`type: ignore`, empty catches, print() in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (features working together, not isolation). Test edge cases: empty response, invalid arg combos, last_n > seq_len. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Commit | Message | Files | Pre-commit |
|--------|---------|-------|------------|
| C1 | `feat(utils): add "all" and "last_n" token selection modes` | `utils.py`, `tests/unit/test_utils.py` | `uv run pytest tests/unit/test_utils.py` |
| C2 | `feat(config): add data_mode, token_select, last_n, seed to ExtractionConfig` | `config.py`, `tests/unit/test_extract.py` (config section) | `uv run pytest tests/unit/test_extract.py -k "config"` |
| C3 | `feat(extract): replace refusal loader with dual-dataset approach` | `extract.py`, `tests/unit/test_extract.py` | `uv run pytest tests/unit/test_extract.py -k "refusal"` |
| C4 | `feat(extract): wire token selection dispatch in extract_steering_vector` | `extract.py`, `tests/unit/test_extract.py` | `uv run pytest tests/unit/test_extract.py -k "token_select"` |
| C5 | `feat(extract): add CLI args for data-mode, token-select, last-n, seed` | `extract.py` | `uv run python -m steering_geometry.extract --concept refusal --dry-run --data-mode prompt_only --token-select all` |
| C6 | `test(extract): end-to-end integration tests + dead code cleanup` | `extract.py`, `tests/unit/test_extract.py` | `uv run pytest` |

---

## Success Criteria

### Verification Commands
```bash
uv run ruff check src/ tests/          # Expected: 0 violations
uv run ruff format --check src/ tests/  # Expected: already formatted
uv run mypy src/                        # Expected: Success, 0 errors
uv run pytest                           # Expected: all pass
```

### CLI Smoke Tests
```bash
uv run python -m steering_geometry.extract --concept refusal --dry-run --data-mode prompt_only --token-select all
uv run python -m steering_geometry.extract --concept refusal --dry-run --data-mode prompt_only --token-select last_n --last-n 5
uv run python -m steering_geometry.extract --concept refusal --dry-run --data-mode prompt_response --token-select all
uv run python -m steering_geometry.extract --concept refusal --dry-run --data-mode prompt_response --token-select last_n --last-n 10
# All should exit 0 and print contrast pair statistics
```

### Backward Compatibility
```bash
uv run pytest tests/unit/test_extract.py -k "sentiment or polite" -v
# Expected: all pass (other concepts unaffected)
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] All tests pass
- [x] `_REFUSAL_PREFIX` and `_COMPLIANCE_PREFIX` constants deleted
- [x] Old `load_refusal_data()` completely replaced
- [x] CLI `--help` shows new arguments
