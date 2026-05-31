# Add Local LLM-as-Judge Support via CLI Extension

## TL;DR

> **Quick Summary**: Add `--judge-api-base` CLI option to enable using local vLLM servers (e.g., Qwen 3.5 9B) as LLM-as-judge, extending the existing `JudgeEvaluator` instead of creating a new class.
>
> **Deliverables**:
> - `--judge-api-base` CLI argument in `apply_steering.py`
> - `--judge-api-base` option in `quick_pipeline.sh`
> - Tests for custom API base handling
> - Helpful error messages for connection failures
>
> **Estimated Effort**: Quick
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 3 → Task 5 → Tests

---

## Context

### Original Request
User wants to use local Qwen 3.5 9B (served via vLLM at `http://localhost:8000/v1`) as the LLM-as-judge in Step 3 of `quick_pipeline.sh`, instead of the default cloud-based Gemini model.

### Interview Summary
**Key Discussions**:
- **Server**: vLLM running at `http://localhost:8000/v1` (OpenAI-compatible API)
- **Approach**: Extend existing `JudgeEvaluator` with CLI args (NOT new class) - Metis analysis revealed this is sufficient
- **Test strategy**: TDD with pytest
- **Error handling**: Fail-fast with helpful error message if server not running

**Research Findings**:
- `JudgeEvaluator` already uses `AsyncOpenAI` with configurable `base_url` via `JudgeConfig.api_base`
- Current gap: `--judge-api-base` CLI argument is MISSING from both Python CLI and shell script
- Test patterns: `AsyncMock` + `patch.object` for mocking API calls

### Metis Review
**Identified Gaps** (addressed):
- **Class design**: User confirmed simpler approach (extend CLI, not new class)
- **API key**: vLLM accepts dummy key; will use `"local-vllm"` when localhost detected
- **Error messages**: Must include helpful message with URL and suggestion to start server

---

## Work Objectives

### Core Objective
Enable local vLLM servers as LLM-as-judge by adding `--judge-api-base` CLI option, reusing the existing `JudgeEvaluator` infrastructure.

### Concrete Deliverables
- `src/steering_geometry/apply_steering.py`: New `--judge-api-base` CLI arg + parameter threading
- `scripts/pipeline/quick_pipeline.sh`: New `--judge-api-base` shell option
- `tests/unit/test_evaluation.py`: Tests for custom API base handling

### Definition of Done
- [x] `uv run python -m steering_geometry.apply_steering --help` shows `--judge-api-base`
- [x] `./scripts/pipeline/quick_pipeline.sh --help` shows `--judge-api-base`
- [x] `uv run pytest` passes with new tests
- [x] `uv run mypy src/` passes
- [x] `uv run ruff check src/ tests/` passes

### Must Have
- `--judge-api-base` CLI argument with default `https://openrouter.ai/api/v1`
- Parameter threaded from CLI → `apply_steering()` → `JudgeConfig` → `JudgeEvaluator`
- Helpful connection error messages for local servers

### Must NOT Have (Guardrails)
- NO new `LocalJudgeEvaluator` class
- NO changes to `MMLUEvaluator`
- NO changes to `run_pipeline.sh` or other scripts (only `quick_pipeline.sh`)
- NO additional CLI args not explicitly requested (`--judge-api-key`, `--judge-retries`, etc.)
- NO modifications to default `JudgeEvaluator` behavior with OpenRouter

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (TDD)
- **Framework**: pytest
- **TDD Flow**: Each task follows RED → GREEN → REFACTOR

### QA Policy
Every task includes agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **CLI**: Use Bash — Run commands, check exit codes, verify output
- **Python**: Use Bash (uv run pytest) — Run tests, assert pass/fail
- **Type check**: Use Bash (uv run mypy) — Verify no errors

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — CLI changes):
├── Task 1: Add --judge-api-base to apply_steering.py CLI [quick]
├── Task 2: Thread judge_api_base through apply_steering() function [quick]
└── Task 3: Add --judge-api-base to quick_pipeline.sh [quick]

Wave 2 (After Wave 1 — tests):
├── Task 4: Test JudgeConfig accepts custom api_base [quick]
├── Task 5: Test JudgeEvaluator uses custom api_base [quick]
└── Task 6: Test connection error with helpful message [quick]

Critical Path: Task 1 → Task 2 → Task 4-6 → Verify
Parallel Speedup: ~50% faster than sequential
Max Concurrent: 3 (Wave 1)
```

### Dependency Matrix

- **1**: — — 2, 3, 1
- **2**: 1 — 4, 5, 6, 2
- **3**: — — —
- **4**: 2 — 5, 3
- **5**: 2, 4 — 6, 3
- **6**: 2, 5 — — 3

### Agent Dispatch Summary

- **1**: **3** — T1-T3 → `quick`
- **2**: **3** — T4-T6 → `quick`

---

## TODOs

- [x] 1. Add `--judge-api-base` CLI argument to `apply_steering.py`

  **What to do**:
  - Add `--judge-api-base` argument to `_build_parser()` in `apply_steering.py`
  - Default value: `"https://openrouter.ai/api/v1"`
  - Help text should mention vLLM: `"For local vLLM, use: http://localhost:8000/v1"`
  - Add `judge_api_base: str` to `_Args` Protocol

  **Must NOT do**:
  - Do NOT add `--judge-api-key` or other extra args
  - Do NOT change default behavior for existing users

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple argparse addition, well-defined pattern to follow
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 2 (needs CLI arg defined)
  - **Blocked By**: None

  **References**:
  - `apply_steering.py:769-779` — Existing `--judge-model` arg pattern to follow
  - `apply_steering.py:705-718` — `_Args` Protocol to extend

  **Acceptance Criteria**:
  - [ ] `--judge-api-base` appears in `--help` output
  - [ ] Default is `https://openrouter.ai/api/v1`
  - [ ] Help mentions vLLM example

  **QA Scenarios**:
  ```
  Scenario: CLI help shows new option
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.apply_steering --help
      2. Grep for "judge-api-base"
    Expected Result: Exit code 0, output contains "--judge-api-base"
    Evidence: .omo/evidence/task-1-cli-help.txt
  ```

  **Commit**: NO (groups with Task 3)

- [x] 2. Thread `judge_api_base` through `apply_steering()` to `JudgeConfig`

  **What to do**:
  - Add `judge_api_base: str` parameter to `apply_steering()` function signature
  - Pass `judge_api_base` to `JudgeConfig(model=judge_model, api_base=judge_api_base)`
  - Update `main()` to read `args.judge_api_base` and pass to `apply_steering()`
  - Handle localhost auto-detection: if `api_base` contains "localhost" or "127.0.0.1", use dummy API key

  **Must NOT do**:
  - Do NOT change function signature for unrelated params
  - Do NOT break backward compatibility (default must work)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Parameter threading, straightforward changes
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4, 5, 6 (need this to work for tests)
  - **Blocked By**: Task 1 (needs CLI arg defined)

  **References**:
  - `apply_steering.py:536-544` — Current `apply_steering()` signature
  - `apply_steering.py:640` — Where `JudgeConfig` is created
  - `apply_steering.py:790-805` — `main()` function that calls `apply_steering()`
  - `config.py:102-115` — `JudgeConfig` dataclass with `api_base` field

  **Acceptance Criteria**:
  - [ ] `apply_steering()` accepts `judge_api_base` parameter
  - [ ] `JudgeConfig` receives correct `api_base` value
  - [ ] Localhost URLs use dummy API key

  **QA Scenarios**:
  ```
  Scenario: Localhost uses dummy API key
    Tool: Bash (pytest)
    Steps:
      1. Write test that verifies JudgeEvaluator uses dummy key for localhost
      2. Run: uv run pytest tests/unit/test_evaluation.py -k "test_localhost" -v
    Expected Result: Test passes
    Evidence: .omo/evidence/task-2-localhost-key.txt

  Scenario: Remote URL uses environment API key
    Tool: Bash (pytest)
    Steps:
      1. Write test that verifies OPENROUTER_API_KEY is used for remote URLs
      2. Run: uv run pytest tests/unit/test_evaluation.py -k "test_remote" -v
    Expected Result: Test passes
    Evidence: .omo/evidence/task-2-remote-key.txt
  ```

  **Commit**: NO (groups with Task 3)

- [x] 3. Add `--judge-api-base` option to `quick_pipeline.sh`

  **What to do**:
  - Add `--judge-api-base` case in `while` loop (around line 157-160)
  - Add `JUDGE_API_BASE` variable with default `"https://openrouter.ai/api/v1"`
  - Add to `EVAL_FLAGS` construction (line 338)
  - Update `usage()` help text to document the option

  **Must NOT do**:
  - Do NOT modify `run_pipeline.sh` or other scripts
  - Do NOT change default behavior

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple shell script addition, follow existing patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `quick_pipeline.sh:157-160` — Existing `--judge-model` pattern to follow
  - `quick_pipeline.sh:338` — `EVAL_FLAGS` construction
  - `quick_pipeline.sh:62-64` — Existing evaluation options in usage()

  **Acceptance Criteria**:
  - [ ] `--judge-api-base` appears in `--help` output
  - [ ] `JUDGE_API_BASE` variable is set correctly
  - [ ] `EVAL_FLAGS` includes `--judge-api-base $JUDGE_API_BASE`

  **QA Scenarios**:
  ```
  Scenario: Shell help shows new option
    Tool: Bash
    Steps:
      1. Run: ./scripts/pipeline/quick_pipeline.sh --help
      2. Grep for "judge-api-base"
    Expected Result: Exit code 0, output contains "--judge-api-base"
    Evidence: .omo/evidence/task-3-shell-help.txt
  ```

  **Commit**: YES
  - Message: `feat(evaluation): add --judge-api-base CLI option for local LLM judges`
  - Files: `src/steering_geometry/apply_steering.py`, `scripts/pipeline/quick_pipeline.sh`, `tests/unit/test_evaluation.py`

- [x] 4. Test `JudgeConfig` accepts custom `api_base`

  **What to do**:
  - Add test `test_judge_config_with_custom_api_base` to `tests/unit/test_evaluation.py`
  - Verify `JudgeConfig(api_base="http://localhost:8000/v1")` works
  - Verify default is `"https://openrouter.ai/api/v1"`

  **Must NOT do**:
  - Do NOT test unrelated JudgeConfig fields
  - Do NOT add integration tests

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple unit test, follows existing test patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: Task 5
  - **Blocked By**: Task 2

  **References**:
  - `tests/unit/test_evaluation.py:22-35` — Existing `TestJudgeEvaluator.test_init` pattern
  - `config.py:102-115` — `JudgeConfig` dataclass

  **Acceptance Criteria**:
  - [ ] Test passes with `uv run pytest tests/unit/test_evaluation.py -k "test_judge_config" -v`

  **QA Scenarios**:
  ```
  Scenario: Test passes
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_evaluation.py -k "test_judge_config" -v
    Expected Result: Exit code 0, test passes
    Evidence: .omo/evidence/task-4-test-pass.txt
  ```

  **Commit**: NO (included in Task 3 commit)

- [x] 5. Test `JudgeEvaluator` uses custom `api_base`

  **What to do**:
  - Add test `test_judge_evaluator_uses_custom_api_base` to `tests/unit/test_evaluation.py`
  - Mock `AsyncOpenAI` constructor, verify it receives `base_url="http://localhost:8000/v1"`
  - Use `patch` to intercept `AsyncOpenAI` instantiation

  **Must NOT do**:
  - Do NOT make actual API calls
  - Do NOT test retry logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Unit test with mocking, follows existing patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6)
  - **Blocks**: Task 6
  - **Blocked By**: Tasks 2, 4

  **References**:
  - `tests/unit/test_evaluation.py:36-50` — Existing test pattern with `AsyncMock`
  - `apply_steering.py:196-199` — Where `AsyncOpenAI` is instantiated

  **Acceptance Criteria**:
  - [ ] Test verifies `AsyncOpenAI(base_url="http://localhost:8000/v1", ...)` is called

  **QA Scenarios**:
  ```
  Scenario: Test passes
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_evaluation.py -k "test_judge_evaluator_uses_custom_api_base" -v
    Expected Result: Exit code 0, test passes
    Evidence: .omo/evidence/task-5-test-pass.txt
  ```

  **Commit**: NO (included in Task 3 commit)

- [x] 6. Test connection error with helpful message

  **What to do**:
  - Add test `test_judge_evaluator_connection_error_helpful_message` to `tests/unit/test_evaluation.py`
  - Mock `AsyncOpenAI.chat.completions.create` to raise `httpx.ConnectError` or similar
  - Verify error message includes:
    - The exact URL that failed
    - Suggestion to check if vLLM server is running

  **Must NOT do**:
  - Do NOT make actual network calls
  - Do NOT test all possible error types

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Unit test for error handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: None
  - **Blocked By**: Tasks 2, 5

  **References**:
  - `tests/unit/test_evaluation.py:150-170` — Existing retry test pattern
  - `apply_steering.py:201-227` — `_call_api` with retry logic

  **Acceptance Criteria**:
  - [ ] Test verifies error message contains URL
  - [ ] Test verifies error message contains "vLLM" or "server"

  **QA Scenarios**:
  ```
  Scenario: Test passes
    Tool: Bash (pytest)
    Steps:
      1. Run: uv run pytest tests/unit/test_evaluation.py -k "test_judge_evaluator_connection_error" -v
    Expected Result: Exit code 0, test passes
    Evidence: .omo/evidence/task-6-test-pass.txt
  ```

  **Commit**: NO (included in Task 3 commit)

---

## Final Verification Wave (MANDATORY)

- [x] F1. **Plan Compliance Audit** — `oracle`
  Verify all CLI args added, tests pass, no scope creep.

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run mypy src/` + `uv run pytest`.

- [x] F3. **Real Manual QA** — `unspecified-high`
  Run `./scripts/pipeline/quick_pipeline.sh --help` and verify `--judge-api-base` appears.

- [x] F4. **Scope Fidelity Check** — `deep`
  Verify no changes to MMLUEvaluator, no new class created, only quick_pipeline.sh modified.

---

## Commit Strategy

- **1**: `feat(evaluation): add --judge-api-base CLI option for local LLM judges` — apply_steering.py, quick_pipeline.sh, test_evaluation.py

---

## Success Criteria

### Verification Commands
```bash
# CLI help shows new option
uv run python -m steering_geometry.apply_steering --help | grep -q "judge-api-base"

# Shell help shows new option
./scripts/pipeline/quick_pipeline.sh --help | grep -q "judge-api-base"

# All tests pass
uv run pytest

# Type check passes
uv run mypy src/

# Lint passes
uv run ruff check src/ tests/
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
