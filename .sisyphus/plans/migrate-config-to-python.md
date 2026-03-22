# Migrate Model/Concept Lists to Python Config

## TL;DR

> **Quick Summary**: Centralize hardcoded model and concept lists from 5 shell scripts into Python config.py with a shell export helper (`--shell` flag) for shell scripts to read config dynamically.
>
> **Deliverables**:
> - Centralized constants in `config.py` (7 models, 3 concepts)
> - Shell export helper in `__main__.py` (`--shell` flag)
> - Updated 3 Python modules to import from config
> - Updated 5 shell scripts to use centralized config
> - Unit tests for new functionality

> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Task 1 → Task 2 → Task 7 → Final Verification

---

## Context

### Original Request
User wants to move model and concept lists from shell scripts to Python config files. Specifically wants 7 models (Qwen3/3.5 variants, Gemma2, Olmo3) and 3 concepts (refusal, polite, sentiment).

### Interview Summary
**Key Discussions**:
- Model paths: Use standard HuggingFace IDs
- Concept naming: Keep "polite" (existing codebase convention)
- Shell integration: Create `--shell` flag in config module
- Concept scope: Only 3 concepts (remove honesty, toxicity, sycophancy from canonical list)

**Research Findings**:
- **5 shell scripts** have hardcoded arrays with inconsistent values
- **3 Python modules** have fragmented VALID_CONCEPTS (different values!)
- **Pre-existing bug**: `extract.py` has 3 concepts, `unembed_analysis.py` has 5, `token_analysis.py` has 5 hardcoded

### Metis Review
**Identified Gaps** (addressed):
- `token_analysis.py` also has hardcoded concept choices - must update
- Need error handling in shell export helper
- Must verify no circular import issues
- Data artifacts in `outputs/` for removed concepts - out of scope (separate task)

---

## Work Objectives

### Core Objective
Create a single source of truth for model and concept lists that both Python modules and shell scripts can reference, eliminating current inconsistencies.

### Concrete Deliverables
- `src/steering_geometry/config.py`: Add `SUPPORTED_MODELS`, `SUPPORTED_CONCEPTS`, `DEFAULT_MODEL`
- `src/steering_geometry/__main__.py`: New file with `--shell` flag
- `src/steering_geometry/extract.py`: Import from config
- `src/steering_geometry/unembed_analysis.py`: Import from config
- `src/steering_geometry/token_analysis.py`: Import from config
- `tests/unit/test_config_main.py`: Unit tests for shell export
- 5 shell scripts updated to use centralized config

### Definition of Done
- [ ] `uv run python -m steering_geometry --shell` outputs valid bash
- [ ] `uv run mypy src/` → Success with 0 errors
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run pytest` → all tests pass
- [ ] Shell scripts work unchanged from user perspective

### Must Have
- Centralized constants in config.py
- Shell export helper with proper quoting
- All Python modules import from config
- All shell scripts use centralized config

### Must NOT Have (Guardrails)
- NO validation logic beyond what exists
- NO --list-models, --list-concepts, or any CLI beyond --shell
- NO backward compatibility aliases for removed concepts
- NO changes to script behavior (only data source changes)
- NO environment variable overrides
- NO touching outputs/ data files (separate concern)
- NO scope creep: adding -m all to single-model scripts

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest with 12 test files)
- **Automated tests**: YES (TDD for new functionality)
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - 1 task):
└── Task 1: Add centralized constants to config.py [quick]

Wave 2 (Core Implementation - 5 parallel tasks after Task 1):
├── Task 2: Create __main__.py with --shell flag [quick]
├── Task 3: Unit tests for __main__.py [quick]
├── Task 4: Update extract.py to import from config [quick]
├── Task 5: Update unembed_analysis.py to import from config [quick]
└── Task 6: Update token_analysis.py to import from config [quick]

Wave 3 (Shell Scripts - 5 parallel tasks after Task 2):
├── Task 7: Update quick_pipeline.sh [quick]
├── Task 8: Update quick_tdnv.sh [quick]
├── Task 9: Update run_unembed_analysis.sh [quick]
├── Task 10: Update quick_discriminative_heatmaps.sh [quick]
└── Task 11: Update quick_diff_means_heatmaps.sh [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 2 → Task 7 → F1-F4 → user okay
Parallel Speedup: ~70% faster than sequential
Max Concurrent: 5 (Waves 2 & 3)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1 | — | 4, 5, 6 |
| 2 | — | 7, 8, 9, 10, 11 |
| 3 | 2 | — |
| 4 | 1 | — |
| 5 | 1 | — |
| 6 | 1 | — |
| 7 | 2 | — |
| 8 | 2 | — |
| 9 | 2 | — |
| 10 | 2 | — |
| 11 | 2 | — |

### Agent Dispatch Summary

- **Wave 1**: **1** — T1 → `quick`
- **Wave 2**: **5** — T2, T3, T4, T5, T6 → all `quick`
- **Wave 3**: **5** — T7-T11 → all `quick`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Add centralized constants to config.py

  **What to do**:
  - Add `SUPPORTED_MODELS: tuple[str, ...]` with 7 models:
    - "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"
    - "google/gemma-2-2b", "google/gemma-2-9b", "allenai/OLMo-2-1124-7B"
  - Add `SUPPORTED_CONCEPTS: tuple[str, ...]` = ("refusal", "polite", "sentiment")
  - Add `DEFAULT_MODEL: str = "Qwen/Qwen3-1.7B"`
  - Add these to `__all__` list
  - Keep existing dataclasses unchanged

  **Must NOT do**:
  - Remove or modify existing config classes
  - Add validation logic
  - Add environment variable support

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple addition of module-level constants
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundation task)
  - **Parallel Group**: Wave 1 (alone)
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/config.py:1-210` - Existing config structure, add after line 6
  - `src/steering_geometry/extract.py:59` - Current VALID_CONCEPTS pattern to replace

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.config import SUPPORTED_MODELS; print(len(SUPPORTED_MODELS))"` → 7
  - [ ] `uv run python -c "from steering_geometry.config import SUPPORTED_CONCEPTS; print(sorted(SUPPORTED_CONCEPTS))"` → ['polite', 'refusal', 'sentiment']
  - [ ] `uv run mypy src/steering_geometry/config.py` → Success

  **QA Scenarios**:
  ```
  Scenario: Constants are importable
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.config import SUPPORTED_MODELS, SUPPORTED_CONCEPTS, DEFAULT_MODEL; print(len(SUPPORTED_MODELS), len(SUPPORTED_CONCEPTS), DEFAULT_MODEL)"
    Expected Result: "7 3 Qwen/Qwen3-1.7B"
    Evidence: .sisyphus/evidence/task-01-import-constants.txt

  Scenario: Constants are immutable tuples
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.config import SUPPORTED_MODELS; print(type(SUPPORTED_MODELS).__name__)"
    Expected Result: "tuple"
    Evidence: .sisyphus/evidence/task-01-tuple-type.txt
  ```

  **Commit**: NO (groups with Task 2)

---

- [x] 2. Create __main__.py with --shell flag

  **What to do**:
  - Create `src/steering_geometry/__main__.py`
  - Implement `--shell` flag that outputs eval-compatible bash:
    ```bash
    ALL_MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" ...)
    ALL_CONCEPTS=("refusal" "polite" "sentiment")
    DEFAULT_MODEL="Qwen/Qwen3-1.7B"
    ```
  - Use argparse with single `--shell` flag
  - Exit 0 on success, non-zero on error
  - Handle import errors gracefully

  **Must NOT do**:
  - Add --list-models, --list-concepts, or any other CLI flags
  - Add JSON/YAML output formats
  - Add verbose logging

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple CLI module with single flag
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1, no dependency)
  - **Parallel Group**: Wave 2 (with Tasks 3-6)
  - **Blocks**: Tasks 7, 8, 9, 10, 11
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/config.py` - Import SUPPORTED_MODELS, SUPPORTED_CONCEPTS, DEFAULT_MODEL
  - `src/steering_geometry/extract.py:527-600` - Example argparse pattern in this codebase

  **Acceptance Criteria**:
  - [ ] `uv run python -m steering_geometry --shell` outputs valid bash
  - [ ] Output can be eval'd: `eval $(uv run python -m steering_geometry --shell)`
  - [ ] Exit code 0 on success

  **QA Scenarios**:
  ```
  Scenario: Shell export produces valid bash
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry --shell
    Expected Result: Lines starting with ALL_MODELS=, ALL_CONCEPTS=, DEFAULT_MODEL=
    Evidence: .sisyphus/evidence/task-02-shell-output.txt

  Scenario: Shell eval works
    Tool: Bash
    Steps:
      1. eval $(uv run python -m steering_geometry --shell)
      2. echo "${ALL_MODELS[0]}"
      3. echo "${ALL_CONCEPTS[0]}"
    Expected Result: "Qwen/Qwen3-1.7B" then "refusal"
    Evidence: .sisyphus/evidence/task-02-shell-eval.txt

  Scenario: Error handling
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry --invalid-flag 2>/dev/null; echo "EXIT=$?"
    Expected Result: EXIT=2 (argparse error code)
    Evidence: .sisyphus/evidence/task-02-error-handling.txt
  ```

  **Commit**: YES (with Task 3)
  - Message: `feat(config): add centralized constants and shell export helper`
  - Files: `src/steering_geometry/config.py`, `src/steering_geometry/__main__.py`
  - Pre-commit: `uv run mypy src/ && uv run ruff check src/`

---

- [x] 3. Unit tests for __main__.py

  **What to do**:
  - Create `tests/unit/test_config_main.py`
  - Test: shell output format is valid bash
  - Test: shell output contains expected values
  - Test: --shell flag is recognized
  - Test: invalid flags exit with non-zero code
  - Follow existing test patterns in `tests/unit/`

  **Must NOT do**:
  - Test implementation details (test behavior, not internals)
  - Add integration tests (those are in Wave FINAL)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple unit tests for CLI module
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 4, 5, 6)
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `tests/unit/test_extract.py` - Test patterns to follow
  - `tests/conftest.py` - Existing fixtures

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/unit/test_config_main.py` → all pass
  - [ ] At least 4 test functions covering: output format, values, flag recognition, error handling

  **QA Scenarios**:
  ```
  Scenario: All tests pass
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_config_main.py -v
    Expected Result: All tests pass (4+ tests)
    Evidence: .sisyphus/evidence/task-03-tests-pass.txt
  ```

  **Commit**: YES (with Task 2)
  - Message: `feat(cli): add shell export helper with tests`
  - Files: `src/steering_geometry/__main__.py`, `tests/unit/test_config_main.py`
  - Pre-commit: `uv run pytest tests/unit/test_config_main.py`

---

- [x] 4. Update extract.py to import from config

  **What to do**:
  - Add import: `from steering_geometry.config import SUPPORTED_CONCEPTS as VALID_CONCEPTS`
  - Remove local `VALID_CONCEPTS = {"polite", "sentiment", "refusal"}` at line 59
  - Keep `_DATASET_LOADERS` unchanged (concept loaders)
  - Verify no circular import (config.py doesn't import extract.py)

  **Must NOT do**:
  - Change concept loader logic
  - Add new concepts
  - Modify function signatures

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple import change
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3, 5, 6)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:
  - `src/steering_geometry/extract.py:59` - Current VALID_CONCEPTS to replace
  - `src/steering_geometry/extract.py:77` - Where VALID_CONCEPTS is used
  - `src/steering_geometry/config.py` - Import source

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.extract import VALID_CONCEPTS; print(sorted(VALID_CONCEPTS))"` → ['polite', 'refusal', 'sentiment']
  - [ ] `uv run mypy src/steering_geometry/extract.py` → Success
  - [ ] No circular import errors

  **QA Scenarios**:
  ```
  Scenario: Import chain works
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.extract import VALID_CONCEPTS; print(sorted(VALID_CONCEPTS))"
    Expected Result: "['polite', 'refusal', 'sentiment']"
    Evidence: .sisyphus/evidence/task-04-extract-import.txt

  Scenario: No circular import
    Tool: Bash
    Steps:
      1. uv run python -c "import steering_geometry.extract; import steering_geometry.config; print('OK')"
    Expected Result: "OK"
    Evidence: .sisyphus/evidence/task-04-no-circular.txt
  ```

  **Commit**: NO (groups with Tasks 5, 6)

---

- [x] 5. Update unembed_analysis.py to import from config

  **What to do**:
  - Add import: `from steering_geometry.config import SUPPORTED_CONCEPTS as VALID_CONCEPTS`
  - Remove local `VALID_CONCEPTS = ("honesty", "sentiment", "toxicity", "sycophancy", "refusal")` at line 458
  - Also remove local `VALID_METHODS` and `DEFAULT_LAYERS` if they exist (check line 458-460)
  - Update argparse choices to use imported VALID_CONCEPTS

  **Must NOT do**:
  - Change analysis logic
  - Add validation for removed concepts

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple import change
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3, 4, 6)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:
  - `src/steering_geometry/unembed_analysis.py:458` - Current VALID_CONCEPTS to replace
  - `src/steering_geometry/config.py` - Import source

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.unembed_analysis import VALID_CONCEPTS; print(sorted(VALID_CONCEPTS))"` → ['polite', 'refusal', 'sentiment']
  - [ ] `uv run mypy src/steering_geometry/unembed_analysis.py` → Success

  **QA Scenarios**:
  ```
  Scenario: Import works correctly
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.unembed_analysis import VALID_CONCEPTS; print(sorted(VALID_CONCEPTS))"
    Expected Result: "['polite', 'refusal', 'sentiment']"
    Evidence: .sisyphus/evidence/task-05-unembed-import.txt

  Scenario: Module still loads
    Tool: Bash
    Steps:
      1. uv run python -c "import steering_geometry.unembed_analysis; print('OK')"
    Expected Result: "OK"
    Evidence: .sisyphus/evidence/task-05-unembed-load.txt
  ```

  **Commit**: NO (groups with Tasks 4, 6)

---

- [x] 6. Update token_analysis.py to import from config

  **What to do**:
  - Find hardcoded concept choices in argparse (around line 326, 353)
  - Add import: `from steering_geometry.config import SUPPORTED_CONCEPTS`
  - Update argparse `choices=` parameter to use `SUPPORTED_CONCEPTS`
  - Remove any local concept list definitions

  **Must NOT do**:
  - Change token analysis logic
  - Add new analysis methods

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple import and argparse update
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3, 4, 5)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:
  - `src/steering_geometry/token_analysis.py:326,353` - Hardcoded choices to update
  - `src/steering_geometry/config.py` - Import source

  **Acceptance Criteria**:
  - [ ] `uv run python -m steering_geometry.token_analysis visualize --help` shows correct concept choices
  - [ ] `uv run mypy src/steering_geometry/token_analysis.py` → Success

  **QA Scenarios**:
  ```
  Scenario: argparse shows correct choices
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.token_analysis visualize --help 2>&1 | grep -A2 "concept"
    Expected Result: Shows refusal, polite, sentiment as choices
    Evidence: .sisyphus/evidence/task-06-token-choices.txt

  Scenario: Invalid concept is rejected
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.token_analysis visualize --concept invalid_concept 2>&1; echo "EXIT=$?"
    Expected Result: Exit code non-zero, error mentions invalid choice
    Evidence: .sisyphus/evidence/task-06-invalid-concept.txt
  ```

  **Commit**: YES (with Tasks 4, 5)
  - Message: `refactor: use centralized VALID_CONCEPTS from config`
  - Files: `src/steering_geometry/extract.py`, `src/steering_geometry/unembed_analysis.py`, `src/steering_geometry/token_analysis.py`
  - Pre-commit: `uv run pytest`

---

- [x] 7. Update quick_pipeline.sh to use centralized config

  **What to do**:
  - Replace `ALL_CONCEPTS=("sentiment" "refusal" "polite")` at line 26 with:
    ```bash
    eval $(uv run python -m steering_geometry --shell)
    ```
  - Replace `ALL_MODELS=()` at line 27 with the same (already done above)
  - Remove both local array definitions
  - Keep default variable assignments (`CONCEPTS=""`, `MODELS="Qwen/Qwen3.5-2B"`) - these are user overrides

  **Must NOT do**:
  - Change script behavior or options
  - Add new CLI flags
  - Modify error handling

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple variable replacement
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 9, 10, 11)
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `scripts/pipeline/quick_pipeline.sh:26-27` - Lines to replace
  - `scripts/pipeline/quick_pipeline.sh:100-101` - Default variables (keep these)

  **Acceptance Criteria**:
  - [ ] `./scripts/pipeline/quick_pipeline.sh -h` runs without error
  - [ ] `./scripts/pipeline/quick_pipeline.sh -c all` uses centralized concepts
  - [ ] Script behavior unchanged from before

  **QA Scenarios**:
  ```
  Scenario: Help runs successfully
    Tool: Bash
    Steps:
      1. ./scripts/pipeline/quick_pipeline.sh -h 2>&1 | head -5
    Expected Result: Usage information displayed, exit code 0
    Evidence: .sisyphus/evidence/task-07-pipeline-help.txt

  Scenario: All concepts work
    Tool: Bash
    Steps:
      1. ./scripts/pipeline/quick_pipeline.sh -c all --help 2>&1 | grep -i "concept"
    Expected Result: Shows all 3 concepts
    Evidence: .sisyphus/evidence/task-07-pipeline-concepts.txt
  ```

  **Commit**: NO (groups with Tasks 8-11)

---

- [x] 8. Update quick_tdnv.sh to use centralized config

  **What to do**:
  - Replace `ALL_CONCEPTS=("honesty" "sycophancy" "refusal" "sentiment")` at line 30 with:
    ```bash
    eval $(uv run python -m steering_geometry --shell)
    ```
  - Replace `ALL_MODELS=()` at line 31 with the same
  - Remove both local array definitions
  - Note: This script had different concepts (honesty, sycophancy) - they will no longer be available

  **Must NOT do**:
  - Add backward compatibility for removed concepts
  - Change script behavior

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple variable replacement
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 9, 10, 11)
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `scripts/tdnv/quick_tdnv.sh:30-31` - Lines to replace
  - `scripts/tdnv/quick_tdnv.sh:92-93` - Default variables (keep these)

  **Acceptance Criteria**:
  - [ ] `./scripts/tdnv/quick_tdnv.sh -h` runs without error
  - [ ] Script uses centralized config

  **QA Scenarios**:
  ```
  Scenario: Help runs successfully
    Tool: Bash
    Steps:
      1. ./scripts/tdnv/quick_tdnv.sh -h 2>&1 | head -5
    Expected Result: Usage information displayed, exit code 0
    Evidence: .sisyphus/evidence/task-08-tdnv-help.txt

  Scenario: Concepts show centralized list
    Tool: Bash
    Steps:
      1. ./scripts/tdnv/quick_tdnv.sh -h 2>&1 | grep -A5 "Available:"
    Expected Result: Shows refusal, polite, sentiment (not honesty, sycophancy)
    Evidence: .sisyphus/evidence/task-08-tdnv-concepts.txt
  ```

  **Commit**: NO (groups with Tasks 7, 9, 10, 11)

---

- [x] 9. Update run_unembed_analysis.sh to use centralized config

  **What to do**:
  - Replace `ALL_CONCEPTS=("honesty" "sentiment" "toxicity" "sycophancy" "refusal")` at line 25 with:
    ```bash
    eval $(uv run python -m steering_geometry --shell)
    ```
  - Remove local array definition
  - Note: This script also had different concepts - they will no longer be available

  **Must NOT do**:
  - Add backward compatibility for removed concepts
  - Change script behavior

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple variable replacement
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8, 10, 11)
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `scripts/unembed_analysis/run_unembed_analysis.sh:25` - Line to replace
  - `scripts/unembed_analysis/run_unembed_analysis.sh:60` - MODEL default (keep this)

  **Acceptance Criteria**:
  - [ ] `./scripts/unembed_analysis/run_unembed_analysis.sh -h` runs without error
  - [ ] Script uses centralized config

  **QA Scenarios**:
  ```
  Scenario: Help runs successfully
    Tool: Bash
    Steps:
      1. ./scripts/unembed_analysis/run_unembed_analysis.sh -h 2>&1 | head -5
    Expected Result: Usage information displayed, exit code 0
    Evidence: .sisyphus/evidence/task-09-unembed-help.txt

  Scenario: Concepts show centralized list
    Tool: Bash
    Steps:
      1. ./scripts/unembed_analysis/run_unembed_analysis.sh -h 2>&1 | grep -i "concept"
    Expected Result: Shows refusal, polite, sentiment
    Evidence: .sisyphus/evidence/task-09-unembed-concepts.txt
  ```

  **Commit**: NO (groups with Tasks 7, 8, 10, 11)

---

- [x] 10. Update quick_discriminative_heatmaps.sh to use centralized config

  **What to do**:
  - Add at start of script (after shebang and comments):
    ```bash
    eval $(uv run python -m steering_geometry --shell)
    ```
  - Replace `CONCEPTS=("honesty" "sentiment" "toxicity" "sycophancy" "refusal")` at line 23 - this is now from eval
  - Note: This script iterates over CONCEPTS array, so it uses the centralized list

  **Must NOT do**:
  - Change iteration logic
  - Add backward compatibility

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple variable replacement
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8, 9, 11)
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `scripts/vector_analysis/quick_discriminative_heatmaps.sh:23` - CONCEPTS array to replace
  - `scripts/vector_analysis/quick_discriminative_heatmaps.sh:26` - MODEL default (keep this)

  **Acceptance Criteria**:
  - [ ] `./scripts/vector_analysis/quick_discriminative_heatmaps.sh -h` runs without error
  - [ ] Script uses centralized config

  **QA Scenarios**:
  ```
  Scenario: Help runs successfully
    Tool: Bash
    Steps:
      1. ./scripts/vector_analysis/quick_discriminative_heatmaps.sh -h 2>&1 | head -5
    Expected Result: Usage information displayed
    Evidence: .sisyphus/evidence/task-10-discr-help.txt
  ```

  **Commit**: NO (groups with Tasks 7, 8, 9, 11)

---

- [x] 11. Update quick_diff_means_heatmaps.sh to use centralized config

  **What to do**:
  - Add at start of script (after shebang and comments):
    ```bash
    eval $(uv run python -m steering_geometry --shell)
    ```
  - Replace `CONCEPTS=("honesty" "sentiment" "toxicity" "sycophancy" "refusal")` at line 28 - this is now from eval
  - Note: This script iterates over CONCEPTS array, so it uses the centralized list

  **Must NOT do**:
  - Change iteration logic
  - Add backward compatibility

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple variable replacement
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 7, 8, 9, 10)
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `scripts/vector_analysis/quick_diff_means_heatmaps.sh:28` - CONCEPTS array to replace
  - `scripts/vector_analysis/quick_diff_means_heatmaps.sh:31` - MODEL default (keep this)

  **Acceptance Criteria**:
  - [ ] `./scripts/vector_analysis/quick_diff_means_heatmaps.sh -h` runs without error
  - [ ] Script uses centralized config

  **QA Scenarios**:
  ```
  Scenario: Help runs successfully
    Tool: Bash
    Steps:
      1. ./scripts/vector_analysis/quick_diff_means_heatmaps.sh -h 2>&1 | head -5
    Expected Result: Usage information displayed
    Evidence: .sisyphus/evidence/task-11-diff-help.txt
  ```

  **Commit**: YES (with Tasks 7, 8, 9, 10)
  - Message: `refactor(scripts): use centralized config via shell export`
  - Files: All 5 shell scripts
  - Pre-commit: All scripts run with `-h` flag successfully

---

## Final Verification Wave (MANDATORY)

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run mypy src/` + `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, print in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Format [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test shell export: `eval $(uv run python -m steering_geometry --shell)`. Test updated scripts: run with `-h` flag, verify no errors. Test Python imports: `from steering_geometry.config import SUPPORTED_MODELS`. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git diff). Verify 1:1 — everything in spec was built, nothing beyond spec. Check "Must NOT do" compliance. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Commit 1**: Add centralized constants to config.py
  - Message: `feat(config): add SUPPORTED_MODELS, SUPPORTED_CONCEPTS, DEFAULT_MODEL`
  - Files: `src/steering_geometry/config.py`
  - Pre-commit: `uv run mypy src/ && uv run ruff check src/`

- **Commit 2**: Create __main__.py with --shell flag and tests
  - Message: `feat(cli): add shell export helper for model/concept lists`
  - Files: `src/steering_geometry/__main__.py`, `tests/unit/test_config_main.py`
  - Pre-commit: `uv run pytest tests/unit/test_config_main.py`

- **Commit 3**: Update Python modules to import from config
  - Message: `refactor: use centralized VALID_CONCEPTS from config`
  - Files: `src/steering_geometry/extract.py`, `src/steering_geometry/unembed_analysis.py`, `src/steering_geometry/token_analysis.py`
  - Pre-commit: `uv run pytest`

- **Commit 4**: Update shell scripts to use centralized config
  - Message: `refactor(scripts): use centralized config via shell export`
  - Files: `scripts/pipeline/quick_pipeline.sh`, `scripts/tdnv/quick_tdnv.sh`, `scripts/unembed_analysis/run_unembed_analysis.sh`, `scripts/vector_analysis/quick_discriminative_heatmaps.sh`, `scripts/vector_analysis/quick_diff_means_heatmaps.sh`
  - Pre-commit: All scripts run with `-h` flag successfully

---

## Success Criteria

### Verification Commands
```bash
# Test centralized constants
uv run python -c "from steering_geometry.config import SUPPORTED_MODELS, SUPPORTED_CONCEPTS, DEFAULT_MODEL; print(len(SUPPORTED_MODELS), len(SUPPORTED_CONCEPTS))"
# Expected: 7 3

# Test shell export
uv run python -m steering_geometry --shell
# Expected: ALL_MODELS=(...), ALL_CONCEPTS=(...), DEFAULT_MODEL=...

# Test shell eval
eval $(uv run python -m steering_geometry --shell) && echo "${ALL_MODELS[0]}" && echo "${ALL_CONCEPTS[0]}"
# Expected: Qwen/Qwen3-1.7B refusal

# Type check
uv run mypy src/
# Expected: Success: no issues found

# Lint
uv run ruff check src/ tests/
# Expected: 0 violations

# All tests
uv run pytest
# Expected: all pass
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Shell export works correctly
- [ ] All Python modules import from config
- [ ] All shell scripts use centralized config
