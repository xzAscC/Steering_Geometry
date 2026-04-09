# Token Selection Experiments for Steering Geometry

## TL;DR

> **Quick Summary**: Create 4 research experiments investigating token selection strategies for activation steering: (1) token count, (2) token position, (3) prompt vs response, (4) steering scope. Each experiment is a shell script under `scripts/token_experiments/` backed by a new Python module.
> 
> **Deliverables**:
> - Modified `generate_with_steering()` with prefix-only steering support (`steer_tokens` parameter)
> - New module `src/steering_geometry/token_selection_experiments.py` with 4 experiment runner functions
> - 4 shell scripts: `scripts/token_experiments/{1_token_count,2_token_position,3_prompt_vs_response,4_steering_scope}.sh`
> - Test file `tests/test_token_selection_experiments.py`
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 (hook mod) → Task 6 (experiment 4 function + script) → F1-F4

---

## Context

### Original Request
Four research questions on token selection strategies for steering vectors:
1. How many tokens should we use for concept vector extraction? (current practice: ~100 tokens)
2. Where to select tokens? All tokens vs last-n tokens (n=1,2,3,4,5,10)
3. Select from prompt or prompt+response? (data_mode comparison)
4. Should we steer all tokens or only the first N? (prefix steering vs full steering, N=5,10,15,20)

### Interview Summary
**Key Discussions**:
- Model: Qwen/Qwen3-1.7B (single model, matching stability experiments)
- Concept: refusal only
- No evaluation implementation now — will evaluate on MMLU/HarmBench/ORBench later
- Token count range matches stability experiments: n_examples=[10,30,100,300,1000,3000,6000,10000]
- Layer range: [0.4, 0.5, 0.6, 0.7, 0.8]

**Research Findings**:
- `select_token_activations()` already supports "all", "last_n", int index modes
- `data_mode` parameter already supports "prompt_only" vs "prompt_response" for refusal
- Current steering hook applies to ALL positions — NEEDS modification for experiment 4
- `ExtractionConfig.token_select` defaults to `"default"` (sentinel, not bug) — experiments must explicitly set it
- Literature: last-token is dominant but robustness debated; prefix steering identified as future work by CAA

### Metis Review
**Identified Gaps** (addressed):
- `FakeCausalLM.generate()` doesn't simulate KV-cache → tests need enhancement for steer_tokens
- data_mode validation absent in loader → experiment should validate before calling
- Output-shape-based counter approach for hook modification is correct with KV cache assumption
- Must explicitly set `use_cache=True` when steer_tokens is used
- Edge cases: steer_tokens=0 (no steering), steer_tokens>=max_new_tokens (equiv to None), counter reset between calls

---

## Work Objectives

### Core Objective
Create infrastructure for 4 token selection experiments that produce steering vectors and steered outputs under different conditions, enabling systematic comparison of token selection strategies.

### Concrete Deliverables
- `src/steering_geometry/config.py` — `SteeringConfig` extended with `steer_tokens: int | None = None`
- `src/steering_geometry/models.py` — `generate_with_steering()` with prefix-only steering support
- `src/steering_geometry/token_selection_experiments.py` — 4 experiment runner functions
- `scripts/token_experiments/1_token_count.sh` — Token count sweep experiment
- `scripts/token_experiments/2_token_position.sh` — Token position comparison experiment
- `scripts/token_experiments/3_prompt_vs_response.sh` — Prompt vs response experiment
- `scripts/token_experiments/4_steering_scope.sh` — Steering scope experiment
- `tests/test_token_selection_experiments.py` — Tests for new functionality

### Definition of Done
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → already formatted
- [ ] `uv run mypy src/` → 0 errors
- [ ] `uv run pytest` → all tests pass
- [ ] Each shell script passes `bash -n` syntax check

### Must Have
- Backward-compatible `steer_tokens` parameter (None = existing behavior)
- 4 experiment functions following stability_comparison.py pattern
- Cosine similarity heatmaps for extraction experiments (1-3)
- Steered text JSONL outputs for steering scope experiment (4)
- Output directory: `outputs/token_experiments/{experiment_name}/`
- Model loaded ONCE per experiment, reused across all parameter combinations

### Must NOT Have (Guardrails)
- NO evaluation infrastructure (MMLU, HarmBench, ORBench — explicitly deferred)
- NO new token selection modes in `select_token_activations()`
- NO modification to `load_refusal_data()` or any dataset loaders
- NO replacement of `model.generate()` with manual token-by-token loop
- NO Python files in `scripts/` directory
- NO new dependencies in `pyproject.toml`
- NO `typing.Any` or `# type: ignore` in new code
- NO `print()` in new code — use `logging` module
- AI slop: excessive comments, over-abstraction, generic names

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (TDD for hook modification, tests-after for experiment functions)
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python modules**: Use Bash (`uv run pytest`, `uv run mypy`) — run tests, check types
- **Shell scripts**: Use Bash (`bash -n`, `--help`) — syntax check, verify arg parsing
- **Integration**: Use Bash (`uv run python -c "..."`) — import modules, verify API

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation, 2 parallel tasks):
├── Task 1: Add steer_tokens to SteeringConfig + modify hook + tests [deep]
└── Task 2: Create token_selection_experiments.py with experiments 1-3 + tests [unspecified-high]

Wave 2 (After Wave 1 — scripts + experiment 4, 4 parallel tasks):
├── Task 3: Create 1_token_count.sh [quick]
├── Task 4: Create 2_token_position.sh [quick]
├── Task 5: Create 3_prompt_vs_response.sh [quick]
└── Task 6: Add experiment 4 function + create 4_steering_scope.sh [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay

Critical Path: Task 1 → Task 6 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 4 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 6 | 1 |
| 2 | — | 3, 4, 5 | 1 |
| 3 | 2 | F1-F4 | 2 |
| 4 | 2 | F1-F4 | 2 |
| 5 | 2 | F1-F4 | 2 |
| 6 | 1 | F1-F4 | 2 |
| F1 | 1-6 | user okay | FINAL |
| F2 | 1-6 | user okay | FINAL |
| F3 | 1-6 | user okay | FINAL |
| F4 | 1-6 | user okay | FINAL |

### Agent Dispatch Summary

- **Wave 1**: 2 — T1 → `deep`, T2 → `unspecified-high`
- **Wave 2**: 4 — T3-T5 → `quick`, T6 → `unspecified-high`
- **FINAL**: 4 — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Add `steer_tokens` Parameter for Prefix-Only Steering

  **What to do**:
  - Add `steer_tokens: int | None = None` field to `SteeringConfig` dataclass in `src/steering_geometry/config.py` (after `max_new_tokens` field)
  - Add `steer_tokens: int | None = None` parameter to `generate_with_steering()` in `src/steering_geometry/models.py`
  - Modify the `steering_hook` closure inside `generate_with_steering()` to implement step-counting:
    1. Create a mutable counter (`step_counter = [0]`) in the closure
    2. On each hook invocation, increment the counter
    3. If `steer_tokens is not None and step_counter[0] > steer_tokens`, return output unchanged
    4. Otherwise, apply steering as before (add to all positions in tensor)
  - **IMPORTANT**: The counter must reset between calls — it's created fresh in each `generate_with_steering` call closure
  - When `steer_tokens` is not None, ensure `use_cache=True` is set in `gen_kwargs` (add comment explaining the KV-cache dependency)
  - Handle edge cases: `steer_tokens=0` → no steering applied; `steer_tokens >= max_new_tokens` → equivalent to `None`
  - Use `lsp_find_references` on `generate_with_steering` and `SteeringConfig` BEFORE modifying signatures to ensure no callers break
  - Write tests in `tests/test_token_selection_experiments.py`:
    - `test_steer_tokens_backward_compat`: calling without `steer_tokens` produces identical output to before
    - `test_steer_tokens_zero`: `steer_tokens=0` produces unsteered output
    - `test_steer_tokens_large`: `steer_tokens=999` (>= max_new_tokens) produces same output as `steer_tokens=None`
  - Enhance `FakeCausalLM.generate()` in `tests/conftest.py` if needed to trigger hooks during generation (currently computes hidden states in one batch call, never triggers per-step hooks)

  **Must NOT do**:
  - Do NOT replace `model.generate()` with a manual token-by-token loop
  - Do NOT modify `utils.select_token_activations()`
  - Do NOT add new dependencies
  - Do NOT break existing callers of `generate_with_steering` or `SteeringConfig`

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires understanding of HuggingFace KV-cache behavior, closure-based hook modification, and careful backward compatibility
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not UI work
    - `git-master`: No git operations in this task

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 6
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/steering_geometry/models.py:194-258` — `generate_with_steering()` function, hook registration, and generation logic. The steering_hook closure at lines 229-235 is the key modification point
  - `src/steering_geometry/models.py:229-235` — Current hook implementation: `tensor_output = tensor_output + steering * scale`. Must be wrapped in step-counter conditional
  - `src/steering_geometry/config.py:91-106` — `SteeringConfig` dataclass. Add `steer_tokens` after `temperature` field

  **API/Type References**:
  - `src/steering_geometry/config.py:91` — `SteeringConfig` class definition. New field must be `steer_tokens: int | None = None`
  - `tests/conftest.py:94-113` — `FakeCausalLM.generate()` fixture. May need enhancement to trigger hooks per-step

  **Test References**:
  - `tests/test_apply_steering.py` — Existing steering integration tests, shows test patterns for steering behavior

  **WHY Each Reference Matters**:
  - `models.py:194-258`: This is the ONLY file that needs modification for the hook logic. The closure pattern means the counter variable is naturally scoped per-call
  - `config.py:91-106`: Adding the field here makes it available through the config chain
  - `conftest.py:94-113`: If tests fail because FakeCausalLM doesn't trigger hooks during generation, this fixture needs enhancement

  **Acceptance Criteria**:

  - [ ] `SteeringConfig` has `steer_tokens: int | None = None` field with docstring
  - [ ] `generate_with_steering()` accepts `steer_tokens` parameter
  - [ ] `steer_tokens=None` produces identical behavior to original (backward compat)
  - [ ] `steer_tokens=0` produces unsteered output
  - [ ] `uv run pytest tests/test_token_selection_experiments.py -k "steer_tokens" -v` → PASS
  - [ ] `uv run pytest tests/` → ALL pass (no regression)
  - [ ] `uv run mypy src/steering_geometry/models.py src/steering_geometry/config.py` → 0 errors

  **QA Scenarios:**

  ```
  Scenario: Backward compatibility — existing callers unaffected
    Tool: Bash (uv run pytest)
    Preconditions: Existing test suite passes
    Steps:
      1. Run `uv run pytest tests/test_apply_steering.py -v`
      2. Verify all existing tests pass (no regression from signature change)
      3. Run `uv run pytest tests/ -v`
      4. Verify 0 failures
    Expected Result: All existing tests pass. No regression.
    Failure Indicators: Any test failure, mypy error on existing callers
    Evidence: .sisyphus/evidence/task-1-backward-compat.txt

  Scenario: steer_tokens=0 produces unsteered output
    Tool: Bash (uv run pytest)
    Preconditions: test_steer_tokens_zero test exists
    Steps:
      1. Run `uv run pytest tests/test_token_selection_experiments.py -k "steer_tokens_zero" -v`
      2. Verify test passes — output with steer_tokens=0 matches unsteered output
    Expected Result: PASS — no steering applied when steer_tokens=0
    Failure Indicators: Test fails, or steering is still applied
    Evidence: .sisyphus/evidence/task-1-steer-zero.txt
  ```

  **Commit**: YES
  - Message: `feat(steering): add steer_tokens parameter for prefix-only steering`
  - Files: `src/steering_geometry/config.py`, `src/steering_geometry/models.py`, `tests/test_token_selection_experiments.py`, `tests/conftest.py`
  - Pre-commit: `uv run pytest && uv run mypy src/ && uv run ruff check src/ tests/`

- [x] 2. Create Token Selection Experiment Runners (Experiments 1-3)

  **What to do**:
  - Create `src/steering_geometry/token_selection_experiments.py` with 3 experiment runner functions
  - Follow the pattern from `stability_comparison.py:run_diff_means_experiment()` (load data → loop params → extract vectors → save → compute similarity → generate heatmaps)
  - **Function 1: `run_token_count_experiment()`**
    - Parameters: `concept, n_examples_list, layers, model_name, output_dir, method="mean", token_select="all"`
    - For each `n_examples` in `n_examples_list`: extract steering vector using `extract_vector()` or direct `extract_steering_vector()` call
    - Save vectors to `{output_dir}/vectors/{concept}/token_count/n{n_examples}_layer{layer_frac}.pt`
    - Compute pairwise cosine similarity across all n_examples values per layer
    - Generate heatmap PDFs at `{output_dir}/heatmaps/token_count/{concept}_layer{layer_frac}.pdf`
    - Return: `{"vector_paths": ..., "heatmap_paths": ..., "statistics": {...}}`
  - **Function 2: `run_token_position_experiment()`**
    - Parameters: `concept, n_examples, position_configs, layers, model_name, output_dir, method="mean"`
    - `position_configs` is a list of dicts: `[{"mode": "all"}, {"mode": "last_n", "n": 1}, ...]`
    - For each position config: set `ExtractionConfig.token_select` and `ExtractionConfig.last_n` accordingly
    - **CRITICAL**: Must explicitly set `token_select` — never rely on the "default" sentinel
    - Save vectors to `{output_dir}/vectors/{concept}/token_position/{mode}_n{n}_layer{layer_frac}.pt`
    - Compute pairwise cosine similarity across all position configs per layer
    - Generate heatmap PDFs at `{output_dir}/heatmaps/token_position/{concept}_layer{layer_frac}.pdf`
    - Return: same structure
  - **Function 3: `run_prompt_response_experiment()`**
    - Parameters: `concept, n_examples, data_modes, layers, model_name, output_dir, method="mean", token_select="all"`
    - `data_modes` is a list: `["prompt_only", "prompt_response"]`
    - For each data_mode: load contrast pairs with that data_mode, extract vectors
    - Validate data_mode values before calling loader (only "prompt_only" and "prompt_response" are valid)
    - Save vectors to `{output_dir}/vectors/{concept}/prompt_response/{data_mode}_n{n_examples}_layer{layer_frac}.pt`
    - Compute pairwise cosine similarity across data_modes per layer (only 2 values, so a simple comparison)
    - Generate comparison heatmap at `{output_dir}/heatmaps/prompt_response/{concept}_layer{layer_frac}.pdf`
    - Return: same structure
  - Use `logging` module (not `print()`) for all output
  - Import heatmap generation from `stability_comparison.py` or replicate the pattern
  - Write unit tests in `tests/test_token_selection_experiments.py`:
    - Test parameter labeling logic (given params, correct output paths are generated)
    - Test cosine similarity computation with known tensors
    - Test data_mode validation

  **Must NOT do**:
  - Do NOT use `print()` — use `logging` module
  - Do NOT use `typing.Any` — use proper types
  - Do NOT modify `stability_comparison.py`
  - Do NOT add evaluation logic

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires following existing patterns carefully, creating 3 similar but distinct functions, proper type hints throughout
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not UI work
    - `git-master`: No git operations

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1)
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 3, 4, 5
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py:179-309` — `run_diff_means_experiment()`. THIS is the canonical pattern to follow: load data, loop params, extract, save vectors, compute similarity, generate heatmaps, return dict
  - `src/steering_geometry/stability_comparison.py:312-460` — `run_discriminative_experiment()`. Same pattern with different parameter (top_k). Shows how to vary a parameter across extractions
  - `src/steering_geometry/stability_comparison.py:463-540` — Heatmap and similarity computation logic. Reuse or replicate this pattern
  - `scripts/vector_analysis/run_stability_comparison.sh` — Shows how inline Python invokes experiment functions

  **API/Type References**:
  - `src/steering_geometry/extract.py:458` — `extract_steering_vector()` function signature. The core extraction function to call
  - `src/steering_geometry/extract.py:527` — `extract_vector()` convenience wrapper
  - `src/steering_geometry/config.py:47-72` — `ExtractionConfig` fields. Must set `token_select` explicitly (not "default")
  - `src/steering_geometry/extract.py:294-295` — `load_refusal_data()` with `data_mode` parameter
  - `src/steering_geometry/utils.py:87-148` — `select_token_activations()` — existing token selection logic

  **WHY Each Reference Matters**:
  - `stability_comparison.py:179-309`: This is the exact pattern to replicate. Copy the structure, change what's varied
  - `extract.py:458`: The low-level extraction function that accepts `ExtractionConfig` directly — needed for fine-grained parameter control
  - `config.py:47-72`: Must understand all config fields to set them correctly. `token_select="default"` falls through to int-index path, so experiments MUST set it explicitly

  **Acceptance Criteria**:

  - [ ] `src/steering_geometry/token_selection_experiments.py` exists with 3 functions
  - [ ] Each function has proper type hints on all parameters and returns
  - [ ] `run_token_count_experiment()` varies n_examples and produces vectors + heatmaps
  - [ ] `run_token_position_experiment()` varies token selection mode and produces vectors + heatmaps
  - [ ] `run_prompt_response_experiment()` varies data_mode and produces vectors + heatmaps
  - [ ] No `print()` calls — uses `logging` module
  - [ ] No `typing.Any` in the module
  - [ ] `uv run mypy src/steering_geometry/token_selection_experiments.py` → 0 errors
  - [ ] `uv run pytest tests/test_token_selection_experiments.py -k "not steer_tokens" -v` → PASS

  **QA Scenarios:**

  ```
  Scenario: Module imports and functions are callable
    Tool: Bash (uv run python)
    Preconditions: Module file exists
    Steps:
      1. Run `uv run python -c "from steering_geometry.token_selection_experiments import run_token_count_experiment, run_token_position_experiment, run_prompt_response_experiment; print('OK')"`
      2. Verify output contains "OK"
    Expected Result: All 3 functions importable without error
    Failure Indicators: ImportError, ModuleNotFoundError, AttributeError
    Evidence: .sisyphus/evidence/task-2-import-check.txt

  Scenario: No print() calls in module
    Tool: Bash (grep)
    Preconditions: Module file exists
    Steps:
      1. Run `grep -n 'print(' src/steering_geometry/token_selection_experiments.py`
      2. Verify no matches found
    Expected Result: 0 matches — all output uses logging
    Failure Indicators: Any match found
    Evidence: .sisyphus/evidence/task-2-no-print.txt
  ```

  **Commit**: YES
  - Message: `feat(experiments): add token selection experiment runners (1-3)`
  - Files: `src/steering_geometry/token_selection_experiments.py`, `tests/test_token_selection_experiments.py`
  - Pre-commit: `uv run pytest && uv run mypy src/ && uv run ruff check src/ tests/`

- [x] 3. Create Token Count Experiment Script (`1_token_count.sh`)

  **What to do**:
  - Create `scripts/token_experiments/1_token_count.sh`
  - Follow the pattern from `scripts/vector_analysis/run_stability_comparison.sh`
  - Script structure:
    1. `#!/usr/bin/env bash` + `set -euo pipefail`
    2. Default parameters: `CONCEPT="refusal"`, `MODEL="Qwen/Qwen3-1.7B"`, `N_EXAMPLES=(10 30 100 300 1000 3000 6000 10000)`, `LAYERS=(0.4 0.5 0.6 0.7 0.8)`, `OUTPUT_DIR="outputs/token_experiments"`
    3. `while/case` argument parsing with `-c concept -m model -n "10 30 100..." -l "0.4 0.5..." -o output_dir` and `--help`
    4. Convert bash arrays to Python list syntax: `n_examples_str=$(IFS=,; echo "[${N_EXAMPLES[*]}]")`
    5. Invoke `uv run python -u -c "..."` calling `run_token_count_experiment()` with interpolated bash variables
    6. Print summary of output locations
  - Color output for status messages (green=success, yellow=warning) matching existing scripts

  **Must NOT do**:
  - Do NOT put Python logic in the script — only call the experiment function
  - Do NOT hardcode paths — use variables

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single shell script following an existing pattern, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5, 6)
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `scripts/vector_analysis/run_stability_comparison.sh` — Full shell script pattern: shebang, set -euo pipefail, default params, while/case arg parsing, inline Python invocation, color output. COPY THIS STRUCTURE
  - `scripts/vector_analysis/quick_diff_means_heatmaps.sh` — Simpler script pattern for parameter sweeps. Shows array-to-Python-list conversion

  **API/Type References**:
  - `src/steering_geometry/token_selection_experiments.py:run_token_count_experiment()` — The function this script calls (created in Task 2)

  **WHY Each Reference Matters**:
  - `run_stability_comparison.sh`: This is the template. Match its structure exactly — arg parsing style, color codes, Python invocation pattern, error handling

  **Acceptance Criteria**:

  - [ ] File exists at `scripts/token_experiments/1_token_count.sh`
  - [ ] `bash -n scripts/token_experiments/1_token_count.sh` → no syntax errors
  - [ ] `./scripts/token_experiments/1_token_count.sh --help` → prints usage with all flags
  - [ ] Script calls `run_token_count_experiment()` with correct parameters

  **QA Scenarios:**

  ```
  Scenario: Script syntax and help
    Tool: Bash
    Preconditions: Script file exists
    Steps:
      1. Run `bash -n scripts/token_experiments/1_token_count.sh`
      2. Verify exit code 0 (no syntax errors)
      3. Run `bash scripts/token_experiments/1_token_count.sh --help`
      4. Verify output contains "-c", "-m", "-n", "-l", "-o" flags
    Expected Result: Syntax check passes, help output shows all expected flags
    Failure Indicators: bash -n fails, --help doesn't list expected flags
    Evidence: .sisyphus/evidence/task-3-syntax-help.txt
  ```

  **Commit**: YES (groups with Commit 4)
  - Message: `feat(scripts): add token selection experiment shell scripts`
  - Files: `scripts/token_experiments/1_token_count.sh`

- [x] 4. Create Token Position Experiment Script (`2_token_position.sh`)

  **What to do**:
  - Create `scripts/token_experiments/2_token_position.sh`
  - Same structure as Task 3 but for token position comparison
  - Default parameters: `CONCEPT="refusal"`, `MODEL="Qwen/Qwen3-1.7B"`, `N_EXAMPLES=100`, `LAST_N_VALUES=(1 2 3 4 5 10)`, `INCLUDE_ALL=true`, `LAYERS=(0.4 0.5 0.6 0.7 0.8)`, `OUTPUT_DIR="outputs/token_experiments"`
  - Build `position_configs` Python list: `[{"mode": "all"}] + [{"mode": "last_n", "n": N} for N in LAST_N_VALUES]`
  - Invoke `run_token_position_experiment()` with the constructed position_configs

  **Must NOT do**:
  - Do NOT put Python logic in the script beyond constructing the parameter list

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single shell script following Task 3's pattern
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 5, 6)
  - **Parallel Group**: Wave 2
  - **Blocks**: F1-F4
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `scripts/vector_analysis/run_stability_comparison.sh` — Same template as Task 3

  **API/Type References**:
  - `src/steering_geometry/token_selection_experiments.py:run_token_position_experiment()` — The function this script calls

  **WHY Each Reference Matters**:
  - Same template as Task 3 — the only difference is which experiment function and parameters

  **Acceptance Criteria**:

  - [ ] File exists at `scripts/token_experiments/2_token_position.sh`
  - [ ] `bash -n scripts/token_experiments/2_token_position.sh` → no syntax errors
  - [ ] `./scripts/token_experiments/2_token_position.sh --help` → prints usage with all flags
  - [ ] Script constructs position_configs including "all" mode and all last_n values

  **QA Scenarios:**

  ```
  Scenario: Script syntax and help
    Tool: Bash
    Preconditions: Script file exists
    Steps:
      1. Run `bash -n scripts/token_experiments/2_token_position.sh`
      2. Verify exit code 0
      3. Run `bash scripts/token_experiments/2_token_position.sh --help`
      4. Verify output contains "-c", "-m", "-n", "--last-n", "-l", "-o" flags
    Expected Result: Syntax check passes, help shows all flags
    Failure Indicators: bash -n fails, missing flags in help
    Evidence: .sisyphus/evidence/task-4-syntax-help.txt
  ```

  **Commit**: YES (groups with Commit 4)
  - Files: `scripts/token_experiments/2_token_position.sh`

- [x] 5. Create Prompt vs Response Experiment Script (`3_prompt_vs_response.sh`)

  **What to do**:
  - Create `scripts/token_experiments/3_prompt_vs_response.sh`
  - Same structure as Tasks 3-4 but for prompt vs response comparison
  - Default parameters: `CONCEPT="refusal"`, `MODEL="Qwen/Qwen3-1.7B"`, `N_EXAMPLES=100`, `DATA_MODES=("prompt_only" "prompt_response")`, `LAYERS=(0.4 0.5 0.6 0.7 0.8)`, `OUTPUT_DIR="outputs/token_experiments"`
  - Invoke `run_prompt_response_experiment()` with data_modes parameter

  **Must NOT do**:
  - Do NOT put Python logic beyond parameter construction

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single shell script following established pattern
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 4, 6)
  - **Parallel Group**: Wave 2
  - **Blocks**: F1-F4
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `scripts/vector_analysis/run_stability_comparison.sh` — Same template

  **API/Type References**:
  - `src/steering_geometry/token_selection_experiments.py:run_prompt_response_experiment()` — The function this script calls

  **Acceptance Criteria**:

  - [ ] File exists at `scripts/token_experiments/3_prompt_vs_response.sh`
  - [ ] `bash -n scripts/token_experiments/3_prompt_vs_response.sh` → no syntax errors
  - [ ] `./scripts/token_experiments/3_prompt_vs_response.sh --help` → prints usage

  **QA Scenarios:**

  ```
  Scenario: Script syntax and help
    Tool: Bash
    Preconditions: Script file exists
    Steps:
      1. Run `bash -n scripts/token_experiments/3_prompt_vs_response.sh`
      2. Verify exit code 0
      3. Run `bash scripts/token_experiments/3_prompt_vs_response.sh --help`
      4. Verify output contains "-c", "-m", "-n", "--data-modes", "-l", "-o"
    Expected Result: Syntax check passes, help shows all flags
    Evidence: .sisyphus/evidence/task-5-syntax-help.txt
  ```

  **Commit**: YES (groups with Commit 4)
  - Files: `scripts/token_experiments/3_prompt_vs_response.sh`

- [x] 6. Add Steering Scope Experiment Function + Script (`4_steering_scope.sh`)

  **What to do**:
  - Add `run_steering_scope_experiment()` function to `src/steering_geometry/token_selection_experiments.py`
  - **Function: `run_steering_scope_experiment()`**
    - Parameters: `vector_path, model_name, output_dir, steer_tokens_values, layers, multipliers, num_samples, max_new_tokens, temperature`
    - `steer_tokens_values` is a list: `[None, 5, 10, 15, 20]` (None = steer all)
    - Load the pre-extracted steering vector from `vector_path`
    - Load model ONCE as `HookedModel`, reuse across all parameter combinations
    - Load contrast pairs for refusal concept, select negative samples as prompts
    - For each `steer_tokens` value × each layer × each multiplier:
      1. Set `SteeringConfig.steer_tokens = steer_tokens_value`
      2. Call `generate_with_steering()` with the steer_tokens parameter
      3. Save result to `{output_dir}/steered/{concept}/steer_scope/steer_{n}_layer{frac}_mult{m}.jsonl`
    - JSONL format per line: `{"steer_tokens": n, "layer": frac, "multiplier": m, "sample_idx": i, "prompt": "...", "generated_text": "..."}`
    - Return: `{"output_files": [...], "statistics": {...}}`
  - Create `scripts/token_experiments/4_steering_scope.sh`
    - Default: `CONCEPT="refusal"`, `MODEL="Qwen/Qwen3-1.7B"`, `VECTOR_PATH=""` (required, no default), `STEER_TOKENS_VALUES=(5 10 15 20)`, `INCLUDE_FULL=true` (adds None to values), `LAYERS=(0.4 0.5 0.6 0.7 0.8)`, `MULTIPLIERS=(0.01 0.1 1.0 10.0)`, `NUM_SAMPLES=10`, `OUTPUT_DIR="outputs/token_experiments"`
    - First extract a baseline steering vector if VECTOR_PATH not provided: call `extract_vector()` with default params
    - Then invoke `run_steering_scope_experiment()`
  - Write tests:
    - `test_run_steering_scope_experiment_function_exists`: function is importable and callable
    - `test_steering_scope_output_path_generation`: given params, correct paths are generated

  **Must NOT do**:
  - Do NOT implement evaluation logic
  - Do NOT modify `apply_steering.py` directly — the experiment function calls `generate_with_steering()` directly
  - Do NOT use `print()` — use `logging`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integrates the hook modification from Task 1 with experiment logic, requires careful parameter threading
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 4, 5 — but depends on Task 1)
  - **Parallel Group**: Wave 2 (with Tasks 3, 4, 5)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 1 (needs steer_tokens in generate_with_steering)

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py:179-309` — Experiment function pattern
  - `src/steering_geometry/apply_steering.py:540-642` — Steering application logic: how to load vector, compute scale, call generate_with_steering. THIS shows the full steering application flow to replicate

  **API/Type References**:
  - `src/steering_geometry/models.py:194-258` — `generate_with_steering()` — now has `steer_tokens` parameter (from Task 1)
  - `src/steering_geometry/config.py:SteeringConfig` — now has `steer_tokens` field (from Task 1)
  - `src/steering_geometry/types.py:SteeringVector` — The dataclass for loaded steering vectors
  - `src/steering_geometry/extract.py:527` — `extract_vector()` for baseline vector extraction in the script

  **WHY Each Reference Matters**:
  - `apply_steering.py:540-642`: This is the existing steering application flow. The experiment function replicates this but adds the steer_tokens parameter sweep
  - `models.py:194-258`: The function being called with the new parameter
  - `types.py:SteeringVector`: How to load and access the pre-extracted vector

  **Acceptance Criteria**:

  - [ ] `run_steering_scope_experiment()` function exists in `token_selection_experiments.py`
  - [ ] Function loads model once and reuses across all parameter combinations
  - [ ] Output JSONL follows specified format with steer_tokens, layer, multiplier, prompt, generated_text
  - [ ] `steer_tokens_values` includes None (full steering) as baseline
  - [ ] `scripts/token_experiments/4_steering_scope.sh` exists and passes syntax check
  - [ ] `bash -n scripts/token_experiments/4_steering_scope.sh` → no errors
  - [ ] `uv run mypy src/steering_geometry/token_selection_experiments.py` → 0 errors

  **QA Scenarios:**

  ```
  Scenario: Steering scope function is callable
    Tool: Bash (uv run python)
    Preconditions: Module with function exists
    Steps:
      1. Run `uv run python -c "from steering_geometry.token_selection_experiments import run_steering_scope_experiment; print('OK')"`
      2. Verify output contains "OK"
    Expected Result: Function imports without error
    Failure Indicators: ImportError, AttributeError
    Evidence: .sisyphus/evidence/task-6-import-check.txt

  Scenario: Script syntax and help
    Tool: Bash
    Preconditions: Script file exists
    Steps:
      1. Run `bash -n scripts/token_experiments/4_steering_scope.sh`
      2. Verify exit code 0
      3. Run `bash scripts/token_experiments/4_steering_scope.sh --help`
      4. Verify output contains "-v", "-m", "--steer-tokens", "-l", "--multipliers", "-o"
    Expected Result: Syntax check passes, help shows all flags
    Evidence: .sisyphus/evidence/task-6-syntax-help.txt
  ```

  **Commit**: YES (two commits)
  - Commit 3: `feat(experiments): add steering scope experiment (4)` — token_selection_experiments.py, tests
  - Commit 4 (groups with Tasks 3-5): `feat(scripts): add token selection experiment shell scripts` — scripts/token_experiments/

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Review all changed files for: `Any`/`type: ignore`, empty catches, print() in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (experiment functions callable from scripts, hook modification doesn't break existing callers). Test edge cases: steer_tokens=0, steer_tokens>=max_new_tokens. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Commit 1**: `feat(steering): add steer_tokens parameter for prefix-only steering` — config.py, models.py, tests
- **Commit 2**: `feat(experiments): add token selection experiment runners (1-3)` — token_selection_experiments.py, tests
- **Commit 3**: `feat(experiments): add steering scope experiment (4)` — token_selection_experiments.py, tests
- **Commit 4**: `feat(scripts): add token selection experiment shell scripts` — scripts/token_experiments/

---

## Success Criteria

### Verification Commands
```bash
uv sync                                                              # Expected: success
uv run ruff check src/ tests/                                        # Expected: 0 violations
uv run ruff format --check src/ tests/                               # Expected: already formatted
uv run mypy src/                                                     # Expected: 0 errors
uv run pytest                                                        # Expected: all pass
bash -n scripts/token_experiments/1_token_count.sh                   # Expected: no syntax errors
bash -n scripts/token_experiments/2_token_position.sh                # Expected: no syntax errors
bash -n scripts/token_experiments/3_prompt_vs_response.sh            # Expected: no syntax errors
bash -n scripts/token_experiments/4_steering_scope.sh                # Expected: no syntax errors
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Each shell script has --help flag and proper arg parsing
- [ ] steer_tokens=None produces identical behavior to existing code
- [ ] Output directory structure matches convention
