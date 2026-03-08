# TDNV Metrics Implementation

## TL;DR

> **Quick Summary**: Implement TDNV (Topic-Discriminative Normalized Variance) metrics as a separate analysis module that computes separability of pos/neg sets across all layers for 5 behavioral concepts.
>
> **Deliverables**:
> - New `src/steering_geometry/tdnv.py` module with CLI
> - Unit tests in `tests/unit/test_tdnv.py`
> - Visualization in `plot/tdnv/`
> - JSON results in `data/tdnv/`
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Types → Core Logic → Tests → CLI → Visualization

---

## Context

### Original Request
User wants to compute TDNV metrics for steering vector research:
- For each of 5 concepts (honesty, sentiment, toxicity, sycophancy, refusal)
- Treat pos/neg sets as two opposing tasks (T=2)
- Compute TDNV and normalized values per layer per model

### Interview Summary
**Key Discussions**:
- **Integration**: Separate module (not integrated with extraction) - cleaner separation
- **Layer Scope**: ALL layers (0 to num_layers-1) - complete picture
- **Output**: JSON file per concept/model
- **Visualization**: Matplotlib plots showing layer-wise trends
- **Testing**: TDD approach with pytest

**Research Findings**:
- TDNV is similar to inverse Fisher Discriminant Ratio
- Lower TDNV = better separability (easier to steer)
- Requires preserving per-pair activations (cannot use aggregated `extract_steering_vector()`)
- Add epsilon (1e-8) to denominator for numerical stability

### Metis Review
**Identified Gaps** (addressed):
- **Architectural Gap**: Cannot call `extract_steering_vector()` - it aggregates; TDNV needs raw per-pair data → Will implement separate activation collection
- **Model name sanitization**: Reuse `_safe_model_name()` pattern from `apply_steering.py`
- **No custom exceptions**: Use only `ValueError`
- **JSON structure**: Follow existing metadata pattern with concept/model/layers

---

## Work Objectives

### Core Objective
Implement TDNV metrics module that computes topic separability for steering vector analysis across all model layers.

### Concrete Deliverables
- `src/steering_geometry/tdnv.py` - Core TDNV computation module
- `src/steering_geometry/types.py` - Add `TDNVResult` dataclass
- `tests/unit/test_tdnv.py` - Unit tests with mock activations
- CLI: `uv run python -m steering_geometry.tdnv --concept honesty --model Qwen/Qwen3.5-2B`
- Output: `data/tdnv/{concept}_{model}.json`
- Visualization: `plot/tdnv/{concept}_{model}.png`

### Definition of Done
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run mypy src/` → 0 errors
- [ ] `uv run pytest tests/unit/test_tdnv.py` → all pass
- [ ] `uv run python -m steering_geometry.tdnv --concept honesty --dry-run` → loads data
- [ ] TDNV results saved to JSON with correct structure
- [ ] Visualization plots generated correctly

### Must Have
- TDNV computation for all layers (0 to num_layers-1)
- Normalized metrics: NormNum, NormDen
- JSON output with metadata (concept, model, num_pairs, layers)
- Unit tests for TDNV formula correctness
- CLI with `--concept`, `--model`, `--num-pairs`, `--output`, `--dry-run` flags
- **New dependency**: `matplotlib` (for visualization - Task 8)

### Must NOT Have (Guardrails)
- Do NOT modify `extract.py` - TDNV is separate analysis
- Do NOT create custom exception classes - use `ValueError` only
- Do NOT call `extract_steering_vector()` - it aggregates activations
- Do NOT use float16 for variance calculations - use float32
- Do NOT assume specific model architectures - use `HookedModel.num_layers`

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (TDD)
- **Framework**: pytest
- **TDD**: Each task follows RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task includes agent-executed QA scenarios.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (Dependency - First):
└── Task 0: Add matplotlib to pyproject.toml [quick]

Wave 1 (Foundation - Sequential, shared types):
├── Task 1: Add TDNVResult dataclass to types.py [quick]
└── Task 2: Add TDNVConfig to config.py [quick]

Wave 2 (Core Implementation - MAX PARALLEL):
├── Task 3: Implement _compute_per_topic_stats() helper [deep]
├── Task 4: Implement compute_tdnv() core function [deep]
├── Task 5: Implement compute_tdnv_for_concept() orchestrator [deep]
└── Task 6: Write unit tests for TDNV computation [deep]

Wave 3 (CLI + Visualization):
├── Task 7: Implement CLI with argparse [quick]
├── Task 8: Implement visualization (matplotlib) [visual-engineering]
└── Task 9: Integration test with real model [unspecified-high]

Wave FINAL (Verification - 4 parallel):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: T0 → T1 → T3 → T4 → T5 → T7 → T8 → F1-F4
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 4 (Wave 2)
```

### Dependency Matrix

- **0**: — 1, 8
- **1**: 0 — 2
- **2**: 0 — 3, 4, 5
- **3**: 2 — 4
- **4**: 2, 3 — 5, 6
- **5**: 2, 4 — 7, 9
- **6**: 4 — 7
- **7**: 5, 6 — 8, 9
- **8**: 7 — 9
- **9**: 5, 7, 8 — F1-F4

### Agent Dispatch Summary

- **Wave 0**: **1** — T0 → `quick`
- **Wave 1**: **2** — T1 → `quick`, T2 → `quick`
- **Wave 2**: **4** — T3 → `deep`, T4 → `deep`, T5 → `deep`, T6 → `deep`
- **Wave 3**: **3** — T7 → `quick`, T8 → `visual-engineering`, T9 → `unspecified-high`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 0. Add matplotlib to pyproject.toml dependencies

  **What to do**:
  - Add `matplotlib` to `[project].dependencies` in `pyproject.toml`
  - Run `uv sync` to install the new dependency
  - Verify import works: `uv run python -c "import matplotlib.pyplot as plt"`

  **Must NOT do**:
  - Do not pin to a specific version unless necessary
  - Do not add to dev dependencies - this is a runtime requirement for visualization

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dependency addition
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO - foundation for Task 8
  - **Parallel Group**: Wave 0 (alone)
  - **Blocks**: Tasks 1, 8
  - **Blocked By**: None

  **References**:
  - `pyproject.toml:dependencies` - Where to add matplotlib

  **Acceptance Criteria**:
  - [ ] `matplotlib` added to dependencies list
  - [ ] `uv sync` completes successfully
  - [ ] `uv run python -c "import matplotlib.pyplot"` succeeds

  **QA Scenarios**:
  ```
  Scenario: matplotlib import works
    Tool: Bash
    Steps:
      1. Run: uv run python -c "import matplotlib.pyplot as plt; print(plt.__version__)"
      2. Verify version is printed
    Expected Result: Version string printed (e.g., "3.x.x")
    Evidence: .sisyphus/evidence/task-0-matplotlib-import.txt
  ```

  **Commit**: YES
  - Message: `chore: add matplotlib dependency for TDNV visualization`
  - Files: `pyproject.toml`, `uv.lock`

- [ ] 1. Add TDNVResult dataclass to types.py

  **What to do**:
  - Add `TDNVResult` dataclass with fields: `concept`, `model_name`, `num_pairs`, `layers`, `tdnv_values`, `norm_num_values`, `norm_den_values`, `layerwise_energy`
  - Add `TDNVLayerMetrics` dataclass for per-layer breakdown
  - Update `__all__` export list

  **Must NOT do**:
  - Do not add methods to dataclasses (keep them pure data containers)
  - Do not import unnecessary types

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dataclass addition, follows existing patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO - foundation type needed by other tasks
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 2, 3, 4, 5
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/types.py:27-43` - SteeringVector pattern to follow
  - `src/steering_geometry/types.py:9-23` - ContrastPair pattern

  **Acceptance Criteria**:
  - [ ] TDNVResult dataclass defined with all required fields
  - [ ] TDNVLayerMetrics dataclass defined
  - [ ] Both added to `__all__` list
  - [ ] `uv run mypy src/steering_geometry/types.py` → success

  **QA Scenarios**:
  ```
  Scenario: TDNVResult instantiation
    Tool: Bash (uv run python -c)
    Steps:
      1. Import TDNVResult from steering_geometry.types
      2. Create instance with test data
      3. Access all fields
    Expected Result: No errors, all fields accessible
    Evidence: .sisyphus/evidence/task-1-types-import.txt
  ```

  **Commit**: YES
  - Message: `feat(tdnv): add TDNVResult and TDNVLayerMetrics types`
  - Files: `src/steering_geometry/types.py`

- [ ] 2. Add TDNVConfig to config.py

  **What to do**:
  - Add `TDNVConfig` dataclass with fields: `num_pairs`, `batch_size`, `output_dir`, `plot_dir`, `read_token_index`
  - Set sensible defaults (num_pairs=500, batch_size=8, read_token_index=-1)
  - Update `__all__` export list

  **Must NOT do**:
  - Do not duplicate fields from other configs
  - Do not add methods to config dataclass

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple config dataclass, follows existing patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES - with Task 1
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 3, 4, 5
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/config.py:24-38` - ExtractionConfig pattern
  - `src/steering_geometry/config.py:56-73` - SteeringConfig pattern

  **Acceptance Criteria**:
  - [ ] TDNVConfig dataclass defined with all fields
  - [ ] Default values set appropriately
  - [ ] Added to `__all__` list
  - [ ] `uv run mypy src/steering_geometry/config.py` → success

  **QA Scenarios**:
  ```
  Scenario: TDNVConfig default values
    Tool: Bash (uv run python -c)
    Steps:
      1. Import TDNVConfig
      2. Create instance with no args
      3. Verify defaults: num_pairs=500, batch_size=8, read_token_index=-1
    Expected Result: All defaults match expected values
    Evidence: .sisyphus/evidence/task-2-config-defaults.txt
  ```

  **Commit**: YES (grouped with Task 1)
  - Message: `feat(tdnv): add TDNVResult and TDNVConfig types`
  - Files: `src/steering_geometry/config.py`

---

- [ ] 3. Implement _compute_per_topic_stats() helper

  **What to do**:
  - Create helper function that computes per-topic statistics from activations
  - Input: `activations: Tensor` (shape: n_samples, hidden_dim), `topic_labels: list[int]`
  - Output: `dict[int, TopicStats]` with mean, variance, count per topic
  - Use float32 for numerical stability
  - Variance formula: `(1/N) * sum ||h_i - mean||^2`

  **Must NOT do**:
  - Do not use float16 for variance calculations
  - Do not assume only 2 topics - make it general

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core numerical computation, needs careful implementation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES - with Tasks 4, 5, 6 (after Task 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 4
  - **Blocked By**: Task 2 (TDNVConfig)

  **References**:
  - `src/steering_geometry/extract.py:81-92` - Aggregator pattern
  - `src/steering_geometry/apply_steering.py:52-73` - _compute_avg_activation pattern

  **Acceptance Criteria**:
  - [ ] Function signature matches specification
  - [ ] Correctly computes per-topic mean
  - [ ] Correctly computes per-topic variance
  - [ ] Handles edge cases (empty topic, single sample)
  - [ ] `uv run mypy` → success

  **QA Scenarios**:
  ```
  Scenario: Per-topic stats with mock data
    Tool: Bash (uv run python -c)
    Steps:
      1. Create mock activations: topic 0 = [[1,0], [2,0]], topic 1 = [[0,1], [0,2]]
      2. Call _compute_per_topic_stats()
      3. Verify topic 0 mean = [1.5, 0], topic 1 mean = [0, 1.5]
      4. Verify variances computed correctly
    Expected Result: Means and variances match expected values
    Evidence: .sisyphus/evidence/task-3-per-topic-stats.txt
  ```

  **Commit**: NO (grouped with Task 4)

- [ ] 4. Implement compute_tdnv() core function

  **What to do**:
  - Implement main TDNV computation function
  - Input: `pos_activations: Tensor`, `neg_activations: Tensor`
  - Output: `TDNVLayerMetrics` with tdnv, norm_num, norm_den, energy
  - Formula: TDNV = sum over topic pairs of (var_t + var_t') / (2 * ||mean_t - mean_t'||^2 + eps)
  - Add epsilon (1e-8) for numerical stability
  - Compute layerwise activation energy: s = (1/N) * sum ||h||^2

  **Must NOT do**:
  - Do not forget epsilon in denominator
  - Do not use float16

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core TDNV formula implementation, critical correctness
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES - with Tasks 5, 6
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Task 2 (TDNVConfig), Task 3 (helper)

  **References**:
  - User's formula slide: TDNV and normalized formulas
  - `src/steering_geometry/types.py:TDNVLayerMetrics` (from Task 1)

  **Acceptance Criteria**:
  - [ ] Correctly implements TDNV formula
  - [ ] Correctly computes NormNum (avg normalized within-topic variance)
  - [ ] Correctly computes NormDen (avg normalized between-topic distance)
  - [ ] Correctly computes layerwise energy s
  - [ ] Adds epsilon to denominator
  - [ ] `uv run mypy` → success

  **QA Scenarios**:
  ```
  Scenario: TDNV with well-separated topics
    Tool: Bash (uv run python -c)
    Steps:
      1. Create pos_activations clustered at [1,0], neg at [0,1]
      2. Call compute_tdnv()
      3. Verify TDNV is LOW (good separability)
    Expected Result: TDNV < 1.0 (well-separated)
    Evidence: .sisyphus/evidence/task-4-tdnv-separated.txt

  Scenario: TDNV with overlapping topics
    Tool: Bash (uv run python -c)
    Steps:
      1. Create pos and neg activations both near [0.5, 0.5]
      2. Call compute_tdnv()
      3. Verify TDNV is HIGH (poor separability)
    Expected Result: TDNV > 10.0 (overlapping)
    Evidence: .sisyphus/evidence/task-4-tdnv-overlapping.txt
  ```

  **Commit**: YES
  - Message: `feat(tdnv): implement core TDNV computation`
  - Files: `src/steering_geometry/tdnv.py`
  - Pre-commit: `uv run pytest tests/unit/test_tdnv.py -k "test_compute_tdnv"`

- [ ] 5. Implement compute_tdnv_for_concept() orchestrator

  **What to do**:
  - Create orchestrator function that ties everything together
  - Load contrast pairs using `load_contrast_pairs(concept, num_pairs)`
  - Load model using `HookedModel(ModelConfig(model_name))`
  - Get ALL layers: `list(range(model.num_layers))`
  - Extract activations for pos and neg texts in batches
  - Call `compute_tdnv()` for each layer
  - Return `TDNVResult` with all metrics

  **Must NOT do**:
  - Do NOT call `extract_steering_vector()` - it aggregates activations
  - Do not use default extraction layers - must use ALL layers
  - Do not assume specific model architecture

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex orchestration, needs to correctly wire all components
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES - with Task 6
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 7, 9
  - **Blocked By**: Task 2 (TDNVConfig), Task 4 (compute_tdnv)

  **References**:
  - `src/steering_geometry/extract.py:411-477` - extract_steering_vector pattern (activation collection)
  - `src/steering_geometry/extract.py:436-461` - Batch processing pattern
  - `src/steering_geometry/models.py:74-88` - resolve_layers (but use ALL layers)
  - `src/steering_geometry/models.py:101-150` - get_activations signature

  **Acceptance Criteria**:
  - [ ] Correctly loads contrast pairs
  - [ ] Uses ALL layers (0 to num_layers-1)
  - [ ] Processes in batches for memory efficiency
  - [ ] Preserves per-pair activations (does not aggregate)
  - [ ] Calls compute_tdnv() for each layer
  - [ ] Returns complete TDNVResult
  - [ ] `uv run mypy` → success

  **QA Scenarios**:
  ```
  Scenario: Full TDNV computation for concept
    Tool: Bash (uv run python -c)
    Steps:
      1. Call compute_tdnv_for_concept("honesty", "sshleifer/tiny-gpt2", num_pairs=5)
      2. Verify result has metrics for ALL layers
      3. Verify tdnv_values length == model.num_layers
      4. Verify all values are positive finite numbers
    Expected Result: TDNVResult with metrics for all layers
    Evidence: .sisyphus/evidence/task-5-full-computation.txt
  ```

  **Commit**: NO (grouped with Task 4)

- [ ] 6. Write unit tests for TDNV computation

  **What to do**:
  - Create `tests/unit/test_tdnv.py`
  - Test `_compute_per_topic_stats()` with mock data
  - Test `compute_tdnv()` with well-separated and overlapping topics
  - Test edge cases: single sample, empty topic, identical topics
  - Test normalization correctness
  - Follow pytest patterns from `tests/unit/test_extract.py`

  **Must NOT do**:
  - Do not skip edge case tests
  - Do not use real model in unit tests - use mock tensors

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Comprehensive testing requires understanding of all edge cases
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES - with Task 5
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 7
  - **Blocked By**: Task 4 (compute_tdnv)

  **References**:
  - `tests/unit/test_extract.py` - Test patterns
  - `tests/conftest.py` - Fixture patterns

  **Acceptance Criteria**:
  - [ ] test_compute_per_topic_stats_basic() passes
  - [ ] test_compute_tdnv_well_separated() passes
  - [ ] test_compute_tdnv_overlapping() passes
  - [ ] test_compute_tdnv_edge_cases() passes
  - [ ] test_normalization() passes
  - [ ] `uv run pytest tests/unit/test_tdnv.py -v` → all pass

  **QA Scenarios**:
  ```
  Scenario: All unit tests pass
    Tool: Bash (pytest)
    Steps:
      1. Run uv run pytest tests/unit/test_tdnv.py -v
      2. Verify all tests pass
      3. Verify coverage includes all functions
    Expected Result: X passed, 0 failed
    Evidence: .sisyphus/evidence/task-6-tests-pass.txt
  ```

  **Commit**: YES
  - Message: `test(tdnv): add unit tests for TDNV computation`
  - Files: `tests/unit/test_tdnv.py`
  - Pre-commit: `uv run pytest tests/unit/test_tdnv.py`

---

- [ ] 7. Implement CLI with argparse

  **What to do**:
  - Create CLI entry point in `tdnv.py` with `if __name__ == "__main__"`
  - Arguments: `--concept`, `--model`, `--num-pairs`, `--output`, `--plot-dir`, `--dry-run`
  - Validate concept is in VALID_CONCEPTS
  - Call `compute_tdnv_for_concept()`
  - Save JSON output to `data/tdnv/{concept}_{model_sanitized}.json`
  - Create output directories if needed

  **Must NOT do**:
  - Do not create custom exceptions - use ValueError
  - Do not forget to sanitize model name for filename

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard CLI pattern, follows extract.py example
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO - depends on Tasks 5, 6
  - **Parallel Group**: Wave 3 (with Tasks 8, 9)
  - **Blocks**: Tasks 8, 9
  - **Blocked By**: Task 5 (orchestrator), Task 6 (tests)

  **References**:
  - `src/steering_geometry/extract.py:533-611` - CLI pattern
  - `src/steering_geometry/extract.py:601-609` - File saving pattern
  - `src/steering_geometry/apply_steering.py:_safe_model_name` - Model name sanitization

  **Acceptance Criteria**:
  - [ ] CLI accepts all required arguments
  - [ ] Validates concept against VALID_CONCEPTS
  - [ ] Creates output directory if needed
  - [ ] Saves JSON with correct filename format
  - [ ] `--dry-run` loads data without running model
  - [ ] `uv run python -m steering_geometry.tdnv --help` works

  **QA Scenarios**:
  ```
  Scenario: CLI dry-run
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.tdnv --concept honesty --model sshleifer/tiny-gpt2 --dry-run
      2. Verify it loads contrast pairs
      3. Verify it does not load model
    Expected Result: "Dry run complete" message, no model loading
    Evidence: .sisyphus/evidence/task-7-cli-dry-run.txt

  Scenario: Invalid concept
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.tdnv --concept invalid_concept --model x --dry-run
      2. Verify error message
    Expected Result: ValueError with "Invalid concept" message
    Evidence: .sisyphus/evidence/task-7-cli-invalid.txt
  ```

  **Commit**: YES
  - Message: `feat(tdnv): add CLI for TDNV analysis`
  - Files: `src/steering_geometry/tdnv.py`

- [ ] 8. Implement visualization (matplotlib)

  **What to do**:
  - Create visualization function `plot_tdnv_trends()`
  - Plot TDNV vs layer index
  - Plot NormNum and NormDen on same or separate axes
  - Add layerwise energy as secondary y-axis or subplot
  - Save to `plot/tdnv/{concept}_{model_sanitized}.png`
  - Use clear labels, legend, title

  **Must NOT do**:
  - Do not use fancy styling that breaks matplotlib defaults
  - Do not create interactive plots - static PNG only

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Visualization requires matplotlib expertise
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO - depends on Tasks 0, 7
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 9
  - **Blocked By**: Task 0 (matplotlib), Task 7 (CLI)

  **References**:
  - Check existing plot patterns in project if any

  **Acceptance Criteria**:
  - [ ] Creates figure with TDNV trend across layers
  - [ ] Includes NormNum and NormDen trends
  - [ ] Clear axis labels (Layer Index, TDNV/NormNum/NormDen)
  - [ ] Title includes concept and model name
  - [ ] Saves PNG to correct path
  - [ ] `uv run python -m steering_geometry.tdnv --concept honesty --model x --num-pairs 5` generates plot

  **QA Scenarios**:
  ```
  Scenario: Plot generation
    Tool: Bash
    Steps:
      1. Run TDNV CLI with small model
      2. Check plot/tdnv/ directory for PNG file
      3. Verify file is valid image (non-zero size)
    Expected Result: PNG file exists with size > 0
    Evidence: .sisyphus/evidence/task-8-plot-generated.txt
  ```

  **Commit**: YES
  - Message: `feat(tdnv): add visualization for TDNV trends`
  - Files: `src/steering_geometry/tdnv.py`

- [ ] 9. Integration test with real model

  **What to do**:
  - Test full pipeline with `sshleifer/tiny-gpt2` (smallest model)
  - Verify JSON output structure and content
  - Verify plot is generated correctly
  - Test all 5 concepts work
  - Document expected output format in tests

  **Must NOT do**:
  - Do not use large model for integration test - too slow
  - Do not skip JSON validation

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration testing requires careful validation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO - final verification
  - **Parallel Group**: Wave 3 (after Tasks 7, 8)
  - **Blocks**: Final Verification Wave
  - **Blocked By**: Tasks 5, 7, 8

  **References**:
  - All previous tasks

  **Acceptance Criteria**:
  - [ ] Full pipeline works with tiny-gpt2
  - [ ] JSON has correct structure: concept, model_name, num_pairs, layers, tdnv_values, norm_num_values, norm_den_values, layerwise_energy
  - [ ] All 5 concepts run without error
  - [ ] Plots generated for each concept

  **QA Scenarios**:
  ```
  Scenario: Full integration with tiny-gpt2
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.tdnv --concept honesty --model sshleifer/tiny-gpt2 --num-pairs 5
      2. Verify JSON exists: data/tdnv/honesty_sshleifer_tiny-gpt2.json
      3. Verify plot exists: plot/tdnv/honesty_sshleifer_tiny-gpt2.png
      4. Load JSON and verify structure
    Expected Result: Both files exist with correct content
    Evidence: .sisyphus/evidence/task-9-integration.txt
  ```

  **Commit**: YES (if separate changes needed)
  - Message: `test(tdnv): add integration test`
  - Files: `tests/integration/test_tdnv_integration.py` (if created)

---

## Final Verification Wave (MANDATORY)

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify all "Must Have" items present, all "Must NOT Have" absent. Check evidence files.

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `ruff check`, `mypy`, `pytest`. No `as any`, no console.log, no unused imports.

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute TDNV CLI with real model. Verify JSON output structure and plot generation.

- [ ] F4. **Scope Fidelity Check** — `deep`
  Verify no scope creep. Check each task's diff matches its specification.

---

## Commit Strategy

- **Commit 0**: `chore: add matplotlib dependency for TDNV visualization`
- **Commit 1**: `feat(tdnv): add TDNVResult type and TDNVConfig`
- **Commit 2**: `feat(tdnv): implement core TDNV computation functions`
- **Commit 3**: `test(tdnv): add unit tests for TDNV computation`
- **Commit 4**: `feat(tdnv): add CLI for TDNV analysis`
- **Commit 5**: `feat(tdnv): add visualization for TDNV trends`

---

## Success Criteria

### Verification Commands
```bash
# Type check
uv run mypy src/steering_geometry/tdnv.py

# Lint
uv run ruff check src/steering_geometry/tdnv.py tests/unit/test_tdnv.py

# Format check
uv run ruff format --check src/steering_geometry/tdnv.py

# Unit tests
uv run pytest tests/unit/test_tdnv.py -v

# CLI dry-run
uv run python -m steering_geometry.tdnv --concept honesty --model sshleifer/tiny-gpt2 --dry-run

# Full run (small model)
uv run python -m steering_geometry.tdnv --concept honesty --model sshleifer/tiny-gpt2 --num-pairs 10
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] CLI works with all 5 concepts
- [ ] Visualization generates correctly
- [ ] JSON output matches expected structure
