# Steering Vector Cosine Similarity Experiments

## TL;DR

> **Quick Summary**: Create two experiments to analyze steering vector stability across different extraction parameters. Experiment 1 varies example counts using difference-in-means. Experiment 2 varies K values using discriminative token selection. Each generates cosine similarity heatmaps saved as PDFs.
>
> **Deliverables**:
> - New module: `src/steering_geometry/experiments.py` with core analysis functions
> - Experiment scripts: `scripts/experiments/run_diff_means_heatmaps.sh` and `run_discriminative_heatmaps.sh`
> - Unit tests: `tests/test_experiments.py` with GPU markers
> - Documentation: Updated AGENTS.md with experiment commands
> - Output: 50 PDF heatmaps (25 per experiment) + 325 saved vectors (200 + 125)
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 5 waves with clear dependencies
> **Critical Path**: Validation → Module → Tests → Exp1 → Exp2 → Docs

---

## Context

### Original Request
Run two experiments on Qwen3-1.7B using all 5 concepts (honesty, sentiment, toxicity, sycophancy, refusal):
1. Extract steering vectors using difference-in-means with varying example counts [10, 30, 100, 300, 1000, 3000, 6000, 10000], compute N×N cosine similarity matrices, visualize as PDF heatmaps
2. Extract vectors using discriminative token method with varying K values [16, 32, 64, 128, 256], compute N×N cosine similarity matrices, visualize as PDF heatmaps

### Interview Summary
**Key Discussions**:
- Example counts: [10, 30, 100, 300, 1000, 3000, 6000, 10000] (capped at dataset maximums)
- K values: [16, 32, 64, 128, 256] for discriminative token selection
- Layers: [0.4, 0.5, 0.6, 0.7, 0.8] (per-layer heatmaps, not aggregated)
- Heatmap structure: Self-similarity matrix (both axes = parameter values)
- Implementation: New module + scripts, reuse existing extraction infrastructure
- Test strategy: GPU markers for model-dependent tests, unit tests for math, tmp_path for PDFs

**Research Findings**:
- Visualization: Only matplotlib available (no seaborn), follow tdnv.py pattern
- Cosine similarity: Not implemented, will use sklearn.metrics.pairwise.cosine_similarity
- Dataset limits: honesty=800, sycophancy=4000, refusal=1000, others support 10000+
- Test infrastructure: pytest with @pytest.mark.gpu and @pytest.mark.slow markers

### Metis Review
**Identified Gaps** (addressed):
- Layer handling: Per-layer heatmaps (5 per concept per experiment) → 50 total PDFs
- Dataset capping: Silent cap with logging when n_examples exceeds dataset size
- GPU memory: Use existing batch processing from extract.py, no changes needed
- Vector format: `.pt` format with naming pattern `{concept}/{method}/{param}_layer{frac}.pt`
- Validation: Add Wave 0 to verify datasets, methods, and model before implementation

---

## Work Objectives

### Core Objective
Create reusable experiment infrastructure to analyze steering vector stability across extraction parameters, enabling systematic comparison of different extraction strategies.

### Concrete Deliverables
- `src/steering_geometry/experiments.py` - Core analysis functions (cosine similarity, heatmap plotting, experiment orchestration)
- `tests/test_experiments.py` - Unit tests + GPU integration tests
- `scripts/experiments/run_diff_means_heatmaps.sh` - Orchestration for Experiment 1
- `scripts/experiments/run_discriminative_heatmaps.sh` - Orchestration for Experiment 2
- `outputs/vectors/` - 325 saved steering vectors
- `outputs/heatmaps/` - 50 PDF heatmaps

### Definition of Done
- [ ] All code passes: `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/`
- [ ] All tests pass: `uv run pytest tests/test_experiments.py -v`
- [ ] Verification commands produce expected file counts
- [ ] AGENTS.md updated with experiment documentation

### Must Have
- Cosine similarity computation using sklearn (unit tested)
- Matplotlib heatmap generation (follows tdnv.py pattern)
- Vector persistence with clear naming convention
- Experiment orchestration functions for both methods
- Shell scripts to run experiments for all concepts
- Unit tests for all core functions (GPU-independent)
- GPU integration tests with proper markers

### Must NOT Have (Guardrails)
- DO NOT modify existing extraction logic in `extract.py`
- DO NOT add seaborn dependency (matplotlib only)
- DO NOT add new CLI arguments to existing scripts
- DO NOT create cross-experiment comparison (keep experiments isolated)
- DO NOT add parallel processing (sequential execution)
- DO NOT add statistical significance testing beyond cosine similarity
- DO NOT create interactive dashboards (static PDFs only)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest >=8.0.0)
- **Automated tests**: YES (with GPU markers)
- **Framework**: pytest
- **Pattern**: TDD - unit tests for math/logic, GPU tests for model-dependent code

### QA Policy
Every task includes agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/Module**: Use Bash (uv run pytest) - Run tests, verify pass/fail counts
- **Visualization**: Use Bash (pdfinfo, ls) - Verify PDF creation, check file metadata
- **Integration**: Use Bash (uv run python -c) - Run extraction commands, verify outputs

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (Validation - Start Immediately, 1 task):
└── Task 0: Validate assumptions (datasets, methods, model) [quick]

Wave 1 (Foundation - After Wave 0, 2 tasks sequential):
├── Task 1: Create experiments module with core functions [quick]
└── Task 2: Write unit tests for experiments module [quick]

Wave 2 (Experiment 1 - After Wave 1, 3 tasks sequential):
├── Task 3: Implement Experiment 1 (difference-in-means) [quick]
├── Task 4: Add GPU integration tests [quick]
└── Task 5: Create shell script for Experiment 1 [quick]

Wave 3 (Experiment 2 - After Wave 2, 2 tasks sequential):
├── Task 6: Implement Experiment 2 (discriminative) [quick]
└── Task 7: Create shell script for Experiment 2 [quick]

Wave 4 (Finalization - After Wave 3, 2 tasks sequential):
├── Task 8: Update AGENTS.md with experiment commands [quick]
└── Task 9: Final verification and commit [quick]

Critical Path: 0 → 1 → 2 → 3 → 4 → 6 → 8 → 9
Parallel Speedup: Minimal (sequential dependencies dominate)
Max Concurrent: 1 (most tasks depend on previous)
```

### Dependency Matrix

- **0**: — — 1, 1
- **1**: 0 — 2, 2
- **2**: 1 — 3, 3
- **3**: 2 — 4, 4
- **4**: 2 — 5, 5
- **5**: 3 — 6, 6
- **6**: 5 — 7, 7
- **7**: 6 — 8, 8
- **8**: 7 — 9, 9
- **9**: 8 — —, —

### Agent Dispatch Summary

- **All tasks**: `quick` category - straightforward implementation, follow existing patterns

---

## TODOs

- [ ] 0. Validate Assumptions

  **What to do**:
  - Run validation commands to confirm datasets, discriminative method, and model loading work
  - Verify all 5 concepts have loaders in `_DATASET_LOADERS`
  - Test discriminative token aggregator exists and is callable
  - Test model loading with Qwen/Qwen3-1.7B (dry run)
  - Test dataset loading for each concept (small sample)

  **Must NOT do**:
  - DO NOT modify any existing code
  - DO NOT create new files
  - DO NOT run full extractions (just validation)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple validation commands, no implementation work
  - **Skills**: []
    - None needed - just running validation commands

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 0 (Sequential)
  - **Blocks**: Tasks 1-9 (all depend on validation)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `src/steering_geometry/extract.py:417-423` - Dataset loader registry
  - `src/steering_geometry/extract.py:142-177` - Discriminative token aggregator implementation
  - `src/steering_geometry/models.py:20-50` - HookedModel class

  **Acceptance Criteria**:
  - [ ] All 5 concepts found in `_DATASET_LOADERS`
  - [ ] Discriminative method exists and is callable
  - [ ] Model loads successfully (or fails with clear error if not available)
  - [ ] All datasets load small samples (10 pairs) successfully

  **QA Scenarios**:
  ```
  Scenario: Validate dataset loaders exist
    Tool: Bash (grep)
    Preconditions: Repository in clean state
    Steps:
      1. Run: grep -E "(honesty|sentiment|toxicity|sycophancy|refusal)" src/steering_geometry/extract.py | grep "_DATASET_LOADERS"
      2. Count matches (expect 5)
    Expected Result: 5 concept loaders found in registry
    Evidence: .sisyphus/evidence/task-0-validate-loaders.txt
  
  Scenario: Validate discriminative method
    Tool: Bash (grep)
    Steps:
      1. Run: grep -A5 "discriminative_token_aggregator" src/steering_geometry/extract.py
      2. Verify function signature exists
    Expected Result: Function definition found with correct signature
    Evidence: .sisyphus/evidence/task-0-discriminative-method.txt
  
  Scenario: Test dataset loading
    Tool: Bash (uv run python)
    Steps:
      1. Run: uv run python -c "from steering_geometry.extract import load_contrast_pairs; pairs = load_contrast_pairs('honesty', 10); print(f'Loaded {len(pairs)} pairs')"
      2. Repeat for all 5 concepts
    Expected Result: All datasets load successfully
    Evidence: .sisyphus/evidence/task-0-dataset-loading.txt
  ```

  **Commit**: NO
  - This is validation only, no changes to commit

- [ ] 1. Create experiments module with core functions

  **What to do**:
  - Create new file: `src/steering_geometry/experiments.py`
  - Implement core functions:
    - `compute_cosine_similarity_matrix(vectors: list[Tensor]) -> ndarray` - Uses sklearn.metrics.pairwise.cosine_similarity
    - `plot_heatmap(matrix: ndarray, labels: list[str], title: str, output_path: Path) -> Path` - Matplotlib heatmap generation following tdnv.py pattern
    - `save_vector(vector: Tensor, path: Path) -> None` - Save tensor to .pt file
    - `load_vector(path: Path) -> Tensor` - Load tensor from .pt file
    - `cap_examples(requested: int, max_available: int, concept: str) -> int` - Cap with logging
  - Add proper type hints, docstrings, and error handling
  - Follow ruff/mypy standards from AGENTS.md

  **Must NOT do**:
  - DO NOT import seaborn (matplotlib only)
  - DO NOT modify existing modules
  - DO NOT add new dependencies to pyproject.toml

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: New module with straightforward utility functions
  - **Skills**: []
    - None needed - standard Python implementation

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (Sequential, after Task 0)
  - **Blocks**: Tasks 2-9
  - **Blocked By**: Task 0

  **References**:
  - `src/steering_geometry/tdnv.py:244-316` - Matplotlib visualization pattern to follow
  - `src/steering_geometry/utils.py:ensure_dir` - Directory creation utility
  - `src/steering_geometry/utils.py:safe_model_name` - Model name sanitization
  - sklearn docs: `https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity`

  **Acceptance Criteria**:
  - [ ] File created: `src/steering_geometry/experiments.py`
  - [ ] All 5 core functions implemented with type hints
  - [ ] Passes: `uv run ruff check src/steering_geometry/experiments.py`
  - [ ] Passes: `uv run mypy src/steering_geometry/experiments.py`

  **QA Scenarios**:
  ```
  Scenario: Module imports successfully
    Tool: Bash (uv run python)
    Steps:
      1. Run: uv run python -c "from steering_geometry.experiments import compute_cosine_similarity_matrix, plot_heatmap, save_vector, load_vector; print('OK')"
    Expected Result: Import succeeds without errors
    Evidence: .sisyphus/evidence/task-1-module-import.txt
  
  Scenario: Type checking passes
    Tool: Bash (uv run mypy)
    Steps:
      1. Run: uv run mypy src/steering_geometry/experiments.py
      2. Check for errors
    Expected Result: Success with 0 errors
    Evidence: .sisyphus/evidence/task-1-mypy-check.txt
  ```

  **Commit**: NO
  - Commit together with Task 2 (tests)

- [ ] 2. Write unit tests for experiments module

  **What to do**:
  - Create new file: `tests/test_experiments.py`
  - Implement unit tests (no GPU required):
    - `test_cosine_similarity_computation()` - Test with known vectors, verify expected similarities (identity matrix, orthogonal vectors)
    - `test_heatmap_generation()` - Use tmp_path, verify PDF created with valid header
    - `test_vector_save_load()` - Save and load tensor, verify roundtrip preservation
    - `test_dataset_size_capping()` - Test cap_examples function with various inputs
  - Add GPU integration tests (marked):
    - `@pytest.mark.gpu @pytest.mark.slow test_single_extraction_experiment1()` - Extract one vector with real model
    - `@pytest.mark.gpu @pytest.mark.slow test_single_extraction_experiment2()` - Extract one discriminative vector
  - Follow test patterns from existing tests (class-based, type hints, fixtures)

  **Must NOT do**:
  - DO NOT require GPU for unit tests
  - DO NOT create tests that need manual verification
  - DO NOT skip tests for core functionality

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard pytest test implementation
  - **Skills**: []
    - None needed - standard testing patterns

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (Sequential, after Task 1)
  - **Blocks**: Tasks 3-9
  - **Blocked By**: Task 1

  **References**:
  - `tests/conftest.py:158-180` - mock_hooked_model fixture pattern
  - `tests/unit/test_aggregators.py` - Pure tensor unit test pattern
  - `tests/unit/test_evaluation.py` - tmp_path usage pattern
  - pytest docs: `https://docs.pytest.org/en/stable/how-to/tmp_path.html`

  **Acceptance Criteria**:
  - [ ] File created: `tests/test_experiments.py`
  - [ ] 4 unit tests implemented and passing
  - [ ] 2 GPU integration tests implemented (marked)
  - [ ] Passes: `uv run pytest tests/test_experiments.py -v -k "not gpu"`
  - [ ] Unit tests run without GPU

  **QA Scenarios**:
  ```
  Scenario: Unit tests pass without GPU
    Tool: Bash (uv run pytest)
    Steps:
      1. Run: uv run pytest tests/test_experiments.py -v -k "not gpu"
      2. Check all tests pass
    Expected Result: 4 passed, 0 failed
    Evidence: .sisyphus/evidence/task-2-unit-tests.txt
  
  Scenario: PDF test creates valid output
    Tool: Bash (uv run pytest)
    Steps:
      1. Run: uv run pytest tests/test_experiments.py::test_heatmap_generation -v
      2. Verify test passes
    Expected Result: Test passes, PDF has valid header
    Evidence: .sisyphus/evidence/task-2-pdf-test.txt
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `feat(experiments): add cosine similarity and heatmap visualization functions`
  - Files: `src/steering_geometry/experiments.py`, `tests/test_experiments.py`
  - Pre-commit: `uv run pytest tests/test_experiments.py -v -k "not gpu"`

- [ ] 3. Implement Experiment 1 (difference-in-means)

  **What to do**:
  - Add to `src/steering_geometry/experiments.py`:
    - `run_diff_means_experiment(concept: str, n_examples_list: list[int], layers: list[float], model_name: str, output_dir: Path) -> dict`
    - Orchestrates: load data → cap n_examples → extract vectors → compute similarities → plot heatmaps
    - Extract vectors for each n_examples value using `extract_vector()` with method="mean"
    - Save vectors to `outputs/vectors/{concept}/diff_means/n{count}_layer{frac}.pt`
    - Compute N×N cosine similarity matrix for each layer
    - Generate heatmap: `outputs/heatmaps/diff_means/{concept}_layer{frac}.pdf`
    - Return summary dict with file paths and statistics
  - Handle edge cases:
    - Dataset smaller than requested (cap and log)
    - Identical vectors (similarity = 1.0, handle in heatmap)
    - NaN in vectors (validate and skip)

  **Must NOT do**:
  - DO NOT modify extract.py extraction logic
  - DO NOT add parallel processing
  - DO NOT create cross-concept comparisons

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Orchestration function using existing infrastructure
  - **Skills**: []
    - None needed - calling existing functions

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (Sequential, after Task 2)
  - **Blocks**: Tasks 4-9
  - **Blocked By**: Task 2

  **References**:
  - `src/steering_geometry/extract.py:527-563` - extract_vector API
  - `src/steering_geometry/config.py:ExtractionConfig` - Configuration class
  - `src/steering_geometry/types.py:SteeringVector` - Steering vector type

  **Acceptance Criteria**:
  - [ ] Function implemented in experiments.py
  - [ ] Passes: `uv run ruff check src/steering_geometry/experiments.py`
  - [ ] Passes: `uv run mypy src/steering_geometry/experiments.py`
  - [ ] Manual test: Run for single concept with reduced params produces expected outputs

  **QA Scenarios**:
  ```
  Scenario: Single concept extraction produces outputs
    Tool: Bash (uv run python)
    Steps:
      1. Run: uv run python -c "from steering_geometry.experiments import run_diff_means_experiment; from pathlib import Path; run_diff_means_experiment('honesty', [10, 30], [0.5], 'Qwen/Qwen3-1.7B', Path('outputs'))"
      2. Check outputs exist
    Expected Result: 1 PDF and 2 .pt files created
    Evidence: .sisyphus/evidence/task-3-single-concept.txt
  
  Scenario: Dataset capping works correctly
    Tool: Bash (uv run python)
    Steps:
      1. Run: uv run python -c "from steering_geometry.experiments import run_diff_means_experiment; from pathlib import Path; run_diff_means_experiment('honesty', [10, 10000], [0.5], 'Qwen/Qwen3-1.7B', Path('outputs'))"
      2. Check logs show capping to 800
    Expected Result: Warning logged about capping to dataset max
    Evidence: .sisyphus/evidence/task-3-capping.txt
  ```

  **Commit**: NO
  - Commit together with Tasks 4-5 (Experiment 1 complete)

- [ ] 4. Add GPU integration tests

  **What to do**:
  - Add to `tests/test_experiments.py`:
    - `@pytest.mark.gpu @pytest.mark.slow test_single_extraction_experiment1()` - Run single extraction with real model, verify vector shape and file creation
    - `@pytest.mark.gpu @pytest.mark.slow test_single_extraction_experiment2()` - Run single discriminative extraction, verify outputs
  - Skip tests if GPU not available: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")`
  - Use small parameters to keep tests fast (n_examples=10, single layer)

  **Must NOT do**:
  - DO NOT run full experiments in tests (too slow)
  - DO NOT skip tests entirely
  - DO NOT remove @pytest.mark.gpu

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding test cases to existing test file
  - **Skills**: []
    - None needed - standard pytest

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (Sequential, after Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 3

  **References**:
  - `tests/conftest.py:116-118` - pytest marker registration
  - `tests/unit/test_extract.py:111` - Skipif pattern example

  **Acceptance Criteria**:
  - [ ] 2 GPU tests added to test_experiments.py
  - [ ] Both marked with @pytest.mark.gpu and @pytest.mark.slow
  - [ ] Both include skipif for GPU availability
  - [ ] Passes: `uv run pytest tests/test_experiments.py -m gpu -v` (if GPU available)

  **QA Scenarios**:
  ```
  Scenario: GPU tests properly marked
    Tool: Bash (grep)
    Steps:
      1. Run: grep -c "@pytest.mark.gpu" tests/test_experiments.py
      2. Expect count >= 2
    Expected Result: At least 2 GPU markers found
    Evidence: .sisyphus/evidence/task-4-gpu-markers.txt
  
  Scenario: GPU tests run successfully (if GPU available)
    Tool: Bash (uv run pytest)
    Steps:
      1. Run: uv run pytest tests/test_experiments.py -m gpu -v
      2. Check results
    Expected Result: Tests pass or skip (if no GPU)
    Evidence: .sisyphus/evidence/task-4-gpu-tests.txt
  ```

  **Commit**: NO
  - Commit together with Tasks 3 and 5 (Experiment 1 complete)

- [ ] 5. Create shell script for Experiment 1

  **What to do**:
  - Create directory: `scripts/experiments/`
  - Create script: `scripts/experiments/run_diff_means_heatmaps.sh`
  - Script runs experiment for all 5 concepts with all parameters:
    - Example counts: [10, 30, 100, 300, 1000, 3000, 6000, 10000] (capped per concept)
    - Layers: [0.4, 0.5, 0.6, 0.7, 0.8]
    - Model: Qwen/Qwen3-1.7B
  - Logs progress and outputs
  - Uses uv run for Python execution
  - Follows bash best practices (set -e, proper quoting)

  **Must NOT do**:
  - DO NOT hardcode GPU-specific settings
  - DO NOT add parallel execution (run concepts sequentially)
  - DO NOT modify existing scripts

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple bash script orchestrating Python calls
  - **Skills**: [`git-master`]
    - git-master: For proper bash script conventions and error handling

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (Sequential, after Task 4)
  - **Blocks**: Tasks 6-9
  - **Blocked By**: Task 4

  **References**:
  - `scripts/run_pipeline.sh` - Existing script pattern
  - `AGENTS.md` - Project conventions

  **Acceptance Criteria**:
  - [ ] Directory created: `scripts/experiments/`
  - [ ] Script created: `scripts/experiments/run_diff_means_heatmaps.sh`
  - [ ] Script syntax valid: `bash -n scripts/experiments/run_diff_means_heatmaps.sh`
  - [ ] Script has execute permission

  **QA Scenarios**:
  ```
  Scenario: Script syntax is valid
    Tool: Bash (bash -n)
    Steps:
      1. Run: bash -n scripts/experiments/run_diff_means_heatmaps.sh
      2. Check for syntax errors
    Expected Result: No output (syntax OK)
    Evidence: .sisyphus/evidence/task-5-syntax-check.txt
  
  Scenario: Script has execute permission
    Tool: Bash (ls)
    Steps:
      1. Run: ls -l scripts/experiments/run_diff_means_heatmaps.sh
      2. Check for execute bit
    Expected Result: Script has -rwxr-xr-x or similar
    Evidence: .sisyphus/evidence/task-5-permissions.txt
  ```

  **Commit**: YES (groups with Tasks 3-4)
  - Message: `feat(experiments): add difference-in-means similarity experiment`
  - Files: `src/steering_geometry/experiments.py`, `tests/test_experiments.py`, `scripts/experiments/run_diff_means_heatmaps.sh`
  - Pre-commit: `uv run pytest tests/test_experiments.py -v`

- [ ] 6. Implement Experiment 2 (discriminative)

  **What to do**:
  - Add to `src/steering_geometry/experiments.py`:
    - `run_discriminative_experiment(concept: str, k_values: list[int], layers: list[float], model_name: str, output_dir: Path) -> dict`
    - Similar structure to Experiment 1 but uses method="discriminative"
    - Varies K parameter instead of n_examples
    - Save vectors to `outputs/vectors/{concept}/discriminative/k{K}_layer{frac}.pt`
    - Save heatmaps to `outputs/heatmaps/discriminative/{concept}_layer{frac}.pdf`
    - Return summary dict
  - Handle edge cases:
    - K larger than available tokens (cap to dataset size)
    - Identical vectors
    - NaN validation

  **Must NOT do**:
  - DO NOT modify discriminative_token_aggregator implementation
  - DO NOT add cross-experiment comparisons
  - DO NOT parallelize execution

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Similar to Task 3, reusing patterns
  - **Skills**: []
    - None needed - following established pattern

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (Sequential, after Task 5)
  - **Blocks**: Tasks 7-9
  - **Blocked By**: Task 5

  **References**:
  - `src/steering_geometry/extract.py:142-177` - Discriminative aggregator
  - `src/steering_geometry/extract.py:188-190` - Method resolution with top_k parameter

  **Acceptance Criteria**:
  - [ ] Function implemented in experiments.py
  - [ ] Passes: `uv run ruff check src/steering_geometry/experiments.py`
  - [ ] Passes: `uv run mypy src/steering_geometry/experiments.py`
  - [ ] Manual test: Run for single concept produces expected outputs

  **QA Scenarios**:
  ```
  Scenario: Single concept discriminative extraction
    Tool: Bash (uv run python)
    Steps:
      1. Run: uv run python -c "from steering_geometry.experiments import run_discriminative_experiment; from pathlib import Path; run_discriminative_experiment('honesty', [16, 32], [0.5], 'Qwen/Qwen3-1.7B', Path('outputs'))"
      2. Check outputs exist
    Expected Result: 1 PDF and 2 .pt files created
    Evidence: .sisyphus/evidence/task-6-discr-extraction.txt
  ```

  **Commit**: NO
  - Commit together with Task 7 (Experiment 2 complete)

- [ ] 7. Create shell script for Experiment 2

  **What to do**:
  - Create script: `scripts/experiments/run_discriminative_heatmaps.sh`
  - Script runs experiment for all 5 concepts:
    - K values: [16, 32, 64, 128, 256]
    - Layers: [0.4, 0.5, 0.6, 0.7, 0.8]
    - Model: Qwen/Qwen3-1.7B
  - Follows same pattern as Experiment 1 script
  - Logs progress and outputs

  **Must NOT do**:
  - DO NOT duplicate code from Experiment 1 script (consider shared functions if needed)
  - DO NOT run experiments in parallel

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Bash script following established pattern
  - **Skills**: [`git-master`]
    - git-master: Bash script conventions

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (Sequential, after Task 6)
  - **Blocks**: Tasks 8-9
  - **Blocked By**: Task 6

  **References**:
  - `scripts/experiments/run_diff_means_heatmaps.sh` - Pattern to follow

  **Acceptance Criteria**:
  - [ ] Script created: `scripts/experiments/run_discriminative_heatmaps.sh`
  - [ ] Script syntax valid: `bash -n scripts/experiments/run_discriminative_heatmaps.sh`
  - [ ] Script has execute permission

  **QA Scenarios**:
  ```
  Scenario: Script syntax valid
    Tool: Bash (bash -n)
    Steps:
      1. Run: bash -n scripts/experiments/run_discriminative_heatmaps.sh
    Expected Result: No output (syntax OK)
    Evidence: .sisyphus/evidence/task-7-syntax-check.txt
  ```

  **Commit**: YES (groups with Task 6)
  - Message: `feat(experiments): add discriminative token similarity experiment`
  - Files: `src/steering_geometry/experiments.py`, `scripts/experiments/run_discriminative_heatmaps.sh`
  - Pre-commit: `uv run pytest tests/test_experiments.py -v`

- [ ] 8. Update AGENTS.md with experiment commands

  **What to do**:
  - Add section to AGENTS.md documenting experiments:
    - New experiment scripts and their locations
    - How to run experiments (command examples)
    - Output locations and file structure
    - Expected output counts (50 PDFs, 325 vectors)
  - Add to "Where to Look" table if appropriate
  - Follow existing AGENTS.md formatting and structure

  **Must NOT do**:
  - DO NOT reorganize existing AGENTS.md sections
  - DO NOT remove existing documentation
  - DO NOT add overly verbose examples

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Documentation update
  - **Skills**: []
    - None needed - straightforward documentation

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (Sequential, after Task 7)
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References**:
  - `AGENTS.md` - Existing structure and format

  **Acceptance Criteria**:
  - [ ] New "Experiments" section added to AGENTS.md
  - [ ] Commands documented for both experiments
  - [ ] Output locations documented
  - [ ] Section follows AGENTS.md formatting

  **QA Scenarios**:
  ```
  Scenario: Experiments section exists
    Tool: Bash (grep)
    Steps:
      1. Run: grep -A10 "Experiments" AGENTS.md
      2. Verify section exists with commands
    Expected Result: Section found with run_diff_means_heatmaps.sh and run_discriminative_heatmaps.sh mentioned
    Evidence: .sisyphus/evidence/task-8-agents-md.txt
  ```

  **Commit**: NO
  - Commit together with Task 9 (final commit)

- [ ] 9. Final verification and commit

  **What to do**:
  - Run all verification commands:
    - `uv run ruff check src/ tests/` (expect 0 violations)
    - `uv run ruff format --check src/ tests/` (expect already formatted)
    - `uv run mypy src/` (expect success)
    - `uv run pytest tests/test_experiments.py -v` (expect all pass)
  - Create atomic commits:
    - Commit 1: Foundation (Tasks 1-2)
    - Commit 2: Experiment 1 (Tasks 3-5)
    - Commit 3: Experiment 2 (Tasks 6-7)
    - Commit 4: Documentation (Task 8)
  - Run git log to verify commits created
  - Create final summary of deliverables

  **Must NOT do**:
  - DO NOT skip any verification steps
  - DO NOT force push or modify git config
  - DO NOT commit if any verification fails

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Running verification commands and git operations
  - **Skills**: [`git-master`]
    - git-master: For proper git workflow and commit message formatting

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (Sequential, after Task 8)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 8

  **References**:
  - `AGENTS.md` - Definition of Done checklist
  - Git commit strategy from plan

  **Acceptance Criteria**:
  - [ ] All verification commands pass
  - [ ] 4 atomic commits created
  - [ ] Commit messages follow conventional commit format
  - [ ] `git log --oneline -4` shows all commits

  **QA Scenarios**:
  ```
  Scenario: All code quality checks pass
    Tool: Bash (uv run)
    Steps:
      1. Run: uv run ruff check src/ tests/
      2. Run: uv run ruff format --check src/ tests/
      3. Run: uv run mypy src/
    Expected Result: All commands complete with 0 errors
    Evidence: .sisyphus/evidence/task-9-quality-checks.txt
  
  Scenario: All tests pass
    Tool: Bash (uv run pytest)
    Steps:
      1. Run: uv run pytest tests/test_experiments.py -v
      2. Check results
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-9-tests.txt
  
  Scenario: Commits created
    Tool: Bash (git log)
    Steps:
      1. Run: git log --oneline -4
      2. Verify 4 commits with correct messages
    Expected Result: 4 commits visible
    Evidence: .sisyphus/evidence/task-9-commits.txt
  ```

  **Commit**: YES
  - Message: `docs: add experiment commands to AGENTS.md`
  - Files: `AGENTS.md`
  - Pre-commit: None

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify all Must Have items present, all Must NOT Have items absent. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest tests/test_experiments.py -v`. Check for AI slop patterns.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

- [ ] F3. **Output Verification** — `unspecified-high`
  Verify output files: 50 PDFs in outputs/heatmaps/, 325 .pt files in outputs/vectors/. Check file naming follows convention. Test single concept extraction end-to-end.
  Output: `PDFs [N/50] | Vectors [N/325] | End-to-End [PASS/FAIL] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: verify everything in spec was built (no missing), nothing beyond spec was built (no creep). Check Must NOT do compliance.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Commit 1**: Foundation - experiments module + unit tests
  - `feat(experiments): add cosine similarity and heatmap visualization functions`
  - Files: `src/steering_geometry/experiments.py`, `tests/test_experiments.py`
  - Pre-commit: `uv run pytest tests/test_experiments.py -v -k "not gpu"`

- **Commit 2**: Experiment 1 implementation
  - `feat(experiments): add difference-in-means similarity experiment`
  - Files: `src/steering_geometry/experiments.py`, `scripts/experiments/run_diff_means_heatmaps.sh`
  - Pre-commit: `uv run pytest tests/test_experiments.py -v`

- **Commit 3**: Experiment 2 implementation
  - `feat(experiments): add discriminative token similarity experiment`
  - Files: `src/steering_geometry/experiments.py`, `scripts/experiments/run_discriminative_heatmaps.sh`
  - Pre-commit: `uv run pytest tests/test_experiments.py -v`

- **Commit 4**: Documentation
  - `docs: add experiment commands to AGENTS.md`
  - Files: `AGENTS.md`
  - Pre-commit: None

---

## Success Criteria

### Verification Commands
```bash
# Code quality
uv run ruff check src/ tests/  # Expected: 0 violations
uv run ruff format --check src/ tests/  # Expected: already formatted
uv run mypy src/  # Expected: Success with 0 errors

# Unit tests (no GPU required)
uv run pytest tests/test_experiments.py -v -k "not gpu"  # Expected: All pass

# All tests (requires GPU)
uv run pytest tests/test_experiments.py -v  # Expected: All pass

# Output verification
find outputs/heatmaps -name "*.pdf" | wc -l  # Expected: 50
find outputs/vectors -name "*.pt" | wc -l  # Expected: 325

# PDF validity check
pdfinfo outputs/heatmaps/diff_means/honesty_layer0.5.pdf | grep "Pages:"  # Expected: 1
```

### Final Checklist
- [ ] All "Must Have" present (7 items)
- [ ] All "Must NOT Have" absent (7 items)
- [ ] All tests pass
- [ ] 50 PDF heatmaps generated
- [ ] 325 steering vectors saved
- [ ] AGENTS.md updated
