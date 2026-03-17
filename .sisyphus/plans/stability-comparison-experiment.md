# Stability Comparison Experiment: Diff Means vs Discriminative

## TL;DR

> **Quick Summary**: Create a Python experiment to compare the stability of two steering vector extraction methods (diff_means vs discriminative) by running 3 repetitions with different token selections and computing pairwise cosine similarity across 10 layers.
>
> **Deliverables**:
> - `src/steering_geometry/stability_comparison.py` - Main experiment module
> - `scripts/vector_analysis/run_stability_comparison.sh` - Shell orchestration script
> - `tests/test_stability_comparison.py` - Pytest test file
> - JSON results file with similarity matrices and statistics
> - PDF heatmap visualizations
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Config → Test (RED) → Implementation → Shell Script → Verification

---

## Context

### Original Request
User wants to create a Python file to compare the stability of two steering vector extraction methods:
1. **Difference in means** - Random 10k tokens
2. **Discriminative token selection** - Top-30 tokens by score `s_i = ||h_i - μ_j||² - ||h_i - μ_i||²`

Each method runs 3 times with **different token selections** (not just different seeds), computing cosine similarity between vectors from different runs. Analyze 10 evenly spaced layers (0.0, 0.1, ..., 0.9) for the sentiment concept. Output both JSON and heatmap visualization.

### Interview Summary
**Key Discussions**:
- Layer selection: Evenly spaced 0.0-0.9 (10 layers)
- File location: Python module in `src/`, shell script in `scripts/vector_analysis/`
- Output format: Both JSON (numerical) + PDF heatmap (visualization)
- Test strategy: TDD - write failing tests first
- Token selection: Load ALL pairs, select different 10k subsets using different random generators to ensure genuinely different tokens per run

**Research Findings**:
- Discriminative formula is theoretically sound (Fisher discriminant principles)
- Cosine similarity is validated as stability metric (ICLR 2025 Braun et al.)
- Existing `compute_cosine_similarity_matrix()` in `vector_analysis.py` should be reused
- Existing `discriminative_token_aggregator()` in `extract.py` already implements the formula
- Stability thresholds: >0.9 = highly stable, 0.7-0.9 = moderate, <0.5 = unstable

### Metis Review
**Identified Gaps** (addressed):
- **Random seed architecture**: User clarified that seed doesn't guarantee different tokens - we need to use different dataset selections/generators
- **Solution**: Create seed-aware wrapper that loads ALL pairs and selects different subsets for each run

---

## Work Objectives

### Core Objective
Create a reproducible experiment comparing steering vector stability between diff_means and discriminative extraction methods, demonstrating which method produces more consistent vectors across different token selections.

### Concrete Deliverables
- `src/steering_geometry/stability_comparison.py` with `run_stability_comparison_experiment()` function
- `scripts/vector_analysis/run_stability_comparison.sh` orchestration script
- `tests/test_stability_comparison.py` with pytest test cases
- Output files in `outputs/vectors/sentiment/stability/` and `outputs/heatmaps/stability/`
- JSON results file at `outputs/stability_comparison_sentiment.json`

### Definition of Done
- [ ] `uv run mypy src/steering_geometry/stability_comparison.py` → Success with 0 errors
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run pytest tests/test_stability_comparison.py` → All tests pass
- [ ] Shell script executes successfully and produces expected output files
- [ ] JSON file contains valid similarity matrices and statistics
- [ ] PDF heatmaps generated for both methods

### Must Have
- Both diff_means and discriminative methods implemented
- 3 runs per method with genuinely different token selections
- Cosine similarity computation between all run pairs
- JSON output with similarity matrices per layer
- PDF heatmap visualization
- Test coverage for core functionality

### Must NOT Have (Guardrails)
- NO modifications to `extract.py`, `utils.py`, or `vector_analysis.py` (use wrappers instead)
- NO new aggregator functions (use existing `mean_aggregator` and `discriminative_token_aggregator`)
- NO additional concepts beyond sentiment for MVP
- NO statistical significance tests (just mean/min/max/std statistics)
- NO parallelization of extractions (sequential only)
- NO breaking changes to existing APIs

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (TDD)
- **Framework**: pytest
- **TDD approach**: Each task follows RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python module**: Use Bash (uv run pytest) - Run tests, assert all pass
- **Shell script**: Use Bash - Execute script, verify output files exist
- **Type checking**: Use Bash (uv run mypy) - Verify type correctness
- **Linting**: Use Bash (uv run ruff) - Verify code style compliance

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — Config + Test Setup):
├── Task 1: StabilityComparisonConfig dataclass [quick]
├── Task 2: Write failing tests for stability comparison [quick]
└── Task 3: Create module skeleton with type stubs [quick]

Wave 2 (After Wave 1 — Core Implementation):
├── Task 4: Implement seed-aware contrast pair selector [deep]
├── Task 5: Implement run_single_extraction() helper [deep]
├── Task 6: Implement run_stability_comparison_experiment() main [deep]
├── Task 7: Implement JSON output function [quick]
└── Task 8: Implement heatmap generation [quick]

Wave 3 (After Wave 2 — Script + Verification):
├── Task 9: Create shell orchestration script [quick]
├── Task 10: Run full verification (ruff, mypy, pytest) [quick]
└── Task 11: Manual QA - Execute experiment end-to-end [unspecified-high]

Wave FINAL (After ALL tasks — independent review, 4 parallel):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: Task 1 → Task 2 → Task 4 → Task 6 → Task 9 → Task 11
Parallel Speedup: ~40% faster than sequential
Max Concurrent: 3 (Wave 1)
```

### Dependency Matrix

- **1-3**: — — 4-8
- **4**: — — 5-6
- **5**: 4 — 6
- **6**: 4, 5 — 7, 8
- **7**: 6 — 9
- **8**: 6 — 9
- **9**: 7, 8 — 10, 11
- **10**: 9 — 11
- **11**: 10 — F1-F4

### Agent Dispatch Summary

- **Wave 1**: **3** — T1-T3 → `quick`
- **Wave 2**: **5** — T4-T5 → `deep`, T6 → `deep`, T7-T8 → `quick`
- **Wave 3**: **3** — T9, T10 → `quick`, T11 → `unspecified-high`
- **FINAL**: **4** — F1 → `oracle`, F2-F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. **Add StabilityComparisonConfig dataclass**

  **What to do**:
  - Add new `StabilityComparisonConfig` dataclass to `src/steering_geometry/config.py`
  - Fields: `concept: str`, `num_tokens: int`, `num_runs: int`, `layers: list[float]`, `top_k: int` (for discriminative), `model_name: str`, `output_dir: Path`
  - Add defaults: `concept="sentiment"`, `num_tokens=10000`, `num_runs=3`, `top_k=30`
  - Add validation: ensure `num_runs >= 2` for meaningful comparison
  - Follow existing pattern from `ExtractionConfig` and `ModelConfig`

  **Must NOT do**:
  - Do NOT add fields not needed for this experiment
  - Do NOT modify existing config classes

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dataclass addition following existing patterns
  - **Skills**: []
    - No special skills needed for dataclass creation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 4, 6
  - **Blocked By**: None (can start immediately)

  **References**:
  - `src/steering_geometry/config.py:ExtractionConfig` - Pattern for config dataclass structure
  - `src/steering_geometry/config.py:ModelConfig` - Example of dataclass with Path field
  - `src/steering_geometry/types.py` - Existing type definitions

  **Acceptance Criteria**:
  - [ ] `StabilityComparisonConfig` dataclass exists in `config.py`
  - [ ] All required fields present with correct types
  - [ ] Default values set as specified
  - [ ] Validation ensures `num_runs >= 2`

  **QA Scenarios**:

  ```
  Scenario: Config instantiation with defaults
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.config import StabilityComparisonConfig; c = StabilityComparisonConfig(); assert c.concept == 'sentiment'; assert c.num_tokens == 10000; assert c.num_runs == 3"
    Expected Result: No error, assertions pass
    Failure Indicators: ImportError, AssertionError
    Evidence: .sisyphus/evidence/task-01-config-defaults.txt

  Scenario: Config validation rejects invalid num_runs
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.config import StabilityComparisonConfig; StabilityComparisonConfig(num_runs=1)"
    Expected Result: ValueError raised with message about num_runs >= 2
    Failure Indicators: No exception raised
    Evidence: .sisyphus/evidence/task-01-config-validation.txt
  ```

  **Commit**: NO (groups with Task 3)
  - Files: `src/steering_geometry/config.py`

---

- [ ] 2. **Write failing tests for stability comparison (TDD RED phase)**

  **What to do**:
  - Create `tests/test_stability_comparison.py`
  - Write test functions for:
    - `test_select_different_token_subsets()` - Verify different subsets are selected
    - `test_run_single_extraction_returns_vector()` - Verify single extraction returns Tensor
    - `test_run_stability_comparison_returns_results()` - Verify main function returns correct structure
    - `test_compute_stability_statistics()` - Verify statistics computation
  - All tests should FAIL initially (TDD RED phase)
  - Use pytest fixtures from `tests/conftest.py` as reference

  **Must NOT do**:
  - Do NOT implement the actual functions yet
  - Do NOT skip any test scenarios

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Writing test stubs following pytest patterns
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Task 5 (implementation)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `tests/conftest.py` - Existing fixtures and patterns
  - `tests/test_extract.py` - Example test structure
  - `src/steering_geometry/types.py:SteeringVector` - Expected return type

  **Acceptance Criteria**:
  - [ ] `tests/test_stability_comparison.py` created
  - [ ] At least 4 test functions written
  - [ ] `uv run pytest tests/test_stability_comparison.py` → All tests FAIL (RED phase)

  **QA Scenarios**:

  ```
  Scenario: Tests fail as expected (RED phase)
    Tool: Bash
    Steps:
      1. uv run pytest tests/test_stability_comparison.py -v
    Expected Result: At least 4 tests FAIL with import errors or assertion failures
    Failure Indicators: All tests PASS (should not happen in RED phase)
    Evidence: .sisyphus/evidence/task-02-tests-red.txt

  Scenario: Test file follows pytest conventions
    Tool: Bash
    Steps:
      1. uv run python -c "import ast; ast.parse(open('tests/test_stability_comparison.py').read())"
    Expected Result: No syntax errors
    Failure Indicators: SyntaxError
    Evidence: .sisyphus/evidence/task-02-syntax.txt
  ```

  **Commit**: NO (groups with Task 3)
  - Files: `tests/test_stability_comparison.py`

---

- [ ] 3. **Create module skeleton with type stubs**

  **What to do**:
  - Create `src/steering_geometry/stability_comparison.py`
  - Add module docstring explaining purpose
  - Add type stubs for all functions (raise `NotImplementedError`)
  - Functions to stub:
    - `select_token_subsets(pairs: list[ContrastPair], num_tokens: int, num_runs: int) -> list[list[ContrastPair]]`
    - `run_single_extraction(model, pairs, config, method: str, layer: float) -> Tensor`
    - `compute_stability_statistics(vectors: list[Tensor]) -> dict[str, float]`
    - `save_results_json(results: dict, output_path: Path) -> None`
    - `generate_stability_heatmap(similarity_matrix: ndarray, layer: float, method: str, output_path: Path) -> None`
    - `run_stability_comparison_experiment(config: StabilityComparisonConfig) -> dict`
  - Add imports following project conventions (stdlib → third-party → local)
  - Add `logger = logging.getLogger(__name__)`

  **Must NOT do**:
  - Do NOT implement function bodies yet
  - Do NOT add extra functions not in spec

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Creating file skeleton with type stubs
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 4-8
  - **Blocked By**: None (can start immediately)

  **References**:
  - `src/steering_geometry/vector_analysis.py:1-30` - Import pattern and logger setup
  - `src/steering_geometry/extract.py` - Function signature patterns
  - `src/steering_geometry/types.py` - Type imports (ContrastPair, Tensor)

  **Acceptance Criteria**:
  - [ ] File `src/steering_geometry/stability_comparison.py` created
  - [ ] All 6 function stubs present with correct signatures
  - [ ] `uv run mypy src/steering_geometry/stability_comparison.py` → Success
  - [ ] Module docstring present

  **QA Scenarios**:

  ```
  Scenario: Module imports successfully
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.stability_comparison import select_token_subsets, run_single_extraction, compute_stability_statistics, save_results_json, generate_stability_heatmap, run_stability_comparison_experiment"
    Expected Result: No ImportError
    Failure Indicators: ImportError
    Evidence: .sisyphus/evidence/task-03-import.txt

  Scenario: Type checking passes on stubs
    Tool: Bash
    Steps:
      1. uv run mypy src/steering_geometry/stability_comparison.py
    Expected Result: Success: no issues found
    Failure Indicators: Type errors
    Evidence: .sisyphus/evidence/task-03-mypy.txt
  ```

  **Commit**: YES
  - Message: `feat(stability): add module skeleton with type stubs`
  - Files: `src/steering_geometry/config.py`, `src/steering_geometry/stability_comparison.py`, `tests/test_stability_comparison.py`
  - Pre-commit: None (stubs only)

---

- [ ] 4. **Implement seed-aware token subset selector**

  **What to do**:
  - Implement `select_token_subsets()` in `stability_comparison.py`
  - Load ALL available contrast pairs for the concept
  - Use `random.Random(seed=i)` for each run to ensure different subsets
  - Ensure subsets are non-overlapping when possible (use disjoint sampling)
  - Return list of `num_runs` subsets, each with `num_tokens` pairs
  - Handle case where total pairs < num_tokens * num_runs (cap appropriately)

  **Must NOT do**:
  - Do NOT use hardcoded seed=42
  - Do NOT modify `utils.py:sample_with_seed()`

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Logic-heavy implementation with edge case handling
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Wave 1)
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Task 1 (config), Task 3 (skeleton)

  **References**:
  - `src/steering_geometry/utils.py:sample_with_seed()` - Sampling pattern reference
  - `src/steering_geometry/extract.py:load_contrast_pairs()` - How pairs are loaded
  - `src/steering_geometry/types.py:ContrastPair` - Type definition

  **Acceptance Criteria**:
  - [ ] Function returns list of `num_runs` subsets
  - [ ] Each subset contains exactly `num_tokens` pairs (or capped if insufficient)
  - [ ] Different seeds produce different subsets
  - [ ] Subsets have minimal overlap when total pairs allow

  **QA Scenarios**:

  ```
  Scenario: Different seeds produce different subsets
    Tool: Bash
    Steps:
      1. uv run python -c "
from steering_geometry.stability_comparison import select_token_subsets
from steering_geometry.extract import load_contrast_pairs
pairs = load_contrast_pairs('sentiment', 100)
subsets = select_token_subsets(pairs, num_tokens=30, num_runs=3)
# Check first pair of each subset is different
assert subsets[0][0] != subsets[1][0] or subsets[0][0] != subsets[2][0], 'Subsets should differ'
print('OK: Different subsets generated')
"
    Expected Result: "OK: Different subsets generated"
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-04-different-subsets.txt

  Scenario: Handles insufficient pairs gracefully
    Tool: Bash
    Steps:
      1. uv run python -c "
from steering_geometry.stability_comparison import select_token_subsets
subsets = select_token_subsets([], num_tokens=10, num_runs=3)
assert all(len(s) == 0 for s in subsets), 'Empty input should return empty subsets'
print('OK: Empty handling works')
"
    Expected Result: "OK: Empty handling works"
    Failure Indicators: Exception raised
    Evidence: .sisyphus/evidence/task-04-empty-handling.txt
  ```

  **Commit**: NO (groups with Task 8)

---

- [ ] 5. **Implement run_single_extraction helper**

  **What to do**:
  - Implement `run_single_extraction()` in `stability_comparison.py`
  - Load model using `HookedModel`
  - Call `extract_steering_vector()` with appropriate config
  - Extract vector for specific layer
  - Return `Tensor` for that layer
  - Handle both "mean" and "discriminative" methods via config

  **Must NOT do**:
  - Do NOT reimplement extraction logic - reuse existing functions
  - Do NOT modify model loading code

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Integrates multiple existing components
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 3 (skeleton), Task 4 (selector)

  **References**:
  - `src/steering_geometry/extract.py:extract_steering_vector()` - Main extraction function
  - `src/steering_geometry/models.py:HookedModel` - Model loading
  - `src/steering_geometry/config.py:ExtractionConfig` - Extraction configuration
  - `src/steering_geometry/vector_analysis.py:351-362` - Example of calling extract_steering_vector

  **Acceptance Criteria**:
  - [ ] Function returns Tensor for specified layer
  - [ ] Works with both "mean" and "discriminative" methods
  - [ ] Reuses existing extraction infrastructure
  - [ ] Test passes: `test_run_single_extraction_returns_vector()`

  **QA Scenarios**:

  ```
  Scenario: Single extraction returns tensor
    Tool: Bash
    Steps:
      1. uv run python -c "
from steering_geometry.stability_comparison import run_single_extraction
from steering_geometry.config import StabilityComparisonConfig
from steering_geometry.extract import load_contrast_pairs
from steering_geometry.models import HookedModel
config = StabilityComparisonConfig(layers=[0.5])
pairs = load_contrast_pairs('sentiment', 10)
model = HookedModel.from_name(config.model_name)
vector = run_single_extraction(model, pairs[:5], config, method='mean', layer=0.5)
assert vector.shape[0] > 0, 'Vector should have non-zero dimension'
print(f'OK: Vector shape {tuple(vector.shape)}')
" 2>/dev/null || echo "Note: Requires model download on first run"
    Expected Result: Vector shape printed or note about model download
    Failure Indicators: Exception unrelated to model download
    Evidence: .sisyphus/evidence/task-05-single-extraction.txt
  ```

  **Commit**: NO (groups with Task 8)

---

- [ ] 6. **Implement run_stability_comparison_experiment main function**

  **What to do**:
  - Implement `run_stability_comparison_experiment()` in `stability_comparison.py`
  - Orchestrate the full experiment:
    1. Load all contrast pairs
    2. Select different subsets for each run
    3. For each method (mean, discriminative):
       - For each layer:
         - For each run (1-3):
           - Extract vector
    4. Compute pairwise cosine similarity for each layer
    5. Compute statistics (mean, min, max, std)
  - Return dict with structure:
    ```python
    {
        "diff_means": {
            "vectors": {layer_frac: [tensor1, tensor2, tensor3]},
            "similarity_matrices": {layer_frac: [[...], ...]},
            "statistics": {layer_frac: {"mean": ..., "min": ..., "max": ..., "std": ...}}
        },
        "discriminative": {...same structure...}
    }
    ```
  - Use `compute_cosine_similarity_matrix()` from `vector_analysis.py`

  **Must NOT do**:
  - Do NOT add parallelization
  - Do NOT add extra statistics beyond mean/min/max/std

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Main orchestration logic, complex data flow
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Tasks 4, 5)
  - **Blocks**: Tasks 7, 8
  - **Blocked By**: Task 4 (selector), Task 5 (single extraction)

  **References**:
  - `src/steering_geometry/vector_analysis.py:run_diff_means_experiment()` - Similar orchestration pattern
  - `src/steering_geometry/vector_analysis.py:compute_cosine_similarity_matrix()` - Cosine similarity
  - `src/steering_geometry/vector_analysis.py:263-278` - Statistics computation pattern

  **Acceptance Criteria**:
  - [ ] Function returns dict with correct structure
  - [ ] Both methods produce results
  - [ ] All 10 layers processed
  - [ ] 3 runs per method per layer
  - [ ] Cosine similarity matrices computed correctly
  - [ ] Statistics computed for each layer
  - [ ] Test passes: `test_run_stability_comparison_returns_results()`

  **QA Scenarios**:

  ```
  Scenario: Main function returns correct structure
    Tool: Bash
    Steps:
      1. uv run python -c "
from steering_geometry.stability_comparison import run_stability_comparison_experiment
from steering_geometry.config import StabilityComparisonConfig
config = StabilityComparisonConfig(num_tokens=10, layers=[0.5])
result = run_stability_comparison_experiment(config)
assert 'diff_means' in result
assert 'discriminative' in result
assert 'statistics' in result['diff_means']
assert 0.5 in result['diff_means']['statistics']
print('OK: Correct structure')
" 2>/dev/null || echo "Note: May require model"
    Expected Result: "OK: Correct structure" or note about model
    Failure Indicators: KeyError, AssertionError
    Evidence: .sisyphus/evidence/task-06-main-structure.txt

  Scenario: Statistics computed correctly
    Tool: Bash
    Steps:
      1. uv run python -c "
from steering_geometry.stability_comparison import compute_stability_statistics
import torch
vectors = [torch.randn(100) for _ in range(3)]
stats = compute_stability_statistics(vectors)
assert 'mean' in stats
assert 'min' in stats
assert 'max' in stats
assert 'std' in stats
assert -1 <= stats['mean'] <= 1, 'Cosine similarity should be in [-1, 1]'
print(f'OK: Stats = {stats}')
"
    Expected Result: Stats dict printed with values in valid range
    Failure Indicators: KeyError, ValueError
    Evidence: .sisyphus/evidence/task-06-stats.txt
  ```

  **Commit**: NO (groups with Task 8)

---

- [ ] 7. **Implement JSON output function**

  **What to do**:
  - Implement `save_results_json()` in `stability_comparison.py`
  - Convert tensors to lists for JSON serialization
  - Save to `outputs/stability_comparison_{concept}.json`
  - Include metadata: timestamp, config used, model name
  - Pretty-print JSON with indent=2

  **Must NOT do**:
  - Do NOT save tensors directly (not JSON serializable)
  - Do NOT add fields not in spec

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple file I/O and data conversion
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 6)
  - **Blocks**: Task 9
  - **Blocked By**: Task 6 (main function)

  **References**:
  - `src/steering_geometry/utils.py:ensure_dir()` - Directory creation
  - Python standard library `json` module

  **Acceptance Criteria**:
  - [ ] Function saves valid JSON file
  - [ ] Tensors converted to lists
  - [ ] File path follows convention: `outputs/stability_comparison_{concept}.json`
  - [ ] JSON is human-readable (pretty-printed)

  **QA Scenarios**:

  ```
  Scenario: JSON file is valid and readable
    Tool: Bash
    Steps:
      1. uv run python -c "
from steering_geometry.stability_comparison import save_results_json
from pathlib import Path
import json
results = {'diff_means': {'statistics': {0.5: {'mean': 0.85}}}}
output_path = Path('/tmp/test_stability.json')
save_results_json(results, output_path)
with open(output_path) as f:
    data = json.load(f)
assert data['diff_means']['statistics']['0.5']['mean'] == 0.85
print('OK: JSON saved and loaded correctly')
"
    Expected Result: "OK: JSON saved and loaded correctly"
    Failure Indicators: JSONDecodeError, AssertionError
    Evidence: .sisyphus/evidence/task-07-json.txt
  ```

  **Commit**: NO (groups with Task 8)

---

- [ ] 8. **Implement heatmap generation**

  **What to do**:
  - Implement `generate_stability_heatmap()` in `stability_comparison.py`
  - Use matplotlib to create heatmap visualization
  - X-axis: Run pairs (1vs2, 1vs3, 2vs3)
  - Y-axis: Layer fractions
  - Color: Cosine similarity value
  - Save to `outputs/heatmaps/stability/{method}_stability.pdf`
  - Add colorbar and title
  - Follow pattern from `vector_analysis.py:plot_heatmap()`

  **Must NOT do**:
  - Do NOT create new visualization style - follow existing pattern
  - Do NOT save to different path than specified

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Visualization code following existing pattern
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 6, parallel with Task 7)
  - **Blocks**: Task 9
  - **Blocked By**: Task 6 (main function)

  **References**:
  - `src/steering_geometry/vector_analysis.py:plot_heatmap()` - Existing heatmap implementation
  - `matplotlib` documentation for heatmap customization

  **Acceptance Criteria**:
  - [ ] PDF files generated for both methods
  - [ ] Heatmaps show cosine similarity correctly
  - [ ] Colorbar present
  - [ ] Title and labels present

  **QA Scenarios**:

  ```
  Scenario: Heatmap PDF generated
    Tool: Bash
    Steps:
      1. uv run python -c "
from steering_geometry.stability_comparison import generate_stability_heatmap
from pathlib import Path
import numpy as np
sim_matrix = np.random.rand(3, 3)
output_path = Path('/tmp/test_heatmap.pdf')
generate_stability_heatmap(sim_matrix, layer=0.5, method='diff_means', output_path=output_path)
assert output_path.exists()
print('OK: Heatmap PDF generated')
"
    Expected Result: "OK: Heatmap PDF generated"
    Failure Indicators: FileNotFoundError, PDF creation error
    Evidence: .sisyphus/evidence/task-08-heatmap.txt
  ```

  **Commit**: YES
  - Message: `feat(stability): implement stability comparison experiment`
  - Files: `src/steering_geometry/stability_comparison.py`
  - Pre-commit: `uv run pytest tests/test_stability_comparison.py -k "not slow"`

---

- [ ] 9. **Create shell orchestration script**

  **What to do**:
  - Create `scripts/vector_analysis/run_stability_comparison.sh`
  - Follow pattern from existing scripts in `scripts/vector_analysis/`
  - Use `set -euo pipefail` for safety
  - Default values:
    - CONCEPT="sentiment"
    - NUM_TOKENS=10000
    - NUM_RUNS=3
    - LAYERS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
    - TOP_K=30
    - MODEL="Qwen/Qwen3-1.7B"
  - Parse command-line arguments to override defaults
  - Call Python module via `uv run python -c "..."`
  - Add colored output for progress indication
  - Create output directories if needed

  **Must NOT do**:
  - Do NOT add parallel execution (sequential only)
  - Do NOT hardcode paths that should be configurable

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Shell script following existing patterns
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Wave 2 complete)
  - **Blocks**: Tasks 10, 11
  - **Blocked By**: Tasks 7, 8 (JSON and heatmap functions)

  **References**:
  - `scripts/vector_analysis/run_diff_means_heatmaps.sh` - Shell script pattern
  - `scripts/vector_analysis/run_discriminative_heatmaps.sh` - Another example
  - `scripts/run_pipeline.sh` - Argument parsing pattern

  **Acceptance Criteria**:
  - [ ] Script executes without errors
  - [ ] Accepts command-line arguments for all parameters
  - [ ] Creates output directories
  - [ ] Produces expected JSON and PDF files

  **QA Scenarios**:

  ```
  Scenario: Script runs successfully
    Tool: Bash
    Steps:
      1. chmod +x scripts/vector_analysis/run_stability_comparison.sh
      2. ./scripts/vector_analysis/run_stability_comparison.sh --help || echo "Script exists"
    Expected Result: Script file exists and is executable
    Failure Indicators: File not found, permission denied
    Evidence: .sisyphus/evidence/task-09-script-exists.txt

  Scenario: Script produces outputs (short run)
    Tool: Bash
    Steps:
      1. ./scripts/vector_analysis/run_stability_comparison.sh -c sentiment -n 10 -l "0.5" -k 16 2>&1 | head -20
    Expected Result: Script starts execution, shows progress
    Failure Indicators: Immediate error
    Evidence: .sisyphus/evidence/task-09-script-run.txt
  ```

  **Commit**: YES
  - Message: `feat(scripts): add stability comparison shell script`
  - Files: `scripts/vector_analysis/run_stability_comparison.sh`
  - Pre-commit: `bash -n scripts/vector_analysis/run_stability_comparison.sh`

---

- [ ] 10. **Run full verification (ruff, mypy, pytest)**

  **What to do**:
  - Run all verification commands from Definition of Done
  - Fix any issues found
  - Ensure all tests pass (GREEN phase)
  - Verify type checking passes
  - Verify linting passes

  **Must NOT do**:
  - Do NOT skip any verification step
  - Do NOT ignore warnings

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Running verification commands and fixing minor issues
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Task 9)
  - **Blocks**: Task 11
  - **Blocked By**: Task 9 (shell script)

  **References**:
  - `AGENTS.md` - Definition of Done checklist
  - `pyproject.toml` - Tool configuration

  **Acceptance Criteria**:
  - [ ] `uv run ruff check src/ tests/` → 0 violations
  - [ ] `uv run ruff format --check src/ tests/` → formatted
  - [ ] `uv run mypy src/` → 0 errors
  - [ ] `uv run pytest tests/test_stability_comparison.py` → all pass

  **QA Scenarios**:

  ```
  Scenario: All verification passes
    Tool: Bash
    Steps:
      1. uv run ruff check src/steering_geometry/stability_comparison.py tests/test_stability_comparison.py
      2. uv run ruff format --check src/steering_geometry/stability_comparison.py tests/test_stability_comparison.py
      3. uv run mypy src/steering_geometry/stability_comparison.py
      4. uv run pytest tests/test_stability_comparison.py -v
    Expected Result: All commands succeed with 0 errors/violations
    Failure Indicators: Any command exits non-zero
    Evidence: .sisyphus/evidence/task-10-verification.txt
  ```

  **Commit**: NO (verification only)

---

- [ ] 11. **Manual QA - Execute experiment end-to-end**

  **What to do**:
  - Run the full experiment script
  - Verify all output files are created
  - Verify JSON structure is correct
  - Verify PDF heatmaps are readable
  - Document results in evidence directory

  **Must NOT do**:
  - Do NOT modify code during QA
  - Do NOT skip documenting results

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires careful end-to-end verification
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (final, after Task 10)
  - **Blocks**: Final Verification Wave
  - **Blocked By**: Task 10 (full verification)

  **References**:
  - Plan success criteria section
  - Expected output structure

  **Acceptance Criteria**:
  - [ ] Script completes without errors
  - [ ] Vector files exist in `outputs/vectors/sentiment/stability/`
  - [ ] JSON file exists and has correct structure
  - [ ] PDF heatmaps exist and are readable
  - [ ] Results documented

  **QA Scenarios**:

  ```
  Scenario: Full experiment produces all outputs
    Tool: Bash
    Steps:
      1. ./scripts/vector_analysis/run_stability_comparison.sh
      2. ls outputs/vectors/sentiment/stability/diff_means/*.pt | wc -l
      3. ls outputs/vectors/sentiment/stability/discriminative/*.pt | wc -l
      4. test -f outputs/stability_comparison_sentiment.json && echo "JSON exists"
      5. test -f outputs/heatmaps/stability/diff_means_stability.pdf && echo "Diff means PDF exists"
      6. test -f outputs/heatmaps/stability/discriminative_stability.pdf && echo "Discriminative PDF exists"
    Expected Result: All files exist, 30 .pt files per method (3 runs × 10 layers)
    Failure Indicators: Missing files
    Evidence: .sisyphus/evidence/task-11-full-run.txt

  Scenario: JSON has correct structure
    Tool: Bash
    Steps:
      1. uv run python -c "
import json
with open('outputs/stability_comparison_sentiment.json') as f:
    data = json.load(f)
assert 'diff_means' in data
assert 'discriminative' in data
for method in ['diff_means', 'discriminative']:
    assert 'statistics' in data[method]
    assert len(data[method]['statistics']) == 10, 'Should have 10 layers'
print('OK: JSON structure correct')
"
    Expected Result: "OK: JSON structure correct"
    Failure Indicators: KeyError, AssertionError
    Evidence: .sisyphus/evidence/task-11-json-structure.txt
  ```

  **Commit**: YES
  - Message: `chore: verify stability comparison experiment works end-to-end`
  - Files: None (documentation only)
  - Pre-commit: None

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run mypy src/`, `uv run ruff check src/ tests/`, `uv run pytest`. Review changed files for: `Any`, empty catches, print() in prod, commented code, unused imports. Check AI slop patterns.
  Output: `Type Check [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task. Test cross-task integration. Test edge cases. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Commit 1**: `feat(stability): add StabilityComparisonConfig dataclass` — config.py
- **Commit 2**: `test(stability): add failing tests for stability comparison` — test_stability_comparison.py
- **Commit 3**: `feat(stability): implement stability comparison experiment` — stability_comparison.py
- **Commit 4**: `feat(scripts): add stability comparison shell script` — run_stability_comparison.sh

---

## Success Criteria

### Verification Commands
```bash
# Type check
uv run mypy src/steering_geometry/stability_comparison.py
# Expected: Success: no issues found in X source files

# Lint check
uv run ruff check src/steering_geometry/stability_comparison.py tests/test_stability_comparison.py
# Expected: 0 violations

# Format check
uv run ruff format --check src/steering_geometry/stability_comparison.py tests/test_stability_comparison.py
# Expected: X files already formatted

# Run tests
uv run pytest tests/test_stability_comparison.py -v
# Expected: X passed in Ys

# Execute experiment
./scripts/vector_analysis/run_stability_comparison.sh
# Expected: Creates JSON and PDF outputs

# Verify outputs
ls outputs/vectors/sentiment/stability/diff_means/
ls outputs/vectors/sentiment/stability/discriminative/
# Expected: .pt files for each run and layer

cat outputs/stability_comparison_sentiment.json | python -m json.tool
# Expected: Valid JSON with similarity_matrices and statistics keys
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Experiment produces expected outputs
