# Discriminative Token Experiments

## TL;DR

> **Quick Summary**: Create two experimental scripts for discriminative token analysis: (1) visualize top-50 pos/neg tokens with decoded literals, (2) train linear probes to test separability across 10 layers.
> 
> **Deliverables**:
> - `src/steering_geometry/token_analysis.py` - unified module with two subcommands
> - New types in `types.py`: `TokenRecord`, `DiscriminativeTokenResult`, `ProbeLayerResult`, `ProbeExperimentResult`
> - New config in `config.py`: `TokenAnalysisConfig`
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential layer processing for memory safety
> **Critical Path**: Types → Config → Token Extraction → Visualization → Probe

---

## Context

### Original Request
Create a new branch for experiments with two Python files:
1. Use discriminative method to find pos/neg top-50 tokens, display decoded literals for each concept
2. Train linear probes on tokens closest to pos/neg centers, test separability across 10 evenly-spaced layers

### Interview Summary
**Key Discussions**:
- Model: Qwen/Qwen3-1.7B
- Data unit: TOKENS not examples - collect 10k tokens per class
- Token collection: Flatten ALL tokens from sequences
- Token display: Detokenized (merged subwords)
- Probe implementation: PyTorch nn.Linear
- Train/Test split: 80/20 stratified

**Research Findings**:
- Discriminative scoring: `s_i = ||h_i - μ_j||² - ||h_i - μ_i||²` (from `extract.py:142-177`)
- Tokenizer available via `model.tokenizer` (`models.py:50-56`)
- Layer resolution: `model.resolve_layers()` for evenly-spaced layers
- Need to create per-token extraction (current code aggregates)

### Metis Review
**Identified Gaps** (addressed):
- Token-to-text mapping: Create `TokenRecord` dataclass
- Memory management: Process layers sequentially, clear GPU cache
- Probe metrics: Track train/test accuracy, AUC, confusion matrix

---

## Work Objectives

### Core Objective
Build infrastructure for discriminative token analysis with visualization and probe experiments.

### Concrete Deliverables
- `src/steering_geometry/token_analysis.py` - CLI module with `visualize` and `probe` subcommands
- Types added to `src/steering_geometry/types.py`
- Config added to `src/steering_geometry/config.py`
- Git branch: `experiment/token-analysis`

### Definition of Done
- [ ] `uv run python -m steering_geometry.token_analysis visualize --concept honesty` produces JSON + console output
- [ ] `uv run python -m steering_geometry.token_analysis probe --concept honesty` produces probe metrics JSON
- [ ] All 5 concepts work: honesty, sentiment, toxicity, sycophancy, refusal
- [ ] `uv run ruff check src/` passes with 0 violations
- [ ] `uv run mypy src/` passes with 0 errors

### Must Have
- Per-token discriminative scoring with token ID tracking
- Detokenized text output for visualization
- Linear probe with 80/20 train/test split per layer
- 10 evenly-spaced layers from 0.0 to 1.0

### Must NOT Have (Guardrails)
- NO dimensionality reduction (t-SNE, clustering, PCA on token space)
- NO attention analysis
- NO cross-concept comparison in single run
- NO modification to existing `discriminative_token_aggregator` function
- NO over-abstraction (keep as functions, not class hierarchies)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (unit tests for new types and functions)
- **Framework**: pytest
- **Test strategy**: Tests-after (implement first, then add tests)

### QA Policy
Every task includes agent-executed QA scenarios with evidence capture.

- **CLI/Python**: Use Bash — Run commands, verify exit codes, check output files
- **Output verification**: JSON structure validation, metric range checks

---

## Execution Strategy

### Sequential Execution (Memory Safety)

```
Task 1: Create git branch
    ↓
Task 2: Add types to types.py
    ↓
Task 3: Add config to config.py
    ↓
Task 4: Create token_analysis.py skeleton
    ↓
Task 5: Implement token extraction function
    ↓
Task 6: Implement discriminative scoring
    ↓
Task 7: Implement visualize subcommand
    ↓
Task 8: Implement probe subcommand
    ↓
Task 9: Add unit tests
    ↓
Task 10: Final verification
```

**Why Sequential**: 
- Per-token extraction creates large tensors (10k × hidden_dim)
- Processing 10 layers sequentially prevents GPU OOM
- Each layer cleared before processing next

### Dependency Matrix

- **2, 3**: — — 4, 1
- **4**: 2, 3 — 5, 1
- **5**: 4 — 6, 1
- **6**: 5 — 7, 8, 1
- **7**: 6 — 9, 1
- **8**: 6 — 9, 1
- **9**: 7, 8 — 10, 1
- **10**: 9 — 1

---

## TODOs

- [x] 1. Create Git Branch

  **What to do**:
  - Create new branch `experiment/token-analysis` from current branch
  - Verify branch is created correctly

  **Must NOT do**:
  - Push to remote without user request

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single git command, trivial task
  - **Skills**: [`git-master`]
    - `git-master`: Git branch creation

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 2-10
  - **Blocked By**: None

  **References**:
  - Current branch: experiment/test-steering (from git status)

  **Acceptance Criteria**:
  - [ ] Branch `experiment/token-analysis` exists locally
  - [ ] `git branch --show-current` returns `experiment/token-analysis`

  **QA Scenarios**:
  ```
  Scenario: Branch creation
    Tool: Bash
    Steps:
      1. git checkout -b experiment/token-analysis
      2. git branch --show-current
    Expected Result: "experiment/token-analysis"
    Evidence: .sisyphus/evidence/task-01-branch.txt
  ```

  **Commit**: NO (branch only)

- [x] 2. Add Types to types.py

  **What to do**:
  - Add `TokenRecord` dataclass for per-token data
  - Add `DiscriminativeTokenResult` dataclass for visualization output
  - Add `ProbeLayerResult` dataclass for per-layer probe metrics
  - Add `ProbeExperimentResult` dataclass for complete probe results
  - Update `__all__` list

  **Must NOT do**:
  - Modify existing types
  - Add unrelated types

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding dataclasses, straightforward
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundation for other tasks)
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 3, 4
  - **Blocked By**: Task 1

  **References**:
  - Pattern: `types.py:77-91` (ContrastPair dataclass example)
  - Pattern: `types.py:169-213` (TDNVResult dataclass example)

  **Acceptance Criteria**:
  - [ ] `TokenRecord` has: token_id, token_text, activation, contrast_pair_idx, position_in_sequence, label
  - [ ] `DiscriminativeTokenResult` has: concept, layer, top_positive (50), top_negative (50)
  - [ ] `ProbeLayerResult` has: layer_idx, train_accuracy, test_accuracy, auc_score
  - [ ] `ProbeExperimentResult` has: concept, model_name, tokens_per_class, layer_results
  - [ ] `uv run mypy src/steering_geometry/types.py` passes

  **QA Scenarios**:
  ```
  Scenario: Types are importable
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.types import TokenRecord, DiscriminativeTokenResult, ProbeLayerResult, ProbeExperimentResult; print('OK')"
    Expected Result: "OK"
    Evidence: .sisyphus/evidence/task-02-types.txt
  ```

  **Commit**: NO (groups with task 3)

- [x] 3. Add Config to config.py

  **What to do**:
  - Add `TokenAnalysisConfig` dataclass with parameters:
    - top_k: int = 50
    - tokens_per_class: int = 10000
    - test_size: float = 0.2
    - layers: list[float] = [i/9 for i in range(10)]
    - batch_size: int = 8
    - random_seed: int = 42
  - Update `__all__` list

  **Must NOT do**:
  - Modify existing configs

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding dataclass, straightforward
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References**:
  - Pattern: `config.py` (existing config dataclasses)
  - Pattern: `TDNVConfig` as reference

  **Acceptance Criteria**:
  - [ ] `TokenAnalysisConfig` exists with all fields
  - [ ] Default layers = [0.0, 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1.0]
  - [ ] `uv run mypy src/steering_geometry/config.py` passes

  **QA Scenarios**:
  ```
  Scenario: Config is importable with defaults
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.config import TokenAnalysisConfig; c = TokenAnalysisConfig(); print(len(c.layers))"
    Expected Result: "10"
    Evidence: .sisyphus/evidence/task-03-config.txt
  ```

  **Commit**: YES
  - Message: `feat(token-analysis): add types and config for token analysis`
  - Files: src/steering_geometry/types.py, src/steering_geometry/config.py
  - Pre-commit: `uv run mypy src/`

- [x] 4. Create token_analysis.py Skeleton

  **What to do**:
  - Create `src/steering_geometry/token_analysis.py`
  - Add module docstring
  - Add imports (argparse, torch, etc.)
  - Add `_Args` Protocol for CLI
  - Add `_build_parser()` with visualize/probe subcommands
  - Add `main()` function skeleton
  - Add `if __name__ == "__main__"` block
  - Add `__all__` list

  **Must NOT do**:
  - Implement actual logic (just skeleton)
  - Add business logic functions

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Boilerplate setup
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 5-8
  - **Blocked By**: Tasks 2, 3

  **References**:
  - Pattern: `extract.py:571-683` (CLI pattern with _Args, _build_parser, main)
  - Pattern: `tdnv.py:324-400` (CLI pattern)
  - Subcommand pattern: `argparse.SubParser`

  **Acceptance Criteria**:
  - [ ] File exists at `src/steering_geometry/token_analysis.py`
  - [ ] `uv run python -m steering_geometry.token_analysis --help` works
  - [ ] `uv run python -m steering_geometry.token_analysis visualize --help` works
  - [ ] `uv run python -m steering_geometry.token_analysis probe --help` works
  - [ ] `uv run mypy src/steering_geometry/token_analysis.py` passes

  **QA Scenarios**:
  ```
  Scenario: CLI help works
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.token_analysis --help
    Expected Result: Exit code 0, shows help text with visualize and probe subcommands
    Evidence: .sisyphus/evidence/task-04-cli-help.txt
  ```

  **Commit**: NO (groups with task 7)

- [x] 5. Implement Token Extraction Function

  **What to do**:
  - Add `extract_all_token_activations()` function
  - Input: model, texts, layers, tokenizer
  - Output: dict[layer_idx, list[TokenRecord]]
  - For each text: tokenize, get activations, track ALL token positions
  - Flatten tokens until reaching tokens_per_class limit
  - Handle memory: process in batches, move to CPU immediately
  - Add detokenization helper: merge subwords to readable text

  **Must NOT do**:
  - Modify existing functions
  - Use only last token (must extract ALL tokens)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core logic, requires careful tensor handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:
  - Model activation: `models.py:101-150` (get_activations method)
  - Tokenizer: `models.py:50-56` (tokenizer initialization)
  - Token decoding: `tokenizer.convert_ids_to_tokens()`, `tokenizer.convert_tokens_to_string()`
  - Batch processing: `extract.py:486-508`

  **Acceptance Criteria**:
  - [ ] Function returns dict[int, list[TokenRecord]]
  - [ ] Each TokenRecord has correct token_id, token_text, activation, position
  - [ ] Tokens are flattened from all sequences
  - [ ] Detokenization merges subwords correctly (e.g., "Ġhonest" + "ness" → "honestness")
  - [ ] Memory managed with CPU offload

  **QA Scenarios**:
  ```
  Scenario: Token extraction returns correct count
    Tool: Bash
    Steps:
      1. uv run python -c "
from steering_geometry.token_analysis import extract_all_token_activations
from steering_geometry.models import HookedModel
from steering_geometry.config import ModelConfig, TokenAnalysisConfig

model = HookedModel(ModelConfig(model_name='Qwen/Qwen3-1.7B'))
texts = ['Hello world', 'Test sentence']
result = extract_all_token_activations(model, texts, [0], config=TokenAnalysisConfig(tokens_per_class=100))
print(f'Tokens collected: {len(result[0])}')
"
    Expected Result: Prints token count > 0
    Evidence: .sisyphus/evidence/task-05-extraction.txt
  ```

  **Commit**: NO (groups with task 7)

- [x] 6. Implement Discriminative Scoring Function

  **What to do**:
  - Add `compute_discriminative_scores()` function
  - Input: list[TokenRecord] for pos, list[TokenRecord] for neg
  - Compute class centers: μ_pos, μ_neg
  - Score each token: s_i = ||h_i - μ_other||² - ||h_i - μ_own||²
  - Return tokens sorted by score (descending for pos, ascending for neg)
  - Add `select_top_k_tokens()` to get top-50

  **Must NOT do**:
  - Modify existing `discriminative_token_aggregator`
  - Change scoring formula

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Mathematical computation, tensor operations
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 7, 8
  - **Blocked By**: Task 5

  **References**:
  - Scoring formula: `extract.py:162-163`
    ```python
    pos_scores = ((pos - neg_center) ** 2).sum(dim=1) - ((pos - pos_center) ** 2).sum(dim=1)
    ```
  - Top-K selection: `extract.py:168-172`

  **Acceptance Criteria**:
  - [ ] Scores computed correctly per formula
  - [ ] Positive tokens sorted descending (highest score = most discriminative)
  - [ ] Negative tokens sorted descending (highest score = most discriminative)
  - [ ] Top-K selection returns exactly K tokens

  **QA Scenarios**:
  ```
  Scenario: Scoring produces valid results
    Tool: Bash
    Steps:
      1. Create mock TokenRecords with known activations
      2. Call compute_discriminative_scores()
      3. Verify scores are finite and sorted
    Expected Result: Scores are finite floats, sorted correctly
    Evidence: .sisyphus/evidence/task-06-scoring.txt
  ```

  **Commit**: NO (groups with task 7)

- [x] 7. Implement Visualize Subcommand

  **What to do**:
  - Add `run_visualize()` function
  - Load contrast pairs for concept
  - Extract tokens using `extract_all_token_activations()`
  - Compute discriminative scores
  - Select top-50 pos and top-50 neg
  - Format output: print to console with detokenized text
  - Save to JSON: concept, layer, top_positive (with token_text, score), top_negative
  - Process each layer sequentially with memory cleanup

  **Must NOT do**:
  - Add t-SNE or other dimensionality reduction
  - Add attention analysis

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration of multiple components
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 9
  - **Blocked By**: Task 6

  **References**:
  - Contrast pair loading: `extract.py:426-450`
  - JSON output pattern: `tdnv.py:212-241`
  - Layer loop: `vector_analysis.py:161-291`

  **Acceptance Criteria**:
  - [ ] CLI: `uv run python -m steering_geometry.token_analysis visualize --concept honesty --output outputs/token_viz/` works
  - [ ] Console output shows 50 pos tokens and 50 neg tokens per layer
  - [ ] JSON file created at `outputs/token_viz/{concept}_{model}.json`
  - [ ] JSON has structure: {concept, model, layers: [{layer, top_positive, top_negative}]}
  - [ ] Token texts are detokenized (merged subwords)
  - [ ] All 5 concepts work

  **QA Scenarios**:
  ```
  Scenario: Visualize produces valid output
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.token_analysis visualize --concept honesty --model Qwen/Qwen3-1.7B --output outputs/token_viz/ --top-k 50
      2. ls outputs/token_viz/
      3. cat outputs/token_viz/honesty_Qwen_Qwen3-1.7B.json | head -50
    Expected Result: JSON file exists, contains top_positive and top_negative arrays with 50 items each
    Evidence: .sisyphus/evidence/task-07-visualize.txt

  Scenario: All concepts work
    Tool: Bash
    Steps:
      1. for concept in honesty sentiment toxicity sycophancy refusal; do
           uv run python -m steering_geometry.token_analysis visualize --concept $concept --model Qwen/Qwen3-1.7B --output outputs/token_viz/ --top-k 10
         done
      2. ls outputs/token_viz/*.json | wc -l
    Expected Result: 5 JSON files created
    Evidence: .sisyphus/evidence/task-07-all-concepts.txt
  ```

  **Commit**: YES
  - Message: `feat(token-analysis): implement visualize subcommand`
  - Files: src/steering_geometry/token_analysis.py
  - Pre-commit: `uv run ruff check && uv run mypy`

- [x] 8. Implement Probe Subcommand

  **What to do**:
  - Add `train_linear_probe()` function using PyTorch
    - nn.Linear(hidden_dim, 2) for binary classification
    - CrossEntropyLoss, Adam optimizer
    - Train for N epochs (e.g., 100)
  - Add `evaluate_probe()` function
    - Compute accuracy, AUC score
    - Return ProbeLayerResult
  - Add `run_probe()` main function
    - Load contrast pairs
    - Extract 10k tokens per class
    - Split 80/20 train/test
    - For each layer: train probe, evaluate
    - Save results to JSON
  - Handle memory: process layers sequentially

  **Must NOT do**:
  - Use sklearn (use PyTorch as requested)
  - Add cross-concept comparison

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: ML training loop, careful implementation needed
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 9
  - **Blocked By**: Task 6

  **References**:
  - PyTorch training loop pattern (standard)
  - Output pattern: `tdnv.py:212-241` (JSON save)
  - Config pattern: `TokenAnalysisConfig`

  **Acceptance Criteria**:
  - [ ] CLI: `uv run python -m steering_geometry.token_analysis probe --concept honesty --output outputs/probes/` works
  - [ ] Uses PyTorch nn.Linear for probe
  - [ ] 80/20 train/test split implemented
  - [ ] 10 layer results in output
  - [ ] Each layer has: layer_idx, train_accuracy, test_accuracy, auc_score
  - [ ] JSON file created at `outputs/probes/{concept}_{model}_probe.json`
  - [ ] All 5 concepts work

  **QA Scenarios**:
  ```
  Scenario: Probe produces valid metrics
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.token_analysis probe --concept honesty --model Qwen/Qwen3-1.7B --output outputs/probes/ --tokens-per-class 1000
      2. cat outputs/probes/honesty_Qwen_Qwen3-1.7B_probe.json | python -c "import json,sys; d=json.load(sys.stdin); print(f'Layers: {len(d[\"layer_results\"])}'); print(f'Accuracies: {[r[\"test_accuracy\"] for r in d[\"layer_results\"]]}')"
    Expected Result: 10 layers, all accuracies between 0 and 1
    Evidence: .sisyphus/evidence/task-08-probe.txt

  Scenario: Probe uses PyTorch
    Tool: Bash
    Steps:
      1. grep -n "nn.Linear\|CrossEntropyLoss" src/steering_geometry/token_analysis.py
    Expected Result: Found PyTorch imports
    Evidence: .sisyphus/evidence/task-08-pytorch.txt
  ```

  **Commit**: YES
  - Message: `feat(token-analysis): implement probe subcommand`
  - Files: src/steering_geometry/token_analysis.py
  - Pre-commit: `uv run ruff check && uv run mypy`

- [x] 9. Add Unit Tests

  **What to do**:
  - Create `tests/test_token_analysis.py`
  - Test types instantiation
  - Test config defaults
  - Test token extraction (mock model)
  - Test discriminative scoring (known inputs)
  - Test output file format validation

  **Must NOT do**:
  - Add integration tests that require model loading (too slow)
  - Test existing code

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard unit tests
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 7, 8

  **References**:
  - Test pattern: `tests/test_hello.py`
  - Fixture pattern: `tests/conftest.py`

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/test_token_analysis.py` passes
  - [ ] Tests cover: types, config, scoring logic
  - [ ] All tests use mocks (no real model loading)

  **QA Scenarios**:
  ```
  Scenario: Tests pass
    Tool: Bash
    Steps:
      1. uv run pytest tests/test_token_analysis.py -v
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-09-tests.txt
  ```

  **Commit**: YES
  - Message: `test(token-analysis): add unit tests`
  - Files: tests/test_token_analysis.py
  - Pre-commit: `uv run pytest`

- [x] 10. Final Verification

  **What to do**:
  - Run full quality check: ruff, mypy, pytest
  - Test both subcommands with all 5 concepts
  - Verify output JSON structure
  - Clean up any debug code

  **Must NOT do**:
  - Add new features

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verification only
  - **Skills**: [`git-master`]
    - `git-master`: For final commit verification

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None (final task)
  - **Blocked By**: Task 9

  **References**:
  - Quality commands in AGENTS.md

  **Acceptance Criteria**:
  - [ ] `uv run ruff check src/ tests/` → 0 violations
  - [ ] `uv run ruff format --check src/ tests/` → formatted
  - [ ] `uv run mypy src/` → 0 errors
  - [ ] `uv run pytest` → all pass
  - [ ] All 5 concepts work for both subcommands

  **QA Scenarios**:
  ```
  Scenario: Full quality check
    Tool: Bash
    Steps:
      1. uv run ruff check src/ tests/
      2. uv run ruff format --check src/ tests/
      3. uv run mypy src/
      4. uv run pytest
    Expected Result: All checks pass
    Evidence: .sisyphus/evidence/task-10-quality.txt
  ```

  **Commit**: NO (verification only)

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
  Verify all types added, both subcommands work, all 5 concepts tested, output files generated.

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `ruff check`, `ruff format --check`, `mypy`, `pytest`. All must pass.

- [x] F3. **Real Manual QA** — `unspecified-high`
  Run both subcommands for honesty concept, verify JSON output structure and console output.

- [x] F4. **Scope Fidelity Check** — `deep`
  Verify no dimensionality reduction added, no attention analysis, no cross-concept logic.

---

## Commit Strategy

- **1**: `feat(token-analysis): add TokenRecord and probe types` — types.py, config.py
- **2**: `feat(token-analysis): implement visualize subcommand` — token_analysis.py (partial)
- **3**: `feat(token-analysis): implement probe subcommand` — token_analysis.py (complete)
- **4**: `test(token-analysis): add unit tests` — tests/test_token_analysis.py

---

## Success Criteria

### Verification Commands
```bash
# Create branch
git checkout -b experiment/token-analysis

# Verify visualize works
uv run python -m steering_geometry.token_analysis visualize \
  --concept honesty --model Qwen/Qwen3-1.7B --output outputs/token_viz/

# Verify probe works
uv run python -m steering_geometry.token_analysis probe \
  --concept honesty --model Qwen/Qwen3-1.7B --output outputs/probes/

# Quality checks
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All 5 concepts work for both subcommands
- [ ] JSON outputs have correct structure
- [ ] No memory errors during execution
