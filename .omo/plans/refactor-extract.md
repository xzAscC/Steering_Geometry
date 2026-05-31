# Refactor: Consolidate Vector Extraction

## TL;DR

> **Quick Summary**: Consolidate 5 vector extraction modules into a single `extract.py` file, replace synthetic data with HuggingFace datasets, and remove steering/evaluation functionality for a minimal, focused extraction library.
>
> **Deliverables**:
> - Single `extract.py` with unified CLI for all 5 concepts
> - Simplified `models.py` (no steering), `types.py` (no EvaluationResult), `config.py` (no EvaluationConfig)
> - Updated tests and scripts
> - Deleted obsolete files
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Dataset Verification → extract.py → Dataset Loaders → Tests → Verification

---

## Context

### Original Request
重构这个文件：
1. 把所有的vector抽取都集中到一个文件里面，而不是用很多个文件
2. 做vector抽取的时候，利用huggingface里面相关的数据集，而不是用生成的数据
3. 获得vector就可以，不要做steering也不要做evaluation
4. 用尽可能少的文件，构造尽可能少的目录

### Interview Summary
**Key Discussions**:
- Dataset choices: google/civil_comments (toxicity), Anthropic/model-written-evals (sycophancy), Human-CentricAI/chatbot-arena-llm-refusal (refusal)
- CLI design: Unified entry point with `--concept` argument
- Test strategy: Add pytest tests for new functionality

**Research Findings**:
- honesty: truthfulqa/truthful_qa (existing, works)
- sentiment: glue/sst2 (existing, works)
- toxicity: google/civil_comments has toxicity field (0-1 float)
- sycophancy: Anthropic/model-written-evals has sycophancy/ subdirectory
- refusal: Human-CentricAI/chatbot-arena-llm-refusal has REFUSAL_ETHICAL/NORMAL labels

### Metis Review
**Identified Gaps** (addressed):
- Need to verify dataset field structures before implementation
- CLI must include all existing options: --model, --method, --num-pairs, --output, --dry-run
- Need to update scripts/run_extractions.sh for new CLI
- Remove EvaluationResult from types.py, EvaluationConfig from config.py
- Remove generate() method with steering from models.py

---

## Work Objectives

### Core Objective
Create a minimal, focused vector extraction library with:
- Single entry point for all 5 concepts
- Real HuggingFace datasets (no synthetic data)
- No steering or evaluation functionality

### Concrete Deliverables
- `src/steering_geometry/extract.py` - Unified extraction module with CLI
- `src/steering_geometry/models.py` - Simplified (remove steering)
- `src/steering_geometry/types.py` - Simplified (remove EvaluationResult)
- `src/steering_geometry/config.py` - Simplified (remove EvaluationConfig)
- `tests/unit/test_extract.py` - New unified tests
- Updated `scripts/run_extractions.sh`

### Definition of Done
- [ ] `uv run python -m steering_geometry.extract --concept honesty --model sshleifer/tiny-gpt2 --dry-run` works
- [ ] `uv run python -m steering_geometry.extract --concept toxicity --model sshleifer/tiny-gpt2 --dry-run` works
- [ ] `uv run python -m steering_geometry.extract --concept sycophancy --model sshleifer/tiny-gpt2 --dry-run` works
- [ ] `uv run python -m steering_geometry.extract --concept refusal --model sshleifer/tiny-gpt2 --dry-run` works
- [ ] `uv run python -m steering_geometry.extract --concept sentiment --model sshleifer/tiny-gpt2 --dry-run` works
- [ ] All quality checks pass: `uv run ruff check`, `uv run mypy`, `uv run pytest`

### Must Have
- 5 concepts: honesty, sentiment, toxicity, sycophancy, refusal
- All use real HuggingFace datasets
- Unified CLI with `--concept` argument
- Output format: `.pt` file with `vector` and `num_pairs` keys

### Must NOT Have (Guardrails)
- NO steering functionality
- NO evaluation functionality
- NO new concepts beyond the 5 specified
- NO new abstraction layers for datasets
- NO dataset caching or progress bars
- NO custom dataset support via CLI

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (TDD)
- **Framework**: pytest

### QA Policy
Every task includes agent-executed QA scenarios with:
- Exact commands to run
- Expected outputs
- Evidence capture to `.omo/evidence/`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - sequential prerequisite):
└── Task 1: Verify HuggingFace dataset loading [quick]

Wave 2 (Core implementation - parallel after Wave 1):
├── Task 2: Create extract.py with unified CLI [deep]
├── Task 3: Simplify models.py [quick]
└── Task 4: Simplify types.py and config.py [quick]

Wave 3 (Dataset loaders and cleanup - parallel after Wave 2):
├── Task 5: Implement all 5 dataset loaders [deep]
└── Task 6: Delete obsolete files [quick]

Wave 4 (Tests and verification - sequential):
├── Task 7: Update tests [deep]
├── Task 8: Update scripts/run_extractions.sh [quick]
└── Task 9: Final verification [quick]
```

### Dependency Matrix
- **1**: — 2, 3, 4, 5
- **2**: 1 — 5, 7
- **3**: — 5, 7
- **4**: — 5, 7
- **5**: 1, 2, 3, 4 — 7
- **6**: 2, 3, 4, 5 — 7
- **7**: 2, 3, 4, 5, 6 — 8
- **8**: 7 — 9
- **9**: 8 —

---

## TODOs

- [ ] 1. **Verify HuggingFace Dataset Loading**

  **What to do**:
  - Test-load all 5 HuggingFace datasets to verify accessibility and document exact field structures
  - For each dataset, print: split names, field names, sample counts, sample rows
  - Document findings in a comment in extract.py for reference

  **Datasets to verify**:
  - `truthfulqa/truthful_qa` (generation split, question field)
  - `glue/sst2` (train split, sentence/label fields)
  - `google/civil_comments` (find toxicity field, text field)
  - `Anthropic/model-written-evals` (sycophancy/*.jsonl files)
  - `Human-CentricAI/chatbot-arena-llm-refusal` (refusal_label_a/b fields)

  **Must NOT do**:
  - DO NOT modify any existing code
  - DO NOT create new files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple verification task, just running test loads
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (standalone)
  - **Blocks**: Tasks 2, 5
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/concepts/honesty.py:66-70` - Example of loading truthfulqa
  - `src/steering_geometry/concepts/sentiment.py:77-78` - Example of loading glue/sst2

  **Acceptance Criteria**:
  - [ ] All 5 datasets load successfully without authentication errors
  - [ ] Field structures documented in terminal output

  **QA Scenarios**:
  ```
  Scenario: Load all datasets successfully
    Tool: Bash
    Steps:
      1. uv run python -c "from datasets import load_dataset; d = load_dataset('truthfulqa/truthful_qa', 'generation'); print('truthfulqa:', list(d.keys()))"
      2. uv run python -c "from datasets import load_dataset; d = load_dataset('glue', 'sst2'); print('sst2:', list(d.keys()))"
      3. uv run python -c "from datasets import load_dataset; d = load_dataset('google/civil_comments'); print('civil_comments:', list(d.keys()))"
    Expected Result: All commands print split names without errors
    Evidence: .omo/evidence/task-01-dataset-verify.txt
  ```

  **Commit**: NO (verification only)

- [ ] 2. **Create extract.py with Unified CLI**

  **What to do**:
  - Create new `src/steering_geometry/extract.py`
  - Implement argparse CLI with options: `--concept`, `--model`, `--method`, `--num-pairs`, `--output`, `--dry-run`
  - Add main() function with concept validation (only accept: honesty, sentiment, toxicity, sycophancy, refusal)
  - Add placeholder dataset loader functions (will be implemented in Task 5)
  - Copy core extraction logic from existing `extraction.py`

  **CLI contract**:
  ```bash
  python -m steering_geometry.extract --concept honesty --model sshleifer/tiny-gpt2 --method mean --num-pairs 500 --output data/vectors/ --dry-run
  ```

  **Must NOT do**:
  - DO NOT implement dataset loaders yet (Task 5)
  - DO NOT import from concepts/ directory
  - DO NOT add evaluation logic

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core module creation with CLI design
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (but can run with Tasks 3, 4)
  - **Parallel Group**: Wave 2 (with Tasks 3, 4)
  - **Blocks**: Tasks 5, 7
  - **Blocked By**: Task 1

  **References**:
  - `src/steering_geometry/extraction.py:59-113` - Core extraction logic to copy
  - `src/steering_geometry/extract_honesty.py:21-32` - CLI pattern to follow
  - `src/steering_geometry/config.py:24-39` - ExtractionConfig to use

  **Acceptance Criteria**:
  - [ ] File created: src/steering_geometry/extract.py
  - [ ] `uv run python -m steering_geometry.extract --help` shows all options
  - [ ] `uv run python -m steering_geometry.extract --concept invalid` exits with error

  **QA Scenarios**:
  ```
  Scenario: CLI help works
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.extract --help
    Expected Result: Shows --concept, --model, --method, --num-pairs, --output, --dry-run options
    Evidence: .omo/evidence/task-02-cli-help.txt

  Scenario: Invalid concept rejected
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.extract --concept invalid --model sshleifer/tiny-gpt2
    Expected Result: Exit code != 0, error message mentions valid concepts
    Evidence: .omo/evidence/task-02-invalid-concept.txt
  ```

  **Commit**: NO (wait for complete implementation)

- [ ] 4. **Simplify types.py and config.py**

  **What to do**:
  - Remove `EvaluationResult` class from types.py (lines 63-77)
  - Remove `EvaluationConfig` class from config.py (lines 41-51)
  - Update `__all__` exports in both files
  - Remove any unused imports

  **Must NOT do**:
  - DO NOT modify ContrastPair or SteeringVector types
  - DO NOT modify ModelConfig or ExtractionConfig
  - DO NOT add new types

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple deletion of unused types
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3)
  - **Blocks**: Task 7
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/types.py:63-77` - EvaluationResult to remove
  - `src/steering_geometry/config.py:41-51` - EvaluationConfig to remove

  **Acceptance Criteria**:
  - [ ] EvaluationResult removed from types.py
  - [ ] EvaluationConfig removed from config.py
  - [ ] `uv run mypy src/steering_geometry/types.py src/steering_geometry/config.py` passes

  **QA Scenarios**:
  ```
  Scenario: Type check passes
    Tool: Bash
    Steps:
      1. uv run mypy src/steering_geometry/types.py src/steering_geometry/config.py
    Expected Result: Success with 0 errors
    Evidence: .omo/evidence/task-04-mypy.txt
  ```

  **Commit**: NO (wait for complete implementation)

- [ ] 5. **Implement All 5 Dataset Loaders**

  **What to do**:
  - Implement `load_honesty_data()` using `truthfulqa/truthful_qa` (copy from concepts/honesty.py)
  - Implement `load_sentiment_data()` using `glue/sst2` (copy from concepts/sentiment.py)
  - Implement `load_toxicity_data()` using `google/civil_comments`:
    - Use toxicity field (threshold > 0.5 for toxic, < 0.2 for non-toxic)
    - Create contrast pairs with text field
  - Implement `load_sycophancy_data()` using `Anthropic/model-written-evals`:
    - Load from sycophancy/sycophancy_on_nlp_survey.jsonl
    - Parse bio + question format, create sycophantic vs objective pairs
  - Implement `load_refusal_data()` using `Human-CentricAI/chatbot-arena-llm-refusal`:
    - Use REFUSAL_ETHICAL as positive, NORMAL as negative
    - Create contrast pairs from prompt field

  **Dataset loader pattern**:
  ```python
  def load_<concept>_data(config: ConceptConfig) -> list[ContrastPair]:
      # Load from HuggingFace
      # Filter/sample based on config.num_pairs
      # Return list of ContrastPair objects
  ```

  **Must NOT do**:
  - DO NOT add evaluation functions
  - DO NOT create separate files for each concept
  - DO NOT add caching or progress bars

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex data loading logic with new datasets
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (with Task 6)
  - **Blocks**: Task 7
  - **Blocked By**: Task 1, 2

  **References**:
  - `src/steering_geometry/concepts/honesty.py:61-117` - Pattern for honesty loader
  - `src/steering_geometry/concepts/sentiment.py:72-138` - Pattern for sentiment loader
  - Task 1 output - Verified dataset field structures

  **Acceptance Criteria**:
  - [ ] All 5 loader functions implemented in extract.py
  - [ ] `uv run python -m steering_geometry.extract --concept honesty --dry-run` prints contrast pair count
  - [ ] `uv run python -m steering_geometry.extract --concept toxicity --dry-run` prints contrast pair count
  - [ ] `uv run python -m steering_geometry.extract --concept sycophancy --dry-run` prints contrast pair count
  - [ ] `uv run python -m steering_geometry.extract --concept refusal --dry-run` prints contrast pair count
  - [ ] `uv run python -m steering_geometry.extract --concept sentiment --dry-run` prints contrast pair count

  **QA Scenarios**:
  ```
  Scenario: All concepts work with dry-run
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.extract --concept honesty --model sshleifer/tiny-gpt2 --dry-run
      2. uv run python -m steering_geometry.extract --concept sentiment --model sshleifer/tiny-gpt2 --dry-run
      3. uv run python -m steering_geometry.extract --concept toxicity --model sshleifer/tiny-gpt2 --dry-run
      4. uv run python -m steering_geometry.extract --concept sycophancy --model sshleifer/tiny-gpt2 --dry-run
      5. uv run python -m steering_geometry.extract --concept refusal --model sshleifer/tiny-gpt2 --dry-run
    Expected Result: All print "Loaded N contrast pairs" and exit with code 0
    Evidence: .omo/evidence/task-05-all-concepts-dryrun.txt
  ```

  **Commit**: NO (wait for complete implementation)

- [ ] 7. **Update Tests**

  **What to do**:
  - Create new `tests/unit/test_extract.py` with tests for:
    - CLI argument parsing
    - Concept validation (valid/invalid concepts)
    - Each dataset loader function
    - Full extraction flow (with tiny model)
  - Update `tests/conftest.py` if needed
  - Remove old test files that were deleted in Task 6

  **Test structure**:
  ```python
  # tests/unit/test_extract.py
  def test_cli_valid_concept():
      # Test --concept honesty, sentiment, etc.
  
  def test_cli_invalid_concept():
      # Test --concept invalid
  
  def test_load_honesty_data():
      # Test dataset loading
  
  # ... similar for other concepts
  
  def test_extract_vector_dry_run():
      # Test full flow with dry-run
  ```

  **Must NOT do**:
  - DO NOT create overly complex tests
  - DO NOT test steering or evaluation (deleted functionality)
  - DO NOT add slow integration tests

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Test design and implementation
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential)
  - **Blocks**: Task 8
  - **Blocked By**: Task 2, 3, 4, 5, 6

  **References**:
  - `tests/unit/test_honesty.py` - Pattern to follow (before deletion)
  - `tests/unit/test_extraction.py` - Pattern to follow (before deletion)

  **Acceptance Criteria**:
  - [ ] File created: tests/unit/test_extract.py
  - [ ] `uv run pytest tests/unit/test_extract.py` passes
  - [ ] Test coverage includes all 5 concepts

  **QA Scenarios**:
  ```
  Scenario: All tests pass
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_extract.py -v
    Expected Result: All tests pass, 0 failures
    Evidence: .omo/evidence/task-07-tests-pass.txt
  ```

  **Commit**: NO (wait for complete implementation)

- [ ] 8. **Update scripts/run_extractions.sh**

  **What to do**:
  - Rewrite script to use unified CLI
  - Change from:
    ```bash
    uv run python -m steering_geometry.extract_honesty --model "$MODEL"
    ```
  - To:
    ```bash
    uv run python -m steering_geometry.extract --concept honesty --model "$MODEL"
    ```
  - Update all 5 concept invocations
  - Keep all existing options (-c, -m, -p, -M, -o)

  **Must NOT do**:
  - DO NOT change script behavior or options
  - DO NOT add new features to the script
  - DO NOT break existing usage patterns

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple script update
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential)
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References**:
  - `scripts/run_extractions.sh` - Current script to update

  **Acceptance Criteria**:
  - [ ] Script updated to use unified CLI
  - [ ] `./scripts/run_extractions.sh -c honesty --dry-run` works (if dry-run supported)

  **QA Scenarios**:
  ```
  Scenario: Script works with new CLI
    Tool: Bash
    Steps:
      1. grep -q "steering_geometry.extract --concept" scripts/run_extractions.sh
    Expected Result: grep finds matches
    Evidence: .omo/evidence/task-08-script-updated.txt
  ```

  **Commit**: NO (wait for complete implementation)

- [ ] 9. **Final Verification**

  **What to do**:
  - Run all quality checks
  - Verify all acceptance criteria
  - Test full extraction with tiny model
  - Verify output file format

  **Quality checks**:
  ```bash
  uv sync
  uv run ruff check src/ tests/
  uv run ruff format --check src/ tests/
  uv run mypy src/
  uv run pytest
  ```

  **Extraction test**:
  ```bash
  uv run python -m steering_geometry.extract --concept honesty --model sshleifer/tiny-gpt2 --num-pairs 10 --output /tmp/test_vectors/
  ```

  **Output verification**:
  ```bash
  uv run python -c "import torch; d = torch.load('/tmp/test_vectors/honesty_sshleifer_tiny-gpt2_mean.pt'); assert 'vector' in d; assert 'num_pairs' in d; assert 'evaluation' not in d"
  ```

  **Must NOT do**:
  - DO NOT skip any quality checks
  - DO NOT proceed if any check fails

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verification and cleanup
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential, final)
  - **Blocks**: None
  - **Blocked By**: Task 8

  **References**:
  - AGENTS.md - Definition of Done checklist

  **Acceptance Criteria**:
  - [ ] `uv sync` completes without errors
  - [ ] `uv run ruff check src/ tests/` → 0 violations
  - [ ] `uv run ruff format --check src/ tests/` → already formatted
  - [ ] `uv run mypy src/` → Success with 0 errors
  - [ ] `uv run pytest` → all tests pass
  - [ ] Extraction produces valid .pt file

  **QA Scenarios**:
  ```
  Scenario: All quality checks pass
    Tool: Bash
    Steps:
      1. uv sync
      2. uv run ruff check src/ tests/
      3. uv run ruff format --check src/ tests/
      4. uv run mypy src/
      5. uv run pytest
    Expected Result: All commands exit with code 0
    Evidence: .omo/evidence/task-09-quality-checks.txt

  Scenario: Extraction produces valid output
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.extract --concept honesty --model sshleifer/tiny-gpt2 --num-pairs 10 --output /tmp/test_vectors/
      2. uv run python -c "import torch; d = torch.load('/tmp/test_vectors/honesty_sshleifer_tiny-gpt2_mean.pt'); print('vector' in d, 'num_pairs' in d, 'evaluation' not in d)"
    Expected Result: Prints "True True True"
    Evidence: .omo/evidence/task-09-extraction-output.txt
  ```

  **Commit**: YES
  - Message: `refactor: consolidate vector extraction into single module`
  - Files: All changed files
  - Pre-commit: `uv run pytest`

---

## Final Verification Wave (MANDATORY)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run mypy src/` + `uv run ruff check src/ tests/` + `uv run pytest`. Review all changed files for: `Any` types, empty catches, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run actual extraction with tiny model for all 5 concepts. Verify CLI works. Check output files. Test error cases (invalid concept, missing args).
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", verify implementation matches spec. Check "Must NOT do" compliance. Detect scope creep.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Single commit**: All changes in one atomic commit after Task 9
  - Message: `refactor: consolidate vector extraction into single module

  - Merge all extract_*.py into single extract.py
  - Replace synthetic data with HuggingFace datasets
  - Remove steering and evaluation functionality
  - Simplify types.py, models.py, config.py
  - Update tests and scripts`
  - Files: All changed files
  - Pre-commit: `uv run pytest`

---

## Success Criteria

### Verification Commands
```bash
# All quality checks pass
uv sync
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest

# CLI works for all concepts
uv run python -m steering_geometry.extract --concept honesty --model sshleifer/tiny-gpt2 --dry-run
uv run python -m steering_geometry.extract --concept sentiment --model sshleifer/tiny-gpt2 --dry-run
uv run python -m steering_geometry.extract --concept toxicity --model sshleifer/tiny-gpt2 --dry-run
uv run python -m steering_geometry.extract --concept sycophancy --model sshleifer/tiny-gpt2 --dry-run
uv run python -m steering_geometry.extract --concept refusal --model sshleifer/tiny-gpt2 --dry-run

# Obsolete files are gone
test ! -f src/steering_geometry/evaluation.py
test ! -f src/steering_geometry/extract_honesty.py
test ! -d src/steering_geometry/concepts
```

### Final Checklist
- [ ] All "Must Have" present (5 concepts, unified CLI, HuggingFace datasets)
- [ ] All "Must NOT Have" absent (no steering, no evaluation, no new concepts)
- [ ] All tests pass
- [ ] All quality checks pass
- [ ] Obsolete files deleted
- [ ] Scripts updated


  **What to do**:
  - Delete `src/steering_geometry/evaluation.py`
  - Delete `src/steering_geometry/compare_concepts.py`
  - Delete `src/steering_geometry/hello.py`
  - Delete `src/steering_geometry/extract_honesty.py`
  - Delete `src/steering_geometry/extract_sentiment.py`
  - Delete `src/steering_geometry/extract_toxicity.py`
  - Delete `src/steering_geometry/extract_sycophancy.py`
  - Delete `src/steering_geometry/extract_refusal.py`
  - Delete entire `src/steering_geometry/concepts/` directory
  - Delete related test files:
    - `tests/unit/test_honesty.py`
    - `tests/unit/test_sentiment.py`
    - `tests/unit/test_toxicity.py`
    - `tests/unit/test_sycophancy.py`
    - `tests/unit/test_refusal.py`
    - `tests/unit/test_models.py` (if only tests steering)
    - `tests/unit/test_extraction.py` (will be replaced)
    - `tests/integration/test_pipeline.py`
  - Update `src/steering_geometry/__init__.py` to export from extract.py

  **Must NOT do**:
  - DO NOT delete types.py, models.py, config.py
  - DO NOT delete tests/conftest.py
  - DO NOT delete tests/__init__.py

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple file deletion
  - **Skills**: [`git-master`]
    - git-master: For clean deletion tracking

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 2, 3, 4, 5 (must complete implementation first)

  **References**:
  - None (deletion task)

  **Acceptance Criteria**:
  - [ ] All obsolete files deleted
  - [ ] `test ! -f src/steering_geometry/evaluation.py` passes
  - [ ] `test ! -d src/steering_geometry/concepts` passes
  - [ ] `test ! -f src/steering_geometry/extract_honesty.py` passes
  - [ ] `uv run python -c "from steering_geometry import extract_vector"` works

  **QA Scenarios**:
  ```
  Scenario: All obsolete files deleted
    Tool: Bash
    Steps:
      1. test ! -f src/steering_geometry/evaluation.py
      2. test ! -f src/steering_geometry/compare_concepts.py
      3. test ! -f src/steering_geometry/hello.py
      4. test ! -d src/steering_geometry/concepts
      5. test ! -f src/steering_geometry/extract_honesty.py
    Expected Result: All tests pass (files don't exist)
    Evidence: .omo/evidence/task-06-files-deleted.txt
  ```

  **Commit**: NO (wait for complete implementation)



  **What to do**:
  - Remove `generate()` method with `steering_vector` parameter (lines 149-203)
  - Keep model loading and activation extraction functionality
  - Remove `SteeringVector` import if no longer needed

  **Must NOT do**:
  - DO NOT modify model loading logic
  - DO NOT modify activation extraction logic
  - DO NOT remove HookedModel class

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple deletion of unused method
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 4)
  - **Blocks**: Task 7
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/models.py:149-203` - generate() method to remove
  - `src/steering_geometry/models.py:1-11` - Imports to clean up

  **Acceptance Criteria**:
  - [ ] generate() method removed from models.py
  - [ ] `uv run mypy src/steering_geometry/models.py` passes
  - [ ] `uv run ruff check src/steering_geometry/models.py` passes

  **QA Scenarios**:
  ```
  Scenario: Type check passes
    Tool: Bash
    Steps:
      1. uv run mypy src/steering_geometry/models.py
    Expected Result: Success with 0 errors
    Evidence: .omo/evidence/task-03-mypy.txt
  ```

  **Commit**: NO (wait for complete implementation)

