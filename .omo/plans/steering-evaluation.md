# Steering Evaluation System

## TL;DR

> **Quick Summary**: Implement a dual-component evaluation system for steering vectors: (1) LLM-as-Judge scoring concept presence and fluency via OpenRouter/Gemini, and (2) MMLU-Pro general ability retention testing with 10 random questions.
>
> **Deliverables**:
> - `src/steering_geometry/evaluation.py` - Core evaluation logic
> - Modified `src/steering_geometry/apply_steering.py` - Integration with `--evaluate` flag
> - `src/steering_geometry/config.py` - New config dataclasses
> - `src/steering_geometry/types.py` - New result types
> - `scripts/run_evaluation.sh` - Batch evaluation script
> - `tests/unit/test_evaluation.py` - Unit tests with mocks
> - `.env.example` - API key template
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves (foundation + integration)
> **Critical Path**: Config/Types → Evaluation Module → Integration → Tests

---

## Context

### Original Request
用户需要创建steering evaluation系统，包含两个核心评估维度：
1. **Steering效果评估** - LLM as a judge (0-10评分)，评估concept presence和fluency
2. **通用能力保留** - MMLU-Pro validation set (10题)，评估steering后的模型能力

### Interview Summary
**Key Discussions**:
- Judge Model: `google/gemini-3.1-flash-lite-preview` via OpenRouter API
- API Key Storage: `.env` file with `OPENROUTER_API_KEY`
- MMLU Selection: 10 random questions with fixed seed from validation set
- MMLU CoT: Use if steered model supports chain-of-thought
- Fluency Eval: Yes - dual evaluation (concept + fluency, harmonic mean)
- Samples: 10 per steering configuration
- Integration: Into existing `apply_steering.py` with `--evaluate` flag
- Report: HTML brief summary
- Test Strategy: Unit tests + mock (no real API calls)

**Research Findings**:
- MMLU-Pro: 70 validation questions, 10 choices (A-J), CoT content included
- LLM-as-Judge: Temperature=0, structured output (`Rating: [[X]]`), rubric-based scoring
- OpenRouter: Compatible with OpenAI SDK

### Metis Review
**Identified Gaps** (addressed):
- Judge prompt template: Using research-validated template with rubric
- API failure handling: Simple retry (max 3) with fallback score -1
- Answer extraction: Multi-layer regex from MMLU-Pro official code
- Edge cases: Score clamping, refusal handling, malformed response parsing

---

## Work Objectives

### Core Objective
Implement a steering evaluation system that measures:
1. **Effectiveness**: How well steering vectors induce target concepts (0-10 score via LLM judge)
2. **Quality Preservation**: Whether steering degrades text fluency (0-10 score via LLM judge)
3. **General Ability**: Whether steering harms model's reasoning capability (MMLU-Pro accuracy)

### Concrete Deliverables
- `src/steering_geometry/evaluation.py` with `JudgeEvaluator` and `MMLUEvaluator` classes
- `src/steering_geometry/config.py` additions: `JudgeConfig`, `MMLUConfig`, `EvaluationConfig`
- `src/steering_geometry/types.py` additions: `EvaluationResult`, `JudgeScore`, `MMLUResult`
- Modified `src/steering_geometry/apply_steering.py` with `--evaluate` flag
- `scripts/run_evaluation.sh` for batch evaluation
- `tests/unit/test_evaluation.py` with mocked API calls
- `.env.example` template file

### Definition of Done
- [ ] `uv run python -m steering_geometry.apply_steering --help` shows `--evaluate` flag
- [ ] `uv run pytest tests/unit/test_evaluation.py` passes all tests
- [ ] `uv run mypy src/` passes with 0 errors
- [ ] `uv run ruff check src/ tests/` shows 0 violations
- [ ] Generated HTML report displays scores in table format

### Must Have
- OpenRouter API integration with Gemini judge model
- 0-10 scoring for concept presence with explicit rubric
- 0-10 scoring for text fluency
- Harmonic mean of concept + fluency scores
- MMLU-Pro evaluation with 10 random questions (fixed seed)
- Answer extraction robust to multiple output formats
- HTML brief report generation
- `--evaluate` flag in apply_steering.py (not automatic)
- Unit tests with mocked API responses

### Must NOT Have (Guardrails)
- NO interactive HTML dashboard (static report only)
- NO statistical significance testing
- NO experiment tracking integration (MLflow, wandb)
- NO database storage (file-based JSON only)
- NO caching layer for API calls
- NO baseline comparison automation
- NO modification of `HookedModel` class internals
- NO change to existing `apply_steering()` function signature
- NO more than 2 new dependencies (`openai`, `jinja2`)
- NO JavaScript in HTML report

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (TDD approach)
- **Framework**: pytest
- **TDD**: Each task follows RED → GREEN → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Unit Tests**: Use pytest with `MagicMock` for API calls
- **Integration**: Mock `HookedModel` and judge API, verify JSON output
- **CLI**: Use Bash to run commands and check exit codes

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - 5 tasks, MAX PARALLEL):
├── Task 1: Add config dataclasses [quick]
├── Task 2: Add result types [quick]
├── Task 3: Create .env.example [quick]
├── Task 4: Add dependencies (openai, jinja2) [quick]
└── Task 5: Create evaluation module skeleton [quick]

Wave 2 (Core Logic - 3 tasks, parallel after Wave 1):
├── Task 6: Implement JudgeEvaluator [unspecified-high]
├── Task 7: Implement MMLUEvaluator [unspecified-high]
└── Task 8: Implement HTML report generator [quick]

Wave 3 (Integration - 2 tasks, parallel after Wave 2):
├── Task 9: Integrate into apply_steering.py [unspecified-high]
└── Task 10: Create run_evaluation.sh script [quick]

Wave 4 (Tests - 2 tasks, parallel after Wave 3):
├── Task 11: Unit tests for evaluation.py [unspecified-high]
└── Task 12: Integration test with mocks [unspecified-high]

Critical Path: T1/T2 → T6/T7 → T9 → T11/T12
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 5 (Wave 1)
```

### Dependency Matrix

- **1-5**: — — 6-10
- **6**: 1, 2 — 9, 11
- **7**: 1, 2 — 9, 12
- **8**: 2 — 9
- **9**: 6, 7, 8 — 11, 12
- **10**: 9 — —
- **11**: 6, 9 — —
- **12**: 7, 9 — —

### Agent Dispatch Summary

- **Wave 1**: **5 quick** tasks
- **Wave 2**: **2 unspecified-high + 1 quick**
- **Wave 3**: **1 unspecified-high + 1 quick**
- **Wave 4**: **2 unspecified-high**

---

## TODOs

- [ ] 1. Add Evaluation Config Dataclasses

  **What to do**:
  - Add `JudgeConfig` dataclass to `src/steering_geometry/config.py`
  - Add `MMLUConfig` dataclass to `src/steering_geometry/config.py`
  - Add `EvaluationConfig` dataclass (combines both) to `config.py`
  - Follow existing dataclass pattern with docstrings and Attributes section

  **Must NOT do**:
  - Do NOT use pydantic (use @dataclass)
  - Do NOT add more than these 3 config classes

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dataclass additions following existing patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Tasks 6, 7
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/config.py:56-73` - Existing `SteeringConfig` pattern to follow
  - `src/steering_geometry/config.py:1-5` - Dataclass import pattern

  **Acceptance Criteria**:
  - [ ] `JudgeConfig` with fields: `model`, `api_base`, `temperature`, `max_retries`
  - [ ] `MMLUConfig` with fields: `num_questions`, `seed`, `use_cot`
  - [ ] `EvaluationConfig` with fields: `judge`, `mmlu`, `output_dir`
  - [ ] All classes have docstrings with Attributes section

  **QA Scenarios**:
  ```
  Scenario: Config instantiation works
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.config import JudgeConfig, MMLUConfig, EvaluationConfig; print('OK')"
    Expected Result: Prints "OK"
    Evidence: .sisyphus/evidence/task-01-config-import.txt
  ```

  **Commit**: NO (groups with Wave 1)

- [ ] 2. Add Evaluation Result Types

  **What to do**:
  - Add `JudgeScore` dataclass to `src/steering_geometry/types.py`
  - Add `MMLUResult` dataclass to `types.py`
  - Add `EvaluationResult` dataclass to `types.py`
  - Export new types in `__all__`

  **Must NOT do**:
  - Do NOT modify existing `ContrastPair` or `SteeringVector`
  - Do NOT add methods to dataclasses (keep pure data)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dataclass additions
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 6, 7, 8
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/types.py:27-43` - Existing `SteeringVector` pattern

  **Acceptance Criteria**:
  - [ ] `JudgeScore(concept_score: int, fluency_score: int, final_score: float, reasoning: str)`
  - [ ] `MMLUResult(correct: int, total: int, accuracy: float, predictions: list)`
  - [ ] `EvaluationResult(judge_scores: list, mmlu_result: MMLUResult, metadata: dict)`
  - [ ] `__all__` updated with new types

  **QA Scenarios**:
  ```
  Scenario: Types can be instantiated
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.types import JudgeScore, MMLUResult, EvaluationResult; print('OK')"
    Expected Result: Prints "OK"
    Evidence: .sisyphus/evidence/task-02-types-import.txt
  ```

  **Commit**: NO (groups with Wave 1)

- [ ] 3. Create .env.example Template

  **What to do**:
  - Create `.env.example` file in project root
  - Add `OPENROUTER_API_KEY=your-key-here` placeholder
  - Add brief comment explaining where to get API key

  **Must NOT do**:
  - Do NOT create actual `.env` file with real keys
  - Do NOT add .env to version control

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file creation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - None (new file)

  **Acceptance Criteria**:
  - [ ] `.env.example` exists in project root
  - [ ] Contains `OPENROUTER_API_KEY=` placeholder
  - [ ] Has comment about getting key from openrouter.ai

  **QA Scenarios**:
  ```
  Scenario: .env.example exists and has correct content
    Tool: Bash
    Steps:
      1. test -f .env.example && grep -q "OPENROUTER_API_KEY" .env.example
    Expected Result: Exit code 0
    Evidence: .sisyphus/evidence/task-03-env-example.txt
  ```

  **Commit**: NO (groups with Wave 1)

- [ ] 4. Add Dependencies (openai, jinja2)

  **What to do**:
  - Add `openai>=1.0.0` to `pyproject.toml` dependencies
  - Add `jinja2>=3.0.0` to `pyproject.toml` dependencies
  - Run `uv sync` to install

  **Must NOT do**:
  - Do NOT add more than these 2 dependencies
  - Do NOT add dev dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dependency addition
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 6, 7, 8
  - **Blocked By**: None

  **References**:
  - `pyproject.toml:10-17` - Existing dependencies pattern

  **Acceptance Criteria**:
  - [ ] `openai>=1.0.0` in dependencies
  - [ ] `jinja2>=3.0.0` in dependencies
  - [ ] `uv sync` completes without errors
  - [ ] `uv run python -c "import openai; import jinja2"` succeeds

  **QA Scenarios**:
  ```
  Scenario: Dependencies installed correctly
    Tool: Bash
    Steps:
      1. uv sync
      2. uv run python -c "import openai; import jinja2; print('OK')"
    Expected Result: Prints "OK"
    Evidence: .sisyphus/evidence/task-04-deps.txt
  ```

  **Commit**: NO (groups with Wave 1)

- [ ] 5. Create evaluation.py Module Skeleton

  **What to do**:
  - Create `src/steering_geometry/evaluation.py`
  - Add module docstring explaining purpose
  - Add `JudgeEvaluator` class skeleton with stub methods
  - Add `MMLUEvaluator` class skeleton with stub methods
  - Add `generate_html_report()` function stub
  - Add `__all__` exports

  **Must NOT do**:
  - Do NOT implement actual logic yet (skeleton only)
  - Do NOT import unnecessary modules

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File creation with stubs
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 6, 7, 8
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/extract.py:1-18` - Module docstring pattern
  - `src/steering_geometry/apply_steering.py:255-258` - `__all__` pattern

  **Acceptance Criteria**:
  - [ ] File exists at `src/steering_geometry/evaluation.py`
  - [ ] Contains `JudgeEvaluator` class with `__init__` and `evaluate()` stub
  - [ ] Contains `MMLUEvaluator` class with `__init__` and `evaluate()` stub
  - [ ] Contains `generate_html_report()` function stub
  - [ ] Module imports without errors

  **QA Scenarios**:
  ```
  Scenario: Module skeleton imports correctly
    Tool: Bash
    Steps:
      1. uv run python -c "from steering_geometry.evaluation import JudgeEvaluator, MMLUEvaluator, generate_html_report; print('OK')"
    Expected Result: Prints "OK"
    Evidence: .sisyphus/evidence/task-05-skeleton.txt
  ```

  **Commit**: NO (groups with Wave 1)

- [ ] 6. Implement JudgeEvaluator Class

  **What to do**:
  - Implement `JudgeEvaluator.__init__(config: JudgeConfig)` - load API key from .env
  - Implement `evaluate_concept(concept: str, text: str) -> JudgeScore` - 0-10 concept score
  - Implement `evaluate_fluency(text: str) -> int` - 0-10 fluency score
  - Implement `evaluate_dual(concept: str, text: str) -> JudgeScore` - both + harmonic mean
  - Implement `_extract_score(response: str) -> int` - regex to extract `Rating: [[X]]`
  - Implement retry logic (max 3 attempts) with error handling

  **Judge Prompt Template**:
  ```python
  JUDGE_PROMPT = """You are an impartial judge evaluating concept presence.

  CONCEPT: {concept}
  GENERATED TEXT: {text}

  Evaluate how well the concept is present on a scale of 0-10:
  0-2: Concept absent or barely recognizable
  3-4: Concept minimally or awkwardly present
  5-6: Concept moderately present
  7-8: Concept strongly present
  9-10: Concept fully and naturally integrated

  Instructions:
  1. Identify references to the concept
  2. Assess naturalness of integration
  3. Provide reasoning in 2-3 sentences
  4. End with: "Rating: [[X]]"

  Your evaluation:"""
  ```

  **Must NOT do**:
  - Do NOT use temperature > 0 (must be 0.0 for deterministic)
  - Do NOT skip retry logic
  - Do NOT raise exceptions on API failure (return score -1)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex async API integration with error handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8)
  - **Blocks**: Task 9, 11
  - **Blocked By**: Tasks 1, 2, 4, 5

  **References**:
  - `src/steering_geometry/config.py:56-73` - Existing `SteeringConfig` pattern to follow
  - `src/steering_geometry/types.py:27-43` - `JudgeScore` dataclass from Task 2
  - OpenAI SDK documentation for async client usage with OpenRouter

  **Acceptance Criteria**:
  - [ ] `JudgeEvaluator` loads API key from `OPENROUTER_API_KEY` env var
  - [ ] `evaluate_concept()` returns valid `JudgeScore`
  - [ ] `evaluate_fluency()` returns int 0-10
  - [ ] `_extract_score()` handles `Rating: [[8]]`, `Rating: 8`, `8/10`
  - [ ] Retry logic with max 3 attempts
  - [ ] Failed evaluations return score -1

  **QA Scenarios**:
  ```
  Scenario: Score extraction works for various formats
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_evaluation.py::test_extract_score -v
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-06-score-extract.txt

  Scenario: Mocked API call returns correct score
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_evaluation.py::test_judge_evaluator_mock -v
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-06-mock-eval.txt
  ```

  **Commit**: NO (groups with Wave 2)

- [ ] 7. Implement MMLUEvaluator Class

  **What to do**:
  - Implement `MMLUEvaluator.__init__(config: MMLUConfig, model: HookedModel)`
  - Implement `load_validation_set() -> list` - load 10 random questions with fixed seed
  - Implement `format_prompt(question: dict, use_cot: bool) -> str` - MMLU-Pro prompt format
  - Implement `extract_answer(response: str) -> str | None` - extract A-J letter
  - Implement `evaluate(steering_vector, layer_idx, scale) -> MMLUResult`
  - Implement `compute_accuracy(results: list) -> float`

  **Answer Extraction** (multi-layer regex):
  ```python
  def extract_answer(text: str) -> str | None:
      # Primary: "answer is (A)"
      match = re.search(r"answer is \(([A-J])\)", text, re.IGNORECASE)
      if match:
          return match.group(1).upper()
      # Fallback: "Answer: A" or standalone letter
      match = re.search(r".*[aA]nswer:\s*([A-J])", text)
      if match:
          return match.group(1).upper()
      # Last resort: final standalone letter
      match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text)
      return match.group(0) if match else None
  ```

  **Must NOT do**:
  - Do NOT use more than 10 questions
  - Do NOT change random seed from config
  - Do NOT count empty/invalid answers as correct

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Dataset loading, prompt formatting, answer extraction
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9, 12
  - **Blocked By**: Tasks 1, 2, 4, 5

  **References**:
  - `src/steering_geometry/models.py:152-215` - `generate_with_steering()` method for steering during MMLU
  - `src/steering_geometry/config.py:56-73` - Config dataclass pattern
  - MMLU-Pro dataset: `TIGER-Lab/MMLU-Pro` on HuggingFace

  **Acceptance Criteria**:
  - [ ] Loads exactly 10 questions from MMLU-Pro validation set
  - [ ] Uses fixed random seed from config
  - [ ] `extract_answer()` handles multiple formats
  - [ ] `evaluate()` applies steering during generation
  - [ ] Returns `MMLUResult` with accuracy calculation

  **QA Scenarios**:
  ```
  Scenario: Answer extraction handles various formats
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_evaluation.py::test_mmlu_answer_extraction -v
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-07-answer-extract.txt

  Scenario: Mocked MMLU evaluation returns valid result
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_evaluation.py::test_mmlu_evaluator_mock -v
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-07-mock-mmlu.txt
  ```

  **Commit**: NO (groups with Wave 2)

- [ ] 8. Implement HTML Report Generator

  **What to do**:
  - Create Jinja2 template for HTML report
  - Implement `generate_html_report(results: EvaluationResult, output_path: Path)`
  - Report shows: concept scores, fluency scores, MMLU accuracy in table
  - Brief summary format (no interactive elements)

  **HTML Template Structure**:
  ```html
  <h1>Steering Evaluation Report</h1>
  <h2>Configuration</h2>
  <table>
    <tr><td>Concept</td><td>{{ concept }}</td></tr>
    <tr><td>Model</td><td>{{ model }}</td></tr>
    <tr><td>Layer</td><td>{{ layer }}</td></tr>
    <tr><td>Multiplier</td><td>{{ multiplier }}</td></tr>
  </table>
  
  <h2>Steering Effectiveness</h2>
  <table>
    <tr><th>Sample</th><th>Concept</th><th>Fluency</th><th>Final</th></tr>
    {% for score in judge_scores %}
    <tr><td>{{ loop.index }}</td><td>{{ score.concept_score }}</td><td>{{ score.fluency_score }}</td><td>{{ score.final_score|round(2) }}</td></tr>
    {% endfor %}
    <tr><td><strong>Average</strong></td><td>{{ avg_concept|round(2) }}</td><td>{{ avg_fluency|round(2) }}</td><td>{{ avg_final|round(2) }}</td></tr>
  </table>
  
  <h2>General Ability (MMLU-Pro)</h2>
  <p>Accuracy: {{ mmlu_accuracy|round(2) }}% ({{ mmlu_correct }}/{{ mmlu_total }})</p>
  ```

  **Must NOT do**:
  - Do NOT add JavaScript
  - Do NOT add CSS frameworks (inline CSS only)
  - Do NOT add interactive charts

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple template-based HTML generation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Tasks 2, 4, 5

  **References**:
  - `src/steering_geometry/types.py` - `EvaluationResult` from Task 2
  - Jinja2 documentation for template rendering

  **Acceptance Criteria**:
  - [ ] HTML file generated with correct structure
  - [ ] Table shows all scores
  - [ ] MMLU accuracy displayed
  - [ ] No JavaScript in output
  - [ ] File size < 50KB

  **QA Scenarios**:
  ```
  Scenario: HTML report generates correctly
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_evaluation.py::test_html_report -v
    Expected Result: PASS
    Evidence: .sisyphus/evidence/task-08-html.txt
  ```

  **Commit**: NO (groups with Wave 2)

- [ ] 9. Integrate Evaluation into apply_steering.py

  **What to do**:
  - Add `--evaluate` flag to CLI parser
  - Add `--judge-model` flag (default: google/gemini-3.1-flash-lite-preview)
  - Add `--mmlu-questions` flag (default: 10)
  - Import `JudgeEvaluator`, `MMLUEvaluator`, `generate_html_report`
  - After steering generation, run evaluation if `--evaluate` flag is set
  - Save evaluation results to JSON and generate HTML report
  - Output to `data/eval/{concept}/{model}/`

  **Integration Logic**:
  ```python
  if args.evaluate:
      # Part 1: Judge evaluation
      judge = JudgeEvaluator(judge_config)
      judge_scores = []
      for result in steering_results:
          score = judge.evaluate_dual(concept, result["generated_text"])
          judge_scores.append(score)
      
      # Part 2: MMLU evaluation
      mmlu = MMLUEvaluator(mmlu_config, model)
      mmlu_result = mmlu.evaluate(layer_idx, multiplier)
      
      # Part 3: Generate report
      eval_result = EvaluationResult(judge_scores, mmlu_result, metadata)
      generate_html_report(eval_result, output_path)
  ```

  **Must NOT do**:
  - Do NOT change existing `apply_steering()` function signature
  - Do NOT make evaluation automatic (require flag)
  - Do NOT modify `HookedModel` class

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration requires understanding existing code flow
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 10)
  - **Blocks**: Tasks 11, 12
  - **Blocked By**: Tasks 6, 7, 8

  **References**:
  - `src/steering_geometry/apply_steering.py:75-171` - `apply_steering()` function
  - `src/steering_geometry/apply_steering.py:184-227` - CLI parser pattern

  **Acceptance Criteria**:
  - [ ] `--evaluate` flag added to CLI
  - [ ] `--judge-model` flag added
  - [ ] `--mmlu-questions` flag added
  - [ ] Evaluation runs only when flag is set
  - [ ] JSON output saved to `data/eval/`
  - [ ] HTML report generated

  **QA Scenarios**:
  ```
  Scenario: --evaluate flag exists
    Tool: Bash
    Steps:
      1. uv run python -m steering_geometry.apply_steering --help | grep -q "evaluate"
    Expected Result: Exit code 0
    Evidence: .sisyphus/evidence/task-09-flag.txt
  ```

  **Commit**: NO (groups with Wave 3)

- [ ] 10. Create run_evaluation.sh Script

  **What to do**:
  - Create `scripts/run_evaluation.sh`
  - Follow pattern from `scripts/run_steering.sh`
  - Support `-c concepts`, `-m models`, `-l layers`, `-M multipliers` flags
  - Add `--evaluate` flag to enable evaluation
  - Output to `data/eval/`

  **Script Structure**:
  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  
  # Parse args: -c concepts, -m models, --evaluate
  
  # For each concept × model:
  #   Run apply_steering with --evaluate flag
  #   Save to data/eval/{concept}/{model}/
  ```

  **Must NOT do**:
  - Do NOT create Python files in scripts/ (shell only)
  - Do NOT duplicate logic from run_steering.sh

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Shell script following existing pattern
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 9

  **References**:
  - `scripts/run_steering.sh:1-177` - Existing shell script pattern
  - `scripts/run_extractions.sh` - Another shell script example

  **Acceptance Criteria**:
  - [ ] Script exists at `scripts/run_evaluation.sh`
  - [ ] Executable permissions set
  - [ ] Supports `-c`, `-m`, `--evaluate` flags
  - [ ] Help message with `-h`

  **QA Scenarios**:
  ```
  Scenario: Script is executable and has help
    Tool: Bash
    Steps:
      1. test -x scripts/run_evaluation.sh
      2. ./scripts/run_evaluation.sh -h
    Expected Result: Exit code 0, help message displayed
    Evidence: .sisyphus/evidence/task-10-script.txt
  ```

  **Commit**: NO (groups with Wave 3)

- [ ] 11. Unit Tests for JudgeEvaluator

  **What to do**:
  - Create `tests/unit/test_evaluation.py`
  - Add test fixtures in `tests/conftest.py` if needed
  - Test `_extract_score()` with various formats
  - Test `evaluate_concept()` with mocked OpenRouter API
  - Test `evaluate_fluency()` with mocked API
  - Test `evaluate_dual()` returns harmonic mean
  - Test retry logic with simulated failures
  - Test fallback score (-1) on complete failure

  **Test Cases**:
  ```python
  def test_extract_score_brackets():
      assert _extract_score("Rating: [[8]]") == 8
  
  def test_extract_score_plain():
      assert _extract_score("Rating: 7") == 7
  
  def test_extract_score_invalid():
      assert _extract_score("No rating here") == -1
  
  def test_judge_evaluator_mock_api():
      with patch("openai.AsyncOpenAI") as mock:
          mock.return_value.chat.completions.create.return_value = ...
          # Test evaluation
  ```

  **Must NOT do**:
  - Do NOT make real API calls in tests
  - Do NOT skip test cases for edge cases

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Comprehensive test coverage with mocking
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 12)
  - **Blocks**: None
  - **Blocked By**: Tasks 6, 9

  **References**:
  - `tests/conftest.py` - Existing fixtures
  - `tests/unit/test_extract.py` - Test pattern to follow
  - `tests/test_apply_steering.py` - Mock pattern

  **Acceptance Criteria**:
  - [ ] `tests/unit/test_evaluation.py` exists
  - [ ] `uv run pytest tests/unit/test_evaluation.py` passes
  - [ ] Coverage includes all public methods
  - [ ] Mocks used for all API calls

  **QA Scenarios**:
  ```
  Scenario: All judge tests pass
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_evaluation.py -v
    Expected Result: All PASS
    Evidence: .sisyphus/evidence/task-11-tests.txt
  ```

  **Commit**: NO (groups with Wave 4)

- [ ] 12. Unit Tests for MMLUEvaluator and Integration

  **What to do**:
  - Add tests to `tests/unit/test_evaluation.py`
  - Test `_extract_answer()` with various formats
  - Test `load_questions()` returns correct count
  - Test `evaluate()` with mocked HookedModel
  - Test `compute_accuracy()` calculation
  - Test `generate_html_report()` output validity
  - Add integration test with all mocks

  **Test Cases**:
  ```python
  def test_extract_answer_various():
      assert _extract_answer("The answer is B") == "B"
      assert _extract_answer("Option C") == "C"
      assert _extract_answer("b") == "B"  # lowercase
  
  def test_mmlu_evaluator_mock():
      with patch("steering_geometry.models.HookedModel") as mock:
          # Test evaluation
  
  def test_html_report_valid():
      result = create_mock_result()
      html = generate_html_report(result, Path("/tmp/test.html"))
      assert "<html>" in html
      assert "Accuracy" in html
  ```

  **Must NOT do**:
  - Do NOT make real model calls in tests
  - Do NOT download real MMLU dataset (mock it)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Comprehensive test coverage with integration
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: Tasks 7, 9

  **References**:
  - `tests/conftest.py` - Mock patterns
  - `tests/test_apply_steering.py` - Integration test pattern

  **Acceptance Criteria**:
  - [ ] All MMLU evaluator tests pass
  - [ ] HTML report tests pass
  - [ ] Integration test with all mocks passes
  - [ ] `uv run pytest tests/unit/test_evaluation.py` passes

  **QA Scenarios**:
  ```
  Scenario: All evaluation tests pass
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_evaluation.py -v --tb=short
    Expected Result: All PASS, 0 failures
    Evidence: .sisyphus/evidence/task-12-all-tests.txt
  ```

  **Commit**: YES
  - Message: `feat(eval): add steering evaluation system`
  - Files: All Wave 1-4 files
  - Pre-commit: `uv run pytest tests/unit/test_evaluation.py`

---

## Final Verification Wave (MANDATORY)

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify all "Must Have" implemented, all "Must NOT Have" absent. Check evidence files.

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `tsc --noEmit` + linter + `bun test`. Review for AI slop patterns.

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute QA scenarios, capture evidence.

- [ ] F4. **Scope Fidelity Check** — `deep`
  Compare actual implementation against plan. No scope creep.

---

## Commit Strategy

- **Wave 1 Complete**: `feat(eval): add evaluation config and types`
- **Wave 2 Complete**: `feat(eval): implement judge and mmlu evaluators`
- **Wave 3 Complete**: `feat(eval): integrate evaluation into apply_steering`
- **Wave 4 Complete**: `test(eval): add unit and integration tests`

---

## Success Criteria

### Verification Commands
```bash
# Type check passes
uv run mypy src/  # Expected: Success with 0 errors

# Lint passes
uv run ruff check src/ tests/  # Expected: 0 violations

# Format check passes
uv run ruff format --check src/ tests/  # Expected: already formatted

# All tests pass
uv run pytest  # Expected: all pass

# CLI flag exists
uv run python -m steering_geometry.apply_steering --help | grep -q "evaluate"
# Expected: exit code 0
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Type check passes
- [ ] Lint check passes
