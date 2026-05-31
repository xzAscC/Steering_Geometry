# Apply Steering Vector - Work Plan

## TL;DR

> **Quick Summary**: 创建 steering 应用模块，加载已提取的 steering vector，对模型逐层做 activation steering，保存生成的文本结果到 JSONL 文件。
> 
> **Deliverables**:
> - `src/steering_geometry/apply_steering.py` - 主模块 + CLI 入口
> - `src/steering_geometry/config.py` - 新增 `SteeringConfig` dataclass
> - `src/steering_geometry/models.py` - 扩展 `generate_with_steering()` 方法
> - `tests/test_apply_steering.py` - 集成测试
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: SteeringConfig → generate_with_steering → apply_steering module → CLI

---

## Context

### Original Request
提取 model steering vector 之后，这个 steering vector 应该是一个 layer * dim 的 vector，然后在 model 的每一层固定选择（不要随机）数据集里面 10 个 neg case 去加上这个 vector，这就是向 pos 方向做 steering，根据层数和 concept 的不同保存输出结果，避免不同运行彼此覆盖，不要做 eval，只做 steering 然后保存 steering 的结果就可以。

### Interview Summary
**Key Discussions**:
- Steering 系数策略：normalize vector (norm=1) + 计算 avg activation + 使用 [0.01, 0.1, 1, 10] × avg_act
- 层处理：逐层单独 steering，遍历所有层
- 输出：JSONL 汇总，每层一个文件（10 samples × 4 multipliers = 40 条）
- 数据集：自动从 steering vector metadata 推断 concept
- 生成参数：temperature=0, max_new_tokens=100

**Research Findings**:
- `HookedModel` 现有 `get_activations()` 方法用于提取 activation
- 需要新增 `generate_with_steering()` 方法支持 steering + 生成
- 现有 hooks 是只读的，需要写 hooks 来修改 activation
- `SteeringVector` 类型包含 `layer_activations: dict[int, Tensor]`

### Metis Review
**Identified Gaps** (addressed):
- **Context Manager Pattern**: 需要用 context manager 管理 hook 生命周期，确保生成后清理 hooks
- **Write Hooks**: 现有 hooks 是只读的，需要修改为支持 in-place 修改 activation
- **SteeringConfig**: 需要新增配置 dataclass 封装 steering 参数

---

## Work Objectives

### Core Objective
创建一个完整的 steering 应用流程：加载 steering vector → 准备 neg samples → 计算系数 → 逐层 steering + 生成 → 保存结果。

### Concrete Deliverables
- `src/steering_geometry/config.py` - 新增 `SteeringConfig` dataclass
- `src/steering_geometry/models.py` - 新增 `generate_with_steering()` 方法
- `src/steering_geometry/apply_steering.py` - 主模块（含 CLI）
- `tests/test_apply_steering.py` - 集成测试

### Definition of Done
- [ ] `uv run python -m steering_geometry.apply_steering --help` 显示帮助
- [ ] `uv run python -m steering_geometry.apply_steering --vector ... --model ...` 成功运行
- [ ] 生成 `data/steered/{concept}/{model}/layer{idx}.jsonl` 文件
- [ ] 每个 JSONL 文件包含 40 行（10 samples × 4 multipliers）
- [ ] `uv run mypy src/` → 0 errors
- [ ] `uv run ruff check src/` → 0 violations
- [ ] `uv run pytest tests/test_apply_steering.py` → all pass

### Must Have
- Normalize steering vector (norm=1)
- 计算每层 avg activation（用同样的 10 个 neg case）
- 4 个 multiplier 值：avg_act × [0.01, 0.1, 1, 10]
- 固定选择 10 个 negative samples（seed=42）
- 逐层单独 steering
- JSONL 输出格式

### Must NOT Have (Guardrails)
- **NO evaluation metrics** - 只保存结果，不做 eval
- **NO random sampling** - 固定 seed=42 选择
- **NO batching of generation** - 逐个样本生成
- **NO multi-concept in single run** - 一次只处理一个 concept
- **NO over-abstraction** - 保持简单直接

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (integration tests)
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.

- **CLI/TUI**: Use Bash — Run command, validate output, check exit code
- **Library/Module**: Use Bash (uv run python) — Import, call functions, verify output

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation):
├── Task 1: Add SteeringConfig dataclass to config.py [quick]
└── Task 2: Add generate_with_steering() to models.py [deep]

Wave 2 (After Wave 1 — core module):
├── Task 3: Create apply_steering.py module [deep]
└── Task 4: Add CLI entry point [quick]

Wave 3 (After Wave 2 — verification):
└── Task 5: Integration tests [quick]

Critical Path: Task 1 → Task 3 → Task 4 → Task 5
Parallel Speedup: Wave 1 (T1, T2 parallel)
```

### Dependency Matrix

- **1**: — — 3
- **2**: — — 3
- **3**: 1, 2 — 4
- **4**: 3 — 5
- **5**: 4 — —

### Agent Dispatch Summary

- **Wave 1**: **2** — T1 → `quick`, T2 → `deep`
- **Wave 2**: **2** — T3 → `deep`, T4 → `quick`
- **Wave 3**: **1** — T5 → `quick`

---

## TODOs

- [ ] 1. Add SteeringConfig dataclass to config.py

  **What to do**:
  - Add `SteeringConfig` dataclass to `src/steering_geometry/config.py`
  - Fields: `multipliers: list[float]`, `num_samples: int`, `seed: int`, `max_new_tokens: int`, `temperature: float`
  - Default values: multipliers=[0.01, 0.1, 1.0, 10.0], num_samples=10, seed=42, max_new_tokens=100, temperature=0.0
  - Export in `__all__`

  **Must NOT do**:
  - DO NOT add evaluation-related fields
  - DO NOT over-complicate with optional fields

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dataclass addition, straightforward task
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/config.py:ExtractionConfig` - Follow this pattern for dataclass structure

  **Acceptance Criteria**:
  - [ ] `SteeringConfig` dataclass exists in config.py
  - [ ] `uv run python -c "from steering_geometry.config import SteeringConfig; print(SteeringConfig())"` works

  **QA Scenarios**:
  ```
  Scenario: Import and instantiate SteeringConfig
    Tool: Bash
    Steps:
      1. Run: uv run python -c "from steering_geometry.config import SteeringConfig; c = SteeringConfig(); print(c.multipliers)"
    Expected Result: Output contains "[0.01, 0.1, 1.0, 10.0]"
    Evidence: .omo/evidence/task-1-import-config.txt
  ```

  **Commit**: NO (groups with other tasks)

- [ ] 2. Add generate_with_steering() method to models.py

  **What to do**:
  - Add `generate_with_steering()` method to `HookedModel` class in `src/steering_geometry/models.py`
  - Method signature: `generate_with_steering(prompt: str, layer_idx: int, steering_vector: Tensor, scale: float, max_new_tokens: int = 100, temperature: float = 0.0) -> str`
  - Implementation:
    1. Register forward hook on specified layer that adds `steering_vector * scale` to activation
    2. Run `model.generate()` with given parameters
    3. Remove hook after generation
    4. Return generated text (decode tokens)
  - Use context manager pattern internally for hook lifecycle

  **Must NOT do**:
  - DO NOT modify existing `get_activations()` method
  - DO NOT add batch generation support

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires understanding of PyTorch hooks and transformers generate API
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/models.py:HookedModel.get_activations` - Follow hook registration pattern
  - `src/steering_geometry/models.py:HookedModel._get_layers_module` - Use this to get layers module

  **Acceptance Criteria**:
  - [ ] `generate_with_steering()` method exists in HookedModel
  - [ ] Method accepts prompt, layer_idx, steering_vector, scale, max_new_tokens, temperature
  - [ ] Returns generated text string

  **QA Scenarios**:
  ```
  Scenario: Generate with steering on tiny model
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
import torch
from steering_geometry.models import HookedModel
from steering_geometry.config import ModelConfig
model = HookedModel(ModelConfig(model_name='sshleifer/tiny-gpt2'))
vector = torch.randn(model.model.config.hidden_size)
result = model.generate_with_steering('Hello', layer_idx=0, steering_vector=vector, scale=0.1, max_new_tokens=10)
print(type(result), len(result) > 0)
"
    Expected Result: Output contains "<class 'str'> True"
    Evidence: .omo/evidence/task-2-generate-steering.txt
  ```

  **Commit**: NO (groups with other tasks)

- [ ] 3. Create apply_steering.py module

  **What to do**:
  - Create `src/steering_geometry/apply_steering.py` module
  - Implement core function: `apply_steering(vector_path: Path, model_name: str, output_dir: Path, config: SteeringConfig) -> None`
  - Implementation steps:
    1. Load steering vector from file: `torch.load(vector_path)["vector"]`
    2. Extract concept from `vector.concept` metadata
    3. Load contrast pairs using `load_contrast_pairs(concept, num_pairs=config.num_samples)`
    4. Extract first 10 negative samples (fixed order, seed=42 in load function already)
    5. Normalize steering vector for each layer: `v / v.norm()`
    6. Compute avg activation for each layer using the 10 neg samples
    7. For each layer in vector.layer_activations:
       - For each sample (10):
         - For each multiplier (4):
           - Call `model.generate_with_steering()` with scale = avg_act * multiplier
           - Append result to list
       - Write all results to JSONL file
  - Output path: `{output_dir}/{concept}/{model_name_safe}/layer{idx}.jsonl`
  - JSONL format: `{"sample_idx": int, "multiplier": float, "prompt": str, "generated_text": str}`

  **Must NOT do**:
  - DO NOT add evaluation logic
  - DO NOT random sample selection
  - DO NOT batch generation

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex orchestration of loading, computing, steering, and saving
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 4
  - **Blocked By**: Task 1, Task 2

  **References**:
  - `src/steering_geometry/extract.py:load_contrast_pairs` - Use to load data
  - `src/steering_geometry/types.py:SteeringVector` - Understand vector structure
  - `src/steering_geometry/extract.py:601-610` - Follow file naming pattern (safe_model_name)

  **Acceptance Criteria**:
  - [ ] `apply_steering()` function exists
  - [ ] Function loads vector, extracts concept, loads data, normalizes, computes avg, generates, saves
  - [ ] Output JSONL files created with correct format

  **QA Scenarios**:
  ```
  Scenario: Run apply_steering module
    Tool: Bash
    Steps:
      1. Run: uv run python -c "
from pathlib import Path
from steering_geometry.apply_steering import apply_steering
from steering_geometry.config import SteeringConfig
# This test assumes vector file exists; skip if not
import os
vector_path = Path('data/vectors/honesty_sshleifer_tiny-gpt2_mean.pt')
if vector_path.exists():
    apply_steering(vector_path, 'sshleifer/tiny-gpt2', Path('data/steered_test/'), SteeringConfig(num_samples=2))
    print('OK')
else:
    print('SKIP: vector file not found')
"
    Expected Result: Output contains "OK" or "SKIP"
    Evidence: .omo/evidence/task-3-apply-steering.txt
  ```

  **Commit**: NO (groups with other tasks)

- [ ] 4. Add CLI entry point to apply_steering.py

  **What to do**:
  - Add CLI entry point in `src/steering_geometry/apply_steering.py`
  - Use argparse with arguments:
    - `--vector` (required): Path to steering vector file
    - `--model` (required): Model name (e.g., "Qwen/Qwen3.5-2B")
    - `--output` (default: "data/steered/"): Output directory
    - `--samples` (default: 10): Number of negative samples
    - `--multipliers` (default: "0.01,0.1,1.0,10.0"): Comma-separated multiplier scale factors
  - Add `if __name__ == "__main__": main()` block
  - Create `main()` function that parses args and calls `apply_steering()`

  **Must NOT do**:
  - DO NOT add --eval or --metrics flags
  - DO NOT add --random-seed flag (always use 42)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard argparse CLI, straightforward
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 3

  **References**:
  - `src/steering_geometry/extract.py:533-571` - Follow argparse pattern from extract.py

  **Acceptance Criteria**:
  - [ ] `uv run python -m steering_geometry.apply_steering --help` shows usage
  - [ ] CLI accepts --vector, --model, --output, --samples, --multipliers

  **QA Scenarios**:
  ```
  Scenario: CLI help works
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.apply_steering --help
    Expected Result: Output contains "--vector" and "--model"
    Evidence: .omo/evidence/task-4-cli-help.txt
  ```

  **Commit**: NO (groups with other tasks)

- [ ] 5. Add integration tests

  **What to do**:
  - Create `tests/test_apply_steering.py`
  - Test cases:
    1. `test_steering_config_defaults()` - Verify SteeringConfig default values
    2. `test_apply_steering_creates_output()` - Run apply_steering with mock/tiny data, verify files created
    3. `test_jsonl_format()` - Verify JSONL row format has required keys
  - Use `tests/conftest.py` fixtures if available for test isolation
  - Mock model loading if tests are slow

  **Must NOT do**:
  - DO NOT test with large models (use tiny-gpt2 or mock)
  - DO NOT add evaluation tests

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard pytest tests
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Task 4)
  - **Blocks**: None
  - **Blocked By**: Task 4

  **References**:
  - `tests/conftest.py` - Use existing fixtures
  - `tests/unit/test_extract.py` - Follow test pattern

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/test_apply_steering.py -v` passes
  - [ ] At least 3 test functions exist

  **QA Scenarios**:
  ```
  Scenario: Tests pass
    Tool: Bash
    Steps:
      1. Run: uv run pytest tests/test_apply_steering.py -v
    Expected Result: All tests pass, exit code 0
    Evidence: .omo/evidence/task-5-tests.txt
  ```

  **Commit**: YES
  - Message: `feat(steering): add apply_steering module for activation steering`
  - Files: `src/steering_geometry/config.py`, `src/steering_geometry/models.py`, `src/steering_geometry/apply_steering.py`, `tests/test_apply_steering.py`
  - Pre-commit: `uv run mypy src/ && uv run ruff check src/ && uv run pytest`

---

## Final Verification Wave (MANDATORY)

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify all Must Have items implemented, all Must NOT Have absent.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run mypy src/`, `uv run ruff check src/`, `uv run pytest`.
  Output: `Type Check [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run CLI with tiny model, verify JSONL output format and content.
  Output: `CLI [PASS/FAIL] | JSONL Format [PASS/FAIL] | Content [PASS/FAIL] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  Verify no scope creep, no over-abstraction, no evaluation code added.
  Output: `Scope [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Single Commit**: After all tasks complete
- Message: `feat(steering): add apply_steering module for activation steering`
- Pre-commit: `uv run mypy src/ && uv run ruff check src/ && uv run pytest`

---

## Success Criteria

### Verification Commands
```bash
# CLI help works
uv run python -m steering_geometry.apply_steering --help

# Run on tiny model
uv run python -m steering_geometry.apply_steering \
    --vector data/vectors/honesty_sshleifer_tiny-gpt2_mean.pt \
    --model sshleifer/tiny-gpt2 \
    --output data/steered/

# Verify JSONL row count (10 samples × 4 multipliers = 40)
wc -l data/steered/honesty/sshleifer_tiny-gpt2/layer*.jsonl

# Verify JSONL format
head -1 data/steered/honesty/sshleifer_tiny-gpt2/layer0.jsonl | python -m json.tool
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] CLI works with tiny model
- [ ] JSONL output format correct
