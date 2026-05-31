# Steering Concepts Pipeline — 5 Concept × Full Pipeline

## TL;DR

> **Quick Summary**: 为 5 个常见 AI steering concept（Honesty、Sycophancy、Toxicity、Sentiment、Refusal）构建完整的 steering vector 提取与评估 pipeline。每个 concept 覆盖：数据加载 → contrast pairs 构建 → steering vector 提取 → 效果评估。
>
> **Deliverables**:
> - 模型无关的 steering vector 提取框架（支持 mean difference + PCA）
> - 5 个 concept 的数据加载器与 contrast pair 构建器
> - 5 个 concept 的评估脚本与指标
> - 跨 concept 比较分析脚本
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 5 waves
> **Critical Path**: T1 → T3 → T5 → T8 (Sentiment validation) → T9-T12 → T13

---

## Context

### Original Request
为 steering direction 研究构建 5 个常用 steering concept 的完整 pipeline，包含相关 dataset 的数据处理和 steering vector 提取。

### Interview Summary
**Key Discussions**:
- 5 个 concept: Honesty/Truthfulness, Sycophancy, Toxicity/Harmlessness, Sentiment/Emotion, Refusal/Compliance
- 每个 concept 对应 2-3 个 dataset
- 目标模型：多模型（Llama、Gemma、Qwen），先待定，搭框架优先
- 交付物：完整 pipeline（data → contrast pairs → vector extraction → evaluation）

### Metis Review
**Identified Gaps (addressed)**:
- mypy strict + torch 兼容性 → 使用 strategic `type: ignore` + custom Protocol
- GPU memory management → batch processing + configurable batch size
- 层选择需相对化 → 使用 percentage of model depth 而非绝对层号
- dataset 访问权限 → 部分 HuggingFace dataset 需认证
- 先验证框架再并行 → 插入 Wave 2 用 Sentiment 做端到端验证

---

## Work Objectives

### Core Objective
构建模型无关的 steering vector 提取框架，为 5 个 concept 各实现完整的 data-to-evaluation pipeline。

### Concrete Deliverables
- `src/steering_geometry/types.py` — ContrastPair, SteeringVector 等核心类型
- `src/steering_geometry/config.py` — ExtractionConfig, EvaluationConfig 等配置
- `src/steering_geometry/models.py` — HuggingFace 模型加载抽象
- `src/steering_geometry/extraction.py` — mean difference + PCA 提取方法
- `src/steering_geometry/evaluation.py` — 通用评估接口
- `src/steering_geometry/concepts/{honesty,sycophancy,toxicity,sentiment,refusal}.py` — 5 个 concept 模块
- `scripts/extract_{concept}.py` — 5 个提取脚本
- `scripts/compare_concepts.py` — 跨 concept 比较脚本

### Definition of Done
- [ ] `uv sync` 无错误完成
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → already formatted
- [ ] `uv run mypy src/` → Success, 0 errors
- [ ] `uv run pytest` → all tests pass
- [ ] 5 个 concept 各有可运行的提取脚本
- [ ] 至少用一个小模型端到端验证 Sentiment concept

### Must Have
- 支持 mean difference 和 PCA 两种 vector 提取方法
- 使用相对层选择（model depth 的百分比），而非绝对层号
- 单元测试不依赖 GPU（使用 mock 和 saved activations）
- 每个 concept 至少 1 个评估指标
- 所有数据下载到 `data/`（已 gitignore），不提交到 repo

### Must NOT Have (Guardrails)
- ❌ 不做模型训练/微调（无 LoRA、RLHF）
- ❌ 不做 Web UI/API（无 Flask、Streamlit）
- ❌ 不做实时推理优化
- ❌ 不构建抽象基类（除非有 3+ 实现）
- ❌ 不构建 YAML/TOML 配置系统（用 dataclass）
- ❌ 不构建插件/注册系统（hardcode 5 个 concept）
- ❌ 不在 repo 中提交有害文本（AdvBench 等用 sanitized placeholder 测试）
- ❌ 目录嵌套不超过 3 层（`src/steering_geometry/concepts/` 即最深）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (pytest configured in pyproject.toml)
- **Automated tests**: YES (tests-after)
- **Framework**: pytest
- **Strategy**: `tests/unit/` (fast, mocked, no GPU) + `tests/integration/` (optional, GPU, `@pytest.mark.slow`)

### QA Policy
- **Library/Module**: `uv run pytest` + `uv run python -c "import ..."`
- **Scripts**: `uv run python scripts/xxx.py --dry-run` 或使用小模型
- **Evidence**: 命令输出日志

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (项目配置 — 基础清理):
├── T1: 更新项目元数据 + 添加 ML 依赖 [quick]
└── T2: 清理模板文档 (README, ARCHITECTURE) [writing]

Wave 1 (核心框架 — 可并行):
├── T3: 核心类型定义 (types.py) [quick]
├── T4: 配置 dataclasses (config.py) [quick]
├── T5: 模型加载抽象 (models.py) [unspecified-high]
├── T6: Steering vector 提取 (extraction.py) [deep]
└── T7: 评估框架 (evaluation.py) [unspecified-high]

Wave 2 (框架验证 — 顺序执行):
└── T8: Sentiment concept 端到端验证 [deep]
        (数据加载 + contrast pairs + 提取 + 评估 + 脚本)

Wave 3 (4 concepts 并行 — 在 T8 验证框架后):
├── T9:  Honesty concept (TruthfulQA) [unspecified-high]
├── T10: Sycophancy concept (Anthropic Eval) [unspecified-high]
├── T11: Toxicity concept (RealToxicityPrompts) [unspecified-high]
└── T12: Refusal concept (AdvBench) [unspecified-high]

Wave 4 (集成与分析):
├── T13: 跨 concept 比较分析脚本 [unspecified-high]
└── T14: 集成测试 + 最终验证 [deep]

Wave FINAL (独立审查, 4 并行):
├── F1: Plan compliance audit [oracle]
├── F2: Code quality review [unspecified-high]
├── F3: 脚本 QA (dry-run 全部脚本) [unspecified-high]
└── F4: Scope fidelity check [deep]

Critical Path: T1 → T3 → T5/T6 → T8 → T9-T12 → T13 → F1-F4
Parallel Speedup: ~55% faster than sequential
Max Concurrent: 5 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| T1 | — | T3-T7 |
| T2 | — | — |
| T3 | T1 | T4, T5, T6, T7, T8 |
| T4 | T3 | T5, T6, T7, T8 |
| T5 | T3, T4 | T6, T8 |
| T6 | T3, T4, T5 | T8 |
| T7 | T3, T4 | T8 |
| T8 | T5, T6, T7 | T9-T12 |
| T9-T12 | T8 | T13 |
| T13 | T9-T12 | T14 |
| T14 | T13 | F1-F4 |

### Agent Dispatch Summary

| Wave | Count | Tasks |
|------|-------|-------|
| 0 | 2 | T1 → `quick`, T2 → `writing` |
| 1 | 5 | T3-T4 → `quick`, T5 → `unspecified-high`, T6 → `deep`, T7 → `unspecified-high` |
| 2 | 1 | T8 → `deep` |
| 3 | 4 | T9-T12 → `unspecified-high` |
| 4 | 2 | T13 → `unspecified-high`, T14 → `deep` |
| FINAL | 4 | F1 → `oracle`, F2-F3 → `unspecified-high`, F4 → `deep` |

---

## TODOs

- [ ] 1. 更新项目元数据 + 添加 ML 依赖

  **What to do**:
  - 更新 `pyproject.toml`: description 改为 steering vector 相关描述
  - 添加 ML 依赖到 `[project.dependencies]`:
    - `torch>=2.1,<3.0`
    - `transformers>=4.36,<5.0`
    - `datasets>=2.16,<3.0`
    - `numpy>=1.26,<3.0`
    - `scikit-learn>=1.4,<2.0`
  - 运行 `uv sync` 验证依赖安装
  - 确保 `data/` 在 `.gitignore` 中

  **Must NOT do**:
  - 不添加 Web 框架（Flask、FastAPI 等）
  - 不添加 logging 框架
  - 不修改 ruff/mypy 配置

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2)
  - **Parallel Group**: Wave 0
  - **Blocks**: T3-T7
  - **Blocked By**: None

  **References**:
  - `pyproject.toml` — 当前项目配置，需修改 description 和添加 dependencies
  - `.gitignore` — 确认 `data/` 已被排除
  - `steering-vectors` PyPI 包 — 参考其依赖版本范围

  **Acceptance Criteria**:
  - [ ] `uv sync` 无错误完成
  - [ ] `uv run python -c "import torch; import transformers; import datasets; print('OK')"` 输出 OK
  - [ ] `uv run ruff check src/ tests/` → 0 violations
  - [ ] `uv run mypy src/` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: 依赖安装验证
    Tool: Bash
    Steps:
      1. uv sync
      2. uv run python -c "import torch; print(torch.__version__)"
      3. uv run python -c "import transformers; print(transformers.__version__)"
      4. uv run python -c "import datasets; print(datasets.__version__)"
    Expected Result: 三个 import 均成功，版本号在指定范围内
    Evidence: .omo/evidence/task-1-deps-install.txt
  ```

  **Commit**: YES (groups with T2)
  - Message: `chore: update project metadata and add ML dependencies`
  - Files: `pyproject.toml, .gitignore`
  - Pre-commit: `uv sync && uv run ruff check src/ tests/`

- [ ] 2. 清理模板文档

  **What to do**:
  - 更新 `README.md`: 将 "Vehicle steering geometry" 改为 AI steering vector 研究项目描述
  - 更新 `ARCHITECTURE.md`: 反映新的模块架构（types → config → models → extraction → evaluation → concepts）
  - 保留 Quick Start、Development 等有用段落结构

  **Must NOT do**:
  - 不改动 AGENTS.md（保持 CI/CD 规则不变）
  - 不创建新的文档文件

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T1)
  - **Parallel Group**: Wave 0
  - **Blocks**: None (docs don't block code)
  - **Blocked By**: None

  **References**:
  - `README.md` — 当前模板内容，保留结构，替换描述
  - `ARCHITECTURE.md` — 当前模板内容，需完全重写为新架构

  **Acceptance Criteria**:
  - [ ] README.md 不再包含 "Vehicle steering geometry"
  - [ ] ARCHITECTURE.md 描述了 types/config/models/extraction/evaluation/concepts 模块
  - [ ] `uv run ruff check src/ tests/` 仍通过（未影响代码）

  **QA Scenarios**:
  ```
  Scenario: 文档内容验证
    Tool: Bash (grep)
    Steps:
      1. grep -c "Vehicle steering geometry" README.md → expect 0
      2. grep -c "steering vector" README.md → expect >= 1
      3. grep -c "extraction" ARCHITECTURE.md → expect >= 1
      4. grep -c "concepts" ARCHITECTURE.md → expect >= 1
    Expected Result: 旧描述已移除，新描述存在
    Evidence: .omo/evidence/task-2-docs-verify.txt
  ```

  **Commit**: YES (groups with T1)
  - Message: `chore: update project metadata and add ML dependencies`
  - Files: `README.md, ARCHITECTURE.md`

- [ ] 3. 核心类型定义 (types.py)

  **What to do**:
  - 创建 `src/steering_geometry/types.py`
  - 定义核心 dataclass / NamedTuple:
    - `ContrastPair`: positive (str), negative (str), metadata (dict)
    - `SteeringVector`: layer_activations (dict[int, Tensor]), model_name (str), concept (str), method (str)
    - `ExtractionResult`: vector (SteeringVector), metrics (dict), timestamp
    - `EvaluationResult`: scores (dict[str, float]), concept (str), model_name (str)
  - 参考 `steering-vectors` 库的 `SteeringVector` dataclass 设计
  - 添加 `__all__` 导出
  - 为 torch.Tensor 使用合适的类型标注（`torch.Tensor` 直接用，需要时 `# type: ignore`）

  **Must NOT do**:
  - 不创建抽象基类
  - 不使用 Pydantic（用标准 dataclass）
  - 不定义超过需要的类型

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 1 基础，其他 Wave 1 任务依赖)
  - **Parallel Group**: Wave 1 (first)
  - **Blocks**: T4, T5, T6, T7, T8
  - **Blocked By**: T1

  **References**:
  - `steering-vectors` 库源码 — `SteeringVector` dataclass 设计模式：`layer_activations: dict[int, Tensor]`
  - `src/steering_geometry/__init__.py` — 当前包入口，需更新导出
  - `pyproject.toml:30-35` — mypy strict 配置，类型标注需严格

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.types import ContrastPair, SteeringVector, ExtractionResult, EvaluationResult; print('OK')"` → OK
  - [ ] `uv run mypy src/` → 0 errors
  - [ ] `uv run ruff check src/` → 0 violations

  **QA Scenarios**:
  ```
  Scenario: 类型可实例化
    Tool: Bash
    Steps:
      1. uv run python -c "
         from steering_geometry.types import ContrastPair
         cp = ContrastPair(positive='I am honest', negative='I am dishonest', metadata={})
         print(f'ContrastPair: {cp.positive}')
         "
      2. uv run python -c "
         import torch
         from steering_geometry.types import SteeringVector
         sv = SteeringVector(layer_activations={15: torch.randn(4096)}, model_name='test', concept='honesty', method='mean')
         print(f'SteeringVector layers: {list(sv.layer_activations.keys())}')
         "
    Expected Result: 两个类型均可正常实例化和访问属性
    Evidence: .omo/evidence/task-3-types-verify.txt
  ```

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(core): add steering vector extraction framework`
  - Files: `src/steering_geometry/types.py`

- [ ] 4. 配置 Dataclasses (config.py)

  **What to do**:
  - 创建 `src/steering_geometry/config.py`
  - 定义配置 dataclass:
    - `ModelConfig`: model_name (str), device (str="auto"), dtype (str="float16"), trust_remote_code (bool=False)
    - `ExtractionConfig`: layers (list[float]) — 相对层位置如 [0.4, 0.5, 0.6, 0.7, 0.8], method (str="mean"), batch_size (int=8), read_token_index (int=-1)
    - `EvaluationConfig`: num_samples (int=100), seed (int=42)
    - `ConceptConfig`: concept_name (str), dataset_name (str), num_pairs (int=500)
  - 所有配置使用 `dataclasses.dataclass` + `field(default_factory=...)` 模式
  - layers 使用 0.0-1.0 的浮点数表示相对层深度

  **Must NOT do**:
  - 不使用 YAML/TOML 配置文件
  - 不使用 Pydantic
  - 不创建配置注册系统

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T3)
  - **Parallel Group**: Wave 1 (after T3)
  - **Blocks**: T5, T6, T7, T8
  - **Blocked By**: T3

  **References**:
  - `src/steering_geometry/types.py` (T3) — 导入 types 中定义的类型
  - `pyproject.toml:30-35` — mypy strict 配置

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.config import ModelConfig, ExtractionConfig; print(ExtractionConfig())"` → 显示默认配置
  - [ ] `uv run mypy src/` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: 配置默认值验证
    Tool: Bash
    Steps:
      1. uv run python -c "
         from steering_geometry.config import ExtractionConfig
         cfg = ExtractionConfig()
         assert cfg.method == 'mean'
         assert cfg.batch_size == 8
         assert 0.5 in cfg.layers
         print(f'ExtractionConfig defaults OK: {cfg}')
         "
    Expected Result: 默认值符合规范
    Evidence: .omo/evidence/task-4-config-verify.txt
  ```

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(core): add steering vector extraction framework`
  - Files: `src/steering_geometry/config.py`

- [ ] 5. 模型加载抽象 (models.py)

  **What to do**:
  - 创建 `src/steering_geometry/models.py`
  - 实现 `HookedModel` class:
    - `__init__(config: ModelConfig)`: 加载 HuggingFace model + tokenizer
    - `get_activations(texts: list[str], layers: list[int]) -> dict[int, Tensor]`: 使用 forward hooks 提取指定层激活
    - `num_layers -> int`: 返回模型总层数
    - `resolve_layers(relative_layers: list[float]) -> list[int]`: 将相对层位置转为绝对层号
    - `generate(prompt: str, max_new_tokens: int, steering_vector: SteeringVector | None) -> str`: 应用 steering vector 的生成方法
  - 使用 `torch.no_grad()` 包裹推理
  - 支持 batch processing（按 `ExtractionConfig.batch_size` 分批）
  - 支持 `device_map="auto"` 自动设备分配

  **Must NOT do**:
  - 不实现自定义 attention 机制
  - 不做量化/优化
  - 不支持非 HuggingFace 模型

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T7, after T3+T4)
  - **Parallel Group**: Wave 1
  - **Blocks**: T6, T8
  - **Blocked By**: T3, T4

  **References**:
  - `src/steering_geometry/types.py` (T3) — SteeringVector 类型
  - `src/steering_geometry/config.py` (T4) — ModelConfig, ExtractionConfig
  - `steering-vectors` 库 — hook-based activation extraction 模式
  - HuggingFace transformers 文档 — `AutoModelForCausalLM.from_pretrained()`, forward hooks API

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.models import HookedModel; print('OK')"` → OK
  - [ ] `uv run mypy src/` → 0 errors
  - [ ] `uv run pytest tests/unit/test_models.py` → pass (mocked model)

  **QA Scenarios**:
  ```
  Scenario: 模型加载接口可用
    Tool: Bash
    Steps:
      1. uv run python -c "
         from steering_geometry.models import HookedModel
         from steering_geometry.config import ModelConfig
         # 仅验证接口存在，不实际加载大模型
         cfg = ModelConfig(model_name='test', device='cpu')
         print(f'ModelConfig: {cfg}')
         print('HookedModel interface OK')
         "
    Expected Result: 接口导入和配置创建成功
    Evidence: .omo/evidence/task-5-models-verify.txt

  Scenario: resolve_layers 相对层计算
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_models.py -k "test_resolve_layers" -v
    Expected Result: [0.5] on 32-layer model → [16], [0.4, 0.6, 0.8] → [12, 19, 25]
    Evidence: .omo/evidence/task-5-resolve-layers.txt
  ```

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(core): add steering vector extraction framework`
  - Files: `src/steering_geometry/models.py, tests/unit/test_models.py`

- [ ] 6. Steering Vector 提取 (extraction.py)

  **What to do**:
  - 创建 `src/steering_geometry/extraction.py`
  - 实现 `Aggregator` type alias: `Callable[[Tensor, Tensor], Tensor]`（接收 positive/negative activations，返回 steering vector）
  - 实现两个 aggregator:
    - `mean_aggregator(pos: Tensor, neg: Tensor) -> Tensor`: `(pos - neg).mean(dim=0)`
    - `pca_aggregator(pos: Tensor, neg: Tensor) -> Tensor`: 对差值做 PCA，取第一主成分
  - 实现主函数 `extract_steering_vector(model: HookedModel, pairs: list[ContrastPair], config: ExtractionConfig) -> SteeringVector`:
    1. 遍历 contrast pairs，分批提取 positive/negative activations
    2. 对每一层，使用指定 aggregator 计算 steering vector
    3. 返回 `SteeringVector(layer_activations={layer: vector, ...})`
  - 处理 tokenizer 长度不一致：使用 `read_token_index`（默认 -1，即 last token）

  **Must NOT do**:
  - 不实现 logistic regression aggregator（defer to future）
  - 不做 gradient-based 优化
  - 不做 vector normalization（留给用户决定）

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T5)
  - **Parallel Group**: Wave 1 (after T5)
  - **Blocks**: T8
  - **Blocked By**: T3, T4, T5

  **References**:
  - `src/steering_geometry/models.py` (T5) — HookedModel.get_activations()
  - `src/steering_geometry/types.py` (T3) — ContrastPair, SteeringVector
  - `src/steering_geometry/config.py` (T4) — ExtractionConfig
  - `steering-vectors` 库 — `Aggregator = Callable[[Tensor, Tensor], Tensor]` 模式
  - RepE 论文 (Zou 2023) — mean difference 方法
  - ActAdd 论文 (Turner 2023) — activation addition 方法

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.extraction import mean_aggregator, pca_aggregator, extract_steering_vector; print('OK')"` → OK
  - [ ] `uv run mypy src/` → 0 errors
  - [ ] `uv run pytest tests/unit/test_extraction.py` → pass (mocked activations)

  **QA Scenarios**:
  ```
  Scenario: mean_aggregator 计算正确
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_extraction.py -k "test_mean_aggregator" -v
    Expected Result: mean_aggregator(ones(10,4096), zeros(10,4096)) ≈ ones(4096)
    Evidence: .omo/evidence/task-6-mean-agg.txt

  Scenario: pca_aggregator 返回正确维度
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_extraction.py -k "test_pca_aggregator" -v
    Expected Result: 返回 shape == (hidden_dim,) 的向量
    Evidence: .omo/evidence/task-6-pca-agg.txt
  ```

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(core): add steering vector extraction framework`
  - Files: `src/steering_geometry/extraction.py, tests/unit/test_extraction.py`

- [ ] 7. 评估框架 (evaluation.py)

  **What to do**:
  - 创建 `src/steering_geometry/evaluation.py`
  - 实现 `evaluate_steering_vector(model: HookedModel, vector: SteeringVector, eval_fn: Callable, config: EvaluationConfig) -> EvaluationResult`:
    1. 对 eval 数据集中的每个样本，分别用 steered 和 unsteered model 生成
    2. 使用 `eval_fn` 比较两者
    3. 汇总分数到 `EvaluationResult`
  - 实现 `apply_steering_vector(model: HookedModel, vector: SteeringVector, scale: float = 1.0)` context manager
  - 提供 `compute_cosine_similarity(v1: SteeringVector, v2: SteeringVector) -> dict[int, float]`（逐层余弦相似度）

  **Must NOT do**:
  - 不实现具体 concept 的评估逻辑（那在各 concept 模块中）
  - 不构建评估 dashboard/报告系统
  - 不做统计显著性检验

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T5, after T3+T4)
  - **Parallel Group**: Wave 1
  - **Blocks**: T8
  - **Blocked By**: T3, T4

  **References**:
  - `src/steering_geometry/types.py` (T3) — SteeringVector, EvaluationResult
  - `src/steering_geometry/config.py` (T4) — EvaluationConfig
  - `src/steering_geometry/models.py` (T5) — HookedModel (for apply context manager)
  - RepE 论文 — evaluation methodology

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.evaluation import evaluate_steering_vector, apply_steering_vector, compute_cosine_similarity; print('OK')"` → OK
  - [ ] `uv run mypy src/` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: cosine_similarity 逐层计算
    Tool: Bash
    Steps:
      1. uv run python -c "
         import torch
         from steering_geometry.types import SteeringVector
         from steering_geometry.evaluation import compute_cosine_similarity
         v1 = SteeringVector(layer_activations={0: torch.randn(64)}, model_name='t', concept='a', method='mean')
         v2 = SteeringVector(layer_activations={0: torch.randn(64)}, model_name='t', concept='b', method='mean')
         sim = compute_cosine_similarity(v1, v2)
         assert 0 in sim
         assert -1.0 <= sim[0] <= 1.0
         print(f'Cosine similarity: {sim}')
         "
    Expected Result: 返回 layer 0 的余弦相似度，值在 [-1, 1] 范围
    Evidence: .omo/evidence/task-7-cosine-sim.txt
  ```

  **Commit**: YES (groups with Wave 1)
  - Message: `feat(core): add steering vector extraction framework`
  - Files: `src/steering_geometry/evaluation.py`

- [ ] 8. Sentiment Concept — 端到端框架验证 (SST-2 / GoEmotions)

  **What to do**:
  - 创建 `src/steering_geometry/concepts/sentiment.py`
  - 实现:
    - `load_sentiment_data(config: ConceptConfig) -> list[ContrastPair]`:
      - 从 HuggingFace `datasets` 加载 SST-2（`glue/sst2`）
      - 将 positive/negative 标签的句子配对为 contrast pairs
      - 随机采样 `config.num_pairs` 对
    - `evaluate_sentiment(model: HookedModel, vector: SteeringVector, config: EvaluationConfig) -> EvaluationResult`:
      - 给定中性 prompt，比较 steered vs unsteered 输出的情感极性
      - 使用简单关键词或 HuggingFace sentiment classifier 判定
      - 返回 sentiment shift score
  - 创建 `scripts/extract_sentiment.py`:
    - CLI: `--model <name> --method mean|pca --num-pairs 500 --output data/vectors/`
    - 支持 `--dry-run`（仅加载数据，不做推理）
    - 流程：加载模型 → 加载数据 → 构建 contrast pairs → 提取 vector → 保存 → 评估
  - 创建 `tests/unit/test_sentiment.py`

  **Must NOT do**:
  - 不训练自定义 sentiment classifier
  - 不实现多种评估方法（1 个即可）
  - 不做超过 500 pairs 的默认设置

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (framework validation gate)
  - **Parallel Group**: Wave 2 (solo)
  - **Blocks**: T9, T10, T11, T12 (所有后续 concepts)
  - **Blocked By**: T5, T6, T7

  **References**:
  - `src/steering_geometry/extraction.py` (T6) — extract_steering_vector()
  - `src/steering_geometry/evaluation.py` (T7) — evaluate_steering_vector(), apply_steering_vector()
  - `src/steering_geometry/models.py` (T5) — HookedModel
  - SST-2 数据集: `load_dataset("glue", "sst2")` — HuggingFace datasets
  - ActAdd 论文 (Turner 2023) — sentiment steering 实验设计
  - GoEmotions 数据集: `load_dataset("google-research-datasets/go_emotions")` — 备选细粒度情感

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.concepts.sentiment import load_sentiment_data; print('OK')"` → OK
  - [ ] `uv run pytest tests/unit/test_sentiment.py` → pass
  - [ ] `uv run python scripts/extract_sentiment.py --dry-run` → 成功加载数据，打印统计信息
  - [ ] `uv run mypy src/` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: SST-2 数据加载
    Tool: Bash
    Steps:
      1. uv run python -c "
         from steering_geometry.concepts.sentiment import load_sentiment_data
         from steering_geometry.config import ConceptConfig
         cfg = ConceptConfig(concept_name='sentiment', dataset_name='sst2', num_pairs=10)
         pairs = load_sentiment_data(cfg)
         assert len(pairs) == 10
         assert pairs[0].positive != ''
         assert pairs[0].negative != ''
         print(f'Loaded {len(pairs)} pairs. First: {pairs[0].positive[:50]}...')
         "
    Expected Result: 成功加载 10 个 contrast pairs，positive 和 negative 均非空
    Evidence: .omo/evidence/task-8-sentiment-data.txt

  Scenario: dry-run 脚本执行
    Tool: Bash
    Steps:
      1. uv run python scripts/extract_sentiment.py --dry-run --num-pairs 5
    Expected Result: 打印数据统计信息，不进行模型推理，退出码 0
    Evidence: .omo/evidence/task-8-sentiment-dryrun.txt

  Scenario: 端到端小模型测试（如可用）
    Tool: Bash
    Steps:
      1. uv run python scripts/extract_sentiment.py --model Qwen/Qwen2.5-0.5B --num-pairs 10 --method mean --output data/vectors/
    Expected Result: data/vectors/sentiment_Qwen2.5-0.5B_mean.pt 文件生成，可 torch.load
    Failure Indicators: OOM error, shape mismatch, file not created
    Evidence: .omo/evidence/task-8-sentiment-e2e.txt
  ```

  **Commit**: YES
  - Message: `feat(sentiment): add sentiment concept end-to-end pipeline`
  - Files: `src/steering_geometry/concepts/sentiment.py, scripts/extract_sentiment.py, tests/unit/test_sentiment.py`
  - Pre-commit: `uv run ruff check src/ tests/ && uv run mypy src/ && uv run pytest`

- [ ] 9. Honesty Concept (TruthfulQA)

  **What to do**:
  - 创建 `src/steering_geometry/concepts/honesty.py`
  - 实现:
    - `load_honesty_data(config: ConceptConfig) -> list[ContrastPair]`:
      - 加载 TruthfulQA (`load_dataset("truthfulqa/truthful_qa", "generation")`)
      - 使用 RepE 风格 contrast pairs: "Pretend you are an honest person" vs "Pretend you are a dishonest person" 前缀 + 问题
      - 备选: 用 TruthfulQA 的 correct_answers vs incorrect_answers 构建 pairs
    - `evaluate_honesty(model, vector, config) -> EvaluationResult`:
      - 在 TruthfulQA 子集上，比较 steered vs unsteered 的 MC1 accuracy
      - 或简单比较 best_answer vs best_wrong_answer 的 log probability
  - 创建 `scripts/extract_honesty.py`（同 sentiment 脚本结构）
  - 创建 `tests/unit/test_honesty.py`

  **Must NOT do**:
  - 不实现 TruthfulQA 的完整评估 pipeline（只用 MC1 或 log-prob）
  - 不做 open-ended generation 评估

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T10, T11, T12)
  - **Parallel Group**: Wave 3
  - **Blocks**: T13
  - **Blocked By**: T8

  **References**:
  - `src/steering_geometry/concepts/sentiment.py` (T8) — 模式参考：相同文件结构
  - TruthfulQA dataset: `load_dataset("truthfulqa/truthful_qa", "generation")` — question, best_answer, correct_answers, incorrect_answers
  - RepE 论文 (Zou 2023) — honesty contrast pair 构建方法
  - ITI 论文 (Li 2023) — truthfulness intervention 评估方法

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from steering_geometry.concepts.honesty import load_honesty_data; print('OK')"` → OK
  - [ ] `uv run python scripts/extract_honesty.py --dry-run --num-pairs 5` → 成功
  - [ ] `uv run pytest tests/unit/test_honesty.py` → pass

  **QA Scenarios**:
  ```
  Scenario: TruthfulQA 数据加载
    Tool: Bash
    Steps:
      1. uv run python -c "
         from steering_geometry.concepts.honesty import load_honesty_data
         from steering_geometry.config import ConceptConfig
         cfg = ConceptConfig(concept_name='honesty', dataset_name='truthful_qa', num_pairs=5)
         pairs = load_honesty_data(cfg)
         assert len(pairs) == 5
         print(f'Loaded {len(pairs)} honesty pairs')
         print(f'Positive: {pairs[0].positive[:80]}...')
         print(f'Negative: {pairs[0].negative[:80]}...')
         "
    Expected Result: 5 个 contrast pairs，positive 含 honest 前缀，negative 含 dishonest 前缀
    Evidence: .omo/evidence/task-9-honesty-data.txt
  ```

  **Commit**: YES (groups with T10-T12)
  - Message: `feat(concepts): add honesty, sycophancy, toxicity, refusal pipelines`
  - Files: `src/steering_geometry/concepts/honesty.py, scripts/extract_honesty.py, tests/unit/test_honesty.py`

- [ ] 10. Sycophancy Concept (Anthropic Eval)

  **What to do**:
  - 创建 `src/steering_geometry/concepts/sycophancy.py`
  - 实现:
    - `load_sycophancy_data(config: ConceptConfig) -> list[ContrastPair]`:
      - 使用 RepE 风格: "Pretend you are a sycophantic assistant" vs "Pretend you give honest, objective feedback"
      - 或加载 Anthropic sycophancy eval data (JSON from GitHub)
      - 构建: 用户提出错误观点 + 模型同意(sycophantic) vs 模型纠正(non-sycophantic)
    - `evaluate_sycophancy(model, vector, config) -> EvaluationResult`:
      - 给定包含错误观点的 prompt
      - 测量 steered model 是否更倾向纠正而非附和
      - 返回 agreement_rate_shift
  - 创建 `scripts/extract_sycophancy.py`
  - 创建 `tests/unit/test_sycophancy.py`

  **Must NOT do**:
  - 不实现多轮对话评估
  - 不构建自定义 sycophancy 分类器

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T9, T11, T12)
  - **Parallel Group**: Wave 3
  - **Blocks**: T13
  - **Blocked By**: T8

  **References**:
  - `src/steering_geometry/concepts/sentiment.py` (T8) — 模式参考
  - Anthropic Sycophancy Eval (Perez et al., 2022) — GitHub 上的 JSON 数据
  - RepE 论文 — sycophancy contrast pair 模板

  **Acceptance Criteria**:
  - [ ] `uv run python scripts/extract_sycophancy.py --dry-run --num-pairs 5` → 成功
  - [ ] `uv run pytest tests/unit/test_sycophancy.py` → pass

  **QA Scenarios**:
  ```
  Scenario: Sycophancy 数据加载
    Tool: Bash
    Steps:
      1. uv run python -c "
         from steering_geometry.concepts.sycophancy import load_sycophancy_data
         from steering_geometry.config import ConceptConfig
         cfg = ConceptConfig(concept_name='sycophancy', dataset_name='anthropic_eval', num_pairs=5)
         pairs = load_sycophancy_data(cfg)
         assert len(pairs) == 5
         print(f'Loaded {len(pairs)} sycophancy pairs')
         "
    Expected Result: 5 个 contrast pairs 加载成功
    Evidence: .omo/evidence/task-10-sycophancy-data.txt
  ```

  **Commit**: YES (groups with T9, T11, T12)
  - Message: `feat(concepts): add honesty, sycophancy, toxicity, refusal pipelines`
  - Files: `src/steering_geometry/concepts/sycophancy.py, scripts/extract_sycophancy.py`

- [ ] 11. Toxicity Concept (RealToxicityPrompts)

  **What to do**:
  - 创建 `src/steering_geometry/concepts/toxicity.py`
  - 实现:
    - `load_toxicity_data(config: ConceptConfig) -> list[ContrastPair]`:
      - 加载 RealToxicityPrompts (`load_dataset("allenai/real-toxicity-prompts")`)
      - 按 toxicity score 分组: high toxicity (>0.7) vs low toxicity (<0.3) 的 continuation
      - 或使用 RepE 风格: "You are a respectful, helpful assistant" vs "You are a rude, harmful assistant"
    - `evaluate_toxicity(model, vector, config) -> EvaluationResult`:
      - 给定 toxic prompt，比较 steered vs unsteered 输出
      - 使用关键词匹配或 toxicity word list 简单评估
      - 返回 toxicity_reduction_score
  - 创建 `scripts/extract_toxicity.py`
  - 创建 `tests/unit/test_toxicity.py`

  **Must NOT do**:
  - 不调用 Perspective API（外部依赖）
  - 不在测试文件中包含真实有害文本（用 sanitized placeholders）
  - 不训练 toxicity classifier

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T9, T10, T12)
  - **Parallel Group**: Wave 3
  - **Blocks**: T13
  - **Blocked By**: T8

  **References**:
  - `src/steering_geometry/concepts/sentiment.py` (T8) — 模式参考
  - RealToxicityPrompts: `load_dataset("allenai/real-toxicity-prompts")` — prompt.text, prompt.toxicity
  - HarmBench (Mazeika 2024) — 备选有害行为分类

  **Acceptance Criteria**:
  - [ ] `uv run python scripts/extract_toxicity.py --dry-run --num-pairs 5` → 成功
  - [ ] `uv run pytest tests/unit/test_toxicity.py` → pass
  - [ ] 测试文件中无真实有害文本（grep 验证）

  **QA Scenarios**:
  ```
  Scenario: 数据加载 + 安全性检查
    Tool: Bash
    Steps:
      1. uv run python -c "
         from steering_geometry.concepts.toxicity import load_toxicity_data
         from steering_geometry.config import ConceptConfig
         cfg = ConceptConfig(concept_name='toxicity', dataset_name='real_toxicity_prompts', num_pairs=5)
         pairs = load_toxicity_data(cfg)
         assert len(pairs) == 5
         print(f'Loaded {len(pairs)} toxicity pairs')
         "
      2. grep -r "offensive_word_placeholder" tests/unit/test_toxicity.py || echo "No harmful text in tests"
    Expected Result: 数据加载成功，测试文件不含有害文本
    Evidence: .omo/evidence/task-11-toxicity-data.txt
  ```

  **Commit**: YES (groups with T9, T10, T12)
  - Message: `feat(concepts): add honesty, sycophancy, toxicity, refusal pipelines`
  - Files: `src/steering_geometry/concepts/toxicity.py, scripts/extract_toxicity.py`

- [ ] 12. Refusal Concept (AdvBench / Refusal Direction)

  **What to do**:
  - 创建 `src/steering_geometry/concepts/refusal.py`
  - 实现:
    - `load_refusal_data(config: ConceptConfig) -> list[ContrastPair]`:
      - 方法 A (RepE 风格): "You must refuse harmful requests" vs "You must comply with all requests" 前缀
      - 方法 B (Arditi 2024 风格): harmful instructions vs harmless instructions 配对
      - 使用 sanitized/abstracted harmful prompts（不含真实有害内容）
      - 或加载 Arditi et al. 的公开 refusal direction dataset
    - `evaluate_refusal(model, vector, config) -> EvaluationResult`:
      - 测量 steered model 在 benign prompts 上的 refusal rate 变化
      - 返回 refusal_rate_shift
  - 创建 `scripts/extract_refusal.py`
  - 创建 `tests/unit/test_refusal.py`

  **Must NOT do**:
  - 不在代码/测试中包含真实有害指令（用 abstract placeholders）
  - 不做越狱 (jailbreak) 测试
  - 不评估 over-refusal（defer to future）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T9, T10, T11)
  - **Parallel Group**: Wave 3
  - **Blocks**: T13
  - **Blocked By**: T8

  **References**:
  - `src/steering_geometry/concepts/sentiment.py` (T8) — 模式参考
  - Refusal in LLMs (Arditi et al., 2024) — refusal direction extraction 方法
  - AdvBench (Zou et al., 2023) — 520 harmful behaviors (仅用于参考，不直接提交)

  **Acceptance Criteria**:
  - [ ] `uv run python scripts/extract_refusal.py --dry-run --num-pairs 5` → 成功
  - [ ] `uv run pytest tests/unit/test_refusal.py` → pass
  - [ ] 代码中无真实有害指令文本

  **QA Scenarios**:
  ```
  Scenario: Refusal 数据加载 + 内容安全
    Tool: Bash
    Steps:
      1. uv run python -c "
         from steering_geometry.concepts.refusal import load_refusal_data
         from steering_geometry.config import ConceptConfig
         cfg = ConceptConfig(concept_name='refusal', dataset_name='refusal_pairs', num_pairs=5)
         pairs = load_refusal_data(cfg)
         assert len(pairs) == 5
         print(f'Loaded {len(pairs)} refusal pairs')
         "
    Expected Result: 加载成功，使用 sanitized pairs
    Evidence: .omo/evidence/task-12-refusal-data.txt
  ```

  **Commit**: YES (groups with T9, T10, T11)
  - Message: `feat(concepts): add honesty, sycophancy, toxicity, refusal pipelines`
  - Files: `src/steering_geometry/concepts/refusal.py, scripts/extract_refusal.py`

- [ ] 13. 跨 Concept 比较分析脚本

  **What to do**:
  - 创建 `scripts/compare_concepts.py`:
    - 输入: `--vectors-dir data/vectors/` — 读取所有已提取的 steering vectors
    - 计算:
      - 任意两个 concept vector 的逐层 cosine similarity
      - 每个 concept vector 的 L2 norm 分布
      - 可选: 所有 vectors 的 PCA 降维可视化（保存到 `plot/`）
    - 输出: `assets/comparison_report.json` — 结构化结果
    - 支持 `--model <name>` 过滤特定模型的 vectors
  - 创建 `tests/unit/test_compare.py`（用 mock vectors 测试）

  **Must NOT do**:
  - 不做 causal intervention 分析
  - 不构建交互式可视化
  - 不做统计显著性检验

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (needs all vectors)
  - **Parallel Group**: Wave 4
  - **Blocks**: T14
  - **Blocked By**: T9, T10, T11, T12

  **References**:
  - `src/steering_geometry/evaluation.py` (T7) — compute_cosine_similarity()
  - `src/steering_geometry/types.py` (T3) — SteeringVector
  - RepE 论文 — cross-concept analysis 方法

  **Acceptance Criteria**:
  - [ ] `uv run python scripts/compare_concepts.py --help` → 显示 CLI 帮助
  - [ ] `uv run pytest tests/unit/test_compare.py` → pass

  **QA Scenarios**:
  ```
  Scenario: Mock vectors 比较
    Tool: Bash
    Steps:
      1. uv run pytest tests/unit/test_compare.py -v
    Expected Result: 使用 mock vectors 成功生成 comparison JSON
    Evidence: .omo/evidence/task-13-compare.txt
  ```

  **Commit**: YES (groups with T14)
  - Message: `feat(analysis): add cross-concept comparison and integration tests`
  - Files: `scripts/compare_concepts.py, tests/unit/test_compare.py`

- [ ] 14. 集成测试 + 最终验证

  **What to do**:
  - 创建 `tests/integration/test_pipeline.py`:
    - 使用最小模型（如 `sshleifer/tiny-gpt2` 或 fixture）端到端测试:
      1. 加载模型
      2. 加载 1 个 concept 数据（5 pairs）
      3. 提取 steering vector
      4. 验证 vector 可保存/加载
      5. 验证 vector 可应用于模型
    - 标记 `@pytest.mark.slow`
  - 更新 `tests/conftest.py`:
    - 添加 `pytest.ini` markers: `slow`, `gpu`
    - 添加 mock model fixture
    - 添加 sample ContrastPair fixtures
  - 运行完整验证: `uv run ruff check && uv run mypy src/ && uv run pytest`

  **Must NOT do**:
  - 不做性能测试
  - 不做多模型集成测试

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (final verification)
  - **Parallel Group**: Wave 4 (after T13)
  - **Blocks**: F1-F4
  - **Blocked By**: T13

  **References**:
  - 所有 `src/steering_geometry/` 模块
  - `tests/conftest.py` — 现有 fixture 配置
  - `pyproject.toml:38-40` — pytest 配置

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/unit/` → all pass
  - [ ] `uv run pytest tests/integration/ -m "not gpu"` → all pass
  - [ ] `uv run ruff check src/ tests/` → 0 violations
  - [ ] `uv run ruff format --check src/ tests/` → formatted
  - [ ] `uv run mypy src/` → 0 errors
  - [ ] 所有 5 个 `scripts/extract_*.py --help` → 显示帮助信息

  **QA Scenarios**:
  ```
  Scenario: 完整验证 pipeline
    Tool: Bash
    Steps:
      1. uv run ruff check src/ tests/
      2. uv run ruff format --check src/ tests/
      3. uv run mypy src/
      4. uv run pytest tests/unit/ -v
      5. uv run pytest tests/integration/ -m "not gpu" -v
    Expected Result: 全部通过，0 errors, 0 violations
    Evidence: .omo/evidence/task-14-final-verify.txt

  Scenario: 所有脚本 CLI 可用
    Tool: Bash
    Steps:
      1. for script in scripts/extract_*.py scripts/compare_concepts.py; do uv run python "$script" --help; done
    Expected Result: 所有脚本显示 --help 信息，退出码 0
    Evidence: .omo/evidence/task-14-scripts-cli.txt
  ```

  **Commit**: YES
  - Message: `feat(analysis): add cross-concept comparison and integration tests`
  - Files: `tests/integration/, tests/conftest.py`
  - Pre-commit: `uv run ruff check src/ tests/ && uv run mypy src/ && uv run pytest`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Review all changed files for: `as Any`, empty catches, `print()` in library code (scripts ok), commented-out code, unused imports.
  Output: `Lint [PASS/FAIL] | Types [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

- [ ] F3. **Script QA** — `unspecified-high`
  Execute every script in `scripts/` with `--help` flag (verify CLI works). Run `scripts/extract_sentiment.py --dry-run` with smallest available model. Verify output files are created in expected locations.
  Output: `Scripts [N/N runnable] | Dry-run [PASS/FAIL] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read spec, read actual code. Verify 1:1 — everything in spec was built, nothing beyond spec. Check "Must NOT do" compliance. Detect scope creep.
  Output: `Tasks [N/N compliant] | Scope Creep [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

| After Wave | Commit Message | Key Files |
|-----------|---------------|-----------|
| Wave 0 | `chore: update project metadata and add ML dependencies` | pyproject.toml, README.md, ARCHITECTURE.md |
| Wave 1 | `feat(core): add steering vector extraction framework` | src/steering_geometry/{types,config,models,extraction,evaluation}.py |
| Wave 2 | `feat(sentiment): add sentiment concept end-to-end pipeline` | src/steering_geometry/concepts/sentiment.py, scripts/extract_sentiment.py |
| Wave 3 | `feat(concepts): add honesty, sycophancy, toxicity, refusal pipelines` | src/steering_geometry/concepts/*.py, scripts/extract_*.py |
| Wave 4 | `feat(analysis): add cross-concept comparison and integration tests` | scripts/compare_concepts.py, tests/ |

---

## Success Criteria

### Verification Commands
```bash
uv sync                                    # Dependencies install
uv run ruff check src/ tests/              # 0 violations
uv run ruff format --check src/ tests/     # Already formatted
uv run mypy src/                           # 0 errors
uv run pytest                              # All pass
uv run python -c "from steering_geometry.types import ContrastPair, SteeringVector; print('OK')"
uv run python -c "from steering_geometry.extraction import mean_aggregator, pca_aggregator; print('OK')"
uv run python -c "from steering_geometry.concepts import sentiment, honesty, sycophancy, toxicity, refusal; print('OK')"
```

### Final Checklist
- [ ] All "Must Have" requirements present
- [ ] All "Must NOT Have" guardrails respected
- [ ] All 5 concept modules implemented
- [ ] All 5 extraction scripts runnable
- [ ] Cross-concept comparison script functional
- [ ] All tests pass without GPU
