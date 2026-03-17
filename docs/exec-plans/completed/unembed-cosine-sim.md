# Unembedding Matrix Cosine Similarity Analysis

## TL;DR

> **Quick Summary**: 分析 steering vectors 与 unembedding matrix 的关系，计算每个 concept/method/layer 的 top-5 cosine similarity 对应的 tokens，输出 JSON 和可视化图表。
>
> **Deliverables**:
> - 新增 `unembed_analysis.py` 模块
> - 扩展 `HookedModel` 添加 unembedding 访问
> - CLI 脚本支持批量分析
> - JSON 结果文件 (5 concepts × 2 methods × 10 layers = 100 个结果)
> - 可视化图表 (热力图 + 条形图)
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: HookedModel extension → Analysis function → CLI → Visualization

---

## Context

### Original Request
用户想要分析 steering vectors 与 unembedding matrix 的关系：
- 使用 discriminative dim 和 difference in means 两种方法
- 计算每一层的 steering vector
- 计算 vector 与 unembedding matrix 的 top-5 cosine similarity
- 输出对应的 tokens

### Interview Summary
**Key Discussions**:
- Concepts: 全部 5 个 (honesty, toxicity, sentiment, sycophancy, refusal)
- Model: Qwen/Qwen3-1.7B
- Methods: discriminative (K=30) + diff_means (n=1000)
- Layers: 10 层均匀分布 (0.1-1.0)
- Output: JSON + 可视化（热力图+条形图）

**Research Findings**:
- 现有 `vector_analysis.py` 有 cosine similarity 计算模式
- `HookedModel` 需要扩展以访问 unembedding matrix
- Qwen 模型的 unembedding 通常在 `model.lm_head.weight`

### Metis Review
**Identified Gaps** (addressed):
- Unembedding access: 需要添加到 `HookedModel`
- Special token filtering: 排除 BOS/EOS/PAD
- Output format: Text only (decoded tokens)

---

## Work Objectives

### Core Objective
构建一个分析工具，计算 steering vectors 与 vocabulary embeddings 的 cosine similarity，找出每个 vector 最对应的语义方向。

### Concrete Deliverables
- `src/steering_geometry/unembed_analysis.py` - 核心分析模块
- `src/steering_geometry/models.py` - 扩展 `HookedModel.get_unembedding_matrix()`
- `scripts/run_unembed_analysis.sh` - 批量分析脚本
- `outputs/unembed_analysis/json/*.json` - 结果文件
- `outputs/unembed_analysis/plots/*.pdf` - 可视化图表

### Definition of Done
- [ ] `uv run pytest tests/test_unembed_analysis.py` → 所有测试通过
- [ ] `uv run mypy src/` → 0 errors
- [ ] 运行分析生成 100 个结果 (5 concepts × 2 methods × 10 layers)
- [ ] 生成可视化图表 (热力图 + 条形图)

### Must Have
- 支持 5 个 concepts + 2 种方法 + 10 层
- 排除特殊 tokens
- 输出 decoded text (不是 token IDs)
- JSON 格式结果
- 热力图和条形图可视化

### Must NOT Have (Guardrails)
- 不要修改现有 `extract.py` 的逻辑
- 不要添加新的 concept loaders
- 不要创建交互式 dashboard
- 避免在循环中重复加载模型

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (TDD)
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python Module**: Use Bash (uv run pytest) — Run tests, check coverage
- **CLI Scripts**: Use Bash — Execute script, verify output files exist

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - can start immediately):
├── Task 1: Add get_unembedding_matrix() to HookedModel [quick]
├── Task 2: Define types for unembed analysis [quick]
└── Task 3: Write test fixtures for unembed analysis [quick]

Wave 2 (Core Implementation - depends on Wave 1):
├── Task 4: Implement compute_topk_similar_tokens() [quick]
├── Task 5: Implement analyze_steering_vector() [quick]
├── Task 6: Implement run_unembed_experiment() [deep]
└── Task 7: Write unit tests for core functions [quick]

Wave 3 (CLI + Visualization - depends on Wave 2):
├── Task 8: Add CLI interface to unembed_analysis.py [quick]
├── Task 9: Implement JSON output formatting [quick]
├── Task 10: Implement heatmap visualization [visual-engineering]
└── Task 11: Implement bar chart visualization [visual-engineering]

Wave 4 (Integration - depends on Wave 3):
├── Task 12: Create batch analysis script [quick]
├── Task 13: Run full analysis (5 concepts × 2 methods × 10 layers) [deep]
├── Task 14: Verify output integrity [quick]
└── Task 15: Generate final report [writing]
```

### Dependency Matrix

- **1-3**: — — 4-7
- **4-7**: 1, 2, 3 — 8-11
- **8-11**: 4-7 — 12-15
- **12**: 8 — 13
- **13**: 8, 9, 10, 11, 12 — 14
- **14**: 13 — 15

### Agent Dispatch Summary

- **Wave 1**: 3 tasks → `quick` × 3
- **Wave 2**: 4 tasks → `quick` × 3, `deep` × 1
- **Wave 3**: 4 tasks → `quick` × 2, `visual-engineering` × 2
- **Wave 4**: 4 tasks → `quick` × 2, `deep` × 1, `writing` × 1

---

## TODOs

- [x] 1. Add `get_unembedding_matrix()` to `HookedModel`

  **What to do**:
  - 在 `models.py` 的 `HookedModel` 类中添加 `get_unembedding_matrix()` 方法
  - 返回模型的 unembedding matrix (通常是 `lm_head.weight` 或 `embed_out.weight`)
  - 处理不同模型架构的兼容性 (Qwen, Gemma 等)
  - 添加 `get_special_token_ids()` 方法返回特殊 token IDs

  **Must NOT do**:
  - 不要修改现有的 `get_activations()` 或 `generate_with_steering()` 方法
  - 不要在 unembedding matrix 上做任何预处理

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单个方法添加，逻辑简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/models.py:20-100` - HookedModel 类结构
  - `src/steering_geometry/models.py:90-99` - `_get_layers_module()` 模式参考

  **Acceptance Criteria**:
  - [ ] `get_unembedding_matrix()` 返回 shape 为 (vocab_size, hidden_dim) 的 Tensor
  - [ ] `get_special_token_ids()` 返回 set[int] 包含 BOS/EOS/PAD token IDs
  - [ ] `uv run mypy src/steering_geometry/models.py` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: Verify unembedding matrix shape
    Tool: Bash (uv run python -c "...")
    Steps:
      1. Load HookedModel with Qwen/Qwen3-1.7B
      2. Call get_unembedding_matrix()
      3. Assert shape[0] == 151936 (Qwen vocab size)
      4. Assert shape[1] == 2048 (hidden dim)
    Expected Result: Shape matches (151936, 2048)
    Evidence: .sisyphus/evidence/task-1-unembed-shape.txt
  ```

  **Commit**: YES (with Task 2, 3)
  - Message: `feat(models): add unembedding matrix access methods`

- [x] 2. Define types for unembed analysis

  **What to do**:
  - 在 `types.py` 中添加新的数据类型：
    - `UnembedAnalysisResult`: 单个 vector 的分析结果 (layer, method, top5 tokens, similarities)
    - `ConceptAnalysisResult`: 整个 concept 的分析结果 (多个 layers 的结果)
  - 使用 dataclass 或 TypedDict

  **Must NOT do**:
  - 不要修改现有的 `SteeringVector` 或 `ContrastPair` 类型

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `src/steering_geometry/types.py:95-110` - SteeringVector dataclass 模式
  - `src/steering_geometry/types.py` - 现有类型定义风格

  **Acceptance Criteria**:
  - [ ] `UnembedAnalysisResult` 包含 layer, method, tokens, similarities 字段
  - [ ] `ConceptAnalysisResult` 包含 concept, model, results 字段
  - [ ] `uv run mypy src/steering_geometry/types.py` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: Verify type structure
    Tool: Bash (uv run python -c "...")
    Steps:
      1. Import UnembedAnalysisResult
      2. Create instance with test data
      3. Access all fields
    Expected Result: No TypeError or AttributeError
    Evidence: .sisyphus/evidence/task-2-types.txt
  ```

  **Commit**: YES (with Task 1, 3)

- [x] 3. Write test fixtures for unembed analysis

  **What to do**:
  - 在 `tests/conftest.py` 中添加测试 fixtures：
    - `mock_unembedding_matrix`: 小型模拟 unembedding matrix (100 tokens × 64 dim)
    - `sample_steering_vector`: 模拟 steering vector
    - `expected_top5_tokens`: 预期的 top-5 结果
  - 在 `tests/test_unembed_analysis.py` 中创建空测试文件结构

  **Must NOT do**:
  - 不要在 fixtures 中使用真实模型加载

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 7
  - **Blocked By**: None

  **References**:
  - `tests/conftest.py` - 现有 fixtures 结构
  - `src/steering_geometry/types.py:95` - SteeringVector 结构

  **Acceptance Criteria**:
  - [ ] `mock_unembedding_matrix` fixture 返回 (100, 64) Tensor
  - [ ] `sample_steering_vector` fixture 返回 SteeringVector 实例
  - [ ] `tests/test_unembed_analysis.py` 文件已创建

  **QA Scenarios**:
  ```
  Scenario: Verify fixtures work
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/test_unembed_analysis.py -v`
    Expected Result: Empty test file passes (0 tests collected is OK)
    Evidence: .sisyphus/evidence/task-3-fixtures.txt
  ```

  **Commit**: YES (with Task 1, 2)

- [x] 4. Implement `compute_topk_similar_tokens()`

  **What to do**:
  - 在 `unembed_analysis.py` 中实现核心函数：
    ```python
    def compute_topk_similar_tokens(
        vector: Tensor,
        unembed_matrix: Tensor,
        tokenizer: Any,
        k: int = 5,
        exclude_tokens: set[int] | None = None,
    ) -> list[tuple[str, float]]:
    ```
  - 计算 cosine similarity: `F.cosine_similarity(vector, unembed_matrix, dim=-1)`
  - 使用 `torch.topk()` 找 top-k
  - 排除 exclude_tokens 中的特殊 tokens
  - 返回 decoded text 和 similarity scores

  **Must NOT do**:
  - 不要使用 sklearn (保持 PyTorch 实现)
  - 不要在函数内加载模型或 tokenizer

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单函数实现，逻辑清晰
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: Tasks 8, 9
  - **Blocked By**: Task 1 (get_unembedding_matrix)

  **References**:
  - `src/steering_geometry/vector_analysis.py:24-44` - cosine similarity 模式
  - `src/steering_geometry/models.py:50-56` - tokenizer 使用模式

  **Acceptance Criteria**:
  - [ ] 函数返回 `list[tuple[str, float]]` 长度为 k
  - [ ] 排除的特殊 tokens 不出现在结果中
  - [ ] Similarity scores 范围在 [-1, 1]
  - [ ] `uv run mypy src/steering_geometry/unembed_analysis.py` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: Verify top-k computation
    Tool: Bash (uv run python -c "...")
    Steps:
      1. Create vector [1, 0, 0, ...] and matrix [[1,0,...], [0,1,...], ...]
      2. Call compute_topk_similar_tokens(k=3)
      3. Assert first result has similarity 1.0
      4. Assert results are sorted by similarity descending
    Expected Result: Top-3 results with correct similarities
    Evidence: .sisyphus/evidence/task-4-topk.txt
  ```

  **Commit**: NO (groups with Task 5, 6, 7)

- [x] 5. Implement `analyze_steering_vector()`

  **What to do**:
  - 实现单 vector 分析函数：
    ```python
    def analyze_steering_vector(
        vector: Tensor,
        model: HookedModel,
        layer_frac: float,
        method: str,
        k: int = 5,
    ) -> UnembedAnalysisResult:
    ```
  - 调用 `model.get_unembedding_matrix()`
  - 调用 `model.get_special_token_ids()`
  - 调用 `compute_topk_similar_tokens()`
  - 返回 `UnembedAnalysisResult`

  **Must NOT do**:
  - 不要在函数内加载模型
  - 不要打印到控制台 (使用 logging)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6, 7)
  - **Blocks**: Tasks 8, 9
  - **Blocked By**: Tasks 1, 2, 4

  **References**:
  - `src/steering_geometry/types.py` - UnembedAnalysisResult 类型
  - `src/steering_geometry/extract.py:481-540` - 返回 SteeringVector 模式

  **Acceptance Criteria**:
  - [ ] 返回 `UnembedAnalysisResult` 实例
  - [ ] 结果包含 layer, method, top5 tokens, similarities
  - [ ] `uv run mypy src/steering_geometry/unembed_analysis.py` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: Verify analysis result structure
    Tool: Bash (uv run pytest)
    Steps:
      1. Use mock_unembedding_matrix fixture
      2. Call analyze_steering_vector()
      3. Assert result.layer == input layer
      4. Assert len(result.tokens) == 5
    Expected Result: Correct UnembedAnalysisResult structure
    Evidence: .sisyphus/evidence/task-5-analyze.txt
  ```

  **Commit**: NO (groups with Task 4, 6, 7)

- [x] 6. Implement `run_unembed_experiment()`

  **What to do**:
  - 实现完整实验函数：
    ```python
    def run_unembed_experiment(
        concept: str,
        model_name: str,
        method: str,  # "diff_means" or "discriminative"
        layers: list[float],
        num_pairs: int = 1000,
        top_k: int = 30,  # for discriminative
        output_dir: Path | str = "outputs",
    ) -> ConceptAnalysisResult:
    ```
  - 加载模型 (`HookedModel`)
  - 提取 steering vectors (调用 `extract_vector()`)
  - 对每个 layer 调用 `analyze_steering_vector()`
  - 保存 JSON 结果
  - 返回 `ConceptAnalysisResult`

  **Must NOT do**:
  - 不要在循环中重复加载模型
  - 不要覆盖现有的 vector 文件

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 整合多个组件，需要理解整体流程
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 7)
  - **Blocks**: Tasks 8, 9, 12, 13
  - **Blocked By**: Tasks 1-5

  **References**:
  - `src/steering_geometry/vector_analysis.py:161-291` - `run_diff_means_experiment()` 模式
  - `src/steering_geometry/extract.py:550-620` - `extract_vector()` 入口

  **Acceptance Criteria**:
  - [ ] 函数支持 "diff_means" 和 "discriminative" 两种方法
  - [ ] 对每个 layer 生成 `UnembedAnalysisResult`
  - [ ] JSON 结果保存到 `outputs/unembed_analysis/json/{concept}_{method}.json`
  - [ ] `uv run mypy src/steering_geometry/unembed_analysis.py` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: Verify full experiment run
    Tool: Bash (uv run python -m steering_geometry.unembed_analysis --concept honesty --method diff_means --layers 0.5)
    Steps:
      1. Run single layer experiment
      2. Check JSON output file exists
      3. Parse JSON, verify it contains "layer_0.5" key
      4. Verify top-5 tokens are present
    Expected Result: JSON file with correct structure
    Evidence: .sisyphus/evidence/task-6-experiment.json
  ```

  **Commit**: YES (with Tasks 4, 5, 7)
  - Message: `feat(analysis): add unembed cosine similarity analysis core`

- [x] 7. Write unit tests for core functions

  **What to do**:
  - 在 `tests/test_unembed_analysis.py` 中编写测试：
    - `test_compute_topk_similar_tokens_basic()`: 基本功能
    - `test_compute_topk_similar_tokens_exclude()`: 排除特殊 tokens
    - `test_analyze_steering_vector()`: 整合测试
    - `test_run_unembed_experiment_mock()`: 使用 mock 的端到端测试

  **Must NOT do**:
  - 不要在单元测试中加载真实模型
  - 不要跳过测试或使用 `@pytest.mark.skip`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6)
  - **Blocks**: None
  - **Blocked By**: Tasks 3, 4, 5

  **References**:
  - `tests/conftest.py` - 测试 fixtures
  - `tests/test_hello.py` - 测试风格参考

  **Acceptance Criteria**:
  - [ ] `uv run pytest tests/test_unembed_analysis.py -v` → 4 tests pass
  - [ ] 测试覆盖率 >= 80%

  **QA Scenarios**:
  ```
  Scenario: All tests pass
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/test_unembed_analysis.py -v --cov=src/steering_geometry/unembed_analysis`
    Expected Result: 4 passed, coverage >= 80%
    Evidence: .sisyphus/evidence/task-7-tests.txt
  ```

  **Commit**: YES (with Tasks 4, 5, 6)

- [x] 8. Add CLI interface to `unembed_analysis.py`

  **What to do**:
  - 添加 `if __name__ == "__main__":` 入口
  - 使用 argparse 解析命令行参数：
    - `--concept`: concept 名称 (或 "all")
    - `--method`: "diff_means" | "discriminative" | "both"
    - `--model`: 模型名称
    - `--layers`: 层列表 (逗号分隔或 "default")
    - `--output`: 输出目录
  - 遵循 `extract.py` 的 CLI 模式

  **Must NOT do**:
  - 不要使用 click 或其他 CLI 库 (保持 argparse)
  - 不要在 CLI 代码中放业务逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10, 11)
  - **Blocks**: Task 12
  - **Blocked By**: Task 6

  **References**:
  - `src/steering_geometry/extract.py:550-620` - CLI 模式参考

  **Acceptance Criteria**:
  - [ ] `uv run python -m steering_geometry.unembed_analysis --help` 显示帮助
  - [ ] 支持所有指定参数
  - [ ] `uv run mypy src/steering_geometry/unembed_analysis.py` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: CLI help works
    Tool: Bash
    Steps:
      1. Run `uv run python -m steering_geometry.unembed_analysis --help`
    Expected Result: Shows usage with all arguments
    Evidence: .sisyphus/evidence/task-8-cli-help.txt
  ```

  **Commit**: NO (groups with Tasks 9, 10, 11)

- [x] 9. Implement JSON output formatting

  **What to do**:
  - 在 `unembed_analysis.py` 中添加 `save_analysis_results()` 函数
  - 格式化为可读的 JSON 结构：
    ```json
    {
      "concept": "honesty",
      "model": "Qwen/Qwen3-1.7B",
      "method": "diff_means",
      "results": {
        "layer_0.1": {"tokens": [...], "similarities": [...]},
        "layer_0.2": {...},
        ...
      }
    }
    ```
  - 使用 `json.dump(..., indent=2, ensure_ascii=False)`

  **Must NOT do**:
  - 不要在 JSON 中包含 tensor 或 numpy 类型 (转换为 Python 原生类型)
  - 不要硬编码文件路径

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 10, 11)
  - **Blocks**: Task 13
  - **Blocked By**: Task 6

  **References**:
  - `src/steering_geometry/vector_analysis.py:100-111` - save_vector 模式

  **Acceptance Criteria**:
  - [ ] JSON 文件格式正确，可被 `json.load()` 解析
  - [ ] 包含 concept, model, method, results 字段
  - [ ] 每层结果包含 tokens 和 similarities

  **QA Scenarios**:
  ```
  Scenario: JSON output is valid
    Tool: Bash (uv run python -c "import json; json.load(open('outputs/unembed_analysis/json/honesty_diff_means.json'))")
    Steps:
      1. Run analysis for single concept
      2. Load JSON file with json.load()
    Expected Result: No JSONDecodeError
    Evidence: .sisyphus/evidence/task-9-json.txt
  ```

  **Commit**: NO (groups with Tasks 8, 10, 11)

- [x] 10. Implement heatmap visualization

  **What to do**:
  - 在 `unembed_analysis.py` 中添加 `plot_topk_heatmap()` 函数
  - 热力图显示各层的 top-5 tokens：
    - X 轴: Top-K position (1-5)
    - Y 轴: Layer (0.1-1.0)
    - 单元格: Token text
  - 使用 matplotlib
  - 保存为 PDF 到 `outputs/unembed_analysis/plots/`

  **Must NOT do**:
  - 不要使用 seaborn (保持 matplotlib only)
  - 不要生成 PNG (使用 PDF)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: 需要设计清晰的可视化布局
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 9, 11)
  - **Blocks**: Task 13
  - **Blocked By**: Task 6

  **References**:
  - `src/steering_geometry/vector_analysis.py:47-98` - plot_heatmap 模式

  **Acceptance Criteria**:
  - [ ] 热力图显示 10 层 × 5 tokens
  - [ ] 每个 cell 显示 token text
  - [ ] PDF 保存成功

  **QA Scenarios**:
  ```
  Scenario: Heatmap generated correctly
    Tool: Bash
    Steps:
      1. Run analysis for honesty concept
      2. Check PDF file exists at outputs/unembed_analysis/plots/honesty_diff_means_heatmap.pdf
      3. Verify file size > 0
    Expected Result: PDF file exists with content
    Evidence: .sisyphus/evidence/task-10-heatmap.txt
  ```

  **Commit**: NO (groups with Tasks 8, 9, 11)

- [x] 11. Implement bar chart visualization

  **What to do**:
  - 在 `unembed_analysis.py` 中添加 `plot_topk_bar_chart()` 函数
  - 条形图显示单个层的 top-5 tokens 和相似度：
    - Y 轴: Token text
    - X 轴: Cosine similarity
    - 颜色: 按 similarity 值渐变
  - 为每个层生成单独的条形图，或使用 subplots 显示多层

  **Must NOT do**:
  - 不要在一个图中显示所有 10 层 (太拥挤)
  - 不要使用 3D 效果

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: 需要设计清晰的可视化布局
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 9, 10)
  - **Blocks**: Task 13
  - **Blocked By**: Task 6

  **References**:
  - `src/steering_geometry/vector_analysis.py:47-98` - matplotlib 模式

  **Acceptance Criteria**:
  - [ ] 条形图显示 5 个 tokens 和对应 similarities
  - [ ] 颜色渐变表示相似度
  - [ ] PDF 保存成功

  **QA Scenarios**:
  ```
  Scenario: Bar chart generated correctly
    Tool: Bash
    Steps:
      1. Run analysis for honesty concept
      2. Check PDF file exists at outputs/unembed_analysis/plots/honesty_diff_means_bars.pdf
      3. Verify file size > 0
    Expected Result: PDF file exists with content
    Evidence: .sisyphus/evidence/task-11-bars.txt
  ```

  **Commit**: YES (with Tasks 8, 9, 10)
  - Message: `feat(analysis): add CLI and visualization for unembed analysis`

- [x] 12. Create batch analysis script

  **What to do**:
  - 创建 `scripts/run_unembed_analysis.sh`
  - 循环运行所有 5 concepts × 2 methods = 10 个分析
  - 参数：模型 Qwen/Qwen3-1.7B，10 层 (0.1-1.0)
  - 输出汇总信息到控制台
  - 遵循 `scripts/run_extractions.sh` 的模式

  **Must NOT do**:
  - 不要在脚本中放 Python 代码
  - 不要硬编码路径 (使用变量)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 8)
  - **Parallel Group**: Sequential
  - **Blocks**: Task 13
  - **Blocked By**: Tasks 8-11

  **References**:
  - `scripts/run_extractions.sh` - 批量脚本模式
  - `scripts/run_pipeline.sh` - 参数处理模式

  **Acceptance Criteria**:
  - [ ] 脚本可执行 (`chmod +x`)
  - [ ] 支持 `-c` (concepts), `-m` (model), `-l` (layers) 参数
  - [ ] 运行成功后生成 10 个 JSON 文件

  **QA Scenarios**:
  ```
  Scenario: Batch script runs all concepts
    Tool: Bash
    Steps:
      1. Run `./scripts/run_unembed_analysis.sh -c all -m "Qwen/Qwen3-1.7B" -l "0.5"`
      2. Wait for completion
      3. Count JSON files in outputs/unembed_analysis/json/
    Expected Result: 10 JSON files (5 concepts × 2 methods)
    Evidence: .sisyphus/evidence/task-12-batch.txt
  ```

  **Commit**: NO (groups with Tasks 13, 14, 15)

- [x] 13. Run full analysis (5 concepts × 2 methods × 10 layers)

  **What to do**:
  - 运行批量分析脚本
  - 监控进度和错误
  - 确保所有 100 个结果正确生成 (5 concepts × 2 methods × 10 layers)
  - 检查 JSON 文件完整性

  **Must NOT do**:
  - 不要跳过任何 concept 或 method
  - 不要忽略错误或警告

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 长时间运行，需要监控和调试
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 14, 15
  - **Blocked By**: Tasks 8-12

  **References**:
  - `outputs/unembed_analysis/json/` - 输出目录

  **Acceptance Criteria**:
  - [ ] 10 个 JSON 文件生成 (5 concepts × 2 methods)
  - [ ] 每个 JSON 包含 10 层结果
  - [ ] 每层结果包含 top-5 tokens 和 similarities

  **QA Scenarios**:
  ```
  Scenario: All results generated
    Tool: Bash
    Steps:
      1. Run full analysis
      2. Verify 10 JSON files exist
      3. For each file, verify it has 10 layer keys
      4. For each layer, verify 5 tokens present
    Expected Result: 100 layer analyses (10 files × 10 layers)
    Evidence: .sisyphus/evidence/task-13-full-run.txt
  ```

  **Commit**: NO (groups with Tasks 12, 14, 15)

- [x] 14. Verify output integrity

  **What to do**:
  - 编写验证脚本检查输出：
    - JSON 格式正确
    - 所有 tokens 是有效字符串
    - Similarity 值在 [-1, 1] 范围内
    - 没有空结果或 NaN
  - 生成验证报告

  **Must NOT do**:
  - 不要修改任何结果文件
  - 不要假设数据正确

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 15
  - **Blocked By**: Task 13

  **References**:
  - `outputs/unembed_analysis/json/` - 输出目录

  **Acceptance Criteria**:
  - [ ] 验证脚本运行成功
  - [ ] 所有 100 个结果通过验证
  - [ ] 生成验证报告

  **QA Scenarios**:
  ```
  Scenario: All outputs pass validation
    Tool: Bash
    Steps:
      1. Run validation script
      2. Check exit code is 0
      3. Verify report shows 100% pass rate
    Expected Result: Validation passes
    Evidence: .sisyphus/evidence/task-14-validation.txt
  ```

  **Commit**: NO (groups with Tasks 12, 13, 15)

- [x] 15. Generate final report

  **What to do**:
  - 生成分析报告 Markdown 文件：`outputs/unembed_analysis/REPORT.md`
  - 内容包括：
    - 分析概述
    - 每个 concept 的发现摘要
    - 有趣的 token 模式观察
    - 可视化图表链接
  - 更新 `docs/PLANS.md` 添加此功能

  **Must NOT do**:
  - 不要在报告中包含原始 JSON 数据
  - 不要包含敏感信息

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 需要撰写清晰的技术报告
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Tasks 13, 14

  **References**:
  - `outputs/unembed_analysis/json/` - 分析结果
  - `docs/PLANS.md` - 项目路线图

  **Acceptance Criteria**:
  - [ ] REPORT.md 文件存在
  - [ ] 包含所有 5 个 concepts 的摘要
  - [ ] 包含可视化图表链接

  **QA Scenarios**:
  ```
  Scenario: Report generated correctly
    Tool: Bash
    Steps:
      1. Check REPORT.md exists
      2. Verify it contains all concept names
      3. Verify it has links to visualization files
    Expected Result: Complete report with all sections
    Evidence: .sisyphus/evidence/task-15-report.txt
  ```

  **Commit**: YES (with Tasks 12, 13, 14)
  - Message: `feat(scripts): add batch unembed analysis and final report`

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist. Compare deliverables against plan.
  Output: `Must Have [5/5] | Must NOT Have [4/4] | Tasks [15/15] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Review all changed files for: `as any`/`# type: ignore`, empty catches, print() in prod, commented-out code, unused imports. Check AI slop patterns.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Run full analysis from scratch. Verify: 10 JSON files exist, each has 10 layers, each layer has 5 tokens + similarities, all similarities in [-1, 1], no special tokens in results. Verify visualizations: heatmap and bar charts exist and are readable.
  Output: `JSON [10/10] | Layers [100/100] | Viz [N/N] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", check actual implementation. Verify nothing extra was added. Check "Must NOT do" compliance. Flag any scope creep.
  Output: `Tasks [15/15 compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Commit 1** (after Wave 1-2): `feat(models): add unembedding matrix access`
- **Commit 2** (after Wave 3): `feat(analysis): add unembed cosine similarity analysis`
- **Commit 3** (after Wave 4): `feat(scripts): add batch analysis and visualization`

---

## Success Criteria

### Verification Commands
```bash
uv run pytest tests/test_unembed_analysis.py -v  # All tests pass
uv run mypy src/steering_geometry/unembed_analysis.py  # 0 errors
ls outputs/unembed_analysis/json/*.json | wc -l  # Should be 10 (5 concepts × 2 methods)
ls outputs/unembed_analysis/plots/*.pdf | wc -l  # Should be >= 10
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] JSON outputs contain top-5 tokens per layer
- [ ] Visualizations generated correctly
