# Import Ordering Enforcement — Reusable Cookiecutter Template

## TL;DR

> **Quick Summary**: 为 Python 项目添加完整的 import 排序强制执行方案（ruff isort 精细化配置 + pre-commit hooks + GitHub Actions CI），并创建可复用的 Cookiecutter 模板。
> 
> **Deliverables**:
> - ruff isort 精细化配置（pyproject.toml）
> - pre-commit 配置文件（.pre-commit-config.yaml）
> - GitHub Actions CI 新增 pre-commit job
> - Cookiecutter 可复用模板（templates/cookiecutter-ruff-imports/）
> - AGENTS.md 更新（import 排序指引）
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 3 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 5

---

## Context

### Original Request
用户希望增加一个步骤来处理 Python 代码中的 import 排序问题，要求通用且可复用。具体需求：
1. Python 代码中 import 语句按规范排序（stdlib → third-party → local）
2. CI/Pre-commit 自动化强制检查
3. 可复用的 Cookiecutter 模板

### Interview Summary
**Key Discussions**:
- 项目已使用 ruff + `I`（isort）规则，当前零违规
- 缺少 pre-commit hooks、CI 中无专门的 import 检查步骤
- 用户希望以 Cookiecutter 模板形式输出可复用方案
- 用户要求包含完整方案（pre-commit + GitHub Actions）
- 用户要求修复 extract.py 的 E402 问题

**Research Findings**:
- `extract.py` 的 E402 是**运行时约束**（`warnings.filterwarnings()` 必须在 `datasets`/`sklearn` import 之前执行，否则 RuntimeWarning 会泄漏）— per-file-ignores 是正确的解决方案，不应"修复"
- 3 个源文件使用绝对 import（`from steering_geometry.X`），6 个使用相对 import（`from .X`）— 这是风格一致性问题，不是排序问题，不在本次 scope 内
- CI 已存在（`.github/workflows/ci.yml`），已有 `ruff check`（包含 `I` 规则）
- Ruff 有 32 个 isort 特定设置可配置

### Metis Review
**Identified Gaps** (addressed):
- E402 不能简单"修复" → 改为添加注释说明 per-file-ignore 的原因
- 绝对/相对 import 不一致 → 明确排除在 scope 外
- pre-commit ruff 版本漂移 → 必须与 pyproject.toml 保持同步
- Cookiecutter 模板放置位置 → 放在 `templates/cookiecutter-ruff-imports/`
- CI 已包含 ruff check → 不需要单独的 `--select I` 步骤，只需添加 pre-commit CI job

---

## Work Objectives

### Core Objective
为当前项目建立完整的 import 排序强制执行管道，并创建可复制到其他 Python 项目的 Cookiecutter 模板。

### Concrete Deliverables
- `pyproject.toml` 中新增 `[tool.ruff.lint.isort]` 配置段
- `.pre-commit-config.yaml` 文件（含 ruff-check 和 ruff-format hooks）
- `.github/workflows/ci.yml` 新增 pre-commit job
- `templates/cookiecutter-ruff-imports/` 完整模板目录
- `AGENTS.md` 更新

### Definition of Done
- [ ] `uv run ruff check --select I src/ tests/` → "All checks passed!"
- [ ] `uv run pre-commit run --all-files` → 所有 hooks 通过
- [ ] CI workflow YAML 语法有效
- [ ] Cookiecutter 模板可成功生成
- [ ] 完整 DoD: `uv sync && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/ && uv run pytest` → 全部通过

### Must Have
- `[tool.ruff.lint.isort]` 配置段含 `known-first-party`
- `.pre-commit-config.yaml` 使用官方 `astral-sh/ruff-pre-commit` 镜像
- pre-commit hook 顺序：ruff-check 在 ruff-format 之前
- ruff 版本在 pre-commit config 和 pyproject.toml 间同步
- Cookiecutter 模板包含 `cookiecutter.json` 和所有必要模板文件
- `pre-commit` 添加到 dev 依赖

### Must NOT Have (Guardrails)
- **不要移动 `extract.py` 中的 `warnings.filterwarnings()`** — 它必须在 datasets/sklearn imports 之前执行，E402 per-file-ignore 是正确的
- **不要规范化绝对/相对 import 风格** — 这是独立的技术债
- **不要添加 `from __future__ import annotations` 的强制要求**
- **不要清理 `.ipynb_checkpoints/`** — 独立问题
- **不要添加单独的 `--select I` CI 步骤** — 现有 `ruff check` 已包含 `I`
- **不要过度配置 isort** — 只添加 `known-first-party`，使用其余默认值
- **不要使用 `repo: local`** — 使用官方 ruff-pre-commit 镜像

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest, 231 tests)
- **Automated tests**: NO — 这是配置/工具变更，无新 Python 代码
- **Agent-Executed QA**: YES — 通过运行 ruff、pre-commit、cookiecutter 命令验证

### QA Policy
- **配置变更**: 用 `ruff check --select I` 和 `pre-commit run --all-files` 验证
- **YAML 语法**: 用 `python -c "import yaml; yaml.safe_load(...)"` 验证
- **Cookiecutter**: 用 `cookiecutter . --no-input` 验证生成成功
- **回归**: 每步后运行完整 DoD 命令确保无破坏

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — config foundation):
├── Task 1: ruff isort 精细化配置 [quick]
├── Task 2: pre-commit 配置 [quick]
└── Task 3: AGENTS.md 更新 [quick]

Wave 2 (After Wave 1 — CI + template, MAX PARALLEL):
├── Task 4: GitHub Actions CI 新增 pre-commit job (depends: 2) [quick]
└── Task 5: Cookiecutter 模板创建 (depends: 1, 2) [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 2 → Task 4 → F1-F4 → user okay
Parallel Speedup: ~40% faster than sequential
Max Concurrent: 3 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | - | 5 | 1 |
| 2 | - | 4, 5 | 1 |
| 3 | - | - | 1 |
| 4 | 2 | - | 2 |
| 5 | 1, 2 | - | 2 |
| F1 | 1-5 | user okay | FINAL |
| F2 | 1-5 | user okay | FINAL |
| F3 | 1-5 | user okay | FINAL |
| F4 | 1-5 | user okay | FINAL |

### Agent Dispatch Summary

- **Wave 1**: 3 agents — T1 → `quick`, T2 → `quick`, T3 → `quick`
- **Wave 2**: 2 agents — T4 → `quick`, T5 → `unspecified-high`
- **FINAL**: 4 agents — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. ruff isort 精细化配置

  **What to do**:
  - 在 `pyproject.toml` 中添加 `[tool.ruff.lint.isort]` 配置段，位于现有 `[tool.ruff.lint]` 之后、`[tool.ruff.format]` 之前
  - 添加 `known-first-party = ["steering_geometry"]` — 显式声明（虽然 ruff 可通过 `src = ["src"]` 自动检测，但显式声明更适合模板）
  - 在 `per-file-ignores` 的 E402 行添加注释说明原因：`# filterwarnings must precede datasets/sklearn imports to suppress RuntimeWarning`
  - 运行 `uv run ruff check --select I src/ tests/` 确认零违规
  - 运行完整 DoD 确认无回归

  **Must NOT do**:
  - 不要移动 `extract.py` 中的 `warnings.filterwarnings()`
  - 不要添加 `known-third-party`、`force-single-line` 等用户未要求的配置
  - 不要修改 `[tool.ruff.lint] select` 列表

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单文件配置变更，步骤清晰
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `git-master`: 简单变更，不需要高级 git 操作

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `pyproject.toml:36-48` — 现有 ruff 配置结构，新 `[tool.ruff.lint.isort]` 应插入在 `[tool.ruff.lint]` 和 `[tool.ruff.format]` 之间
  - `pyproject.toml:44-45` — 现有 per-file-ignores，需在此处添加注释

  **API/Type References** (contracts to implement against):
  - Ruff isort 文档：`https://docs.astral.sh/ruff/settings/#lint_isort` — 可用的 isort 配置选项
  - `known-first-party` 说明：`https://docs.astral.sh/ruff/settings/#lint_isort_known-first-party`

  **WHY Each Reference Matters**:
  - `pyproject.toml:36-48` — 确保新配置段放在正确的位置，保持 TOML 结构一致性
  - `pyproject.toml:44-45` — E402 注释需要添加到这个位置，解释为何保留 per-file-ignore

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: isort 配置正确生效
    Tool: Bash
    Preconditions: pyproject.toml 已更新
    Steps:
      1. 运行 `uv run ruff check --select I src/ tests/`
      2. 检查输出
    Expected Result: 输出 "All checks passed!" 或 "0 violations"
    Failure Indicators: 任何 isort 违规报告
    Evidence: .sisyphus/evidence/task-1-isort-config.txt

  Scenario: 完整 DoD 无回归
    Tool: Bash
    Preconditions: 配置变更已完成
    Steps:
      1. 运行 `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/ && uv run pytest`
    Expected Result: 所有检查通过，231 tests pass
    Failure Indicators: 任何 lint/type/test 错误
    Evidence: .sisyphus/evidence/task-1-full-dod.txt

  Scenario: known-first-party 被正确识别
    Tool: Bash
    Preconditions: 配置变更已完成
    Steps:
      1. 运行 `uv run ruff check --select I src/ tests/ --output-format json 2>/dev/null | python -c "import sys,json; violations=json.load(sys.stdin); print(f'{len(violations)} violations')" `
      2. 或者简单地检查 `uv run python -c "from ruff.__main__ import main; ..."` — 实际上直接验证配置加载即可
      3. 验证方法: 运行 `grep -A2 '\[tool.ruff.lint.isort\]' pyproject.toml` 确认配置存在
    Expected Result: 配置段存在且包含 `known-first-party = ["steering_geometry"]`
    Failure Indicators: 配置段不存在或格式错误
    Evidence: .sisyphus/evidence/task-1-config-verify.txt
  ```

  **Commit**: YES
  - Message: `feat(lint): add explicit ruff isort configuration`
  - Files: `pyproject.toml`
  - Pre-commit: `uv run ruff check --select I src/ tests/`

- [ ] 2. pre-commit 配置

  **What to do**:
  - 创建 `.pre-commit-config.yaml`，包含以下 hooks：
    ```yaml
    repos:
      - repo: https://github.com/astral-sh/ruff-pre-commit
        rev: v0.8.0  # Keep in sync with pyproject.toml dev dependencies
        hooks:
          - id: ruff-check
            args: [--fix]
          - id: ruff-format
    ```
  - **关键**: `ruff-check` 必须在 `ruff-format` 之前（因为 `--fix` 会修改文件，format 需要在 fix 之后运行）
  - 在 `pyproject.toml` 的 `[dependency-groups] dev` 中添加 `"pre-commit>=4.0.0"` 依赖
  - 运行 `uv sync` 安装新依赖
  - 运行 `uv run pre-commit run --all-files` 确认通过
  - 运行 `uv run pre-commit install` 设置 git hooks（仅说明，不在计划中执行）

  **Must NOT do**:
  - 不要使用 `repo: local` — 使用官方 ruff-pre-commit 镜像
  - 不要添加 `--select I` 参数到 ruff-check — ruff-check 应运行完整的 lint 配置
  - 不要添加 standalone isort hook — ruff 的 `I` 规则已覆盖
  - 不要在 hooks 中添加 mypy 或 pytest — 那些不是 pre-commit 的职责

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 创建一个新文件 + 添加一行依赖
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 4, 5
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `.github/workflows/ci.yml` — 现有 CI 的 ruff 命令，pre-commit hooks 应与 CI 保持一致
  - `pyproject.toml:29-34` — 现有 dev 依赖格式

  **API/Type References**:
  - ruff-pre-commit 官方仓库: `https://github.com/astral-sh/ruff-pre-commit` — hook ID 和配置选项
  - pre-commit 配置文档: `https://pre-commit.com/#configuration`

  **WHY Each Reference Matters**:
  - `.github/workflows/ci.yml` — 确保 pre-commit hooks 和 CI 运行相同的检查，避免不一致
  - `pyproject.toml:29-34` — dev 依赖的格式和分组约定

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: pre-commit 配置文件语法正确
    Tool: Bash
    Preconditions: .pre-commit-config.yaml 已创建
    Steps:
      1. 运行 `uv run python -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml')); print('YAML valid')"`
    Expected Result: 输出 "YAML valid"
    Failure Indicators: Python 异常
    Evidence: .sisyphus/evidence/task-2-yaml-valid.txt

  Scenario: pre-commit hooks 通过现有代码
    Tool: Bash
    Preconditions: uv sync 已完成, pre-commit 已安装
    Steps:
      1. 运行 `uv sync`
      2. 运行 `uv run pre-commit run --all-files`
    Expected Result: 所有 hooks 显示 "Passed"
    Failure Indicators: 任何 hook 显示 "Failed"
    Evidence: .sisyphus/evidence/task-2-precommit-run.txt

  Scenario: hook 顺序正确（ruff-check 在 ruff-format 之前）
    Tool: Bash
    Preconditions: .pre-commit-config.yaml 已创建
    Steps:
      1. 运行 `uv run python -c "
      import yaml
      config = yaml.safe_load(open('.pre-commit-config.yaml'))
      hooks = config['repos'][0]['hooks']
      hook_ids = [h['id'] for h in hooks]
      assert hook_ids.index('ruff-check') < hook_ids.index('ruff-format'), 'ruff-check must come before ruff-format'
      print(f'Hook order correct: {hook_ids}')
      "`
    Expected Result: 输出 "Hook order correct: ['ruff-check', 'ruff-format']"
    Failure Indicators: AssertionError 或顺序错误
    Evidence: .sisyphus/evidence/task-2-hook-order.txt
  ```

  **Commit**: YES
  - Message: `feat(hooks): add pre-commit configuration with ruff hooks`
  - Files: `.pre-commit-config.yaml`, `pyproject.toml`
  - Pre-commit: `uv sync && uv run pre-commit run --all-files`

- [ ] 3. AGENTS.md 更新

  **What to do**:
  - 更新 AGENTS.md section 2 (Build & Verify Commands)，添加 pre-commit 相关命令：
    ```bash
    # Pre-commit checks (runs automatically on git commit)
    uv run pre-commit run --all-files  # Run all hooks manually
    ```
  - 更新 AGENTS.md section 4 (When Writing Code)，添加 import 排序指引：
    - Import 顺序由 ruff isort 自动处理，无需手动排序
    - 运行 `uv run ruff check --fix` 可自动修复排序问题
    - `extract.py` 的 E402 per-file-ignore 是有意为之（运行时约束）
  - 修复 AGENTS.md 中对 `scripts/validate_analysis_json.py` 的过时引用（如果存在）
  - 更新 section 5 (When Writing Tests) 如果需要

  **Must NOT do**:
  - 不要重写整个 AGENTS.md
  - 不要添加 `from __future__ import annotations` 的强制要求
  - 不要讨论绝对/相对 import 规范化问题

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 文件编辑，几个小段落
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `AGENTS.md:17-38` — 现有 Build & Verify Commands 区域，新增命令应插入此处
  - `AGENTS.md:40-50` — 现有 When Writing Code 区域，import 指引应插入此处

  **WHY Each Reference Matters**:
  - 需要在现有文档结构中插入新内容，保持格式和风格一致

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: AGENTS.md 包含 pre-commit 命令
    Tool: Bash
    Preconditions: AGENTS.md 已更新
    Steps:
      1. 运行 `grep -c "pre-commit" AGENTS.md`
      2. 验证计数 >= 2
    Expected Result: grep 返回计数 >= 2
    Failure Indicators: 计数为 0 或 1（说明内容不充分）
    Evidence: .sisyphus/evidence/task-3-agents-update.txt

  Scenario: 文档引用准确
    Tool: Bash
    Preconditions: AGENTS.md 已更新
    Steps:
      1. 验证 section 2 包含 "pre-commit run" 命令
      2. 验证 section 4 包含 import 排序相关指引
    Expected Result: 两个 section 都包含相关内容
    Failure Indicators: 任一 section 缺失内容
    Evidence: .sisyphus/evidence/task-3-docs-verify.txt
  ```

  **Commit**: YES
  - Message: `docs: update AGENTS.md with import enforcement guidelines`
  - Files: `AGENTS.md`
  - Pre-commit: `uv run ruff check src/ tests/`

- [ ] 4. GitHub Actions CI 新增 pre-commit job

  **What to do**:
  - 在 `.github/workflows/ci.yml` 中新增一个 `pre-commit` job（与现有 `check` job 平级）
  - Job 结构：
    ```yaml
    pre-commit:
      runs-on: ubuntu-latest
      steps:
        - name: Checkout
          uses: actions/checkout@v4
        - name: Install uv
          uses: astral-sh/setup-uv@v4
        - name: Set up Python
          uses: actions/setup-python@v5
          with:
            python-version-file: .python-version
        - name: Install dependencies
          run: uv sync
        - name: Run pre-commit
          run: uv run pre-commit run --all-files
    ```
  - 此 job 独立于现有 `check` job，两者并行运行
  - 不需要单独的 `--select I` 步骤 — 现有 `check` job 的 `ruff check` 已覆盖

  **Must NOT do**:
  - 不要修改现有 `check` job
  - 不要添加 `--select I` 的单独步骤
  - 不要添加 mypy/pytest 到 pre-commit hooks 或 CI
  - 不要使用 pre-commit GitHub Action（`pre-commit/action`）— 使用 `uv run` 确保环境一致

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 在现有 CI 文件中添加一个 job，结构明确
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (within Wave 2)
  - **Parallel Group**: Wave 2 (with Task 5)
  - **Blocks**: None
  - **Blocked By**: Task 2 (需要 pre-commit 配置文件已存在)

  **References**:

  **Pattern References**:
  - `.github/workflows/ci.yml:1-38` — 完整的现有 CI workflow，新 job 应复制相同的 setup 步骤
  - `.github/workflows/ci.yml:10-37` — 现有 `check` job 结构，新 `pre-commit` job 应保持相同的 setup 模式

  **WHY Each Reference Matters**:
  - 现有 CI 是新 job 的模板 — 保持相同的 checkout、uv setup、python setup 流程确保一致性

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: CI YAML 语法有效
    Tool: Bash
    Preconditions: ci.yml 已更新
    Steps:
      1. 运行 `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('CI YAML valid')"`
    Expected Result: 输出 "CI YAML valid"
    Failure Indicators: Python 异常
    Evidence: .sisyphus/evidence/task-4-ci-yaml.txt

  Scenario: CI 包含 pre-commit job
    Tool: Bash
    Preconditions: ci.yml 已更新
    Steps:
      1. 运行 `uv run python -c "
      import yaml
      config = yaml.safe_load(open('.github/workflows/ci.yml'))
      jobs = list(config['jobs'].keys())
      assert 'pre-commit' in jobs, f'pre-commit job not found, got: {jobs}'
      print(f'Jobs: {jobs}')
      "`
    Expected Result: 输出包含 "pre-commit" 的 jobs 列表
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-4-precommit-job.txt

  Scenario: 原有 check job 未被修改
    Tool: Bash
    Preconditions: ci.yml 已更新
    Steps:
      1. 运行 `uv run python -c "
      import yaml
      config = yaml.safe_load(open('.github/workflows/ci.yml'))
      check_steps = [s.keys() for s in config['jobs']['check']['steps']]
      assert len(check_steps) == 6, f'check job has {len(check_steps)} steps, expected 6'
      print(f'check job has {len(check_steps)} steps - unchanged')
      "`
    Expected Result: check job 仍有 6 个步骤
    Failure Indicators: 步骤数不匹配
    Evidence: .sisyphus/evidence/task-4-check-unchanged.txt
  ```

  **Commit**: YES
  - Message: `ci: add pre-commit job to GitHub Actions workflow`
  - Files: `.github/workflows/ci.yml`
  - Pre-commit: YAML syntax validation

- [ ] 5. Cookiecutter 可复用模板

  **What to do**:
  - 创建 `templates/cookiecutter-ruff-imports/` 目录结构：
    ```
    templates/cookiecutter-ruff-imports/
    ├── cookiecutter.json                    # 模板变量定义
    ├── README.md                            # 模板使用说明
    ├── hooks/
    │   └── post_gen_project.py              # 生成后提示
    └── {{cookiecutter.project_slug}}/
        ├── .pre-commit-config.yaml          # pre-commit 配置模板
        ├── .github/
        │   └── workflows/
        │       └── ci.yml                   # CI workflow 模板（pre-commit job 片段）
        ├── pyproject.toml.snippet           # ruff isort 配置片段（需手动合并）
        └── README.md                        # 使用说明（如何将片段集成到项目）
    ```
  - `cookiecutter.json` 包含变量：
    ```json
    {
      "project_slug": "my-project",
      "package_name": "my_package",
      "python_version": "3.12",
      "line_length": "100",
      "quote_style": "double",
      "ruff_version": "0.8.0",
      "use_github_actions": true
    }
    ```
  - `.pre-commit-config.yaml` 模板使用 `{{cookiecutter.ruff_version}}` 变量
  - `pyproject.toml.snippet` 包含 `[tool.ruff.lint.isort]` 配置模板
  - CI workflow 模板使用 `{{cookiecutter.package_name}}` 替代硬编码值
  - README.md 说明如何使用模板和集成步骤

  **Must NOT do**:
  - 不要创建独立的 Python 包 — 这只是配置模板
  - 不要使用 cruft — 纯 Cookiecutter 即可
  - 不要在模板中硬编码 `steering_geometry` — 使用 `{{cookiecutter.package_name}}`
  - 不要生成完整的项目模板 — 只生成 import 排序相关的配置文件
  - 模板不要包含 mypy/pytest 配置 — 只关注 ruff isort + pre-commit

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 多文件模板创建，需要细心处理 Cookiecutter 语法和变量替换
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (within Wave 2)
  - **Parallel Group**: Wave 2 (with Task 4)
  - **Blocks**: None
  - **Blocked By**: Task 1 (需要了解最终 isort 配置), Task 2 (需要了解最终 pre-commit 配置)

  **References**:

  **Pattern References**:
  - `.pre-commit-config.yaml` (Task 2 创建) — 模板应使用相同的 hook 结构，但用变量替换版本号
  - `pyproject.toml:36-48` (Task 1 更新后) — 模板的 isort 配置应与此一致
  - `.github/workflows/ci.yml` (Task 4 更新后) — 模板的 CI 片段应与此一致

  **External References**:
  - Cookiecutter 文档: `https://www.cookiecutter.io/` — 模板结构和变量语法
  - Cookiecutter JSON schema: `https://cookiecutter.readthedocs.io/en/stable/cookiecutter.html`

  **WHY Each Reference Matters**:
  - Task 1-2 的输出是模板的"源 truth" — 模板必须是这些配置的参数化版本
  - Cookiecutter 文档确保模板语法正确

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Cookiecutter 模板可成功生成
    Tool: Bash
    Preconditions: 模板文件已创建
    Steps:
      1. 安装 cookiecutter: `uv run pip install cookiecutter` 或 `uv tool install cookiecutter`
      2. 运行 `cd templates/cookiecutter-ruff-imports && cookiecutter . --no-input`
      3. 检查生成的文件是否存在
    Expected Result: 无错误，生成目录包含 .pre-commit-config.yaml, pyproject.toml.snippet 等
    Failure Indicators: cookiecutter 报错或文件缺失
    Evidence: .sisyphus/evidence/task-5-cookiecutter-generate.txt

  Scenario: 生成的 .pre-commit-config.yaml 包含正确的变量替换
    Tool: Bash
    Preconditions: cookiecutter 已生成项目
    Steps:
      1. 运行 cookiecutter 生成
      2. 检查生成的 .pre-commit-config.yaml 中 `rev:` 值
      3. 验证不包含 Jinja2 模板语法（`{{` / `}}`）
    Expected Result: rev 值为实际版本号（如 "v0.8.0"），无模板残留
    Failure Indicators: 文件中包含 `{{cookiecutter.xxx}}`
    Evidence: .sisyphus/evidence/task-5-template-output.txt

  Scenario: cookiecutter.json 是有效 JSON
    Tool: Bash
    Preconditions: cookiecutter.json 已创建
    Steps:
      1. 运行 `python -c "import json; json.load(open('templates/cookiecutter-ruff-imports/cookiecutter.json')); print('JSON valid')"`
    Expected Result: 输出 "JSON valid"
    Failure Indicators: JSON 解析错误
    Evidence: .sisyphus/evidence/task-5-json-valid.txt

  Scenario: 自定义变量生成正确输出
    Tool: Bash
    Preconditions: 模板已创建
    Steps:
      1. 运行 `cookiecutter templates/cookiecutter-ruff-imports --no-input package_name="my_lib" line_length="120"`
      2. 检查生成的文件中 line-length 值
    Expected Result: 生成的配置中 line-length 为 "120"，package_name 相关处为 "my_lib"
    Failure Indicators: 值未替换或仍为默认值
    Evidence: .sisyphus/evidence/task-5-custom-vars.txt
  ```

  **Commit**: YES
  - Message: `feat(template): create Cookiecutter template for ruff import enforcement`
  - Files: `templates/cookiecutter-ruff-imports/` (整个目录)
  - Pre-commit: cookiecutter smoke test

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Review all changed files for: config errors, version mismatches, YAML syntax issues.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute EVERY QA scenario from EVERY task. Run `uv run pre-commit run --all-files` to verify hooks work. Run `cookiecutter` with `--no-input` to verify template generates. Test edge cases: dirty files, missing config.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec. Check "Must NOT do" compliance. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Commit 1**: `feat(lint): add explicit ruff isort configuration` — `pyproject.toml`
  - Pre-commit: `uv run ruff check --select I src/ tests/`
- **Commit 2**: `feat(hooks): add pre-commit configuration with ruff hooks` — `.pre-commit-config.yaml`, `pyproject.toml`
  - Pre-commit: `uv run pre-commit run --all-files`
- **Commit 3**: `docs: update AGENTS.md with import enforcement guidelines` — `AGENTS.md`
  - Pre-commit: `uv run ruff check src/ tests/`
- **Commit 4**: `ci: add pre-commit job to GitHub Actions workflow` — `.github/workflows/ci.yml`
  - Pre-commit: YAML syntax validation
- **Commit 5**: `feat(template): create Cookiecutter template for ruff import enforcement` — `templates/cookiecutter-ruff-imports/`
  - Pre-commit: `cookiecutter . --no-input` smoke test

---

## Success Criteria

### Verification Commands
```bash
# Import ordering check
uv run ruff check --select I src/ tests/  # Expected: All checks passed!

# Pre-commit hooks
uv run pre-commit run --all-files  # Expected: Passed

# Full CI equivalent
uv sync && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/ && uv run pytest
# Expected: All pass, 0 errors

# Cookiecutter template
cd templates/cookiecutter-ruff-imports && cookiecutter . --no-input
# Expected: Generates project without errors

# CI YAML valid
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
# Expected: No exception
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All 231 existing tests still pass
- [ ] `pre-commit run --all-files` passes
- [ ] Cookiecutter template generates valid output
