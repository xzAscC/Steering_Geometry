# Logging System: Replace print() with Unified Python Logging

## TL;DR

> **Quick Summary**: Replace all 57 `print()` calls across 5 files with Python stdlib `logging`, adding a shared `configure_logging()` function in `utils.py` that outputs to both console and per-run log files in `logs/`.
>
> **Deliverables**:
> - `configure_logging()` function in `utils.py` (dual output: console + file)
> - `tests/unit/test_logging.py` with TDD coverage
> - 53 `print()` → `logger.*()` replacements across 5 files
> - `--log-level` CLI flag on all 5 CLI modules
> - `logs/` directory managed via `.gitignore`
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: T1 (infra) → T3-T8 (migrations) → T9 (docs) → Final

---

## Context

### Original Request
用户要求将项目中所有 `print()` 替换为统一日志系统，日志需同时输出到控制台和 `.log` 文件，建立专门的 `logs/` 文件夹管理。

### Interview Summary
**Key Discussions**:
- 日志库选择: 确认使用 Python stdlib `logging`（4个文件已在用，无需新增依赖）
- 日志文件夹: 项目根目录 `logs/`
- 日志文件命名: 按运行实例 `steering_YYYYMMDD_HHMMSS.log`
- CLI 日志控制: 每个模块加 `--log-level` 选项（DEBUG/INFO/WARNING/ERROR）
- 测试策略: TDD（测试优先）

**Research Findings**:
- 57 个 `print()` 分布在 5 个文件中（extract.py 10, apply_steering.py 17, tdnv.py 25, token_analysis.py 12, __main__.py 3）
- 4 个文件已有部分 logging 但配置不一致（token_analysis.py, unembed_analysis.py, stability_comparison.py, token_selection_experiments.py）
- `unembed_analysis.py` 有孤立的 `logging.basicConfig()` 调用需替换
- `__main__.py` 有 3 行 shell eval 输出必须保留为 `print()`
- `token_analysis.py` 有格式化表格输出（CLI 表格），建议保留为 `print()`
- `stability_comparison.py` 使用 lazy `%s` 格式是黄金标准
- `test_experiments.py` 已使用 `caplog` fixture

### Metis Review
**Identified Gaps** (addressed):
- **`__main__.py` shell eval 输出不能迁移**: 3 行 `print()` 被 shell 脚本通过 `eval` 消费 → 保留
- **幂等性问题**: `configure_logging()` 被多次调用会导致重复 handler → 需幂等守卫
- **basicConfig 冲突**: 不能用 `logging.basicConfig()`，需用 `getLogger().addHandler()` 模式
- **CLI 格式化表格**: `token_analysis.py` 的表格输出（`=== Layer 3 ===`、编号列表）在日志文件中会变丑 → 保留为 `print()`
- **重复行**: `token_analysis.py` 第 652 行 print 重复了第 651 行 logger → 删除
- **现有 logger 静默问题**: 4 个文件有 logger 但无 handler → `configure_logging()` 自动修复
- **日志格式风格**: 使用 lazy `%s` 格式（匹配 `stability_comparison.py` 黄金标准）

---

## Work Objectives

### Core Objective
将项目中所有 `print()` 替换为 Python stdlib `logging`，建立统一的日志配置基础设施，使日志同时输出到控制台和按运行实例命名的日志文件中。

### Concrete Deliverables
- `src/steering_geometry/utils.py` 中的 `configure_logging()` 函数
- `tests/unit/test_logging.py` 测试文件
- 5 个模块的 `print()` → `logger` 迁移
- 5 个 CLI 模块的 `--log-level` 参数
- `.gitignore` 中的 `logs/` 条目

### Definition of Done
- [ ] `uv run pytest` → 所有测试通过（含新增 logging 测试）
- [ ] `uv run mypy src/` → 0 errors
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `grep -r "print(" src/steering_geometry/` → 仅剩 `__main__.py` 的 3 行 shell eval + `token_analysis.py` 的 CLI 表格输出
- [ ] `uv run python -m steering_geometry --shell` → shell eval 输出不变

### Must Have
- `configure_logging()` 必须幂等（多次调用不重复添加 handler）
- 双输出：StreamHandler（控制台）+ FileHandler（文件）
- 按运行实例命名日志文件 `steering_YYYYMMDD_HHMMSS.log`
- 每个带 main() 的 CLI 模块加 `--log-level` 参数
- 使用 lazy `%s` 日志格式（`logger.info("Saved %s", path)`）
- 所有库函数中的 `print()` 必须迁移（`extract_vector()`, `apply_steering()`, `compute_tdnv_*()`）

### Must NOT Have (Guardrails)
- ❌ 不添加第三方日志依赖（loguru, structlog 等）
- ❌ 不创建新文件（如 `logging_config.py`）— 放在 `utils.py`
- ❌ 不迁移 `__main__.py` 第 40-42 行的 shell eval `print()`
- ❌ 不迁移 `token_analysis.py` 的 CLI 格式化表格输出
- ❌ 不使用 `logging.basicConfig()`（避免冲突）
- ❌ 不修改 `scripts/` 目录下的任何文件
- ❌ 不改变任何库函数的行为 — 只切换输出机制
- ❌ 不添加 ruff `G` 规则（单独 scope）
- ❌ 不使用 f-string 日志格式 — 使用 lazy `%s`

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest, 21 tests)
- **Automated tests**: TDD
- **Framework**: pytest (with `caplog` and `tmp_path` fixtures)
- **TDD flow**: RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Infrastructure**: Use Bash (pytest) — Run tests, assert pass/fail
- **Module migration**: Use Bash (grep + pytest + mypy) — Verify print removal, type check, test pass
- **CLI integration**: Use Bash (python -m) — Run module, check output, verify log file creation

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation):
├── Task 1: Logging infrastructure (utils.py) - TDD [deep]
└── Task 2: Add logs/ to .gitignore [quick]

Wave 2 (After Task 1 - module migration, MAX PARALLEL):
├── Task 3: Migrate extract.py [unspecified-high]
├── Task 4: Migrate apply_steering.py [unspecified-high]
├── Task 5: Migrate tdnv.py [unspecified-high]
├── Task 6: Migrate __main__.py [quick]
├── Task 7: Migrate token_analysis.py [unspecified-high]
└── Task 8: Migrate unembed_analysis.py [quick]

Wave 3 (After Wave 2 - docs cleanup):
└── Task 9: Update AGENTS.md + QUALITY_SCORE.md [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay

Critical Path: T1 → T3-T8 → T9 → FINAL
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 6 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| T1 | - | T3, T4, T5, T6, T7, T8 | 1 |
| T2 | - | - | 1 |
| T3 | T1 | T9 | 2 |
| T4 | T1 | T9 | 2 |
| T5 | T1 | T9 | 2 |
| T6 | T1 | T9 | 2 |
| T7 | T1 | T9 | 2 |
| T8 | T1 | T9 | 2 |
| T9 | T3-T8 | FINAL | 3 |

### Agent Dispatch Summary

- **Wave 1**: 2 tasks — T1 → `deep`, T2 → `quick`
- **Wave 2**: 6 tasks — T3 → `unspecified-high`, T4 → `unspecified-high`, T5 → `unspecified-high`, T6 → `quick`, T7 → `unspecified-high`, T8 → `quick`
- **Wave 3**: 1 task — T9 → `quick`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Logging Infrastructure (utils.py) — TDD

  **What to do**:
  - **TDD RED**: 先创建 `tests/unit/test_logging.py`，编写以下测试（此时应失败，因为 `configure_logging` 不存在）：
    1. `test_configure_logging_creates_log_file` — 调用 `configure_logging(level="DEBUG", log_dir=tmp_path, log_name="test.log")`，验证文件被创建
    2. `test_configure_logging_writes_to_file` — 调用后 `logger.info("test message")`，验证文件包含该消息
    3. `test_configure_logging_console_output` — 验证 StreamHandler 存在且级别正确
    4. `test_configure_logging_idempotent` — 调用两次，验证 handler 不重复（只有 1 个 FileHandler + 1 个 StreamHandler）
    5. `test_configure_logging_level_filtering` — `level="WARNING"` 时，`logger.info("suppressed")` 不出现在输出中，但 `logger.warning("visible")` 出现
    6. `test_configure_logging_default_level` — 无 level 参数时默认 INFO
    7. `test_configure_logging_creates_log_dir` — 当 `log_dir` 不存在时自动创建
  - **TDD GREEN**: 在 `src/steering_geometry/utils.py` 中实现 `configure_logging()`：
    - 函数签名: `configure_logging(level: str = "INFO", log_dir: Path | None = None, log_name: str | None = None) -> None`
    - 使用 `logging.getLogger("steering_geometry")` 作为父 logger（不是 root logger）
    - 添加 StreamHandler（控制台，简洁格式：`%(levelname)s - %(name)s - %(message)s`）
    - 添加 FileHandler（文件，详细格式：`%(asctime)s - %(name)s - %(levelname)s - %(message)s`）
    - 日志文件路径: `log_dir / log_name`，默认 `logs/steering_YYYYMMDD_HHMMSS.log`
    - 幂等守卫: 检查 `"steering_geometry"` logger 是否已有 handler，若有则跳过（除非 force 参数）
    - 不要使用 `logging.basicConfig()`
  - 更新 `utils.py` 的 `__all__` 导出 `configure_logging`

  **Must NOT do**:
  - 不使用 `logging.basicConfig()`
  - 不创建新的 Python 文件（如 `logging_config.py`）
  - 不使用 root logger（用命名 logger `"steering_geometry"`）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: TDD 要求先写失败测试再实现，需要理解 logging handler 体系和幂等性设计
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 3, 4, 5, 6, 7, 8
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `src/steering_geometry/utils.py` — 现有共享工具模块，`configure_logging()` 应添加在此文件中，遵循现有的 `ensure_dir()` 等工具函数模式
  - `src/steering_geometry/stability_comparison.py:34` — `logger = logging.getLogger(__name__)` 黄金标准模式，所有新迁移的文件应遵循此模式

  **API/Type References**:
  - `src/steering_geometry/utils.py` — 查看现有 `__all__` 导出列表，确保 `configure_logging` 被添加

  **Test References**:
  - `tests/test_experiments.py` — 已使用 `caplog.set_level("WARNING")` + `assert "..." in caplog.text` 模式，新测试应遵循此模式
  - `tests/conftest.py` — 查看 `tmp_path` fixture 用法（pytest 内置，直接使用即可）

  **WHY Each Reference Matters**:
  - `utils.py` 是添加 `configure_logging` 的位置，需理解现有结构和导出模式
  - `stability_comparison.py` 的 logging 模式是所有其他文件的对齐目标
  - `test_experiments.py` 的 `caplog` 用法是测试 logging 的成熟模式，直接复用

  **Acceptance Criteria**:

  **If TDD (tests enabled)**:
  - [ ] Test file created: tests/unit/test_logging.py
  - [ ] `uv run pytest tests/unit/test_logging.py -v` → PASS (7 tests, 0 failures)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Logging infrastructure creates dual output
    Tool: Bash (pytest)
    Preconditions: utils.py has configure_logging() implemented
    Steps:
      1. Run: uv run pytest tests/unit/test_logging.py -v
      2. Assert all 7 tests pass with 0 failures
      3. Specifically check test_configure_logging_idempotent passes
    Expected Result: 7 passed, 0 failed
    Failure Indicators: Any test failure, especially idempotency test
    Evidence: .sisyphus/evidence/task-1-test-results.txt

  Scenario: Type checking passes for new code
    Tool: Bash (mypy)
    Preconditions: configure_logging() implemented
    Steps:
      1. Run: uv run mypy src/steering_geometry/utils.py
      2. Assert 0 errors
    Expected Result: Success: no issues found
    Failure Indicators: Any type error in utils.py
    Evidence: .sisyphus/evidence/task-1-mypy.txt
  ```

  **Commit**: YES
  - Message: `feat: add configure_logging() to utils.py with tests`
  - Files: `src/steering_geometry/utils.py`, `tests/unit/test_logging.py`
  - Pre-commit: `uv run pytest tests/unit/test_logging.py && uv run mypy src/steering_geometry/utils.py`

- [x] 2. Add logs/ to .gitignore

  **What to do**:
  - 在项目根目录的 `.gitignore` 中添加 `logs/` 条目（如文件不存在则创建）
  - 确保条目格式正确（每行一条规则，按字母顺序插入到合适位置）

  **Must NOT do**:
  - 不修改其他 .gitignore 规则
  - 不创建 logs/ 目录本身（运行时自动创建）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单行变更，无需复杂推理
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1)
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `.gitignore` — 查看现有忽略规则，确保新条目风格一致

  **WHY Each Reference Matters**:
  - 需确认 `.gitignore` 是否已存在及现有格式

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: logs/ is ignored by git
    Tool: Bash
    Preconditions: .gitignore updated
    Steps:
      1. mkdir -p logs/
      2. touch logs/test.log
      3. Run: git status logs/
      4. Assert logs/ does not appear in untracked files
      5. rm -rf logs/
    Expected Result: logs/ directory is ignored by git
    Failure Indicators: logs/ appears in git status output
    Evidence: .sisyphus/evidence/task-2-gitignore.txt
  ```

  **Commit**: NO (groups with Task 9 docs commit)

- [x] 3. Migrate extract.py from print() to logging

  **What to do**:
  - 在文件顶部添加 `import logging` 和 `logger = logging.getLogger(__name__)`
  - 在 `_build_parser()` 中添加 `--log-level` 参数：`parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")`
  - 在 `main()` 函数开头调用 `configure_logging(level=args.log_level)`
  - 将所有 10 个 `print()` 替换为对应的 `logger.info()` / `logger.warning()` / `logger.error()` 调用
  - 使用 lazy `%s` 格式：`logger.info("Extracting %s vector for %s", concept, model_name)`
  - 注意 `extract_vector()` 是公共库函数（有 1 个 print），也必须迁移
  - 运行 `uv run ruff check src/steering_geometry/extract.py` 确认无违规

  **Must NOT do**:
  - 不使用 f-string 日志格式
  - 不改变 extract_vector() 的任何业务逻辑
  - 不添加 print() 之外的新日志消息

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解每个 print 的上下文来决定正确的日志级别，10 处替换需要仔细判断
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5, 6, 7, 8)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1 (needs configure_logging)

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py` — 黄金标准：`logger = logging.getLogger(__name__)` + lazy `%s` 格式 + `logger.info("Saved to %s", path)` 模式

  **API/Type References**:
  - `src/steering_geometry/utils.py:configure_logging()` — Task 1 实现的配置函数，在 main() 中调用
  - `src/steering_geometry/extract.py:_build_parser()` — 需在此添加 `--log-level` 参数
  - `src/steering_geometry/extract.py:main()` — 需在此开头调用 `configure_logging()`

  **Test References**:
  - `tests/test_experiments.py` — 查看 caplog 测试模式

  **WHY Each Reference Matters**:
  - `stability_comparison.py` 是已正确使用 logging 的范例，extract.py 的迁移应与其风格一致
  - `_build_parser()` 和 `main()` 是需要修改的具体位置

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No print() remains in extract.py (except zero tolerance)
    Tool: Bash (grep)
    Preconditions: Migration complete
    Steps:
      1. Run: grep -n "print(" src/steering_geometry/extract.py
      2. Assert: no output (zero matches)
    Expected Result: grep returns exit code 1 (no matches found)
    Failure Indicators: Any print() line found
    Evidence: .sisyphus/evidence/task-3-no-print.txt

  Scenario: --log-level flag accepted and configure_logging called
    Tool: Bash (python)
    Preconditions: extract.py migrated
    Steps:
      1. Run: uv run python -m steering_geometry.extract --help 2>&1 | grep -q "\-\-log-level"
      2. Assert: --log-level appears in help output
    Expected Result: --log-level flag present in CLI help
    Failure Indicators: Flag not found in help output
    Evidence: .sisyphus/evidence/task-3-cli-help.txt

  Scenario: Type checking passes
    Tool: Bash (mypy)
    Preconditions: extract.py migrated
    Steps:
      1. Run: uv run mypy src/steering_geometry/extract.py
    Expected Result: Success: no issues found
    Failure Indicators: Any type error
    Evidence: .sisyphus/evidence/task-3-mypy.txt
  ```

  **Commit**: YES
  - Message: `refactor: migrate extract.py from print() to logging`
  - Files: `src/steering_geometry/extract.py`
  - Pre-commit: `uv run ruff check src/steering_geometry/extract.py && uv run mypy src/steering_geometry/extract.py && uv run pytest`

- [x] 4. Migrate apply_steering.py from print() to logging

  **What to do**:
  - 在文件顶部添加 `import logging` 和 `logger = logging.getLogger(__name__)`
  - 在 `_build_parser()` 中添加 `--log-level` 参数（同 Task 3 模式）
  - 在 `main()` 函数开头调用 `configure_logging(level=args.log_level)`
  - 将所有 17 个 `print()` 替换为对应的 `logger.info()` / `logger.warning()` / `logger.error()` 调用
  - 特别注意 `apply_steering()` 是公共库函数，其中的 print 必须全部迁移
  - 使用 lazy `%s` 格式

  **Must NOT do**:
  - 不使用 f-string 日志格式
  - 不改变任何业务逻辑
  - 不修改 JudgeEvaluator / MMLUEvaluator 的评估逻辑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 17 处替换，apply_steering.py 是最大最复杂的模块（1632 行），需要理解上下文
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 5, 6, 7, 8)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py` — 黄金标准 logging 模式
  - `src/steering_geometry/extract.py` (after Task 3) — 同项目的迁移范例

  **API/Type References**:
  - `src/steering_geometry/utils.py:configure_logging()` — 配置函数
  - `src/steering_geometry/apply_steering.py:_build_parser()` — 添加 --log-level
  - `src/steering_geometry/apply_steering.py:main()` — 调用 configure_logging

  **WHY Each Reference Matters**:
  - apply_steering.py 是最长的文件，需要仔细识别每个 print 的上下文来分配正确的日志级别

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No print() remains in apply_steering.py
    Tool: Bash (grep)
    Preconditions: Migration complete
    Steps:
      1. Run: grep -n "print(" src/steering_geometry/apply_steering.py
      2. Assert: no output
    Expected Result: grep returns exit code 1
    Failure Indicators: Any print() found
    Evidence: .sisyphus/evidence/task-4-no-print.txt

  Scenario: --log-level flag present
    Tool: Bash
    Preconditions: Migration complete
    Steps:
      1. Run: uv run python -m steering_geometry.apply_steering --help 2>&1 | grep -q "\-\-log-level"
    Expected Result: Flag found
    Evidence: .sisyphus/evidence/task-4-cli-help.txt

  Scenario: Existing tests still pass
    Tool: Bash (pytest)
    Preconditions: Migration complete
    Steps:
      1. Run: uv run pytest
      2. Assert all tests pass
    Expected Result: All existing tests pass (no regression)
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-4-test-results.txt
  ```

  **Commit**: YES
  - Message: `refactor: migrate apply_steering.py from print() to logging`
  - Files: `src/steering_geometry/apply_steering.py`
  - Pre-commit: `uv run ruff check src/steering_geometry/apply_steering.py && uv run mypy src/steering_geometry/apply_steering.py && uv run pytest`

- [x] 5. Migrate tdnv.py from print() to logging

  **What to do**:
  - 在文件顶部添加 `import logging` 和 `logger = logging.getLogger(__name__)`
  - 在 `_build_parser()` 中添加 `--log-level` 参数（同 Task 3 模式）
  - 在 `main()` 函数开头调用 `configure_logging(level=args.log_level)`
  - 将所有 25 个 `print()` 替换为对应的 `logger.info()` / `logger.debug()` / `logger.warning()` / `logger.error()` 调用
  - 注意区分：库函数中的 print（如 `compute_tdnv_for_concept()`, `compute_tdnv_for_mmlu()`）中的为 `info`，调试信息为 `debug`
  - `"Dry run complete"` 类状态消息用 `info`
  - 使用 lazy `%s` 格式

  **Must NOT do**:
  - 不使用 f-string 日志格式
  - 不改变任何业务逻辑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 25 处替换（最多），需要理解 TDNV 计算上下文来分配级别
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 4, 6, 7, 8)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py` — 黄金标准 logging 模式

  **API/Type References**:
  - `src/steering_geometry/utils.py:configure_logging()` — 配置函数
  - `src/steering_geometry/tdnv.py:_build_parser()` — 添加 --log-level
  - `src/steering_geometry/tdnv.py:main()` — 调用 configure_logging

  **WHY Each Reference Matters**:
  - tdnv.py 有最多 print，需逐个判断日志级别

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No print() remains in tdnv.py
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n "print(" src/steering_geometry/tdnv.py
      2. Assert: no output
    Expected Result: grep returns exit code 1
    Evidence: .sisyphus/evidence/task-5-no-print.txt

  Scenario: --log-level flag present
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.tdnv --help 2>&1 | grep -q "\-\-log-level"
    Expected Result: Flag found
    Evidence: .sisyphus/evidence/task-5-cli-help.txt
  ```

  **Commit**: YES
  - Message: `refactor: migrate tdnv.py from print() to logging`
  - Files: `src/steering_geometry/tdnv.py`
  - Pre-commit: `uv run ruff check src/steering_geometry/tdnv.py && uv run mypy src/steering_geometry/tdnv.py && uv run pytest`

- [x] 6. Migrate __main__.py from print() to logging

  **What to do**:
  - 在文件顶部添加 `import logging` 和 `logger = logging.getLogger(__name__)`
  - **仅迁移第 33 行的 stderr 错误**：将 `print("Error: ...", file=sys.stderr)` 替换为 `logger.error(...)`
  - **保留第 40-42 行的 shell eval 输出**：`ALL_MODELS=(...)`, `ALL_CONCEPTS=(...)`, `DEFAULT_MODEL="..."` 这些必须保持为 `print()` — 它们被 shell 脚本通过 `eval $(uv run python -m steering_geometry --shell)` 消费
  - 在 main() 的非 shell 分支中调用 `configure_logging()`（仅在非 --shell 模式下）

  **Must NOT do**:
  - ❌ 绝对不能迁移第 40-42 行的 shell eval print() — shell 脚本依赖这些
  - 不给 __main__.py 添加 --log-level（它只有 --shell 模式，不需要日志级别控制）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 仅 1 行 print 需迁移，3 行需保留，变更量极小
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 4, 5, 7, 8)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/steering_geometry/__main__.py:33` — `print("Error: ...", file=sys.stderr)` → `logger.error("...")`
  - `src/steering_geometry/__main__.py:40-42` — shell eval 输出，必须保持为 print()

  **API/Type References**:
  - `src/steering_geometry/utils.py:configure_logging()` — 在 main() 非分支中调用

  **WHY Each Reference Matters**:
  - `__main__.py` 有两种 print：shell eval（保留）和错误输出（迁移），必须区分
  - Shell 脚本如 `scripts/pipeline/run_pipeline.sh` 依赖 eval 输出

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Shell eval output unchanged
    Tool: Bash (python)
    Preconditions: __main__.py migrated
    Steps:
      1. Run: uv run python -m steering_geometry --shell 2>&1
      2. Assert output contains exactly:
         ALL_MODELS=(...)
         ALL_CONCEPTS=(...)
         DEFAULT_MODEL="..."
      3. Assert output does NOT contain timestamp or "INFO" prefix
    Expected Result: Shell eval output identical to pre-migration
    Failure Indicators: Output changed, contains logging prefix, or missing variables
    Evidence: .sisyphus/evidence/task-6-shell-eval.txt

  Scenario: Error case uses logger instead of print
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n "print(" src/steering_geometry/__main__.py
      2. Assert: only 3 remaining prints (shell eval lines 40-42)
      3. Run: grep -n "logger" src/steering_geometry/__main__.py
      4. Assert: logger.error present
    Expected Result: 3 prints remain (shell eval), 1 logger.error added
    Evidence: .sisyphus/evidence/task-6-no-extra-print.txt
  ```

  **Commit**: YES (groups with Task 7)
  - Message: `refactor: migrate __main__.py and token_analysis.py to logging`
  - Files: `src/steering_geometry/__main__.py`
  - Pre-commit: `uv run ruff check src/steering_geometry/__main__.py && uv run mypy src/steering_geometry/__main__.py`

- [x] 7. Migrate token_analysis.py from print() to logging

  **What to do**:
  - 文件已有 `import logging` 和 `logger = logging.getLogger(__name__)` — 无需添加
  - 在 `_build_parser()` 中添加 `--log-level` 参数
  - 在 `main()` 函数开头调用 `configure_logging(level=args.log_level)`
  - **删除第 652 行的重复 print**（与第 651 行 logger 重复）
  - **保留 CLI 格式化表格输出**：如 `print(f"\n=== Layer {layer} ===")`, `print(f"  {i:3d}. '{token.token_text}'...")` 等 — 这些是 CLI 展示用途，不适合日志
  - 将其余信息性 print（非表格格式化）迁移为 logger 调用
  - 使用 lazy `%s` 格式

  **Must NOT do**:
  - ❌ 不迁移 CLI 格式化表格输出（section headers、numbered lists、aligned metrics）
  - 不改变已有的 `logger = logging.getLogger(__name__)` 模式
  - 不使用 f-string 日志格式

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要区分 CLI 格式化输出（保留）和信息性输出（迁移），12 处 print 需逐个判断
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 4, 5, 6, 8)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py` — 黄金标准 logging 模式
  - `src/steering_geometry/token_analysis.py:41` — 已有 `logger = logging.getLogger(__name__)`
  - `src/steering_geometry/token_analysis.py:651-652` — 重复行（logger + print 同一消息），删除 652

  **API/Type References**:
  - `src/steering_geometry/utils.py:configure_logging()` — 在 main() 中调用
  - `src/steering_geometry/token_analysis.py:_build_parser()` — 添加 --log-level

  **WHY Each Reference Matters**:
  - token_analysis.py 是混合使用 logger 和 print 的典型文件，需要精确判断哪些保留哪些迁移
  - 重复行是需要清理的技术债务

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Only CLI table prints remain, no duplicate
    Tool: Bash (grep)
    Steps:
      1. Run: grep -n "print(" src/steering_geometry/token_analysis.py
      2. Assert: remaining prints are only CLI formatting (=== headers, numbered lists, aligned metrics)
      3. Run: grep -n "logger\." src/steering_geometry/token_analysis.py | wc -l
      4. Assert: logger call count increased from 14 to >= 20
    Expected Result: Only CLI table prints remain, all info prints migrated
    Failure Indicators: Info-level prints still using print()
    Evidence: .sisyphus/evidence/task-7-remaining-prints.txt

  Scenario: --log-level flag present
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.token_analysis --help 2>&1 | grep -q "\-\-log-level"
    Expected Result: Flag found
    Evidence: .sisyphus/evidence/task-7-cli-help.txt
  ```

  **Commit**: YES (groups with Task 6)
  - Message: `refactor: migrate __main__.py and token_analysis.py to logging`
  - Files: `src/steering_geometry/token_analysis.py`
  - Pre-commit: `uv run ruff check src/steering_geometry/token_analysis.py && uv run mypy src/steering_geometry/token_analysis.py`

- [x] 8. Migrate unembed_analysis.py — replace basicConfig with configure_logging

  **What to do**:
  - 文件已有 `import logging` 和 `logger = logging.getLogger(__name__)` — 无需添加
  - **替换 `logging.basicConfig()` 调用**（约在第 512-515 行的 main() 中）为 `configure_logging(level=args.log_level)`
  - 在 `_build_parser()` 中添加 `--log-level` 参数
  - 检查是否有其他 print() 需要迁移（如有则迁移，但预期此文件已使用 logger）
  - 使用 lazy `%s` 格式

  **Must NOT do**:
  - 不保留任何 `logging.basicConfig()` 调用
  - 不使用 f-string 日志格式

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 主要是替换一个 basicConfig() 调用 + 添加 --log-level，变更量小
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 4, 5, 6, 7)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/steering_geometry/stability_comparison.py` — 黄金标准

  **API/Type References**:
  - `src/steering_geometry/unembed_analysis.py:512-515` — 需替换的 `logging.basicConfig()` 调用
  - `src/steering_geometry/utils.py:configure_logging()` — 替换目标

  **WHY Each Reference Matters**:
  - `basicConfig()` 与新的 `configure_logging()` 会冲突，必须彻底替换

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No logging.basicConfig() remains
    Tool: Bash (grep)
    Steps:
      1. Run: grep -rn "logging.basicConfig" src/steering_geometry/
      2. Assert: no output (zero matches across all files)
    Expected Result: grep returns exit code 1
    Failure Indicators: Any basicConfig() found
    Evidence: .sisyphus/evidence/task-8-no-basicconfig.txt

  Scenario: --log-level flag present
    Tool: Bash
    Steps:
      1. Run: uv run python -m steering_geometry.unembed_analysis --help 2>&1 | grep -q "\-\-log-level"
    Expected Result: Flag found
    Evidence: .sisyphus/evidence/task-8-cli-help.txt
  ```

  **Commit**: YES
  - Message: `refactor: migrate unembed_analysis.py to configure_logging()`
  - Files: `src/steering_geometry/unembed_analysis.py`
  - Pre-commit: `uv run ruff check src/steering_geometry/unembed_analysis.py && uv run mypy src/steering_geometry/unembed_analysis.py`

- [x] 9. Update documentation (AGENTS.md + QUALITY_SCORE.md)

  **What to do**:
  - 更新 `AGENTS.md` 技术债务表：
    - 删除或标记已解决的 `print()` 条目：`extract.py`, `tdnv.py`, `token_analysis.py`, `apply_steering.py` 的 "print() for CLI" 行
    - 在 Section 9 "Where to Look" 中添加 `utils.py` 的 `configure_logging()` 说明
  - 更新 `docs/QUALITY_SCORE.md`：更新测试计数（新增的 logging 测试数量）
  - 验证 `.gitignore` 中已有 `logs/` 条目（Task 2 添加的）

  **Must NOT do**:
  - 不添加新的文档文件
  - 不修改 README.md（除非有明确的 print 相关引用）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯文档更新，变更量小
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after all migrations)
  - **Blocks**: FINAL
  - **Blocked By**: Tasks 3-8 (all migrations must complete first)

  **References**:

  **Pattern References**:
  - `AGENTS.md` Section 10 "Anti-Patterns" → "Current Technical Debt" 表 — 需更新的位置
  - `AGENTS.md` Section 9 "Where to Look" — 添加 configure_logging 条目
  - `docs/QUALITY_SCORE.md` — 测试计数更新

  **WHY Each Reference Matters**:
  - AGENTS.md 是 AI agent 的工作指南，必须反映最新状态
  - QUALITY_SCORE.md 跟踪测试覆盖，需更新以反映新增测试

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: AGENTS.md tech debt updated
    Tool: Bash (grep)
    Steps:
      1. Run: grep -c "print() for CLI" AGENTS.md
      2. Assert: count decreased from original (4 entries removed or marked resolved)
      3. Run: grep "configure_logging" AGENTS.md
      4. Assert: configure_logging mentioned in "Where to Look" section
    Expected Result: Tech debt table reflects completed migration
    Failure Indicators: Old print() entries still present as unresolved
    Evidence: .sisyphus/evidence/task-9-agents-md.txt
  ```

  **Commit**: YES (includes Task 2 .gitignore change)
  - Message: `chore: add logs/ to .gitignore, update docs for logging migration`
  - Files: `.gitignore`, `AGENTS.md`, `docs/QUALITY_SCORE.md`
  - Pre-commit: `uv run pytest`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Review all changed files for: `as any`/type ignore, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration: run `uv run python -m steering_geometry.extract --concept sentiment --model Qwen/Qwen3-1.7B --log-level DEBUG` and verify log file appears in logs/ with timestamp. Verify `uv run python -m steering_geometry --shell` still works. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes. Specifically verify: `__main__.py` shell eval prints preserved, `token_analysis.py` table prints preserved, no `logging.basicConfig()` usage, no f-string logging.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1**: `test: add logging configuration tests` - tests/unit/test_logging.py
- **2**: `feat: add configure_logging() to utils.py` - src/steering_geometry/utils.py, tests/unit/test_logging.py (all pass)
- **3**: `refactor: migrate extract.py from print() to logging` - src/steering_geometry/extract.py
- **4**: `refactor: migrate apply_steering.py from print() to logging` - src/steering_geometry/apply_steering.py
- **5**: `refactor: migrate tdnv.py from print() to logging` - src/steering_geometry/tdnv.py
- **6**: `refactor: migrate __main__.py and token_analysis.py to logging` - src/steering_geometry/__main__.py, src/steering_geometry/token_analysis.py
- **7**: `refactor: migrate unembed_analysis.py to configure_logging()` - src/steering_geometry/unembed_analysis.py
- **8**: `chore: add logs/ to .gitignore, update docs` - .gitignore, AGENTS.md

---

## Success Criteria

### Verification Commands
```bash
uv run pytest                    # Expected: all tests pass (including new logging tests)
uv run mypy src/                 # Expected: 0 errors
uv run ruff check src/ tests/    # Expected: 0 violations
uv run ruff format --check src/ tests/  # Expected: already formatted
grep -rn "print(" src/steering_geometry/ | grep -v "# shell eval" | grep -v "# CLI table"
# Expected: only __main__.py shell eval prints + token_analysis.py CLI table prints remain
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass (21 existing + new logging tests)
- [ ] Log files created in `logs/` with correct naming pattern
- [ ] `--log-level` works on all 5 CLI modules
- [ ] `uv run python -m steering_geometry --shell` output unchanged
