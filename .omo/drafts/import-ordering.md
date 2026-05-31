# Draft: Import Ordering Enforcement (Reusable Template)

## Requirements (confirmed)
- **Python 代码中 import 排序**: 确保 .py 文件中 import 按规范排序（stdlib → third-party → local）
- **CI/Pre-commit 自动化**: 在提交和 CI 流程中自动检查/修复 import 排序
- **可复用模板**: 做成可复制到其他 Python 项目的通用配置

## Research Findings

### Current State (Steering_Geometry)
- **Ruff isort**: 已在 `pyproject.toml` 中启用 `select = ["E", "F", "I", ...]`
- **当前违规**: `ruff check --select I` → "All checks passed!" — 无现有违规
- **Pre-commit**: 不存在 `.pre-commit-config.yaml`
- **CI workflow**: 未找到 GitHub Actions 或其他 CI 配置
- **已知问题**: `extract.py` 有 `E402` per-file-ignore（module level import not at top of file），因为 `warnings.filterwarnings()` 被插在 stdlib imports 之间

### Observed Import Patterns
- `models.py`: 标准 stdlib → third-party → local ✅
- `apply_steering.py`: 标准 stdlib → third-party → local ✅
- `extract.py`: 有 `warnings.filterwarnings()` 夹在 stdlib imports 之间，导致需要 E402 豁免 ⚠️

### Ruff Isort Config (Current - MINIMAL)
```toml
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "N"]

[tool.ruff.lint.per-file-ignores]
"src/steering_geometry/extract.py" = ["E402"]
```
**缺失**: 无 `isort.known-first-party`, `isort.section-order`, `isort.force-*` 等精细化配置

## Technical Decisions (confirmed)
- [x] 修复 extract.py 中的 E402 问题（重构 import 结构，移除 per-file-ignores）
- [x] 使用 pre-commit 框架
- [x] 包含 GitHub Actions CI workflow
- [x] 模板形式：Cookiecutter 模板

## Scope Boundaries
- INCLUDE:
  - ruff isort 精细化配置（known-first-party, section-order 等）
  - pre-commit 集成（.pre-commit-config.yaml）
  - GitHub Actions CI workflow
  - Cookiecutter 可复用模板
  - 修复 extract.py 的 E402 问题
  - AGENTS.md 更新（新增 import 排序相关指引）
- EXCLUDE:
  - 修改 import 的业务逻辑
  - 添加新的 ruff lint 规则（非 isort 相关）
  - 非 ruff 的 import 排序工具（如 standalone isort）

## Test Strategy Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: NO (config/tooling change, no new Python code)
- **Agent-Executed QA**: YES — verify ruff check passes, pre-commit runs, CI workflow syntax valid
