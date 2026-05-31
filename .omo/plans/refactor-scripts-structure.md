# Plan: Refactor Scripts Directory Structure

## TL;DR

> **Quick Summary**: 将 scripts/ 目录重构为每个实验有独立子目录的结构，确保每个实验都有对应的 shell 脚本可以前台运行。
> 
> **Deliverables**: 
> - 7 个实验子目录 (extract/, apply_steering/, tdnv/, unembed_analysis/, vector_analysis/, token_analysis/, stability_comparison/)
> - pipeline/ 目录存放完整流程脚本
> - 保留 quick/ 子目录
> - 更新文档引用
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Wave 1 (Research) → Wave 2 (Create + Move) → Wave 3 (Document + Verify)

---

## Context

### Original Request
用户希望：
1. 总结目前代码实现的功能
2. 每个实验代码放到一个独立文件（保留公共 utils.py）
3. src 下每个 py 对应一个实验（除了 utils）
4. scripts 下每个实验有对应的 sub目录存放 sh 启动文件
5. sh 文件必须能实际运行代码（前台运行，不后台输出）

### Interview Summary
**Key Discussions**:
- evaluation.py 保留为独立模块，被 apply_steering 导入使用
- 旧脚本移动到对应子目录
- 确认 7 个实验模块完整

**Research Findings**:
- 现有脚本: run_pipeline.sh, run_discriminative.sh, run_tdnv.sh, run_weighted_mean.sh, run_unembed_analysis.sh
- 现有目录: scripts/quick/, scripts/vector_analysis/
- 需创建目录: extract/, apply_steering/, tdnv/, unembed_analysis/, token_analysis/, stability_comparison/, pipeline/
- Python 模块确认存在: extract.py, apply_steering.py, tdnv.py, unembed_analysis.py, vector_analysis.py, token_analysis.py, stability_comparison.py

### Metis Review
**Identified Gaps** (addressed):
- 脚本如何调用对应的 Python 模块需要验证（通过 grep 搜索解决）
- 现有 vector_analysis/ 目录处理方式需要决策（保留并更新）
- 文档引用路径需要更新

---

## Work Objectives

### Core Objective
重构 scripts/ 目录，使每个实验有独立的子目录和对应的 shell 脚本，所有脚本能前台运行。

### Concrete Deliverables
- [ ] scripts/extract/ 目录 (转向向量提取)
- [ ] scripts/apply_steering/ 目录 (应用转向)
- [ ] scripts/tdnv/ 目录 (TDNV 分析)
- [ ] scripts/unembed_analysis/ 目录 (反嵌入分析)
- [ ] scripts/vector_analysis/ 目录 (向量稳定性)
- [ ] scripts/token_analysis/ 目录 (Token 分析)
- [ ] scripts/stability_comparison/ 目录 (稳定性对比)
- [ ] scripts/pipeline/ 目录 (完整流程)
- [ ] 更新 AGENTS.md 中的脚本路径引用

### Definition of Done
- [ ] `ls scripts/` 显示新的目录结构
- [ ] 每个子目录下的脚本都能运行 `--help`
- [ ] 文档中的路径引用已更新
- [ ] 所有脚本使用前台运行方式（无后台 `&`）

### Must Have
- 每个实验有对应的 shell 脚本
- 脚本必须能实际运行（调用正确的 Python 模块）
- 保留 scripts/quick/ 目录

### Must NOT Have (Guardrails)
- 不修改 Python 模块代码逻辑
- 不修改模块间的依赖关系
- 不删除任何现有的功能
- 不在脚本中添加后台运行 (`&`)
- 不添加新功能

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Shell Scripts**: Use Bash — Run `--help`, check exit code, verify module import
- **Directory Structure**: Use Bash — `ls -la`, `find`, verify structure

---

## Execution Strategy

### Parallel Execution Waves

> Maximize throughput by grouping independent tasks into parallel waves.
> Each wave completes before the next begins.

```
Wave 1 (Start Immediately — research + verification):
├── Task 1: Map scripts to Python modules [quick]
└── Task 2: Document current script locations [quick]

Wave 2 (After Wave 1 — create structure):
├── Task 3: Create target directory structure [quick]

Wave 3 (After Wave 2 — move scripts):
├── Task 4: Move extraction scripts [quick]
├── Task 5: Move pipeline script [quick]
├── Task 6: Move tdnv script [quick]
├── Task 7: Move unembed script [quick]
├── Task 8: Handle vector_analysis/ [quick]
├── Task 9: Create apply_steering script [quick]

Wave 4 (After Wave 3 — documentation):
├── Task 10: Update documentation references [quick]

Wave FINAL (After ALL tasks — verification):
├── Task F1: Plan compliance audit [oracle]
├── Task F2: Script functionality verification [unspecified-high]
├── Task F3: Directory structure verification [quick]
└── Task F4: Commit changes [quick]

Critical Path: Task 1,2 → Task 3 → Task 4-9 → Task 10 → F1-F4 → commit
Parallel Speedup: ~50% faster than sequential
Max Concurrent: 6 (Wave 3)
```

### Dependency Matrix

- **1-2**: — — 3, 4-9
- **3**: 1, 2 — 4, 5, 6, 7, 8, 9, 2
- **4-9**: 3 — 10, 1
- **10**: 4, 5, 6, 7, 8, 9 — F1-F4, 2

### Agent Dispatch Summary

- **Wave 1**: 2 tasks — both `quick`
- **Wave 2**: 1 task — `quick`
- **Wave 3**: 6 tasks — all `quick` (with git-master for moves)
- **Wave 4**: 1 task — `quick`
- **FINAL**: 4 tasks — F1 → `oracle`, F2-F4 → `quick`

---

## TODOs

- [ ] 1. **Map Scripts to Python Modules**
  
  **What to do**:
  - 在所有 .sh 文件中搜索 `uv run python -m` 模式
  - 构建脚本到 Python 模块的映射表
  - 验证每个 Python 模块是否存在
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 简单的文件搜索和分析任务
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3
  - **Blocked By**: None (can start immediately)
  
  **References**:
  - `scripts/*.sh` - 所有 shell 脚本
  - `src/steering_geometry/*.py` - Python 模块
  
  **Acceptance Criteria**:
  - [ ] 输出完整的映射表
  - [ ] 验证所有 Python 模块都存在
  
  **QA Scenarios**:
  ```
  Scenario: Verify all Python modules exist
    Tool: Bash
    Steps:
      1. ls src/steering_geometry/*.py | grep -v __pycache__ | grep -v __init__
      2. For each script, check if the module it calls exists
    Expected Result: All modules exist
    Failure Indicators: Module file not found
    Evidence: .omo/evidence/task-01-module-verification.txt
  ```

- [ ] 2. **Document Current Script Locations**
  
  **What to do**:
  - 运行 `find scripts/ -name "*.sh" -type f`
  - 记录每个脚本的当前路径
  - 确认 scripts/quick/ 和 scripts/vector_analysis/ 子目录
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 简单的文件系统操作
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3
  - **Blocked By**: None (can start immediately)
  
  **References**:
  - `scripts/` - scripts 目录
  
  **Acceptance Criteria**:
  - [ ] 完整的脚本列表
  - [ ] 确认哪些需要移动
  
  **QA Scenarios**:
  ```
  Scenario: Verify script inventory
    Tool: Bash
    Steps:
      1. find scripts/ -name "*.sh" -type f | sort
      2. Count total scripts
    Expected Result: Complete list with ~10 scripts
    Failure Indicators: Missing scripts
    Evidence: .omo/evidence/task-02-script-inventory.txt
  ```

- [ ] 3. **Create Target Directory Structure**
  
  **What to do**:
  - 创建以下目录: extract/, apply_steering/, tdnv/, unembed_analysis/, token_analysis/, stability_comparison/, pipeline/
  - 确保目录结构正确
  - 保留 scripts/quick/ 不变
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 简单的目录创建
  
  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 4, 5, 6, 7, 8, 9
  - **Blocked By**: Tasks 1, 2
  
  **References**:
  - `scripts/` - 目标目录
  
  **Acceptance Criteria**:
  - [ ] `ls scripts/` 显示所有新目录
  - [ ] scripts/quick/ 保留不变
  
  **QA Scenarios**:
  ```
  Scenario: Verify directory structure
    Tool: Bash
    Steps:
      1. ls -la scripts/ | grep "^d"
      2. Verify extract, apply_steering, tdnv, unembed_analysis, token_analysis, stability_comparison, pipeline, quick directories exist
    Expected Result: 8 directories present (including quick/)
    Failure Indicators: Missing directories
    Evidence: .omo/evidence/task-03-directory-structure.txt
  ```

- [ ] 4. **Move Extraction Scripts**
  
  **What to do**:
  - 移动 run_discriminative.sh 到 scripts/extract/
  - 移动 run_weighted_mean.sh 到 scripts/extract/
  - 创建 scripts/extract/run_extract.sh (通用提取脚本)
  - 更新脚本内部路径（SCRIPT_DIR 计算）
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`git-master`]
  - **Reason**: 需要 git mv 操作
  - `git-master`: 用于安全的文件移动和跟踪
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 5,6,7,8,9)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 10
  - **Blocked By**: Task 3
  
  **References**:
  - `scripts/run_discriminative.sh` - 源脚本
  - `scripts/run_weighted_mean.sh` - 源脚本
  - `src/steering_geometry/extract.py` - Python 模块
  
  **Must NOT do**:
  - 修改脚本的逻辑功能
  - 添加后台运行
  
  **Acceptance Criteria**:
  - [ ] 脚本已移动到正确位置
  - [ ] `./scripts/extract/run_discriminative.sh --help` 正常工作
  - [ ] `./scripts/extract/run_weighted_mean.sh --help` 正常工作
  
  **QA Scenarios**:
  ```
  Scenario: Verify extraction scripts work
    Tool: Bash
    Steps:
      1. cd scripts/extract && ./run_discriminative.sh --help
      2. Verify output shows usage info
      3. ./run_weighted_mean.sh --help
      4. Verify output shows usage info
    Expected Result: Both scripts show help text
    Failure Indicators: Script not found or error
    Evidence: .omo/evidence/task-04-extraction-scripts.txt
  ```

- [ ] 5. **Move Pipeline Script**
  
  **What to do**:
  - 移动 run_pipeline.sh 到 scripts/pipeline/
  - 更新脚本内部路径
  - 更新 usage 文档中的示例路径
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`git-master`]
  - **Reason**: 需要 git mv 操作
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4,6,7,8,9)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 10
  - **Blocked By**: Task 3
  
  **References**:
  - `scripts/run_pipeline.sh` - 源脚本
  - `src/steering_geometry/extract.py` - 被调用的模块
  - `src/steering_geometry/apply_steering.py` - 被调用的模块
  
  **Acceptance Criteria**:
  - [ ] `./scripts/pipeline/run_pipeline.sh --help` 正常工作
  
  **QA Scenarios**:
  ```
  Scenario: Verify pipeline script works
    Tool: Bash
    Steps:
      1. cd scripts/pipeline && ./run_pipeline.sh --help
      2. Verify output shows usage info with correct paths
    Expected Result: Script shows help text
    Failure Indicators: Script not found or path errors
    Evidence: .omo/evidence/task-05-pipeline-script.txt
  ```

- [ ] 6. **Move TDNV Script**
  
  **What to do**:
  - 移动 run_tdnv.sh 到 scripts/tdnv/
  - 更新脚本内部路径
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`git-master`]
  - **Reason**: 需要 git mv 操作
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4,5,7,8,9)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 10
  - **Blocked By**: Task 3
  
  **References**:
  - `scripts/run_tdnv.sh` - 源脚本
  - `src/steering_geometry/tdnv.py` - Python 模块
  
  **Acceptance Criteria**:
  - [ ] `./scripts/tdnv/run_tdnv.sh --help` 正常工作
  
  **QA Scenarios**:
  ```
  Scenario: Verify tdnv script works
    Tool: Bash
    Steps:
      1. cd scripts/tdnv && ./run_tdnv.sh --help
      2. Verify output shows usage info
    Expected Result: Script shows help text
    Failure Indicators: Script not found
    Evidence: .omo/evidence/task-06-tdnv-script.txt
  ```

- [ ] 7. **Move Unembed Script**
  
  **What to do**:
  - 移动 run_unembed_analysis.sh 到 scripts/unembed_analysis/
  - 更新脚本内部路径
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`git-master`]
  - **Reason**: 需要 git mv 操作
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4,5,6,8,9)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 10
  - **Blocked By**: Task 3
  
  **References**:
  - `scripts/run_unembed_analysis.sh` - 源脚本
  - `src/steering_geometry/unembed_analysis.py` - Python 模块
  
  **Acceptance Criteria**:
  - [ ] `./scripts/unembed_analysis/run_unembed_analysis.sh --help` 正常工作
  
  **QA Scenarios**:
  ```
  Scenario: Verify unembed script works
    Tool: Bash
    Steps:
      1. cd scripts/unembed_analysis && ./run_unembed_analysis.sh --help
      2. Verify output shows usage info
    Expected Result: Script shows help text
    Failure Indicators: Script not found
    Evidence: .omo/evidence/task-07-unembed-script.txt
  ```

- [ ] 8. **Handle vector_analysis/ Directory**
  
  **What to do**:
  - 检查 scripts/vector_analysis/ 目录是否存在
  - 如果存在，保留并验证脚本正确性
  - 如果不存在，创建目录和对应的脚本
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 条件性创建/验证
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4,5,6,7,9)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 10
  - **Blocked By**: Task 3
  
  **References**:
  - `scripts/vector_analysis/` - 目标目录
  - `src/steering_geometry/vector_analysis.py` - Python 模块
  
  **Acceptance Criteria**:
  - [ ] scripts/vector_analysis/ 目录存在
  - [ ] 脚本可以运行
  
  **QA Scenarios**:
  ```
  Scenario: Verify vector_analysis directory
    Tool: Bash
    Steps:
      1. ls -la scripts/vector_analysis/
      2. If scripts exist, run --help on each
    Expected Result: Directory exists with working scripts
    Failure Indicators: Directory missing
    Evidence: .omo/evidence/task-08-vector-analysis-dir.txt
  ```

- [ ] 9. **Create apply_steering Script**
  
  **What to do**:
  - 创建 scripts/apply_steering/ 目录
  - 创建 run_steering.sh 脚本
  - 脚本应调用 `uv run python -m steering_geometry.apply_steering`
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 基于模板创建新脚本
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4,5,6,7,8)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 10
  - **Blocked By**: Task 3
  
  **References**:
  - `src/steering_geometry/apply_steering.py` - Python 模块
  - `scripts/extract/run_discriminative.sh` - 模板参考
  
  **Acceptance Criteria**:
  - [ ] scripts/apply_steering/run_steering.sh 存在
  - [ ] `./scripts/apply_steering/run_steering.sh --help` 正常工作
  
  **QA Scenarios**:
  ```
  Scenario: Verify apply_steering script works
    Tool: Bash
    Steps:
      1. cd scripts/apply_steering && ./run_steering.sh --help
      2. Verify output shows usage info
    Expected Result: Script shows help text
    Failure Indicators: Script not found
    Evidence: .omo/evidence/task-09-apply-steering-script.txt
  ```

- [ ] 10. **Create token_analysis Scripts**
  
  **What to do**:
  - 创建 scripts/token_analysis/ 目录
  - 创建 run_visualize.sh 脚本（调用 visualize 子命令）
  - 创建 run_probe.sh 脚本（调用 probe 子命令）
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 基于模板创建新脚本
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4,5,6,7,8,9)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 11
  - **Blocked By**: Task 3
  
  **References**:
  - `src/steering_geometry/token_analysis.py` - Python 模块（有 visualize 和 probe 子命令）
  
  **Acceptance Criteria**:
  - [ ] scripts/token_analysis/run_visualize.sh 存在
  - [ ] scripts/token_analysis/run_probe.sh 存在
  
  **QA Scenarios**:
  ```
  Scenario: Verify token_analysis scripts work
    Tool: Bash
    Steps:
      1. cd scripts/token_analysis
      2. ./run_visualize.sh --help
      3. ./run_probe.sh --help
    Expected Result: Both scripts show help text
    Failure Indicators: Scripts not found
    Evidence: .omo/evidence/task-10-token-analysis-scripts.txt
  ```

- [ ] 11. **Create stability_comparison Script**
  
  **What to do**:
  - 创建 scripts/stability_comparison/ 目录
  - 创建 run_stability.sh 脚本
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 基于模板创建新脚本
  
  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4,5,6,7,8,9,10)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 12
  - **Blocked By**: Task 3
  
  **References**:
  - `src/steering_geometry/stability_comparison.py` - Python 模块
  
  **Acceptance Criteria**:
  - [ ] scripts/stability_comparison/run_stability.sh 存在
  - [ ] `./scripts/stability_comparison/run_stability.sh --help` 正常工作
  
  **QA Scenarios**:
  ```
  Scenario: Verify stability_comparison script works
    Tool: Bash
    Steps:
      1. cd scripts/stability_comparison && ./run_stability.sh --help
      2. Verify output shows usage info
    Expected Result: Script shows help text
    Failure Indicators: Script not found
    Evidence: .omo/evidence/task-11-stability-script.txt
  ```

- [ ] 12. **Update Documentation References**
  
  **What to do**:
  - 更新 AGENTS.md 中的脚本路径
  - 更新 README.md 中的脚本路径（如果有）
  - 确保所有引用指向新的目录结构
  
  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: 文档更新
  
  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4
  - **Blocks**: Final Verification
  - **Blocked By**: Tasks 4-11
  
  **References**:
  - `AGENTS.md` - 需要更新
  - `README.md` - 需要更新
  
  **Acceptance Criteria**:
  - [ ] 文档中的脚本路径已更新
  - [ ] 没有指向旧路径的引用
  
  **QA Scenarios**:
  ```
  Scenario: Verify documentation paths updated
    Tool: Bash
    Steps:
      1. grep -r "scripts/run_" AGENTS.md README.md
      2. Verify no old paths remain
      3. grep "scripts/extract/" AGENTS.md
      4. Verify new paths present
    Expected Result: No old paths, new paths present
    Failure Indicators: Old paths still referenced
    Evidence: .omo/evidence/task-12-doc-paths.txt
  ```

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE.
> Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, check directory). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Script Functionality Review** — `unspecified-high`
  Run `--help` on every script in scripts/ subdirectories. Verify each script:
  - Calls correct Python module
  - Has no background execution (`&`)
  - Has proper error handling (`set -euo pipefail`)
  - Output: `Scripts [N/N functional] | Issues [list] | VERDICT`

- [ ] F3. **Directory Structure Verification** — `quick`
  Run `find scripts/ -type d` and verify structure matches plan:
  - extract/, apply_steering/, tdnv/, unembed_analysis/, vector_analysis/, token_analysis/, stability_comparison/, pipeline/, quick/
  - Output: `Directories [N/N match] | Missing [list] | VERDICT`

- [ ] F4. **Commit Changes** — `quick` (+ `git-master`)
  After all verification passes:
  ```bash
  git add scripts/
  git commit -m "refactor(scripts): reorganize into experiment-specific subdirectories

  - Extract: run_discriminative.sh, run_weighted_mean.sh → scripts/extract/
  - Analysis: run_tdnv.sh → scripts/tdnv/, run_unembed_analysis.sh → scripts/unembed_analysis/
  - Pipeline: run_pipeline.sh → scripts/pipeline/
  - New: apply_steering/, token_analysis/, stability_comparison scripts
  - Updated: documentation references"
  ```
  Output: `Commit [hash] | Files [N] | VERDICT`

---

## Commit Strategy

- **1**: `refactor(scripts): reorganize into experiment-specific subdirectories`
  - All scripts/ changes
  - Pre-commit: Verify each script runs `--help`

---

## Success Criteria

### Verification Commands
```bash
# Verify directory structure
ls -la scripts/ | grep "^d"

# Verify each script works
./scripts/extract/run_discriminative.sh --help
./scripts/tdnv/run_tdnv.sh --help

# Verify no old paths in docs
grep -r "scripts/run_" AGENTS.md README.md
```

### Final Checklist
- [ ] All "Must Have" present (8 directories with scripts)
- [ ] All "Must NOT Have" absent (no background execution, no code changes)
- [ ] All scripts pass `--help`
- [ ] Documentation updated
