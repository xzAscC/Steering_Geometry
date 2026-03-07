# Agent Engineering Template — Python/uv Setup

## TL;DR

> **Quick Summary**: Create a complete, reusable Python template repo for AI agent engineering workflows. The template provides AGENTS.md (AI operational policy), documentation structure (design-docs, exec-plans, quality tracking), working tooling (ruff + mypy + pytest via uv), and GitHub integration (CI, PR template, YAML issue forms).
> 
> **Deliverables**:
> - AGENTS.md — concise AI operational policy (<150 lines)
> - ARCHITECTURE.md + README.md — system design template + usage guide
> - docs/ tree — design-docs, exec-plans, PLANS.md, QUALITY_SCORE.md
> - pyproject.toml — ruff + mypy + pytest configs, uv managed
> - src/my_project/ + tests/ — minimal scaffold with one passing test
> - .github/ — CI workflow + AI-aware PR template + YAML issue forms
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 3 waves + verification + final review
> **Critical Path**: Task 1 (pyproject.toml) → Task 9 (src/) → Task 10 (tests/) → Task 14 (verify all tools)

---

## Context

### Original Request
User wants to build a reusable GitHub template repo for "Agent Engineering" — a workflow where:
1. User defines tasks and writes exec plans
2. AI agent reads AGENTS.md / ARCHITECTURE.md / relevant docs
3. AI implements code → runs lint/typecheck → writes tests/docs → self-reviews
4. AI opens PR with summary and risks
5. User reviews (judge issues only) → merge
6. Periodic agent does doc-gardening + refactor + tech debt cleanup

The repo currently only has: `.git/`, empty `.gitignore`, empty `architecture.md` (lowercase), empty `README.md`.

### Interview Summary
**Key Discussions**:
- **Purpose**: Template repo (clone/fork for new projects), not a specific project
- **Stack**: Python + uv + ruff + mypy + pytest
- **AI Agent**: Claude Code (opencode) — primary consumer of AGENTS.md
- **Scope**: Full setup — docs + working tooling + GitHub integration
- **Content**: Sensible defaults for all documentation files

**Research Findings**:
- AGENTS.md should be under 150 lines — command-first, not prose (from Metis/librarian research on 2,500+ repos)
- Issue templates should use YAML forms (.yml), not markdown (.md) — structured fields are better for AI agents
- PR template needs AI-specific sections: self-review checklist, verification output, agent metadata block
- QUALITY_SCORE.md should be a tracking framework (automated gates + trend metrics + human rubric)
- AGENTS.md must include escalation rules (when to stop and report vs. improvise)

### Metis Review
**Identified Gaps** (addressed):
- YAML issue forms instead of markdown templates — adopted
- AGENTS.md line limit (<150) — enforced
- Escalation rules in AGENTS.md — included
- Definition of Done section in AGENTS.md — included
- AI-specific PR template sections — included
- Quality tracking framework structure — included
- No CLAUDE.md needed (scope creep for a template) — excluded
- No AI-specific GitHub Actions (ai-pr-review.yml) — excluded from template scope

---

## Work Objectives

### Core Objective
Create a production-ready Python template repo that enables an AI agent (Claude Code/opencode) to autonomously execute the full development cycle: read docs → implement → lint/typecheck → test → self-review → open PR.

### Concrete Deliverables
28 files total across 4 layers:
- **Root**: AGENTS.md, ARCHITECTURE.md, README.md, .gitignore, .python-version, pyproject.toml
- **Source**: src/my_project/__init__.py, py.typed, hello.py
- **Tests**: tests/__init__.py, conftest.py, test_hello.py
- **Docs**: design-docs/ (index.md, core-beliefs.md), exec-plans/ (active/, completed/, tech-debt-tracker.md), PLANS.md, QUALITY_SCORE.md
- **GitHub**: workflows/ci.yml, pull_request_template.md, ISSUE_TEMPLATE/ (config.yml, ai-task.yml, bug-report.yml, feature-request.yml)

### Definition of Done
- [ ] `uv sync` completes without errors
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → already formatted
- [ ] `uv run mypy src/` → Success with 0 errors
- [ ] `uv run pytest` → 1 test passed, 0 failures
- [ ] All 28 files exist and have non-empty content
- [ ] AGENTS.md is under 150 lines

### Must Have
- Working pyproject.toml with uv, ruff, mypy, pytest configs
- AGENTS.md with: commands section, Definition of Done, escalation rules, code style, architecture cross-refs
- CI workflow that runs ruff + mypy + pytest on push/PR
- PR template with AI-specific sections (self-review, verification output, agent metadata)
- YAML issue forms (.yml) with structured fields
- docs/ tree matching the target structure from user's screenshot
- Placeholder package name `my_project` with clear rename instructions in README

### Must NOT Have (Guardrails)
- No CLAUDE.md file — AGENTS.md is sufficient for the template
- No AI-specific workflow automation (.github/workflows/ai-pr-review.yml) — beyond template scope
- No actual application code beyond one placeholder function and its test
- No deployment configs (Docker, cloud, etc.)
- No database/infra setup
- No pre-commit hooks (user can add later if desired)
- AGENTS.md must NOT exceed 150 lines — front-load commands, minimize prose
- No `any` types in Python code — use proper typing
- No prose-heavy AGENTS.md — operational policy with verifiable commands only

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (will be created by Wave 1)
- **Automated tests**: YES (Tests-after — pytest, one test for placeholder)
- **Framework**: pytest (via uv run pytest)
- **TDD**: Not applicable — this is a template scaffold, not feature code

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Config files**: Use Bash — run tool commands, verify exit codes, check output
- **Markdown files**: Use Bash — check file exists, verify line count, grep for required sections
- **GitHub YAML**: Use Bash — validate YAML syntax, check required fields
- **Full integration**: Use Bash — run all tools end-to-end

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — ALL foundation files, MAX PARALLEL):
├── Task 1: pyproject.toml + .python-version + uv init [quick]
├── Task 2: .gitignore (Python-specific) [quick]
├── Task 3: AGENTS.md (AI operational policy) [writing]
├── Task 4: ARCHITECTURE.md (rename + populate) [writing]
├── Task 5: README.md (template usage guide) [writing]
├── Task 6: docs/design-docs/ (index.md + core-beliefs.md) [writing]
├── Task 7: docs/PLANS.md + docs/QUALITY_SCORE.md [writing]
└── Task 8: docs/exec-plans/ structure + tech-debt-tracker.md [quick]

Wave 2 (After Wave 1 — code scaffolding + GitHub, parallel):
├── Task 9: src/my_project/ package structure (depends: 1) [quick]
├── Task 10: tests/ structure + test_hello.py (depends: 1, 9) [quick]
├── Task 11: .github/workflows/ci.yml (depends: 1) [quick]
├── Task 12: .github/pull_request_template.md [writing]
└── Task 13: .github/ISSUE_TEMPLATE/ YAML forms [quick]

Wave 3 (After Wave 2 — full integration verification):
└── Task 14: Run all tools end-to-end, fix any issues (depends: 1-13) [deep]

Wave FINAL (After ALL tasks — independent review, 4 parallel):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: Task 1 → Task 9 → Task 10 → Task 14 → F1-F4
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 8 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 9, 10, 11, 14 | 1 |
| 2 | — | 14 | 1 |
| 3 | — | 14 | 1 |
| 4 | — | 14 | 1 |
| 5 | — | 14 | 1 |
| 6 | — | 14 | 1 |
| 7 | — | 14 | 1 |
| 8 | — | 14 | 1 |
| 9 | 1 | 10, 14 | 2 |
| 10 | 1, 9 | 14 | 2 |
| 11 | 1 | 14 | 2 |
| 12 | — | 14 | 2 |
| 13 | — | 14 | 2 |
| 14 | 1-13 | F1-F4 | 3 |
| F1-F4 | 14 | — | FINAL |

### Agent Dispatch Summary

- **Wave 1**: **8 tasks** — T1 → `quick`, T2 → `quick`, T3 → `writing`, T4 → `writing`, T5 → `writing`, T6 → `writing`, T7 → `writing`, T8 → `quick`
- **Wave 2**: **5 tasks** — T9 → `quick`, T10 → `quick`, T11 → `quick`, T12 → `writing`, T13 → `quick`
- **Wave 3**: **1 task** — T14 → `deep`
- **FINAL**: **4 tasks** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. Initialize Python project with uv + pyproject.toml + .python-version

  **What to do**:
  - Run `uv init` to create the project skeleton OR create `pyproject.toml` manually with proper structure
  - Set `.python-version` to `3.12`
  - Configure pyproject.toml with:
    - `[project]` section: name=`my_project`, version=`0.1.0`, requires-python `>=3.12`, description placeholder
    - `[tool.ruff]` section: target-version=`py312`, line-length=100, select rules (E, F, I, UP, B, SIM, N), src=`["src"]`
    - `[tool.ruff.format]` section: quote-style=`double`
    - `[tool.mypy]` section: python_version=`3.12`, strict=true, warn_return_any=true, warn_unused_configs=true, packages=`["my_project"]`, mypy_path=`src`
    - `[tool.pytest.ini_options]` section: testpaths=`["tests"]`, pythonpath=`["src"]`
    - `[dependency-groups]` or `[project.optional-dependencies]`: dev deps = ruff, mypy, pytest
  - Run `uv sync` to generate uv.lock and .venv

  **Must NOT do**:
  - Do NOT add application dependencies — only dev tools
  - Do NOT use poetry or pip format — uv native only
  - Do NOT set Python version below 3.12

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward config file creation with well-known structure
  - **Skills**: []
    - No special skills needed — standard pyproject.toml authoring
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not relevant — no browser interaction
    - `frontend-ui-ux`: Not relevant — backend/config work

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-8)
  - **Blocks**: Tasks 9, 10, 11, 14
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL):

  **Pattern References**:
  - No existing files to follow — this is the first file being created

  **API/Type References**:
  - uv pyproject.toml format: https://docs.astral.sh/uv/concepts/projects/config/
  - ruff configuration: https://docs.astral.sh/ruff/configuration/
  - mypy configuration: https://mypy.readthedocs.io/en/stable/config_file.html#using-a-pyproject-toml-file

  **WHY Each Reference Matters**:
  - uv docs: Ensures correct project metadata format and dependency group syntax
  - ruff docs: Correct rule selection syntax and src directory configuration
  - mypy docs: Proper strict mode settings and package path configuration

  **Acceptance Criteria**:

  - [ ] pyproject.toml exists with all required sections ([project], [tool.ruff], [tool.mypy], [tool.pytest])
  - [ ] .python-version contains `3.12`
  - [ ] `uv sync` completes with exit code 0
  - [ ] uv.lock file is generated
  - [ ] .venv directory is created

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: uv sync creates working environment
    Tool: Bash
    Preconditions: pyproject.toml and .python-version exist
    Steps:
      1. Run `uv sync` — expected: exit code 0, no errors in output
      2. Run `ls .venv/bin/python` — expected: file exists
      3. Run `uv run python --version` — expected: output contains "3.12"
      4. Run `uv run ruff --version` — expected: exits 0, shows ruff version
      5. Run `uv run mypy --version` — expected: exits 0, shows mypy version
      6. Run `uv run pytest --version` — expected: exits 0, shows pytest version
    Expected Result: All 6 commands exit 0, tools are installed and runnable
    Failure Indicators: Any command returns non-zero exit code, "not found" errors
    Evidence: .sisyphus/evidence/task-1-uv-sync.txt

  Scenario: pyproject.toml has all required sections
    Tool: Bash
    Preconditions: pyproject.toml exists
    Steps:
      1. Run `grep -c '\[project\]' pyproject.toml` — expected: 1
      2. Run `grep -c '\[tool.ruff\]' pyproject.toml` — expected: 1
      3. Run `grep -c '\[tool.mypy\]' pyproject.toml` — expected: 1
      4. Run `grep -c '\[tool.pytest' pyproject.toml` — expected: 1
      5. Run `grep 'name.*=.*"my_project"' pyproject.toml` — expected: match found
    Expected Result: All sections present, project name is my_project
    Failure Indicators: grep returns 0 matches for any section
    Evidence: .sisyphus/evidence/task-1-pyproject-sections.txt
  ```

  **Evidence to Capture:**
  - [ ] task-1-uv-sync.txt — full output of uv sync command
  - [ ] task-1-pyproject-sections.txt — grep verification output

  **Commit**: YES (groups with Wave 1)
  - Message: `feat: initialize project foundation`
  - Files: `pyproject.toml`, `.python-version`, `uv.lock`
  - Pre-commit: `uv sync`

- [ ] 2. Create Python-specific .gitignore

  **What to do**:
  - Create `.gitignore` with comprehensive Python patterns:
    - Python: `__pycache__/`, `*.py[cod]`, `*$py.class`, `*.so`
    - Virtual envs: `.venv/`, `venv/`, `env/`
    - Distribution: `dist/`, `build/`, `*.egg-info/`, `*.egg`
    - IDE: `.idea/`, `.vscode/`, `*.swp`, `*.swo`, `.DS_Store`
    - Testing: `.coverage`, `htmlcov/`, `.pytest_cache/`, `.mypy_cache/`
    - uv: `.uv/` (if applicable, check uv docs)
    - Sisyphus: `.sisyphus/evidence/`
    - Environment: `.env`, `.env.local`

  **Must NOT do**:
  - Do NOT ignore `uv.lock` — it should be committed (like package-lock.json)
  - Do NOT ignore `.python-version` — it should be committed
  - Do NOT add overly broad patterns that might catch intended files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple file creation with well-known patterns
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None relevant

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3-8)
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References**:

  **External References**:
  - GitHub's Python .gitignore template: https://github.com/github/gitignore/blob/main/Python.gitignore

  **WHY Each Reference Matters**:
  - GitHub's template covers all standard Python patterns — use as starting point

  **Acceptance Criteria**:

  - [ ] .gitignore exists and is non-empty
  - [ ] Contains `__pycache__/`, `.venv/`, `.mypy_cache/`, `.pytest_cache/`
  - [ ] Does NOT contain `uv.lock` or `.python-version`

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: .gitignore covers standard Python patterns
    Tool: Bash
    Preconditions: .gitignore exists
    Steps:
      1. Run `grep '__pycache__' .gitignore` — expected: match
      2. Run `grep '.venv' .gitignore` — expected: match
      3. Run `grep '.mypy_cache' .gitignore` — expected: match
      4. Run `grep '.pytest_cache' .gitignore` — expected: match
      5. Run `grep 'uv.lock' .gitignore` — expected: NO match (should NOT be ignored)
    Expected Result: All standard patterns present, uv.lock not ignored
    Failure Indicators: Missing essential patterns, or uv.lock found in ignore list
    Evidence: .sisyphus/evidence/task-2-gitignore-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-2-gitignore-check.txt

  **Commit**: YES (groups with Wave 1)
  - Message: `feat: initialize project foundation`
  - Files: `.gitignore`

- [ ] 3. Create AGENTS.md — AI operational policy (<150 lines)

  **What to do**:
  - Create `AGENTS.md` at project root — this is THE most critical file in the template
  - Must be under 150 lines — command-first, operational, not prose
  - Structure (follow this exact order):
    1. **Repository Snapshot** (~5 lines): Stack summary, package manager, key files
    2. **Build & Verify Commands** (~15 lines): Exact commands for install, lint, format, typecheck, test, single-test
    3. **Definition of Done** (~10 lines): All commands that must exit 0 before a PR is opened
    4. **When Writing Code** (~12 lines): Import ordering, typing rules, error handling, naming conventions
    5. **When Writing Tests** (~10 lines): Test patterns, fixtures, isolation rules, coverage
    6. **When Opening a PR** (~10 lines): What to include in description, self-review checklist reference
    7. **Escalation Rules** (~10 lines): When to STOP and report (e.g., "If modifying more than 5 files not in the plan, STOP"), when to ask for clarification, what to never do without permission
    8. **Code Style Summary** (~8 lines): What ruff enforces (defer to tooling), naming conventions, docstring style
    9. **Architecture Cross-References** (~5 lines): Point to ARCHITECTURE.md, docs/design-docs/, docs/exec-plans/
  - Total must be under 150 lines
  - Commands must be uv-based: `uv run ruff check`, `uv run mypy`, `uv run pytest`
  - Content must be Python/uv-specific, NOT generic or JavaScript

  **Must NOT do**:
  - Do NOT exceed 150 lines — this is a hard limit
  - Do NOT write prose paragraphs — use bullet points and code blocks
  - Do NOT include JavaScript/Node/Bun references (the global AGENTS.md is JS — ignore it)
  - Do NOT include agent-specific instructions (Claude-specific, Cursor-specific) — keep tool-agnostic
  - Do NOT duplicate ruff/mypy rules — say "enforced by tooling, see pyproject.toml"

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Critical content creation requiring precise structure, conciseness, and operational clarity
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `git-master`: Not needed — no git operations
    - `playwright`: Not relevant

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-2, 4-8)
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `~/.config/Claude/AGENTS.md` — The currently active AGENTS.md for reference on STRUCTURE only (content is JS-specific, ignore that). Look at section ordering and formatting style.

  **External References**:
  - Research finding: AGENTS.md files over 150 lines get truncated by context windows (from analysis of 2,500+ repos)
  - Best practice: front-load commands, minimize prose, make every line actionable

  **WHY Each Reference Matters**:
  - Global AGENTS.md: Copy the structural pattern (sections, formatting) but replace ALL content with Python/uv equivalents
  - Research finding: Hard limit on line count ensures AI agents actually read the full file

  **Acceptance Criteria**:

  - [ ] AGENTS.md exists at project root
  - [ ] File is under 150 lines (`wc -l AGENTS.md` < 150)
  - [ ] Contains sections: Repository Snapshot, Build & Verify Commands, Definition of Done, Escalation Rules
  - [ ] All commands use `uv run` prefix
  - [ ] No JavaScript/Node/Bun references
  - [ ] Contains at least 3 escalation rules

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: AGENTS.md is concise and complete
    Tool: Bash
    Preconditions: AGENTS.md exists
    Steps:
      1. Run `wc -l AGENTS.md` — expected: number < 150
      2. Run `grep -c 'uv run' AGENTS.md` — expected: >= 5 (multiple command references)
      3. Run `grep -i 'node\|npm\|bun\|yarn' AGENTS.md` — expected: NO matches
      4. Run `grep -i 'escalat' AGENTS.md` — expected: at least 1 match
      5. Run `grep -i 'definition of done' AGENTS.md` — expected: at least 1 match
      6. Run `grep -i 'STOP\|stop and' AGENTS.md` — expected: at least 1 match (escalation rule)
    Expected Result: Under 150 lines, all Python/uv commands, no JS references, has escalation rules
    Failure Indicators: Over 150 lines, contains JS references, missing escalation section
    Evidence: .sisyphus/evidence/task-3-agents-md-check.txt

  Scenario: AGENTS.md has no prose bloat
    Tool: Bash
    Preconditions: AGENTS.md exists
    Steps:
      1. Count lines that are headers, bullets, or code blocks vs. plain prose paragraphs
      2. Run `grep -c '^#\|^-\|^  -\|^\`\`\`\|^$' AGENTS.md` — expected: >70% of total lines
    Expected Result: Majority of lines are structured (headers, bullets, code) not flowing prose
    Failure Indicators: Large blocks of paragraph text
    Evidence: .sisyphus/evidence/task-3-agents-md-structure.txt
  ```

  **Evidence to Capture:**
  - [ ] task-3-agents-md-check.txt
  - [ ] task-3-agents-md-structure.txt

  **Commit**: YES (groups with Wave 1)
  - Message: `feat: initialize project foundation`
  - Files: `AGENTS.md`

- [ ] 4. Rename and populate ARCHITECTURE.md

  **What to do**:
  - Rename `architecture.md` (lowercase) to `ARCHITECTURE.md` (uppercase) — use `git mv`
  - Populate with a template structure for system architecture documentation:
    - **System Overview**: High-level description placeholder + architecture diagram placeholder
    - **Core Components**: Table with component name, responsibility, key files, dependencies
    - **Data Flow**: How data moves through the system (placeholder diagram)
    - **Key Design Decisions**: ADR (Architecture Decision Record) style entries with context, decision, consequences
    - **Technology Stack**: Table with category (language, runtime, package manager, lint, typecheck, test, CI) and the chosen tool
    - **Directory Structure**: The target file tree with descriptions of each directory's purpose
    - **Integration Points**: External APIs, databases, services (placeholder section)
    - **How to Update This Document**: Instructions for keeping architecture docs current
  - Content should be template-style: headers + placeholder text explaining what goes in each section
  - Pre-fill the Technology Stack table with the actual choices (Python, uv, ruff, mypy, pytest, GitHub Actions)

  **Must NOT do**:
  - Do NOT write actual architecture for a specific project — this is template content
  - Do NOT make it overly long — aim for 80-120 lines of useful template structure

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation template requiring clear structure and helpful placeholder content
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-3, 5-8)
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `architecture.md` (current, empty) — rename target

  **External References**:
  - ADR format: https://adr.github.io/ — Architecture Decision Records template style

  **WHY Each Reference Matters**:
  - ADR format provides a proven template for documenting design decisions

  **Acceptance Criteria**:

  - [ ] ARCHITECTURE.md exists (uppercase) at project root
  - [ ] Old `architecture.md` (lowercase) no longer exists
  - [ ] Contains sections: System Overview, Core Components, Technology Stack, Directory Structure
  - [ ] Technology Stack table pre-filled with Python, uv, ruff, mypy, pytest, GitHub Actions

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: ARCHITECTURE.md properly renamed and populated
    Tool: Bash
    Preconditions: Project root accessible
    Steps:
      1. Run `ls ARCHITECTURE.md` — expected: file exists
      2. Run `ls architecture.md 2>&1` — expected: "No such file" (old file removed)
      3. Run `grep -i 'technology stack' ARCHITECTURE.md` — expected: match
      4. Run `grep -i 'ruff\|mypy\|pytest\|uv' ARCHITECTURE.md` — expected: multiple matches
      5. Run `wc -l ARCHITECTURE.md` — expected: 40-150 lines
    Expected Result: Renamed, populated, contains tech stack info
    Failure Indicators: Old file still exists, new file empty or missing tech stack
    Evidence: .sisyphus/evidence/task-4-architecture-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-4-architecture-check.txt

  **Commit**: YES (groups with Wave 1)
  - Message: `feat: initialize project foundation`
  - Files: `ARCHITECTURE.md` (renamed from `architecture.md`)

- [ ] 5. Create README.md — template usage guide

  **What to do**:
  - Create `README.md` with content focused on HOW TO USE THIS TEMPLATE:
    - **Title**: `# Agent Engineering Template` with brief tagline
    - **What This Is**: 1-2 sentences explaining this is a template for AI-assisted Python development
    - **Quick Start**: Step-by-step to use this template:
      1. Click "Use this template" on GitHub (or clone)
      2. Rename `my_project` → your project name (list all places: pyproject.toml, src/ directory, imports)
      3. Run `uv sync`
      4. Verify: `uv run ruff check . && uv run mypy src/ && uv run pytest`
    - **What's Included**: Brief list of what the template provides (AGENTS.md, docs structure, CI, etc.)
    - **Workflow**: The Agent Engineering workflow diagram (text-based):
      ```
      You define tasks → Write exec plan → AI reads docs → AI codes → AI lint/test → AI self-reviews → AI opens PR → You review → Merge
      ```
    - **Customization Guide**: What to change for your specific project
    - **File Structure**: The complete file tree with one-line descriptions
    - **Renaming Guide**: Exact sed/find-replace commands to rename `my_project` everywhere

  **Must NOT do**:
  - Do NOT write README for a specific project — this is meta-documentation about the template
  - Do NOT include badges (CI badge can be added after first CI run)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: User-facing documentation requiring clear, welcoming writing
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-4, 6-8)
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References**:

  **External References**:
  - User's screenshot showing the Agent Engineering workflow (the numbered steps)

  **WHY Each Reference Matters**:
  - The workflow diagram from the screenshot is the core concept to communicate in the README

  **Acceptance Criteria**:

  - [ ] README.md exists and is non-empty
  - [ ] Contains "Quick Start" section with numbered steps
  - [ ] Contains renaming instructions for `my_project`
  - [ ] Contains the workflow description
  - [ ] Contains file structure overview

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: README has all essential sections
    Tool: Bash
    Preconditions: README.md exists
    Steps:
      1. Run `grep -i 'quick start' README.md` — expected: match
      2. Run `grep -i 'uv sync' README.md` — expected: match
      3. Run `grep -i 'my_project' README.md` — expected: match (rename instructions)
      4. Run `grep -i 'workflow\|AI.*reads\|AI.*codes\|exec plan' README.md` — expected: at least 1 match
      5. Run `wc -l README.md` — expected: 50-200 lines
    Expected Result: All key sections present, reasonable length
    Failure Indicators: Missing sections, too short (<30 lines), no rename instructions
    Evidence: .sisyphus/evidence/task-5-readme-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-5-readme-check.txt

  **Commit**: YES (groups with Wave 1)
  - Message: `feat: initialize project foundation`
  - Files: `README.md`

- [ ] 6. Create docs/design-docs/ — index.md + core-beliefs.md

  **What to do**:
  - Create `docs/design-docs/index.md`:
    - Title: "Design Documents"
    - Purpose: Explain this directory holds design decisions and architectural rationale
    - Index table: columns for Doc Name, Status (Draft/Approved/Superseded), Date, Summary
    - One example entry for core-beliefs.md
    - Instructions for adding new design docs (naming convention, required sections)
  - Create `docs/design-docs/core-beliefs.md`:
    - Title: "Core Engineering Beliefs"
    - Sensible defaults for AI-assisted development beliefs:
      1. **Code is read more than written** — Clarity over cleverness
      2. **Tests are documentation** — Tests describe intended behavior
      3. **Automate everything repeatable** — If an AI agent can verify it, it should be automated
      4. **Small, frequent changes** — Small PRs over large rewrites
      5. **Fail fast, fail loud** — Errors should surface immediately, not silently
      6. **Documentation is code** — Docs are versioned, reviewed, and maintained like code
      7. **Trust but verify** — AI writes code, humans verify intent
    - Each belief: Title + 2-3 sentence explanation + practical implication

  **Must NOT do**:
  - Do NOT create more than these two files — other design docs are project-specific
  - Do NOT make beliefs overly specific to one domain

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Thoughtful content creation requiring engineering philosophy articulation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-5, 7-8)
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References**:

  **External References**:
  - Google's engineering practices: https://google.github.io/eng-practices/ — inspiration for belief framing

  **Acceptance Criteria**:

  - [ ] `docs/design-docs/index.md` exists with index table
  - [ ] `docs/design-docs/core-beliefs.md` exists with at least 5 beliefs
  - [ ] Each belief has title + explanation + practical implication

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Design docs directory properly structured
    Tool: Bash
    Preconditions: docs/design-docs/ exists
    Steps:
      1. Run `ls docs/design-docs/index.md` — expected: file exists
      2. Run `ls docs/design-docs/core-beliefs.md` — expected: file exists
      3. Run `grep -c '^##\|^###' docs/design-docs/core-beliefs.md` — expected: >= 5 (beliefs as headers)
      4. Run `grep -i 'core-beliefs' docs/design-docs/index.md` — expected: match (referenced in index)
    Expected Result: Both files exist, core-beliefs has 5+ beliefs, index references core-beliefs
    Failure Indicators: Missing files, too few beliefs, index doesn't reference core-beliefs
    Evidence: .sisyphus/evidence/task-6-design-docs-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-6-design-docs-check.txt

  **Commit**: YES (groups with Wave 1)
  - Message: `feat: initialize project foundation`
  - Files: `docs/design-docs/index.md`, `docs/design-docs/core-beliefs.md`

- [ ] 7. Create docs/PLANS.md + docs/QUALITY_SCORE.md

  **What to do**:
  - Create `docs/PLANS.md`:
    - Title: "Project Plans & Roadmap"
    - Kanban-style sections: Backlog, In Progress, Done
    - Each item format: `- [ ] [Priority] Task description — linked exec plan`
    - Example entries showing the format
    - Instructions for maintaining the plan (how to add items, how to promote from backlog to in-progress)
    - Cross-reference to `docs/exec-plans/` for detailed execution plans

  - Create `docs/QUALITY_SCORE.md`:
    - Title: "Quality Tracking Framework"
    - **Layer 1 — Automated Gates** (binary pass/fail):
      - `uv run ruff check` → PASS/FAIL
      - `uv run mypy` → PASS/FAIL
      - `uv run pytest` → PASS/FAIL + coverage %
      - CI pipeline → PASS/FAIL
    - **Layer 2 — Trend Metrics** (tracked over time):
      - Test coverage % (current + target)
      - Type coverage % (files with type hints)
      - Open tech debt items (from tech-debt-tracker.md)
      - Average PR review turnaround
    - **Layer 3 — Human Rubric** (periodic assessment):
      - Code readability (1-5)
      - Architecture fitness (1-5)
      - Documentation freshness (1-5)
      - Onboarding friction (1-5 — how easy for new agent/human to start)
    - Scoring template table with date, scores, notes
    - Instructions for when to update (every sprint/milestone)

  **Must NOT do**:
  - Do NOT create a dashboard or automation — this is a markdown tracking template
  - Do NOT add fake scores — provide empty template rows

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Framework documentation requiring structured thinking about quality dimensions
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-6, 8)
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - User's screenshot showing PLANS.md and QUALITY_SCORE.md in the docs/ directory

  **External References**:
  - Quality framework research from Metis: three-layer approach (automated gates + trend metrics + human rubric)

  **Acceptance Criteria**:

  - [ ] `docs/PLANS.md` exists with Backlog/In Progress/Done sections
  - [ ] `docs/QUALITY_SCORE.md` exists with 3-layer quality framework
  - [ ] Quality framework includes automated gates, trend metrics, and human rubric
  - [ ] Both files have example entries showing the format

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: PLANS.md and QUALITY_SCORE.md properly structured
    Tool: Bash
    Preconditions: docs/ directory exists
    Steps:
      1. Run `ls docs/PLANS.md docs/QUALITY_SCORE.md` — expected: both files exist
      2. Run `grep -i 'backlog\|in progress\|done' docs/PLANS.md` — expected: all 3 match
      3. Run `grep -i 'automated.*gate\|trend.*metric\|human.*rubric' docs/QUALITY_SCORE.md` — expected: all 3 layers present
      4. Run `grep -i 'ruff\|mypy\|pytest' docs/QUALITY_SCORE.md` — expected: tools referenced in automated gates
    Expected Result: Both files have proper structure and content
    Failure Indicators: Missing files, missing kanban sections, missing quality layers
    Evidence: .sisyphus/evidence/task-7-plans-quality-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-7-plans-quality-check.txt

  **Commit**: YES (groups with Wave 1)
  - Message: `feat: initialize project foundation`
  - Files: `docs/PLANS.md`, `docs/QUALITY_SCORE.md`

- [ ] 8. Create docs/exec-plans/ directory structure + tech-debt-tracker.md

  **What to do**:
  - Create directory structure:
    - `docs/exec-plans/active/` — with `.gitkeep` to track empty dir
    - `docs/exec-plans/completed/` — with `.gitkeep` to track empty dir
  - Create `docs/exec-plans/tech-debt-tracker.md`:
    - Title: "Technical Debt Tracker"
    - Table format: ID, Description, Severity (High/Medium/Low), Category (code/test/docs/infra), Introduced (date/PR), Planned Resolution, Status
    - Categories explained: code debt, test debt, documentation debt, infrastructure debt
    - Instructions for adding new debt items
    - Instructions for resolving and archiving debt items
    - Cross-reference to periodic agent maintenance workflow (from user's screenshot: "周期性 agent 做 doc-gardening + refactor + tech debt cleanup")
    - 2-3 example entries (empty/template rows)

  **Must NOT do**:
  - Do NOT create actual exec plan files — those are project-specific
  - Do NOT create subdirectories beyond active/ and completed/

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mostly structural — directories, gitkeep, one markdown file with table template
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-7)
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - User's screenshot: `docs/exec-plans/` with `active/`, `completed/`, `tech-debt-tracker.md`

  **Acceptance Criteria**:

  - [ ] `docs/exec-plans/active/.gitkeep` exists
  - [ ] `docs/exec-plans/completed/.gitkeep` exists
  - [ ] `docs/exec-plans/tech-debt-tracker.md` exists with table structure
  - [ ] Tech debt tracker has columns: ID, Description, Severity, Category, Status

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Exec plans directory properly structured
    Tool: Bash
    Preconditions: docs/ directory exists
    Steps:
      1. Run `ls docs/exec-plans/active/.gitkeep` — expected: file exists
      2. Run `ls docs/exec-plans/completed/.gitkeep` — expected: file exists
      3. Run `ls docs/exec-plans/tech-debt-tracker.md` — expected: file exists
      4. Run `grep -i 'severity\|category\|status' docs/exec-plans/tech-debt-tracker.md` — expected: matches
    Expected Result: Directory structure exists, tracker has proper columns
    Failure Indicators: Missing directories or .gitkeep files, tracker missing table structure
    Evidence: .sisyphus/evidence/task-8-exec-plans-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-8-exec-plans-check.txt

  **Commit**: YES (groups with Wave 1)
  - Message: `feat: initialize project foundation`
  - Files: `docs/exec-plans/active/.gitkeep`, `docs/exec-plans/completed/.gitkeep`, `docs/exec-plans/tech-debt-tracker.md`

- [ ] 9. Create src/my_project/ package structure

  **What to do**:
  - Create `src/my_project/__init__.py`:
    - Module docstring: brief description placeholder
    - `__version__ = "0.1.0"` — matches pyproject.toml
  - Create `src/my_project/py.typed`:
    - Empty file (PEP 561 marker for type checkers)
  - Create `src/my_project/hello.py`:
    - One simple, properly-typed function as a placeholder:
      ```python
      def greet(name: str) -> str:
          """Return a greeting message."""
          return f"Hello, {name}!"
      ```
    - This serves as:
      - Proof that imports work
      - Example of type annotation style
      - Target for the test in Task 10
      - Something for ruff/mypy to check

  **Must NOT do**:
  - Do NOT add more than one module — this is a minimal scaffold
  - Do NOT add application logic beyond the placeholder
  - Do NOT use `any` type or missing type annotations
  - Do NOT add unnecessary imports

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Minimal Python files with one trivial function
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 10-13 after Wave 1)
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 10, 14
  - **Blocked By**: Task 1 (needs pyproject.toml for package name and structure)

  **References**:

  **API/Type References**:
  - PEP 561: https://peps.python.org/pep-0561/ — py.typed marker file standard
  - Task 1's pyproject.toml — package name must match (`my_project`)

  **WHY Each Reference Matters**:
  - PEP 561: The py.typed file tells mypy/pyright this package ships inline types
  - pyproject.toml: Package name in src/ must match `[project] name` and `[tool.mypy] packages`

  **Acceptance Criteria**:

  - [ ] `src/my_project/__init__.py` exists with `__version__`
  - [ ] `src/my_project/py.typed` exists (can be empty)
  - [ ] `src/my_project/hello.py` exists with typed `greet()` function
  - [ ] `uv run mypy src/` passes on these files

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Package imports correctly
    Tool: Bash
    Preconditions: uv sync completed, src/my_project/ exists
    Steps:
      1. Run `uv run python -c "from my_project import __version__; print(__version__)"` — expected: prints "0.1.0"
      2. Run `uv run python -c "from my_project.hello import greet; print(greet('World'))"` — expected: prints "Hello, World!"
      3. Run `uv run mypy src/my_project/hello.py` — expected: "Success: no issues found"
      4. Run `uv run ruff check src/my_project/` — expected: no violations
    Expected Result: All imports work, all tools pass
    Failure Indicators: ImportError, mypy errors, ruff violations
    Evidence: .sisyphus/evidence/task-9-package-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-9-package-check.txt

  **Commit**: YES (groups with Wave 2)
  - Message: `feat: add source scaffold, tests, and GitHub integration`
  - Files: `src/my_project/__init__.py`, `src/my_project/py.typed`, `src/my_project/hello.py`

- [ ] 10. Create tests/ structure with test_hello.py

  **What to do**:
  - Create `tests/__init__.py`: Empty file (required for pytest to find tests)
  - Create `tests/conftest.py`:
    - Docstring: "Shared test fixtures"
    - One example fixture (optional, can be empty with just docstring)
  - Create `tests/test_hello.py`:
    - Import `greet` from `my_project.hello`
    - Test happy path: `test_greet_returns_greeting()` — asserts `greet("World") == "Hello, World!"`
    - Test edge case: `test_greet_with_empty_string()` — asserts `greet("") == "Hello, !"`
    - Tests must pass with `uv run pytest`

  **Must NOT do**:
  - Do NOT add more test files — one test file for the one module
  - Do NOT use unittest — use pytest style (plain functions with assert)
  - Do NOT add unnecessary fixtures or complex setup

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Two simple test functions for a trivial module
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 11-13)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 14
  - **Blocked By**: Task 1 (pytest config), Task 9 (module to test)

  **References**:

  **Pattern References**:
  - Task 9's `src/my_project/hello.py` — the function being tested
  - Task 1's pyproject.toml `[tool.pytest.ini_options]` — test paths config

  **WHY Each Reference Matters**:
  - hello.py: Import path and function signature to test against
  - pyproject.toml: Ensures testpaths and pythonpath are correct for imports

  **Acceptance Criteria**:

  - [ ] `tests/__init__.py` exists
  - [ ] `tests/conftest.py` exists
  - [ ] `tests/test_hello.py` exists with at least 2 test functions
  - [ ] `uv run pytest` → 2 tests passed, 0 failures

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All tests pass
    Tool: Bash
    Preconditions: uv sync completed, src/my_project/ exists, tests/ exists
    Steps:
      1. Run `uv run pytest -v` — expected: 2 passed, 0 failed
      2. Run `uv run pytest --tb=short` — expected: no traceback output (all pass)
      3. Run `uv run ruff check tests/` — expected: no violations
    Expected Result: 2 tests pass, test code is clean
    Failure Indicators: Any test failure, ruff violations in test files
    Evidence: .sisyphus/evidence/task-10-tests-check.txt

  Scenario: Tests actually test the right thing
    Tool: Bash
    Preconditions: tests/test_hello.py exists
    Steps:
      1. Run `grep 'def test_' tests/test_hello.py` — expected: at least 2 test functions
      2. Run `grep 'assert' tests/test_hello.py` — expected: at least 2 assertions
      3. Run `grep 'greet' tests/test_hello.py` — expected: greet function is called
    Expected Result: Tests contain proper assertions against the greet function
    Failure Indicators: No assert statements, greet not imported/called
    Evidence: .sisyphus/evidence/task-10-test-content-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-10-tests-check.txt
  - [ ] task-10-test-content-check.txt

  **Commit**: YES (groups with Wave 2)
  - Message: `feat: add source scaffold, tests, and GitHub integration`
  - Files: `tests/__init__.py`, `tests/conftest.py`, `tests/test_hello.py`

- [ ] 11. Create .github/workflows/ci.yml — CI pipeline

  **What to do**:
  - Create `.github/workflows/ci.yml` with:
    - **Name**: `CI`
    - **Triggers**: push to `main`, pull_request to `main`
    - **Jobs**: Single job `check` on `ubuntu-latest`
    - **Steps**:
      1. Checkout code
      2. Install uv (use `astral-sh/setup-uv@v4` action)
      3. Set up Python 3.12 (use `actions/setup-python@v5` with `python-version-file: .python-version`)
      4. Install deps: `uv sync`
      5. Lint: `uv run ruff check src/ tests/`
      6. Format check: `uv run ruff format --check src/ tests/`
      7. Type check: `uv run mypy src/`
      8. Test: `uv run pytest`
    - Use proper caching for uv (uv action handles this)
    - All steps should fail-fast (default behavior)

  **Must NOT do**:
  - Do NOT add deployment steps — CI only
  - Do NOT add AI-specific workflows (ai-pr-review.yml) — beyond template scope
  - Do NOT add matrix builds — keep it simple for a template
  - Do NOT use `pip install` — use uv throughout

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard GitHub Actions YAML with well-known patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9, 10, 12, 13)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 14
  - **Blocked By**: Task 1 (needs to know exact commands from pyproject.toml)

  **References**:

  **External References**:
  - astral-sh/setup-uv: https://github.com/astral-sh/setup-uv — official uv GitHub Action
  - actions/setup-python: https://github.com/actions/setup-python — Python setup action

  **WHY Each Reference Matters**:
  - setup-uv: Correct action version and caching configuration
  - setup-python: Proper Python version setup with .python-version file support

  **Acceptance Criteria**:

  - [ ] `.github/workflows/ci.yml` exists
  - [ ] Triggers on push to main AND pull_request to main
  - [ ] Runs: ruff check, ruff format --check, mypy, pytest
  - [ ] Uses uv for all package operations
  - [ ] YAML is valid (parseable)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: CI workflow YAML is valid and complete
    Tool: Bash
    Preconditions: .github/workflows/ci.yml exists
    Steps:
      1. Run `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` — expected: parses without error (install PyYAML if needed, or use `uv run python -c "..."`)
      2. Run `grep 'ruff check' .github/workflows/ci.yml` — expected: match
      3. Run `grep 'ruff format' .github/workflows/ci.yml` — expected: match
      4. Run `grep 'mypy' .github/workflows/ci.yml` — expected: match
      5. Run `grep 'pytest' .github/workflows/ci.yml` — expected: match
      6. Run `grep 'push\|pull_request' .github/workflows/ci.yml` — expected: both triggers present
      7. Run `grep 'setup-uv\|astral-sh' .github/workflows/ci.yml` — expected: uv action used
    Expected Result: Valid YAML, all 4 tool steps present, proper triggers
    Failure Indicators: YAML parse error, missing tool steps, missing triggers
    Evidence: .sisyphus/evidence/task-11-ci-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-11-ci-check.txt

  **Commit**: YES (groups with Wave 2)
  - Message: `feat: add source scaffold, tests, and GitHub integration`
  - Files: `.github/workflows/ci.yml`

- [ ] 12. Create .github/pull_request_template.md — AI-aware PR template

  **What to do**:
  - Create `.github/pull_request_template.md` with AI-specific sections:
    1. **Summary**: Brief description of changes (2-3 sentences)
    2. **Linked Issue**: `Closes #` reference
    3. **Type of Change**: Checkboxes (bug fix, new feature, refactor, docs, test, CI/CD, other)
    4. **What Changed**: Specific file-level changes with brief descriptions
    5. **Verification Performed**: Section for pasting actual command output:
       ```
       ### Lint
       <output of uv run ruff check>
       
       ### Type Check
       <output of uv run mypy>
       
       ### Tests
       <output of uv run pytest>
       ```
    6. **Self-Review Checklist** (for AI agent):
       - [ ] Code follows project style (ruff passes)
       - [ ] Types are correct (mypy passes)
       - [ ] Tests pass and cover new behavior
       - [ ] No secrets or credentials in code
       - [ ] No unnecessary dependencies added
       - [ ] Documentation updated if needed
       - [ ] Changes are within the scope of the linked issue
    7. **Agent Metadata** (YAML block):
       ```yaml
       agent: <agent name>
       model: <model version>
       exec_plan: <link to exec plan if applicable>
       files_changed: <count>
       ```
    8. **Risks & Considerations**: Known risks, trade-offs, areas needing extra review attention
    9. **Human Reviewer Checklist**:
       - [ ] Intent matches the issue/plan
       - [ ] No scope creep beyond the plan
       - [ ] Architecture decisions are sound
       - [ ] Edge cases considered

  **Must NOT do**:
  - Do NOT make it overly verbose — each section should be concise
  - Do NOT remove the agent metadata section — it's key for the AI workflow

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Template authoring requiring clear structure and thoughtful section design
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9-11, 13)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References**:

  **External References**:
  - User's screenshot: "AI 开 PR, 附结果摘要和风险" — PR must include results summary and risks
  - Metis research: AI-specific PR template sections (self-review, verification output, agent metadata, human checklist)

  **WHY Each Reference Matters**:
  - User's workflow: The PR template must support "results + risks" requirement
  - Metis: Best practices for AI-generated PRs from research

  **Acceptance Criteria**:

  - [ ] `.github/pull_request_template.md` exists
  - [ ] Contains self-review checklist section
  - [ ] Contains verification output section (lint, typecheck, tests)
  - [ ] Contains agent metadata YAML block
  - [ ] Contains human reviewer checklist
  - [ ] Contains risks section

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: PR template has all required sections
    Tool: Bash
    Preconditions: .github/pull_request_template.md exists
    Steps:
      1. Run `grep -i 'self.review\|self-review' .github/pull_request_template.md` — expected: match
      2. Run `grep -i 'verification\|ruff\|mypy\|pytest' .github/pull_request_template.md` — expected: matches
      3. Run `grep -i 'agent.*metadata\|agent:' .github/pull_request_template.md` — expected: match
      4. Run `grep -i 'risk\|consideration' .github/pull_request_template.md` — expected: match
      5. Run `grep -i 'human.*review\|reviewer.*checklist' .github/pull_request_template.md` — expected: match
      6. Run `grep -c '\- \[ \]' .github/pull_request_template.md` — expected: >= 8 (checkboxes)
    Expected Result: All sections present with enough checkboxes for thorough review
    Failure Indicators: Missing sections, too few checkboxes
    Evidence: .sisyphus/evidence/task-12-pr-template-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-12-pr-template-check.txt

  **Commit**: YES (groups with Wave 2)
  - Message: `feat: add source scaffold, tests, and GitHub integration`
  - Files: `.github/pull_request_template.md`

- [ ] 13. Create .github/ISSUE_TEMPLATE/ YAML forms

  **What to do**:
  - Create `.github/ISSUE_TEMPLATE/config.yml`:
    - `blank_issues_enabled: true`
    - Contact links (optional, can leave empty array)
  
  - Create `.github/ISSUE_TEMPLATE/ai-task.yml`:
    - Name: "AI Task"
    - Description: "Define a task for AI agent execution"
    - Labels: `["ai-task"]`
    - Fields:
      - Task Type: dropdown (feature, bugfix, refactor, documentation, test, maintenance)
      - Problem Statement: textarea (required, description: "What needs to be done and why")
      - Acceptance Criteria: textarea (required, description: "Checkboxes of what 'done' looks like")
      - Files Likely Involved: textarea (optional)
      - Scope Assessment: dropdown (small: 1-2 files, medium: 3-5 files, large: 6+ files)
      - Explicitly Out of Scope: textarea (optional)

  - Create `.github/ISSUE_TEMPLATE/bug-report.yml`:
    - Name: "Bug Report"
    - Description: "Report a bug"
    - Labels: `["bug"]`
    - Fields:
      - Description: textarea (required)
      - Steps to Reproduce: textarea (required)
      - Expected Behavior: textarea (required)
      - Actual Behavior: textarea (required)
      - Environment: input (Python version, OS)

  - Create `.github/ISSUE_TEMPLATE/feature-request.yml`:
    - Name: "Feature Request"
    - Description: "Suggest a new feature"
    - Labels: `["enhancement"]`
    - Fields:
      - Problem: textarea (required, "What problem does this solve?")
      - Proposed Solution: textarea (required)
      - Alternatives Considered: textarea (optional)
      - Additional Context: textarea (optional)

  **Must NOT do**:
  - Do NOT use .md format for issue templates — use .yml (YAML forms)
  - Do NOT create more than these 3 templates + config

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Structured YAML files with well-defined fields — no creative writing needed
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9-12)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References**:

  **External References**:
  - GitHub YAML issue forms docs: https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms
  - Metis research: YAML forms are superior for AI agents due to structured fields

  **WHY Each Reference Matters**:
  - GitHub docs: Exact YAML syntax for form fields (type, id, attributes, validations)
  - Metis: YAML forms provide structured data that AI agents can parse more reliably

  **Acceptance Criteria**:

  - [ ] `.github/ISSUE_TEMPLATE/config.yml` exists
  - [ ] `.github/ISSUE_TEMPLATE/ai-task.yml` exists with form fields
  - [ ] `.github/ISSUE_TEMPLATE/bug-report.yml` exists with form fields
  - [ ] `.github/ISSUE_TEMPLATE/feature-request.yml` exists with form fields
  - [ ] All YAML files are valid (parseable)
  - [ ] ai-task.yml has dropdown for task type and required fields

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All issue templates exist and are valid YAML
    Tool: Bash
    Preconditions: .github/ISSUE_TEMPLATE/ exists
    Steps:
      1. Run `ls .github/ISSUE_TEMPLATE/` — expected: 4 files (config.yml, ai-task.yml, bug-report.yml, feature-request.yml)
      2. For each .yml file, run `python -c "import yaml; yaml.safe_load(open('FILE'))"` — expected: all parse
      3. Run `grep 'type: dropdown' .github/ISSUE_TEMPLATE/ai-task.yml` — expected: match
      4. Run `grep 'required: true' .github/ISSUE_TEMPLATE/ai-task.yml` — expected: match
      5. Run `grep -l '.yml' .github/ISSUE_TEMPLATE/` — expected: no .md template files
    Expected Result: All 4 YAML files exist and parse, ai-task has dropdowns and required fields
    Failure Indicators: YAML parse errors, missing files, .md files instead of .yml
    Evidence: .sisyphus/evidence/task-13-issue-templates-check.txt

  Scenario: ai-task template has all required fields
    Tool: Bash
    Preconditions: .github/ISSUE_TEMPLATE/ai-task.yml exists
    Steps:
      1. Run `grep -i 'task type\|problem statement\|acceptance criteria\|scope' .github/ISSUE_TEMPLATE/ai-task.yml` — expected: all 4 present
      2. Run `grep -c 'type:' .github/ISSUE_TEMPLATE/ai-task.yml` — expected: >= 4 (field type declarations)
    Expected Result: All essential fields present with proper types
    Failure Indicators: Missing fields, no type declarations
    Evidence: .sisyphus/evidence/task-13-ai-task-fields-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-13-issue-templates-check.txt
  - [ ] task-13-ai-task-fields-check.txt

  **Commit**: YES (groups with Wave 2)
  - Message: `feat: add source scaffold, tests, and GitHub integration`
  - Files: `.github/ISSUE_TEMPLATE/config.yml`, `.github/ISSUE_TEMPLATE/ai-task.yml`, `.github/ISSUE_TEMPLATE/bug-report.yml`, `.github/ISSUE_TEMPLATE/feature-request.yml`

- [ ] 14. Full integration verification — run all tools end-to-end

  **What to do**:
  - This is the CRITICAL verification task. Run everything from a clean state:
    1. Verify all 28 files exist (list and count)
    2. Run `uv sync` — verify clean install
    3. Run `uv run ruff check src/ tests/` — verify 0 violations
    4. Run `uv run ruff format --check src/ tests/` — verify already formatted
    5. Run `uv run mypy src/` — verify 0 errors
    6. Run `uv run pytest -v` — verify all tests pass
    7. Run `wc -l AGENTS.md` — verify under 150 lines
    8. Validate all YAML files parse correctly
    9. Check no forbidden files exist (no CLAUDE.md, no deployment configs)
    10. Fix ANY issues found — update the relevant files
  - If fixes are needed, make them and re-run all checks until everything is green

  **Must NOT do**:
  - Do NOT skip any verification step
  - Do NOT leave failing checks unresolved
  - Do NOT add new files — only fix existing ones

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires methodical verification of all components, debugging any integration issues, and ensuring everything works together
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (solo)
  - **Blocks**: F1-F4
  - **Blocked By**: ALL Tasks 1-13

  **References**:

  **Pattern References**:
  - All tasks 1-13 — this task verifies their combined output
  - Success Criteria section of this plan — run all verification commands listed there

  **Acceptance Criteria**:

  - [ ] All 28 files exist
  - [ ] `uv sync` → exit 0
  - [ ] `uv run ruff check src/ tests/` → 0 violations
  - [ ] `uv run ruff format --check src/ tests/` → already formatted
  - [ ] `uv run mypy src/` → Success, 0 errors
  - [ ] `uv run pytest` → all pass
  - [ ] `wc -l AGENTS.md` < 150
  - [ ] All YAML files parse
  - [ ] No forbidden files (CLAUDE.md, etc.)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Complete end-to-end verification
    Tool: Bash
    Preconditions: All tasks 1-13 completed
    Steps:
      1. Count all expected files: `find . -not -path './.git/*' -not -path './.venv/*' -not -path './.sisyphus/*' -not -path './node_modules/*' -not -name 'uv.lock' -type f | wc -l` — expected: ~28
      2. Run `uv sync` — expected: exit 0
      3. Run `uv run ruff check src/ tests/` — expected: "All checks passed" or similar
      4. Run `uv run ruff format --check src/ tests/` — expected: "N files already formatted"
      5. Run `uv run mypy src/` — expected: "Success: no issues found"
      6. Run `uv run pytest -v` — expected: "N passed" with 0 failures
      7. Run `wc -l AGENTS.md` — expected: <150
      8. Validate YAML: Check all .yml files in .github/ parse correctly
      9. Run `ls CLAUDE.md 2>&1` — expected: "No such file"
    Expected Result: All 9 checks pass green
    Failure Indicators: Any non-zero exit code, any failing test, AGENTS.md over 150 lines
    Evidence: .sisyphus/evidence/task-14-full-verification.txt

  Scenario: Template is usable from scratch
    Tool: Bash
    Preconditions: Full repo with all files
    Steps:
      1. Verify `uv run python -c "from my_project.hello import greet; print(greet('Template'))"` outputs "Hello, Template!"
      2. Verify README.md mentions `my_project` (for rename instructions)
      3. Verify AGENTS.md references `uv run` commands
    Expected Result: Template is functional and self-documenting
    Failure Indicators: Import failures, missing rename instructions
    Evidence: .sisyphus/evidence/task-14-usability-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-14-full-verification.txt
  - [ ] task-14-usability-check.txt

  **Commit**: YES (only if fixes were made)
  - Message: `fix: resolve integration issues from verification`
  - Files: any files fixed
  - Pre-commit: all verification commands must pass

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, check content). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.sisyphus/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run ruff check src/ tests/` + `uv run mypy src/` + `uv run pytest`. Review all files for: type errors, unused imports, formatting issues. Check AGENTS.md is under 150 lines. Validate all YAML files parse correctly. Check pyproject.toml has all required sections.
  Output: `Ruff [PASS/FAIL] | Mypy [PASS/FAIL] | Pytest [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state (`rm -rf .venv uv.lock && uv sync`). Run EVERY verification command from Definition of Done. Test `uv run ruff check`, `uv run mypy`, `uv run pytest` from scratch. Verify all docs have content. Verify all GitHub YAML files are valid. Save to `.sisyphus/evidence/final-qa/`.
  Output: `DoD Items [N/N pass] | Docs [N/N populated] | YAML [N/N valid] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual files. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance (no CLAUDE.md, no AI workflow, no deployment configs). Count files: should be exactly 28 created. Flag any unaccounted files.
  Output: `Tasks [N/N compliant] | File Count [N/28] | Scope [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Wave 1 Commit**: `feat: initialize project foundation — pyproject.toml, AGENTS.md, docs structure` — pyproject.toml, .python-version, .gitignore, AGENTS.md, ARCHITECTURE.md, README.md, docs/**
- **Wave 2 Commit**: `feat: add source scaffold, tests, and GitHub integration` — src/**, tests/**, .github/**
- **Wave 3 Commit**: `fix: resolve integration issues from verification` — any files fixed during verification
- **Final Commit** (if needed): `fix: address review findings` — any fixes from final review

---

## Success Criteria

### Verification Commands
```bash
uv sync                                    # Expected: resolves deps, creates .venv
uv run ruff check src/ tests/             # Expected: All checks passed!
uv run ruff format --check src/ tests/    # Expected: N files already formatted
uv run mypy src/                           # Expected: Success: no issues found
uv run pytest                              # Expected: 1 passed
wc -l AGENTS.md                            # Expected: <150 lines
ls .github/workflows/ci.yml               # Expected: file exists
ls .github/pull_request_template.md       # Expected: file exists
ls .github/ISSUE_TEMPLATE/*.yml           # Expected: 4 YAML files
ls docs/design-docs/index.md docs/design-docs/core-beliefs.md  # Expected: files exist
ls docs/exec-plans/tech-debt-tracker.md   # Expected: file exists
ls docs/PLANS.md docs/QUALITY_SCORE.md    # Expected: files exist
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tools pass (ruff, mypy, pytest)
- [ ] AGENTS.md under 150 lines
- [ ] All 28 files created with non-empty content
- [ ] Template is immediately usable after clone + `uv sync`
