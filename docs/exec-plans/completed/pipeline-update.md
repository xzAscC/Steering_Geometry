# Execution Plan: Pipeline Workflow Update

## Metadata

- **Created**: 2026-03-06
- **Status**: Completed
- **Priority**: High
- **Type**: Process Improvement

## Objective

Update Sisyphus agent behavior to enforce a strict pipeline workflow for all task execution.

## Pipeline Definition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SISYPHUS PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. READ PLAN                                                            │
│     └── Read PLAN.md → Parse tasks → Understand requirements            │
│                                                                          │
│  2. CODE IMPLEMENTATION                                                  │
│     └── Write/Edit code → Follow AGENTS.md conventions                  │
│                                                                          │
│  3. VERIFY (lint/typecheck/test)                                         │
│     ├── uv run ruff check src/ tests/                                   │
│     ├── uv run ruff format --check src/ tests/                          │
│     ├── uv run mypy src/                                                │
│     └── uv run pytest                                                   │
│                                                                          │
│  4. MOVE PLAN                                                            │
│     ├── If complete → docs/exec-plans/completed/                        │
│     └── If in progress → docs/exec-plans/active/                        │
│                                                                          │
│  5. UPDATE DOCS                                                          │
│     ├── Update docs/PLANS.md (roadmap status)                           │
│     ├── Update docs/QUALITY_SCORE.md (if applicable)                    │
│     └── Update ARCHITECTURE.md (if structural changes)                  │
│                                                                          │
│  6. COMMIT/PR                                                            │
│     ├── Commit when: logical unit complete, all checks pass             │
│     └── PR when: feature complete, ready for review                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Commit Criteria

Commit when ALL of the following are true:
- [ ] Logical unit of work is complete
- [ ] `uv run ruff check src/ tests/` → 0 violations
- [ ] `uv run ruff format --check src/ tests/` → already formatted
- [ ] `uv run mypy src/` → 0 errors
- [ ] `uv run pytest` → all tests pass
- [ ] Related documentation updated

## Tasks

### Phase 1: Update AGENTS.md

- [ ] Add Pipeline section to AGENTS.md
- [ ] Document commit criteria explicitly
- [ ] Add plan movement rules

### Phase 2: Update PLAN.md

- [ ] Create PLAN.md with current task reference
- [ ] Link to this execution plan

### Phase 3: Verify

- [ ] Run all verification commands
- [ ] Ensure pipeline is documented correctly

### Phase 4: Commit

- [ ] Commit with message: `docs: add pipeline workflow to AGENTS.md`
- [ ] Move this plan to completed/

## Success Criteria

- [ ] AGENTS.md contains complete pipeline documentation
- [ ] All lint/typecheck/test commands pass
- [ ] Commit created with appropriate message
- [ ] This plan moved to completed/

## Risks

- None identified (documentation-only change)
