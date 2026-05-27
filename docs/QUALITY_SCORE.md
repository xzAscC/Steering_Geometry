# Quality Tracking Framework

## Layer 1: Automated Gates (Pass/Fail)

| Check | Command | Status |
|-------|---------|--------|
| Lint | `uv run ruff check src/ tests/` | OK |
| Format | `uv run ruff format --check src/ tests/` | OK |
| Type Check | `uv run mypy src/` | OK |
| Tests | `uv run pytest` | OK (191 pass, 2 skip) |
| CI | GitHub Actions | Pending |

## Layer 2: Trend Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Test Coverage | ~60% | 80% | 191 tests across unit/integration |
| Type Coverage | 100% | 100% | All files have type hints |
| Open Tech Debt | 0 | 0 | From tech-debt-tracker.md |
| PR Review Time | - hrs | <24 hrs | Average turnaround |

## Layer 3: Human Rubric (1-5 Scale)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Code Readability | 4 | Clean, well-documented code |
| Architecture Fitness | 4 | Paper-aligned modular design |
| Documentation Freshness | 5 | All docs updated for NeurIPS 2026 paper |
| Onboarding Friction | 4 | Clear AGENTS.md and pipeline docs |

## Scoring History

| Date | Gates | Coverage | Readability | Architecture | Docs | Onboarding |
|------|-------|----------|-------------|--------------|------|------------|
| 2026-03-07 | 4/5 | ~60% | 4 | 4 | 4 | 4 |
| 2026-05-26 | 4/5 | ~60% | 4 | 4 | 5 | 4 |

## Update Schedule

- **Automated Gates**: Every PR
- **Trend Metrics**: Weekly
- **Human Rubric**: Every sprint/milestone
