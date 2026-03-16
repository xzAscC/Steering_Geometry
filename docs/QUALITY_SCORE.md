# Quality Tracking Framework

## Layer 1: Automated Gates (Pass/Fail)

| Check | Command | Status |
|-------|---------|--------|
| Lint | `uv run ruff check src/ tests/` | ✅ |
| Format | `uv run ruff format --check src/ tests/` | ✅ |
| Type Check | `uv run mypy src/` | ✅ |
| Tests | `uv run pytest` | ✅ (21 pass) |
| CI | GitHub Actions | ⬜ |

## Layer 2: Trend Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Test Coverage | ~60% | 80% | 21 tests across unit/integration |
| Type Coverage | 100% | 100% | All files have type hints |
| Open Tech Debt | 0 | 0 | From tech-debt-tracker.md |
| PR Review Time | - hrs | <24 hrs | Average turnaround |

## Layer 3: Human Rubric (1-5 Scale)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Code Readability | 4 | Clean, well-documented code |
| Architecture Fitness | 4 | Modular design with clear separation |
| Documentation Freshness | 4 | README/ARCHITECTURE updated 2026-03 |
| Onboarding Friction | 4 | Clear AGENTS.md and pipeline docs |

## Scoring History

| Date | Gates | Coverage | Readability | Architecture | Docs | Onboarding |
|------|-------|----------|-------------|--------------|------|------------|
| 2026-03-07 | 4/5 | ~60% | 4 | 4 | 4 | 4 |

## Update Schedule

- **Automated Gates**: Every PR
- **Trend Metrics**: Weekly
- **Human Rubric**: Every sprint/milestone
