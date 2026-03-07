# Quality Tracking Framework

## Layer 1: Automated Gates (Pass/Fail)

| Check | Command | Status |
|-------|---------|--------|
| Lint | `uv run ruff check src/ tests/` | ⬜ |
| Format | `uv run ruff format --check src/ tests/` | ⬜ |
| Type Check | `uv run mypy src/` | ⬜ |
| Tests | `uv run pytest` | ⬜ |
| CI | GitHub Actions | ⬜ |

## Layer 2: Trend Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Test Coverage | - % | 80% | |
| Type Coverage | - % | 100% | Files with type hints |
| Open Tech Debt | - | 0 | From tech-debt-tracker.md |
| PR Review Time | - hrs | <24 hrs | Average turnaround |

## Layer 3: Human Rubric (1-5 Scale)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Code Readability | - | How easy is code to understand? |
| Architecture Fitness | - | Does structure support growth? |
| Documentation Freshness | - | Are docs up to date? |
| Onboarding Friction | - | How easy for new dev/agent to start? |

## Scoring History

| Date | Gates | Coverage | Readability | Architecture | Docs | Onboarding |
|------|-------|----------|-------------|--------------|------|------------|
| - | -/5 | -% | - | - | - | - |

## Update Schedule

- **Automated Gates**: Every PR
- **Trend Metrics**: Weekly
- **Human Rubric**: Every sprint/milestone
