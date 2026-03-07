# Technical Debt Tracker

Track technical debt items for periodic cleanup.

## Active Debt

| ID | Description | Severity | Category | Introduced | Resolution Plan | Status |
|----|-------------|----------|----------|------------|-----------------|--------|
| - | - | - | - | - | - | - |

## Severity Levels

- **High**: Blocks development or causes frequent issues
- **Medium**: Causes friction but has workarounds
- **Low**: Minor inconvenience, fix when convenient

## Categories

- **code**: Code quality issues (duplication, complexity)
- **test**: Missing or inadequate tests
- **docs**: Missing or outdated documentation
- **infra**: Build, deploy, or infrastructure issues

## Adding New Debt

1. Assign unique ID (TD-001, TD-002, etc.)
2. Describe the issue and its impact
3. Categorize and set severity
4. Note when/where it was introduced
5. Plan resolution approach

## Resolving Debt

1. Create exec plan in `docs/exec-plans/active/`
2. Implement fix
3. Move entry to **Resolved** section below
4. Archive exec plan to `docs/exec-plans/completed/`

## Resolved Debt

| ID | Description | Resolved | PR |
|----|-------------|----------|-----|
| - | - | - | - |

## Periodic Maintenance

This tracker supports the periodic agent workflow:
- Review weekly during sprint planning
- Prioritize High severity items
- Schedule dedicated debt-reduction sprints
