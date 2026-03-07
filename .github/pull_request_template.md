## Summary

<!-- 2-3 sentences describing the changes -->

## Linked Issue (Optional)

Closes #

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor
- [ ] Documentation
- [ ] Test
- [ ] CI/CD
- [ ] Other

## What Changed

<!-- List specific file-level changes -->

## Verification Performed

### Lint
```
<output of uv run ruff check src/ tests/ >
```

### Type Check
```
<output of uv run mypy src/ >
```

### Tests
```
<output of uv run pytest>
```

## Self-Review Checklist

- [ ] Code follows project style (ruff passes)
- [ ] Types are correct (mypy passes)
- [ ] Tests pass and cover new behavior
- [ ] No secrets or credentials in code
- [ ] No unnecessary dependencies added
- [ ] Documentation updated if needed
- [ ] Changes are within the scope of the linked issue

## Agent Metadata

```yaml
agent:
model:
exec_plan:
files_changed:
```

## Risks & Considerations

<!-- Known risks, trade-offs, areas needing extra review -->

## Human Reviewer Checklist

- [ ] Intent matches the issue/plan
- [ ] No scope creep beyond the plan
- [ ] Architecture decisions are sound
- [ ] Edge cases considered
