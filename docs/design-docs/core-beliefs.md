# Core Engineering Beliefs

These beliefs guide development in this project.

## 1. Code is Read More Than Written

**Explanation**: Every line of code will be read many more times than it was written. Clarity always beats cleverness.

**Practical Implication**: Prefer explicit over implicit. Name things for what they do, not how they do it.

## 2. Tests are Documentation

**Explanation**: Tests describe intended behavior. They're the most reliable documentation because they're executable and must pass.

**Practical Implication**: Write tests that tell a story. Test names should describe the behavior being verified.

## 3. Automate Everything Repeatable

**Explanation**: If an AI agent can verify it, it should be automated. Manual verification doesn't scale.

**Practical Implication**: Every verification step should be a command in AGENTS.md that exits 0 on success.

## 4. Small, Frequent Changes

**Explanation**: Small PRs are easier to review, easier to revert, and cause fewer merge conflicts.

**Practical Implication**: If a change touches more than 5 files, consider splitting it.

## 5. Fail Fast, Fail Loud

**Explanation**: Errors should surface immediately, not silently. Silent failures are the worst kind.

**Practical Implication**: Use strict mode in type checkers. Fail on warnings in CI. Never catch and swallow exceptions silently.

## 6. Documentation is Code

**Explanation**: Docs are versioned, reviewed, and maintained like code. Stale docs are bugs.

**Practical Implication**: Update docs in the same PR as code changes. Review docs for accuracy.

## 7. Trust but Verify

**Explanation**: AI writes code, humans verify intent. The AI is a force multiplier, not a replacement for judgment.

**Practical Implication**: Always review AI-generated code. Focus on intent, not just syntax.
