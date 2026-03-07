# AI Research Template

A Python template optimized for AI-assisted research projects. Designed for experiments, prototypes, and research codebases where AI agents handle implementation.

## What This Is

This template provides a complete foundation for **research projects** where AI agents (Claude Code, Opencode, Aider, etc.) handle implementation. It emphasizes:

- **Rapid experimentation** — Minimal boilerplate, quick iteration cycles
- **Reproducibility** — Structured documentation for experiments and results
- **AI-friendly structure** — Clear conventions that AI agents understand
- **Quality tracking** — Built-in quality metrics for research code

## Quick Start

1. **Use this template**
   - Click "Use this template" on GitHub, or
   - Clone: `git clone <repo-url>`

2. **Rename the package**
   ```bash
   # Replace 'ai_research_template' with your project name
   find . -type f -name "*.py" -o -name "*.toml" | xargs sed -i 's/ai_research_template/your_project_name/g'
   mv src/ai_research_template src/your_project_name
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Verify everything works**
   ```bash
   uv run ruff check src/ tests/
   uv run mypy src/
   uv run pytest
   ```

## What's Included

- **AGENTS.md** — AI agent operational policy (commands, escalation rules)
- **ARCHITECTURE.md** — System design template
- **docs/** — Documentation structure (design-docs, exec-plans, quality tracking)
- **pyproject.toml** — Configured with ruff, mypy, pytest
- **.github/** — CI workflow, PR template, issue templates

## Research Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Define     │───▶│  Document   │───▶│  AI reads   │
│  hypothesis │    │  experiment │    │    docs     │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Review     │◀───│  AI opens   │◀───│  AI codes,  │
│  results    │    │     PR      │    │  tests,     │
└─────────────┘    └─────────────┘    │  verifies   │
       │                              └─────────────┘
       ▼
┌─────────────┐
│  Document   │───▶ Repeat for next experiment
│  findings   │
└─────────────┘
```

## File Structure

```
.
├── AGENTS.md                    # AI agent instructions
├── ARCHITECTURE.md              # System design
├── README.md                    # This file
├── pyproject.toml               # Project config
├── .python-version              # Python version
├── src/ai_research_template/    # Source code
├── tests/                       # Test files
├── docs/
│   ├── design-docs/             # Design documents
│   ├── exec-plans/              # Execution plans
│   ├── PLANS.md                 # Project roadmap
│   └── QUALITY_SCORE.md         # Quality tracking
└── .github/
    ├── workflows/ci.yml         # CI pipeline
    ├── pull_request_template.md
    └── ISSUE_TEMPLATE/          # Issue templates
```

## Renaming Guide

After cloning, replace all instances of `ai_research_template`:

1. **pyproject.toml**: Change `name = "ai_research_template"`
2. **src/ai_research_template/**: Rename directory
3. **Imports**: Update all `from ai_research_template` imports
4. **AGENTS.md**: Update package references

## Research-Specific Features

- **Experiment tracking**: Use `docs/design-docs/` for hypothesis and experiment documentation
- **Quality metrics**: `docs/QUALITY_SCORE.md` tracks code quality over time
- **Execution plans**: `docs/exec-plans/` structure your research iterations

## Related Templates

- **ai-general-template** — For general-purpose projects (coming soon)

## License

[Add your license here]
