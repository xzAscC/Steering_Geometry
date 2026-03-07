# Architecture

This document describes the system architecture for this project.

## System Overview

[Describe the high-level purpose and architecture of the system.]

## Core Components

| Component | Responsibility | Key Files | Dependencies |
|-----------|---------------|-----------|--------------|
| [Name] | [What it does] | [Files] | [Deps] |

## Technology Stack

| Category | Tool | Version |
|----------|------|---------|
| Language | Python | 3.12+ |
| Package Manager | uv | latest |
| Linting | ruff | 0.8+ |
| Type Checking | mypy | 1.13+ |
| Testing | pytest | 8.0+ |
| CI | GitHub Actions | N/A |

## Directory Structure

```
.
├── src/ai_research_template/  # Source code
│   ├── __init__.py     # Package init
│   ├── py.typed        # PEP 561 marker
│   └── *.py            # Modules
├── tests/              # Test files
├── docs/               # Documentation
│   ├── design-docs/    # Design documents
│   └── exec-plans/     # Execution plans
└── .github/            # GitHub configs
```

## Key Design Decisions

### ADR-001: [Decision Title]

- **Context**: [Why this decision was needed]
- **Decision**: [What was decided]
- **Consequences**: [Impact of this decision]

## How to Update This Document

Update this file when:
- Adding new major components
- Changing technology stack
- Making significant architectural decisions
