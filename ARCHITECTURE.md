# Architecture

This document describes the system architecture for the Steering Geometry project.

## System Overview

The Steering Geometry project provides tools for analyzing and visualizing vehicle steering geometry parameters. It focuses on mathematical modeling and visualization of steering systems, including Ackermann geometry, turning radius calculations, and steering angle relationships.

## Core Components

| Component | Responsibility | Key Files | Dependencies |
|-----------|---------------|-----------|--------------|
| Core Calculations | Steering geometry math and physics | `src/steering_geometry/` | numpy, scipy |
| Visualization | Plot generation and graphical output | `scripts/`, `plot/` | matplotlib |
| Data Processing | Input/output handling | `data/`, `assets/` | pyyaml |
| Testing | Unit tests and validation | `tests/` | pytest |

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
├── src/steering_geometry/  # Source code
│   ├── __init__.py         # Package init
│   ├── py.typed            # PEP 561 marker
│   └── *.py                # Modules
├── tests/                  # Test files
├── scripts/                # Executable analysis scripts
├── data/                   # Input data files
├── plot/                   # Generated plots and visualizations
├── assets/                 # Analysis results and outputs
├── docs/                   # Documentation
│   ├── design-docs/        # Design documents
│   └── exec-plans/         # Execution plans
└── .github/                # GitHub configs
```

## Directory Purposes

- **src/steering_geometry/** - Core Python modules for steering geometry calculations
- **scripts/** - Standalone scripts for running analyses and generating reports
- **data/** - Input data files (vehicle parameters, measurement data)
- **plot/** - Generated plots, graphs, and visualizations (PNG, PDF, SVG)
- **assets/** - Analysis results, reports, and calculated outputs
- **tests/** - Unit tests and integration tests

## Key Design Decisions

### ADR-001: Project Structure for Research Code

- **Context**: Need to organize code for both development and research outputs
- **Decision**: Separate directories for source code (src/), scripts (scripts/), data (data/), plots (plot/), and results (assets/)
- **Consequences**: Clear separation between code and outputs, easy to version control data and track results

### ADR-002: Python 3.12+ with Modern Tooling

- **Context**: Need modern Python features and reliable tooling
- **Decision**: Use Python 3.12+, uv for package management, ruff for linting, mypy for type checking
- **Consequences**: Better type safety, faster dependency management, consistent code style

## How to Update This Document

Update this file when:
- Adding new major components or modules
- Changing technology stack
- Making significant architectural decisions
- Adding new data processing pipelines
- Introducing new visualization methods
