# Architecture

This document describes the system architecture for the AI Steering Vector Extraction Framework.

## System Overview

The framework provides tools for extracting and analyzing steering vectors in Large Language Models (LLMs). It focuses on identifying activation patterns that correspond to specific behavioral concepts (e.g., honesty, toxicity) and using these patterns to steer model behavior.

## Core Components

| Component | Responsibility | Key Files | Dependencies |
|-----------|---------------|-----------|--------------|
| Types | Core data structures and type hints | `src/steering_geometry/types.py` | typing |
| Configuration | Model and extraction settings | `src/steering_geometry/config.py` | pydantic |
| Model Wrapper | LLM loading and activation hooks | `src/steering_geometry/models.py` | torch, transformers |
| Extraction | Activation collection and vector calculation | `src/steering_geometry/extraction.py` | numpy, scikit-learn |
| Evaluation | Steering effect validation and metrics | `src/steering_geometry/evaluation.py` | scipy |
| Concepts | Concept-specific datasets and prompts | `src/steering_geometry/concepts/` | pyyaml |

## Data Flow

The extraction pipeline follows a linear data flow:

1. **Dataset Selection**: Choose a behavioral concept (e.g., Honesty).
2. **Contrast Pairs**: Generate or load pairs of prompts that differ only in the target concept (e.g., "Tell the truth" vs. "Lie").
3. **Activation Extraction**: Run the LLM on both prompts and collect activations from specific layers.
4. **Steering Vector Calculation**: Compute the difference in activations (e.g., via mean difference or PCA) to find the steering vector.
5. **Evaluation**: Apply the steering vector to new prompts and measure the change in model behavior.

## Technology Stack

| Category | Tool | Version |
|----------|------|---------|
| Language | Python | 3.12+ |
| Deep Learning | PyTorch | 2.0+ |
| LLM Library | Transformers | 4.40+ |
| Package Manager | uv | latest |
| Linting | ruff | 0.8+ |
| Type Checking | mypy | 1.13+ |
| Testing | pytest | 8.0+ |

## Directory Structure

```
.
├── src/steering_geometry/  # Source code
│   ├── concepts/           # Concept-specific logic and data
│   ├── types.py            # Core type definitions
│   ├── config.py           # Configuration management
│   ├── models.py           # Model loading and wrapping
│   ├── extraction.py       # Activation extraction logic
│   └── evaluation.py       # Vector evaluation and validation
├── tests/                  # Test files
├── scripts/                # Executable extraction scripts
├── data/                   # Raw datasets and contrast pairs
├── plot/                   # Visualizations of activation spaces
├── assets/                 # Saved steering vectors and results
├── docs/                   # Documentation
│   ├── design-docs/        # Design documents
│   └── exec-plans/         # Execution plans
└── .github/                # GitHub configs
```

## Key Design Decisions

### ADR-001: Modular Extraction Pipeline

- **Context**: Need to support multiple concepts and extraction methods.
- **Decision**: Separate extraction logic from model wrapping and evaluation.
- **Consequences**: Easier to add new concepts or test different vector calculation methods without modifying the core model logic.

### ADR-002: Type-Safe Configuration

- **Context**: LLM experiments require many hyperparameters.
- **Decision**: Use Pydantic for configuration management and strict type hints throughout the codebase.
- **Consequences**: Reduced runtime errors and better IDE support for experiment configuration.

## How to Update This Document

Update this file when:
- Adding new major components or modules
- Changing the extraction or evaluation pipeline
- Making significant architectural decisions
- Updating the technology stack
