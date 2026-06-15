# Architecture

This document describes the architecture for the research code supporting the NeurIPS 2026 paper, "Not All Tokens Are Equally Useful for Steering: Robust Directions and Prefix Steering for Activation Steering."

## System Overview

`steering_geometry` is a research framework for activation steering experiments in causal language models. The codebase is scoped to the paper's experiments: extracting reliable steering directions, diagnosing which tokens produce useful directions, applying steering during generation, and evaluating the resulting behavior.

The framework studies four paper models:

- OLMo3-7B
- OLMo3-32B
- Qwen3-1.7B
- Qwen3-14B

It supports three paper concepts:

- safety, represented by refusal behavior
- sentiment
- politeness

The two central methods are Robust DiM and Prefix Steering. Robust DiM scores activation candidates with a relative margin so selected tokens produce more stable directions. Prefix Steering applies a steering vector only to the first `N` generated tokens, which tests whether early intervention can shape later generation without steering the full response.

Evaluation is limited to the paper benchmarks: HarmBench, LLM-as-judge with three labels, and MMLU-Pro.

## Core Components

| Component | Responsibility | Key File | Dependencies |
|-----------|----------------|----------|--------------|
| Package exports | Public package surface for common imports | `src/steering_geometry/__init__.py` | local modules |
| CLI shell evaluation | Small command-line entry point for shell-driven evaluation | `src/steering_geometry/__main__.py` | argparse, local modules |
| Types | Core data structures shared by extraction, steering, and evaluation | `src/steering_geometry/types.py` | dataclasses, typing |
| Configuration | Type-safe settings for the four paper models, three concepts, extraction, steering, and evaluation | `src/steering_geometry/config.py` | dataclasses, local types |
| Model wrapper | HuggingFace model loading, tokenizer handling, and activation hooks | `src/steering_geometry/models.py` | PyTorch, Transformers |
| Extraction | Steering vector extraction with DiM, PCA, weighted mean, and discriminative variants | `src/steering_geometry/extract.py` | PyTorch, NumPy, datasets |
| Steering and evaluation | Prefix Steering, full-scope steering, HarmBench, LLM-as-judge, and MMLU-Pro evaluation | `src/steering_geometry/apply_steering.py` | PyTorch, Transformers |
| Sweep evaluation | Strength × steer_tokens grid evaluation with concept-specific evaluators and MMLU-Pro, producing heatmap plots | `src/steering_geometry/sweep_evaluation.py` | PyTorch, Transformers, Matplotlib |
| Prefix analysis | Per-token KL divergence and attention pattern analysis diagnosing why Prefix Steering works | `src/steering_geometry/prefix_analysis.py` | PyTorch, NumPy, Matplotlib |
| Stability experiments | Cosine similarity sweeps for direction stability across methods, layers, and sample counts | `src/steering_geometry/stability_comparison.py` | NumPy, PyTorch, Matplotlib |
| Token selection experiments | Construction diagnosis experiments for token count, token position, prompt versus response tokens, and steering scope | `src/steering_geometry/token_selection_experiments.py` | PyTorch, NumPy, Matplotlib |
| Utilities | Shared filesystem, naming, sampling, and logging helpers | `src/steering_geometry/utils.py` | pathlib, logging |
| Typed package marker | PEP 561 marker for downstream type checking | `src/steering_geometry/py.typed` | none |

## Data Flow

The main workflow follows the paper experiment path from vector construction to behavioral evaluation.

1. **Configure experiment**
   - Select one of the four paper models.
   - Select one paper concept: safety, sentiment, or politeness.
   - Choose the extraction method and layer schedule.

2. **Load contrast data**
   - Build or load contrast pairs for the selected concept.
   - Keep prompt formatting consistent across models so activation comparisons isolate the target concept.

3. **Collect activations**
   - `HookedModel` loads the model and tokenizer.
   - Forward hooks capture layer activations for prompt and response tokens.
   - Token metadata is preserved so construction diagnosis can compare token count, position, and prompt versus response choices.

4. **Extract steering vectors**
   - Standard DiM computes difference-in-means directions.
   - PCA captures the dominant contrast direction.
   - Weighted mean uses scored token contributions.
   - Discriminative extraction ranks activation candidates.
   - Robust DiM uses relative-margin scoring to select activations before forming the final direction.

5. **Run stability experiments**
   - `stability_comparison.py` measures cosine similarity across sample counts, layers, and extraction settings.
   - The output supports the paper's claim that not all tokens produce equally stable steering directions.

6. **Run token selection experiments**
   - `token_selection_experiments.py` varies token count, token position, prompt versus response source, and steering scope.
   - These experiments diagnose why some construction choices produce stronger or more stable directions.

7. **Apply steering**
   - `apply_steering.py` injects steering vectors during generation.
   - Prefix Steering applies the vector only to the first `N` generated tokens.
   - Full-scope steering remains available as a comparison condition.

8. **Evaluate outputs**
   - HarmBench measures safety behavior.
   - LLM-as-judge assigns one of three labels for concept-specific behavior.
   - MMLU-Pro measures capability retention under steering.

## Technology Stack

| Category | Tool |
|----------|------|
| Language | Python 3.12+ |
| Deep learning | PyTorch |
| Model loading | Transformers |
| Package manager | uv |
| Linting and formatting | ruff |
| Type checking | mypy |
| Testing | pytest |

## Directory Structure

```text
.
├── src/steering_geometry/
│   ├── __init__.py
│   ├── __main__.py
│   ├── types.py
│   ├── config.py
│   ├── models.py
│   ├── extract.py
│   ├── apply_steering.py
│   ├── sweep_evaluation.py
│   ├── prefix_analysis.py
│   ├── stability_comparison.py
│   ├── token_selection_experiments.py
│   ├── utils.py
│   └── py.typed
├── scripts/
│   ├── apply_steering/
│   ├── extract/
│   ├── experiments/
│   ├── pipeline/
│   ├── prefix_analysis/
│   ├── stability_comparison/
│   ├── token_experiments/
│   └── vector_analysis/
├── tests/
│   ├── conftest.py
│   ├── test_apply_steering.py
│   ├── test_experiments.py
│   ├── test_stability_comparison.py
│   ├── test_stability_sweep.py
│   ├── test_sweep_evaluation.py
│   ├── test_token_selection_experiments.py
│   └── unit/
├── data/
├── outputs/
├── docs/
├── pyproject.toml
├── README.md
└── ARCHITECTURE.md
```

## File Placement Rules

| File Type | Location | Rule |
|-----------|----------|------|
| Python source modules | `src/steering_geometry/` | Importable package code lives here. |
| Shell scripts | `scripts/` | Orchestration scripts use `.sh` files only. |
| Tests | `tests/` | Test modules and fixtures stay outside `src/`. |
| Generated artifacts | `outputs/` | Vectors, evaluations, and PDF figures are not source files. |
| Input data | `data/` | Contrast data and benchmark inputs stay separate from code. |

## Key Design Decisions

### ADR-001: Modular Extraction Pipeline

- **Context**: The paper compares several construction methods and token selection choices across multiple models, concepts, and layers.
- **Decision**: Keep extraction in `extract.py`, model hooks in `models.py`, configuration in `config.py`, and evaluation in `apply_steering.py`.
- **Consequences**: Extraction methods can change without rewriting model loading or evaluation code. The same activation collection path supports DiM, PCA, weighted mean, discriminative extraction, and Robust DiM.

### ADR-002: Type-Safe Configuration

- **Context**: Paper experiments need repeatable model, concept, layer, steering, and evaluation settings.
- **Decision**: Represent supported models, concepts, and experiment options with typed configuration objects.
- **Consequences**: Invalid experiment settings fail early. Strict typing also keeps scripts and tests aligned with the paper scope.

### ADR-003: PDF-Only Visualization Output

- **Context**: Stability and token selection experiments produce figures for analysis and paper use.
- **Decision**: Save visualizations as PDF files only.
- **Consequences**: Figures remain publication-ready, file handling stays consistent, and scripts avoid multiple output branches for the same plot.

### ADR-004: Module Consolidation

- **Context**: The refactor narrowed the codebase to experiments needed for the paper.
- **Decision**: Keep steering and evaluation together in `apply_steering.py`, and keep vector stability experiments in `stability_comparison.py`.
- **Consequences**: The workflow is easier to follow: extract vectors, apply steering, evaluate results, and run stability comparisons from focused modules.

### ADR-005: CLI via python -m Pattern

- **Context**: Experiments are launched by shell scripts and direct command-line calls.
- **Decision**: CLI modules support invocation through `uv run python -m steering_geometry.<module>`.
- **Consequences**: Entrypoints work without separate console script registration. Shell scripts can call modules in a consistent way across local runs and batch jobs.

### ADR-006: Paper-Aligned Module Scope

- **Context**: The repository now supports a specific NeurIPS 2026 paper rather than a broad steering analysis toolkit.
- **Decision**: Keep only modules needed for the paper's four models, three concepts, two steering methods, construction diagnosis, stability sweeps, and benchmark evaluations.
- **Consequences**: Architecture, tests, scripts, and docs should not describe non-paper experiments. New modules require a clear link to Robust DiM, Prefix Steering, HarmBench, LLM-as-judge, MMLU-Pro, stability analysis, or token selection diagnosis.

## How to Update This Document

Update this file when:

- A paper experiment changes its data flow, supported models, supported concepts, or evaluation set.
- A source module in `src/steering_geometry/` is added, removed, renamed, or given a new responsibility.
- A script directory changes how extraction, steering, stability, or token selection experiments are run.
- A design decision changes the experiment scope or reproducibility contract.

When updating this document, keep it aligned with the actual file tree and the paper scope. Do not describe modules, benchmarks, models, or concepts that are outside the current implementation.
