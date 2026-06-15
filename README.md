# RobustDiM-PrefixSteering

[![Paper](https://img.shields.io/badge/paper-PDF-red)](docs/paper.pdf)
[![Project Page](https://img.shields.io/badge/🌐-Project_Page-3273dc)](https://xzascc.github.io/RobustDiM-PrefixSteering/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code accompanying the arXiv preprint *"Not All Tokens Are Equally Useful for Steering: Robust Directions and Prefix Steering"*.

This repository implements the paper's two activation-steering methods — **Robust DiM** for token-selective steering-direction construction and **Prefix Steering** for early-token intervention — together with experiments on safety/refusal, sentiment, and politeness across four models, evaluated with steering performance and general ability.

## Paper Scope

- **Models**: OLMo3-7B (`allenai/Olmo-3-1025-7B`), OLMo3-32B (`allenai/Olmo-3-1125-32B`), Qwen3-1.7B (`Qwen/Qwen3-1.7B`), Qwen3-14B (`Qwen/Qwen3-14B`)
- **Concepts**: safety/refusal, sentiment, politeness
- **Datasets**: `LLM-LAT/benign-dataset` + `LLM-LAT/harmful-dataset` (safety), SST-2 (sentiment), PoliteGuard (politeness)
- **Evaluations**: HarmBench (safety), 3-label LLM-as-judge (sentiment/politeness), MMLU-Pro (capability retention)

## Methods

- **Robust DiM** — token-selective steering-direction construction: score each candidate activation by its relative margin between class means, keep the top-K per class, and average. Filters out non-target-dominated activations before mean estimation.
- **Prefix Steering** — apply the steering update only to an initial prefix of generated tokens. Most steering effect emerges early, so restricting intervention there retains control while reducing the distributional shift that degrades general capability.

## Quick Start

```bash
uv sync                                                                    # install dependencies
uv run python -m steering_geometry.extract \
    --concept sentiment --model "Qwen/Qwen3-1.7B"                          # extract a steering vector
uv run python -m steering_geometry.apply_steering \
    --vector outputs/vectors/sentiment/layer0.7.pt                         # apply a saved vector + evaluate
uv run python -m steering_geometry --shell                                 # print package metadata / shell config
```

The experiment modules are driven by the shell scripts in `scripts/` (see [Project Structure](#project-structure)) rather than standalone CLIs.

## Project Structure

The tree below reflects the actual contents of the repository. Every tracked file and folder is annotated; runtime/cache artifacts (`__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.venv/`, `.git/`) are omitted for brevity.

```text
# ─── Root configuration & docs ──────────────────────────────────────────────
pyproject.toml              # Hatchling build config, ruff (E/F/I/UP/B/SIM/N, line 100), mypy --strict, pytest
uv.lock                     # Locked dependency set for reproducible `uv sync` installs
requirements.in             # High-level pip requirements (source of requirements.txt)
requirements.txt            # Generated pip requirements (compatibility / non-uv environments)
.python-version             # Pins Python 3.12 for uv / pyenv
.env.example                # Template env file (OPENROUTER_API_KEY for LLM-as-judge)
.gitignore                  # Ignores outputs/, logs/, .venv/, caches, .env, etc.
.ignore                     # Extra ignore rules for ripgrep / fuzzy finders
LICENSE                     # MIT license text
README.md                   # This file
AGENTS.md                   # Rules every AI agent working in this repo must follow
ARCHITECTURE.md             # System-level design and module relationships
SPEC.md                     # Project specification / scope description
src/steering_geometry/
├── __init__.py                       # Public package interface; re-exports key symbols
├── __main__.py                       # `python -m steering_geometry` entry; `--shell` prints shell config (print() allowed here)
├── types.py                          # Paper domain objects (dataclasses) and result TypedDicts
├── config.py                         # Four paper models, three concepts, extraction & evaluation configs
├── models.py                         # `HookedModel`: loads a HF causal LM with forward hooks on residual layers
├── extract.py                        # Robust DiM extraction: contrast-pair loading, margin scoring, top-K averaging
├── apply_steering.py                 # Prefix Steering application + HarmBench / LLM-as-judge / MMLU-Pro evaluation
├── sweep_evaluation.py               # Strength × steer_tokens grid sweep with HarmBench/LLM-as-judge and MMLU-Pro
├── prefix_analysis.py                # KL divergence and attention-pattern analysis for Prefix Steering diagnostics
├── stability_comparison.py           # Robust DiM stability sweeps and vector comparison helpers
├── token_selection_experiments.py    # Construction diagnosis: token position, prompt vs response, example count, scope
├── utils.py                          # Shared helpers: ensure_dir(), safe_model_name(), sample_with_seed(), configure_logging()
└── py.typed                          # PEP 561 marker declaring this package is typed
scripts/
├── extract/
│   └── quick_discriminative.sh       # Quick Robust DiM extraction run across a concept/model
├── apply_steering/
│   └── run_steering.sh               # Apply a saved vector and run the evaluation path
├── pipeline/
│   └── quick_pipeline.sh             # End-to-end: extract → steer → evaluate
├── prefix_analysis/
│   ├── run_kl_divergence.sh           # KL-only analysis (fast, no eager-attention reload)
│   ├── run_analysis.sh               # KL divergence + attention-pattern analysis for one concept
│   └── run_all_concepts.sh           # Loop run_analysis.sh over safety/sentiment/politeness
├── token_experiments/
│   ├── 1_token_count.sh              # Construction diagnosis: number of selected tokens
│   ├── 2_token_position.sh           # Construction diagnosis: token position effect
│   ├── 3_prompt_vs_response.sh       # Construction diagnosis: prompt vs response tokens
│   └── 4_steering_scope.sh           # Prefix Steering: steering scope × prefix length
├── vector_analysis/
│   ├── run_stability_sweep.sh        # Generate stability-sweep raw data
│   ├── plot_stability_sweep.sh       # Plot stability-sweep figures
│   ├── run_stability_comparison.sh   # Compare vector stability across construction methods
│   ├── run_k_ablation.sh             # Ablation over the top-K margin parameter
│   ├── run_candidate_pool_ablation.sh# Ablation over the candidate activation pool
│   ├── quick_discriminative_heatmaps.sh  # Generate Robust DiM margin heatmaps
│   └── quick_diff_means_heatmaps.sh  # Generate diff-of-means comparison heatmaps
├── experiments/
│   └── steering_strength_prefix_sweep.sh  # Full steering-strength × prefix-length sweep
└── stability_comparison/
    └── quick_vector_stability.sh     # Quick Robust DiM vector stability run
assets/                               # Static binary assets (figures/screenshots); currently empty (.gitkeep)
data/
├── vectors/                          # Pre-saved steering vectors
└── steered/                          # Example steered-generation outputs
outputs/                              # Run outputs (git-ignored in practice; see .gitignore)
├── vectors/                          # Extracted steering vectors per concept/model
├── steering/                         # Steering-run JSON results (prefix_steering_*.json)
├── stability/                        # Vector-stability experiment results
├── stability_sweep/                  # Stability-sweep raw data
├── heatmaps/                         # Generated margin / diff-of-means heatmaps
├── prefix_analysis/                  # KL divergence and attention-analysis outputs
├── prefix_vs_full/                   # Prefix vs. full-steering comparison (sentiment/politeness)
├── prefix_vs_full_refusal/           # Prefix vs. full-steering comparison (refusal)
└── posters/                          # Generated poster / summary artifacts
logs/                                 # Timestamped run logs (steering_YYYYMMDD_HHMMSS.log); runtime artifacts
```

## Checks

This is a research codebase for a paper — contributions aren't expected. If you want to verify your environment reproduces the checked-in state, run the same checks CI uses:

```bash
uv run ruff check src/ tests/          # lint
uv run ruff format --check src/ tests/ # format check
uv run mypy src/                       # type check (strict)
uv run pytest                          # tests
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
```

## License

MIT License. See `LICENSE` for details. The project page under `docs/` is adapted from [Nerfies](https://nerfies.github.io) and licensed separately under CC BY-SA 4.0; see `docs/LICENSE`.
