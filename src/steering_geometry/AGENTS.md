# src/steering_geometry/

**Generated:** 2026-03-19
**Package:** `steering_geometry` — Steering vector extraction for LLM representation engineering

## OVERVIEW

Core extraction framework: load models → extract activations → compute steering vectors → apply steering → evaluate.

## STRUCTURE

```
steering_geometry/
├── __init__.py           # Exports: extract_vector, load_contrast_pairs, VALID_CONCEPTS
├── types.py              # 8 TypedDict + 6 dataclasses (ContrastPair, SteeringVector, etc.)
├── config.py             # 8 @dataclass configs (ModelConfig, ExtractionConfig, etc.)
├── models.py             # HookedModel — model loading, activation hooks, steering
├── extract.py            # Unified extraction CLI + 5 concept loaders + 4 aggregators
├── apply_steering.py     # Apply steering + evaluation (JudgeEvaluator, MMLUEvaluator merged here)
├── stability_comparison.py # Vector stability experiments (renamed from vector_analysis)
├── token_analysis.py     # Token-level analysis (visualize, probe subcommands)
├── unembed_analysis.py   # Unembedding cosine similarity analysis
├── tdnv.py               # TDNV separability metrics (Topic-Discriminative Normalized Variance)
└── utils.py              # Shared: validate_positive_int, sample_with_seed, ensure_dir, safe_model_name
```

## WHERE TO LOOK

| Task | Module | Key Functions |
|------|--------|---------------|
| Extract steering vector | `extract.py` | `extract_vector()`, `load_contrast_pairs()` |
| Add new concept | `extract.py` | Add loader to `_DATASET_LOADERS`, prefix constants |
| Add aggregation method | `extract.py` | Add to `_resolve_aggregator()` registry |
| Load model with hooks | `models.py` | `HookedModel`, `get_activations()`, `generate_with_steering()` |
| Apply steering | `apply_steering.py` | `apply_steering()` |
| Evaluate steering | `apply_steering.py` | `JudgeEvaluator`, `MMLUEvaluator` (lines 181, 321) |
| Add new type | `types.py` | Add dataclass or TypedDict |
| Add new config | `config.py` | Add @dataclass |
| Vector stability | `stability_comparison.py` | `run_diff_means_experiment()`, `run_discriminative_experiment()` |
| Token analysis | `token_analysis.py` | `visualize()`, `probe()` |
| Unembed analysis | `unembed_analysis.py` | `analyze_unembed_cosine()` |
| TDNV analysis | `tdnv.py` | `compute_tdnv()`, `compute_tdnv_for_concept()` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `VALID_CONCEPTS` | const | extract.py:77 | ["polite", "sentiment", "refusal"] |
| `ContrastPair` | dataclass | types.py:77 | Positive/negative text + metadata |
| `SteeringVector` | dataclass | types.py:94 | Layer activations + model/concept/method |
| `HookedModel` | class | models.py:20 | Model wrapper with activation hooks |
| `extract_vector` | func | extract.py:527 | Main extraction entry point |
| `extract_steering_vector` | func | extract.py:458 | Core extraction logic |
| `apply_steering` | func | apply_steering.py:536 | Apply + evaluate steering |
| `JudgeEvaluator` | class | apply_steering.py:181 | LLM-as-judge evaluation |
| `MMLUEvaluator` | class | apply_steering.py:321 | Benchmark evaluation |
| `compute_tdnv` | func | tdnv.py:73 | TDNV metric calculation |
| `run_diff_means_experiment` | func | stability_comparison.py:179 | Differential means stability |
| `run_discriminative_experiment` | func | stability_comparison.py:312 | Discriminative token selection |

## CONVENTIONS (This Package)

### Module Execution Pattern
All CLI modules follow this pattern:
```python
class _Args(Protocol):
    arg_name: type

def _build_parser() -> argparse.ArgumentParser: ...
def main() -> None: ...

if __name__ == "__main__":
    main()
```

### Aggregator Pattern
```python
Aggregator = Callable[[Tensor, Tensor], Tensor]

def mean_aggregator(pos: Tensor, neg: Tensor) -> Tensor: ...
def pca_aggregator(pos: Tensor, neg: Tensor) -> Tensor: ...

AGGREGATORS = {"mean": mean_aggregator, "pca": pca_aggregator}
```

### Layer Resolution
Layers specified as relative positions 0.0-1.0:
```python
layers = model.resolve_layers([0.4, 0.6])  # Returns absolute indices
```

### Config Usage
```python
config = ExtractionConfig(layers=[0.4, 0.6], method="mean", batch_size=8)
```

## ANTI-PATTERNS (Found Here)

| File | Issue | Fix |
|------|-------|-----|
| `models.py:90,174,229,240` | `Any` in hook params | Use Protocol or specific tensor types |
| `stability_comparison.py:17,543,589,618` | `Any` in result dicts | Use TypedDict |
| `unembed_analysis.py:12,38` | `Any` for tokenizer | Use Protocol |
| `apply_steering.py:32,328,607` | `Any` for model/config | Use specific types |
| `extract.py` + `tdnv.py` | `_select_token_activations` duplicated | Extract to utils.py |
| `extract.py`, `apply_steering.py`, `tdnv.py`, `token_analysis.py` | `print()` for CLI output | Use logging module |

## IMPORT GRAPH

```
__init__.py
    └── extract.py

extract.py
    ├── config.py (ExtractionConfig, ModelConfig, ConceptConfig)
    ├── models.py (HookedModel)
    ├── types.py (ContrastPair, SteeringVector)
    └── utils.py (ensure_dir, safe_model_name, sample_with_seed)

apply_steering.py
    ├── config.py (SteeringConfig, EvaluationConfig)
    ├── models.py (HookedModel)
    └── types.py (EvaluationResult, SteeringVector, JudgeScore, MMLUResult)
    (JudgeEvaluator and MMLUEvaluator are defined HERE, not in separate file)

stability_comparison.py
    ├── config.py (ExtractionConfig)
    ├── extract.py (load_contrast_pairs)
    ├── models.py (HookedModel)
    └── types.py (SteeringVector)

tdnv.py
    ├── config.py (TDNVConfig)
    ├── extract.py (load_contrast_pairs)
    └── models.py (HookedModel)

token_analysis.py
    ├── config.py (TokenAnalysisConfig)
    ├── models.py (HookedModel)
    └── types.py (TokenRecord, DiscriminativeTokenResult, ProbeExperimentResult)

unembed_analysis.py
    ├── config.py (UnembedAnalysisConfig)
    ├── models.py (HookedModel)
    └── types.py (UnembedAnalysisResult, ConceptAnalysisResult)
```
