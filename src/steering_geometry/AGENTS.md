# src/steering_geometry/

**Package:** `steering_geometry` — Steering vector extraction for LLM representation engineering

## OVERVIEW

Core extraction framework: load models → extract activations → compute steering vectors → apply steering → evaluate.

## STRUCTURE

```
steering_geometry/
├── __init__.py      # Exports: extract_vector, load_contrast_pairs, VALID_CONCEPTS
├── types.py         # 8 TypedDict + 6 dataclasses (ContrastPair, SteeringVector, etc.)
├── config.py        # 8 @dataclass configs (ModelConfig, ExtractionConfig, etc.)
├── models.py        # HookedModel — model loading, activation hooks, steering
├── extract.py       # Unified extraction CLI + 5 concept loaders + 4 aggregators
├── apply_steering.py # Apply steering vectors + evaluation orchestration
├── evaluation.py    # JudgeEvaluator, MMLUEvaluator, HTML report generation
├── tdnv.py          # TDNV separability metrics (Topic-Discriminative Normalized Variance)
└── utils.py         # Shared: validate_positive_int, sample_with_seed, ensure_dir, safe_model_name
```

## WHERE TO LOOK

| Task | Module | Key Functions |
|------|--------|---------------|
| Extract steering vector | `extract.py` | `extract_vector()`, `load_contrast_pairs()` |
| Add new concept | `extract.py` | Add loader to `_DATASET_LOADERS`, prefix constants |
| Add aggregation method | `extract.py` | Add to `_resolve_aggregator()` registry |
| Load model with hooks | `models.py` | `HookedModel`, `get_activations()`, `generate_with_steering()` |
| Apply steering | `apply_steering.py` | `apply_steering()` |
| Evaluate steering | `evaluation.py` | `JudgeEvaluator`, `MMLUEvaluator` |
| Add new type | `types.py` | Add dataclass or TypedDict |
| Add new config | `config.py` | Add @dataclass |
| TDNV analysis | `tdnv.py` | `compute_tdnv()`, `compute_tdnv_for_concept()` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `VALID_CONCEPTS` | const | extract.py:71 | ["honesty", "sycophancy", "toxicity", "sentiment", "refusal"] |
| `ContrastPair` | dataclass | types.py:78 | Positive/negative text + metadata |
| `SteeringVector` | dataclass | types.py:95 | Layer activations + model/concept/method |
| `HookedModel` | class | models.py:20 | Model wrapper with activation hooks |
| `extract_vector` | func | extract.py:550 | Main extraction entry point |
| `extract_steering_vector` | func | extract.py:481 | Core extraction logic |
| `apply_steering` | func | apply_steering.py:72 | Apply + evaluate steering |
| `JudgeEvaluator` | class | evaluation.py:159 | LLM-as-judge evaluation |
| `MMLUEvaluator` | class | evaluation.py:299 | Benchmark evaluation |
| `compute_tdnv` | func | tdnv.py:73 | TDNV metric calculation |

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
| `models.py:133,187` | `Any` in hook params | Use Protocol or specific tensor types |
| `extract.py` + `tdnv.py` | `_select_token_activations` duplicated | Extract to utils.py |
| `extract.py:573+` | `print()` for CLI output | Use logging module |

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
    ├── evaluation.py (JudgeEvaluator, MMLUEvaluator)
    ├── models.py (HookedModel)
    └── types.py (EvaluationResult, SteeringVector)

tdnv.py
    ├── config.py (TDNVConfig)
    ├── extract.py (load_contrast_pairs)
    └── models.py (HookedModel)
```
