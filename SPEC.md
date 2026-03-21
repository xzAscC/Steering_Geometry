# Technical Specification

**Generated:** 2026-03-19
**Version:** 0.1.0
**Branch:** refactor/architecture

This document specifies the technical interfaces, data formats, and protocols for the Steering Geometry framework.

---

## 1. Supported Concepts

The framework extracts steering vectors for five behavioral concepts:

| Concept | Description | Dataset Source |
|---------|-------------|----------------|
| `honesty` | Truthful vs. deceptive responses | `真理/TruthfulQA` |
| `sycophancy` | Objective vs. user-aligned responses | `EleutherAI/sycophancy` |
| `toxicity` | Non-toxic vs. toxic language | `SkolkovoInstitute/toxic_comments` |
| `sentiment` | Positive vs. negative sentiment | `stanfordnlp/sst2` |
| `refusal` | Compliance vs. refusal behavior | `HuggingFaceH4/ultrafeedback_binarized` |

**Constant:** `VALID_CONCEPTS = ["honesty", "sycophancy", "toxicity", "sentiment", "refusal"]`

---

## 2. Aggregation Methods

Four methods for computing steering vectors from contrast pairs:

| Method | Function | Description |
|--------|----------|-------------|
| `mean` | `mean_aggregator()` | Simple mean difference: `pos.mean() - neg.mean()` |
| `pca` | `pca_aggregator()` | First principal component of concatenated deltas |
| `weighted_mean` | `weighted_mean_aggregator()` | Inverse-variance weighted mean |
| `discriminative` | `discriminative_token_aggregator()` | Top-K discriminative tokens by Fisher score |

**Default:** `mean`

---

## 3. Layer Specification

Layers are specified as relative positions (0.0 to 1.0):

| Layer Spec | Absolute Index (24-layer model) |
|------------|--------------------------------|
| 0.0 | Layer 0 (embedding) |
| 0.25 | Layer 6 |
| 0.5 | Layer 12 (middle) |
| 0.75 | Layer 18 |
| 1.0 | Layer 23 (final) |

**Resolution:** `model.resolve_layers([0.4, 0.6])` → absolute indices

---

## 4. Data Formats

### 4.1 SteeringVector

```python
@dataclass
class SteeringVector:
    layer_activations: dict[int, Tensor]  # {layer_idx: steering_vector}
    model_name: str                       # "Qwen/Qwen3.5-2B"
    concept: str                          # "honesty"
    method: str                           # "mean"
```

**Storage:** PyTorch `.pt` files via `torch.save()`

**File naming:** `{concept}/{method}/n{count}_layer{frac}.pt`

### 4.2 ContrastPair

```python
@dataclass
class ContrastPair:
    positive: str                    # "I always provide facts."
    negative: str                    # "I might make up answers."
    metadata: ContrastPairMetadata   # concept, dataset, source, pair_index
```

### 4.3 EvaluationResult

```python
@dataclass
class EvaluationResult:
    judge_scores: list[JudgeScore]   # Per-sample LLM-as-judge scores
    mmlu_result: MMLUResult | None   # MMLU benchmark results
    metadata: EvaluationMetadata     # model, concept, layer, multiplier
```

---

## 5. CLI Interfaces

### 5.1 Extract

```bash
uv run python -m steering_geometry.extract \
    --concept <concept> \
    --model <model_name> \
    [--method <mean|pca|weighted_mean|discriminative>] \
    [--num-pairs <N>] \
    [--top-k <K>] \
    [--layers <0.4,0.6,0.8>] \
    [--output <dir>]
```

**Required:** `--concept`, `--model`

### 5.2 Apply Steering

```bash
uv run python -m steering_geometry.apply_steering \
    --vector <path.pt> \
    --model <model_name> \
    [--output <dir>] \
    [--samples <N>] \
    [--multipliers <1.0,1.5,2.0>] \
    [--evaluate] \
    [--judge-model <model>] \
    [--mmlu-questions <N>]
```

**Required:** `--vector`, `--model`

### 5.3 Token Analysis

```bash
uv run python -m steering_geometry.token_analysis \
    (visualize | probe) \
    --concept <concept> \
    --model <model_name> \
    [--output <dir>]
```

**Subcommands:** `visualize`, `probe`

### 5.4 TDNV Metrics

```bash
uv run python -m steering_geometry.tdnv \
    --concept <concept> \
    --model <model_name> \
    [--num-pairs <N>]
```

### 5.5 Unembed Analysis

```bash
uv run python -m steering_geometry.unembed_analysis \
    --concept <concept> \
    --model <model_name> \
    [--top-k <K>]
```

---

## 6. Shell Script Interface

### 6.1 Pipeline Script

```bash
./scripts/pipeline/run_pipeline.sh \
    -c <concepts> \           # comma-separated: honesty,toxicity
    -m <models> \             # comma-separated: Qwen/Qwen3.5-2B
    [-l <layers>] \           # comma-separated: 0.4,0.6
    [--extract-only] \        # Skip steering and evaluation
    [--steer-only] \          # Skip extraction (use existing)
    [--eval-only]             # Skip extraction and steering
```

### 6.2 Quick Scripts

```bash
# Quick extraction (single concept/layer)
./scripts/quick/quick_extract.sh -c honesty -l 0.7

# Quick steering (single concept/layer)
./scripts/quick/quick_steering.sh -c honesty -l 0.7

# Quick evaluation (single concept/layer)
./scripts/quick/quick_eval.sh -c honesty -l 0.7
```

---

## 7. Output Structure

```
outputs/
├── vectors/
│   └── {concept}/
│       ├── diff_means/
│       │   └── n{count}_layer{frac}.pt
│       └── discriminative/
│           └── k{K}_layer{frac}.pt
├── heatmaps/
│   ├── diff_means/
│   │   └── {concept}_layer{frac}.pdf
│   └── discriminative/
│       └── {concept}_layer{frac}.pdf
├── token_analysis/
│   └── {concept}/
│       ├── visualize/
│       └── probe/
├── unembed_analysis/
│   ├── plots/
│   └── json/
└── stability/
    └── {experiment}/
```

---

## 8. Configuration Classes

### 8.1 ModelConfig

```python
@dataclass
class ModelConfig:
    model_name: str           # "Qwen/Qwen3.5-2B"
    device: str = "auto"      # "cpu", "cuda", "auto"
    dtype: str = "auto"       # "float16", "float32", "auto"
```

### 8.2 ExtractionConfig

```python
@dataclass
class ExtractionConfig:
    layers: list[float]       # [0.4, 0.6, 0.8]
    method: str = "mean"      # "mean", "pca", "weighted_mean", "discriminative"
    batch_size: int = 8
    top_k: int = 64           # For discriminative method
```

### 8.3 SteeringConfig

```python
@dataclass
class SteeringConfig:
    multipliers: list[float]  # [1.0, 1.5, 2.0]
    max_new_tokens: int = 64
    temperature: float = 1.0
    num_samples: int = 10
```

---

## 9. API Reference

### 9.1 extract_vector()

```python
def extract_vector(
    concept: str,
    model_name: str,
    num_pairs: int = 500,
    method: str = "mean",
    layers: list[float] | None = None,
    batch_size: int = 8,
) -> SteeringVector:
    """Extract steering vector for a concept."""
```

### 9.2 load_contrast_pairs()

```python
def load_contrast_pairs(
    concept: str,
    num_pairs: int = 500,
) -> list[ContrastPair]:
    """Load contrast pairs for a concept."""
```

### 9.3 apply_steering()

```python
def apply_steering(
    vector_path: str,
    model_name: str,
    output_dir: str,
    config: SteeringConfig,
    evaluate: bool = False,
) -> EvaluationResult | None:
    """Apply steering vector to model and optionally evaluate."""
```

---

## 10. Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `ValueError: Invalid concept` | Unknown concept name | Use one of `VALID_CONCEPTS` |
| `ValueError: No dataset loader` | Missing loader in `_DATASET_LOADERS` | Add loader function |
| `RuntimeError: CUDA out of memory` | Batch size too large | Reduce `batch_size` |
| `FileNotFoundError: Vector not found` | Missing steering vector file | Run extraction first |

---

## 11. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.1,<3.0 | Tensor operations, model inference |
| transformers | >=4.36,<5.0 | HuggingFace model loading |
| datasets | >=2.16,<3.0 | Dataset loading |
| numpy | >=1.26,<3.0 | Numerical operations |
| scikit-learn | >=1.4,<2.0 | PCA, logistic regression |
| accelerate | >=1.13.0 | Model parallelism |
| openai | >=1.0.0,<2.0 | LLM-as-judge evaluation |
| matplotlib | >=3.8.0,<4.0 | Visualization |
| jinja2 | >=3.0.0,<4.0 | HTML report templates |
