# Technical Specification

**Generated:** 2026-06-15
**Version:** 0.1.0
**Branch:** experiment/pipeline

This document specifies the paper experiment scope for the Steering Geometry framework.
The repository supports the NeurIPS 2026 paper, "Not All Tokens Are Equally Useful for
Steering: Robust Directions and Prefix Steering for Activation Steering".

The implementation is limited to the paper's concepts, models, extraction methods,
steering intervention, and evaluation protocols.

---

## 1. Supported Concepts

The paper studies three behavioral concepts. Code uses canonical concept names where
needed, while paper text may use display names.

| Paper concept | Canonical code name | Positive construction data | Negative construction data | Evaluation data |
|---|---|---|---|---|
| Safety/refusal | `refusal` | `LLM-LAT/benign-dataset` | `LLM-LAT/harmful-dataset` | HarmBench |
| Sentiment | `sentiment` | SST-2 train, positive labels | SST-2 train, negative labels | SST-2 validation |
| Politeness | `polite` | PoliteGuard train, polite labels | PoliteGuard train, impolite labels | PoliteGuard test, polite and impolite labels only |

**Constant:** `SUPPORTED_CONCEPTS = ("refusal", "polite", "sentiment")`

Paper aliases accepted by stability sweep configuration:

| Paper name | Canonical name |
|---|---|
| `safety` | `refusal` |
| `refusal` | `refusal` |
| `politeness` | `polite` |
| `polite` | `polite` |
| `sentiment` | `sentiment` |

---

## 2. Supported Models

The paper experiments use four Hugging Face causal language models.

| Display name | Hugging Face identifier | Notes |
|---|---|---|
| OLMo3-7B | `allenai/Olmo-3-1025-7B` | Uses `trust_remote_code=True` through `ModelConfig` |
| OLMo3-32B | `allenai/Olmo-3-1125-32B` | Uses `trust_remote_code=True` through `ModelConfig` |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | Default model |
| Qwen3-14B | `Qwen/Qwen3-14B` | Paper model |

**Constant:**

```python
SUPPORTED_MODELS = (
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-14B",
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-1125-32B",
)
```

**Default:** `DEFAULT_MODEL = "Qwen/Qwen3-1.7B"`

---

## 3. Aggregation Methods

The extraction pipeline supports four aggregation methods for converting contrastive
activation data into a layer-wise steering direction.

| Method | Meaning | Use in paper scope |
|---|---|---|
| `mean` | Difference in means between positive and negative activations | Baseline DiM direction |
| `pca` | First principal component of contrastive activation differences | Directional baseline |
| `weighted_mean` | Mean direction with token or activation weighting | Weighted baseline |
| `discriminative` | Top-K high-margin activations selected by Robust DiM scoring | Main Robust DiM method |

**Default extraction method:** `mean`

**Configured methods:** `"mean"`, `"pca"`, `"weighted_mean"`, `"discriminative"`

---

## 4. Robust DiM Specification

Robust DiM selects the most useful activations before constructing a steering direction.
The goal is to avoid treating every token as equally informative.

For an activation vector `h`, let `μ+` be the positive class centroid and `μ-` be the
negative class centroid. The relative-margin score is:

```text
s+(h) = (||h - μ-||² - ||h - μ+||²) / (||h - μ-||² + ||h - μ+||²)
```

Interpretation:

| Score behavior | Meaning |
|---|---|
| High positive `s+(h)` | `h` is closer to `μ+` than `μ-` with a large relative margin |
| Near zero | `h` is weakly separated or ambiguous |
| Negative | `h` is closer to `μ-` than `μ+` |

Selection rule:

1. Collect activations from contrast pairs at the configured relative layers.
2. Compute the relative-margin score for candidate activations.
3. Select the top-K high-margin activations.
4. Aggregate selected activations into a steering direction for each layer.

`ExtractionConfig.top_k` controls K. When `top_k` is `None`, the discriminative path uses
its internal default.

---

## 5. Prefix Steering Specification

Prefix Steering applies a steering direction only during the first generated tokens rather
than during the full generation.

| Parameter | Meaning | Paper default |
|---|---|---|
| `steer_tokens` | Number of generation steps that receive the steering vector | `5` |
| `multipliers` | Scale factors applied to the steering vector | Configured by `SteeringConfig` |
| `max_new_tokens` | Maximum generated tokens | Configured by `SteeringConfig` |
| `temperature` | Decoding temperature | Configured by `SteeringConfig` |

The intervention is prefix-local:

```text
For generated token positions t = 1, ..., m, add the steering vector at the target layer.
For generated token positions t > m, run the model without the steering addition.
```

The paper default is `m = 5`. In code, this is represented by `SteeringConfig.steer_tokens`
when running Prefix Steering experiments.

---

## 6. Layer Specification

Layers are specified as relative positions from `0.0` to `1.0`, then resolved to absolute
model layer indices for the selected model.

| Relative layer | Meaning |
|---|---|
| `0.0` | Earliest available layer position |
| `0.4` | 40 percent depth |
| `0.5` | Middle depth |
| `0.6` | 60 percent depth |
| `0.7` | 70 percent depth |
| `0.8` | 80 percent depth |
| `1.0` | Final available layer position |

Default extraction layers:

```python
[0.4, 0.5, 0.6, 0.7, 0.8]
```

Stability sweeps may scan a denser layer grid across `0.0` to `1.0` using
`StabilitySweepConfig.layers` or `StabilityComparisonConfig.layers`.

---

## 7. Evaluation Protocols

The repository implements the paper's evaluation paths only.

### 7.1 HarmBench

Safety/refusal steering is evaluated with the standard HarmBench protocol.

Tracked result type: `HarmBenchResult`

Key fields:

| Field | Meaning |
|---|---|
| `asr` | Attack Success Rate, from 0 to 100 |
| `total` | Number of evaluated behaviors |
| `harmful` | Count classified as harmful |
| `safe` | Count classified as safe |
| `unknown` | Count with unknown classification |
| `predictions` | Per-behavior `HarmBenchPrediction` records |

### 7.2 LLM-as-judge

Sentiment and politeness are evaluated with deterministic LLM-as-judge classification.

Sentiment labels:

```text
POSITIVE
NEGATIVE
NEUTRAL_OR_MIXED
```

Politeness labels:

```text
POLITE
IMPOLITE
NEUTRAL_OR_MIXED
```

Tracked result type: `JudgeScore`

Key fields:

| Field | Meaning |
|---|---|
| `concept_score` | Concept match score |
| `fluency_score` | Naturalness score |
| `final_score` | Combined score |
| `reasoning` | Judge rationale |

### 7.3 MMLU-Pro

General capability preservation is evaluated with MMLU-Pro.

Tracked result type: `MMLUProResult`

Key fields:

| Field | Meaning |
|---|---|
| `accuracy` | Overall accuracy percentage |
| `total` | Number of evaluated questions |
| `correct` | Number answered correctly |
| `refused` | Number refused |
| `extract_failed` | Number with failed answer extraction |
| `per_category` | Accuracy by category |
| `per_category_counts` | Question counts by category |
| `predictions` | Per-question `MMLUProPrediction` records |

### 7.4 Cosine Similarity Stability

Directional stability is measured with cosine similarity between steering directions from
independent trials.

Tracked result type: `StabilitySweepResult`

Key fields:

| Field | Meaning |
|---|---|
| `model_name` | Hugging Face model identifier |
| `concept` | Canonical concept name |
| `display_concept` | Paper display name |
| `selected_layer` | Layer fraction with best average stability |
| `per_n_data` | Mean and standard deviation by sample size |
| `all_layers_data` | Full layer by sample-size stability matrix |

---

## 8. Type Definitions

Core public types are defined in `src/steering_geometry/types.py`.

### 8.1 Extraction Types

| Type | Kind | Purpose |
|---|---|---|
| `ContrastPair` | dataclass | Positive and negative text pair with metadata |
| `ContrastPairMetadata` | `TypedDict` | Concept, dataset, source, pair index, and source fields |
| `SteeringVector` | dataclass | Layer-indexed activation tensors plus model, concept, and method |

### 8.2 Judge and Evaluation Types

| Type | Kind | Purpose |
|---|---|---|
| `JudgeScore` | dataclass | Per-response judge scores and reasoning |
| `EvaluationResult` | dataclass | Combined judge, benchmark, and metadata result |
| `EvaluationMetadata` | `TypedDict` | Concept, model, layer, and multiplier metadata |

### 8.3 HarmBench Types

| Type | Kind | Purpose |
|---|---|---|
| `HarmBenchBehavior` | `TypedDict` | HarmBench behavior record |
| `HarmBenchPrediction` | `TypedDict` | Per-behavior classifier output |
| `HarmBenchResult` | dataclass | Aggregate HarmBench metrics |

### 8.4 Capability Evaluation Types

| Type | Kind | Purpose |
|---|---|---|
| `MMLUProQuestion` | `TypedDict` | MMLU-Pro question record |
| `MMLUProPrediction` | `TypedDict` | Per-question MMLU-Pro prediction |
| `MMLUProResult` | dataclass | Aggregate MMLU-Pro metrics |
| `MMLUQuestion` | `TypedDict` | MMLU question record retained for compatibility |
| `MMLUPrediction` | `TypedDict` | Per-question MMLU prediction retained for compatibility |
| `MMLUResult` | dataclass | Aggregate MMLU metrics retained for compatibility |

### 8.5 Stability Types

| Type | Kind | Purpose |
|---|---|---|
| `StabilitySweepResult` | dataclass | Cosine-similarity stability sweep result |

---

## 9. Configuration

Configuration dataclasses and constants are defined in `src/steering_geometry/config.py`.

### 9.1 Constants

| Name | Value |
|---|---|
| `SUPPORTED_MODELS` | Four paper model identifiers |
| `SUPPORTED_CONCEPTS` | `("refusal", "polite", "sentiment")` |
| `DEFAULT_MODEL` | `"Qwen/Qwen3-1.7B"` |

### 9.2 Model and Extraction Configs

| Config | Key fields | Purpose |
|---|---|---|
| `ModelConfig` | `model_name`, `device`, `dtype`, `trust_remote_code` | Model loading and inference |
| `ExtractionConfig` | `layers`, `method`, `batch_size`, `read_token_index`, `top_k`, `data_mode`, `token_select`, `last_n`, `seed` | Steering vector extraction |
| `ConceptConfig` | `concept_name`, `dataset_name`, `num_pairs` | Concept dataset and sample count |

### 9.3 Steering and Evaluation Configs

| Config | Key fields | Purpose |
|---|---|---|
| `SteeringConfig` | `multipliers`, `num_samples`, `seed`, `max_new_tokens`, `temperature`, `steer_tokens` | Prefix Steering generation settings |
| `JudgeConfig` | `model`, `api_base`, `temperature`, `max_retries` | LLM-as-judge settings |
| `EvaluationConfig` | `judge`, `mmlu`, `output_dir` | Shared evaluation settings |
| `HarmBenchConfig` | `classifier_model`, `classifier_api_base`, `classifier_api_key`, `behaviors_file`, `max_completion_tokens`, `max_retries` | HarmBench classifier settings |
| `MMLUProConfig` | `num_questions`, `n_shot`, `use_cot`, `seed`, `categories`, `max_new_tokens` | MMLU-Pro settings |

### 9.4 Stability Configs

| Config | Key fields | Purpose |
|---|---|---|
| `StabilityComparisonConfig` | `concept`, `num_tokens`, `num_runs`, `layers`, `top_k`, `model_name`, `output_dir` | Independent vector stability comparison |
| `StabilitySweepConfig` | `model_name`, `concept`, `n_values`, `layers`, `num_runs`, `seed`, `output_dir`, `device`, `dtype`, `reference_n` | Sample-size and layer stability sweep |
| `StabilitySweepBatchConfig` | `model_name`, `concepts`, `n_values`, `layers`, `num_runs`, `seed`, `output_dir`, `device`, `dtype`, `reference_n` | Batched stability sweeps for multiple concepts |

---

## 10. Output Structure

Paper experiment outputs are written under `outputs/`.

```text
outputs/
├── vectors/
│   └── {concept}/
│       └── {model}/
│           └── robust_dim_layer{frac}.pt
├── steering/
│   └── {concept}/
│       └── {model}/
│           └── prefix_steering_*.json
├── token_experiments/
│   ├── token_count/
│   ├── token_position/
│   ├── prompt_vs_response/
│   └── steering_scope/
└── vector_analysis/
    ├── stability_sweep/
    └── heatmaps/
```

---

## 11. Definition of Done for Paper Experiments

An experiment run is complete when it records:

1. Model identifier from `SUPPORTED_MODELS`.
2. Canonical concept from `SUPPORTED_CONCEPTS`.
3. Extraction method and layer fraction.
4. Prefix Steering settings, including `steer_tokens` when steering is applied.
5. Concept-specific evaluation result.
6. MMLU-Pro capability result when capability preservation is part of the run.
7. Cosine-similarity stability result when comparing independent direction trials.
