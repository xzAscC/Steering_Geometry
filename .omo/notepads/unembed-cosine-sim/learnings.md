
## 2026-03-16: Test Fixtures for Unembedding Analysis

### Fixture Patterns
- `mock_unembedding_matrix`: Tensor (100, 64) - simulates vocab embeddings
- `sample_unembed_vector`: Tensor (64,) - simulates steering vector
- `mock_tokenizer`: Reuses existing `FakeTokenizer` class with decode() method
- `mock_special_token_ids`: set[int] with {0, 1, 2}

### Key Decisions
- Reused existing `FakeTokenizer` instead of creating new mock - simpler, consistent
- Followed conftest.py pattern: no docstrings on fixtures (unlike other project files)
- Used `torch.randn()` for deterministic random tensors with float32 dtype

### File Locations
- Fixtures: `tests/conftest.py` lines 183-204
- Test file: `tests/test_unembed_analysis.py` (placeholder for Wave 2)

## 2026-03-16: HookedModel Methods for Unembedding Analysis

### Methods Added
- `get_unembedding_matrix() -> Tensor`: Returns (vocab_size, hidden_dim) tensor from model
- `get_special_token_ids() -> set[int]`: Returns {bos, eos, pad} token IDs

### Architecture Handling
- Primary: `model.lm_head.weight` (Qwen, most HF models)
- Fallback 1: `model.embed_out.weight` (some architectures)
- Fallback 2: `model.model.embed_tokens.weight` (shared embedding/output)

### Type Safety
- Used `cast(Tensor, ...)` for dynamic attribute access to satisfy mypy
- All tensor returns are `.detach().cpu()` to avoid gradient tracking

### Code Location
- File: `src/steering_geometry/models.py` lines 101-138

## 2026-03-16: compute_topk_similar_tokens() Implementation

### Implementation Pattern
- Normalization: `torch.clamp(torch.norm(...), min=1e-8)` for numerical stability
- Cosine similarity: `torch.mv(normalized_matrix, normalized_vector)` - matrix-vector dot product
- Token exclusion: Set similarity to `float("-inf")` before topk
- Strict zip: Use `strict=True` in zip() per ruff B905 rule

### Key Code Patterns
```python
# Unit vector normalization with numerical stability
vector_norm = vector / torch.norm(vector)
unembed_norms = torch.clamp(torch.norm(unembed_matrix, dim=1, keepdim=True), min=1e-8)
unembed_normalized = unembed_matrix / unembed_norms

# Cosine similarity via dot product
similarities = torch.mv(unembed_normalized, vector_norm)

# Token exclusion via -inf masking
similarities[token_id] = float("-inf")
```

### File Location
- File: `src/steering_geometry/unembed_analysis.py`
- Function: `compute_topk_similar_tokens()`

## 2026-03-16: analyze_steering_vector() Implementation

### Function Pattern
- Uses TYPE_CHECKING guard for HookedModel import (avoids circular import)
- Calls existing `compute_topk_similar_tokens()` for the core logic
- Returns `UnembedAnalysisResult` dataclass directly

### Code Structure
```python
def analyze_steering_vector(
    vector: Tensor,
    model: "HookedModel",
    layer_frac: float,
    method: str,
    k: int = 5,
) -> UnembedAnalysisResult:
    unembed_matrix = model.get_unembedding_matrix()
    special_token_ids = model.get_special_token_ids()
    token_results = compute_topk_similar_tokens(...)
    tokens = [text for text, _ in token_results]
    similarities = [sim for _, sim in token_results]
    return UnembedAnalysisResult(layer=layer_frac, method=method, ...)
```

### Key Decisions
- Used forward reference `"HookedModel"` with TYPE_CHECKING guard for clean imports
- Logging at INFO level for analysis start and top tokens result
- No inline comments - code is self-explanatory
- Added to `__all__` for public API export

### File Location
- File: `src/steering_geometry/unembed_analysis.py`
- Function: `analyze_steering_vector()`
- Lines: 86-132

## 2026-03-16: Unit Tests for unembed_analysis Module

### Tests Added
- `test_compute_topk_similar_tokens_basic`: Validates correct result count, tuple types, similarity range [-1,1], descending sort
- `test_compute_topk_similar_tokens_exclude`: Validates excluded tokens don't appear, -inf masking works
- `test_compute_topk_similar_tokens_invalid_vector_dim`: Validates 2D vector raises ValueError
- `test_compute_topk_similar_tokens_mismatched_hidden_dim`: Validates dimension mismatch raises ValueError  
- `test_analyze_steering_vector`: Integration test with mocked HookedModel, validates UnembedAnalysisResult structure

### Mocking Pattern for analyze_steering_vector
- Can't use `mock_hooked_model` from conftest.py (FakeCausalLM lacks lm_head)
- Instead, create `MagicMock(spec=HookedModel)` with mocked methods:
  - `get_unembedding_matrix.return_value = mock_unembedding_matrix`
  - `get_special_token_ids.return_value = mock_special_token_ids`
  - `.tokenizer = mock_tokenizer`

### File Location
- File: `tests/test_unembed_analysis.py`
- 5 tests total (4 for compute_topk_similar_tokens, 1 for analyze_steering_vector)

## 2026-03-16: run_unembed_experiment() Implementation

### Function Pattern
- Loads model ONCE, reuses for all layer analyses (avoids repeated loading overhead)
- Imports extract_vector inside function to avoid circular import at module load time
- Maps method names: "diff_means" → "mean" for extract_vector() API
- Output path: `outputs/unembed_analysis/json/{concept}_{method}.json`

### Key Code Structure
```python
def run_unembed_experiment(
    concept: str,
    model_name: str,
    method: str,  # "diff_means" or "discriminative"
    layers: list[float],
    num_pairs: int = 1000,
    top_k: int = 30,
    output_dir: Path | str = "outputs",
) -> ConceptAnalysisResult:
    # Import here to avoid circular import
    from steering_geometry.extract import extract_vector
    
    model = HookedModel(ModelConfig(model_name=model_name))
    extract_method = "mean" if method == "diff_means" else method
    steering_vector = extract_vector(...)
    
    for layer_frac, abs_idx in zip(layers, ...):
        analysis = analyze_steering_vector(vector, model, layer_frac, method, k)
        results[f"layer_{layer_frac}"] = analysis
    
    save_analysis_results(result, json_output_path)
    return result
```

### Method Validation
- Only accepts "diff_means" or "discriminative" - raises ValueError for other methods
- Maps "diff_means" to "mean" internally for extract_vector() compatibility

### JSON Output Format
```json
{
  "concept": "honesty",
  "model": "Qwen/Qwen3-1.7B",
  "method": "diff_means",
  "results": {
    "layer_0.5": {
      "layer": 0.5,
      "method": "diff_means",
      "tokens": ["token1", "token2", ...],
      "similarities": [0.85, 0.72, ...]
    }
  }
}
```

### File Location
- File: `src/steering_geometry/unembed_analysis.py`
- Functions: `save_analysis_results()`, `run_unembed_experiment()`
- Added to `__all__` for public API export
