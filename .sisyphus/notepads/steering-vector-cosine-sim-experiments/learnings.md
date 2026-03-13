# Learnings

## 2026-03-13: experiments.py Module Creation

### Type Safety with Untyped Libraries
- `sklearn.metrics.pairwise` lacks type stubs → use `# type: ignore[import-untyped]`
- `torch.load()` and `cosine_similarity()` return `Any` → use `cast()` for type safety
- Import `cast` from `typing` module

### Matplotlib Pattern (from tdnv.py)
- Import matplotlib inside function to avoid eager import
- Use `fig, ax = plt.subplots(figsize=(w, h))` pattern
- Always call `fig.tight_layout()` before `plt.savefig()`
- Close with `plt.close()` to prevent memory leaks
- Use `logger.info()` not `print()` for output

### Vector Persistence
- `torch.save(tensor, path)` for saving
- `torch.load(path, weights_only=True)` for safe loading
- Use `ensure_dir(path.parent)` before saving

### Heatmap Color Maps
- "RdYlBu_r" (reversed) works well for similarity (-1 to 1)
- Set `vmin=-1, vmax=1` for cosine similarity

## 2026-03-13: experiments.py Unit Tests

### Pytest Patterns
- Use `tmp_path: Path` fixture for file-based tests (PDFs, tensors)
- Use `caplog: pytest.LogCaptureFixture` to verify logging behavior
- `caplog.set_level("WARNING")` needed to capture warning-level logs
- Mark GPU tests with both `@pytest.mark.gpu` and `@pytest.mark.slow`
- Use `pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")` for conditional GPU skip

### Testing Cosine Similarity
- sklearn's `cosine_similarity` returns symmetric matrix
- Identical vectors produce all 1.0 similarities
- Orthogonal vectors (unit basis) produce identity matrix (1.0 on diagonal, 0.0 off)
- cos(45°) = sqrt(2)/2 ≈ 0.707 - useful known value for tests

### Testing Matplotlib PDF Output
- Check `content.startswith(b"%PDF")` to verify valid PDF header
- Use `tmp_path / "filename.pdf"` for output path

### Tensor Save/Load Testing
- `torch.equal(a, b)` for exact tensor comparison
- Test dtype preservation with `torch.float64`
- Test multi-dimensional tensors (not just 1D)

## 2026-03-13: run_diff_means_experiment Implementation

### Type Annotations for Complex Dict Returns
- Use `dict[str, dict[str, str] | dict[str, dict[str, float]]]` for nested heterogeneous dicts
- Avoid `Any` - use union types for value variation

### zip() with strict=True
- Always use `strict=True` with zip() when iterating parallel sequences
- Catches length mismatches at runtime

### Layer Index Mapping
- `extract_vector()` returns `layer_activations` keyed by absolute indices
- Use `zip(layers, layer_activations.keys(), strict=True)` to map back to fractions

### Edge Case Handling
- Check `torch.isnan(vector).any()` before saving vectors
- Warn on identical vectors (would produce constant similarity matrix)
- Cap n_examples using `cap_examples()` with max from dataset

## 2026-03-13: GPU Integration Test Implementation

### Type Narrowing for Complex Return Types
- When function returns `dict[str, A | B | C]`, mypy can't narrow type after key access
- Use `cast("dict[str, str]", result["key"])` to narrow nested dict types
- Alternative: `typing.TypeGuard` or `@overload` (more complex)
- Pattern: assign casted result to variable, then access nested values

### GPU Test Parameters
- Keep tests minimal: n_examples=10, single layer (0.5)
- Use small model: Qwen/Qwen3-1.7B
- Verify: file creation, vector shape (1D, non-zero), statistics structure
- Use `tmp_path` fixture for output directory

## 2026-03-13: run_discriminative_experiment Implementation

### Using Lower-Level APIs
- `extract_vector()` doesn't support `top_k` parameter for discriminative method
- Use `extract_steering_vector()` directly with custom `ExtractionConfig` when need fine-grained control
- Import `HookedModel`, `ModelConfig`, `ExtractionConfig` from their respective modules

### Model Loading Strategy
- For discriminative experiment, load model once and reuse for all K values
- Avoids repeated model loading overhead compared to calling `extract_vector()` in a loop

### K Value Validation
- Validate `k_values` is not empty
- Validate all K values are positive integers (K > 0)
- Different from `n_examples` validation in diff_means which only checks for empty list

### Output Path Patterns
- Vectors: `{output_dir}/vectors/{concept}/discriminative/k{K}_layer{layer_frac}.pt`
- Heatmaps: `{output_dir}/heatmaps/discriminative/{concept}_layer{layer_frac}.pdf`
- Return dict keys: `k{K}_layer{layer_frac}` (vs `n{n}_layer{layer_frac}` in diff_means)

## 2026-03-13: AGENTS.md Documentation Update

### Documentation Placement
- Added new "Experiments" subsection after "Extraction Scripts" in Section 11 (Pipeline Workflow)
- Updated "Where to Look" table with experiment entries
- Followed existing markdown formatting patterns

### Key Documentation Elements
- Command examples for both experiment scripts
- Parameter lists (concepts, model, layers, n_examples, K values)
- Output directory structure diagram
- Expected output counts (50 PDFs, 325 vectors)
