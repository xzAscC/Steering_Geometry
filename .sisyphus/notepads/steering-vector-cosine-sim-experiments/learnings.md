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
