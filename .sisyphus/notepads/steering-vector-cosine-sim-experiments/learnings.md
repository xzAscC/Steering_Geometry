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
