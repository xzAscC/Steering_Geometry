# Learnings

## 2026-04-08 Session Start

### Codebase Patterns
- `load_refusal_data()` (extract.py:288-323) has early-break pattern at line 296-297: `if pair_index >= config.num_pairs: break`
- `load_sentiment_data()` (extract.py:192-237) and `load_polite_data()` (extract.py:240-285) iterate entire dataset then sample
- `sample_with_seed()` uses `random.Random(42).sample()` for deterministic sampling
- `_run_concept()` (tdnv.py:962-989) loads data at line 967 for printing, then `compute_tdnv_for_concept()` loads again at line 426
- `_Args` protocol (tdnv.py:860-872) has: mode, concept, model, num_pairs, num_questions, mmlu_seed, categories, output, plot_dir, dry_run, last_n, top_k

### Test Patterns
- Tests use `unittest.mock.patch` and `unittest.mock.MagicMock`
- Existing test files: `tests/unit/test_extract.py` (175 lines), `tests/unit/test_tdnv.py` (552 lines)
- `test_extract.py` has classes: `TestLoadContrastPairs`, `TestDatasetLoaders`, `TestPoliteLoader`
- `test_tdnv.py` has classes: `TestComputePerTopicStats`, `TestTDNVFormula`, `TestTDNVEnergy`, `TestComputeTDNVMultiConcept`, `TestSelectLastNTokens`, `TestSelectTopKDiscriminative`, `TestComputeTDNVMMLU`
- Some tests use `@pytest.mark.skipif(not HAS_ACCELERATE, reason="requires accelerate package")`

### Architecture
- Project uses `uv` for package management, `ruff` for linting, `mypy --strict` for type checking, `pytest` for testing
- Line length: 100, double quotes

## Test Writing for Bug Fixes (2026-04-07)

- `_run_concept` in tdnv.py uses `_Args` Protocol with fields: mode, concept, model, num_pairs, num_questions, mmlu_seed, categories, output, plot_dir, dry_run, last_n, top_k
- For tdnv tests, mock `_run_concept` dependencies at `steering_geometry.tdnv.<func>` path
- `_run_concept` dry_run branch calls `load_contrast_pairs` directly; non-dry-run delegates to `compute_tdnv_for_concept`
- `load_sentiment_data` uses `load_dataset("glue", "sst2")` returning dict with "train" key; `load_polite_data` uses `load_dataset("Intel/polite-guard", split="train")` returning iterable directly
- For early-stop testing, `_TrackingIterable` pattern tracks consumed items — sentiment consumed ~44 items for num_pairs=10 (oversample=20, balanced so needs ~20 of each), well under 20000 total
- `sample_with_seed` is called twice per loader (once for positives, once for negatives) — assert `call_count >= 2`
- Patch paths: `steering_geometry.extract.load_dataset`, `steering_geometry.extract.sample_with_seed`, `steering_geometry.tdnv.load_contrast_pairs`, etc.
- Full suite: 150 passed, 2 skipped (accelerate-dependent tests)
