# Decisions

## 2026-04-08 Plan: fix-early-data-loading

### Oversample Buffer Strategy
- Collect `num_pairs * 2` items per class, then `sample_with_seed` down to `num_pairs`
- This preserves random sampling guarantee while dramatically reducing loaded data
- For `num_pairs=500`, currently iterates ~67k rows; with fix, iterates ~2000 rows

### Double-Load Fix Strategy
- Move `load_contrast_pairs` call into `if args.dry_run:` branch only
- For non-dry-run, `compute_tdnv_for_concept()` already handles loading + printing
- No API changes to `compute_tdnv_for_concept()`

### Parallelization
- Wave 1: Tasks 1 & 2 in parallel (different files: tdnv.py vs extract.py)
- Wave 2: Task 3 (tests) after both fixes
- Final: F1 quality gate
