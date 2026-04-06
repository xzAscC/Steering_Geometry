# Learnings — switch-polite-dataset

## Key Patterns
- `load_sentiment_data()` (extract.py:194-239) is the canonical pattern for dataset loaders using `load_dataset()`
- Tests use integration pattern (real HF API calls), NOT mocks
- `load_contrast_pairs()` sets `dataset_name=concept` (line 360), so `dataset_name="polite"` is correct
- Intel/polite-guard: labels are strings `polite`, `somewhat polite`, `neutral`, `impolite`; text in column `text`; use `split="train"`
- Filter to extreme labels only: `polite` (positive) and `impolite` (negative)

## File Locations
- extract.py:28 — `import pandas as pd` (to remove)
- extract.py:31 — `from huggingface_hub import hf_hub_download` (to remove)
- extract.py:242-292 — `load_polite_data()` (to rewrite)
- extract.py:7 — Module docstring referencing "Cleanlab/stanford-politeness" (to update)
- test_extract.py:133-175 — TestPoliteLoader class (to update)
- test_extract.py:148 — Source assertion `"Cleanlab/stanford-politeness"` (to change)
- test_extract.py:140,154,165 — `dataset_name="politeness"` (to change to `"polite"`)
- pyproject.toml:14 — `"pandas>=2.0,<3.0"` (to remove)
- pyproject.toml:35 — `"pandas-stubs>=2.0,<3.0"` (to remove)
