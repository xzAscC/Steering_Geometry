# Decisions — refusal-extraction plan

## 2026-04-09

### Architecture Decisions
- `read_token_index` parameter type changes from `int` to `int | str` (backward compat for existing int usage)
- "all" and "last_n" modes return flattened (total_tokens, hidden_dim) — aggregator interface unchanged
- New ExtractionConfig fields added alongside existing ones — `read_token_index` preserved for other concepts
- Dual-dataset loader replaces old single-dataset loader in `_DATASET_LOADERS["refusal"]` slot
- Dead code (`_REFUSAL_PREFIX`, `_COMPLIANCE_PREFIX`) deleted in Task 6, not earlier
