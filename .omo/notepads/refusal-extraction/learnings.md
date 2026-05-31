# Learnings — refusal-extraction plan

## 2026-04-09 Session Start

### Codebase State
- `select_token_activations()` in utils.py:82-113 — currently accepts only `int` for `read_token_index`
  - Has 2D passthrough (line 95-96), 3D with -1 index (lines 102-107), 3D with explicit index (lines 109-113)
  - Padding detection: `non_zero_mask = activations.abs().sum(dim=-1) > 0` (line 103)
  - Returns `(batch, hidden_dim)` always
- `ExtractionConfig` in config.py:46-64 — fields: layers, method, batch_size, read_token_index, top_k
- `load_refusal_data()` in extract.py:296-331 — single dataset (LLM-LAT/harmful-dataset), sequential (no shuffle)
- `_REFUSAL_PREFIX` (line 64) and `_COMPLIANCE_PREFIX` (line 65) — constants to delete
- `_DATASET_LOADERS` registry (line 334-339) — maps concept name to loader function
- `extract_steering_vector()` (lines 374-440) — calls `select_token_activations` at lines 415-422
- No `tests/unit/test_utils.py` exists yet — must create

### Test Suite
- 150 tests pass, 2 skipped
- Test framework: pytest
- Key fixtures in conftest.py: mock_hooked_model, sample_contrast_pairs, FakeTokenizer, FakeCausalLM
- FakeCausalLM: 4 layers, hidden_dim=8

### Key Patterns
- Sentiment loader (extract.py:192-241) is the canonical pattern: oversample → collect → sample_with_seed → zip into ContrastPairs
- All loaders use `ContrastPairMetadata` TypedDict with concept, dataset, source, pair_index fields
- `sample_with_seed()` from utils.py:28-40 uses `random.Random(seed).sample()`

### Aggregator Interface
- All aggregators: `(pos: Tensor, neg: Tensor) -> Tensor` — MUST NOT change
- Currently receive (N, hidden_dim) 2D tensors from `select_token_activations`
- The "all" and "last_n" modes produce flattened (total_tokens, hidden_dim) — compatible with existing aggregators
