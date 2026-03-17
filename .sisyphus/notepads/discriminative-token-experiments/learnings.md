# Learnings - Discriminative Token Experiments

## 2026-03-16 Session Start

### Key Patterns from Existing Code

1. **Dataclass Pattern** (from types.py):
   - All types use `@dataclass` decorator
   - Fields with type hints
   - No `__init__` - auto-generated
   - Example: `ContrastPair` at line 77-91

2. **Config Pattern** (from config.py):
   - Use `@dataclass` with `field(default_factory=...)` for mutable defaults
   - Simple immutable defaults can be inline
   - Example: `TDNVConfig` for reference

3. **CLI Pattern** (from extract.py:571-683):
   ```python
   class _Args(Protocol):
       arg_name: type
   
   def _build_parser() -> argparse.ArgumentParser: ...
   def main() -> None: ...
   
   if __name__ == "__main__":
       main()
   ```

4. **Discriminative Scoring Formula** (extract.py:162-163):
   ```python
   pos_scores = ((pos - neg_center) ** 2).sum(dim=1) - ((pos - pos_center) ** 2).sum(dim=1)
   neg_scores = ((neg - pos_center) ** 2).sum(dim=1) - ((neg - neg_center) ** 2).sum(dim=1)
   ```

5. **Token Decoding**:
   - `tokenizer.convert_ids_to_tokens(token_ids)` → list of subword tokens
   - `tokenizer.convert_tokens_to_string(tokens)` → merged readable text
   - Subwords have prefixes like `Ġ` (space) in BPE tokenizers

### Conventions
- ESM-style imports: `from x import y`
- Type hints on ALL function parameters and returns
- Never use `Any`
- Follow ruff formatting (line-length 100, double quotes)

### 2026-03-16: TokenAnalysisConfig Added

- Added `TokenAnalysisConfig` dataclass to config.py
- Pattern followed: same as other configs with `field(default_factory=...)` for mutable defaults
- Default layers: `[i / 9 for i in range(10)]` produces 10 evenly-spaced values from 0.0 to 1.0
- All configs have docstrings with Attributes section (project convention)

### 2026-03-16: extract_all_token_activations() Implementation

**Key Implementation Details:**
1. **Tokenizer Access**: Use `model.tokenizer` directly (HookedModel exposes it as public attribute)
2. **Attention Mask**: Use `inputs["attention_mask"]` to find actual sequence length and skip padding
3. **Activation Shape**: `activations[layer]` has shape `(batch_size, seq_len, hidden_dim)` - index as `[batch_idx, pos]`
4. **Memory Safety**: Call `.detach().cpu()` immediately to move tensors off GPU
5. **Token Limit**: Track per-layer token count and break early when all layers hit `tokens_per_class`

**Detokenization Pattern:**
```python
raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]  # Get subword token
decoded = tokenizer.convert_tokens_to_string([raw_token])   # Merge subwords
```

**Import Pattern:**
- Use direct imports: `from .config import TokenAnalysisConfig`
- Avoid redundant relative imports that cause F811 redefinition errors

**Unused Parameter Note:**
- `token_ids_context` in `_detokenize_token()` is kept for future use (may need full context for some tokenizers)

### 2026-03-16: run_visualize() Implementation

**Key Implementation Pattern:**
1. Load contrast pairs once using `load_contrast_pairs(concept, num_pairs=500)`
2. Load model once with `ModelConfig` and `HookedModel`
3. Use `model.resolve_layers()` to convert relative (0.0-1.0) to absolute layer indices
4. Extract token activations for positive and negative texts separately
5. Process layers SEQUENTIALLY for memory safety
6. Store results without activation tensors (too large for JSON)

**Memory Safety Pattern:**
```python
for layer in absolute_layers:
    # Process layer...
    torch.cuda.empty_cache()  # Required after each layer
```

**JSON Output Pattern (from tdnv.py):**
```python
output_dir = ensure_dir(Path(args.output))
model_slug = safe_model_name(args.model)
output_file = output_dir / f"{concept}_{model_slug}.json"
```

**Console Output:**
- Print layer header with `=== Layer {layer} ===`
- Show top-k tokens with rank, token_text, and score
- Include direction (toward/away from concept)

**Comment Discipline:**
- Comments required for: memory management (`torch.cuda.empty_cache()`), non-obvious formulas (`i / 9`), data structure differences (excluding tensors)
- Comments NOT required for: function calls with clear names, print statements

### 2026-03-15: Probe Subcommand Implementation

**train_linear_probe() Pattern:**
```python
probe = nn.Linear(hidden_dim, 2)  # Binary classification
optimizer = Adam(probe.parameters(), lr=0.01)
criterion = CrossEntropyLoss()
dataset = TensorDataset(train_activations, train_labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

**evaluate_probe() Pattern:**
```python
probe.eval()
with torch.no_grad():
    logits = probe(test_activations)
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)
    # Use probs[:, 1] for positive class probability in AUC
    auc = float(roc_auc_score(test_labels.numpy(), probs[:, 1].numpy()))
```

**80/20 Stratified Split Pattern:**
```python
train_acts, test_acts, train_labels, test_labels = train_test_split(
    all_activations.numpy(),
    all_labels.numpy(),
    test_size=0.2,
    stratify=all_labels.numpy(),
    random_state=42,
)
train_acts = torch.from_numpy(train_acts)  # Convert back to tensor
```

**Type Safety for sklearn:**
- sklearn modules lack type stubs, use `# type: ignore[import-untyped]`
- sklearn metric functions return `float | ndarray`, wrap with `float()` for strict typing

**Memory Cleanup in Loop:**
```python
del probe, train_acts, test_acts, train_labels, test_labels_tensor
del pos_activations, neg_activations, all_activations, all_labels
torch.cuda.empty_cache()
```

**JSON Output Pattern for Probe:**
- Output file: `{concept}_{model_slug}_probe.json`
- Structure: `concept`, `model_name`, `tokens_per_class`, `layer_results[]`
- Each layer result: `layer_idx`, `train_accuracy`, `test_accuracy`, `auc_score`

## Task 9: Unit Tests for Token Analysis (2026-03-15)

### Files Created
- `tests/test_token_analysis.py` - 20 unit tests for token_analysis module

### Test Coverage
1. **Types instantiation tests** (8 tests):
   - TokenRecord creation with valid data, default score
   - DiscriminativeTokenResult with custom/empty lists
   - ProbeLayerResult with accuracy metrics
   - ProbeExperimentResult with layer results

2. **Config tests** (3 tests):
   - TokenAnalysisConfig default values (top_k=50, tokens_per_class=10000)
   - Layers default is [0.0, 0.111..., 1.0] with 10 values (i/9 pattern)
   - Custom value assignment

3. **Discriminative scoring tests** (4 tests):
   - Scores are finite numbers
   - Scores sorted in descending order
   - ValueError on empty positive/negative lists

4. **Top-k selection tests** (5 tests):
   - Returns exactly k tokens when available
   - Returns all tokens when k > len(list)
   - Returns highest scored tokens
   - Result contains concept and layer metadata
   - Handles empty/mismatched length lists

### Testing Patterns Used
- Helper methods for mock record creation (`_create_mock_records`, `_create_scored_records`)
- `torch.manual_seed()` for deterministic random data
- `pytest.approx()` for floating point comparison
- `pytest.raises()` for exception testing

### Pre-existing Issue
- `tests/test_experiments.py` has import error (module `steering_geometry.experiments` not found)
- Not related to current changes; file should be removed or module created
