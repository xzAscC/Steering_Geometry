# Learnings - Token Selection Experiments

## Codebase Patterns
- `stability_comparison.py` is the canonical pattern for experiment runners: load data → loop params → extract vectors → save → compute similarity → generate heatmaps
- `models.py:steering_hook` uses closure pattern — mutable state in closure is per-call scoped
- `ExtractionConfig.token_select` defaults to "default" (sentinel) — experiments MUST explicitly set it
- Shell scripts follow `run_stability_comparison.sh` pattern: shebang, set -euo pipefail, defaults, while/case parsing, inline Python
- FakeCausalLM.generate() computes hidden states in one batch, does NOT trigger per-step hooks

## Key API Signatures
- `generate_with_steering(prompt, layer_idx, steering_vector, scale, max_new_tokens=100, temperature=0.0)` → str
- `SteeringConfig(multipliers, num_samples=10, seed=42, max_new_tokens=100, temperature=0.0)` 
- `extract_vector(concept, model_name, num_pairs=500, method="mean", layers=None, batch_size=8)` → SteeringVector
- `extract_steering_vector(model, pairs, config: ExtractionConfig)` → SteeringVector
- `load_contrast_pairs(concept, num_pairs, **kwargs)` → list[ContrastPair]

## Conventions
- Use `logging` module, never `print()` in production code
- No `typing.Any` or `# type: ignore`
- ruff line-length 100, double quotes
- All function params and returns need type hints

## steer_tokens Implementation (Task 5)
- `SteeringConfig` now has `steer_tokens: int | None = None` after `temperature`
- `generate_with_steering()` accepts `steer_tokens` parameter with step-counting hook closure
- Step counter uses mutable list `step_counter = [0]` in closure — standard Python closure mutation pattern
- When `steer_tokens is not None`, `use_cache=True` is set in gen_kwargs for KV-cache support
- `steer_tokens=0` → counter hits 1 on first step, 1 > 0 → no steering. Natural edge case.
- `steer_tokens >= max_new_tokens` → all steps steered, equivalent to None. Natural edge case.
- FakeCausalLM.generate() now calls `self(input_ids)` per step (triggers registered forward hooks)
- FakeCausalLM.generate() accepts `**kwargs` to absorb extra args like `use_cache`
- The only caller of `generate_with_steering` is `apply_steering.py:618` — uses kwargs, safe to add optional param
- Updated API signatures:
  - `generate_with_steering(prompt, layer_idx, steering_vector, scale, max_new_tokens=100, temperature=0.0, steer_tokens=None)` → str
  - `SteeringConfig(multipliers, num_samples=10, seed=42, max_new_tokens=100, temperature=0.0, steer_tokens=None)`
