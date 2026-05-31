# Refusal Benchmarks - Learnings

## 2026-04-09: Task 2 - Benchmark Config Dataclasses

- Added `HarmBenchConfig`, `ORBenchConfig`, `MMLUProConfig` to `config.py`
- Pattern: simple `@dataclass` with typed fields and defaults, docstring with `Attributes:` section
- `list[str] | None = None` works fine in dataclasses without `field(default=None)` — dataclass handles `None` defaults natively
- **Pre-existing issue**: `types.py:155` references `HarmBenchResult` which is not defined, causing `NameError` on import. This blocks `pytest`. Likely needs Task 1 to add `HarmBenchResult` to types.py first.
- Config file now has 10 dataclass configs total (7 existing + 3 new)

## Task 1: Benchmark Evaluation Types

- Dataclass fields with forward references (types defined later in file) cause `NameError` at runtime. Must define types BEFORE they're used in dataclass fields. mypy handles forward refs fine, but Python runtime doesn't without `from __future__ import annotations`.
- TypedDict + @dataclass pattern for benchmark types: TypedDict for per-record, @dataclass for aggregate results.
- Placement: new benchmark types go between `MMLUResult` and `EvaluationResult` so `EvaluationResult` can reference them.

## Task 5: MMLUProEvaluator Implementation

- **Parallel task race conditions**: HarmBench and ORBench tasks write to the same test file (`test_evaluation.py`) concurrently. Must re-read file before each edit and handle overwrites gracefully. Include both HarmBench and MMLUPro tests together when appending.
- **TypedDict vs dict for mypy**: `_filter_na` and `_format_cot_example` must use `MMLUProQuestion` as parameter type, not bare `dict`. Use `cast("MMLUProQuestion", dict(q))` when converting from HuggingFace dataset rows.
- **HuggingFace datasets return non-typed rows**: `load_dataset()` returns rows that need `dict(q)` conversion + `cast` to TypedDict for mypy strict mode.
- **MMLUEvaluator pattern**: `__init__(config, model)` → `load_dataset()` → `evaluate(steering_vector, layer_idx, scale)` — exact same signature across all evaluators.
- **Test mock pattern**: `patch("steering_geometry.apply_steering.load_dataset")` with `side_effect=[test_data, val_data]` for two sequential `load_dataset` calls (test split, then validation split).
- **extract_answer 3-layer regex**: Layer 1 catches "answer is (X)", Layer 2 catches "Answer: X", Layer 3 catches last standalone A-J letter. Same as MMLUEvaluator but with optional parens in Layer 1.
- **`Any` for model param**: Accepted as technical debt per project conventions — matches existing `MMLUEvaluator` pattern.
- **Unused imports from parallel tasks**: ORBench imports added by parallel task cause F401 ruff violations in apply_steering.py. Not my concern — parallel tasks need to clean up.

## Task 3: HarmBenchEvaluator

- Followed JudgeEvaluator pattern exactly: `__init__` with config → `_call_classifier` with retry → evaluation methods
- HarmBench classifier prompts are very long (>100 chars per line). Used string concatenation with parenthesized implicit joining to stay within ruff's 100-char line limit
- ruff SIM110: `for x in items: if cond(x): return True; return False` → `return any(cond(x) for x in items)`
- `_is_refusal` uses prefix matching against canonical HarmBench REFUSAL_PREFIXES list — refusals skip classifier API call (always label=0)
- Truncation uses character approximation: `max_completion_tokens * 4` chars (~4 chars/token)
- `load_behaviors` uses csv.DictReader with HarmBench CSV column names (BehaviorID, Behavior, FunctionalCategory, etc.)
- Test pattern: `_make_evaluator()` helper creates evaluator with localhost config, tests use `patch.object(evaluator.client.chat.completions, "create", new_callable=AsyncMock, ...)`
- Concurrent tasks may modify the same files — had to handle race conditions where TestMMLUProEvaluator was added mid-session

## Task 4: ORBenchEvaluator Implementation

- Followed MMLUEvaluator pattern exactly: `__init__(config, model)` → `load_prompts()` → `evaluate(steering_vector, layer_idx, scale)`
- `_is_refused()` checks first 100 chars of response (case-insensitive) for keyword startswith match
- `_compute_orr()` returns `tuple[float, dict[str, float]]` for overall + per-category ORR
- TypedDict construction: `ORBenchPrediction(prompt=..., category=..., response=..., is_refused=..., refusal_type=...)` works with TypedDict since Python 3.12 treats TypedDict constructors as keyword-arg constructors
- Parallel task file contention: other tasks (HarmBenchEvaluator, MMLUProEvaluator) modify test_evaluation.py concurrently. Had to re-apply edits after parallel modifications overwrote them. Always re-read files before editing when parallel tasks are active.
- `random` module already imported in apply_steering.py — no new stdlib imports needed
- Added `ORBenchEvaluator` to `__all__` in apply_steering.py, alphabetically after `MMLUProEvaluator`
- 9 tests: 1 load_dataset, 5 _is_refused variants, 1 _compute_orr, 1 per_category_orr, 1 full evaluate integration
- Mock pattern: `patch("steering_geometry.apply_steering.load_dataset")` with `MagicMock` model

## Task 6: Wiring Evaluators into apply_steering()

- All three evaluators (HarmBench, OR-Bench, MMLU-Pro) already existed in apply_steering.py with their config/result types already imported
- EvaluationResult in types.py already had `harmbench_result`, `orbench_result`, `mmlu_pro_result` optional fields from Task 1
- Pattern: each evaluator gated behind a boolean flag, instantiated with config, called with `evaluate(normalized_v, layer_idx, scale)` (except HarmBench which uses `asyncio.run(hb_evaluator.evaluate(completions))`)
- HarmBench is async (uses AsyncOpenAI classifier), so needs `asyncio.run()` wrapper — different from the sync OR-Bench and MMLU-Pro evaluators
- JSON serialization uses conditional dict updates (`if result is not None:`) to only include benchmarks that were actually run
- Changed json.dump from inline dict literal to building `result_dict` variable first, then serializing — cleaner with conditional fields
- All new CLI flags default to disabled (False/0/"") for backward compatibility
- `_Args` Protocol mirrors argparse defaults exactly
- No new imports needed — configs were already imported from previous tasks
