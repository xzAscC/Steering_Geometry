# Logging System - Learnings

## 2026-04-10 Session Start
- Plan has 9 implementation tasks + 4 final wave tasks
- Wave 1: T1 (infra) + T2 (.gitignore) — parallel
- Wave 2: T3-T8 (migrations) — parallel, depends on T1
- Wave 3: T9 (docs) — depends on all migrations
- Final: F1-F4 — parallel review
- Key constraints: No basicConfig, no f-string logging, lazy %s format only
- __main__.py shell eval prints MUST be preserved
- token_analysis.py CLI table prints MUST be preserved

## 2026-04-10 TDD: configure_logging()
- `from __future__ import annotations` was needed in utils.py for `Path | None` syntax without 3.10+ runtime cost; also enabled removing quoted type annotations (`"Tensor"` → `Tensor`) for pre-existing code (UP037 fix)
- `pytest.Generator` doesn't exist — use `collections.abc.Generator` for fixture return type hints
- Logging cleanup fixture: must reset logger level AND remove handlers between tests; `autouse=True` fixture with yield pattern works well
- Idempotency via checking `logger.handlers` list is simple and effective
- `ensure_dir()` from utils.py is reusable for log_dir creation
- All 7 tests pass, mypy strict clean, ruff clean, full suite 231 passed

## 2026-04-10 Migration: extract.py
- 10 print() calls replaced: 1 in extract_vector() (public API), 9 in main() (CLI)
- Pattern: `print(f"... {var}")` → `logger.info("... %s", var)` using lazy %s
- For float formatting: `logger.info("Avg %.2f words", val)` — % format specifiers work in lazy logging
- `configure_logging()` imported from `.utils` alongside existing imports
- `--log-level` arg added after `--seed`, before `return parser`
- `_Args` Protocol must be updated to include `log_level: str` for mypy strict
- `configure_logging(level=args.log_level)` placed right after `parse_args()`, before any work
- extract_vector() is both public API and called by CLI — logging works for both paths
- All verifications: 0 print(), ruff clean, mypy clean, 231 tests pass

## 2026-04-10 tdnv.py Migration (21 print() calls)
- 21 print() calls found (not 25 as estimated — grep was the source of truth)
- Distribution: compute_tdnv_for_concept (5), compute_tdnv_for_mmlu (6), save_tdnv_result (1), plot_tdnv_trends (1), plot_stability_trend (1), _run_concept (3), _run_mmlu (3), _run_mmlu saved msg (1)
- Pattern for f-string with format specs: `f"TDNV={metrics.tdnv:.4f}"` → `logger.info("TDNV=%.4f", metrics.tdnv)` — Python logging % format supports `%.4f`
- Error messages use `logger.error()` (e.g., missing --concept arg)
- Library functions (compute_tdnv_for_concept, compute_tdnv_for_mmlu) use `logger.info()` for status messages
- Must add `log_level: str` to _Args Protocol when adding --log-level CLI arg
- `configure_logging()` added as early call in `main()` before branching to mode handlers
- All verifications clean: ruff 0 violations, mypy 0 errors, 231 tests pass, grep exit code 1 (no prints)
