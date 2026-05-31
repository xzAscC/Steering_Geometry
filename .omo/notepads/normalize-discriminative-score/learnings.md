# Learnings - normalize-discriminative-score

## 2026-04-09 Session Start
- Project uses Python 3.12+, uv, ruff, mypy --strict, pytest
- 3 independent implementations of discriminative scoring exist:
  1. `token_analysis.py:149-197` - `compute_discriminative_scores()` - already has `.float()` cast
  2. `extract.py:126-161` - `discriminative_token_aggregator()` - NO `.float()` cast yet
  3. `tdnv.py:59-112` - `select_top_k_discriminative()` - NO `.float()` cast yet, has own `EPS = 1e-8`
- `tdnv.py:38` has `EPS = 1e-8` module-level constant
- `utils.py` has `__all__` at line 151 - must add new constant there
- Formula change: raw diff → normalized with total-variance denominator
- Scope: 3 source files + 3 test files, NO other files
