# Decisions - normalize-discriminative-score

## 2026-04-09
- Use `DISCRIMINATIVE_EPS` as shared constant name in `utils.py`
- Replace `tdnv.py`'s `EPS` with import from utils
- Add `.float()` cast in `extract.py` and `tdnv.py` (already in `token_analysis.py`)
- No new parameters to function signatures
- Commit strategy: 2 commits (1 for utils.py, 1 for all formula + test changes)
