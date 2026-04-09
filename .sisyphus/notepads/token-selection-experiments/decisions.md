# Decisions - Token Selection Experiments

## 2026-04-09 Initial Setup
- Wave 1: Tasks 1 and 2 in parallel (independent foundation work)
- Wave 2: Tasks 3-6 in parallel (depend on Wave 1)
- Final Wave: F1-F4 in parallel (depend on all tasks)
- Using existing `stability_comparison.py` patterns for experiment functions
- Hook modification uses mutable counter in closure (step_counter = [0]) pattern
