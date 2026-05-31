# QA Report: switch-polite-dataset

**Date:** 2026-04-05
**Scope:** `load_polite_data()` switched from `Cleanlab/stanford-politeness` → `Intel/polite-guard`

---

## Scenario 1: Verify Intel/polite-guard references

### src/steering_geometry/extract.py — 5 matches
```
7: - polite: Intel/polite-guard
241: """Load politeness contrast pairs from Intel/polite-guard."""
244: dataset = load_dataset("Intel/polite-guard", split="train")
259: msg = "Intel/polite-guard dataset did not provide both polite and impolite texts"
278: source="Intel/polite-guard",
```

### tests/unit/test_extract.py — 2 matches
```
137: """Polite data should load from Intel/polite-guard."""
148: assert pair.metadata["source"] == "Intel/polite-guard"
```

**Result: PASS** — Multiple references in both files.

---

## Scenario 2: Verify no old references

| Pattern | Scope | Result |
|---------|-------|--------|
| `Cleanlab` | src/ + tests/ | No matches |
| `stanford-politeness` | src/ + tests/ | No matches |
| `pandas` | src/ | No matches |
| `hf_hub_download` | src/ | No matches |

**Result: PASS** — All old references removed.

---

## Scenario 3: Verify pyproject.toml clean

```
grep "pandas" pyproject.toml → No matches
```

**Result: PASS** — pandas dependency removed.

---

## Scenario 4: Verify function signature unchanged

```python
def load_polite_data(config: ConceptConfig) -> list[ContrastPair]:
```

- Takes `ConceptConfig` ✅
- Returns `list[ContrastPair]` ✅

**Result: PASS**

---

## Scenario 5: Verify filtering logic

Lines 248-256 of extract.py:
```python
for row in dataset:
    text = row["text"]
    label = row["label"]
    if not text or not text.strip():
        continue
    if label == "polite":
        polite_texts.append(text.strip())
    elif label == "impolite":
        impolite_texts.append(text.strip())
```

Analysis:
- ✅ Only `"polite"` and `"impolite"` labels are collected
- ✅ `"somewhat polite"` and `"neutral"` are implicitly skipped (not matched by if/elif)
- ✅ Empty/whitespace-only text is filtered out
- ✅ Text is stripped before storage

**Result: PASS**

---

## Scenario 6: Verify metadata fields

Lines 275-280 of extract.py:
```python
metadata=ContrastPairMetadata(
    concept=config.concept_name,    ✅ dynamic from config
    dataset=config.dataset_name,    ✅ dynamic from config
    source="Intel/polite-guard",    ✅ hardcoded correctly
    pair_index=pair_index,          ✅ sequential index
)
```

**Result: PASS**

---

## Scenario 7: Error message quality

Two error paths:
1. Line 259: `"Intel/polite-guard dataset did not provide both polite and impolite texts"` — ✅ references new dataset
2. Line 265: `"not enough data to construct politeness contrast pairs"` — ✅ generic, no stale reference

**Result: PASS** — No references to "Stanford Politeness" in error messages.

---

## Summary

| Scenario | Result |
|----------|--------|
| S1: Intel/polite-guard references | PASS |
| S2: No old references | PASS |
| S3: pyproject.toml clean | PASS |
| S4: Function signature unchanged | PASS |
| S5: Filtering logic correct | PASS |
| S6: Metadata fields correct | PASS |
| S7: Error messages updated | PASS |

```
Scenarios [7/7 pass] | Integration [0/0 — unit tests only] | Edge Cases [0 tested — no live data] | VERDICT: APPROVE
```
