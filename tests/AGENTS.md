# tests/

**Framework:** pytest — unit and integration tests for steering_geometry

## OVERVIEW

Test suite with mock models, fixtures for contrast pairs, and evaluation mocks.

## STRUCTURE

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures + custom markers
├── test_apply_steering.py   # Integration-style tests for steering
├── integration/             # Placeholder (empty)
│   └── __init__.py
└── unit/                    # Unit tests by module
    ├── __init__.py
    ├── test_aggregators.py  # Extraction aggregator tests
    ├── test_evaluation.py   # Judge/MMLU evaluator tests
    ├── test_extract.py      # Extraction logic tests
    └── test_tdnv.py         # TDNV metrics tests
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add test fixture | `conftest.py` |
| Mock model setup | `conftest.py:FakeTokenizer`, `FakeCausalLM` |
| Test extraction | `unit/test_extract.py` |
| Test evaluation | `unit/test_evaluation.py` |
| Test aggregators | `unit/test_aggregators.py` |

## FIXTURES (conftest.py)

| Fixture | Type | Description |
|---------|------|-------------|
| `sample_fixture` | `str` | Basic string ("value") |
| `sample_contrast_pairs` | `list[ContrastPair]` | 5 test contrast pairs |
| `mock_hooked_model` | `HookedModel` | Monkeypatched mock model |
| `FakeTokenizer` | class | Mock tokenizer (ord-based encoding) |
| `FakeCausalLM` | class | Minimal model for hook testing |
| `tmp_path` | `Path` | Pytest built-in temp directory |
| `monkeypatch` | `MonkeyPatch` | Pytest built-in env mocking |

## CUSTOM MARKERS

```python
@pytest.mark.slow   # Long-running tests
@pytest.mark.gpu    # Tests requiring GPU
```

## CONVENTIONS (This Directory)

### Test Structure
```python
"""Tests for module_name."""

import pytest
from steering_geometry.module import ClassUnderTest


class TestClassName:
    """Tests for ClassName."""

    def test_behavior_description(self) -> None:
        """Short description of expected behavior."""
        # arrange
        obj = ClassUnderTest()
        # act
        result = obj.method()
        # assert
        assert result == expected
```

### Required Elements
- Type hints: `def test_xxx(self) -> None:`
- Docstrings: Describe expected behavior
- Class grouping: `class Test<Feature>:`

### Patterns
```python
# Exception testing
with pytest.raises(ValueError, match="expected message"):
    func()

# Conditional skip
@pytest.mark.skipif(not HAS_ACCELERATE, reason="accelerate not installed")
def test_xxx() -> None: ...

# Mocking
from unittest.mock import MagicMock, patch, AsyncMock

with patch("module.Class") as mock_class:
    mock_class.return_value.method.return_value = "result"
```

## COMMANDS

```bash
uv run pytest                          # All tests
uv run pytest tests/unit/test_extract.py  # Single file
uv run pytest -k "test_aggregator"     # By name pattern
uv run pytest -m "not slow"            # Exclude slow tests
uv run pytest -m gpu                   # Only GPU tests
```

## NOTES

- `tests/integration/` is empty placeholder
- `test_apply_steering.py` at root (inconsistent with `unit/` structure)
- No coverage configuration in pyproject.toml
