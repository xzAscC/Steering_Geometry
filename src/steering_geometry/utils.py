"""Shared utility functions for steering geometry package."""

import random
from pathlib import Path


def validate_positive_int(value: int, name: str = "value") -> None:
    """Validate that an integer is positive.

    Args:
        value: The integer to validate.
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def sample_with_seed[T](items: list[T], k: int, seed: int = 42) -> list[T]:
    """Sample items randomly with a fixed seed.

    Args:
        items: List of items to sample from.
        k: Number of items to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled items.
    """
    rng = random.Random(seed)
    return rng.sample(items, k=min(k, len(items)))


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        The path that was created.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_model_name(model_name: str) -> str:
    """Convert model name to safe filesystem name.

    Args:
        model_name: Original model name (e.g., "Qwen/Qwen3.5-2B").

    Returns:
        Safe name for filesystem use (e.g., "Qwen_Qwen3.5-2B").
    """
    return model_name.replace("/", "_")


def clamp_score(score: int, min_val: int = 0, max_val: int = 10) -> int:
    """Clamp a score to a valid range.

    Args:
        score: The score to clamp.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Score clamped to [min_val, max_val].
    """
    return max(min_val, min(max_val, score))


__all__ = [
    "validate_positive_int",
    "sample_with_seed",
    "ensure_dir",
    "safe_model_name",
    "clamp_score",
]
