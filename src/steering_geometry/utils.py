"""Shared utility functions for steering geometry package."""

from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

DISCRIMINATIVE_EPS: float = 1e-8


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


def _non_padding_mask(activations: Tensor) -> Tensor:
    """Return 2D boolean mask of non-padding positions from a 3D activation tensor."""
    return activations.abs().sum(dim=-1) > 0


def select_token_activations(
    activations: Tensor,
    read_token_index: int | str,
    last_n: int | None = None,
) -> Tensor:
    """Select activations from token positions.

    Args:
        activations: Tensor of shape (batch, seq_len, hidden_dim) or (batch, hidden_dim).
        read_token_index: Token selection mode:
            - int: specific position index (-1 = last non-padding token).
            - "all": all non-padding tokens from every sample.
            - "last_n": last N non-padding tokens per sample (requires ``last_n``).
        last_n: Number of trailing tokens per sample when ``read_token_index="last_n"``.

    Returns:
        Selected activations. Int index → (batch, hidden_dim).
        "all" / "last_n" → (N, hidden_dim) where N depends on non-padding token count.

    Raises:
        ValueError: If activations tensor has unexpected dimensions or last_n misused.
    """
    if activations.ndim == 2:
        return activations
    if activations.ndim != 3:
        msg = f"Expected 2D or 3D activation tensor, got shape {tuple(activations.shape)}"
        raise ValueError(msg)

    # String-based modes: "all" or "last_n"
    if isinstance(read_token_index, str):
        if read_token_index == "all":
            mask = _non_padding_mask(activations)
            return activations[mask]
        if read_token_index == "last_n":
            if last_n is None:
                msg = "last_n must be specified when read_token_index='last_n'"
                raise ValueError(msg)
            mask = _non_padding_mask(activations)
            batch = activations.shape[0]
            collected: list[Tensor] = []
            for i in range(batch):
                real_tokens = activations[i][mask[i]]
                n_take = min(last_n, real_tokens.shape[0])
                collected.append(real_tokens[-n_take:])
            return torch.cat(collected, dim=0)
        msg = f"Unknown token selection mode: {read_token_index!r}"
        raise ValueError(msg)

    # Legacy int-index path (unchanged)
    sequence_length = activations.shape[1]
    if read_token_index == -1:
        non_zero_mask = _non_padding_mask(activations)
        token_indices = non_zero_mask.long().sum(dim=1) - 1
        token_indices = torch.clamp(token_indices, min=0, max=sequence_length - 1)
        batch_indices = torch.arange(activations.shape[0], device=activations.device)
        return activations[batch_indices, token_indices, :]

    index = read_token_index
    if index < 0:
        index += sequence_length
    index = max(0, min(sequence_length - 1, index))
    return activations[:, index, :]


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    log_name: str | None = None,
) -> None:
    """Set up dual-output logging (console + file) for the package.

    Creates a named logger ``steering_geometry`` with a StreamHandler
    (console) and a FileHandler (file).  Idempotent — subsequent calls
    are no-ops when handlers are already attached.

    Args:
        level: Logging level string (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
        log_dir: Directory for the log file.  Created automatically if missing.
                 Defaults to ``logs/`` relative to cwd.
        log_name: Log file name.  Defaults to
                  ``steering_YYYYMMDD_HHMMSS.log``.
    """
    for _name in ("httpx", "datasets", "filelock", "transformers", "PIL"):
        logging.getLogger(_name).setLevel(logging.WARNING)

    logger = logging.getLogger("steering_geometry")

    if logger.handlers:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    console_fmt = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(console_fmt)
    logger.addHandler(stream_handler)

    if log_dir is None:
        log_dir = Path("logs")
    ensure_dir(log_dir)

    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"steering_{timestamp}.log"

    file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_dir / log_name)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)


__all__ = [
    "DISCRIMINATIVE_EPS",
    "configure_logging",
    "validate_positive_int",
    "sample_with_seed",
    "ensure_dir",
    "safe_model_name",
    "clamp_score",
    "select_token_activations",
]
