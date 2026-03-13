"""Experimental utilities for steering vector analysis.

Provides functions for computing cosine similarity matrices, generating heatmaps,
and managing vector persistence for steering vector experiments.
"""

import logging
from pathlib import Path
from typing import cast

import torch
from numpy import ndarray
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]
from torch import Tensor

from steering_geometry.utils import ensure_dir

logger = logging.getLogger(__name__)


def compute_cosine_similarity_matrix(vectors: list[Tensor]) -> ndarray:
    """Compute pairwise cosine similarity matrix for a list of vectors.

    Args:
        vectors: List of 1D tensors to compute similarities between.

    Returns:
        2D numpy array of shape (n_vectors, n_vectors) containing pairwise
        cosine similarities. Values range from -1 to 1.

    Raises:
        ValueError: If vectors list is empty or tensors have incompatible shapes.
    """
    if not vectors:
        msg = "Cannot compute similarity matrix for empty vector list"
        raise ValueError(msg)

    stacked = torch.stack(vectors)
    matrix = stacked.numpy()

    return cast("ndarray", cosine_similarity(matrix))


def plot_heatmap(
    matrix: ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> Path:
    """Generate and save a heatmap visualization of a similarity matrix.

    Args:
        matrix: 2D numpy array to visualize as heatmap.
        labels: List of labels for both axes (must match matrix dimensions).
        title: Title for the plot.
        output_path: Path where the plot will be saved (PDF format).

    Returns:
        Path to the saved plot file.

    Raises:
        ValueError: If labels length doesn't match matrix dimensions.
    """
    import matplotlib.pyplot as plt

    if len(labels) != matrix.shape[0] or len(labels) != matrix.shape[1]:
        msg = (
            f"Labels length ({len(labels)}) must match matrix dimensions "
            f"({matrix.shape[0]}x{matrix.shape[1]})"
        )
        raise ValueError(msg)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", fontsize=10)

    ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()

    logger.info("Saved heatmap to %s", output_path)
    return output_path


def save_vector(vector: Tensor, path: Path) -> None:
    """Save a tensor to disk as a .pt file.

    Args:
        vector: Tensor to save.
        path: Destination path (will create parent directories if needed).
    """
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(vector, path)
    logger.debug("Saved vector to %s", path)


def load_vector(path: Path) -> Tensor:
    """Load a tensor from a .pt file.

    Args:
        path: Path to the .pt file.

    Returns:
        Loaded tensor.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Vector file not found: {path}"
        raise FileNotFoundError(msg)

    vector = cast("Tensor", torch.load(path, weights_only=True))
    logger.debug("Loaded vector from %s", path)
    return vector


def cap_examples(requested: int, max_available: int, concept: str) -> int:
    """Cap the number of requested examples to available maximum.

    Logs a warning if the requested amount exceeds availability.

    Args:
        requested: Number of examples requested.
        max_available: Maximum number of examples available.
        concept: Name of the concept (for logging).

    Returns:
        The capped number of examples (min of requested and max_available).
    """
    if requested > max_available:
        logger.warning(
            "Requested %d examples for concept '%s' but only %d available. Using %d instead.",
            requested,
            concept,
            max_available,
            max_available,
        )
        return max_available
    return requested


__all__ = [
    "compute_cosine_similarity_matrix",
    "plot_heatmap",
    "save_vector",
    "load_vector",
    "cap_examples",
]
