"""Compatibility shim for vector_analysis imports.

.. deprecated::
    This module was merged into `stability_comparison.py`. Use the new import path:

        from steering_geometry.stability_comparison import (
            run_diff_means_experiment,
            run_discriminative_experiment,
            plot_heatmap,
            load_vector,
            compute_cosine_similarity_matrix,
        )

    This shim is provided for backward compatibility only and will be removed
    in a future version.
"""

from steering_geometry.stability_comparison import (
    compute_cosine_similarity_matrix,
    load_vector,
    plot_heatmap,
    run_diff_means_experiment,
    run_discriminative_experiment,
)

__all__ = [
    "run_diff_means_experiment",
    "run_discriminative_experiment",
    "plot_heatmap",
    "load_vector",
    "compute_cosine_similarity_matrix",
]
