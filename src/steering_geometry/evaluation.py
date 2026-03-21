"""Compatibility shim for evaluation module.

This module re-exports evaluation symbols from apply_steering.py and types.py
to maintain backward compatibility with the old import path.

.. deprecated::
    Import directly from steering_geometry.apply_steering instead:

        # Old (deprecated but works):
        from steering_geometry.evaluation import JudgeEvaluator

        # New (preferred):
        from steering_geometry.apply_steering import JudgeEvaluator

The evaluation functionality was merged into apply_steering.py in PR6.
This shim will be maintained for backward compatibility.
"""

from steering_geometry.apply_steering import (
    JudgeEvaluator,
    MMLUEvaluator,
    generate_html_report,
)
from steering_geometry.config import MMLUConfig
from steering_geometry.types import (
    EvaluationMetadata,
    EvaluationResult,
    JudgeScore,
    MMLUPrediction,
    MMLUQuestion,
    MMLUResult,
)

__all__ = [
    "JudgeEvaluator",
    "MMLUEvaluator",
    "generate_html_report",
    "MMLUConfig",
    "EvaluationMetadata",
    "EvaluationResult",
    "JudgeScore",
    "MMLUPrediction",
    "MMLUQuestion",
    "MMLUResult",
]
