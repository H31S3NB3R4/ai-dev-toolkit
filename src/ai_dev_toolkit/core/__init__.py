"""Core evaluation models, config, errors, and evaluation orchestrator."""

from ai_dev_toolkit.core.config import EvaluatorConfig
from ai_dev_toolkit.core.errors import (
    AIDevToolkitError,
    MetricError,
    ProviderError,
)
from ai_dev_toolkit.core.evaluator import Evaluator, evaluate
from ai_dev_toolkit.core.result import (
    EvaluationMetadata,
    EvaluationResult,
    MetricResult,
    calculate_overall,
    clamp_score,
)

__all__ = [
    "AIDevToolkitError",
    "ProviderError",
    "MetricError",
    "EvaluatorConfig",
    "EvaluationMetadata",
    "EvaluationResult",
    "MetricResult",
    "Evaluator",
    "evaluate",
    "calculate_overall",
    "clamp_score",
]
