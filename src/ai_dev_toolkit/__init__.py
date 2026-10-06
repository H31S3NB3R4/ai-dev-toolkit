"""AI Dev Toolkit.

Open-source evaluation and testing toolkit for LLM, RAG, and AI agents.
"""

from ai_dev_toolkit._version import __version__
from ai_dev_toolkit.core.config import EvaluatorConfig
from ai_dev_toolkit.core.dataset import (
    DatasetEvaluationResult,
    MetricSummary,
    evaluate_dataset,
)
from ai_dev_toolkit.core.errors import (
    AIDevToolkitError,
    MetricError,
    ProviderError,
)
from ai_dev_toolkit.core.evaluator import Evaluator, aevaluate, evaluate
from ai_dev_toolkit.core.result import (
    EvaluationMetadata,
    EvaluationResult,
    MetricResult,
)
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.registry import (
    get_metric,
    list_metrics,
    register_metric,
    reset_metrics,
)

__all__ = [
    "__version__",
    "evaluate",
    "aevaluate",
    "evaluate_dataset",
    "Evaluator",
    "EvaluatorConfig",
    "EvaluationResult",
    "EvaluationMetadata",
    "MetricResult",
    "DatasetEvaluationResult",
    "MetricSummary",
    "Metric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "reset_metrics",
    "AIDevToolkitError",
    "MetricError",
    "ProviderError",
]
