"""AI Dev Toolkit.

Open-source evaluation and testing toolkit for LLM, RAG, and AI agents.
"""

from ai_dev_toolkit._version import __version__
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
)

__all__ = [
    "__version__",
    "evaluate",
    "Evaluator",
    "EvaluatorConfig",
    "EvaluationResult",
    "EvaluationMetadata",
    "MetricResult",
    "AIDevToolkitError",
    "MetricError",
    "ProviderError",
]
