"""AI Dev Toolkit.

Open-source evaluation and testing toolkit for LLM, RAG, and AI agents.
"""

from ai_dev_toolkit._version import __version__
from ai_dev_toolkit.core.config import EvaluatorConfig
from ai_dev_toolkit.core.dataset import (
    DatasetEvaluationResult,
    MetricSummary,
    evaluate_batch,
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
from ai_dev_toolkit.metrics.answer_relevance import AnswerRelevanceMetric
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.citation import CitationCorrectnessMetric
from ai_dev_toolkit.metrics.completeness import CompletenessMetric
from ai_dev_toolkit.metrics.context_recall import ContextRecallMetric
from ai_dev_toolkit.metrics.context_relevance import ContextRelevanceMetric
from ai_dev_toolkit.metrics.faithfulness import FaithfulnessMetric
from ai_dev_toolkit.metrics.registry import (
    get_metric,
    list_metrics,
    register_metric,
    reset_metrics,
)
from ai_dev_toolkit.metrics.relevance import RelevanceMetric
from ai_dev_toolkit.reports.generator import (
    generate_html_report,
    generate_markdown_report,
)

__all__ = [
    "__version__",
    "evaluate",
    "aevaluate",
    "evaluate_dataset",
    "evaluate_batch",
    "generate_markdown_report",
    "generate_html_report",
    "Evaluator",
    "EvaluatorConfig",
    "EvaluationResult",
    "EvaluationMetadata",
    "MetricResult",
    "DatasetEvaluationResult",
    "MetricSummary",
    "Metric",
    "RelevanceMetric",
    "CompletenessMetric",
    "FaithfulnessMetric",
    "ContextRelevanceMetric",
    "ContextRecallMetric",
    "AnswerRelevanceMetric",
    "CitationCorrectnessMetric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "reset_metrics",
    "AIDevToolkitError",
    "MetricError",
    "ProviderError",
]
