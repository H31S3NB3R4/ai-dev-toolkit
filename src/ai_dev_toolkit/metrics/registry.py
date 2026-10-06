"""Custom metric registration and lookup registry."""

import inspect

from ai_dev_toolkit.core.errors import MetricError
from ai_dev_toolkit.metrics.answer_relevance import AnswerRelevanceMetric
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.citation import CitationCorrectnessMetric
from ai_dev_toolkit.metrics.completeness import CompletenessMetric
from ai_dev_toolkit.metrics.context_recall import ContextRecallMetric
from ai_dev_toolkit.metrics.context_relevance import ContextRelevanceMetric
from ai_dev_toolkit.metrics.faithfulness import FaithfulnessMetric
from ai_dev_toolkit.metrics.relevance import RelevanceMetric

_DEFAULT_METRICS: dict[str, type[Metric]] = {
    "relevance": RelevanceMetric,
    "completeness": CompletenessMetric,
    "faithfulness": FaithfulnessMetric,
    "context_relevance": ContextRelevanceMetric,
    "context_recall": ContextRecallMetric,
    "answer_relevance": AnswerRelevanceMetric,
    "citation_correctness": CitationCorrectnessMetric,
}

_REGISTRY: dict[str, Metric | type[Metric]] = dict(_DEFAULT_METRICS)


def register_metric(
    metric: Metric | type[Metric],
    name: str | None = None,
) -> None:
    """Register a custom metric in the global registry.

    Args:
        metric: An instance or class inheriting from Metric.
        name: Optional custom identifier. If not provided, uses metric.name.

    Raises:
        ValueError: If metric is not an instance/subclass of Metric
            or has no valid name.
    """
    if inspect.isclass(metric):
        if not issubclass(metric, Metric):
            raise ValueError(
                f"Class '{metric.__name__}' must inherit from Metric base class."
            )
        metric_name: str | None = name
        if not metric_name:
            cls_name_attr = getattr(metric, "name", None)
            if isinstance(cls_name_attr, str):
                metric_name = cls_name_attr
            else:
                try:
                    inst = metric()
                    metric_name = inst.name
                except Exception as e:
                    raise ValueError(
                        f"Could not determine name for metric class "
                        f"'{metric.__name__}'. Please provide explicit 'name' argument."
                    ) from e
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError(
                f"Metric name must be a non-empty string, got {metric_name!r}"
            )
        _REGISTRY[metric_name.lower().strip()] = metric
    elif isinstance(metric, Metric):
        metric_name = name or metric.name
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError(
                f"Metric name must be a non-empty string, got {metric_name!r}"
            )
        _REGISTRY[metric_name.lower().strip()] = metric
    else:
        raise ValueError(
            f"Expected Metric instance or subclass, got {type(metric).__name__}."
        )


def get_metric(name: str) -> Metric:
    """Retrieve and instantiate a metric by name.

    Args:
        name: Unique identifier of the metric (case-insensitive).

    Returns:
        An instantiated Metric object.

    Raises:
        MetricError: If no metric with the given name is registered.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = list_metrics()
        raise MetricError(
            f"Metric '{name}' not found. Available metrics: {available}",
            metric_name=name,
        )

    entry = _REGISTRY[key]
    if inspect.isclass(entry):
        return entry()
    return entry


def list_metrics() -> list[str]:
    """Return a list of all registered metric names."""
    return sorted(_REGISTRY.keys())


def reset_metrics() -> None:
    """Reset the registry to the default built-in metric suite."""
    global _REGISTRY
    _REGISTRY = dict(_DEFAULT_METRICS)
