"""Abstract base class for all evaluation metrics."""

from abc import ABC, abstractmethod

from ai_dev_toolkit.core.result import MetricResult
from ai_dev_toolkit.providers.base import LLMProvider


class Metric(ABC):
    """Base class that all evaluation metrics must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this metric (e.g. 'relevance')."""
        ...

    @property
    @abstractmethod
    def requires_context(self) -> bool:
        """Whether this metric needs a context string to operate."""
        ...

    @abstractmethod
    def score(
        self,
        prompt: str,
        response: str,
        context: str | None,
        provider: LLMProvider,
        *,
        temperature: float = 0.0,
        include_reasoning: bool = True,
    ) -> MetricResult:
        """Evaluate response quality and return a MetricResult.

        Args:
            prompt: The original user prompt.
            response: The LLM-generated response to evaluate.
            context: Optional reference context for grounding.
            provider: LLM provider to use as the judge.
            temperature: Sampling temperature for the judge call.
            include_reasoning: Whether to request reasoning.

        Returns:
            A MetricResult with a score in [0.0, 1.0].
        """
        ...
