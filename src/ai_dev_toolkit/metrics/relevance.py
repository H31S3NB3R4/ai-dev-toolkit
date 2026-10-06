"""Relevance metric: does the response address the prompt?"""

from ai_dev_toolkit.core.result import MetricResult
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.parsing import judge_with_retry
from ai_dev_toolkit.metrics.prompts import RELEVANCE_PROMPT
from ai_dev_toolkit.providers.base import LLMProvider


class RelevanceMetric(Metric):
    """Evaluate how well a response addresses the original prompt."""

    @property
    def name(self) -> str:
        return "relevance"

    @property
    def requires_context(self) -> bool:
        return False

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
        """Score how relevant the response is to the prompt."""
        judge_prompt = RELEVANCE_PROMPT.format(
            prompt=prompt,
            response=response,
        )

        data = judge_with_retry(
            provider=provider,
            prompt=judge_prompt,
            metric_name=self.name,
            required_keys=["score"],
            temperature=temperature,
        )

        return MetricResult(
            score=float(data["score"]),
            reasoning=data.get("reasoning") if include_reasoning else None,
        )
