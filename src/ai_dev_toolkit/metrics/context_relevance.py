"""Context relevance metric: does retrieved context directly relate to prompt?"""

from typing import Any

from ai_dev_toolkit.core.result import MetricResult
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.parsing import judge_with_retry
from ai_dev_toolkit.metrics.prompts.templates import CONTEXT_RELEVANCE_PROMPT
from ai_dev_toolkit.providers.base import LLMProvider


class ContextRelevanceMetric(Metric):
    """Evaluate how relevant retrieved context chunks are to the user prompt."""

    @property
    def name(self) -> str:
        return "context_relevance"

    @property
    def requires_context(self) -> bool:
        return True

    def score(
        self,
        prompt: str,
        response: str,
        context: str | None,
        provider: LLMProvider,
        *,
        temperature: float = 0.0,
        include_reasoning: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Score relevance of context to the prompt."""
        if not context or not context.strip():
            return MetricResult(
                score=0.0,
                reasoning=(
                    "No context was provided to evaluate context relevance."
                    if include_reasoning
                    else None
                ),
            )

        judge_prompt = CONTEXT_RELEVANCE_PROMPT.format(
            prompt=prompt,
            context=context,
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
