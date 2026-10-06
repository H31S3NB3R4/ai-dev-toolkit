"""Answer relevance metric: does the generated answer directly address the prompt?"""

from typing import Any

from ai_dev_toolkit.core.result import MetricResult
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.parsing import judge_with_retry
from ai_dev_toolkit.metrics.prompts.templates import ANSWER_RELEVANCE_PROMPT
from ai_dev_toolkit.providers.base import LLMProvider


class AnswerRelevanceMetric(Metric):
    """Evaluate how directly and relevantly the response answers the prompt."""

    @property
    def name(self) -> str:
        return "answer_relevance"

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
        **kwargs: Any,
    ) -> MetricResult:
        """Score how relevant the answer is to the user question."""
        judge_prompt = ANSWER_RELEVANCE_PROMPT.format(
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
