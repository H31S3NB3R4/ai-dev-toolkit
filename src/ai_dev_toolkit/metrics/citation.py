"""Citation correctness metric: are inline citations accurately grounded in context?"""

from typing import Any

from ai_dev_toolkit.core.result import MetricResult
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.parsing import judge_with_retry
from ai_dev_toolkit.metrics.prompts.templates import CITATION_CORRECTNESS_PROMPT
from ai_dev_toolkit.providers.base import LLMProvider


class CitationCorrectnessMetric(Metric):
    """Evaluate whether source citations in the response accurately reflect context."""

    @property
    def name(self) -> str:
        return "citation_correctness"

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
        """Score accuracy of citations in response against context."""
        if not context or not context.strip():
            return MetricResult(
                score=0.0,
                reasoning="No context provided to verify citations."
                if include_reasoning
                else None,
            )

        judge_prompt = CITATION_CORRECTNESS_PROMPT.format(
            context=context,
            response=response,
        )

        data = judge_with_retry(
            provider=provider,
            prompt=judge_prompt,
            metric_name=self.name,
            required_keys=["score"],
            temperature=temperature,
        )

        metadata: dict[str, Any] = {}
        if "citations_checked" in data:
            metadata["citations_checked"] = data["citations_checked"]
        if "citations_correct" in data:
            metadata["citations_correct"] = data["citations_correct"]

        return MetricResult(
            score=float(data["score"]),
            reasoning=data.get("reasoning") if include_reasoning else None,
            metadata=metadata if metadata else None,
        )
