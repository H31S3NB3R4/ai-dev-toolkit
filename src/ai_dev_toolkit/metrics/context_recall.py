"""Context recall metric: does context contain facts needed to answer prompt?"""

from typing import Any

from ai_dev_toolkit.core.result import MetricResult
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.parsing import judge_with_retry
from ai_dev_toolkit.metrics.prompts.templates import CONTEXT_RECALL_PROMPT
from ai_dev_toolkit.providers.base import LLMProvider


class ContextRecallMetric(Metric):
    """Evaluate whether context contains sufficient facts / ground truth."""

    @property
    def name(self) -> str:
        return "context_recall"

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
        ground_truth: str | None = None,
        reference: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Score how completely the retrieved context covers the necessary facts."""
        if not context or not context.strip():
            return MetricResult(
                score=0.0,
                reasoning=(
                    "No context provided to evaluate context recall."
                    if include_reasoning
                    else None
                ),
            )

        ref = ground_truth or reference or ""
        ref_text = ref if ref else "(No ground truth provided; evaluate against prompt)"
        judge_prompt = CONTEXT_RECALL_PROMPT.format(
            prompt=prompt,
            context=context,
            reference=ref_text,
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
