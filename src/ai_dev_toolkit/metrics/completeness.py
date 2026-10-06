"""Completeness metric: are all parts of the prompt answered?"""

from ai_dev_toolkit.core.result import MetricResult
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.parsing import judge_with_retry
from ai_dev_toolkit.metrics.prompts import COMPLETENESS_PROMPT
from ai_dev_toolkit.providers.base import LLMProvider


class CompletenessMetric(Metric):
    """Evaluate whether the response covers all aspects of the prompt."""

    @property
    def name(self) -> str:
        return "completeness"

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
        """Score how completely the response covers the prompt."""
        judge_prompt = COMPLETENESS_PROMPT.format(
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

        metadata: dict[str, object] = {}
        if "sub_questions" in data:
            metadata["sub_questions"] = data["sub_questions"]

        return MetricResult(
            score=float(data["score"]),
            reasoning=(data.get("reasoning") if include_reasoning else None),
            metadata=metadata,
        )
