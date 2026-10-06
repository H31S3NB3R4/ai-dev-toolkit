"""Faithfulness metric: is every claim supported by the context?"""

import json

from ai_dev_toolkit.core.result import MetricResult
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.parsing import judge_with_retry
from ai_dev_toolkit.metrics.prompts import (
    FAITHFULNESS_CLAIMS_PROMPT,
    FAITHFULNESS_VERIFY_PROMPT,
)
from ai_dev_toolkit.providers.base import LLMProvider


class FaithfulnessMetric(Metric):
    """Evaluate whether the response is faithful to the provided context.

    Uses a two-step process:
    1. Extract factual claims from the response.
    2. Verify each claim against the provided context.
    """

    @property
    def name(self) -> str:
        return "faithfulness"

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
    ) -> MetricResult:
        """Score faithfulness of response claims against context.

        If context is None, returns score 1.0 (no verification needed).
        """
        if context is None:
            return MetricResult(
                score=1.0,
                reasoning=(
                    "No context provided; faithfulness check skipped."
                    if include_reasoning
                    else None
                ),
            )

        # Step 1: Extract claims
        claims_prompt = FAITHFULNESS_CLAIMS_PROMPT.format(
            response=response,
        )
        claims_data = judge_with_retry(
            provider=provider,
            prompt=claims_prompt,
            metric_name=self.name,
            required_keys=["claims"],
            temperature=temperature,
        )

        claims: list[str] = claims_data.get("claims", [])
        if not claims:
            return MetricResult(
                score=1.0,
                reasoning=(
                    "No factual claims found in the response."
                    if include_reasoning
                    else None
                ),
                metadata={"claims_count": 0},
            )

        # Step 2: Verify claims against context
        verify_prompt = FAITHFULNESS_VERIFY_PROMPT.format(
            context=context,
            claims_json=json.dumps(claims, indent=2),
        )
        verify_data = judge_with_retry(
            provider=provider,
            prompt=verify_prompt,
            metric_name=self.name,
            required_keys=["score"],
            temperature=temperature,
        )

        verdicts = verify_data.get("verdicts", [])
        metadata: dict[str, object] = {
            "claims_count": len(claims),
            "claims": claims,
        }
        if verdicts:
            metadata["verdicts"] = verdicts

        return MetricResult(
            score=float(verify_data["score"]),
            reasoning=(verify_data.get("reasoning") if include_reasoning else None),
            metadata=metadata,
        )
