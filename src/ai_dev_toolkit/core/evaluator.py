"""Evaluation engine and top-level execution coordination."""

import asyncio
import concurrent.futures
import logging
import time
from typing import Any

from ai_dev_toolkit.core.config import EvaluatorConfig
from ai_dev_toolkit.core.result import (
    EvaluationMetadata,
    EvaluationResult,
    MetricResult,
    calculate_overall,
)
from ai_dev_toolkit.metrics.base import Metric
from ai_dev_toolkit.metrics.completeness import CompletenessMetric
from ai_dev_toolkit.metrics.faithfulness import FaithfulnessMetric
from ai_dev_toolkit.metrics.relevance import RelevanceMetric
from ai_dev_toolkit.providers.base import LLMProvider

logger = logging.getLogger(__name__)


def _create_default_provider(config: EvaluatorConfig) -> LLMProvider:
    """Create a default provider based on configuration."""
    provider_name = config.provider.lower().strip()
    if provider_name == "gemini":
        from ai_dev_toolkit.providers.gemini import GeminiProvider

        return GeminiProvider(
            model=config.model or "gemini-2.5-flash",
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
    elif provider_name == "openai":
        from ai_dev_toolkit.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=config.model or "gpt-4o-mini",
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
    elif provider_name == "ollama":
        from ai_dev_toolkit.providers.ollama import OllamaProvider

        return OllamaProvider(
            model=config.model or "llama3",
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
    elif provider_name == "fake":
        from ai_dev_toolkit.providers.fake import FakeProvider

        return FakeProvider()
    else:
        raise ValueError(
            f"Unsupported provider '{config.provider}'. "
            f"Supported providers: 'gemini', 'openai', 'ollama', 'fake'."
        )


def _get_default_metrics() -> list[Metric]:
    """Return standard default metric suite."""
    return [
        RelevanceMetric(),
        CompletenessMetric(),
        FaithfulnessMetric(),
    ]


class Evaluator:
    """Core evaluation orchestrator for scoring LLM outputs across multiple metrics."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        config: EvaluatorConfig | None = None,
        metrics: list[Metric] | None = None,
    ) -> None:
        self.config = config or EvaluatorConfig.find_and_load()
        self.provider = provider or _create_default_provider(self.config)
        self.metrics = metrics if metrics is not None else _get_default_metrics()

    def evaluate(
        self,
        prompt: str,
        response: str,
        context: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a prompt/response pair with optional reference context.

        Args:
            prompt: The user prompt or instruction (non-empty string).
            response: The generated response to evaluate (non-empty string).
            context: Optional grounding context for RAG evaluation.

        Returns:
            An EvaluationResult containing metric scores, reasoning, and metadata.

        Raises:
            ValueError: If prompt or response is empty or whitespace-only.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Response must be a non-empty string.")

        clean_context: str | None = (
            context.strip() if context and context.strip() else None
        )

        # Determine active metrics based on context availability
        active_metrics = [
            m
            for m in self.metrics
            if not m.requires_context or clean_context is not None
        ]

        start_time = time.perf_counter()
        results: dict[str, MetricResult] = {}

        if len(active_metrics) == 1:
            m = active_metrics[0]
            results[m.name] = m.score(
                prompt=prompt,
                response=response,
                context=clean_context,
                provider=self.provider,
                temperature=self.config.temperature,
                include_reasoning=self.config.include_reasoning,
            )
        elif len(active_metrics) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(active_metrics)
            ) as executor:
                future_to_metric = {
                    executor.submit(
                        m.score,
                        prompt=prompt,
                        response=response,
                        context=clean_context,
                        provider=self.provider,
                        temperature=self.config.temperature,
                        include_reasoning=self.config.include_reasoning,
                    ): m
                    for m in active_metrics
                }
                for future in concurrent.futures.as_completed(future_to_metric):
                    m = future_to_metric[future]
                    results[m.name] = future.result()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        relevance_res = results.get("relevance")
        completeness_res = results.get("completeness")
        faithfulness_res = results.get("faithfulness")

        relevance_score = relevance_res.score if relevance_res else 0.0
        completeness_score = completeness_res.score if completeness_res else 0.0
        faithfulness_score = faithfulness_res.score if faithfulness_res else None

        overall_score = calculate_overall(
            relevance=relevance_score,
            completeness=completeness_score,
            faithfulness=faithfulness_score,
            config=self.config,
        )

        reasoning: dict[str, str] = {}
        for name, res in results.items():
            if res.reasoning:
                reasoning[name] = res.reasoning

        provider_name = getattr(self.provider, "name", "unknown")
        model_name = getattr(self.provider, "model", "") or ""

        metadata = EvaluationMetadata(
            provider=provider_name,
            model=model_name,
            latency_ms=round(elapsed_ms, 2),
        )

        return EvaluationResult(
            relevance=relevance_score,
            completeness=completeness_score,
            faithfulness=faithfulness_score,
            overall=overall_score,
            reasoning=reasoning,
            metadata=metadata,
        )


def evaluate(
    prompt: str,
    response: str,
    context: str | None = None,
    *,
    provider: LLMProvider | None = None,
    config: EvaluatorConfig | None = None,
    **kwargs: Any,
) -> EvaluationResult:
    """Evaluate an LLM response against a prompt and optional context.

    This is the primary synchronous public entrypoint of ai-dev-toolkit.

    Example:
        >>> from ai_dev_toolkit import evaluate
        >>> result = evaluate(
        ...     prompt="What is the capital of France?",
        ...     response="The capital of France is Paris.",
        ... )
        >>> print(result.overall)

    Args:
        prompt: The prompt or question asked to the LLM.
        response: The LLM output to evaluate.
        context: Optional reference text for faithfulness / hallucination checks.
        provider: Optional LLMProvider instance.
        config: Optional EvaluatorConfig for weights and settings.
        **kwargs: Additional overrides for EvaluatorConfig (e.g. model, temperature).

    Returns:
        EvaluationResult containing scores, reasoning, and execution metadata.
    """
    if kwargs:
        cfg_dict = config.model_dump() if config else EvaluatorConfig().model_dump()
        cfg_dict.update(kwargs)
        config = EvaluatorConfig(**cfg_dict)

    evaluator = Evaluator(provider=provider, config=config)
    return evaluator.evaluate(prompt=prompt, response=response, context=context)


async def aevaluate(
    prompt: str,
    response: str,
    context: str | None = None,
    *,
    provider: LLMProvider | None = None,
    config: EvaluatorConfig | None = None,
    **kwargs: Any,
) -> EvaluationResult:
    """Asynchronously evaluate an LLM response against a prompt and optional context.

    Example:
        >>> import asyncio
        >>> from ai_dev_toolkit import aevaluate
        >>> result = asyncio.run(aevaluate(
        ...     prompt="What is the capital of France?",
        ...     response="Paris",
        ... ))
        >>> print(result.overall)
    """
    return await asyncio.to_thread(
        evaluate,
        prompt=prompt,
        response=response,
        context=context,
        provider=provider,
        config=config,
        **kwargs,
    )
