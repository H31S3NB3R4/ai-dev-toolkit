"""Unit and integration tests for Evaluator and evaluate() public API."""

import json

import pytest

from ai_dev_toolkit import (
    EvaluationResult,
    Evaluator,
    EvaluatorConfig,
    evaluate,
)
from ai_dev_toolkit.providers.fake import FakeProvider


class TestEvaluatorValidation:
    """Test input validation on Evaluator and evaluate()."""

    def test_empty_prompt_raises_value_error(self) -> None:
        provider = FakeProvider(responses=["{}"])
        evaluator = Evaluator(provider=provider)
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            evaluator.evaluate(prompt="", response="Some response")

    def test_whitespace_prompt_raises_value_error(self) -> None:
        provider = FakeProvider(responses=["{}"])
        evaluator = Evaluator(provider=provider)
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            evaluator.evaluate(prompt="   \n\t  ", response="Some response")

    def test_empty_response_raises_value_error(self) -> None:
        provider = FakeProvider(responses=["{}"])
        evaluator = Evaluator(provider=provider)
        with pytest.raises(ValueError, match="Response must be a non-empty string"):
            evaluator.evaluate(prompt="Some prompt", response="")

    def test_whitespace_response_raises_value_error(self) -> None:
        provider = FakeProvider(responses=["{}"])
        evaluator = Evaluator(provider=provider)
        with pytest.raises(ValueError, match="Response must be a non-empty string"):
            evaluator.evaluate(prompt="Some prompt", response="   \n  ")

    def test_unsupported_provider_raises_value_error(self) -> None:
        config = EvaluatorConfig(provider="unsupported_llm_provider")
        with pytest.raises(
            ValueError, match="Unsupported provider 'unsupported_llm_provider'"
        ):
            Evaluator(config=config)


class TestEvaluatorExecution:
    """Test evaluation logic with FakeProvider."""

    def test_evaluate_without_context(self) -> None:
        # Without context: relevance and completeness run, faithfulness is skipped
        def fake_generate(prompt: str, *args: object, **kwargs: object) -> str:
            if "sub-questions" in prompt:
                return json.dumps(
                    {
                        "sub_questions": ["What is Python?"],
                        "score": 1.0,
                        "reasoning": "All questions answered",
                    }
                )
            return json.dumps(
                {
                    "score": 0.9,
                    "reasoning": "Very relevant",
                }
            )

        provider = FakeProvider(responses=fake_generate)
        result = evaluate(
            prompt="What is Python?",
            response="Python is a programming language.",
            provider=provider,
        )

        assert isinstance(result, EvaluationResult)
        assert result.relevance == 0.9
        assert result.completeness == 1.0
        assert result.faithfulness is None
        assert result.hallucination is None
        # Weight renormalization:
        # (0.35 * 0.9 + 0.25 * 1.0) / 0.60 = 0.565 / 0.60 = 0.9417
        assert 0.94 <= result.overall <= 0.95
        assert result.reasoning["relevance"] == "Very relevant"
        assert result.reasoning["completeness"] == "All questions answered"
        assert result.metadata.provider == "fake"
        assert result.metadata.latency_ms is not None
        assert result.metadata.latency_ms >= 0

    def test_evaluate_with_context(self) -> None:
        # With context: relevance, completeness, and faithfulness all run
        def fake_generate(prompt: str, *args: object, **kwargs: object) -> str:
            if "sub-questions" in prompt:
                return json.dumps(
                    {
                        "sub_questions": ["What is X?"],
                        "score": 1.0,
                        "reasoning": "Complete answer",
                    }
                )
            elif (
                "verify" in prompt.lower()
                or "verdicts" in prompt.lower()
                or "supported" in prompt.lower()
            ):
                return json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim": "Paris is the capital of France.",
                                "supported": True,
                            }
                        ],
                        "score": 1.0,
                        "reasoning": "Fully faithful",
                    }
                )
            elif "extract" in prompt.lower():
                return json.dumps({"claims": ["Paris is the capital of France."]})
            return json.dumps({"score": 1.0, "reasoning": "Direct answer"})

        provider = FakeProvider(responses=fake_generate)
        result = evaluate(
            prompt="What is the capital of France?",
            response="Paris is the capital of France.",
            context="Paris is the capital city of France.",
            provider=provider,
        )

        assert result.relevance == 1.0
        assert result.completeness == 1.0
        assert result.faithfulness == 1.0
        assert result.hallucination == 0.0
        assert result.overall == 1.0
        assert "relevance" in result.reasoning
        assert "faithfulness" in result.reasoning

    def test_evaluate_with_whitespace_context_treated_as_none(self) -> None:
        def fake_generate(prompt: str, *args: object, **kwargs: object) -> str:
            return json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "OK",
                    "sub_questions": ["Q1"],
                }
            )

        provider = FakeProvider(responses=fake_generate)
        result = evaluate(
            prompt="Tell me a joke",
            response="Why did the chicken cross the road?",
            context="    \n   ",
            provider=provider,
        )

        assert result.faithfulness is None
        assert result.hallucination is None

    def test_evaluate_with_custom_config(self) -> None:
        config = EvaluatorConfig(
            relevance_weight=0.8,
            completeness_weight=0.2,
            faithfulness_weight=0.0,
            temperature=0.2,
        )

        def fake_generate(prompt: str, *args: object, **kwargs: object) -> str:
            if "sub-questions" in prompt:
                return json.dumps(
                    {
                        "sub_questions": ["Q1"],
                        "score": 0.0,
                        "reasoning": "Incomplete",
                    }
                )
            return json.dumps({"score": 1.0, "reasoning": "Perfect relevance"})

        provider = FakeProvider(responses=fake_generate)
        result = evaluate(
            prompt="Explain quantum computing",
            response="Quantum computing uses qubits.",
            provider=provider,
            config=config,
        )

        # overall = 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        assert result.overall == 0.8
        assert result.relevance == 1.0
        assert result.completeness == 0.0

    def test_evaluate_kwargs_override_config(self) -> None:
        provider = FakeProvider(
            responses=lambda p, *args, **kwargs: json.dumps(
                {
                    "score": 1.0,
                    "sub_questions": [],
                    "reasoning": "OK",
                }
            )
        )
        result = evaluate(
            prompt="Hello",
            response="Hi there!",
            provider=provider,
            temperature=0.5,
        )
        assert result.overall == 1.0

    def test_serialization_methods(self) -> None:
        provider = FakeProvider(
            responses=lambda p, *args, **kwargs: json.dumps(
                {
                    "score": 0.85,
                    "reasoning": "Good",
                    "sub_questions": ["Q1"],
                }
            )
        )
        result = evaluate(
            prompt="Summarize the article",
            response="Here is the summary.",
            provider=provider,
        )

        data_dict = result.to_dict()
        assert isinstance(data_dict, dict)
        assert "relevance" in data_dict
        assert "completeness" in data_dict
        assert "overall" in data_dict
        assert "metadata" in data_dict

        json_str = result.to_json(indent=2)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["overall"] == result.overall
