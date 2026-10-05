"""Unit tests for core models, errors, config, and results."""

import json

import pytest
from pydantic import ValidationError

from ai_dev_toolkit.core.config import EvaluatorConfig
from ai_dev_toolkit.core.errors import (
    AIDevToolkitError,
    MetricError,
    ProviderError,
)
from ai_dev_toolkit.core.result import (
    EvaluationMetadata,
    EvaluationResult,
    MetricResult,
    calculate_overall,
    clamp_score,
)


def test_errors_hierarchy() -> None:
    """Test custom error classes and their attributes."""
    base_err = AIDevToolkitError("Base error", details={"code": 123})
    assert str(base_err) == "Base error"
    assert base_err.details == {"code": 123}

    prov_err = ProviderError(
        "Rate limit exceeded",
        provider_name="gemini",
        status_code=429,
        details={"retry_after": 5},
    )
    assert isinstance(prov_err, AIDevToolkitError)
    assert prov_err.provider_name == "gemini"
    assert prov_err.status_code == 429
    assert prov_err.details == {"retry_after": 5}

    metric_err = MetricError(
        "Failed to parse json",
        metric_name="relevance",
        raw_output="invalid json string",
    )
    assert isinstance(metric_err, AIDevToolkitError)
    assert metric_err.metric_name == "relevance"
    assert metric_err.raw_output == "invalid json string"


def test_evaluator_config_defaults() -> None:
    """Test default values of EvaluatorConfig."""
    cfg = EvaluatorConfig()
    assert cfg.relevance_weight == 0.35
    assert cfg.completeness_weight == 0.25
    assert cfg.faithfulness_weight == 0.40
    assert cfg.temperature == 0.0
    assert cfg.timeout == 30.0
    assert cfg.max_retries == 3
    assert cfg.include_reasoning is True


def test_evaluator_config_validation() -> None:
    """Test validation constraints on EvaluatorConfig."""
    with pytest.raises(ValidationError):
        EvaluatorConfig(relevance_weight=-0.1)

    with pytest.raises(ValidationError):
        EvaluatorConfig(temperature=-1.0)

    with pytest.raises(ValidationError):
        EvaluatorConfig(timeout=0.0)

    with pytest.raises(ValidationError):
        EvaluatorConfig(
            relevance_weight=0.0,
            completeness_weight=0.0,
            faithfulness_weight=0.0,
        )


def test_clamp_score() -> None:
    """Test clamp_score utility function."""
    assert clamp_score(-0.5) == 0.0
    assert clamp_score(1.5) == 1.0
    assert clamp_score(0.75) == 0.75


def test_metric_result_clamping() -> None:
    """Test MetricResult clamping and field initialization."""
    res_low = MetricResult(score=-0.2, reasoning="Too low")
    assert res_low.score == 0.0
    assert res_low.reasoning == "Too low"

    res_high = MetricResult(score=1.8, metadata={"tokens": 42})
    assert res_high.score == 1.0
    assert res_high.metadata == {"tokens": 42}


def test_calculate_overall_with_faithfulness() -> None:
    """Test calculate_overall with all three metrics provided."""
    # 0.35 * 1.0 + 0.25 * 0.8 + 0.40 * 0.9 = 0.35 + 0.20 + 0.36 = 0.91
    score = calculate_overall(relevance=1.0, completeness=0.8, faithfulness=0.9)
    assert score == pytest.approx(0.91, abs=1e-4)


def test_calculate_overall_without_faithfulness() -> None:
    """Test calculate_overall when faithfulness is None (renormalization)."""
    # Relevance (0.35) and completeness (0.25) -> sum = 0.60
    # rel = 0.9, comp = 0.6 -> (0.9 * 0.35 + 0.6 * 0.25) / 0.60 = 0.465 / 0.60 = 0.775
    score = calculate_overall(relevance=0.9, completeness=0.6, faithfulness=None)
    assert score == pytest.approx(0.775, abs=1e-4)


def test_calculate_overall_with_custom_config() -> None:
    """Test calculate_overall with custom weights."""
    cfg = EvaluatorConfig(
        relevance_weight=0.5,
        completeness_weight=0.5,
        faithfulness_weight=0.0,
    )
    score = calculate_overall(
        relevance=0.8,
        completeness=0.6,
        faithfulness=0.2,
        config=cfg,
    )
    assert score == pytest.approx(0.7, abs=1e-4)


def test_calculate_overall_invalid_weights() -> None:
    """Test error when renormalized weights sum to zero."""
    cfg = EvaluatorConfig(
        relevance_weight=0.0,
        completeness_weight=0.0,
        faithfulness_weight=1.0,
    )
    with pytest.raises(ValueError, match="Sum of relevance and completeness"):
        calculate_overall(
            relevance=0.8,
            completeness=0.8,
            faithfulness=None,
            config=cfg,
        )

    # Bypass Pydantic validation to test defensive runtime check in calculate_overall
    cfg_zero = EvaluatorConfig.model_construct(
        relevance_weight=0.0,
        completeness_weight=0.0,
        faithfulness_weight=0.0,
    )
    with pytest.raises(ValueError, match="Sum of all metric weights must be > 0"):
        calculate_overall(
            relevance=0.8,
            completeness=0.8,
            faithfulness=0.8,
            config=cfg_zero,
        )


def test_evaluation_result_with_faithfulness() -> None:
    """Test EvaluationResult and hallucination computation with context."""
    res = EvaluationResult(
        relevance=0.92,
        completeness=0.86,
        faithfulness=0.94,
        overall=0.91,
        reasoning={
            "relevance": "Directly answers the question.",
            "completeness": "Covers definition; omits advantages.",
            "faithfulness": "All claims supported by context.",
        },
        metadata=EvaluationMetadata(
            provider="gemini",
            model="gemini-2.5-flash",
            latency_ms=1840,
        ),
    )

    assert res.relevance == 0.92
    assert res.completeness == 0.86
    assert res.faithfulness == 0.94
    assert res.hallucination == pytest.approx(0.06, abs=1e-4)
    assert res.overall == 0.91
    assert res.reasoning["relevance"] == "Directly answers the question."
    assert res.metadata.latency_ms == 1840


def test_evaluation_result_without_faithfulness() -> None:
    """Test EvaluationResult when faithfulness is None."""
    res = EvaluationResult(
        relevance=0.90,
        completeness=0.80,
        faithfulness=None,
        overall=0.85,
    )

    assert res.faithfulness is None
    assert res.hallucination is None
    assert res.relevance == 0.90
    assert res.completeness == 0.80


def test_evaluation_result_clamping() -> None:
    """Test clamping of out-of-range values in EvaluationResult."""
    res = EvaluationResult(
        relevance=-0.1,
        completeness=1.5,
        faithfulness=1.2,
        overall=2.0,
    )
    assert res.relevance == 0.0
    assert res.completeness == 1.0
    assert res.faithfulness == 1.0
    assert res.hallucination == 0.0
    assert res.overall == 1.0


def test_evaluation_result_to_dict_and_to_json() -> None:
    """Test dictionary and JSON serialization matching PRD schema."""
    res = EvaluationResult(
        relevance=0.92,
        completeness=0.86,
        faithfulness=0.94,
        overall=0.91,
        reasoning={"relevance": "Good"},
        metadata=EvaluationMetadata(
            provider="gemini",
            model="gemini-2.5-flash",
            toolkit_version="0.1.0",
            latency_ms=100,
        ),
    )

    d = res.to_dict()
    assert d["relevance"] == 0.92
    assert d["completeness"] == 0.86
    assert d["faithfulness"] == 0.94
    assert d["hallucination"] == 0.06
    assert d["overall"] == 0.91
    assert d["reasoning"]["relevance"] == "Good"
    assert d["metadata"]["provider"] == "gemini"
    assert d["metadata"]["latency_ms"] == 100

    raw_json = res.to_json(indent=2)
    parsed = json.loads(raw_json)
    assert parsed == d
