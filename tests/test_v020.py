"""Unit tests for Phase 11 (v0.2.0):
OpenAI, Ollama, Registry, Config files, Datasets, Async.
"""

import asyncio
import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_toolkit import (
    DatasetEvaluationResult,
    EvaluationResult,
    EvaluatorConfig,
    Metric,
    MetricError,
    MetricResult,
    ProviderError,
    aevaluate,
    evaluate_dataset,
    get_metric,
    list_metrics,
    register_metric,
    reset_metrics,
)
from ai_dev_toolkit.providers.fake import FakeProvider
from ai_dev_toolkit.providers.ollama import OllamaProvider
from ai_dev_toolkit.providers.openai import OpenAIProvider


class TestOpenAIProvider:
    """Tests for OpenAIProvider implementation."""

    def test_missing_api_key_raises_error(self) -> None:
        with patch("ai_dev_toolkit.providers.openai._find_env_key", return_value=None):
            with pytest.raises(ProviderError, match="OPENAI_API_KEY is not set"):
                OpenAIProvider(api_key=None)

    def test_successful_generation(self) -> None:
        mock_response_data = {
            "choices": [
                {"message": {"content": '{"score": 1.0, "reasoning": "Great"}'}}
            ]
        }
        mock_response_bytes = json.dumps(mock_response_data).encode("utf-8")

        mock_http_response = MagicMock()
        mock_http_response.read.return_value = mock_response_bytes
        mock_http_response.__enter__.return_value = mock_http_response

        with patch("urllib.request.urlopen", return_value=mock_http_response):
            provider = OpenAIProvider(api_key="sk-test-key-12345")
            text = provider.generate("Say hello")
            assert '{"score": 1.0' in text
            assert provider.name == "openai"

    def test_sanitizes_api_key_on_error(self) -> None:
        error_with_key = urllib.error.HTTPError(
            url="https://api.openai.com",
            code=401,
            msg="Unauthorized with sk-test-key-12345",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error": "Invalid sk-test-key-12345"}'),
        )

        with patch("urllib.request.urlopen", side_effect=error_with_key):
            provider = OpenAIProvider(api_key="sk-test-key-12345", max_retries=0)
            with pytest.raises(ProviderError) as exc_info:
                provider.generate("Test prompt")
            assert "sk-test-key-12345" not in str(exc_info.value)
            assert "[REDACTED_API_KEY]" in str(exc_info.value)


class TestOllamaProvider:
    """Tests for OllamaProvider implementation."""

    def test_successful_generation(self) -> None:
        mock_response_data = {"response": '{"score": 0.95, "reasoning": "Good"}'}
        mock_response_bytes = json.dumps(mock_response_data).encode("utf-8")

        mock_http_response = MagicMock()
        mock_http_response.read.return_value = mock_response_bytes
        mock_http_response.__enter__.return_value = mock_http_response

        with patch("urllib.request.urlopen", return_value=mock_http_response):
            provider = OllamaProvider(model="llama3")
            text = provider.generate("Evaluate this")
            assert '{"score": 0.95' in text
            assert provider.name == "ollama"

    def test_connection_error_handling(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(reason="Connection refused"),
        ):
            provider = OllamaProvider(max_retries=1, initial_backoff_sec=0.01)
            with pytest.raises(ProviderError, match="Ollama request failed after"):
                provider.generate("Hello")


class TestMetricRegistry:
    """Tests for custom metric registration."""

    def setup_method(self) -> None:
        reset_metrics()

    def teardown_method(self) -> None:
        reset_metrics()

    def test_builtin_metrics_registered(self) -> None:
        metrics = list_metrics()
        assert "relevance" in metrics
        assert "completeness" in metrics
        assert "faithfulness" in metrics

    def test_register_custom_metric(self) -> None:
        class WordCountMetric(Metric):
            @property
            def name(self) -> str:
                return "word_count"

            @property
            def requires_context(self) -> bool:
                return False

            def score(
                self,
                prompt: str,
                response: str,
                context: str | None,
                provider: object,
                *,
                temperature: float = 0.0,
                include_reasoning: bool = True,
            ) -> MetricResult:
                count = len(response.split())
                return MetricResult(
                    score=min(1.0, count / 10.0), reasoning=f"{count} words"
                )

        register_metric(WordCountMetric)
        assert "word_count" in list_metrics()

        metric_inst = get_metric("word_count")
        assert metric_inst.name == "word_count"
        res = metric_inst.score("prompt", "one two three", None, FakeProvider())
        assert res.score == 0.3

    def test_get_unknown_metric_raises_error(self) -> None:
        with pytest.raises(MetricError, match="Metric 'nonexistent' not found"):
            get_metric("nonexistent")


class TestConfigFileLoading:
    """Tests for EvaluatorConfig from TOML/YAML/JSON files."""

    def test_load_from_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "ai-dev.toml"
        toml_file.write_text(
            """
            [evaluator]
            relevance_weight = 0.5
            completeness_weight = 0.3
            faithfulness_weight = 0.2
            temperature = 0.1
            provider = "fake"
            """,
            encoding="utf-8",
        )
        config = EvaluatorConfig.from_file(toml_file)
        assert config.relevance_weight == 0.5
        assert config.completeness_weight == 0.3
        assert config.temperature == 0.1
        assert config.provider == "fake"

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "ai-dev.yaml"
        yaml_file.write_text(
            """
            relevance_weight: 0.6
            completeness_weight: 0.4
            faithfulness_weight: 0.0
            provider: fake
            """,
            encoding="utf-8",
        )
        config = EvaluatorConfig.from_file(yaml_file)
        assert config.relevance_weight == 0.6
        assert config.completeness_weight == 0.4

    def test_load_from_json(self, tmp_path: Path) -> None:
        json_file = tmp_path / "ai-dev.json"
        json_file.write_text(
            json.dumps(
                {
                    "relevance_weight": 0.7,
                    "completeness_weight": 0.3,
                    "faithfulness_weight": 0.0,
                }
            ),
            encoding="utf-8",
        )
        config = EvaluatorConfig.from_file(json_file)
        assert config.relevance_weight == 0.7


class TestDatasetEvaluation:
    """Tests for dataset evaluation and summary metrics."""

    def test_evaluate_dataset_jsonl(self, tmp_path: Path) -> None:
        data_file = tmp_path / "dataset.jsonl"
        lines = [
            json.dumps({"id": "1", "prompt": "P1", "response": "R1"}),
            json.dumps({"id": "2", "prompt": "P2", "response": "R2", "context": "C2"}),
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        provider = FakeProvider(
            responses=lambda p, *a, **k: json.dumps(
                {
                    "score": 0.9,
                    "reasoning": "Good",
                    "sub_questions": ["Q1"],
                    "claims": ["Claim 1"],
                    "verdicts": [{"claim": "Claim 1", "supported": True}],
                }
            )
        )

        dataset_result = evaluate_dataset(data_file, provider=provider)
        assert isinstance(dataset_result, DatasetEvaluationResult)
        assert dataset_result.total_samples == 2
        assert "overall" in dataset_result.summary
        assert "relevance" in dataset_result.summary
        assert dataset_result.summary["overall"].mean == 0.9

        # Test CSV export
        csv_file = tmp_path / "output.csv"
        dataset_result.to_csv(csv_file)
        assert csv_file.is_file()
        assert "overall" in csv_file.read_text(encoding="utf-8")

    def test_evaluate_dataset_csv_and_json(self, tmp_path: Path) -> None:
        # JSON list format
        json_file = tmp_path / "dataset.json"
        json_file.write_text(
            json.dumps(
                [
                    {"prompt": "P1", "response": "R1"},
                    {"prompt": "P2", "response": "R2"},
                ]
            ),
            encoding="utf-8",
        )
        provider = FakeProvider(
            responses=lambda p, *a, **k: json.dumps(
                {"score": 0.8, "reasoning": "Ok", "sub_questions": []}
            )
        )
        res_json = evaluate_dataset(json_file, provider=provider)
        assert res_json.total_samples == 2

        # CSV format
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "prompt,response\nP1,R1\nP2,R2\n",
            encoding="utf-8",
        )
        res_csv = evaluate_dataset(csv_file, provider=provider)
        assert res_csv.total_samples == 2

    def test_invalid_dataset_format_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "dataset.unknown"
        bad_file.write_text("dummy", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            evaluate_dataset(bad_file, provider=FakeProvider())


class TestAsyncEvaluation:
    """Tests for async aevaluate() API."""

    def test_aevaluate_executes_asynchronously(self) -> None:
        provider = FakeProvider(
            responses=lambda p, *a, **k: json.dumps(
                {
                    "score": 0.95,
                    "reasoning": "Async OK",
                    "sub_questions": [],
                }
            )
        )

        async def run_async_test() -> EvaluationResult:
            return await aevaluate(
                prompt="Async prompt",
                response="Async response",
                provider=provider,
            )

        result = asyncio.run(run_async_test())
        assert isinstance(result, EvaluationResult)
        assert result.overall == 0.95


class TestRetryAndConfigSearch:
    """Additional tests for OpenAI retries and find_and_load config."""

    def test_openai_retry_then_success(self) -> None:
        mock_429 = urllib.error.HTTPError(
            url="https://api.openai.com",
            code=429,
            msg="Rate limit exceeded",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error": "rate limit"}'),
        )
        mock_ok = MagicMock()
        mock_ok.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "Resolved"}}]}
        ).encode("utf-8")
        mock_ok.__enter__.return_value = mock_ok

        with patch("urllib.request.urlopen", side_effect=[mock_429, mock_ok]):
            provider = OpenAIProvider(
                api_key="sk-test",
                max_retries=2,
                initial_backoff_sec=0.01,
            )
            resp = provider.generate("Test prompt")
            assert resp == "Resolved"

    def test_find_and_load_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "ai-dev.toml"
        cfg_file.write_text(
            """
            relevance_weight = 0.5
            completeness_weight = 0.5
            faithfulness_weight = 0.0
            """,
            encoding="utf-8",
        )
        loaded = EvaluatorConfig.find_and_load(directory=tmp_path)
        assert loaded is not None
        assert loaded.relevance_weight == 0.5

    def test_find_and_load_not_found(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        loaded = EvaluatorConfig.find_and_load(directory=empty_dir)
        # Should return default config if none found
        assert isinstance(loaded, EvaluatorConfig)
