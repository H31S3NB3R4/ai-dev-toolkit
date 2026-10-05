"""Unit tests for provider interfaces, FakeProvider, and GeminiProvider."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_dev_toolkit.core.errors import ProviderError
from ai_dev_toolkit.providers import get_provider
from ai_dev_toolkit.providers.base import LLMProvider
from ai_dev_toolkit.providers.fake import FakeProvider
from ai_dev_toolkit.providers.gemini import GeminiProvider, _sanitize_message


def test_fake_provider_protocol_conformance() -> None:
    """Test FakeProvider conforms to LLMProvider protocol."""
    provider = FakeProvider(default_response="ok")
    assert isinstance(provider, LLMProvider)
    assert provider.name == "fake"
    assert provider.generate("test prompt") == "ok"
    assert provider.call_count == 1
    assert provider.prompts == ["test prompt"]


def test_fake_provider_sequential_responses() -> None:
    """Test FakeProvider returns sequential responses."""
    provider = FakeProvider(responses=["first", "second"], default_response="fallback")
    assert provider.generate("p1") == "first"
    assert provider.generate("p2") == "second"
    assert provider.generate("p3") == "fallback"
    assert provider.call_count == 3


def test_fake_provider_mapping_responses() -> None:
    """Test FakeProvider matches prompt substrings."""
    provider = FakeProvider(
        responses={"relevance": '{"score": 0.9}', "completeness": '{"score": 0.8}'},
        default_response="{}",
    )
    assert provider.generate("Evaluate relevance please") == '{"score": 0.9}'
    assert provider.generate("Check completeness here") == '{"score": 0.8}'
    assert provider.generate("Other metric") == "{}"


def test_fake_provider_error_raising() -> None:
    """Test FakeProvider raises configured error."""
    provider = FakeProvider(error_to_raise=RuntimeError("Simulated LLM crash"))
    with pytest.raises(RuntimeError, match="Simulated LLM crash"):
        provider.generate("test")


def test_gemini_provider_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test GeminiProvider raises ProviderError when GEMINI_API_KEY is not set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("os.path.isfile", return_value=False):
        with pytest.raises(ProviderError, match="GEMINI_API_KEY is not set"):
            GeminiProvider()


def test_gemini_provider_env_file_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test reading GEMINI_API_KEY from .env file."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('GEMINI_API_KEY="test-dotenv-key"\n', encoding="utf-8")

    with patch("os.getcwd", return_value=str(tmp_path)):
        provider = GeminiProvider()
        assert provider._api_key == "test-dotenv-key"


def test_gemini_provider_api_key_sanitization() -> None:
    """Ensure API key never appears in str, repr, or error messages."""
    secret_key = "AIzaSySecretTestKey123456789012345"
    provider = GeminiProvider(api_key=secret_key, model="gemini-2.5-flash")

    assert secret_key not in repr(provider)
    assert secret_key not in str(provider)

    raw_msg = f"Error communicating with {secret_key} endpoint"
    sanitized = _sanitize_message(raw_msg, secret_key)
    assert secret_key not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_gemini_provider_mock_generate() -> None:
    """Test GeminiProvider generate call with mocked GenAI client."""
    provider = GeminiProvider(api_key="fake-key-test")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"relevance": 0.95}'
    mock_client.models.generate_content.return_value = mock_response
    provider._client = mock_client

    result = provider.generate("Test query", temperature=0.2)
    assert result == '{"relevance": 0.95}'
    mock_client.models.generate_content.assert_called_once()


def test_gemini_provider_mock_retry_success() -> None:
    """Test GeminiProvider retries on transient error and succeeds."""
    provider = GeminiProvider(
        api_key="fake-key-test",
        max_retries=2,
        initial_backoff_sec=0.01,
        backoff_factor=1.0,
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "success after retry"

    mock_client.models.generate_content.side_effect = [
        RuntimeError("Transient 503 error"),
        mock_response,
    ]
    provider._client = mock_client

    result = provider.generate("Retry test")
    assert result == "success after retry"
    assert mock_client.models.generate_content.call_count == 2


def test_gemini_provider_mock_retry_exhausted() -> None:
    """Test GeminiProvider wraps exhaustion error in ProviderError."""
    provider = GeminiProvider(
        api_key="fake-key-test",
        max_retries=1,
        initial_backoff_sec=0.01,
        backoff_factor=1.0,
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Rate limit 429")
    provider._client = mock_client

    with pytest.raises(ProviderError) as exc_info:
        provider.generate("Fail test")

    assert "Gemini API request failed after 2 attempts" in str(exc_info.value)
    assert exc_info.value.provider_name == "gemini"


def test_fake_provider_single_string_response() -> None:
    """Test FakeProvider with constant string response and repr."""
    provider = FakeProvider(responses="fixed output")
    assert provider.generate("hello") == "fixed output"
    assert "FakeProvider(call_count=1)" in repr(provider)


def test_gemini_provider_none_text_response() -> None:
    """Test GeminiProvider handles None text attribute gracefully."""
    provider = GeminiProvider(api_key="fake-key-test")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = None
    mock_client.models.generate_content.return_value = mock_response
    provider._client = mock_client

    assert provider.generate("test") == ""


def test_gemini_provider_client_init_failure() -> None:
    """Test GeminiProvider handles client initialization failure."""
    provider = GeminiProvider(api_key="fake-key-test")
    with patch("google.genai.Client", side_effect=RuntimeError("SDK init failed")):
        with pytest.raises(ProviderError, match="Failed to initialize Gemini client"):
            provider._get_client()


def test_get_provider_factory() -> None:
    """Test get_provider factory function."""
    fake = get_provider("fake", default_response="hello")
    assert isinstance(fake, FakeProvider)
    assert fake.generate("test") == "hello"

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        gemini = get_provider("gemini")
        assert isinstance(gemini, GeminiProvider)
        assert isinstance(gemini, LLMProvider)

    with pytest.raises(ProviderError, match="Unknown provider 'unsupported'"):
        get_provider("unsupported")


def test_zero_gemini_imports_in_core() -> None:
    """Verify architectural requirement: core code has zero Gemini imports."""
    import inspect

    import ai_dev_toolkit.core
    import ai_dev_toolkit.core.config
    import ai_dev_toolkit.core.errors
    import ai_dev_toolkit.core.result

    for mod in [
        ai_dev_toolkit.core,
        ai_dev_toolkit.core.config,
        ai_dev_toolkit.core.errors,
        ai_dev_toolkit.core.result,
    ]:
        source = inspect.getsource(mod)
        assert "google.genai" not in source
        assert "GeminiProvider" not in source
