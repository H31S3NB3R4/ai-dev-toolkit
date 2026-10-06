"""Gemini LLM provider implementation using the official google-genai SDK."""

import logging
import os
import re
import time
from typing import Any

from ai_dev_toolkit.core.errors import ProviderError

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _find_env_key(key_name: str = "GEMINI_API_KEY") -> str | None:
    """Retrieve key from environment or local .env file."""
    val = os.getenv(key_name)
    if val:
        return val
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == key_name:
                            return v.strip().strip("'\"")
        except Exception:
            pass
    return None


def _sanitize_message(message: str, api_key: str | None = None) -> str:
    """Remove any API key occurrences from error strings and logs."""
    sanitized = message
    if api_key:
        sanitized = sanitized.replace(api_key, "[REDACTED_API_KEY]")
    # Also catch general Gemini API key patterns if any
    sanitized = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_API_KEY]", sanitized)
    return sanitized


class GeminiProvider:
    """LLM provider implementation for Google Gemini models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        max_retries: int = 3,
        initial_backoff_sec: float = 1.0,
        backoff_factor: float = 2.0,
        timeout: float = 30.0,
    ) -> None:
        """Initialize GeminiProvider.

        Args:
            api_key: Optional Gemini API key. If not provided, reads GEMINI_API_KEY
                from the environment.
            model: Gemini model identifier (default: 'gemini-2.5-flash').
            max_retries: Maximum retry attempts for transient API errors.
            initial_backoff_sec: Initial sleep duration between retries in seconds.
            backoff_factor: Exponential multiplier for retry backoff.
            timeout: Request timeout in seconds.

        Raises:
            ProviderError: If no API key is provided or found in the environment.
        """
        self._api_key = api_key or _find_env_key("GEMINI_API_KEY")
        if not self._api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY "
                "environment variable or pass api_key to GeminiProvider.",
                provider_name="gemini",
            )

        self.model = model
        self.max_retries = max(0, max_retries)
        self.initial_backoff_sec = initial_backoff_sec
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self._client: Any = None

    @property
    def name(self) -> str:
        """Return provider identifier name."""
        return "gemini"

    def _get_client(self) -> Any:
        """Lazy-initialize Google GenAI client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:
                sanitized_msg = _sanitize_message(str(e), self._api_key)
                raise ProviderError(
                    f"Failed to initialize Gemini client: {sanitized_msg}",
                    provider_name="gemini",
                    details={"error": sanitized_msg},
                ) from e
        return self._client

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Generate text using the configured Gemini model with retry and backoff.

        Args:
            prompt: Text prompt to send to Gemini.
            temperature: Sampling temperature for model output.

        Returns:
            The generated response string.

        Raises:
            ProviderError: If the request fails after maximum retries.
        """
        client = self._get_client()

        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
        )

        last_error: Exception | None = None
        current_delay = self.initial_backoff_sec

        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                if response.text is not None:
                    return str(response.text)
                return ""
            except Exception as e:
                last_error = e
                sanitized_err = _sanitize_message(str(e), self._api_key)
                logger.warning(
                    "Gemini API attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    sanitized_err,
                )

                if attempt < self.max_retries:
                    time.sleep(current_delay)
                    current_delay *= self.backoff_factor

        error_message = (
            f"Gemini API request failed after {self.max_retries + 1} attempts: "
            f"{_sanitize_message(str(last_error), self._api_key)}"
        )
        raise ProviderError(
            error_message,
            provider_name="gemini",
            details={"model": self.model, "attempts": self.max_retries + 1},
        ) from last_error

    def __repr__(self) -> str:
        """Safe representation hiding API key."""
        return f"GeminiProvider(model='{self.model}', max_retries={self.max_retries})"

    def __str__(self) -> str:
        """Safe string conversion hiding API key."""
        return self.__repr__()
