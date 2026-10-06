"""OpenAI LLM provider implementation with retry and key sanitization."""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from ai_dev_toolkit.core.errors import ProviderError

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _find_env_key(key_name: str = "OPENAI_API_KEY") -> str | None:
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
    """Remove any secret API key occurrences from error strings and logs."""
    sanitized = message
    if api_key:
        sanitized = sanitized.replace(api_key, "[REDACTED_API_KEY]")
    sanitized = re.sub(r"sk-[0-9A-Za-z\-_]{20,}", "[REDACTED_API_KEY]", sanitized)
    return sanitized


class OpenAIProvider:
    """LLM provider for OpenAI and OpenAI-compatible API endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        max_retries: int = 3,
        initial_backoff_sec: float = 1.0,
        backoff_factor: float = 2.0,
        timeout: float = 30.0,
    ) -> None:
        """Initialize OpenAIProvider.

        Args:
            api_key: OpenAI API key. If not provided, reads OPENAI_API_KEY
                from environment.
            model: Model identifier (default: 'gpt-4o-mini').
            base_url: Base endpoint URL (default: 'https://api.openai.com/v1').
            max_retries: Maximum retry attempts for transient errors.
            initial_backoff_sec: Initial sleep duration in seconds between retries.
            backoff_factor: Exponential multiplier for backoff.
            timeout: Request timeout in seconds.

        Raises:
            ProviderError: If no API key is provided or found in the environment.
        """
        self._api_key = api_key or _find_env_key("OPENAI_API_KEY")
        if not self._api_key:
            raise ProviderError(
                "OPENAI_API_KEY is not set. Please set the OPENAI_API_KEY "
                "environment variable or pass api_key to OpenAIProvider.",
                provider_name="openai",
            )

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(0, max_retries)
        self.initial_backoff_sec = initial_backoff_sec
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return provider identifier name."""
        return "openai"

    def _send_request(self, payload: dict[str, Any]) -> str:
        """Send chat completion request to OpenAI endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data_bytes, headers=headers, method="POST"
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            response_body = response.read().decode("utf-8")
            data = json.loads(response_body)
            choices = data.get("choices", [])
            if not choices:
                raise ProviderError(
                    "OpenAI returned an empty choices list.",
                    provider_name="openai",
                )
            content = choices[0].get("message", {}).get("content", "")
            return str(content).strip()

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Generate text using configured OpenAI model with retry and backoff."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        last_error: Exception | None = None
        delay = self.initial_backoff_sec

        for attempt in range(1, self.max_retries + 2):
            try:
                return self._send_request(payload)
            except urllib.error.HTTPError as e:
                status_code = e.code
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8")
                except Exception:
                    pass
                sanitized_msg = _sanitize_message(
                    f"HTTP {status_code}: {error_body or e.reason}",
                    self._api_key,
                )
                last_error = e

                # Retry on rate limit (429) or server errors (500, 502, 503, 504)
                if (
                    status_code in (429, 500, 502, 503, 504)
                    and attempt <= self.max_retries
                ):
                    logger.warning(
                        "OpenAI attempt %d/%d failed (HTTP %d); retrying in %.2fs",
                        attempt,
                        self.max_retries + 1,
                        status_code,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.backoff_factor
                    continue

                raise ProviderError(
                    f"OpenAI API request failed: {sanitized_msg}",
                    provider_name="openai",
                    details={"status_code": status_code, "error": sanitized_msg},
                ) from e
            except Exception as e:
                sanitized_msg = _sanitize_message(str(e), self._api_key)
                last_error = e
                if attempt <= self.max_retries:
                    logger.warning(
                        "OpenAI API attempt %d/%d failed: %s; retrying in %.2fs",
                        attempt,
                        self.max_retries + 1,
                        sanitized_msg,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.backoff_factor
                    continue

                raise ProviderError(
                    f"OpenAI API request failed after {attempt} attempts: "
                    f"{sanitized_msg}",
                    provider_name="openai",
                    details={"error": sanitized_msg},
                ) from e

        raise ProviderError(
            f"OpenAI API request failed: {last_error}",
            provider_name="openai",
        )
