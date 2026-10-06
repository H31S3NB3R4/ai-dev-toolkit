"""Ollama LLM provider for local, zero-cost model evaluation."""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from ai_dev_toolkit.core.errors import ProviderError

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """LLM provider for locally hosted Ollama instances."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        max_retries: int = 2,
        initial_backoff_sec: float = 1.0,
        backoff_factor: float = 2.0,
        timeout: float = 60.0,
    ) -> None:
        """Initialize OllamaProvider.

        Args:
            model: Ollama model name (default: 'llama3').
            base_url: Ollama server base URL (default: 'http://localhost:11434').
            max_retries: Maximum retry attempts for transient connection failures.
            initial_backoff_sec: Initial backoff sleep duration in seconds.
            backoff_factor: Exponential multiplier for retry backoff.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(0, max_retries)
        self.initial_backoff_sec = initial_backoff_sec
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return provider identifier name."""
        return "ollama"

    def _send_request(self, payload: dict[str, Any]) -> str:
        """Send generation request to Ollama /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data_bytes, headers=headers, method="POST"
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            response_body = response.read().decode("utf-8")
            data = json.loads(response_body)
            response_text = data.get("response", "")
            return str(response_text).strip()

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Generate text from local Ollama model."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        last_error: Exception | None = None
        delay = self.initial_backoff_sec

        for attempt in range(1, self.max_retries + 2):
            try:
                return self._send_request(payload)
            except Exception as e:
                last_error = e
                if attempt <= self.max_retries:
                    logger.warning(
                        "Ollama request attempt %d/%d failed: %s; retrying in %.2fs",
                        attempt,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.backoff_factor
                    continue

                raise ProviderError(
                    f"Ollama request failed after {attempt} attempts: {e}. "
                    f"Ensure Ollama is running at {self.base_url}.",
                    provider_name="ollama",
                    details={
                        "base_url": self.base_url,
                        "model": self.model,
                        "error": str(e),
                    },
                ) from e

        raise ProviderError(
            f"Ollama request failed: {last_error}",
            provider_name="ollama",
        )
