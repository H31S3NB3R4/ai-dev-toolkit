"""Custom exceptions for the AI Dev Toolkit."""

from typing import Any


class AIDevToolkitError(Exception):
    """Base exception class for all errors raised by AI Dev Toolkit."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ProviderError(AIDevToolkitError):
    """Raised when an LLM provider encounters an error."""

    def __init__(
        self,
        message: str,
        provider_name: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.provider_name = provider_name
        self.status_code = status_code


class MetricError(AIDevToolkitError):
    """Raised when a metric computation or output parsing fails."""

    def __init__(
        self,
        message: str,
        metric_name: str | None = None,
        raw_output: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.metric_name = metric_name
        self.raw_output = raw_output
