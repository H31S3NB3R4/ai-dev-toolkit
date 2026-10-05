"""LLM Provider abstraction and implementations."""

from ai_dev_toolkit.core.errors import ProviderError
from ai_dev_toolkit.providers.base import LLMProvider
from ai_dev_toolkit.providers.fake import FakeProvider
from ai_dev_toolkit.providers.gemini import GeminiProvider


def get_provider(
    name: str = "gemini",
    **kwargs: object,
) -> LLMProvider:
    """Factory helper to obtain an LLMProvider instance by name.

    Args:
        name: Provider identifier (e.g. 'gemini', 'fake').
        **kwargs: Additional configuration arguments for provider initialization.

    Returns:
        An instance conforming to the LLMProvider protocol.

    Raises:
        ProviderError: If the provider name is unknown.
    """
    normalized = name.lower().strip()
    if normalized == "fake":
        return FakeProvider(**kwargs)  # type: ignore[arg-type]
    if normalized == "gemini":
        return GeminiProvider(**kwargs)  # type: ignore[arg-type]

    raise ProviderError(
        f"Unknown provider '{name}'. Supported providers: 'gemini', 'fake'",
        provider_name=name,
    )


__all__ = [
    "LLMProvider",
    "FakeProvider",
    "GeminiProvider",
    "get_provider",
]
