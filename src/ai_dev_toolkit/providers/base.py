"""Base interface and protocols for LLM providers."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for all LLM evaluation providers."""

    @property
    def name(self) -> str:
        """Return the unique identifier for the provider (e.g., 'gemini', 'fake')."""
        ...

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Generate response text for the provided prompt.

        Args:
            prompt: Text prompt to submit to the model.
            temperature: Sampling temperature (0.0 for deterministic output).

        Returns:
            The raw text response from the model.

        Raises:
            ProviderError: If the provider encounters an error during generation.
        """
        ...
