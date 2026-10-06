"""Fake LLM provider for zero-cost offline testing and development."""

from collections.abc import Callable, Mapping, Sequence


def _default_fallback_response(prompt: str) -> str:
    if "extract all factual claims" in prompt.lower():
        return '{"claims": ["Mock claim."]}'
    if "verify whether each factual claim" in prompt.lower():
        return (
            '{"verdicts": [{"claim": "Mock claim.", "supported": true}], '
            '"score": 1.0, "reasoning": "Mock verified"}'
        )
    return (
        '{"score": 1.0, "sub_questions": ["Mock question"], "reasoning": "Mock score"}'
    )


class FakeProvider:
    """Mock LLM provider returning predetermined scripted responses."""

    def __init__(
        self,
        responses: (
            str | Sequence[str] | Mapping[str, str] | Callable[..., str] | None
        ) = None,
        default_response: str | None = None,
        error_to_raise: Exception | None = None,
    ) -> None:
        """Initialize FakeProvider with scripted responses or errors.

        Args:
            responses: A single string, a list of sequential response strings,
                or a mapping of substrings to matching responses.
            default_response: Fallback response if no match or list is exhausted.
            error_to_raise: Optional exception to raise when generate is called.
        """
        self._responses = responses
        self._default_response = default_response
        self._error_to_raise = error_to_raise
        self.prompts: list[str] = []
        self.call_count: int = 0

    @property
    def name(self) -> str:
        """Return provider identifier name."""
        return "fake"

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Return a scripted response recorded for this call."""
        self.prompts.append(prompt)
        self.call_count += 1

        if self._error_to_raise is not None:
            raise self._error_to_raise

        if self._responses is None:
            if self._default_response is not None:
                return self._default_response
            return _default_fallback_response(prompt)

        if callable(self._responses):
            try:
                return self._responses(prompt, temperature=temperature)
            except TypeError:
                try:
                    return self._responses(prompt, temperature)
                except TypeError:
                    return self._responses(prompt)

        if isinstance(self._responses, str):
            return self._responses

        if isinstance(self._responses, Mapping):
            for key, val in self._responses.items():
                if key in prompt:
                    return val
            return self._default_response

        if isinstance(self._responses, Sequence):
            idx = self.call_count - 1
            if idx < len(self._responses):
                return self._responses[idx]
            return self._default_response

        return self._default_response

    def __repr__(self) -> str:
        """Return representation without leaking sensitive data."""
        return f"FakeProvider(call_count={self.call_count})"
