"""Basic evaluation example demonstrating prompt/response scoring."""

import os

from ai_dev_toolkit import evaluate
from ai_dev_toolkit.providers.fake import FakeProvider


def main() -> None:
    """Run basic evaluation on an LLM response."""
    prompt = "What are the primary differences between Python lists and tuples?"
    response = (
        "Lists in Python are mutable, meaning their elements can be modified\n"
        "after creation, and they use square brackets []. Tuples are immutable,\n"
        "meaning their elements cannot be altered, and they use parentheses ()."
    )

    print("=== AI Dev Toolkit: Basic Evaluation Example ===\n")
    print(f"Prompt:\n  {prompt}\n")
    print(f"Response:\n  {response}\n")

    # Use live GeminiProvider if GEMINI_API_KEY is present,
    # otherwise fallback to FakeProvider for offline demonstration.
    if os.getenv("GEMINI_API_KEY"):
        print("Evaluating using live GeminiProvider...")
        result = evaluate(prompt=prompt, response=response)
    else:
        print("GEMINI_API_KEY not found; running demo with FakeProvider...")
        fake_provider = FakeProvider()
        result = evaluate(prompt=prompt, response=response, provider=fake_provider)

    print("\n--- Evaluation Results ---")
    print(f"Relevance:    {result.relevance * 100:.1f}%")
    print(f"Completeness: {result.completeness * 100:.1f}%")
    print(f"Overall:      {result.overall * 100:.1f}%")

    if result.reasoning:
        print("\n--- Reasoning ---")
        for metric, reason in result.reasoning.items():
            print(f"[{metric.capitalize()}]: {reason}")

    print("\n--- JSON Output ---")
    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()
