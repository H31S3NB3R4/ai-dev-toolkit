"""RAG evaluation example demonstrating faithfulness and hallucination detection."""

import os

from ai_dev_toolkit import EvaluatorConfig, evaluate
from ai_dev_toolkit.providers.fake import FakeProvider


def main() -> None:
    """Run RAG evaluation with context grounding check."""
    prompt = "What is the capital of Australia and what is its population?"
    context = (
        "Canberra is the capital city of Australia. Founded following the federation "
        "of the colonies of Australia as the seat of government, it has an estimated "
        "population of 456,000 residents."
    )
    # Intentionally accurate response
    response_faithful = (
        "The capital of Australia is Canberra, with an estimated population "
        "of approximately 456,000."
    )

    # Intentionally hallucinated response
    response_hallucinated = (
        "Sydney is the capital of Australia, with a massive population "
        "of over 5 million people."
    )

    print("=== AI Dev Toolkit: RAG & Faithfulness Evaluation ===\n")
    print(f"Context:\n  {context}\n")

    provider = None
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not found; running demo with FakeProvider...")
        provider = FakeProvider()

    custom_config = EvaluatorConfig(
        relevance_weight=0.30,
        completeness_weight=0.30,
        faithfulness_weight=0.40,
    )

    # 1. Evaluate faithful response
    print("\n1. Evaluating Faithful Response:")
    print(f"Response: {response_faithful}")
    res_1 = evaluate(
        prompt=prompt,
        response=response_faithful,
        context=context,
        provider=provider,
        config=custom_config,
    )
    print(f"-> Relevance:     {res_1.relevance * 100:.1f}%")
    print(f"-> Completeness:  {res_1.completeness * 100:.1f}%")
    if res_1.faithfulness is not None:
        print(f"-> Faithfulness:  {res_1.faithfulness * 100:.1f}%")
    if res_1.hallucination is not None:
        print(f"-> Hallucination: {res_1.hallucination * 100:.1f}%")
    print(f"-> Overall Score: {res_1.overall * 100:.1f}%")

    # 2. Evaluate hallucinated response
    print("\n2. Evaluating Hallucinated Response:")
    print(f"Response: {response_hallucinated}")
    res_2 = evaluate(
        prompt=prompt,
        response=response_hallucinated,
        context=context,
        provider=provider,
        config=custom_config,
    )
    print(f"-> Relevance:     {res_2.relevance * 100:.1f}%")
    print(f"-> Completeness:  {res_2.completeness * 100:.1f}%")
    if res_2.faithfulness is not None:
        print(f"-> Faithfulness:  {res_2.faithfulness * 100:.1f}%")
    if res_2.hallucination is not None:
        print(f"-> Hallucination: {res_2.hallucination * 100:.1f}%")
    print(f"-> Overall Score: {res_2.overall * 100:.1f}%")

    # Automated CI threshold check
    min_score_threshold = 0.75
    print(f"\n--- Quality Gate (Threshold >= {min_score_threshold * 100:.0f}%) ---")
    if res_1.overall >= min_score_threshold:
        print("[PASS] Faithful response meets quality threshold.")
    else:
        print("[FAIL] Faithful response below threshold.")


if __name__ == "__main__":
    main()
