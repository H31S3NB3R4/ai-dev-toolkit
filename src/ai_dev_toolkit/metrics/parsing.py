"""Strict JSON parsing utilities for judge output."""

import json
import logging
import re
from typing import Any

from ai_dev_toolkit.core.errors import MetricError

logger = logging.getLogger(__name__)


def extract_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from raw LLM output.

    Tries to find a JSON block in the output, handling markdown
    code fences and surrounding prose.

    Args:
        raw: Raw text output from the judge LLM.

    Returns:
        Parsed dictionary from the JSON content.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Try direct parse first
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON object braces
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(cleaned[start : end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from output: {raw[:200]}")


def parse_judge_output(
    raw: str,
    metric_name: str,
    required_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Parse and validate JSON output from a judge LLM.

    Args:
        raw: Raw text output from the judge.
        metric_name: Name of the metric for error reporting.
        required_keys: Optional list of keys that must be present.

    Returns:
        Validated dictionary with judge output.

    Raises:
        MetricError: If parsing fails or required keys are missing.
    """
    try:
        data = extract_json(raw)
    except ValueError as e:
        raise MetricError(
            f"Failed to parse judge output for '{metric_name}': {e}",
            metric_name=metric_name,
            raw_output=raw,
        ) from e

    if required_keys:
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise MetricError(
                f"Judge output for '{metric_name}' is missing required keys: {missing}",
                metric_name=metric_name,
                raw_output=raw,
            )

    return data


def judge_with_retry(
    provider: Any,
    prompt: str,
    metric_name: str,
    required_keys: list[str] | None = None,
    *,
    temperature: float = 0.0,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Call provider and parse JSON output, retrying once on parse failure.

    Args:
        provider: LLM provider to use for generation.
        prompt: The judge prompt to send.
        metric_name: Name of the metric for error reporting.
        required_keys: Keys that must be present in the JSON output.
        temperature: Sampling temperature for generation.
        max_retries: Number of retry attempts on parse failure.

    Returns:
        Parsed and validated dictionary from the judge output.

    Raises:
        MetricError: If all attempts fail.
    """
    last_error: MetricError | None = None

    for attempt in range(max_retries + 1):
        raw = provider.generate(prompt, temperature=temperature)
        try:
            return parse_judge_output(raw, metric_name, required_keys)
        except MetricError as e:
            last_error = e
            logger.warning(
                "Judge parse attempt %d/%d failed for '%s': %s",
                attempt + 1,
                max_retries + 1,
                metric_name,
                e.message,
            )

    assert last_error is not None
    raise last_error
