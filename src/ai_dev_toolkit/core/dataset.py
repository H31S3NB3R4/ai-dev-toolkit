"""Dataset evaluation utilities and summary statistics aggregation."""

import csv
import io
import json
import statistics
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_dev_toolkit.core.config import EvaluatorConfig
from ai_dev_toolkit.core.evaluator import Evaluator
from ai_dev_toolkit.providers.base import LLMProvider


class MetricSummary(BaseModel):
    """Statistical summary for an individual metric across a dataset."""

    count: int = Field(description="Number of valid scores evaluated.")
    mean: float = Field(description="Arithmetic mean score [0.0, 1.0].")
    median: float = Field(description="Median score [0.0, 1.0].")
    min: float = Field(description="Minimum score [0.0, 1.0].")
    max: float = Field(description="Maximum score [0.0, 1.0].")


class DatasetEvaluationResult(BaseModel):
    """Aggregate result of evaluating an entire dataset."""

    total_samples: int = Field(description="Total number of evaluated samples.")
    summary: dict[str, MetricSummary] = Field(
        description="Statistical summary keyed by metric name."
    )
    results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Individual item evaluation records.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset result to dictionary format."""
        return self.model_dump()

    def to_json(self, indent: int | None = None) -> str:
        """Serialize dataset results to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self, filepath: str | Path) -> None:
        """Export dataset evaluation results to a CSV file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.results:
            out_path.write_text("", encoding="utf-8")
            return

        fieldnames = [
            "id",
            "prompt",
            "response",
            "context",
            "overall",
            "relevance",
            "completeness",
            "faithfulness",
            "hallucination",
        ]

        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in self.results:
                writer.writerow(r)


def _load_dataset_records(
    source: str | Path | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse dataset records from list, JSON, JSONL, or CSV."""
    if isinstance(source, list):
        return source

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset file '{source}' not found.")

    suffix = source_path.suffix.lower()
    content = source_path.read_text(encoding="utf-8")

    if suffix in (".jsonl",):
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
    elif suffix in (".json",):
        data = json.loads(content)
        if isinstance(data, list):
            return data
        raise ValueError(
            f"JSON dataset file '{source}' must contain a JSON array of records."
        )
    elif suffix in (".csv",):
        records = []
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            records.append(dict(row))
        return records
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}'. Supported: .jsonl, .json, .csv"
        )


def _calculate_metric_summary(scores: list[float]) -> MetricSummary:
    """Calculate summary statistics for a list of metric scores."""
    if not scores:
        return MetricSummary(count=0, mean=0.0, median=0.0, min=0.0, max=0.0)

    return MetricSummary(
        count=len(scores),
        mean=round(statistics.mean(scores), 4),
        median=round(statistics.median(scores), 4),
        min=round(min(scores), 4),
        max=round(max(scores), 4),
    )


def evaluate_dataset(
    dataset: str | Path | list[dict[str, Any]],
    provider: LLMProvider | None = None,
    config: EvaluatorConfig | None = None,
    max_workers: int = 4,
) -> DatasetEvaluationResult:
    """Evaluate a complete dataset of prompt/response samples.

    Args:
        dataset: Path to dataset file (.jsonl, .json, .csv) or list of sample dicts.
        provider: Optional LLMProvider judge.
        config: Optional EvaluatorConfig.
        max_workers: Concurrency limit for processing dataset items.

    Returns:
        DatasetEvaluationResult containing item results and summary statistics.
    """
    records = _load_dataset_records(dataset)
    if not records:
        return DatasetEvaluationResult(total_samples=0, summary={}, results=[])

    evaluator = Evaluator(provider=provider, config=config)

    evaluated_records: list[dict[str, Any]] = []
    relevance_scores: list[float] = []
    completeness_scores: list[float] = []
    faithfulness_scores: list[float] = []
    overall_scores: list[float] = []

    for idx, item in enumerate(records):
        item_id = item.get("id", f"sample_{idx + 1}")
        prompt = item.get("prompt", "")
        response = item.get("response", "")
        context = item.get("context")

        if not prompt or not response:
            continue

        result = evaluator.evaluate(prompt=prompt, response=response, context=context)

        relevance_scores.append(result.relevance)
        completeness_scores.append(result.completeness)
        overall_scores.append(result.overall)

        if result.faithfulness is not None:
            faithfulness_scores.append(result.faithfulness)

        record_dict = {
            "id": item_id,
            "prompt": prompt,
            "response": response,
            "context": context,
            "overall": result.overall,
            "relevance": result.relevance,
            "completeness": result.completeness,
            "faithfulness": result.faithfulness,
            "hallucination": result.hallucination,
            "reasoning": result.reasoning,
        }
        evaluated_records.append(record_dict)

    summary: dict[str, MetricSummary] = {
        "overall": _calculate_metric_summary(overall_scores),
        "relevance": _calculate_metric_summary(relevance_scores),
        "completeness": _calculate_metric_summary(completeness_scores),
    }

    if faithfulness_scores:
        summary["faithfulness"] = _calculate_metric_summary(faithfulness_scores)

    return DatasetEvaluationResult(
        total_samples=len(evaluated_records),
        summary=summary,
        results=evaluated_records,
    )
