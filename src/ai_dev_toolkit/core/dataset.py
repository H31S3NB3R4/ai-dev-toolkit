"""Dataset evaluation utilities, batch execution, and summary statistics aggregation."""

import csv
import io
import json
import statistics
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_dev_toolkit.core.config import EvaluatorConfig
from ai_dev_toolkit.core.evaluator import Evaluator
from ai_dev_toolkit.metrics.registry import get_metric
from ai_dev_toolkit.providers.base import LLMProvider
from ai_dev_toolkit.reports.generator import (
    generate_html_report,
    generate_markdown_report,
)


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

        # Determine all unique keys across results
        keys_set: set[str] = set()
        for r in self.results:
            keys_set.update(r.keys())

        # Sort with standard keys first
        priority = [
            "id",
            "prompt",
            "response",
            "context",
            "ground_truth",
            "overall",
            "relevance",
            "completeness",
            "faithfulness",
            "hallucination",
            "context_relevance",
            "context_recall",
            "answer_relevance",
            "citation_correctness",
        ]
        fieldnames = [k for k in priority if k in keys_set] + sorted(
            keys_set - set(priority)
        )

        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in self.results:
                writer.writerow(r)

    def to_markdown(
        self,
        filepath: str | Path | None = None,
        title: str = "AI Dev Toolkit Benchmark Report",
        min_score_threshold: float = 0.7,
    ) -> str:
        """Generate and optionally save a Markdown benchmark report."""
        report = generate_markdown_report(
            self, title=title, min_score_threshold=min_score_threshold
        )
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(report, encoding="utf-8")
        return report

    def to_html(
        self,
        filepath: str | Path | None = None,
        title: str = "AI Dev Toolkit Benchmark Report",
        min_score_threshold: float = 0.7,
    ) -> str:
        """Generate and optionally save a self-contained HTML benchmark report."""
        report = generate_html_report(
            self, title=title, min_score_threshold=min_score_threshold
        )
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(report, encoding="utf-8")
        return report


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
    metrics: list[str] | None = None,
    max_workers: int = 4,
) -> DatasetEvaluationResult:
    """Evaluate a complete dataset of prompt/response samples.

    Args:
        dataset: Path to dataset file (.jsonl, .json, .csv) or list of
            sample dicts.
        provider: Optional LLMProvider judge.
        config: Optional EvaluatorConfig.
        metrics: Optional explicit list of metric names to score
            (e.g. ['relevance', 'context_recall']).
        max_workers: Concurrency limit for processing dataset items.

    Returns:
        DatasetEvaluationResult containing item results and summary statistics.
    """
    records = _load_dataset_records(dataset)
    if not records:
        return DatasetEvaluationResult(total_samples=0, summary={}, results=[])

    evaluator = Evaluator(provider=provider, config=config)
    evaluated_records: list[dict[str, Any]] = []

    # Map of metric_name -> list of float scores
    metric_score_accum: dict[str, list[float]] = {
        "overall": [],
        "relevance": [],
        "completeness": [],
        "faithfulness": [],
    }

    # Instantiate any extra custom/RAG metrics requested
    extra_metric_instances = {}
    if metrics:
        for m_name in metrics:
            if m_name not in ("relevance", "completeness", "faithfulness"):
                extra_metric_instances[m_name] = get_metric(m_name)
                if m_name not in metric_score_accum:
                    metric_score_accum[m_name] = []

    for idx, item in enumerate(records):
        item_id = item.get("id", f"sample_{idx + 1}")
        prompt = item.get("prompt", "")
        response = item.get("response", "")

        # Handle context as str or list of chunks
        raw_context = item.get("context") or item.get("contexts")
        if isinstance(raw_context, list):
            context = "\n\n".join(str(c) for c in raw_context)
        else:
            context = str(raw_context) if raw_context is not None else None

        ground_truth = item.get("ground_truth") or item.get("reference")

        if not prompt or not response:
            continue

        result = evaluator.evaluate(prompt=prompt, response=response, context=context)

        record_dict: dict[str, Any] = {
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

        if ground_truth:
            record_dict["ground_truth"] = ground_truth

        metric_score_accum["overall"].append(result.overall)
        metric_score_accum["relevance"].append(result.relevance)
        metric_score_accum["completeness"].append(result.completeness)

        if result.faithfulness is not None:
            metric_score_accum["faithfulness"].append(result.faithfulness)

        # Run extra metrics if specified
        for m_name, m_inst in extra_metric_instances.items():
            try:
                m_res = m_inst.score(
                    prompt=prompt,
                    response=response,
                    context=context,
                    provider=evaluator.provider,
                    ground_truth=ground_truth,
                )
                record_dict[m_name] = m_res.score
                metric_score_accum[m_name].append(m_res.score)
            except Exception:
                record_dict[m_name] = None

        evaluated_records.append(record_dict)

    summary: dict[str, MetricSummary] = {}
    for m_name, scores in metric_score_accum.items():
        if scores:
            summary[m_name] = _calculate_metric_summary(scores)

    return DatasetEvaluationResult(
        total_samples=len(evaluated_records),
        summary=summary,
        results=evaluated_records,
    )


def evaluate_batch(
    records: list[dict[str, Any]],
    provider: LLMProvider | None = None,
    config: EvaluatorConfig | None = None,
    metrics: list[str] | None = None,
    max_workers: int = 4,
) -> DatasetEvaluationResult:
    """Evaluate a batch list of samples in memory."""
    return evaluate_dataset(
        dataset=records,
        provider=provider,
        config=config,
        metrics=metrics,
        max_workers=max_workers,
    )
