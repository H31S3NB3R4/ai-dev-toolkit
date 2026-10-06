# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-10-07

### Added
- **New Providers**:
  - `OpenAIProvider`: Support for OpenAI Chat Completions API (`gpt-4o-mini`, `gpt-4o`, custom models) with exponential backoff retries and API key redaction.
  - `OllamaProvider`: Local, zero-cost LLM evaluation via Ollama HTTP API (`http://localhost:11434/api/generate`) with automatic retry support.
- **Custom Metric Registry**:
  - Global registry API: `register_metric`, `get_metric`, `list_metrics`, `reset_metrics`.
  - Support for registering custom metric instances and classes subclassing `Metric`.
- **Configuration File Loading**:
  - `EvaluatorConfig.from_file()` supporting `.toml`, `.yaml`, `.yml`, and `.json`.
  - `EvaluatorConfig.find_and_load()` automatic filesystem discovery for `.ai-dev.*` / `ai-dev.*` config files.
- **Dataset Evaluation**:
  - `evaluate_dataset()` supporting JSONL, JSON, and CSV input datasets.
  - `DatasetEvaluationResult` with aggregated statistics (`mean`, `median`, `min`, `max`) across samples and metrics.
  - `.to_csv()` dataset export utility for easy downstream reporting and CI/CD pipelines.
- **Async Evaluation**:
  - Public `aevaluate()` async API for seamless integration with modern asynchronous pipelines and async frameworks.

---

## [0.1.0] - 2026-10-07

### Added
- **Core Evaluation API**:
  - `evaluate()` 1-line / 3-line public function for evaluating LLM responses.
  - `Evaluator` orchestrator with concurrent metric scoring across thread pools.
  - `EvaluationResult` Pydantic model with scores, natural language reasoning, and metadata.
  - Standardized `.to_dict()` and `.to_json()` serialization methods.
  - Automatic weight renormalization when context is absent.
  - Input validation for prompt, response, and context.
- **Evaluation Metrics**:
  - `RelevanceMetric`: LLM-as-a-judge scoring of prompt relevance.
  - `CompletenessMetric`: Prompt sub-question decomposition and coverage scoring.
  - `FaithfulnessMetric`: Two-step claim extraction and factual verification against context.
  - Computed hallucination score (`1.0 - faithfulness`).
  - Versioned judge prompt templates with structured XML/delimiter wrapping and injection defense.
  - Strict JSON parser with automatic 1-retry fallback.
- **LLM Providers**:
  - `LLMProvider` protocol interface.
  - `GeminiProvider`: Integration with official `google-genai` SDK, exponential backoff retries, and API key sanitization.
  - `FakeProvider`: Deterministic offline provider for zero-cost testing and demos.
- **CLI (`ai-dev`)**:
  - `ai-dev evaluate` command with options for direct text, input files (`--prompt-file`, `--response-file`, `--context-file`), provider override, and sampling temperature.
  - Color-coded Rich terminal table output with optional reasoning display (`--reasoning`).
  - Machine-readable `--json` output flag.
  - Quality threshold gating with `--min-score` (exits with code 1 if below threshold).
  - Version flag `--version` / `-v`.
- **CI & Quality**:
  - GitHub Actions matrix workflow on Python 3.11, 3.12, and 3.13.
  - Enforced test coverage threshold of ≥ 80% (achieved 94%+).
  - Pre-commit configuration for Ruff and MyPy.
- **Examples & Documentation**:
  - `examples/basic_evaluation.py`: Basic prompt/response scoring example.
  - `examples/rag_evaluation.py`: RAG grounding and hallucination detection example.
  - Hand-labeled golden test set with 15 scenarios in `tests/data/golden_set.json`.
  - Comprehensive `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
