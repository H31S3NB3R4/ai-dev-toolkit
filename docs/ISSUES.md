# Initial GitHub Issues and Contributor Tickets

This document contains pre-written, well-scoped issue templates for maintainers to create on GitHub with appropriate labels (`good first issue`, `help wanted`, `enhancement`, `provider`, `evaluation`).

---

### Issue 1: Add Ollama Provider for Local Offline Evaluation
- **Labels**: `provider`, `enhancement`, `good first issue`
- **Description**:
  Add an `OllamaProvider` implementation in `src/ai_dev_toolkit/providers/ollama.py` allowing developers to run LLM evaluations against local models (e.g. `llama3`, `mistral`, `gemma2`) with zero API costs.
- **Acceptance Criteria**:
  - Implements `LLMProvider` protocol (`name = "ollama"`, `generate(prompt, temperature=0.0)`).
  - Uses `httpx` or `urllib` to query `http://localhost:11434/api/generate`.
  - Configurable `base_url` and `model` (default `llama3:latest`).
  - Unit tests with mocked HTTP responses in `tests/test_providers.py`.

---

### Issue 2: Add OpenAI Provider Integration
- **Labels**: `provider`, `enhancement`, `help wanted`
- **Description**:
  Implement `OpenAIProvider` in `src/ai_dev_toolkit/providers/openai.py` supporting `gpt-4o-mini`, `gpt-4o`, and compatible OpenAI endpoints.
- **Acceptance Criteria**:
  - Reads `OPENAI_API_KEY` from environment or constructor.
  - Implements `LLMProvider` protocol.
  - Exponential backoff retry for rate limits (429) and timeouts.
  - Sanitizes API key in error messages.
  - Unit tests with mocked client.

---

### Issue 3: Add Response Length and Latency Evaluator
- **Labels**: `evaluation`, `enhancement`, `good first issue`
- **Description**:
  Create a rule-based `LengthMetric` that validates whether response token/character lengths meet specified minimum or maximum constraints without calling an LLM judge.
- **Acceptance Criteria**:
  - Implements `Metric` base class (`requires_context = False`).
  - Accepts `min_chars`, `max_chars`, `min_words`, `max_words`.
  - Returns score `1.0` if within bounds, proportional penalty if outside bounds.
  - Fast, zero-LLM execution.
  - Unit tests in `tests/test_metrics.py`.

---

### Issue 4: Add CSV and JSONL Dataset Exporter
- **Labels**: `cli`, `enhancement`, `good first issue`
- **Description**:
  Add utilities in `src/ai_dev_toolkit/core/export.py` and CLI options to export batch evaluation results into CSV and JSONL files for analytics and reporting.
- **Acceptance Criteria**:
  - `export_csv(results: list[EvaluationResult], filepath: Path)`
  - `export_jsonl(results: list[EvaluationResult], filepath: Path)`
  - Flattens scores, reasoning, and metadata fields into spreadsheet columns.
  - Unit tests in `tests/test_evaluator.py`.

---

### Issue 5: Support YAML / TOML Config File Loading
- **Labels**: `core`, `enhancement`, `help wanted`
- **Description**:
  Allow `EvaluatorConfig` to be loaded directly from `ai-dev.yaml` or `ai-dev.toml` files, configuring metric weights, default provider, timeouts, and thresholds.
- **Acceptance Criteria**:
  - `EvaluatorConfig.from_file(path: str | Path) -> EvaluatorConfig`.
  - CLI automatically detects `./ai-dev.yaml` or `./ai-dev.toml` if present.
  - Validates weights sum to > 0.
  - Unit tests for YAML and TOML parsing.

---

### Issue 6: Add Batch Dataset Evaluation CLI Command
- **Labels**: `cli`, `evaluation`, `help wanted`
- **Description**:
  Add `ai-dev batch --input dataset.jsonl --output results.jsonl` command to evaluate entire test datasets concurrently and print summary statistics (mean, p50, p95 score distribution).
- **Acceptance Criteria**:
  - Accepts JSONL file where each row contains `{"prompt": "...", "response": "...", "context": "..."}`.
  - Executes evaluation with worker pool.
  - Prints aggregate summary table with mean relevance, completeness, faithfulness.
  - Unit tests with Typer `CliRunner`.

---

### Issue 7: Add Windows & WSL Installation and Troubleshooting Guide
- **Labels**: `documentation`, `good first issue`
- **Description**:
  Add a dedicated section in `docs/` or `README.md` covering PowerShell execution policies, PATH configuration for Python Scripts, and virtual environment activation on Windows.
- **Acceptance Criteria**:
  - Clear steps for PowerShell `Set-ExecutionPolicy` guidance.
  - Command line tips for Windows Terminal and WSL2.
  - Verified by Windows developer.

---

### Issue 8: Add Interactive Jupyter Example Notebook
- **Labels**: `documentation`, `examples`, `good first issue`
- **Description**:
  Create `examples/evaluation_walkthrough.ipynb` walking developers through setting up API keys, running basic evaluations, evaluating RAG hallucination cases, and tuning custom weights.
- **Acceptance Criteria**:
  - Clean markdown explanations with runnable code cells.
  - Uses `FakeProvider` by default for zero-setup execution with toggle for live Gemini.
  - Demonstrates score plotting and table inspection.
