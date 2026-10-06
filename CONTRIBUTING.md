# Contributing to AI Dev Toolkit

Thank you for your interest in contributing to **AI Dev Toolkit**! We welcome community contributions of all kinds, including bug fixes, new metrics, provider integrations, documentation enhancements, and feature requests.

Please take a moment to review this guide before opening a Pull Request.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and constructive in all discussions and interactions.

---

## Development Setup

### 1. Prerequisites

- **Python**: Version 3.11, 3.12, or 3.13
- **Git**

### 2. Clone and Setup Virtual Environment

```bash
git clone https://github.com/H31S3NB3R4/ai-dev-toolkit.git
cd ai-dev-toolkit

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies in Editable Mode

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### 4. (Optional) Set up Pre-commit Hooks

```bash
pre-commit install
```

---

## Running Quality Checks and Tests

All tests and quality checks run **100% offline** by default using `FakeProvider`. No paid API keys or active internet connections are required for local testing.

### Run Unit and Integration Tests

```bash
python -m pytest
```

*(Pytest enforces a minimum test coverage threshold of 80% automatically.)*

### Run Linter and Formatting Checks

```bash
# Check code for lint errors
python -m ruff check .

# Fix auto-fixable lint issues
python -m ruff check . --fix

# Check formatting
python -m ruff format --check .

# Format code
python -m ruff format .
```

### Run Type Checker

```bash
python -m mypy src tests
```

---

## Adding a New Metric

Follow these steps to add a new evaluation metric:

1. **Create Prompt Template**:
   Add a versioned prompt template in `src/ai_dev_toolkit/metrics/prompts/templates.py`. Ensure untrusted inputs (prompt, response, context) are wrapped in clear delimiters (`<<<PROMPT_START>>>`, etc.) with prompt-injection defense instructions.
2. **Implement Metric Class**:
   Create a new module in `src/ai_dev_toolkit/metrics/<metric_name>.py` inheriting from `ai_dev_toolkit.metrics.base.Metric`:
   ```python
   from ai_dev_toolkit.metrics.base import Metric
   from ai_dev_toolkit.core.result import MetricResult
   from ai_dev_toolkit.providers.base import LLMProvider


   class CustomMetric(Metric):
       @property
       def name(self) -> str:
           return "custom_metric"

       @property
       def requires_context(self) -> bool:
           return False

       def score(
           self,
           prompt: str,
           response: str,
           context: str | None,
           provider: LLMProvider,
           *,
           temperature: float = 0.0,
           include_reasoning: bool = True,
       ) -> MetricResult:
           # Call judge and parse structured output
           ...
           return MetricResult(score=score, reasoning=reasoning)
   ```
3. **Export the Metric**:
   Export your metric class in `src/ai_dev_toolkit/metrics/__init__.py`.
4. **Add Unit Tests**:
   Add unit tests in `tests/test_metrics.py` verifying parsing, retries, edge cases, and prompt injection defense.

---

## Adding a New Provider

To integrate a new LLM provider (e.g. OpenAI, Ollama, Anthropic):

1. **Implement Provider Class**:
   Create `src/ai_dev_toolkit/providers/<provider_name>.py` conforming to the `LLMProvider` protocol defined in `src/ai_dev_toolkit/providers/base.py`:
   ```python
   class CustomProvider:
       @property
       def name(self) -> str:
           return "custom_provider"

       def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
           # Execute LLM call with retry and backoff
           ...
   ```
2. **Handle Errors & Security**:
   - Wrap provider-specific exceptions in `ai_dev_toolkit.core.errors.ProviderError`.
   - Ensure API keys and secret tokens are sanitized and never leaked in logs or error messages.
3. **Register Provider**:
   Export the class in `src/ai_dev_toolkit/providers/__init__.py` and add support in `src/ai_dev_toolkit/core/evaluator.py`.
4. **Add Offline Tests**:
   Create mock/unit tests in `tests/test_providers.py`.

---

## Pull Request Process

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Commit your changes with clear, descriptive commit messages following the [Conventional Commits](https://www.conventionalcommits.org/) convention (e.g., `feat:`, `fix:`, `docs:`, `test:`).
3. Ensure all quality checks pass locally:
   ```bash
   python -m ruff check .
   python -m ruff format --check .
   python -m pytest
   ```
4. Push your branch to GitHub and open a Pull Request against `main`.
5. Fill out the PR template completely. A maintainer will review your PR within 48 hours.

Thank you for helping make AI Dev Toolkit better for everyone!
