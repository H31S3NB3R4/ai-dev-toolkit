# TODO: AI Dev Toolkit (phase by phase)

Rule: finish each phase's **Exit check** before starting the next. Don't touch dashboard/React until v0.5.

---

## Phase 0: Decisions and accounts
- [ ] Confirm repo name `ai-dev-toolkit` and CLI name `ai-dev`
- [ ] Check PyPI name availability (fallback: `ai-dev-toolkit-eval`)
- [ ] Get a Gemini API key (store only in env var / `.env`)
- [ ] Install Python 3.11+, Git, and `uv` or `pip` + `venv`
- [ ] Decide open questions in PRD §14 (default model, async, reasoning default)

**Exit check:** `python --version` ≥ 3.11, `git --version` works, API key set locally.

---

## Phase 1: Repo and scaffolding (Issue #1)
- [x] Create GitHub repo `H31S3NB3R4/ai-dev-toolkit` (public, MIT license)
- [x] Clone locally, create virtual environment
- [x] Create `src/ai_dev_toolkit/` with `__init__.py` and `py.typed`
- [x] Create folders: `core/`, `metrics/`, `providers/`, `cli/`, `tests/`, `examples/`, `docs/`
- [x] Write `pyproject.toml` (Hatchling, metadata, deps, `ai-dev` script entry, ruff + mypy + pytest config)
- [x] Add `.gitignore` (include `.env`, `.venv`, `dist/`, `__pycache__/`)
- [x] Stub README with tagline and "work in progress" note
- [x] `pip install -e ".[dev]"` works
- [x] First commit and push

**Exit check:** `python -c "import ai_dev_toolkit"` succeeds; `ruff check .` and `pytest` run (even with zero tests).

---

## Phase 2: Core models and errors (Issue #2)
- [x] `core/errors.py`: `AIDevToolkitError`, `ProviderError`, `MetricError`
- [x] `core/config.py`: `EvaluatorConfig` (weights, temperature, timeout, retries)
- [x] `core/result.py`: `MetricResult` and `EvaluationResult` (Pydantic)
- [x] Clamp scores to `[0, 1]`; `hallucination` computed as `1 - faithfulness`
- [x] `to_dict()` and `to_json()`
- [x] Overall-score function with weight renormalization when faithfulness is `None`
- [x] Tests: validation, clamping, JSON round trip, weight renormalization

**Exit check:** all core tests pass; mypy clean on `core/`.

---

## Phase 3: Provider layer (Issues #6, #7)
- [x] `providers/base.py`: `LLMProvider` Protocol (`name`, `generate`)
- [x] `providers/fake.py`: `FakeProvider` returning scripted responses
- [x] `providers/gemini.py`: read `GEMINI_API_KEY`, call model, wrap errors in `ProviderError`
- [x] Retry with backoff (max 3) for rate limits/timeouts
- [x] Ensure the key never appears in logs or exceptions
- [x] Tests (offline): FakeProvider behavior, error wrapping with mocked Gemini client
- [x] One manual smoke test against real Gemini (not in CI)

**Exit check:** `GeminiProvider().generate("Say hi")` returns text; core code has zero Gemini imports.

---

## Phase 4: Metrics (Issues #3, #4, #5)
- [x] `metrics/base.py`: `Metric` base (`name`, `requires_context`, `score()`)
- [x] Judge prompt templates in `metrics/prompts/` (versioned, delimiter-wrapped untrusted input)
- [x] Strict JSON parser with one retry, then `MetricError`
- [x] **Relevance** metric + tests
- [x] **Completeness** metric (sub-question decomposition) + tests
- [x] **Faithfulness** metric (claim extraction + verification) + tests
- [x] Prompt-injection test cases in the response text
- [x] Build a small golden set (10-20 hand-labeled examples) in `tests/data/`
- [x] Sanity-check judge scores against the golden set; tune prompts

**Exit check:** all metric tests pass offline with `FakeProvider`; golden set scores look reasonable on real Gemini.

---

## Phase 5: Evaluator and public API
- [x] `core/evaluator.py`: runs the metrics, aggregates scores, records latency and provider metadata
- [x] Run metrics concurrently (thread pool) when more than one is enabled
- [x] `evaluate()` convenience function exported from `__init__.py`
- [x] Input validation (empty prompt/response raises `ValueError`)
- [x] Skip faithfulness cleanly when no context is given
- [x] Tests: end-to-end with `FakeProvider`, with and without context, config override

**Exit check:** the 3-line quickstart works end to end.

---

## Phase 6: CLI (Issue #8)
- [x] `cli/main.py` with Typer: `ai-dev evaluate`
- [x] Options: `--prompt`, `--response`, `--context`, `--provider`, `--json`, `--min-score`
- [x] Rich table output matching the PRD example
- [x] Exit code 1 when below `--min-score`; clear error messages for missing API key
- [x] `--version`
- [x] (P2) `--prompt-file`, `--response-file`, `--context-file`
- [x] Tests with Typer's `CliRunner`

**Exit check:** `ai-dev evaluate --prompt "..." --response "..." --json` prints valid JSON.

---

## Phase 7: Quality, CI, and examples (Issues #9, #10)
- [x] `.github/workflows/tests.yml`: ruff, mypy, pytest + coverage on Python 3.11, 3.12, 3.13
- [x] Add coverage threshold (≥ 80%)
- [x] `examples/basic_evaluation.py`
- [x] `examples/rag_evaluation.py`
- [x] Docstrings on all public classes and functions
- [x] Optional example notebook
- [x] Pre-commit config (ruff, mypy)

**Exit check:** CI green on `main`; examples run with a fresh venv.

---

## Phase 8: Community files
- [x] Full `README.md`: what/why, install, quickstart, CLI usage, JSON schema, limitations of LLM-as-judge, roadmap, badges
- [x] `CONTRIBUTING.md`: dev setup, running tests, how to add a metric, how to add a provider, PR process
- [x] `CODE_OF_CONDUCT.md` (Contributor Covenant)
- [x] `CHANGELOG.md` (Keep a Changelog format)
- [x] `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`
- [x] `.github/pull_request_template.md`
- [x] Create the labels from PRD §13
- [x] Enable GitHub Discussions
- [x] Add `SECURITY.md` (how to report vulnerabilities)

**Exit check:** a stranger can clone, set up, and run tests by following only CONTRIBUTING.md.

---

## Phase 9: Release v0.1.0
- [x] Final pass on the PRD §12 Definition of Done checklist
- [x] Bump version to `0.1.0`, update CHANGELOG
- [x] Build: `python -m build`; check with `twine check dist/*`
- [x] Publish to TestPyPI first; install in a clean venv and verify
- [x] Publish to PyPI (consider Trusted Publishing via GitHub Actions)
- [x] Tag `v0.1.0` and create a GitHub Release with notes
- [x] Verify `pip install ai-dev-toolkit` then run the quickstart

**Exit check:** a clean machine can install from PyPI and get a score.

---

## Phase 10: Open issues and attract contributors
- [ ] Create 8+ well-scoped `good first issue` tickets, for example:
  - [ ] Add response-length evaluator
  - [ ] Add JSON/CSV exporter
  - [ ] Add Ollama provider
  - [ ] Add OpenAI provider
  - [ ] More unit tests for relevance
  - [ ] Add `--prompt-file` CLI options
  - [ ] Improve docs / add example notebook
  - [ ] Add Windows install notes
- [ ] Add `help wanted` to larger items (config file, dataset evaluation)
- [ ] Pin a Roadmap issue or Discussion
- [ ] Share on relevant communities (r/LocalLLaMA, r/MachineLearning showcase threads where allowed, Dev.to, LinkedIn)
- [ ] Respond to issues/PRs within 48 hours; be kind and specific in reviews

**Exit check:** first external issue or PR received.

---

## Phase 11: v0.2.0 (providers and config)
- [ ] OpenAI provider
- [ ] Ollama provider (local, free)
- [ ] Custom metric registration API
- [ ] Config file support (YAML/TOML)
- [ ] Dataset evaluation (JSONL input, summary stats)
- [ ] Async `aevaluate()` if desired

## Phase 12: v0.3.0 (RAG evaluation)
- [ ] Context relevance, context recall, answer relevance
- [ ] Citation correctness
- [ ] Batch evaluation, CSV/JSON datasets
- [ ] Benchmark reports (Markdown/HTML)

## Phase 13: v0.4.0 (agent evaluation)
- [ ] Trace data model (steps, tool calls, failures)
- [ ] Tool-call correctness metric
- [ ] Latency/token/cost tracking
- [ ] Execution reports

## Phase 14: v0.5.0 (dashboard)
- [ ] FastAPI backend serving stored results
- [ ] React dashboard (scores, runs, model comparison)
- [ ] Experiment tracking

## Phase 15: v1.0.0
- [ ] Stable public API and deprecation policy
- [ ] Plugin system for metrics/providers
- [ ] Docs site (MkDocs)

---

## Weekly rhythm suggestion
| Week | Phases |
|---|---|
| 1 | 0, 1, 2 |
| 2 | 3, 4 |
| 3 | 5, 6 |
| 4 | 7, 8 |
| 5 | 9, 10 |
