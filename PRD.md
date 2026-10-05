# PRD: AI Dev Toolkit

| | |
|---|---|
| **Repository** | `H31S3NB3R4/ai-dev-toolkit` |
| **PyPI package** | `ai-dev-toolkit` |
| **Import** | `from ai_dev_toolkit import evaluate` |
| **CLI** | `ai-dev` |
| **Tagline** | Open-source evaluation and testing toolkit for LLM, RAG, and AI agents |
| **License** | MIT |
| **Status** | Draft v1.0 |
| **Owner** | H31S3NB3R4 |
| **Date** | 2026-10-05 |

---

## 1. Summary

AI Dev Toolkit is a modular, open-source Python library that helps developers **evaluate, test, and debug** AI applications. v0.1.0 starts with one narrow, finishable problem: scoring an LLM response against a prompt (and optional context) on relevance, completeness, and faithfulness, then returning a structured, machine-readable result.

Long-term, it grows into evaluation for RAG pipelines and agents, prompt/model comparison, and a dashboard.

## 2. Problem Statement

Developers building LLM apps usually "eyeball" outputs. This causes:

- No repeatable way to tell if a prompt or model change made things better or worse.
- Hallucinations and incomplete answers slip into production.
- Existing eval frameworks can be heavy, opinionated, or tied to one ecosystem.
- Results are not easily consumable by CI/CD pipelines.

## 3. Goals and Non-Goals

### 3.1 Goals (v0.1.0)

1. One-call evaluation: `evaluate(prompt, response, context=None)`.
2. Three metrics: **relevance**, **completeness**, **faithfulness** (hallucination = 1 − faithfulness).
3. Provider-agnostic design with Gemini as the first provider.
4. Structured output (`to_dict()`, `to_json()`) suitable for CI.
5. A basic CLI (`ai-dev evaluate`).
6. Contributor-ready repo: docs, CI, tests, issue templates, labeled issues.

### 3.2 Non-Goals (v0.1.0)

Dashboard/React UI, auth, database, Docker/K8s, multi-agent systems, more than one provider, LangChain integration, vector DBs, SaaS, user accounts, cloud deployment.

### 3.3 Success Metrics

| Metric | Target for v0.1.0 | 3-month target |
|---|---|---|
| Installable from PyPI | Yes | Yes |
| Test coverage (core + metrics) | ≥ 80% | ≥ 85% |
| CI passing on `main` | 100% | 100% |
| Time from `pip install` to first score | < 5 minutes | < 3 minutes |
| GitHub stars | n/a | 50+ |
| External contributors (merged PR) | n/a | 3+ |
| Good-first-issues open | 8+ | always ≥ 5 |

## 4. Target Users

| Persona | Needs |
|---|---|
| **Indie / student AI developer** | Quick, free-tier-friendly way to check answer quality. |
| **RAG builder** | Check whether answers are grounded in retrieved context. |
| **Backend / ML engineer** | JSON output and exit codes to gate CI pipelines. |
| **Open-source contributor** | Clear structure to add a metric or provider in a small PR. |

## 5. User Stories

1. As a developer, I can score a single prompt/response pair in 3 lines of Python.
2. As a RAG developer, I can pass retrieved context and see if the response is faithful to it.
3. As a CI engineer, I can run a CLI command and get JSON plus a non-zero exit code if the score is below a threshold.
4. As a contributor, I can add a new provider by implementing one interface and one test file.
5. As a maintainer, I can change score weights without editing metric code.

## 6. Functional Requirements

### 6.1 Public API

```python
from ai_dev_toolkit import evaluate

result = evaluate(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    context="France is a country in Europe. Paris is its capital.",  # optional
)

result.relevance       # float 0.0–1.0
result.completeness    # float 0.0–1.0
result.faithfulness    # float 0.0–1.0 or None if no context
result.hallucination   # 1 - faithfulness, or None
result.overall         # float 0.0–1.0
result.to_dict()
result.to_json()
```

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | `evaluate()` accepts `prompt`, `response`, optional `context`, optional `provider`, optional `config`. | P0 |
| FR-2 | Returns an `EvaluationResult` (Pydantic model). | P0 |
| FR-3 | Scores are floats clamped to `[0.0, 1.0]`. | P0 |
| FR-4 | If `context` is absent, faithfulness is `None` and is excluded from overall (weights renormalized). | P0 |
| FR-5 | Input validation: empty prompt/response raises `ValueError` with a clear message. | P0 |
| FR-6 | `to_dict()` and `to_json()` produce stable, documented keys. | P0 |
| FR-7 | Optional `reasoning` string per metric to explain the score. | P1 |

### 6.2 Metrics

| Metric | Inputs | Question answered | Method (v0.1) |
|---|---|---|---|
| **Relevance** | prompt, response | Does the response address the prompt? | LLM-as-judge with a strict rubric, JSON output |
| **Completeness** | prompt, response | Are all parts of the prompt answered? | LLM-as-judge: decompose prompt into sub-questions, check each |
| **Faithfulness** | prompt, context, response | Is every claim supported by the context? | LLM-as-judge: extract claims, verify each against context |

**Overall score**

```
overall = 0.35 * relevance + 0.25 * completeness + 0.40 * faithfulness
```

If faithfulness is `None`, renormalize: `relevance 0.35/0.60`, `completeness 0.25/0.60`. Weights are configurable through `EvaluatorConfig`.

**Design notes**

- Judge prompts live in versioned template files, not inline strings, so contributors can improve them.
- Judge output must be strict JSON; parser retries once on malformed output, then raises `MetricError`.
- Temperature defaults to 0 for reproducibility.
- Each metric implements a common `Metric` base class: `name`, `requires_context`, `score(...) -> MetricResult`.

### 6.3 Provider Abstraction

```python
class LLMProvider(Protocol):
    name: str
    def generate(self, prompt: str, *, temperature: float = 0.0) -> str: ...
```

| ID | Requirement | Priority |
|---|---|---|
| FR-8 | No provider-specific code in `core/` or `metrics/`. | P0 |
| FR-9 | `GeminiProvider` reads `GEMINI_API_KEY` from the environment; never accepts it from logs/commits. | P0 |
| FR-10 | Provider errors are wrapped in `ProviderError` (auth, rate limit, timeout). | P0 |
| FR-11 | Basic retry with backoff for transient errors (max 3). | P1 |
| FR-12 | A `FakeProvider` for tests and offline demos. | P0 |

### 6.4 CLI

```
ai-dev evaluate --prompt "Explain RAG" --response "RAG combines retrieval and generation" \
                [--context "..."] [--provider gemini] [--json] [--min-score 0.7]
```

| ID | Requirement | Priority |
|---|---|---|
| FR-13 | Human-readable table output by default; `--json` for machine output. | P0 |
| FR-14 | `--min-score` returns exit code 1 if overall is below threshold. | P1 |
| FR-15 | Accept `--prompt-file`, `--response-file`, `--context-file`. | P2 |
| FR-16 | `ai-dev --version`. | P0 |

Example output:

```
AI Dev Toolkit
────────────────────
Relevance       92%
Completeness    86%
Faithfulness    94%

Overall         91%
```

### 6.5 Configuration

`EvaluatorConfig` (Pydantic): metric weights, provider, temperature, timeout, retries. Environment variable overrides: `GEMINI_API_KEY`, `AI_DEV_PROVIDER`.

## 7. Non-Functional Requirements

| Area | Requirement |
|---|---|
| **Compatibility** | Python 3.11+; Windows, macOS, Linux. |
| **Performance** | Library overhead < 50 ms excluding LLM latency. Metrics run concurrently where possible. |
| **Reliability** | Deterministic (temperature 0); graceful failure messages. |
| **Security** | No keys in code, logs, or error output. `.env` in `.gitignore`. Inputs are never executed. Judge prompts guard against prompt injection in `response` by delimiting untrusted text. |
| **Privacy** | No telemetry. Data only goes to the configured provider. |
| **Quality** | Ruff clean, mypy strict on `src/`, pytest. All tests must run without network (use `FakeProvider`). |
| **Docs** | README quickstart, docstrings on all public APIs, 2 runnable examples. |
| **Packaging** | `pyproject.toml` with Hatchling, src layout, semantic versioning, `py.typed`. |

## 8. Architecture

```
                    evaluate()
                        │
                        ↓
                   Evaluator ──── EvaluatorConfig
                        │
        ┌───────────────┼────────────────┐
        ↓               ↓                ↓
    Relevance      Completeness     Faithfulness      (metrics/)
        │               │                │
        └───────────────┼────────────────┘
                        ↓
                  LLMProvider (interface)
                        │
              ┌─────────┼─────────┐
              ↓         ↓         ↓
           Gemini     OpenAI*   Ollama*        (* later)
                        │
                        ↓
                 EvaluationResult → dict / JSON / CLI table
```

### Repository layout

```
ai-dev-toolkit/
├── src/ai_dev_toolkit/
│   ├── __init__.py            # exports evaluate, EvaluationResult
│   ├── py.typed
│   ├── core/
│   │   ├── evaluator.py
│   │   ├── result.py
│   │   ├── config.py
│   │   └── errors.py
│   ├── metrics/
│   │   ├── base.py
│   │   ├── relevance.py
│   │   ├── completeness.py
│   │   ├── faithfulness.py
│   │   └── prompts/           # versioned judge templates
│   ├── providers/
│   │   ├── base.py
│   │   ├── gemini.py
│   │   └── fake.py
│   └── cli/
│       └── main.py
├── tests/
├── examples/
│   ├── basic_evaluation.py
│   └── rag_evaluation.py
├── docs/
├── .github/
│   ├── workflows/tests.yml
│   ├── ISSUE_TEMPLATE/{bug_report.md,feature_request.md}
│   └── pull_request_template.md
├── CONTRIBUTING.md  CODE_OF_CONDUCT.md  LICENSE  README.md  CHANGELOG.md
├── pyproject.toml
└── .gitignore
```

### Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.11+ |
| Models/validation | Pydantic v2 |
| CLI | Typer (+ Rich for tables) |
| Build | Hatchling |
| Tests | pytest, pytest-cov |
| Lint/format | Ruff |
| Types | mypy (strict) |
| LLM (first) | Gemini via official Google SDK |
| CI | GitHub Actions (matrix 3.11, 3.12, 3.13) |

## 9. Data Contracts

### EvaluationResult JSON

```json
{
  "relevance": 0.92,
  "completeness": 0.86,
  "faithfulness": 0.94,
  "hallucination": 0.06,
  "overall": 0.91,
  "reasoning": {
    "relevance": "Directly answers the question.",
    "completeness": "Covers definition; omits advantages.",
    "faithfulness": "All claims supported by context."
  },
  "metadata": {
    "provider": "gemini",
    "model": "<model-id>",
    "toolkit_version": "0.1.0",
    "latency_ms": 1840
  }
}
```

This schema is part of the public API. Changes after v0.1.0 follow semantic versioning.

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM-as-judge scores are noisy/biased | Low trust | Temperature 0, strict rubrics, claim-level verification, document limitations, add golden test set |
| Judge is fooled by prompt injection in response | Wrong scores | Delimit untrusted text, instruct judge to ignore embedded instructions, add injection tests |
| Gemini API changes or free-tier limits | Breakage | Provider interface, `FakeProvider` in tests, pin SDK range |
| Over-scoping | Never ships | Hard non-goals list; v0.1.0 definition of done |
| Contributors don't appear | Stagnation | Good-first-issues, clear CONTRIBUTING, quick PR reviews, share in communities |
| Name collision on PyPI | Blocked publish | Check name availability early (Phase 0); fall back to `ai-dev-toolkit-eval` |
| API cost for users | Adoption | Efficient prompts, one call per metric, support local models (Ollama) in v0.2 |

## 11. Roadmap

| Version | Theme | Highlights |
|---|---|---|
| **v0.1.0** | MVP | evaluate(), 3 metrics, Gemini, CLI, CI, docs |
| **v0.2.0** | Providers + config | OpenAI, Ollama, custom metrics, config file (YAML/TOML), dataset evaluation |
| **v0.3.0** | RAG | Context relevance/recall, citation correctness, batch eval, CSV/JSON datasets, reports |
| **v0.4.0** | Agents | Tool-call evaluation, tracing, execution reports |
| **v0.5.0** | Dashboard | React dashboard, experiment tracking, model comparison |
| **v1.0.0** | Stable | Stable API, plugin system, full docs site |

## 12. v0.1.0 Definition of Done

- [ ] Repo public with MIT license, README, CONTRIBUTING, CODE_OF_CONDUCT, CHANGELOG
- [ ] `EvaluationResult` with `to_dict()` / `to_json()`
- [ ] Relevance, completeness, faithfulness metrics with tests
- [ ] Provider interface, `GeminiProvider`, `FakeProvider`
- [ ] CLI `ai-dev evaluate` working with `--json` and `--min-score`
- [ ] CI green (pytest, ruff, mypy) on Python 3.11–3.13
- [ ] Coverage ≥ 80%
- [ ] 2 runnable examples
- [ ] Published to PyPI; `pip install ai-dev-toolkit` works in a clean venv
- [ ] 8+ labeled good-first-issues open
- [ ] Tagged release `v0.1.0` with release notes

## 13. Initial GitHub Issues

| # | Title | Labels |
|---|---|---|
| 1 | Set up initial Python package structure | `enhancement`, `good first issue` |
| 2 | Create `EvaluationResult` model | `core`, `good first issue` |
| 3 | Implement relevance metric | `evaluation`, `testing` |
| 4 | Implement completeness metric | `evaluation`, `testing` |
| 5 | Implement faithfulness metric | `evaluation`, `testing` |
| 6 | Create provider abstraction | `provider`, `core` |
| 7 | Add Gemini provider | `provider` |
| 8 | Add initial CLI | `cli` |
| 9 | Add CI for pytest, ruff, mypy | `enhancement` |
| 10 | Add beginner-friendly usage examples | `documentation`, `good first issue` |

**Labels:** `good first issue`, `help wanted`, `bug`, `enhancement`, `documentation`, `testing`, `core`, `cli`, `provider`, `evaluation`

## 14. Open Questions

1. Which Gemini model should be the default judge (cost vs. quality)?
2. Should scores be 0–1 floats only, or also expose a 0–100 display scale in the CLI/JSON?
3. Should `reasoning` be on by default (more tokens) or opt-in?
4. Will a golden dataset of hand-labeled examples be built in-repo to validate the judge prompts?
5. Sync-only for v0.1, or add `aevaluate()` async from the start?

## 15. Principles

1. Small and finished beats large and unfinished.
2. Every component is understood by the maintainer, not just generated.
3. Provider-neutral core.
4. Tests run offline.
5. Make contributing easy: one metric or provider per PR.
