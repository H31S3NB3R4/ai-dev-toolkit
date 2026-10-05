# AI Dev Toolkit (`ai-dev-toolkit`)

[![PyPI version](https://img.shields.io/pypi/v/ai-dev-toolkit.svg)](https://pypi.org/project/ai-dev-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> Open-source evaluation and testing toolkit for LLM, RAG, and AI agents.

*Status: Work in progress (v0.1.0 in active development)*

## Quick Summary

AI Dev Toolkit is a lightweight, modular library to help developers evaluate, test, and debug LLM outputs. It enables one-call evaluation of LLM answers against relevance, completeness, and faithfulness metrics with structured JSON output ready for CI/CD gates.

## Roadmap & Features (v0.1.0)

- **One-call evaluation**: `evaluate(prompt, response, context=None)`
- **Core metrics**: Relevance, Completeness, Faithfulness (Hallucination = 1 - Faithfulness)
- **Providers**: Gemini (via official SDK), offline FakeProvider for zero-cost testing
- **CLI**: `ai-dev evaluate` with exit code gating on score thresholds
- **Developer-friendly**: Strict types, 100% offline unit tests, clean Pydantic data models
