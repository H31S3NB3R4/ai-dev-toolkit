"""Unit tests for the ai-dev CLI interface."""

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from ai_dev_toolkit._version import __version__
from ai_dev_toolkit.cli.main import app

runner = CliRunner()


class TestCliBasics:
    """Test CLI commands and flags."""

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert f"v{__version__}" in result.stdout

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert f"v{__version__}" in result.stdout

    def test_version_command(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert f"v{__version__}" in result.stdout

    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "evaluate" in result.stdout

    def test_evaluate_help(self) -> None:
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--prompt" in result.stdout
        assert "--response" in result.stdout
        assert "--context" in result.stdout
        assert "--min-score" in result.stdout
        assert "--json" in result.stdout


class TestCliEvaluation:
    """Test evaluate command with various options."""

    def test_missing_prompt_fails(self) -> None:
        result = runner.invoke(app, ["evaluate", "--response", "Some response"])
        assert result.exit_code == 1
        assert "Missing required prompt" in (result.stderr or result.stdout)

    def test_missing_response_fails(self) -> None:
        result = runner.invoke(app, ["evaluate", "--prompt", "Some prompt"])
        assert result.exit_code == 1
        assert "Missing required response" in (result.stderr or result.stdout)

    def test_missing_file_fails(self) -> None:
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--prompt-file",
                "non_existent_prompt_file_123.txt",
                "--response",
                "Some response",
            ],
        )
        assert result.exit_code == 1
        assert "Error reading prompt file" in (result.stderr or result.stdout)

    def test_evaluate_with_json_output(self) -> None:
        fake_json = {
            "score": 0.95,
            "reasoning": "Excellent",
            "sub_questions": ["Q1"],
        }
        with patch.dict("os.environ", {"AI_DEV_PROVIDER": "fake"}):
            with patch(
                "ai_dev_toolkit.providers.fake.FakeProvider.generate",
                return_value=json.dumps(fake_json),
            ):
                result = runner.invoke(
                    app,
                    [
                        "evaluate",
                        "--prompt",
                        "What is Python?",
                        "--response",
                        "Python is a programming language.",
                        "--provider",
                        "fake",
                        "--json",
                    ],
                )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "relevance" in data
        assert "completeness" in data
        assert "overall" in data
        assert data["overall"] == 0.95
        assert data["metadata"]["provider"] == "fake"

    def test_evaluate_table_output(self) -> None:
        fake_json = {
            "score": 0.9,
            "reasoning": "Very relevant",
            "sub_questions": ["Q1"],
        }
        with patch(
            "ai_dev_toolkit.providers.fake.FakeProvider.generate",
            return_value=json.dumps(fake_json),
        ):
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--prompt",
                    "Explain recursion",
                    "--response",
                    "Recursion is a function calling itself.",
                    "--provider",
                    "fake",
                ],
            )
        assert result.exit_code == 0
        assert "Relevance" in result.stdout
        assert "Completeness" in result.stdout
        assert "Overall" in result.stdout

    def test_evaluate_table_with_reasoning(self) -> None:
        fake_json = {
            "score": 1.0,
            "reasoning": "Detailed justification here",
            "sub_questions": ["Q1"],
        }
        with patch(
            "ai_dev_toolkit.providers.fake.FakeProvider.generate",
            return_value=json.dumps(fake_json),
        ):
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--prompt",
                    "Explain recursion",
                    "--response",
                    "Recursion is a function calling itself.",
                    "--provider",
                    "fake",
                    "--reasoning",
                ],
            )
        assert result.exit_code == 0
        assert "Detailed justification here" in result.stdout

    def test_evaluate_with_files(self, tmp_path: Path) -> None:
        p_file = tmp_path / "prompt.txt"
        r_file = tmp_path / "response.txt"
        c_file = tmp_path / "context.txt"

        p_file.write_text("What is AI?", encoding="utf-8")
        r_file.write_text("AI is artificial intelligence.", encoding="utf-8")
        c_file.write_text("Context about AI.", encoding="utf-8")

        def fake_generate(prompt: str, *args: object, **kwargs: object) -> str:
            if "sub-questions" in prompt:
                return json.dumps({"sub_questions": ["Q1"], "score": 1.0})
            elif (
                "verify" in prompt.lower()
                or "verdicts" in prompt.lower()
                or "supported" in prompt.lower()
            ):
                return json.dumps(
                    {
                        "verdicts": [
                            {
                                "claim": "AI is artificial intelligence.",
                                "supported": True,
                            }
                        ],
                        "score": 1.0,
                    }
                )
            elif "extract" in prompt.lower():
                return json.dumps({"claims": ["AI is artificial intelligence."]})
            return json.dumps({"score": 1.0, "reasoning": "Direct answer"})

        with patch(
            "ai_dev_toolkit.providers.fake.FakeProvider.generate",
            side_effect=fake_generate,
        ):
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--prompt-file",
                    str(p_file),
                    "--response-file",
                    str(r_file),
                    "--context-file",
                    str(c_file),
                    "--provider",
                    "fake",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["faithfulness"] == 1.0
        assert data["overall"] == 1.0

    def test_min_score_passes_when_above_threshold(self) -> None:
        fake_json = {
            "score": 0.9,
            "sub_questions": ["Q1"],
        }
        with patch(
            "ai_dev_toolkit.providers.fake.FakeProvider.generate",
            return_value=json.dumps(fake_json),
        ):
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--prompt",
                    "Hello",
                    "--response",
                    "World",
                    "--provider",
                    "fake",
                    "--min-score",
                    "0.8",
                ],
            )
        assert result.exit_code == 0

    def test_min_score_fails_when_below_threshold(self) -> None:
        fake_json = {
            "score": 0.5,
            "sub_questions": ["Q1"],
        }
        with patch(
            "ai_dev_toolkit.providers.fake.FakeProvider.generate",
            return_value=json.dumps(fake_json),
        ):
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--prompt",
                    "Hello",
                    "--response",
                    "World",
                    "--provider",
                    "fake",
                    "--min-score",
                    "0.8",
                ],
            )
        assert result.exit_code == 1

    def test_missing_api_key_displays_clear_error(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "ai_dev_toolkit.providers.gemini._find_env_key",
                return_value=None,
            ):
                result = runner.invoke(
                    app,
                    [
                        "evaluate",
                        "--prompt",
                        "Hello",
                        "--response",
                        "World",
                        "--provider",
                        "gemini",
                    ],
                )
        assert result.exit_code == 1
        assert "GEMINI_API_KEY" in (result.stderr or result.stdout)
