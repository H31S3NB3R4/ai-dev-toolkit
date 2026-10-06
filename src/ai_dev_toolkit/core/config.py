"""Configuration models and file loaders for the evaluation engine."""

import json
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _parse_yaml_content(content: str) -> dict[str, Any]:
    """Parse YAML content using PyYAML if available, or lightweight fallback."""
    try:
        import yaml

        parsed = yaml.safe_load(content)
        return parsed if isinstance(parsed, dict) else {}
    except ImportError:
        # Fallback simple line-by-line key: value parser for basic YAML
        result: dict[str, Any] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            # Convert booleans and numbers if applicable
            if v.lower() == "true":
                result[k] = True
            elif v.lower() == "false":
                result[k] = False
            elif v.lower() in ("null", "none"):
                result[k] = None
            else:
                try:
                    if "." in v:
                        result[k] = float(v)
                    else:
                        result[k] = int(v)
                except ValueError:
                    result[k] = v
        return result


class EvaluatorConfig(BaseModel):
    """Configuration options for evaluation execution and weighting."""

    relevance_weight: float = Field(
        default=0.35,
        ge=0.0,
        description="Weight assigned to relevance metric in overall score calculation.",
    )
    completeness_weight: float = Field(
        default=0.25,
        ge=0.0,
        description=(
            "Weight assigned to completeness metric in overall score calculation."
        ),
    )
    faithfulness_weight: float = Field(
        default=0.40,
        ge=0.0,
        description=(
            "Weight assigned to faithfulness metric in overall score calculation."
        ),
    )
    provider: str = Field(
        default_factory=lambda: os.getenv("AI_DEV_PROVIDER", "gemini"),
        description="Identifier of LLM provider to use for evaluation.",
    )
    model: str | None = Field(
        default=None,
        description="Optional model identifier override for provider.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for LLM judge (default 0.0 for deterministic"
            " evaluation)."
        ),
    )
    timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="Request timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for transient provider/parse failures.",
    )
    include_reasoning: bool = Field(
        default=True,
        description=(
            "Whether to request and record natural-language reasoning "
            "for metric scores."
        ),
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "EvaluatorConfig":
        """Ensure total weight sum is strictly positive."""
        total = (
            self.relevance_weight + self.completeness_weight + self.faithfulness_weight
        )
        if total <= 0.0:
            raise ValueError("Total metric weight sum must be greater than 0.")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluatorConfig":
        """Load configuration from a TOML, YAML, or JSON file.

        Args:
            path: Path to configuration file.

        Returns:
            An instantiated EvaluatorConfig.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If file format is unsupported or contains invalid settings.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file '{path}' not found.")

        suffix = config_path.suffix.lower()
        content = config_path.read_text(encoding="utf-8")

        data: dict[str, Any] = {}
        if suffix in (".toml",):
            data = tomllib.loads(content)
        elif suffix in (".yaml", ".yml"):
            data = _parse_yaml_content(content)
        elif suffix in (".json",):
            data = json.loads(content)
        else:
            raise ValueError(
                f"Unsupported configuration format '{suffix}'. "
                f"Supported formats: .toml, .yaml, .yml, .json"
            )

        # Handle nested [evaluator] or [ai-dev] table if present in TOML/YAML
        if "evaluator" in data and isinstance(data["evaluator"], dict):
            data = data["evaluator"]
        elif "ai-dev" in data and isinstance(data["ai-dev"], dict):
            data = data["ai-dev"]

        return cls(**data)

    @classmethod
    def find_and_load(cls, directory: str | Path | None = None) -> "EvaluatorConfig":
        """Search current or specified directory for standard config files."""
        search_dir = Path(directory) if directory else Path.cwd()
        candidate_names = [
            "ai-dev.toml",
            ".ai-dev.toml",
            "ai-dev.yaml",
            "ai-dev.yml",
            ".ai-dev.yaml",
            ".ai-dev.yml",
            "ai-dev.json",
            ".ai-dev.json",
        ]

        for name in candidate_names:
            candidate = search_dir / name
            if candidate.is_file():
                return cls.from_file(candidate)

        return cls()
