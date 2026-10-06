"""Configuration models for the evaluation engine."""

import os

from pydantic import BaseModel, Field, model_validator


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
