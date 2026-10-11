"""Validate model responses before they influence the graph or its tools."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class TrainingSetting(BaseModel):
    """Bounds keep the proposed study within one five-minute script."""

    model_config = ConfigDict(extra="forbid")
    optimizer: Literal["Adam", "AdamW"]
    learning_rate: float = Field(ge=0.0001, le=0.03)
    steps: int = Field(ge=500, le=800)
    rollout_length: int = Field(ge=16, le=24)
    batch_size: int = Field(ge=1, le=2)


class ScientificPlan(BaseModel):
    """The scientist owns the theory, settings, and ordered coding steps."""

    model_config = ConfigDict(extra="forbid")
    differential_equation: str = Field(max_length=200)
    state_equation: str = Field(max_length=200)
    network_and_scaling: str = Field(max_length=500)
    architectures: list[Annotated[
        list[Annotated[int, Field(ge=1, le=64)]], Field(min_length=1, max_length=3)
    ]] = Field(min_length=2, max_length=2, description="Two arrays of hidden-layer widths only; no input/output dimensions")
    settings: list[TrainingSetting] = Field(min_length=2, max_length=2)
    training_method: str = Field(max_length=500)
    free_run_test: str = Field(max_length=500)
    implementation_steps: list[Annotated[str, Field(max_length=220)]] = Field(min_length=5, max_length=8)

    @model_validator(mode="after")
    def check_comparison(self):
        if self.architectures[0] == self.architectures[1]:
            raise ValueError("Propose two different architectures")
        for widths in self.architectures:
            if not 1 <= len(widths) <= 3 or any(not 1 <= width <= 64 for width in widths):
                raise ValueError("Use 1-3 hidden layers, each with 1-64 neurons")
        for field in ("optimizer", "learning_rate", "steps", "rollout_length"):
            if getattr(self.settings[0], field) == getattr(self.settings[1], field):
                raise ValueError(f"The two settings must differ in {field}")
        return self


class CodeSubmission(BaseModel):
    """Model-generated arguments passed to the executor's write/run tool."""

    model_config = ConfigDict(extra="forbid")
    filename: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*\.py$")
    code: str = Field(min_length=1, max_length=2000,
                      description="Concise runnable Python calling study_utils.run_study")
