"""Bound model-selected experiments to a small, reproducible simulated bench."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FocusSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A 32x32 DMD is grouped into square binary amplitude macropixels.
    macropixel_size: Literal[1, 2, 4, 8]
    focus_x: int = Field(ge=0, le=31)
    focus_y: int = Field(ge=0, le=31)
    focus_size: Literal[1, 3, 5]
    background_inner_radius: int = Field(ge=1, le=10)
    background_outer_radius: int = Field(ge=3, le=14)
    background_regions: int = Field(ge=3, le=8)
    iterations: int = Field(ge=10, le=3000)
    flip_count: int = Field(ge=1, le=16)
    frames_per_measurement: int = Field(ge=1, le=8)
    validation_shots: int = Field(ge=3, le=20)
    seed: int = Field(ge=0, le=99999)

    @model_validator(mode="after")
    def check_geometry(self):
        # Keep the focus square clear of the surrounding background annulus.
        if self.background_inner_radius <= (self.focus_size // 2) * 2**0.5:
            raise ValueError("Background inner radius must lie outside the focus square")
        if self.background_outer_radius <= self.background_inner_radius:
            raise ValueError("Background outer radius must exceed inner radius")
        radius = self.background_outer_radius
        if not (radius <= self.focus_x < 32 - radius and radius <= self.focus_y < 32 - radius):
            raise ValueError("The full background annulus must fit inside the 32x32 camera")
        if self.flip_count > (32 // self.macropixel_size) ** 2:
            raise ValueError("Cannot flip more macropixels than the DMD contains")
        return self


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class FocusPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    physics: str = Field(min_length=1, max_length=500)
    enhancement_definition: str = Field(min_length=1, max_length=300)
    experiments: list[FocusSettings] = Field(min_length=2, max_length=4)
    rationale: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def distinct_experiments(self):
        if len({experiment.model_dump_json() for experiment in self.experiments}) != len(self.experiments):
            raise ValueError("Propose distinct experiments")
        return self
