"""
Pydantic models for request and response payloads.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    SWITCH = "switch"


class PitchType(str, Enum):
    FASTBALL = "fastball"
    FOUR_SEAM = "four_seam_fastball"
    TWO_SEAM = "two_seam_fastball"
    CUTTER = "cutter"
    SINKER = "sinker"
    CHANGEUP = "changeup"
    CURVEBALL = "curveball"
    SLIDER = "slider"
    SPLITTER = "splitter"
    KNUCKLEBALL = "knuckleball"


class PitchLocation(str, Enum):
    UP_AND_IN = "up_and_in"
    UP = "up"
    UP_AND_AWAY = "up_and_away"
    IN = "in"
    MIDDLE = "middle"
    AWAY = "away"
    DOWN_AND_IN = "down_and_in"
    DOWN = "down"
    DOWN_AND_AWAY = "down_and_away"
    WASTE = "waste"


class GameSituation(str, Enum):
    NORMAL = "normal"
    RISP = "risp"                       # runner in scoring position
    BASES_LOADED = "bases_loaded"
    FIRST_PITCH = "first_pitch"
    TWO_OUT = "two_out"
    LEAD_OFF = "lead_off"               # first batter of an inning


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class PitchingScenario(BaseModel):
    """Criteria supplied by the user to generate pitching suggestions."""

    balls: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Current ball count (0–3).",
    )
    strikes: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Current strike count (0–2).",
    )
    batter_handedness: Handedness = Field(
        default=Handedness.RIGHT,
        description="Handedness of the batter at the plate.",
    )
    inning: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Current inning number.",
    )
    outs: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Number of outs in the current inning.",
    )
    runners_on_base: List[int] = Field(
        default_factory=list,
        description="List of occupied bases (1, 2, and/or 3).",
    )
    previous_pitches: List[PitchType] = Field(
        default_factory=list,
        description="Pitch types thrown to this batter so far in the at-bat, oldest first.",
    )
    available_pitch_types: Optional[List[PitchType]] = Field(
        default=None,
        description=(
            "Pitch types the pitcher has available. "
            "If omitted, all pitch types are considered."
        ),
    )
    pitcher_handedness: Handedness = Field(
        default=Handedness.RIGHT,
        description="Handedness of the pitcher.",
    )

    @model_validator(mode="after")
    def validate_runners(self) -> "PitchingScenario":
        valid_bases = {1, 2, 3}
        for base in self.runners_on_base:
            if base not in valid_bases:
                raise ValueError(
                    f"runners_on_base must only contain 1, 2, or 3. Got: {base}"
                )
        return self


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PitchSuggestion(BaseModel):
    """A single suggested pitch."""

    pitch_type: PitchType = Field(description="Recommended pitch type.")
    location: PitchLocation = Field(description="Recommended target location.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this suggestion (0.0–1.0).",
    )
    rationale: str = Field(description="Short explanation for the suggestion.")


class PitchingSuggestionsResponse(BaseModel):
    """Response containing an ordered list of pitching suggestions."""

    balls: int
    strikes: int
    batter_handedness: Handedness
    pitcher_handedness: Handedness
    inning: int
    outs: int
    situation_tags: List[str] = Field(
        description="High-level situation labels applied to this scenario."
    )
    suggestions: List[PitchSuggestion] = Field(
        description="Pitching suggestions ranked by confidence (highest first)."
    )


class PitchTypeInfo(BaseModel):
    """Metadata about a single pitch type."""

    pitch_type: PitchType
    name: str
    description: str
    typical_velocity_mph: str
    movement: str
