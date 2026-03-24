"""
FastAPI application entry-point.
"""

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.models import (
    PitchType,
    PitchTypeInfo,
    PitchingScenario,
    PitchingSuggestionsResponse,
)
from app.services import PITCH_INFO, get_suggestions

app = FastAPI(
    title="Catcher Scenario / Pitching Suggestion API",
    description=(
        "An API service that returns ranked pitching suggestions based on "
        "in-game criteria such as count, batter handedness, inning, "
        "runners on base, and previous pitches thrown."
    ),
    version="1.0.0",
    contact={
        "name": "catcher_scenario_api",
        "url": "https://github.com/swords50001/catcher_scenario_api",
    },
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check", tags=["system"])
def health() -> JSONResponse:
    """Returns a simple alive response so load balancers can verify the service is running."""
    return JSONResponse(content={"status": "ok"})


# ---------------------------------------------------------------------------
# Pitch types catalogue
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/pitch-types",
    response_model=List[PitchTypeInfo],
    summary="List all supported pitch types",
    tags=["reference"],
)
def list_pitch_types() -> List[PitchTypeInfo]:
    """Returns metadata for every pitch type recognised by the suggestion engine."""
    return [
        PitchTypeInfo(
            pitch_type=pt,
            name=info["name"],
            description=info["description"],
            typical_velocity_mph=info["typical_velocity_mph"],
            movement=info["movement"],
        )
        for pt, info in PITCH_INFO.items()
    ]


@app.get(
    "/api/v1/pitch-types/{pitch_type}",
    response_model=PitchTypeInfo,
    summary="Get a single pitch type by key",
    tags=["reference"],
)
def get_pitch_type(pitch_type: str) -> PitchTypeInfo:
    """Returns metadata for a single pitch type.  Use the ``pitch_type`` enum value as the path parameter."""
    try:
        pt = PitchType(pitch_type)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Pitch type '{pitch_type}' not found. "
                   f"Valid values: {[p.value for p in PitchType]}",
        )

    info = PITCH_INFO.get(pt)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Pitch type '{pitch_type}' not found.")

    return PitchTypeInfo(
        pitch_type=pt,
        name=info["name"],
        description=info["description"],
        typical_velocity_mph=info["typical_velocity_mph"],
        movement=info["movement"],
    )


# ---------------------------------------------------------------------------
# Pitching suggestions
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/suggestions",
    response_model=PitchingSuggestionsResponse,
    summary="Get pitching suggestions for a scenario",
    tags=["suggestions"],
    status_code=200,
)
def create_suggestion(scenario: PitchingScenario) -> PitchingSuggestionsResponse:
    """
    Given an in-game scenario, returns a ranked list of pitching suggestions
    along with a confidence score and rationale for each.

    **Scenario criteria**

    | Field | Description |
    |---|---|
    | `balls` | Current ball count (0–3) |
    | `strikes` | Current strike count (0–2) |
    | `batter_handedness` | `left` / `right` / `switch` |
    | `pitcher_handedness` | `left` / `right` |
    | `inning` | Current inning number |
    | `outs` | Number of outs (0–2) |
    | `runners_on_base` | List of occupied bases e.g. `[1, 2]` |
    | `previous_pitches` | Pitches thrown to this batter so far |
    | `available_pitch_types` | Pitch types in the pitcher's repertoire (optional) |
    """
    return get_suggestions(scenario)
