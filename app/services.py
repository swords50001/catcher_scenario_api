"""
Rule-based pitching suggestion service.

Strategy overview
-----------------
* Ahead in count (0-2, 1-2)  → emphasise breaking balls / off-speed away
* Behind in count (3-0, 3-1) → stick to fastballs for strikes
* Even / full count          → mix speed and location to keep batter guessing
* Handedness matchup         → favour same-side breaking balls; away = platoon advantage
* Runners in scoring position → pitch to contact, low-and-away
* Two-out / RISP             → aggressive; strikeout pitches
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from app.models import (
    Handedness,
    PitchLocation,
    PitchSuggestion,
    PitchType,
    PitchingScenario,
    PitchingSuggestionsResponse,
)

# ---------------------------------------------------------------------------
# Pitch catalogue
# ---------------------------------------------------------------------------

PITCH_INFO: Dict[PitchType, Dict] = {
    PitchType.FASTBALL: {
        "name": "Fastball",
        "description": "Generic high-velocity straight pitch.",
        "typical_velocity_mph": "90–96",
        "movement": "Minimal movement; slight rise due to spin.",
    },
    PitchType.FOUR_SEAM: {
        "name": "Four-Seam Fastball",
        "description": "Highest-velocity pitch with a rising trajectory.",
        "typical_velocity_mph": "92–100",
        "movement": "High spin rate, rides up through the zone.",
    },
    PitchType.TWO_SEAM: {
        "name": "Two-Seam Fastball",
        "description": "Sinking fastball with arm-side run.",
        "typical_velocity_mph": "89–94",
        "movement": "Late sink and run into same-handed batters.",
    },
    PitchType.CUTTER: {
        "name": "Cut Fastball",
        "description": "Fastball that cuts toward the glove side late.",
        "typical_velocity_mph": "86–92",
        "movement": "Late glove-side cut.",
    },
    PitchType.SINKER: {
        "name": "Sinker",
        "description": "Hard sinking pitch designed to induce ground balls.",
        "typical_velocity_mph": "88–94",
        "movement": "Heavy downward and arm-side run.",
    },
    PitchType.CHANGEUP: {
        "name": "Changeup",
        "description": "Off-speed pitch designed to disrupt batter timing.",
        "typical_velocity_mph": "78–86",
        "movement": "Fades arm-side with late tumble.",
    },
    PitchType.CURVEBALL: {
        "name": "Curveball",
        "description": "12-to-6 breaking ball with sharp downward break.",
        "typical_velocity_mph": "72–82",
        "movement": "Large downward break, top-spin.",
    },
    PitchType.SLIDER: {
        "name": "Slider",
        "description": "Hard breaking ball with lateral and downward break.",
        "typical_velocity_mph": "82–90",
        "movement": "Tight glove-side sweep.",
    },
    PitchType.SPLITTER: {
        "name": "Splitter",
        "description": "Split-finger fastball with late diving action.",
        "typical_velocity_mph": "82–88",
        "movement": "Sudden late dive, looks like a fastball then drops.",
    },
    PitchType.KNUCKLEBALL: {
        "name": "Knuckleball",
        "description": "Slow, unpredictable pitch with minimal spin.",
        "typical_velocity_mph": "60–75",
        "movement": "Random fluttering — difficult to predict or control.",
    },
}

# Default repertoire if the pitcher hasn't specified their available pitches.
DEFAULT_REPERTOIRE: List[PitchType] = [
    PitchType.FOUR_SEAM,
    PitchType.TWO_SEAM,
    PitchType.SLIDER,
    PitchType.CHANGEUP,
    PitchType.CURVEBALL,
]

# Fastball group used in count-based logic.
FASTBALL_TYPES: Set[PitchType] = {
    PitchType.FASTBALL,
    PitchType.FOUR_SEAM,
    PitchType.TWO_SEAM,
    PitchType.CUTTER,
    PitchType.SINKER,
}

BREAKING_TYPES: Set[PitchType] = {
    PitchType.CURVEBALL,
    PitchType.SLIDER,
    PitchType.CUTTER,
}

OFFSPEED_TYPES: Set[PitchType] = {
    PitchType.CHANGEUP,
    PitchType.SPLITTER,
    PitchType.KNUCKLEBALL,
}


# ---------------------------------------------------------------------------
# Situation tagging helpers
# ---------------------------------------------------------------------------

def _derive_situations(scenario: PitchingScenario) -> List[str]:
    tags = []
    runners = set(scenario.runners_on_base)

    if scenario.balls == 0 and scenario.strikes == 0:
        tags.append("first_pitch")
    if runners & {2, 3}:
        tags.append("risp")
    if runners == {1, 2, 3}:
        tags.append("bases_loaded")
    if scenario.outs == 2:
        tags.append("two_out")
    if not runners and scenario.outs == 0:
        tags.append("lead_off")

    # Count situation
    if scenario.strikes == 2 and scenario.balls < 3:
        tags.append("pitcher_count")
    elif scenario.balls >= 3 and scenario.strikes < 2:
        tags.append("hitter_count")
    elif scenario.balls == 3 and scenario.strikes == 2:
        tags.append("full_count")
    else:
        tags.append("even_count")

    # Late inning pressure
    if scenario.inning >= 7:
        tags.append("late_game")

    return tags


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------

_CANDIDATE = Tuple[PitchType, PitchLocation, float, str]


def _score_candidates(
    scenario: PitchingScenario,
    repertoire: List[PitchType],
    situations: List[str],
) -> List[PitchSuggestion]:
    """Generate and rank pitch suggestions for the given scenario."""

    candidates: List[_CANDIDATE] = []

    is_pitcher_count = "pitcher_count" in situations
    is_hitter_count = "hitter_count" in situations
    is_full_count = "full_count" in situations
    is_risp = "risp" in situations
    is_two_out_risp = is_risp and "two_out" in situations
    is_first_pitch = "first_pitch" in situations

    # Platoon advantage — same-side pitcher/batter
    same_side = (
        scenario.pitcher_handedness == scenario.batter_handedness
        and scenario.batter_handedness != Handedness.SWITCH
    )

    away_location = PitchLocation.AWAY
    in_location = PitchLocation.IN
    down_away = PitchLocation.DOWN_AND_AWAY
    down_in = PitchLocation.DOWN_AND_IN
    up_in = PitchLocation.UP_AND_IN

    recent = set(scenario.previous_pitches[-2:]) if scenario.previous_pitches else set()

    for pitch in repertoire:
        is_fb = pitch in FASTBALL_TYPES
        is_breaking = pitch in BREAKING_TYPES
        is_offspeed = pitch in OFFSPEED_TYPES

        # ----- Base score and defaults -----
        score = 0.5
        location: PitchLocation = down_away
        rationale: str = ""

        # ---- Count adjustments ----
        if is_hitter_count:
            # Behind in count: throw strikes with fastballs
            if is_fb:
                score += 0.20
                location = away_location
                rationale = "Hitter's count — fastball for a strike to work back into the count."
            else:
                score -= 0.10
                location = down_away
                rationale = "Off-speed in a hitter's count — risky; use only to steal a strike."

        elif is_pitcher_count:
            # Ahead in count: expand the zone with breaking balls / off-speed
            if is_breaking:
                score += 0.25
                location = PitchLocation.WASTE if scenario.balls <= 1 else down_away
                rationale = (
                    "Pitcher's count — breaking ball to expand the zone and induce a chase."
                )
            elif is_offspeed:
                score += 0.18
                location = down_away
                rationale = "Pitcher's count — off-speed to disrupt timing and get a swing-and-miss."
            else:
                score += 0.05
                location = up_in
                rationale = "Pitcher's count — high fastball to set up a breaking ball."

        elif is_full_count:
            # Full count: best pitch, must throw a strike
            if is_fb:
                score += 0.15
                location = away_location
                rationale = "Full count — fastball away to get a called strike or weak contact."
            elif is_breaking and scenario.balls < 3:
                score += 0.10
                location = down_away
                rationale = "Full count — breaking ball down-and-away to induce a swing-and-miss."
            else:
                score += 0.05
                location = down_away
                rationale = "Full count — off-speed pitch to keep the batter off-balance."

        else:  # even count / first pitch
            if is_first_pitch and is_fb:
                score += 0.22
                location = away_location
                rationale = "First pitch fastball — establish the strike zone early."
            elif is_fb:
                score += 0.10
                location = away_location
                rationale = "Even count — fastball to get ahead."
            elif is_breaking:
                score += 0.08
                location = down_away
                rationale = "Even count — breaking ball to keep the batter guessing."
            else:
                score += 0.06
                location = down_away
                rationale = "Even count — off-speed to disrupt timing."

        # ---- Platoon / handedness adjustments ----
        if same_side and is_breaking:
            score += 0.08
            location = down_away
            rationale += " Platoon advantage — breaking ball sweeps away from the batter."

        # ---- RISP adjustments ----
        if is_risp:
            if is_fb and (pitch in {PitchType.SINKER, PitchType.TWO_SEAM}):
                score += 0.12
                location = down_in
                rationale += " RISP — sink it for a ground ball double play."
            elif is_two_out_risp and is_breaking:
                score += 0.10
                rationale += " Two outs, RISP — go for the strikeout with a breaking ball."

        # ---- Avoid repeating the same pitch type ----
        if pitch in recent:
            score -= 0.12

        # ---- Clamp ----
        score = round(min(max(score, 0.05), 0.99), 2)

        candidates.append((pitch, location, score, rationale))

    # Sort by confidence descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    return [
        PitchSuggestion(
            pitch_type=pt,
            location=loc,
            confidence=conf,
            rationale=rat,
        )
        for pt, loc, conf, rat in candidates
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_suggestions(scenario: PitchingScenario) -> PitchingSuggestionsResponse:
    """Return pitching suggestions for the provided scenario."""

    repertoire = (
        scenario.available_pitch_types
        if scenario.available_pitch_types
        else DEFAULT_REPERTOIRE
    )

    situations = _derive_situations(scenario)
    suggestions = _score_candidates(scenario, repertoire, situations)

    return PitchingSuggestionsResponse(
        balls=scenario.balls,
        strikes=scenario.strikes,
        batter_handedness=scenario.batter_handedness,
        pitcher_handedness=scenario.pitcher_handedness,
        inning=scenario.inning,
        outs=scenario.outs,
        situation_tags=situations,
        suggestions=suggestions,
    )
