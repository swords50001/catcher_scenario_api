"""
Tests for the Catcher Scenario / Pitching Suggestion API.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Handedness, PitchType
from app.services import (
    DEFAULT_REPERTOIRE,
    FASTBALL_TYPES,
    BREAKING_TYPES,
    _derive_situations,
    get_suggestions,
)
from app.models import PitchingScenario

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Pitch-types catalogue
# ---------------------------------------------------------------------------

def test_list_pitch_types_returns_all():
    resp = client.get("/api/v1/pitch-types")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == len(PitchType)
    keys = {item["pitch_type"] for item in data}
    for pt in PitchType:
        assert pt.value in keys


def test_get_single_pitch_type_fastball():
    resp = client.get("/api/v1/pitch-types/four_seam_fastball")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pitch_type"] == "four_seam_fastball"
    assert "name" in data
    assert "description" in data
    assert "typical_velocity_mph" in data
    assert "movement" in data


def test_get_pitch_type_not_found():
    resp = client.get("/api/v1/pitch-types/notapitch")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Suggestion endpoint — basic shape
# ---------------------------------------------------------------------------

def _default_payload(**overrides):
    base = {
        "balls": 0,
        "strikes": 0,
        "batter_handedness": "right",
        "pitcher_handedness": "right",
        "inning": 1,
        "outs": 0,
        "runners_on_base": [],
        "previous_pitches": [],
    }
    base.update(overrides)
    return base


def test_suggestions_default_scenario():
    resp = client.post("/api/v1/suggestions", json=_default_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert "suggestions" in data
    assert len(data["suggestions"]) > 0
    assert "situation_tags" in data
    first = data["suggestions"][0]
    assert "pitch_type" in first
    assert "location" in first
    assert 0.0 <= first["confidence"] <= 1.0
    assert "rationale" in first


def test_suggestions_are_ranked_descending():
    resp = client.post("/api/v1/suggestions", json=_default_payload())
    assert resp.status_code == 200
    confidences = [s["confidence"] for s in resp.json()["suggestions"]]
    assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# Suggestion logic — pitcher's count
# ---------------------------------------------------------------------------

def test_pitchers_count_tags():
    """0-2 count should be tagged as pitcher_count."""
    scenario = PitchingScenario(balls=0, strikes=2)
    tags = _derive_situations(scenario)
    assert "pitcher_count" in tags


def test_pitchers_count_prefers_breaking_balls():
    scenario = PitchingScenario(
        balls=0,
        strikes=2,
        available_pitch_types=[
            PitchType.FOUR_SEAM,
            PitchType.SLIDER,
            PitchType.CURVEBALL,
            PitchType.CHANGEUP,
        ],
    )
    result = get_suggestions(scenario)
    top = result.suggestions[0]
    assert top.pitch_type in BREAKING_TYPES | {PitchType.CHANGEUP}


# ---------------------------------------------------------------------------
# Suggestion logic — hitter's count
# ---------------------------------------------------------------------------

def test_hitters_count_tags():
    """3-1 count should be tagged as hitter_count."""
    scenario = PitchingScenario(balls=3, strikes=1)
    tags = _derive_situations(scenario)
    assert "hitter_count" in tags


def test_hitters_count_prefers_fastballs():
    scenario = PitchingScenario(
        balls=3,
        strikes=1,
        available_pitch_types=[
            PitchType.FOUR_SEAM,
            PitchType.SLIDER,
            PitchType.CURVEBALL,
            PitchType.CHANGEUP,
        ],
    )
    result = get_suggestions(scenario)
    top = result.suggestions[0]
    assert top.pitch_type in FASTBALL_TYPES


# ---------------------------------------------------------------------------
# Suggestion logic — full count
# ---------------------------------------------------------------------------

def test_full_count_tags():
    scenario = PitchingScenario(balls=3, strikes=2)
    tags = _derive_situations(scenario)
    assert "full_count" in tags


def test_full_count_returns_suggestions():
    resp = client.post("/api/v1/suggestions", json=_default_payload(balls=3, strikes=2))
    assert resp.status_code == 200
    assert len(resp.json()["suggestions"]) > 0


# ---------------------------------------------------------------------------
# Suggestion logic — RISP
# ---------------------------------------------------------------------------

def test_risp_tag_runner_on_second():
    scenario = PitchingScenario(balls=1, strikes=1, runners_on_base=[2])
    tags = _derive_situations(scenario)
    assert "risp" in tags


def test_risp_tag_runner_on_third():
    scenario = PitchingScenario(balls=0, strikes=0, runners_on_base=[3])
    tags = _derive_situations(scenario)
    assert "risp" in tags


def test_no_risp_tag_runner_on_first_only():
    scenario = PitchingScenario(balls=0, strikes=0, runners_on_base=[1])
    tags = _derive_situations(scenario)
    assert "risp" not in tags


def test_bases_loaded_tags():
    scenario = PitchingScenario(balls=0, strikes=0, runners_on_base=[1, 2, 3])
    tags = _derive_situations(scenario)
    assert "bases_loaded" in tags
    assert "risp" in tags


# ---------------------------------------------------------------------------
# Suggestion logic — first pitch
# ---------------------------------------------------------------------------

def test_first_pitch_tag():
    scenario = PitchingScenario(balls=0, strikes=0)
    tags = _derive_situations(scenario)
    assert "first_pitch" in tags


def test_first_pitch_fastball_advantage():
    scenario = PitchingScenario(
        balls=0,
        strikes=0,
        available_pitch_types=[PitchType.FOUR_SEAM, PitchType.CURVEBALL],
    )
    result = get_suggestions(scenario)
    top = result.suggestions[0]
    assert top.pitch_type == PitchType.FOUR_SEAM


# ---------------------------------------------------------------------------
# Suggestion logic — late game
# ---------------------------------------------------------------------------

def test_late_game_tag_inning_7():
    scenario = PitchingScenario(balls=0, strikes=0, inning=7)
    tags = _derive_situations(scenario)
    assert "late_game" in tags


def test_late_game_tag_not_early():
    scenario = PitchingScenario(balls=0, strikes=0, inning=6)
    tags = _derive_situations(scenario)
    assert "late_game" not in tags


# ---------------------------------------------------------------------------
# Suggestion logic — avoid repeat pitches
# ---------------------------------------------------------------------------

def test_avoids_recent_pitch():
    """If the same pitch was just thrown, its confidence should be lower."""
    base_scenario = PitchingScenario(
        balls=1,
        strikes=1,
        available_pitch_types=[PitchType.FOUR_SEAM, PitchType.SLIDER],
    )
    base_result = get_suggestions(base_scenario)
    fb_conf_without_history = next(
        s.confidence for s in base_result.suggestions if s.pitch_type == PitchType.FOUR_SEAM
    )

    repeated_scenario = PitchingScenario(
        balls=1,
        strikes=1,
        available_pitch_types=[PitchType.FOUR_SEAM, PitchType.SLIDER],
        previous_pitches=[PitchType.FOUR_SEAM, PitchType.FOUR_SEAM],
    )
    repeated_result = get_suggestions(repeated_scenario)
    fb_conf_with_history = next(
        s.confidence for s in repeated_result.suggestions if s.pitch_type == PitchType.FOUR_SEAM
    )
    assert fb_conf_with_history < fb_conf_without_history


# ---------------------------------------------------------------------------
# Validation — request model
# ---------------------------------------------------------------------------

def test_invalid_balls_above_max():
    resp = client.post("/api/v1/suggestions", json=_default_payload(balls=4))
    assert resp.status_code == 422


def test_invalid_strikes_above_max():
    resp = client.post("/api/v1/suggestions", json=_default_payload(strikes=3))
    assert resp.status_code == 422


def test_invalid_runner_base_number():
    resp = client.post(
        "/api/v1/suggestions",
        json=_default_payload(runners_on_base=[4]),
    )
    assert resp.status_code == 422


def test_invalid_inning_zero():
    resp = client.post("/api/v1/suggestions", json=_default_payload(inning=0))
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Suggestion endpoint — custom repertoire
# ---------------------------------------------------------------------------

def test_custom_repertoire_restricts_suggestions():
    payload = _default_payload()
    payload["available_pitch_types"] = ["fastball", "curveball"]
    resp = client.post("/api/v1/suggestions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    returned_types = {s["pitch_type"] for s in data["suggestions"]}
    assert returned_types <= {"fastball", "curveball"}


# ---------------------------------------------------------------------------
# Suggestion endpoint — handedness variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batter", ["left", "right", "switch"])
@pytest.mark.parametrize("pitcher", ["left", "right"])
def test_handedness_combinations(batter, pitcher):
    payload = _default_payload(batter_handedness=batter, pitcher_handedness=pitcher)
    resp = client.post("/api/v1/suggestions", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["suggestions"]) > 0


# ---------------------------------------------------------------------------
# Suggestion endpoint — all counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("balls,strikes", [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2),
])
def test_all_counts_return_suggestions(balls, strikes):
    resp = client.post("/api/v1/suggestions", json=_default_payload(balls=balls, strikes=strikes))
    assert resp.status_code == 200
    assert len(resp.json()["suggestions"]) > 0
