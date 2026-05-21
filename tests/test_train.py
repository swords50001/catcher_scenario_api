"""
Tests for train.py data-hygiene (clean_pitch_data).

Reproduces the real-world issue we diagnosed: the 2024 season was ingested
twice — each pitch appears once WITH plate_x/plate_z and once WITHOUT — and the
old pipeline filled the missing coordinates with the center of the zone.

train.py imports xgboost + supabase at module load, so skip cleanly if those
aren't installed in the current environment.
"""

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("supabase")

import numpy as np
import pandas as pd

from train import clean_pitch_data, NATURAL_KEY, get_location_zone, LOCATION_ZONES


def _row(game_pk, ab, pitch, plate_x, plate_z, pitch_label="Fastball"):
    return {
        "game_pk": game_pk,
        "at_bat_number": ab,
        "pitch_number": pitch,
        "plate_x": plate_x,
        "plate_z": plate_z,
        "pitch_label": pitch_label,
    }


def test_drops_null_coordinate_duplicates_keeps_coord_copy():
    # Each pitch appears twice: one coord-bearing, one null (the 2024 pattern).
    rows = []
    for ab in range(1, 6):
        for pn in range(1, 4):
            rows.append(_row(1, ab, pn, plate_x=0.5, plate_z=2.4))   # real
            rows.append(_row(1, ab, pn, plate_x=None, plate_z=None))  # dupe
    df = pd.DataFrame(rows)
    assert len(df) == 30

    clean = clean_pitch_data(df)

    # 15 unique pitches survive, all with real coordinates.
    assert len(clean) == 15
    assert clean["plate_x"].notna().all()
    assert clean["plate_z"].notna().all()
    # No duplicate natural keys remain.
    assert not clean.duplicated(subset=NATURAL_KEY).any()


def test_never_fills_missing_coords_with_center():
    # A pitch present ONLY as a null-coordinate row must be dropped, never
    # converted into a (0.0, 2.5) middle_middle pitch.
    rows = [
        _row(1, 1, 1, plate_x=-0.8, plate_z=3.1),   # tracked
        _row(1, 2, 1, plate_x=None, plate_z=None),  # untracked, no twin
    ]
    df = pd.DataFrame(rows)
    clean = clean_pitch_data(df)

    assert len(clean) == 1
    # The surviving row is the tracked one — center-of-zone was never invented.
    assert clean.iloc[0]["plate_x"] == pytest.approx(-0.8)
    assert not ((clean["plate_x"] == 0.0) & (clean["plate_z"] == 2.5)).any()


def test_coerces_string_coordinates():
    # Supabase JSON can deliver numbers as strings; they should coerce, not drop.
    rows = [
        _row(1, 1, 1, plate_x="0.42", plate_z="2.55"),
        _row(1, 1, 2, plate_x="garbage", plate_z="2.55"),  # unparseable → drop
    ]
    df = pd.DataFrame(rows)
    clean = clean_pitch_data(df)

    assert len(clean) == 1
    assert clean.iloc[0]["plate_x"] == pytest.approx(0.42)
    assert clean.iloc[0]["plate_z"] == pytest.approx(2.55)


def test_clean_data_passes_through_unchanged():
    # Already-clean, unique, fully-tracked data should survive intact.
    rows = [_row(1, ab, 1, plate_x=0.1 * ab, plate_z=2.0) for ab in range(1, 11)]
    df = pd.DataFrame(rows)
    clean = clean_pitch_data(df)
    assert len(clean) == 10


def test_dedupe_skipped_gracefully_without_key_columns():
    # If the natural-key columns are absent, dedupe is skipped (not crashed),
    # but the coordinate filter still applies.
    df = pd.DataFrame([
        {"plate_x": 0.5, "plate_z": 2.4, "pitch_label": "Slider"},
        {"plate_x": None, "plate_z": None, "pitch_label": "Slider"},
    ])
    clean = clean_pitch_data(df)
    assert len(clean) == 1


# ── get_location_zone: out-of-zone taxonomy ──────────────────────────────────

class TestLocationZone:
    def test_in_zone_unchanged(self):
        # Dead center, and a near-edge in-zone pitch, classify as before.
        assert get_location_zone(0.0, 2.5, "R") == "middle_middle"
        assert get_location_zone(0.75, 3.2, "R") == "up_away"
        assert get_location_zone(-0.75, 1.8, "R") == "low_in"

    def test_out_horizontal_rhh(self):
        # Way off the plate horizontally (RHH): inside vs away corners.
        assert get_location_zone(1.5, 2.6, "R") == "out_up_away"   # high & off-away
        assert get_location_zone(-1.5, 2.0, "R") == "out_low_in"   # low & off-inside

    def test_out_vertical(self):
        # Above the top / below the bottom of the zone, over the middle.
        assert get_location_zone(0.0, 4.0, "R") == "out_up_away"   # x==0 → away by convention
        assert get_location_zone(-0.1, 1.0, "R") == "out_low_in"   # slightly inside, in dirt

    def test_lefty_horizontal_flip(self):
        # plate_x = +1.5 is INSIDE to a lefty, so it must read as *_in, not *_away.
        assert get_location_zone(1.5, 2.6, "L") == "out_up_in"

    def test_all_returned_zones_are_known(self):
        for px in (-2.0, -0.8, -0.5, 0.0, 0.5, 0.8, 2.0):
            for pz in (0.8, 1.6, 2.5, 3.4, 4.2):
                for stand in ("L", "R"):
                    assert get_location_zone(px, pz, stand) in LOCATION_ZONES

    def test_taxonomy_has_thirteen_zones(self):
        assert len(LOCATION_ZONES) == 13
        assert sum(z.startswith("out_") for z in LOCATION_ZONES) == 4
