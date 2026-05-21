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

from train import clean_pitch_data, NATURAL_KEY


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
