"""
pipeline.py
============
Pull Statcast + batting data → clean → upload to Supabase.
Run this once to seed the DB, then on a schedule (e.g. nightly) to top up.

Requirements:
    pip install pybaseball supabase pandas numpy python-dotenv
"""

import os
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

from pybaseball import statcast, batting_stats

load_dotenv()  # reads SUPABASE_URL and SUPABASE_KEY from .env

# ─── Supabase client ────────────────────────────────────────────────────────
def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]   # use service-role key for writes
    return create_client(url, key)


# ─── Pitch type mapping ─────────────────────────────────────────────────────
PITCH_TYPE_MAP = {
    "FF": "Fastball", "SI": "Sinker",  "FC": "Cutter",
    "SL": "Slider",   "CU": "Curveball", "KC": "Curveball",
    "CH": "Changeup", "FS": "Splitter",  "ST": "Sweeper",
}

KEEP_COLS = [
    "game_pk", "game_date", "pitcher", "batter", "player_name",
    "pitch_type", "balls", "strikes", "on_1b", "on_2b", "on_3b",
    "inning", "outs_when_up", "stand", "p_throws",
    "home_score", "away_score", "release_speed",
    "at_bat_number", "pitch_number",
]


def clean_pitches(df: pd.DataFrame) -> pd.DataFrame:
    df = df[KEEP_COLS].copy()
    df["pitch_label"] = df["pitch_type"].map(PITCH_TYPE_MAP)
    df = df.dropna(subset=["pitch_label", "balls", "strikes"])

    # Drop rare labels
    counts = df["pitch_label"].value_counts(normalize=True)
    valid  = counts[counts >= 0.005].index
    df = df[df["pitch_label"].isin(valid)]

    # Binary base runners
    df["on_1b"] = df["on_1b"].notna().astype(int)
    df["on_2b"] = df["on_2b"].notna().astype(int)
    df["on_3b"] = df["on_3b"].notna().astype(int)

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)

    # Replace NaN with None for JSON serialisation
    df = df.where(pd.notna(df), None)
    return df


def upload_pitches(df: pd.DataFrame, supabase: Client,
                   batch_size: int = 1000) -> None:
    """Upsert pitches in batches (Supabase has a 1 MB payload limit)."""
    records = df.to_dict(orient="records")
    total   = len(records)
    batches = math.ceil(total / batch_size)
    print(f"   ⬆️  Uploading {total:,} pitches in {batches} batches ...")
    for i in range(batches):
        chunk = records[i * batch_size : (i + 1) * batch_size]
        supabase.table("pitches").upsert(chunk).execute()
        if (i + 1) % 10 == 0:
            print(f"      {(i+1)*batch_size:,} / {total:,} uploaded")
    print("   ✅ Pitches uploaded.\n")


def upload_batting_averages(ba_df: pd.DataFrame,
                             supabase: Client,
                             season: int) -> None:
    records = [
        {"season": season, "name": row["Name"], "batter_avg": float(row["batter_avg"])}
        for _, row in ba_df.iterrows()
        if not pd.isna(row["batter_avg"])
    ]
    supabase.table("batters").upsert(
        records, on_conflict="season,name"
    ).execute()
    print(f"   ✅ {len(records)} batter averages uploaded.\n")


# ─── Main pipeline ──────────────────────────────────────────────────────────

def run_pipeline(start_date: str, end_date: str, season: int):
    sb = get_supabase()

    print(f"📥 Pulling Statcast data {start_date} → {end_date} ...")
    raw = statcast(start_dt=start_date, end_dt=end_date)
    print(f"   {len(raw):,} rows pulled.")

    print("🧹 Cleaning ...")
    clean = clean_pitches(raw)
    print(f"   {len(clean):,} rows after cleaning.")

    upload_pitches(clean, sb)

    print("📥 Pulling batting averages ...")
    ba = batting_stats(season, qual=50)[["Name", "AVG"]].rename(
        columns={"AVG": "batter_avg"}
    )
    ba["batter_avg"] = pd.to_numeric(ba["batter_avg"], errors="coerce")
    upload_batting_averages(ba, sb, season)

    print("🎉 Pipeline complete.")


if __name__ == "__main__":
    run_pipeline(
        start_date="2024-04-01",
        end_date="2024-09-30",
        season=2024,
    )
