"""
fix_batting_avgs.py
====================
Calculate batting averages from Statcast data already in Supabase
and upload to the batters table — no FanGraphs needed.
"""

import os
import math
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd

load_dotenv()

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

print("📥 Loading pitch data from Supabase...")
all_rows, page, page_size = [], 0, 1000
while True:
    resp = (
        sb.table("pitches")
        .select("player_name, pitch_type")
        .range(page * page_size, (page + 1) * page_size - 1)
        .execute()
    )
    if not resp.data:
        break
    all_rows.extend(resp.data)
    page += 1

print(f"   ✅ {len(all_rows):,} rows loaded.")

# Pull at-bat results from Statcast using pybaseball instead
# We'll use a simpler proxy: estimate BA from Statcast events
print("📥 Pulling at-bat results from Statcast for BA calculation...")
from pybaseball import statcast
df = statcast(start_dt="2024-04-01", end_dt="2024-09-30")

# Filter to plate appearance ending events only
hit_events    = ["single", "double", "triple", "home_run"]
ab_events     = hit_events + ["strikeout", "field_out", "grounded_into_double_play",
                               "force_out", "double_play", "fielders_choice_out",
                               "fielders_choice", "strikeout_double_play",
                               "other_out"]

df_ab = df[df["events"].isin(ab_events)].copy()
df_ab["is_hit"] = df_ab["events"].isin(hit_events).astype(int)

ba = (
    df_ab.groupby("player_name")
    .agg(hits=("is_hit", "sum"), abs=("is_hit", "count"))
    .reset_index()
)
ba = ba[ba["abs"] >= 50]   # min 50 at-bats
ba["batter_avg"] = (ba["hits"] / ba["abs"]).round(3)

print(f"   ✅ Calculated BA for {len(ba)} batters.")

# Upload to Supabase
records = [
    {"season": 2024, "name": row["player_name"], "batter_avg": float(row["batter_avg"])}
    for _, row in ba.iterrows()
]

batch_size = 500
for i in range(math.ceil(len(records) / batch_size)):
    chunk = records[i * batch_size : (i + 1) * batch_size]
    sb.table("batters").upsert(chunk, on_conflict="season,name").execute()

print(f"✅ {len(records)} batting averages uploaded to Supabase.")
