"""
Calculate 2025 batting averages directly from Supabase pitch data.
No Statcast pull needed — data is already there.
"""

import os
import math
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

print("📥 Loading 2025 pitches from Supabase...")
all_rows, page, page_size = [], 0, 1000
while True:
    resp = (
        sb.table("pitches")
        .select("player_name, events, game_date")
        .gte("game_date", "2025-01-01")
        .range(page * page_size, (page + 1) * page_size - 1)
        .execute()
    )
    if not resp.data:
        break
    all_rows.extend(resp.data)
    page += 1

print(f"   ✅ {len(all_rows):,} rows loaded.")

df = pd.DataFrame(all_rows)

# Filter to at-bat ending events only
hit_events = ["single", "double", "triple", "home_run"]
ab_events  = hit_events + [
    "strikeout", "field_out", "grounded_into_double_play",
    "force_out", "double_play", "fielders_choice_out",
    "fielders_choice", "strikeout_double_play", "other_out"
]

df_ab = df[df["events"].isin(ab_events)].copy()
df_ab["is_hit"] = df_ab["events"].isin(hit_events).astype(int)

ba = (
    df_ab.groupby("player_name")
    .agg(hits=("is_hit", "sum"), abs=("is_hit", "count"))
    .reset_index()
)
ba = ba[ba["abs"] >= 50]
ba["batter_avg"] = (ba["hits"] / ba["abs"]).round(3)

print(f"   ✅ Calculated BA for {len(ba)} batters.")

# Upload to Supabase
records = [
    {"season": 2025, "name": row["player_name"], "batter_avg": float(row["batter_avg"])}
    for _, row in ba.iterrows()
]

batch_size = 500
for i in range(math.ceil(len(records) / batch_size)):
    chunk = records[i * batch_size : (i + 1) * batch_size]
    sb.table("batters").upsert(chunk, on_conflict="season,name").execute()

print(f"✅ {len(records)} 2025 batting averages uploaded to Supabase.")