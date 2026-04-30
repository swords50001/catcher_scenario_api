"""
Calculate 2025 batting averages from Statcast data in Supabase.
Uses release_speed as a proxy to identify actual pitches (not just events).
Since we don't have events column, we estimate BA from pybaseball directly
using only the at-bat results endpoint which is lighter than full statcast.
"""

import os
import math
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from pybaseball import batting_stats_bref

load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

print("📥 Pulling 2025 batting averages from Baseball Reference...")
# batting_stats_bref hits Baseball Reference, not FanGraphs — no 403 error
ba = batting_stats_bref(2025)
print(f"   ✅ {len(ba)} batters retrieved.")

# Baseball Reference uses 'BA' not 'AVG'
ba = ba[["Name", "BA"]].rename(columns={"BA": "batter_avg"})
ba["batter_avg"] = pd.to_numeric(ba["batter_avg"], errors="coerce")
ba = ba.dropna(subset=["batter_avg"])
ba = ba[ba["batter_avg"] > 0]
ba = ba.groupby("Name")["batter_avg"].mean().reset_index()

print(f"   ✅ {len(ba)} batters with valid BA.")

# Upload to Supabase
records = [
    {"season": 2025, "name": row["Name"], "batter_avg": float(row["batter_avg"])}
    for _, row in ba.iterrows()
]

batch_size = 500
for i in range(math.ceil(len(records) / batch_size)):
    chunk = records[i * batch_size : (i + 1) * batch_size]
    sb.table("batters").upsert(chunk, on_conflict="season,name").execute()

print(f"✅ {len(records)} 2025 batting averages uploaded to Supabase.")