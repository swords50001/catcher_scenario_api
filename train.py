"""
train.py
=========
Load data from Supabase → engineer features → train XGBoost → save model.

Requirements:
    pip install supabase xgboost scikit-learn pandas numpy joblib python-dotenv
"""

import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, log_loss
import xgboost as xgb

load_dotenv()

MODEL_PATH    = "pitch_model.joblib"
ENCODER_PATH  = "label_encoder.joblib"
FEATURES_PATH = "feature_names.joblib"


# ─── Load from Supabase ─────────────────────────────────────────────────────

def load_from_supabase() -> pd.DataFrame:
    sb  = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    
    print("📥 Loading pitches from Supabase ...")
    # Keyset (cursor) pagination on the primary key. The old approach paged with
    # .range()/OFFSET, which forces Postgres to scan and discard every prior row
    # on each page — on a 2M-row table the deep pages exceed the statement
    # timeout (APIError 57014). Filtering `id > last_id` with an ORDER BY id uses
    # the PK index, so every page is fast regardless of depth.
    all_rows, page_size, last_id = [], 1000, 0
    while True:
        resp = (
            sb.table("pitches")
            .select("*")
            .gt("id", last_id)
            .order("id", desc=False)
            .limit(page_size)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_rows.extend(rows)
        last_id = rows[-1]["id"]
        print(f"   ... {len(all_rows):,} rows loaded", end="\r")
    df = pd.DataFrame(all_rows)
    print(f"\n   ✅ {len(df):,} pitches loaded.\n")

    print("📥 Loading batting averages from Supabase ...")
    ba_resp = sb.table("batters").select("name, batter_avg").execute()
    ba_df   = pd.DataFrame(ba_resp.data)
    ba_lookup = dict(zip(ba_df["name"], ba_df["batter_avg"].astype(float)))
    df["batter_avg"] = df["player_name"].map(ba_lookup).fillna(0.250)
    print(f"   ✅ Batting averages merged.\n")

    return df


# ─── Data hygiene ─────────────────────────────────────────────────────────────

# A pitch is uniquely identified within a game by this triplet.
NATURAL_KEY = ["game_pk", "at_bat_number", "pitch_number"]


def clean_pitch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data-hygiene pass that must run before feature engineering.

    Two problems this guards against (diagnosed May 2026):

      1. The 2024 season was ingested twice — once from a source carrying
         plate_x/plate_z and once without. Exactly half of all 2024 rows
         therefore have null coordinates, and each null row is an exact
         duplicate (on game_pk + at_bat_number + pitch_number) of a
         coordinate-bearing row. Confirmed: 686,824 duplicate pairs, zero
         singletons, zero over-duplication.

      2. The old pipeline filled missing coordinates with the center of the
         zone (0.0, 2.5 → middle_middle), silently relabeling every untracked
         pitch as a middle-middle strike — polluting loc_middle_middle and,
         because duplicates also inflated cumcount() and the expanding
         pitcher-tendency means, corrupting those features too.

    Fix: require real coordinates (never fabricate a location), then
    de-duplicate on the natural pitch key, keeping the coordinate-bearing copy.
    """
    start = len(df)
    print("🧹 Cleaning pitch data ...")

    # 1. Require real pitch coordinates — never fill to center-of-zone.
    df["plate_x"] = pd.to_numeric(df["plate_x"], errors="coerce")
    df["plate_z"] = pd.to_numeric(df["plate_z"], errors="coerce")
    has_coords = df["plate_x"].notna() & df["plate_z"].notna()
    n_no_coords = int((~has_coords).sum())
    df = df[has_coords].copy()
    print(f"   • Dropped {n_no_coords:,} rows missing pitch coordinates")

    # 2. De-duplicate on the natural pitch key. After the coordinate filter the
    #    null-coordinate duplicates are already gone; this also catches any
    #    other accidental repeats. Every surviving row has coordinates, so
    #    keep="first" is safe.
    missing_key = [c for c in NATURAL_KEY if c not in df.columns]
    if missing_key:
        print(f"   • ⚠️  Skipping dedupe — missing key column(s): {missing_key}")
    else:
        before = len(df)
        df = (
            df.sort_values(NATURAL_KEY)
              .drop_duplicates(subset=NATURAL_KEY, keep="first")
              .reset_index(drop=True)
        )
        print(f"   • Removed {before - len(df):,} duplicate pitch rows")

    print(f"   ✅ {start:,} → {len(df):,} clean pitches\n")
    return df


# ─── Location zone helper ────────────────────────────────────────────────────

def get_location_zone(plate_x: float, plate_z: float, stand: str) -> str:
    """
    Map plate_x / plate_z coordinates to a 9-zone grid.
    Statcast coords: plate_x is horizontal (-ve = catcher's left / inside to RHH)
                     plate_z is vertical (roughly 1.5 = low, 3.5 = high)

    Zones flip horizontally for LHH so in/out stay batter-relative.

    RHH view:                    LHH view (mirrored):
    up_in | up_middle | up_away  up_away | up_middle | up_in
    middle_in | ... | middle_away         ...flipped...
    low_in | low_middle | low_away low_away | low_middle | low_in
    """
    # Vertical zones
    if plate_z >= 3.0:      v_zone = "up"
    elif plate_z >= 2.0:    v_zone = "middle"
    else:                   v_zone = "low"

    # Horizontal zones — flip for lefties so in/out stay batter-relative
    # For RHH: negative plate_x = inside, positive = outside
    # For LHH: positive plate_x = inside, negative = outside
    if stand == "L":
        plate_x = -plate_x

    if plate_x <= -0.7:     h_zone = "in"
    elif plate_x >= 0.7:    h_zone = "away"
    else:                   h_zone = "middle"

    return f"{v_zone}_{h_zone}"


LOCATION_ZONES = [
    "up_in",      "up_middle",      "up_away",
    "middle_in",  "middle_middle",  "middle_away",
    "low_in",     "low_middle",     "low_away",
]


# ─── Pitcher tendency helper ─────────────────────────────────────────────────

def add_pitcher_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """Add each pitcher's historical pitch mix as features."""
    print("   Adding pitcher tendency features...")

    pitch_dummies = pd.get_dummies(df["pitch_label"], prefix="tends")
    df = pd.concat([df, pitch_dummies], axis=1)

    tend_cols = [c for c in df.columns if c.startswith("tends_")]

    df = df.sort_values(["game_date", "game_pk", "pitcher",
                         "at_bat_number", "pitch_number"])

    for col in tend_cols:
        df[f"pitcher_{col}"] = (
            df.groupby("pitcher")[col]
            .transform(lambda x: x.expanding().mean().shift(1))
            .fillna(0.125)   # default to uniform if no prior history
        )

    df = df.drop(columns=tend_cols)
    print(f"   ✅ Added {len(tend_cols)} pitcher tendency features.")
    return df


# ─── Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    for col in ["balls", "strikes", "on_1b", "on_2b", "on_3b",
                "inning", "outs_when_up", "home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── Count ────────────────────────────────────────────────────────────
    df["count_state"]  = df["balls"] * 3 + df["strikes"]
    
    def count_cat(row):
        b, s = row["balls"], row["strikes"]
        if s == 2:    return "two_strike"
        if b == 3:    return "three_ball"
        if b > s:     return "hitter_ahead"
        if s > b:     return "pitcher_ahead"
        return "even"

    df["count_category"] = df.apply(count_cat, axis=1)

    # ── Base runners ─────────────────────────────────────────────────────
    df["base_state"]   = df["on_1b"] * 4 + df["on_2b"] * 2 + df["on_3b"]
    df["runners_on"]   = ((df["on_1b"] + df["on_2b"] + df["on_3b"]) > 0).astype(int)
    df["scoring_pos"]  = ((df["on_2b"] + df["on_3b"]) > 0).astype(int)

    # ── Game context ─────────────────────────────────────────────────────
    df["score_diff"]   = (df["home_score"] - df["away_score"]).clip(-5, 5)
    df["late_inning"]  = (df["inning"] >= 7).astype(int)
    df["two_outs"]     = (df["outs_when_up"] == 2).astype(int)
    df["same_hand"]    = (df["stand"] == df["p_throws"]).astype(int)

    # ── Batter ───────────────────────────────────────────────────────────
    df["batter_avg"]   = pd.to_numeric(df["batter_avg"], errors="coerce").fillna(0.250)

    df["batter_avg_bucket"] = pd.cut(
        df["batter_avg"],
        bins=[0, .200, .230, .260, .290, .320, 1.0],
        labels=["sub200","200s","230s","260s","290s","300plus"]
    ).astype(str)

    # ── Pitcher fatigue ───────────────────────────────────────────────────
    df = df.sort_values(["game_date", "game_pk", "pitcher",
                         "at_bat_number", "pitch_number"])
    df["pitcher_pitch_count"] = (
        df.groupby(["game_pk", "pitcher"]).cumcount() + 1
    )
    df["pitcher_tired"] = (df["pitcher_pitch_count"] > 80).astype(int)

    # ── Pitch location zones ──────────────────────────────────────────────
    # Coordinates are cleaned in clean_pitch_data(). Coerce defensively, but
    # NEVER fill missing coords with the center of the zone — that silently
    # mislabels untracked pitches as middle_middle. Drop any stragglers.
    df["plate_x"] = pd.to_numeric(df["plate_x"], errors="coerce")
    df["plate_z"] = pd.to_numeric(df["plate_z"], errors="coerce")
    _missing = df["plate_x"].isna() | df["plate_z"].isna()
    if _missing.any():
        print(f"   ⚠️  Dropping {int(_missing.sum()):,} rows with unusable coordinates "
              f"(did clean_pitch_data run?)")
        df = df[~_missing].reset_index(drop=True)

    df["location_zone"] = df.apply(
        lambda row: get_location_zone(row["plate_x"], row["plate_z"], row["stand"]),
        axis=1
    )

    # One-hot encode the 9 zones
    for zone in LOCATION_ZONES:
        df[f"loc_{zone}"] = (df["location_zone"] == zone).astype(int)

    print(f"   ✅ Location zones engineered.")

    # ── Pitcher tendencies ────────────────────────────────────────────────
    df = add_pitcher_tendencies(df)

    print(f"   ✅ Features engineered.\n")
    return df


# ─── Model input ─────────────────────────────────────────────────────────────

BASE_FEATURES = [
    # Count
    "balls", "strikes", "count_state",
    # Base runners
    "on_1b", "on_2b", "on_3b", "base_state", "runners_on", "scoring_pos",
    # Game context
    "inning", "outs_when_up", "late_inning", "two_outs", "score_diff",
    # Matchup
    "same_hand",
    # Batter
    "batter_avg",
    # Pitcher
    "pitcher_pitch_count", "pitcher_tired",
    # Pitcher tendencies
    "pitcher_tends_Fastball", "pitcher_tends_Sinker",  "pitcher_tends_Slider",
    "pitcher_tends_Changeup", "pitcher_tends_Curveball","pitcher_tends_Cutter",
    "pitcher_tends_Splitter", "pitcher_tends_Sweeper",
    # Location zones
    "loc_up_in",     "loc_up_middle",     "loc_up_away",
    "loc_middle_in", "loc_middle_middle", "loc_middle_away",
    "loc_low_in",    "loc_low_middle",    "loc_low_away",
]

CAT_FEATURES = ["count_category", "batter_avg_bucket"]


def prepare_xy(df: pd.DataFrame):
    # Only encode CAT_FEATURES columns that actually exist
    existing_cats = [c for c in CAT_FEATURES if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=False)

    dummy_cols = [c for c in df.columns
                  if c.startswith("count_category_") or
                     c.startswith("batter_avg_bucket_")]

    # Only keep BASE_FEATURES that exist (guards against missing cols)
    valid_base   = [f for f in BASE_FEATURES if f in df.columns]
    all_features = valid_base + dummy_cols

    X  = df[all_features].astype(float)
    le = LabelEncoder()
    y  = le.fit_transform(df["pitch_label"])
    return X, y, le, all_features


# ─── Train ───────────────────────────────────────────────────────────────────

def train(X_train, y_train, num_classes, le):
    from sklearn.utils.class_weight import compute_sample_weight

    val_sz = int(len(X_train) * 0.1)
    X_tr, X_val = X_train.iloc[:-val_sz], X_train.iloc[-val_sz:]
    y_tr, y_val = y_train[:-val_sz], y_train[-val_sz:]

    # Mild inverse-frequency weighting — boosts rare pitches without killing Fastball
    class_counts = np.bincount(y_tr)
    total        = len(y_tr)
    raw_weights  = total / (num_classes * class_counts)
    mild_weights = np.sqrt(raw_weights)
    mild_weights = mild_weights / mild_weights.mean()

    print("   Class weights:")
    for cls, w in zip(le.classes_, mild_weights):
        print(f"      {cls}: {w:.3f}")

    sample_weights = mild_weights[y_tr]

    model = xgb.XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=15,
        subsample=0.8,
        colsample_bytree=0.75,
        gamma=0.5,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=40,
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_from_supabase()
    df = clean_pitch_data(df)
    df = engineer_features(df)

    # Chronological split — never let future games leak into training
    df = df.sort_values("game_date")
    cut = int(len(df) * 0.80)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    X_train, y_train, le, features = prepare_xy(train_df)
    X_test,  y_test,  *_           = prepare_xy(test_df)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print("🚀 Training XGBoost ...")
    model = train(X_train, y_train, num_classes=len(le.classes_), le=le)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = (y_pred == y_test).mean()
    ll  = log_loss(y_test, y_proba)
    print(f"\n✅ Accuracy: {acc:.4f} | Log-Loss: {ll:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    joblib.dump(model,    MODEL_PATH)
    joblib.dump(le,       ENCODER_PATH)
    joblib.dump(features, FEATURES_PATH)
    print(f"\n💾 Saved: {MODEL_PATH}, {ENCODER_PATH}, {FEATURES_PATH}")