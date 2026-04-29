"""
api.py
=======
FastAPI server — exposes pitch prediction as a REST endpoint.
Your buddy's frontend hits POST /predict and gets back probabilities
+ a confidence score for whichever pitch the user selected.

Requirements:
    pip install fastapi uvicorn joblib xgboost scikit-learn pandas numpy python-dotenv

Run:
    uvicorn api:app --reload --port 8000
"""

import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# ─── Load model artifacts (trained by train.py) ─────────────────────────────
MODEL_PATH    = "pitch_model.joblib"
ENCODER_PATH  = "label_encoder.joblib"
FEATURES_PATH = "feature_names.joblib"

model        = joblib.load(MODEL_PATH)
label_enc    = joblib.load(ENCODER_PATH)
feature_cols = joblib.load(FEATURES_PATH)

PITCH_CLASSES = list(label_enc.classes_)   # e.g. ["Changeup","Curveball","Fastball",...]

# ─── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pitch Prediction API",
    description="Returns pitch-type probabilities + confidence score for a selected pitch.",
    version="1.0.0",
)

# Allow any origin so your buddy's frontend can call this freely during dev.
# Lock this down to specific origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ─────────────────────────────────────────────

class PitchRequest(BaseModel):
    # Count
    balls:   int = Field(..., ge=0, le=3, description="Balls in current count")
    strikes: int = Field(..., ge=0, le=2, description="Strikes in current count")

    # Base runners
    on_1b: int = Field(0, ge=0, le=1, description="Runner on 1st (0/1)")
    on_2b: int = Field(0, ge=0, le=1, description="Runner on 2nd (0/1)")
    on_3b: int = Field(0, ge=0, le=1, description="Runner on 3rd (0/1)")

    # Batter
    batter_avg: float = Field(0.260, ge=0.0, le=1.0,
                               description="Batter season batting average")

    # Game context (optional — defaults to neutral scenario)
    inning:              int   = Field(5,   ge=1, le=20)
    outs:                int   = Field(0,   ge=0, le=2)
    score_diff:          int   = Field(0,   description="home_score - away_score")
    pitcher_pitch_count: int   = Field(50,  ge=1, le=150)
    same_hand:           int   = Field(0,   ge=0, le=1,
                                        description="1 if pitcher/batter same hand")

    # The pitch the user selected in the frontend UI (optional)
    selected_pitch: Optional[str] = Field(
        None,
        description=f"Pitch the user chose. One of: {PITCH_CLASSES}"
    )


class PitchProbability(BaseModel):
    pitch_type:  str
    probability: float
    is_selected: bool


class PredictResponse(BaseModel):
    probabilities:     list[PitchProbability]
    top_pitch:         str
    top_probability:   float
    selected_pitch:    Optional[str]
    confidence_score:  Optional[float]
    confidence_label:  Optional[str]
    verdict:           Optional[str]    # "Correct" / "Acceptable" / "Incorrect"
    verdict_emoji:     Optional[str]    # ✅ / ⚠️ / ❌
    verdict_reason:    Optional[str]    # plain english explanation
    situation_summary: str


# ─── Feature builder ────────────────────────────────────────────────────────

def build_feature_row(req: PitchRequest) -> pd.DataFrame:
    b, s = req.balls, req.strikes

    # Count category
    if s == 2:          count_cat = "two_strike"
    elif b == 3:        count_cat = "three_ball"
    elif b > s:         count_cat = "hitter_ahead"
    elif s > b:         count_cat = "pitcher_ahead"
    else:               count_cat = "even"

    # BA bucket
    avg = req.batter_avg
    if avg < .200:      avg_bucket = "sub200"
    elif avg < .230:    avg_bucket = "200s"
    elif avg < .260:    avg_bucket = "230s"
    elif avg < .290:    avg_bucket = "260s"
    elif avg < .320:    avg_bucket = "290s"
    else:               avg_bucket = "300plus"

    row = {f: 0 for f in feature_cols}
    row.update({
        "balls":               b,
        "strikes":             s,
        "count_state":         b * 3 + s,
        "on_1b":               req.on_1b,
        "on_2b":               req.on_2b,
        "on_3b":               req.on_3b,
        "base_state":          req.on_1b * 4 + req.on_2b * 2 + req.on_3b,
        "runners_on":          int(req.on_1b or req.on_2b or req.on_3b),
        "scoring_pos":         int(req.on_2b or req.on_3b),
        "inning":              req.inning,
        "outs_when_up":        req.outs,
        "late_inning":         int(req.inning >= 7),
        "two_outs":            int(req.outs == 2),
        "score_diff":          max(-5, min(5, req.score_diff)),
        "same_hand":           req.same_hand,
        "batter_avg":          req.batter_avg,
        "pitcher_pitch_count": req.pitcher_pitch_count,
        "pitcher_tired":       int(req.pitcher_pitch_count > 80),
    })

    cc_key = f"count_category_{count_cat}"
    ab_key = f"batter_avg_bucket_{avg_bucket}"
    if cc_key in row: row[cc_key] = 1
    if ab_key in row: row[ab_key] = 1

    return pd.DataFrame([row])[feature_cols].astype(float)


def confidence_label(score: float) -> str:
    if score >= 0.55:   return "High"
    if score >= 0.35:   return "Medium"
    return "Low"


def get_verdict(selected_pitch: str,
                top_pitch: str,
                selected_prob: float,
                top_prob: float,
                sorted_probs: list) -> tuple:
    """
    Verdict logic:
    - Correct   → selected pitch IS the top predicted pitch
    - Acceptable → selected pitch is within 10% of the top pitch probability
    - Incorrect  → selected pitch is well below the top prediction
    """
    # Get rank of selected pitch (1 = best)
    ranked = [p for p, _ in sorted_probs]
    selected_rank = ranked.index(selected_pitch) + 1
    prob_gap = top_prob - selected_prob

    if selected_pitch == top_pitch:
        return (
            "Correct",
            "✅",
            f"{selected_pitch} is the model's top pick for this situation "
            f"at {selected_prob*100:.0f}% probability."
        )
    elif selected_rank == 2 or prob_gap <= 0.10:
        return (
            "Acceptable",
            "⚠️",
            f"{selected_pitch} is a reasonable call ({selected_prob*100:.0f}%), "
            f"though {top_pitch} is slightly more likely at {top_prob*100:.0f}%."
        )
    else:
        return (
            "Incorrect",
            "❌",
            f"{selected_pitch} is unlikely here ({selected_prob*100:.0f}%). "
            f"The model strongly favors {top_pitch} at {top_prob*100:.0f}% "
            f"for this situation."
        )


def situation_summary(req: PitchRequest) -> str:
    bases = []
    if req.on_1b: bases.append("1st")
    if req.on_2b: bases.append("2nd")
    if req.on_3b: bases.append("3rd")
    base_str = ", ".join(bases) if bases else "empty"
    return (
        f"{req.balls}-{req.strikes} count | "
        f"Bases: {base_str} | "
        f"Inning {req.inning} | "
        f"BA: {req.batter_avg:.3f}"
    )


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "available_pitch_types": PITCH_CLASSES}


@app.get("/pitch-types")
def get_pitch_types():
    """Return the list of pitch types the model knows about."""
    return {"pitch_types": PITCH_CLASSES}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PitchRequest):
    """
    Main endpoint.

    Send the current at-bat situation and (optionally) the pitch the user
    selected. Returns:
      - probabilities for every pitch type
      - top predicted pitch
      - confidence_score = probability of the selected pitch
    
    Example request body:
    {
        "balls": 2,
        "strikes": 1,
        "on_1b": 1,
        "on_2b": 0,
        "on_3b": 0,
        "batter_avg": 0.285,
        "inning": 7,
        "outs": 1,
        "score_diff": -1,
        "pitcher_pitch_count": 75,
        "same_hand": 0,
        "selected_pitch": "Fastball"
    }
    """
    # Validate selected pitch
    if req.selected_pitch and req.selected_pitch not in PITCH_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pitch '{req.selected_pitch}'. "
                   f"Valid options: {PITCH_CLASSES}"
        )

    X         = build_feature_row(req)
    probs_arr = model.predict_proba(X)[0]           # array of probabilities

    # Build sorted probability list
    prob_map = dict(zip(PITCH_CLASSES, probs_arr))
    sorted_probs = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)

    probabilities = [
        PitchProbability(
            pitch_type  = p,
            probability = round(float(v), 4),
            is_selected = (p == req.selected_pitch),
        )
        for p, v in sorted_probs
    ]

    top_pitch, top_prob = sorted_probs[0]

    # Confidence score for the user's selected pitch
    conf_score = None
    conf_label = None
    if req.selected_pitch:
        conf_score = round(float(prob_map[req.selected_pitch]), 4)
        conf_label = confidence_label(conf_score)

    # Verdict
    verdict, verdict_emoji, verdict_reason = (None, None, None)
    if req.selected_pitch:
        verdict, verdict_emoji, verdict_reason = get_verdict(
            selected_pitch = req.selected_pitch,
            top_pitch      = top_pitch,
            selected_prob  = float(prob_map[req.selected_pitch]),
            top_prob       = float(top_prob),
            sorted_probs   = sorted_probs,
        )

    return PredictResponse(
        probabilities     = probabilities,
        top_pitch         = top_pitch,
        top_probability   = round(float(top_prob), 4),
        selected_pitch    = req.selected_pitch,
        confidence_score  = conf_score,
        confidence_label  = conf_label,
        verdict           = verdict,
        verdict_emoji     = verdict_emoji,
        verdict_reason    = verdict_reason,
        situation_summary = situation_summary(req),
    )
    )


# ─── Dev server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
