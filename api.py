"""
api.py
=======
FastAPI server — exposes pitch prediction as a REST endpoint.
Your buddy's frontend hits POST /predict and gets back probabilities,
a confidence score, and a verdict for whichever pitch the user selected.

Requirements:
    pip install fastapi uvicorn joblib xgboost scikit-learn pandas numpy python-dotenv

Run:
    uvicorn api:app --reload --port 8000
"""

import os
import math
import random
import joblib
import warnings
warnings.filterwarnings("ignore")

import jwt as pyjwt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Literal

load_dotenv()

# ─── Load model artifacts (trained by train.py) ─────────────────────────────
MODEL_PATH    = "pitch_model.joblib"
ENCODER_PATH  = "label_encoder.joblib"
FEATURES_PATH = "feature_names.joblib"

model        = joblib.load(MODEL_PATH)
label_enc    = joblib.load(ENCODER_PATH)
feature_cols = joblib.load(FEATURES_PATH)

PITCH_CLASSES = list(label_enc.classes_)

LOCATION_ZONES = [
    # In-zone 3x3 grid
    "up_in",       "up_middle",     "up_away",
    "middle_in",   "middle_middle", "middle_away",
    "low_in",      "low_middle",    "low_away",
    # Out-of-zone corners (batter-relative). Must match train.py's LOCATION_ZONES.
    "out_up_in",   "out_up_away",   "out_low_in",   "out_low_away",
]

# Zones outside the strike zone — governed by the chase model, not the in-zone
# quality→weights layers.
OUT_OF_ZONE_ZONES = {"out_up_in", "out_up_away", "out_low_in", "out_low_away"}

# ─── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pitch Prediction API",
    description="Returns pitch-type probabilities, confidence score, and verdict for a selected pitch.",
    version="2.0.0",
)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = ["*"] if _raw_origins == "*" else [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Auth ────────────────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
_bearer = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="Server auth not configured")
    try:
        payload = pyjwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


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
    batter_avg: float = Field(0.260, ge=0.0, le=1.0, description="Batter season batting average")
    stand:      str   = Field("R", description="Batter handedness: L or R")

    # Game context
    inning:              int = Field(5,  ge=1,  le=20)
    outs:                int = Field(0,  ge=0,  le=2)
    score_diff:          int = Field(0,  description="home_score - away_score")
    pitcher_pitch_count: int = Field(50, ge=1,  le=150)
    same_hand:           int = Field(0,  ge=0,  le=1, description="1 if pitcher/batter same hand")

    # Pitcher tendency features — fraction of pitches historically thrown of each type.
    # Defaults to 0.125 (uniform prior, matching the training fallback).
    # Pass the pitcher's actual season mix when available for much better predictions.
    pitcher_tends_Fastball:  float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Sinker:    float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Slider:    float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Changeup:  float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Curveball: float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Cutter:    float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Splitter:  float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Sweeper:   float = Field(0.125, ge=0.0, le=1.0)

    # Pitch location zone
    location_zone: Optional[str] = Field(
        None,
        description=(
            "Zone the pitch was thrown to. One of: "
            "up_in, up_middle, up_away, "
            "middle_in, middle_middle, middle_away, "
            "low_in, low_middle, low_away"
        )
    )

    # The pitch the user selected in the frontend UI
    selected_pitch: Optional[str] = Field(
        None,
        description=f"Pitch the user chose. One of: {PITCH_CLASSES}"
    )


class LocationProbability(BaseModel):
    location_zone: str
    probability:   float


class PitchProbability(BaseModel):
    pitch_type:             str
    probability:            float
    is_selected:            bool
    location_probabilities: list[LocationProbability]
    most_likely_location:   str


class PredictResponse(BaseModel):
    probabilities:     list[PitchProbability]
    top_pitch:         str
    top_probability:   float
    selected_pitch:    Optional[str]
    confidence_score:  Optional[float]
    confidence_label:  Optional[str]
    verdict:           Optional[str]
    verdict_emoji:     Optional[str]
    verdict_reason:    Optional[str]
    situation_summary: str


# ─── Helpers ────────────────────────────────────────────────────────────────

def confidence_label(score: float) -> str:
    if score >= 0.55:   return "High"
    if score >= 0.35:   return "Medium"
    return "Low"


def get_verdict(
    selected_pitch: str,
    top_pitch: str,
    selected_prob: float,
    top_prob: float,
    sorted_probs: list,
) -> tuple:
    """
    Correct    → selected pitch IS the model's top pick
    Acceptable → selected pitch is ranked #2 OR within 10% of top probability
    Incorrect  → selected pitch is well below the top prediction
    """
    ranked = [p for p, _ in sorted_probs]
    selected_rank = ranked.index(selected_pitch) + 1
    prob_gap = top_prob - selected_prob

    if selected_pitch == top_pitch:
        return (
            "Correct",
            "✅",
            f"{selected_pitch} is the model's top pick for this situation "
            f"at {selected_prob * 100:.0f}% probability."
        )
    elif selected_rank == 2 or prob_gap <= 0.10:
        return (
            "Acceptable",
            "⚠️",
            f"{selected_pitch} is a reasonable call ({selected_prob * 100:.0f}%), "
            f"though {top_pitch} is slightly more likely at {top_prob * 100:.0f}%."
        )
    else:
        return (
            "Incorrect",
            "❌",
            f"{selected_pitch} is unlikely here ({selected_prob * 100:.0f}%). "
            f"The model strongly favors {top_pitch} at {top_prob * 100:.0f}% "
            f"for this situation."
        )


def situation_summary(req: PitchRequest) -> str:
    bases = []
    if req.on_1b: bases.append("1st")
    if req.on_2b: bases.append("2nd")
    if req.on_3b: bases.append("3rd")
    base_str  = ", ".join(bases) if bases else "empty"
    zone_str  = req.location_zone or "not specified"
    return (
        f"{req.balls}-{req.strikes} count | "
        f"Bases: {base_str} | "
        f"Inning {req.inning} | "
        f"BA: {req.batter_avg:.3f} | "
        f"Location: {zone_str}"
    )


def build_feature_row(req: PitchRequest) -> pd.DataFrame:
    b, s = req.balls, req.strikes

    # Count category
    if s == 2:      count_cat = "two_strike"
    elif b == 3:    count_cat = "three_ball"
    elif b > s:     count_cat = "hitter_ahead"
    elif s > b:     count_cat = "pitcher_ahead"
    else:           count_cat = "even"

    # BA bucket
    avg = req.batter_avg
    if avg < .200:      avg_bucket = "sub200"
    elif avg < .230:    avg_bucket = "200s"
    elif avg < .260:    avg_bucket = "230s"
    elif avg < .290:    avg_bucket = "260s"
    elif avg < .320:    avg_bucket = "290s"
    else:               avg_bucket = "300plus"

    # Start with all features zeroed out
    row = {f: 0 for f in feature_cols}

    # Core features
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
        # Pitcher tendencies — use caller-supplied values (default 0.125 = uniform prior)
        "pitcher_tends_Fastball":  req.pitcher_tends_Fastball,
        "pitcher_tends_Sinker":    req.pitcher_tends_Sinker,
        "pitcher_tends_Slider":    req.pitcher_tends_Slider,
        "pitcher_tends_Changeup":  req.pitcher_tends_Changeup,
        "pitcher_tends_Curveball": req.pitcher_tends_Curveball,
        "pitcher_tends_Cutter":    req.pitcher_tends_Cutter,
        "pitcher_tends_Splitter":  req.pitcher_tends_Splitter,
        "pitcher_tends_Sweeper":   req.pitcher_tends_Sweeper,
    })

    # Count category dummy
    cc_key = f"count_category_{count_cat}"
    if cc_key in row:
        row[cc_key] = 1

    # BA bucket dummy
    ab_key = f"batter_avg_bucket_{avg_bucket}"
    if ab_key in row:
        row[ab_key] = 1

    # Location zone one-hot
    if req.location_zone and req.location_zone in LOCATION_ZONES:
        loc_key = f"loc_{req.location_zone}"
        if loc_key in row:
            row[loc_key] = 1

    return pd.DataFrame([row])[feature_cols].astype(float)


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "available_pitch_types": PITCH_CLASSES,
        "available_location_zones": LOCATION_ZONES,
    }


@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    """Returns the authenticated user's identity from their Supabase JWT."""
    return {
        "user_id": user.get("sub"),
        "email":   user.get("email"),
        "role":    user.get("role"),
    }


@app.get("/pitch-types")
def get_pitch_types():
    """Return pitch types and location zones the model knows about."""
    return {
        "pitch_types":      PITCH_CLASSES,
        "location_zones":   LOCATION_ZONES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PitchRequest, _user=Depends(get_current_user)):
    """
    Main prediction endpoint.

    Send the current at-bat situation and (optionally) the pitch + location
    the user selected. Returns probabilities for every pitch type plus a
    confidence score and verdict for the selected pitch.

    Example request:
    {
        "balls": 0,
        "strikes": 2,
        "on_1b": 1,
        "on_2b": 0,
        "on_3b": 0,
        "batter_avg": 0.285,
        "stand": "R",
        "inning": 7,
        "outs": 1,
        "score_diff": -1,
        "pitcher_pitch_count": 75,
        "same_hand": 0,
        "location_zone": "low_away",
        "selected_pitch": "Slider"
    }
    """
    # Validate selected pitch
    if req.selected_pitch and req.selected_pitch not in PITCH_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pitch '{req.selected_pitch}'. Valid options: {PITCH_CLASSES}"
        )

    # Validate location zone
    if req.location_zone and req.location_zone not in LOCATION_ZONES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown location zone '{req.location_zone}'. Valid options: {LOCATION_ZONES}"
        )

    X         = build_feature_row(req)
    probs_arr = model.predict_proba(X)[0]

    prob_map     = dict(zip(PITCH_CLASSES, probs_arr))
    sorted_probs = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)

    # Build a (9 zones × num_pitch_types) matrix of P(pitch_type | situation, zone)
    # by re-running the model with location_zone swapped to each of the 9 zones.
    # Then for each pitch type, normalize across zones to get
    # P(zone | pitch_type, situation) under a uniform prior over zones.
    zone_request_dicts = []
    base_dict = req.model_dump()
    for zone in LOCATION_ZONES:
        d = dict(base_dict)
        d["location_zone"] = zone
        zone_request_dicts.append(PitchRequest(**d))

    zone_X     = pd.concat(
        [build_feature_row(zr) for zr in zone_request_dicts],
        ignore_index=True,
    )
    zone_probs = model.predict_proba(zone_X)  # shape: (9, num_pitch_types)

    # Per-pitch location distributions (cols = pitch types, rows = zones)
    pitch_to_zone_probs: dict = {}
    for pitch_idx, pitch_name in enumerate(PITCH_CLASSES):
        col = zone_probs[:, pitch_idx]
        total = float(col.sum())
        if total > 0:
            normalized = col / total
        else:
            normalized = np.full_like(col, 1.0 / len(LOCATION_ZONES))
        pitch_to_zone_probs[pitch_name] = normalized

    probabilities = []
    for p, v in sorted_probs:
        zone_dist = pitch_to_zone_probs[p]
        loc_probs = [
            LocationProbability(
                location_zone = zone,
                probability   = round(float(zone_dist[i]), 4),
            )
            for i, zone in enumerate(LOCATION_ZONES)
        ]
        most_likely_zone = LOCATION_ZONES[int(np.argmax(zone_dist))]
        probabilities.append(
            PitchProbability(
                pitch_type             = p,
                probability            = round(float(v), 4),
                is_selected            = (p == req.selected_pitch),
                location_probabilities = loc_probs,
                most_likely_location   = most_likely_zone,
            )
        )

    top_pitch, top_prob = sorted_probs[0]

    # Confidence score for selected pitch
    conf_score = None
    conf_label = None
    if req.selected_pitch:
        conf_score = round(float(prob_map[req.selected_pitch]), 4)
        conf_label = confidence_label(conf_score)

    # Verdict
    verdict, verdict_emoji, verdict_reason = None, None, None
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


# ════════════════════════════════════════════════════════════════════════════
# NEW: /api/v1/evaluate — server-authoritative pitch evaluation.
#
# Mirrors the Swift algorithms documented in pitch-iq-audit.md §2.1–§2.3 and
# §2.6. Quality scoring, outcome simulation, feedback text, and the coach hint
# are now produced here so iOS, web, and Android clients see identical results.
# The existing /predict endpoint is kept untouched for back-compat.
# ════════════════════════════════════════════════════════════════════════════

MODEL_VERSION = "1.0.0"

# 19 possible outcomes. (The audit text says 18; the enumeration it lists is
# actually 19 entries. We include all of them and gate strikeout/walk on count.)
ALL_OUTCOMES = [
    "called_strike", "swinging_strike", "ball", "foul_ball",
    "strikeout_looking", "strikeout_swinging", "walk", "hit_by_pitch",
    "ground_out", "fly_out", "line_out", "pop_out",
    "double_play", "fielders_choice", "sacrifice_fly",
    "single", "double", "triple", "home_run",
]

OUTCOME_DISPLAY_NAMES = {
    "called_strike":       "Called Strike",
    "swinging_strike":     "Swinging Strike",
    "ball":                "Ball",
    "foul_ball":           "Foul Ball",
    "strikeout_looking":   "Strikeout (Looking)",
    "strikeout_swinging":  "Strikeout (Swinging)",
    "walk":                "Walk",
    "hit_by_pitch":        "Hit By Pitch",
    "ground_out":          "Ground Out",
    "fly_out":             "Fly Out",
    "line_out":            "Line Out",
    "pop_out":             "Pop Out",
    "double_play":         "Double Play",
    "fielders_choice":     "Fielder's Choice",
    "sacrifice_fly":       "Sacrifice Fly",
    "single":              "Single",
    "double":              "Double",
    "triple":              "Triple",
    "home_run":            "Home Run",
}

# Quality tier thresholds (audit §2.1)
QUALITY_TIERS = [
    (0.85, "elite"),
    (0.70, "strong"),
    (0.55, "good"),
    (0.40, "neutral"),
    (0.25, "poor"),
    (0.00, "terrible"),
]

# Pitch-type families. Match case-insensitively against PITCH_CLASSES so this
# stays correct whatever spellings the trained label encoder uses.
GROUND_BALL_PITCH_NAMES = {"sinker", "2-seam fastball", "two-seam fastball",
                          "slider", "changeup", "splitter"}
FASTBALL_NAMES   = {"fastball", "4-seam fastball", "four-seam fastball",
                    "sinker", "2-seam fastball", "two-seam fastball", "cutter"}
BREAKING_NAMES   = {"curveball", "slider", "knuckle curve", "sweeper"}
OFFSPEED_NAMES   = {"changeup", "splitter", "screwball"}


def quality_tier(score: float) -> str:
    for threshold, name in QUALITY_TIERS:
        if score >= threshold:
            return name
    return "terrible"


def normalize_pitch_name(name: str) -> str:
    return name.strip().lower() if isinstance(name, str) else ""


# ─── Request / Response schemas for /api/v1/evaluate ────────────────────────

class EvaluateScenario(BaseModel):
    balls:                int   = Field(..., ge=0, le=3)
    strikes:              int   = Field(..., ge=0, le=2)
    outs:                 int   = Field(0,   ge=0, le=2)
    inning:               int   = Field(1,   ge=1, le=20)
    runners_on_base:      list[int] = Field(
        default_factory=list,
        description="Which bases are occupied. Subset of [1, 2, 3].",
    )
    batter_handedness:    Literal["left", "right"] = "right"
    pitcher_handedness:   Literal["left", "right"] = "right"
    batter_avg:           float = Field(0.260, ge=0.0, le=1.0)
    score_diff:           int   = 0
    pitcher_pitch_count:  int   = Field(50, ge=1, le=150)
    previous_pitches:     list[str] = Field(default_factory=list)
    # Pitcher tendency mix — fraction of pitches historically thrown of each type.
    # Defaults to 0.125 (uniform prior). Pass real season mix for better predictions.
    pitcher_tends_Fastball:  float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Sinker:    float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Slider:    float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Changeup:  float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Curveball: float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Cutter:    float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Splitter:  float = Field(0.125, ge=0.0, le=1.0)
    pitcher_tends_Sweeper:   float = Field(0.125, ge=0.0, le=1.0)


class EvaluateSelection(BaseModel):
    pitch_type: str
    location:   str = Field(..., description=f"One of: {LOCATION_ZONES}")


class EvaluateContext(BaseModel):
    mode:      Literal["game", "coach", "practice"] = "game"
    player_id: Optional[str] = None
    # Optional seed so clients (and tests) can reproduce an outcome roll.
    random_seed: Optional[int] = None
    # The player's available pitches this game. None = full arsenal (all types).
    # The model still predicts over all pitch types; we mask + renormalize to
    # this subset so quality/verdict/feedback are relative to what they can throw.
    arsenal: Optional[list[str]] = Field(
        None,
        description=f"Pitches the player can throw (subset of {PITCH_CLASSES}). "
                    f"None means the full arsenal.",
    )


class EvaluateRequest(BaseModel):
    scenario:  EvaluateScenario
    selection: EvaluateSelection
    context:   EvaluateContext = Field(default_factory=EvaluateContext)


class ProbabilityEntry(BaseModel):
    pitch_type:  str
    probability: float
    rank:        int


class EvaluationBlock(BaseModel):
    quality_score:              float
    quality_tier:               str
    rating:                     Literal["excellent", "good", "acceptable", "poor"]
    probabilities:              list[ProbabilityEntry]
    top_pitch:                  str
    top_probability:            float
    selected_pitch_probability: float
    verdict:                    str
    verdict_reason:             str


class OutcomeBlock(BaseModel):
    result:           str
    display_name:     str
    is_terminal:      bool
    is_out:           bool
    is_hit:           bool
    is_strike:        bool
    is_ball:          bool
    outs_recorded:    int
    bases_advanced:   int


class FeedbackBlock(BaseModel):
    pitch_assessment:    str
    location_assessment: str
    combined_assessment: str
    recommended_pitch:   Optional[str]
    coaching_hint:       Optional[str]
    # Set only when the unrestricted model's top pitch is OUTSIDE the player's
    # arsenal — i.e., a pro would throw something the player can't (yet) throw.
    pro_pitch:           Optional[str] = None
    pro_tip:             Optional[str] = None


class SituationBlock(BaseModel):
    tags:    list[str]
    summary: str


class EvaluateMeta(BaseModel):
    model_version:    str
    response_time_ms: int


class EvaluateResponse(BaseModel):
    evaluation: EvaluationBlock
    outcome:    OutcomeBlock
    feedback:   FeedbackBlock
    situation:  SituationBlock
    meta:       EvaluateMeta


# ─── PitchQualityCalculator (audit §2.1) ────────────────────────────────────

def _gap_score(gap: float) -> float:
    """Component 3 — penalty based on probability gap from top pick."""
    if gap <= 0.0:    return 0.10
    if gap <  0.03:   return 0.05
    if gap <  0.08:   return 0.00
    if gap <  0.15:   return -0.05
    # gap >= 0.15: scale from -0.10 (at 0.15) toward -0.15 (at 1.00), capped.
    scaled = (gap - 0.15) / max(1.0 - 0.15, 1e-6)
    return max(-0.15, -0.10 - scaled * 0.05)


def _rank_score(rank: int) -> float:
    """Component 2 — rank bonus/penalty. rank is 1-indexed."""
    if rank == 1: return 0.15
    if rank == 2: return 0.05
    if rank == 3: return 0.00
    # rank 4+
    return max(-0.10, -0.05 * (rank - 3))


def _verdict_modifier(verdict: str) -> float:
    """Component 4 — direct from API verdict string."""
    v = (verdict or "").lower()
    if v == "correct":   return 0.10
    if v == "incorrect": return -0.10
    return 0.0


def _normalized_entropy(probs: list[float]) -> float:
    """Shannon entropy of `probs`, normalized to [0, 1] via log2(count)."""
    n = len([p for p in probs if p > 0])
    if n <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs if p > 0)
    return h / math.log2(n)


def _location_is_middle(loc: str) -> bool:
    return loc in {"middle_in", "middle_middle", "middle_away"}


def compute_pitch_quality(
    sorted_probs:   list[tuple[str, float]],   # [(pitch, prob), ...] sorted desc
    selected_pitch: str,
    verdict:        str,
    scenario:       EvaluateScenario,
    selection:      EvaluateSelection,
) -> tuple[float, str]:
    """
    Implements the 6-component additive quality score from audit §2.1.

    Returns (score in [0,1], tier label).
    """
    prob_map = dict(sorted_probs)
    top_pitch, top_prob = sorted_probs[0]
    selected_prob = prob_map.get(selected_pitch, 0.0)

    # rank: 1-indexed position of selected pitch (worst case: count + 1)
    pitches = [p for p, _ in sorted_probs]
    rank = pitches.index(selected_pitch) + 1 if selected_pitch in pitches \
           else len(pitches) + 1

    # Component 1: base selection (0.25–0.60)
    if top_prob > 0:
        base = 0.25 + (selected_prob / top_prob) * 0.35
    else:
        base = 0.25
    base = min(0.60, max(0.25, base))

    # Component 2: rank
    rank_s = _rank_score(rank)

    # Component 3: gap
    gap = max(0.0, top_prob - selected_prob)
    gap_s = _gap_score(gap)

    # Component 4: verdict
    verdict_s = _verdict_modifier(verdict)

    # Component 5: distribution entropy — only applied when selected isn't top
    # (the audit explicitly frames it as adjusting the penalty for non-top picks).
    if selected_pitch == top_pitch:
        entropy_s = 0.0
    else:
        norm_h = _normalized_entropy([p for _, p in sorted_probs])
        entropy_s = (norm_h - 0.5) * 0.10  # range -0.05..+0.05

    # Component 6: context modifiers
    # `quality_indicator` interpreted as running quality so far clamped to [0,1]
    running = base + rank_s + gap_s + verdict_s + entropy_s
    quality_indicator = max(0.0, min(1.0, running))
    direction = (quality_indicator - 0.5) * 0.04

    context_s = 0.0
    is_full_count   = scenario.balls == 3 and scenario.strikes == 2
    is_risp         = 2 in scenario.runners_on_base or 3 in scenario.runners_on_base
    is_dangerous    = scenario.batter_avg > 0.300
    is_poor_pitch   = selected_prob < 0.50
    is_middle_loc   = _location_is_middle(selection.location)

    if is_full_count:                       context_s += direction
    if is_risp:                             context_s += direction
    if is_dangerous and is_poor_pitch:      context_s -= 0.02
    if is_middle_loc and is_poor_pitch:     context_s -= 0.02
    # cap context_s to declared range
    context_s = max(-0.05, min(0.05, context_s))

    # Count-aware out-of-zone intent (docs/out-of-zone-chase-model.md §9):
    # a chase pitch with two strikes is smart pitching; wasting one while ahead
    # in the count is not. Keeps quality consistent with the chase outcome model.
    intent_s = 0.0
    if selection.location in OUT_OF_ZONE_ZONES:
        if scenario.strikes == 2 and scenario.balls < 3:
            intent_s += 0.03      # 0-2, 1-2, 2-2 — expand the zone, get the chase
        elif (scenario.balls, scenario.strikes) in {(2, 0), (3, 0), (3, 1)}:
            intent_s -= 0.04      # ahead in the count — wasting a pitch / risking a walk

    raw = base + rank_s + gap_s + verdict_s + entropy_s + context_s + intent_s
    score = max(0.0, min(1.0, raw))
    return score, quality_tier(score)


# ─── FeedbackGenerator (audit §2.2) ─────────────────────────────────────────

def compute_rating(selected_prob: float, top_prob: float, is_top: bool) -> str:
    if is_top:
        return "excellent"
    gap = top_prob - selected_prob
    if gap < 0.03: return "good"
    if gap < 0.10: return "acceptable"
    return "poor"


def _count_label(balls: int, strikes: int) -> str:
    return f"{balls}-{strikes}"


def _location_phrase(loc: str) -> str:
    parts = loc.split("_")
    if len(parts) == 2:
        v, h = parts
        v_word = {"up": "up", "middle": "middle", "low": "down"}.get(v, v)
        h_word = {"in": "and in", "middle": "in the middle", "away": "and away"}.get(h, h)
        return f"{v_word} {h_word}".strip()
    return loc.replace("_", " ")


def _assess_pitch_type(
    selected_pitch: str,
    top_pitch:      str,
    selected_prob:  float,
    top_prob:       float,
    in_distribution: bool,
    count_label:    str,
) -> str:
    if selected_pitch == top_pitch:
        return (f"Great pitch selection. {selected_pitch} had the highest probability "
                f"at {selected_prob * 100:.0f}% in a {count_label} count.")
    if in_distribution:
        return (f"{selected_pitch} had {selected_prob * 100:.0f}% probability in this "
                f"{count_label} count. The model favored {top_pitch} at "
                f"{top_prob * 100:.0f}%.")
    return (f"{selected_pitch} wasn't in the model's distribution. {top_pitch} was "
            f"the top choice at {top_prob * 100:.0f}%.")


def _assess_location(location: str) -> str:
    return f"Targeting {_location_phrase(location)}."


def _assess_combination(
    rating:     str,
    top_pitch:  str,
    count_label: str,
    is_pitcher_count: bool,
    is_risp:    bool,
) -> str:
    if rating == "excellent":
        if is_pitcher_count:
            return (f"That's the pitch to throw in a {count_label} count. "
                    f"You're thinking like a catcher.")
        if is_risp:
            return "Right pitch with runners in scoring position. Smart call."
        return "Exactly what a catcher should call here. Right pitch for the situation."
    if rating == "good":
        return (f"Strong pitch call in a {count_label} count. "
                f"Very close to the optimal choice.")
    if rating == "acceptable":
        return (f"Not a bad call, but {top_pitch} gives you a better chance in this "
                f"{count_label} count.")
    # poor
    return f"{top_pitch} is the stronger play here in a {count_label} count."


def generate_feedback(
    scenario:       EvaluateScenario,
    selection:      EvaluateSelection,
    sorted_probs:   list[tuple[str, float]],
    rating:         str,
    coaching_hint:  Optional[str],
    pro_pitch:      Optional[str] = None,
    pro_prob:       float = 0.0,
) -> FeedbackBlock:
    prob_map = dict(sorted_probs)
    top_pitch, top_prob = sorted_probs[0]
    selected_prob = prob_map.get(selection.pitch_type, 0.0)
    in_distribution = selection.pitch_type in prob_map

    is_pitcher_count = (scenario.balls, scenario.strikes) in {(0, 2), (1, 2)}
    is_risp = 2 in scenario.runners_on_base or 3 in scenario.runners_on_base
    count_label = _count_label(scenario.balls, scenario.strikes)

    pitch_assessment = _assess_pitch_type(
        selection.pitch_type, top_pitch, selected_prob, top_prob,
        in_distribution, count_label,
    )
    location_assessment = _assess_location(selection.location)
    combined_assessment = _assess_combination(
        rating, top_pitch, count_label, is_pitcher_count, is_risp,
    )

    recommended_pitch = top_pitch if rating == "poor" else None

    # "What a pro would throw" note — only when the unrestricted model favored a
    # pitch the player doesn't have in their arsenal. Turns a gap into a tip.
    pro_tip = None
    if pro_pitch:
        pro_tip = (f"A pitcher with a full arsenal might go {pro_pitch} here "
                   f"({pro_prob * 100:.0f}%) — one to consider adding.")

    return FeedbackBlock(
        pitch_assessment    = pitch_assessment,
        location_assessment = location_assessment,
        combined_assessment = combined_assessment,
        recommended_pitch   = recommended_pitch,
        coaching_hint       = coaching_hint,
        pro_pitch           = pro_pitch,
        pro_tip             = pro_tip,
    )


# ─── OutcomeEngine (audit §2.3) ─────────────────────────────────────────────

def _base_outcome_weights(quality: float) -> dict[str, float]:
    """Layer 1 — quality-dependent base weights summed by category, then
    distributed across the outcomes in each category."""
    strike_rate = 0.15 + quality * 0.40   # 0.15 .. 0.55
    ball_rate   = 0.30 - quality * 0.22   # 0.30 .. 0.08
    out_rate    = 0.05 + quality * 0.20   # 0.05 .. 0.25
    hit_rate    = 0.30 - quality * 0.25   # 0.30 .. 0.05
    extra_boost = max(0.0, 1.0 - quality) # poor pitches → more extra-base hits

    # Within-category splits. The audit specifies only the category totals; the
    # within-category ratios below are reasonable defaults — adjustable here
    # without an app release.
    w: dict[str, float] = {o: 0.0 for o in ALL_OUTCOMES}

    # Strikes
    w["called_strike"]    = strike_rate * 0.45
    w["swinging_strike"]  = strike_rate * 0.35
    w["foul_ball"]        = strike_rate * 0.20

    # Balls
    w["ball"]             = ball_rate * 0.95
    w["hit_by_pitch"]     = ball_rate * 0.05

    # Outs
    w["ground_out"]       = out_rate * 0.35
    w["fly_out"]          = out_rate * 0.30
    w["line_out"]         = out_rate * 0.15
    w["pop_out"]          = out_rate * 0.20

    # Hits — extra-base boost amplifies double/triple/HR for poor pitches
    eb = 1.0 + extra_boost * 0.5
    w["single"]           = hit_rate * 0.70
    w["double"]           = hit_rate * 0.18 * eb
    w["triple"]           = hit_rate * 0.04 * eb
    w["home_run"]         = hit_rate * 0.08 * eb

    return w


def _apply_count_modifiers(w: dict[str, float], balls: int, strikes: int,
                           quality: float) -> None:
    """Layer 2 — count modifiers."""
    pitcher_counts = {(0, 2), (1, 2)}
    hitter_counts  = {(2, 0), (2, 1), (3, 0), (3, 1)}
    count = (balls, strikes)

    if count in pitcher_counts:
        w["swinging_strike"] *= 1.4
        w["called_strike"]   *= 1.2
        w["single"]          *= 0.7
        w["double"]          *= 0.7

    if count in hitter_counts:
        quality_penalty = 1.0 + (1.0 - quality) * 0.5
        w["ball"]            *= 1.3
        w["single"]          *= 1.2 * quality_penalty
        w["home_run"]        *= 1.2 * quality_penalty
        w["swinging_strike"] *= 0.6

    if count == (3, 2):  # full count
        w["foul_ball"] *= 1.5
        w["single"]    *= 1.1

    if count == (3, 0):
        w["ball"]            *= 1.8
        w["called_strike"]   *= 1.3
        w["swinging_strike"] *= 0.2


def _apply_location_modifiers(w: dict[str, float], loc: str) -> None:
    """Layer 3 — zone-based modifiers (audit §2.3)."""
    # Vertical
    if loc.startswith("low_"):
        w["ground_out"]      *= 1.4
        w["swinging_strike"] *= 1.2
        w["home_run"]        *= 0.5
    elif loc.startswith("up_"):
        w["fly_out"]    *= 1.3
        w["pop_out"]    *= 1.3
        w["ground_out"] *= 0.6

    # Horizontal
    if loc.endswith("_in"):
        w["ground_out"] *= 1.2
        w["foul_ball"]  *= 1.3
    elif loc.endswith("_away"):
        w["swinging_strike"] *= 1.2
        w["ground_out"]      *= 1.1

    # "Middle of plate" — meatball: both axes middle
    if loc == "middle_middle":
        w["single"]        *= 1.5
        w["double"]        *= 1.6
        w["home_run"]      *= 1.8
        w["called_strike"] *= 0.8


def _apply_batter_modifiers(w: dict[str, float], batter_avg: float,
                            quality: float) -> None:
    """Layer 4."""
    deviation     = batter_avg - 0.260
    quality_factor = 1.0 + (1.0 - quality) * 0.5
    hit_mult       = max(0.75, min(1.25,
                                   1.0 + deviation * 3.0 * quality_factor))
    for k in ("single", "double", "triple", "home_run"):
        w[k] *= hit_mult

    if deviation < 0:  # weak hitter
        boost = 1.0 + abs(deviation) * 2.0
        w["swinging_strike"] *= boost


def _apply_runner_modifiers(w: dict[str, float], runners: list[int], outs: int) -> None:
    """Layer 5 — double-play and sacrifice-fly setups."""
    if 1 in runners and outs < 2:
        w["double_play"] = w["ground_out"] * 0.30
    if 3 in runners and outs < 2:
        w["sacrifice_fly"] = w["fly_out"] * 0.25


def _apply_elite_rule(w: dict[str, float], tier: str) -> None:
    """Layer 6 — a perfect pitch is never a ball."""
    if tier == "elite":
        w["ball"] = 0.0


def _is_terminal_outcome(o: str, strikes: int) -> bool:
    """Pitch ends the at-bat?"""
    terminal = {"walk", "hit_by_pitch", "ground_out", "fly_out", "line_out",
                "pop_out", "double_play", "fielders_choice", "sacrifice_fly",
                "single", "double", "triple", "home_run",
                "strikeout_looking", "strikeout_swinging"}
    return o in terminal


def _outs_recorded(o: str) -> int:
    if o == "double_play":   return 2
    if o in {"ground_out", "fly_out", "line_out", "pop_out", "fielders_choice",
             "sacrifice_fly", "strikeout_looking", "strikeout_swinging"}:
        return 1
    return 0


def _bases_advanced(o: str) -> int:
    return {"single": 1, "double": 2, "triple": 3, "home_run": 4,
            "walk": 1, "hit_by_pitch": 1}.get(o, 0)


# ─── Out-of-zone chase model (see docs/out-of-zone-chase-model.md) ──────────
#
# Pitches thrown OUTSIDE the strike zone are governed by the batter's swing
# decision (take → ball, chase → whiff/weak contact), not by pitch-quality
# weight nudges. All constants below are league-average calibration anchors and
# are meant to be tuned (ideally against your own Statcast description/events).

# Each out-of-zone zone maps to a depth band. With a single directional ring we
# only distinguish "chase" today; shadow/waste granularity needs continuous
# coordinates (pending the UI input model) or a second outer ring.
ZONE_DEPTH = {
    "out_up_in":    "chase",
    "out_up_away":  "chase",
    "out_low_in":   "chase",
    "out_low_away": "chase",
}

BASE_SWING = {"shadow": 0.55, "chase": 0.22, "waste": 0.05}

COUNT_SWING_FACTOR = {
    (0, 0): 0.85, (0, 1): 1.00, (0, 2): 1.45,
    (1, 0): 0.80, (1, 1): 1.00, (1, 2): 1.45,
    (2, 0): 0.55, (2, 1): 0.80, (2, 2): 1.40,
    (3, 0): 0.20, (3, 1): 0.55, (3, 2): 1.35,
}

WHIFF_GIVEN_SWING        = {"shadow": 0.25, "chase": 0.42, "waste": 0.62}
CALLED_STRIKE_GIVEN_TAKE = {"shadow": 0.45, "chase": 0.05, "waste": 0.00}

WEAK_CONTACT_DIST = {
    "foul_ball": 0.500, "ground_out": 0.200, "pop_out": 0.120,
    "fly_out":   0.060, "line_out":   0.020, "single":  0.085,
    "double":    0.012, "triple":     0.001, "home_run": 0.002,
}


def _is_out_of_zone(location: str) -> bool:
    return location in OUT_OF_ZONE_ZONES


def _zone_depth(location: str) -> str:
    return ZONE_DEPTH.get(location, "chase")


def out_of_zone_outcome_weights(
    zone_depth:    str,
    balls:         int,
    strikes:       int,
    discipline:    float = 1.0,
    quality_score: Optional[float] = None,
) -> dict[str, float]:
    """Outcome weights for a pitch thrown OUTSIDE the strike zone.

    Two-stage swing/take model (docs/out-of-zone-chase-model.md §3–5). Returns
    weights over the standard outcome vocabulary BEFORE the count-remap step;
    the four branches already sum to 1.0.
    """
    s = BASE_SWING[zone_depth] * COUNT_SWING_FACTOR[(balls, strikes)] * discipline
    s = max(0.02, min(0.95, s))

    whiff = WHIFF_GIVEN_SWING[zone_depth]
    if quality_score is not None:
        # A well-executed chase pitch misses more bats: ±10% at the extremes.
        whiff = max(0.0, min(0.95, whiff * (0.9 + 0.2 * quality_score)))

    cs = CALLED_STRIKE_GIVEN_TAKE[zone_depth]

    weights: dict[str, float] = {
        "swinging_strike": s * whiff,
        "called_strike":   (1.0 - s) * cs,
        "ball":            (1.0 - s) * (1.0 - cs),
    }
    contact_mass = s * (1.0 - whiff)
    for outcome, share in WEAK_CONTACT_DIST.items():
        weights[outcome] = weights.get(outcome, 0.0) + contact_mass * share
    return weights


def resolve_outcome(
    quality_score: float,
    quality_tier_label: str,
    pitch_type:    str,
    location:      str,
    balls:         int,
    strikes:       int,
    outs:          int,
    runners:       list[int],
    batter_avg:    float,
    rng:           random.Random,
) -> OutcomeBlock:
    """Weighted-sample outcome model.

    In-zone pitches use the 6-layer model from audit §2.3. Out-of-zone pitches
    route through the chase model instead (docs/out-of-zone-chase-model.md).
    """
    if _is_out_of_zone(location):
        w = out_of_zone_outcome_weights(
            zone_depth    = _zone_depth(location),
            balls         = balls,
            strikes       = strikes,
            discipline    = 1.0,            # league-average until a chase stat exists
            quality_score = quality_score,  # well-executed chase = nastier
        )
        _apply_runner_modifiers(w, runners, outs)
        # NOTE: deliberately skip _apply_elite_rule here. A chase pitch is a ball
        # by design when taken; zeroing the ball weight would break the mechanic.
    else:
        w = _base_outcome_weights(quality_score)
        _apply_count_modifiers(w, balls, strikes, quality_score)
        _apply_location_modifiers(w, location)
        _apply_batter_modifiers(w, batter_avg, quality_score)
        _apply_runner_modifiers(w, runners, outs)
        _apply_elite_rule(w, quality_tier_label)

    # Map count-conditional outcomes:
    # - On the 3rd strike, called/swinging strike become strikeouts.
    # - On a 4th ball, a ball becomes a walk.
    if strikes == 2:
        w["strikeout_looking"]  = w["called_strike"]
        w["strikeout_swinging"] = w["swinging_strike"]
        w["called_strike"]      = 0.0
        w["swinging_strike"]    = 0.0
    if balls == 3:
        w["walk"]               = w["ball"]
        w["ball"]               = 0.0

    # Disable DP/sac-fly if setup conditions aren't met (defensive — Layer 5
    # already gates these, but guard anyway).
    if not (1 in runners and outs < 2):
        w["double_play"] = 0.0
    if not (3 in runners and outs < 2):
        w["sacrifice_fly"] = 0.0

    # Weighted random sample. Sort for determinism with a seeded RNG.
    items = sorted(w.items())
    total = sum(v for _, v in items)
    if total <= 0:
        result = "ball"
    else:
        r = rng.random() * total
        cumulative = 0.0
        result = "ball"
        for name, weight in items:
            cumulative += weight
            if r < cumulative:
                result = name
                break

    is_out    = _outs_recorded(result) > 0
    is_hit    = result in {"single", "double", "triple", "home_run"}
    is_strike = result in {"called_strike", "swinging_strike", "foul_ball",
                           "strikeout_looking", "strikeout_swinging"}
    is_ball   = result in {"ball", "walk", "hit_by_pitch"}

    return OutcomeBlock(
        result         = result,
        display_name   = OUTCOME_DISPLAY_NAMES.get(result, result),
        is_terminal    = _is_terminal_outcome(result, strikes),
        is_out         = is_out,
        is_hit         = is_hit,
        is_strike      = is_strike,
        is_ball        = is_ball,
        outs_recorded  = _outs_recorded(result),
        bases_advanced = _bases_advanced(result),
    )


# ─── Coach hints + situation tags (audit §2.6) ──────────────────────────────

COACH_HINTS = {
    "full_count":     "Full count — make them put it in play or take a close pitch.",
    "pitcher_count":  "You're ahead in the count — use it. Expand the zone.",
    "hitter_count":   "Batter has the advantage — don't give in. Quality pitch.",
    "risp":           "Runners in scoring position — think about keeping the ball down.",
    "late_game":      "Late in the game — every pitch matters.",
    "two_out":        "Two outs — you can be more aggressive.",
    "first_pitch":    "Set the tone. Establish the zone early.",
    "default":        "Read the situation and make a smart call.",
}

# Hint priority order (most specific → fallback)
HINT_PRIORITY = ["full_count", "pitcher_count", "hitter_count", "risp",
                 "late_game", "two_out", "first_pitch"]


def compute_situation_tags(s: EvaluateScenario) -> list[str]:
    tags: list[str] = []
    count = (s.balls, s.strikes)
    if count == (0, 0):                          tags.append("first_pitch")
    if count in {(0, 2), (1, 2)}:                tags.append("pitcher_count")
    if count in {(2, 0), (2, 1), (3, 0), (3, 1)}: tags.append("hitter_count")
    if count == (3, 2):                          tags.append("full_count")
    if s.strikes == 2:                           tags.append("two_strike")
    if 2 in s.runners_on_base or 3 in s.runners_on_base:
        tags.append("risp")
    if not s.runners_on_base:                    tags.append("bases_empty")
    if s.inning >= 7:                            tags.append("late_game")
    if s.outs == 2:                              tags.append("two_out")
    return tags


def coach_hint_for_tags(tags: list[str]) -> str:
    for t in HINT_PRIORITY:
        if t in tags:
            return COACH_HINTS[t]
    return COACH_HINTS["default"]


def situation_summary_for_tags(tags: list[str], s: EvaluateScenario) -> str:
    bases = [str(b) for b in sorted(s.runners_on_base)] or ["empty"]
    pieces = [
        f"{s.balls}-{s.strikes} count",
        f"{s.outs} out{'s' if s.outs != 1 else ''}",
        f"Inning {s.inning}",
        "Bases: " + (", ".join(bases) if bases != ["empty"] else "empty"),
    ]
    if "risp" in tags:      pieces.append("RISP")
    if "late_game" in tags: pieces.append("late game")
    return " | ".join(pieces)


# ─── Adapter: translate new request shape into the existing feature builder ─

def _scenario_to_pitch_request(
    scenario: EvaluateScenario,
    selection: EvaluateSelection,
) -> PitchRequest:
    on_1b = 1 if 1 in scenario.runners_on_base else 0
    on_2b = 1 if 2 in scenario.runners_on_base else 0
    on_3b = 1 if 3 in scenario.runners_on_base else 0
    same_hand = 1 if scenario.batter_handedness == scenario.pitcher_handedness else 0
    return PitchRequest(
        balls               = scenario.balls,
        strikes             = scenario.strikes,
        on_1b               = on_1b,
        on_2b               = on_2b,
        on_3b               = on_3b,
        batter_avg          = scenario.batter_avg,
        stand               = "R" if scenario.batter_handedness == "right" else "L",
        inning              = scenario.inning,
        outs                = scenario.outs,
        score_diff          = scenario.score_diff,
        pitcher_pitch_count = scenario.pitcher_pitch_count,
        same_hand           = same_hand,
        location_zone       = selection.location,
        selected_pitch      = selection.pitch_type,
        pitcher_tends_Fastball  = scenario.pitcher_tends_Fastball,
        pitcher_tends_Sinker    = scenario.pitcher_tends_Sinker,
        pitcher_tends_Slider    = scenario.pitcher_tends_Slider,
        pitcher_tends_Changeup  = scenario.pitcher_tends_Changeup,
        pitcher_tends_Curveball = scenario.pitcher_tends_Curveball,
        pitcher_tends_Cutter    = scenario.pitcher_tends_Cutter,
        pitcher_tends_Splitter  = scenario.pitcher_tends_Splitter,
        pitcher_tends_Sweeper   = scenario.pitcher_tends_Sweeper,
    )


# ─── /api/v1/evaluate ───────────────────────────────────────────────────────

@app.post("/api/v1/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, _user=Depends(get_current_user)):
    """
    Server-authoritative pitch evaluation.

    Runs the ML model, then computes quality, simulates the outcome, generates
    feedback text, and tags the situation — all on the server so every
    platform sees identical results.
    """
    import time
    t0 = time.perf_counter()

    # Validate pitch + zone (mirrors /predict checks)
    if req.selection.pitch_type not in PITCH_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pitch '{req.selection.pitch_type}'. "
                   f"Valid options: {PITCH_CLASSES}",
        )
    if req.selection.location not in LOCATION_ZONES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown location '{req.selection.location}'. "
                   f"Valid options: {LOCATION_ZONES}",
        )
    invalid_runners = [b for b in req.scenario.runners_on_base if b not in (1, 2, 3)]
    if invalid_runners:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base(s) {invalid_runners} in runners_on_base.",
        )

    # Arsenal: pitches the player can throw. None/empty → full arsenal.
    arsenal_set = set(req.context.arsenal) if req.context.arsenal else set(PITCH_CLASSES)
    unknown_arsenal = arsenal_set - set(PITCH_CLASSES)
    if unknown_arsenal:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pitch(es) in arsenal: {sorted(unknown_arsenal)}. "
                   f"Valid options: {PITCH_CLASSES}",
        )
    if req.selection.pitch_type not in arsenal_set:
        raise HTTPException(
            status_code=400,
            detail=f"Selected pitch '{req.selection.pitch_type}' is not in the "
                   f"player's arsenal {sorted(arsenal_set)}.",
        )

    # ── 1. Model inference (reuses /predict's feature pipeline) ─────────────
    pitch_request = _scenario_to_pitch_request(req.scenario, req.selection)
    X = build_feature_row(pitch_request)
    probs_arr = model.predict_proba(X)[0]
    full_prob_map = dict(zip(PITCH_CLASSES, probs_arr))
    full_top_pitch = max(full_prob_map, key=full_prob_map.get)

    # Condition the distribution on the arsenal: drop unavailable pitches and
    # renormalize so quality/verdict/feedback are relative to what the player
    # can actually throw. (No retrain needed — see arsenal design discussion.)
    masked = {p: v for p, v in full_prob_map.items() if p in arsenal_set}
    total = sum(masked.values())
    if total > 0:
        prob_map = {p: v / total for p, v in masked.items()}
    else:
        prob_map = {p: 1.0 / len(masked) for p in masked}  # uniform fallback

    # "Pro pick": the unrestricted top pitch, surfaced only when the player
    # doesn't have it in their arsenal.
    pro_pitch = full_top_pitch if full_top_pitch not in arsenal_set else None
    pro_prob  = float(full_prob_map[full_top_pitch]) if pro_pitch else 0.0

    sorted_probs = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    top_pitch, top_prob = sorted_probs[0]
    selected_prob = float(prob_map.get(req.selection.pitch_type, 0.0))

    # ── 2. Verdict (reuses existing logic) ──────────────────────────────────
    verdict, _emoji, verdict_reason = get_verdict(
        selected_pitch = req.selection.pitch_type,
        top_pitch      = top_pitch,
        selected_prob  = selected_prob,
        top_prob       = float(top_prob),
        sorted_probs   = sorted_probs,
    )

    # ── 3. Quality score & tier ─────────────────────────────────────────────
    quality_score, q_tier = compute_pitch_quality(
        sorted_probs   = sorted_probs,
        selected_pitch = req.selection.pitch_type,
        verdict        = verdict,
        scenario       = req.scenario,
        selection      = req.selection,
    )

    # ── 4. Rating ───────────────────────────────────────────────────────────
    rating = compute_rating(
        selected_prob = selected_prob,
        top_prob      = float(top_prob),
        is_top        = req.selection.pitch_type == top_pitch,
    )

    # ── 5. Outcome simulation (server-authoritative randomness) ─────────────
    rng = random.Random(req.context.random_seed) if req.context.random_seed is not None \
          else random.Random()
    outcome = resolve_outcome(
        quality_score      = quality_score,
        quality_tier_label = q_tier,
        pitch_type         = req.selection.pitch_type,
        location           = req.selection.location,
        balls              = req.scenario.balls,
        strikes            = req.scenario.strikes,
        outs               = req.scenario.outs,
        runners            = req.scenario.runners_on_base,
        batter_avg         = req.scenario.batter_avg,
        rng                = rng,
    )

    # ── 6. Situation tags + coach hint ──────────────────────────────────────
    tags = compute_situation_tags(req.scenario)
    coaching_hint = coach_hint_for_tags(tags) if req.context.mode == "coach" else None

    # ── 7. Feedback text ────────────────────────────────────────────────────
    feedback = generate_feedback(
        scenario      = req.scenario,
        selection     = req.selection,
        sorted_probs  = sorted_probs,
        rating        = rating,
        coaching_hint = coaching_hint,
        pro_pitch     = pro_pitch,
        pro_prob      = pro_prob,
    )

    # ── 8. Assemble response ────────────────────────────────────────────────
    probabilities = [
        ProbabilityEntry(
            pitch_type  = p,
            probability = round(float(pv), 4),
            rank        = i + 1,
        )
        for i, (p, pv) in enumerate(sorted_probs)
    ]

    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

    return EvaluateResponse(
        evaluation = EvaluationBlock(
            quality_score              = round(quality_score, 4),
            quality_tier               = q_tier,
            rating                     = rating,
            probabilities              = probabilities,
            top_pitch                  = top_pitch,
            top_probability            = round(float(top_prob), 4),
            selected_pitch_probability = round(selected_prob, 4),
            verdict                    = verdict.lower(),
            verdict_reason             = verdict_reason,
        ),
        outcome   = outcome,
        feedback  = feedback,
        situation = SituationBlock(
            tags    = tags,
            summary = situation_summary_for_tags(tags, req.scenario),
        ),
        meta = EvaluateMeta(
            model_version    = MODEL_VERSION,
            response_time_ms = elapsed_ms,
        ),
    )


# ─── Dev server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
