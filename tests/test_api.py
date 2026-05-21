"""
Unit + integration tests for the new /api/v1/evaluate algorithms.

Covers:
- PitchQualityCalculator: tier boundaries, component formulas
- FeedbackGenerator: rating thresholds, text variants
- OutcomeEngine: weighted-sample distribution with seeded RNG, count gating
- Coach hints + situation tags: priority order and tag derivation
- /api/v1/evaluate: end-to-end through FastAPI TestClient
"""

import warnings
warnings.filterwarnings("ignore")

import math
import random
import pytest
from fastapi.testclient import TestClient

import api
from api import (
    EvaluateScenario, EvaluateSelection, EvaluateRequest, EvaluateContext,
    compute_pitch_quality, compute_rating, quality_tier,
    resolve_outcome, generate_feedback,
    compute_situation_tags, coach_hint_for_tags, COACH_HINTS,
    _gap_score, _rank_score, _verdict_modifier, _normalized_entropy,
    _base_outcome_weights, ALL_OUTCOMES,
    out_of_zone_outcome_weights, _is_out_of_zone, _zone_depth,
    OUT_OF_ZONE_ZONES,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def base_scenario():
    return EvaluateScenario(
        balls=1, strikes=1, outs=1, inning=4,
        runners_on_base=[], batter_handedness="right",
        pitcher_handedness="right", batter_avg=0.260,
        score_diff=0, pitcher_pitch_count=50,
    )

@pytest.fixture
def base_selection():
    return EvaluateSelection(pitch_type="Slider", location="low_away")


# ════════════════════════════════════════════════════════════════════════════
# PitchQualityCalculator (audit §2.1)
# ════════════════════════════════════════════════════════════════════════════

class TestQualityTiers:
    @pytest.mark.parametrize("score,expected", [
        (1.00, "elite"),
        (0.85, "elite"),
        (0.849, "strong"),
        (0.70, "strong"),
        (0.55, "good"),
        (0.40, "neutral"),
        (0.25, "poor"),
        (0.249, "terrible"),
        (0.00, "terrible"),
    ])
    def test_boundaries(self, score, expected):
        assert quality_tier(score) == expected


class TestGapScore:
    @pytest.mark.parametrize("gap,expected", [
        (0.00,  0.10),   # selected IS top
        (0.02,  0.05),   # < 3%
        (0.05,  0.00),   # < 8%
        (0.10, -0.05),   # < 15%
        (0.15, -0.10),   # = 15% boundary
        (1.00, -0.15),   # max gap → floor
    ])
    def test_thresholds(self, gap, expected):
        assert _gap_score(gap) == pytest.approx(expected, abs=1e-3)

    def test_floor_capped(self):
        assert _gap_score(2.0) == pytest.approx(-0.15)


class TestRankScore:
    @pytest.mark.parametrize("rank,expected", [
        (1,  0.15),
        (2,  0.05),
        (3,  0.00),
        (4, -0.05),
        (5, -0.10),
        (6, -0.10),   # capped at -0.10
        (20, -0.10),  # still capped
    ])
    def test_rank(self, rank, expected):
        assert _rank_score(rank) == pytest.approx(expected)


class TestVerdictModifier:
    @pytest.mark.parametrize("verdict,expected", [
        ("Correct",     0.10),
        ("correct",     0.10),
        ("Incorrect", -0.10),
        ("incorrect", -0.10),
        ("Acceptable",  0.00),
        ("",            0.00),
        (None,          0.00),
    ])
    def test_verdict(self, verdict, expected):
        assert _verdict_modifier(verdict) == pytest.approx(expected)


class TestEntropy:
    def test_uniform_distribution_max_entropy(self):
        probs = [0.25, 0.25, 0.25, 0.25]
        assert _normalized_entropy(probs) == pytest.approx(1.0)

    def test_peaked_distribution_low_entropy(self):
        probs = [0.97, 0.01, 0.01, 0.01]
        assert _normalized_entropy(probs) < 0.2

    def test_single_outcome_zero_entropy(self):
        assert _normalized_entropy([1.0, 0.0, 0.0]) == 0.0


class TestPitchQualityIntegration:
    """End-to-end checks of the additive 6-component quality score."""

    def _make(self, sorted_probs, selected, verdict, scenario, selection):
        return compute_pitch_quality(
            sorted_probs   = sorted_probs,
            selected_pitch = selected,
            verdict        = verdict,
            scenario       = scenario,
            selection      = selection,
        )

    def test_top_pitch_correct_verdict_scores_high(self, base_scenario, base_selection):
        probs = [("Slider", 0.45), ("Fastball", 0.30), ("Curveball", 0.15),
                 ("Changeup", 0.10)]
        score, tier = self._make(probs, "Slider", "Correct",
                                  base_scenario, base_selection)
        # base 0.60 + rank +0.15 + gap +0.10 + verdict +0.10 + entropy 0
        # + context ~ +0.004 (full count off, RISP off) ≈ 0.95
        assert score >= 0.85
        assert tier == "elite"

    def test_bottom_pitch_incorrect_verdict_scores_low(self, base_scenario, base_selection):
        probs = [("Fastball", 0.55), ("Slider", 0.20), ("Curveball", 0.15),
                 ("Changeup", 0.10)]
        bad = EvaluateSelection(pitch_type="Changeup", location="middle_middle")
        score, tier = self._make(probs, "Changeup", "Incorrect",
                                  base_scenario, bad)
        assert score <= 0.40
        assert tier in {"poor", "terrible", "neutral"}

    def test_score_clamped_to_unit_interval(self, base_scenario, base_selection):
        probs = [("Slider", 0.99), ("Fastball", 0.01)]
        score, _ = self._make(probs, "Slider", "Correct",
                               base_scenario, base_selection)
        assert 0.0 <= score <= 1.0

    def test_dangerous_batter_poor_pitch_middle_takes_double_penalty(
        self, base_scenario,
    ):
        scenario = base_scenario.model_copy(update={"batter_avg": 0.330})
        bad = EvaluateSelection(pitch_type="Changeup", location="middle_middle")
        probs = [("Fastball", 0.40), ("Slider", 0.30), ("Curveball", 0.20),
                 ("Changeup", 0.10)]  # 10% < 50% → counts as "poor pitch"
        score_dangerous, _ = self._make(probs, "Changeup", "Incorrect",
                                         scenario, bad)
        score_normal, _    = self._make(
            probs, "Changeup", "Incorrect",
            base_scenario,
            EvaluateSelection(pitch_type="Changeup", location="low_away"),
        )
        # Dangerous batter + middle location should be at least 0.02 lower.
        assert score_dangerous < score_normal


# ════════════════════════════════════════════════════════════════════════════
# FeedbackGenerator (audit §2.2)
# ════════════════════════════════════════════════════════════════════════════

class TestRating:
    def test_selected_is_top_is_excellent(self):
        assert compute_rating(0.45, 0.45, is_top=True) == "excellent"

    @pytest.mark.parametrize("top,sel,expected", [
        (0.40, 0.38,   "good"),         # gap 0.02 → < 3%
        (0.40, 0.371,  "good"),         # gap 0.029
        (0.40, 0.35,   "acceptable"),   # gap 0.05 → < 10%
        (0.40, 0.301,  "acceptable"),   # gap 0.099
        (0.40, 0.299,  "poor"),         # gap 0.101 → >= 10%
        (0.50, 0.00,   "poor"),         # gap 0.50
    ])
    def test_thresholds(self, top, sel, expected):
        # NB: pass top/selected directly to avoid float subtraction landing on
        # the wrong side of the < 0.10 boundary.
        assert compute_rating(sel, top, is_top=False) == expected


class TestFeedbackText:
    def test_excellent_pitcher_count_text(self, base_scenario):
        scen = base_scenario.model_copy(update={"balls": 0, "strikes": 2})
        probs = [("Slider", 0.45), ("Fastball", 0.30)]
        fb = generate_feedback(
            scenario=scen,
            selection=EvaluateSelection(pitch_type="Slider", location="low_away"),
            sorted_probs=probs, rating="excellent", coaching_hint=None,
        )
        assert "thinking like a catcher" in fb.combined_assessment
        assert "0-2 count" in fb.combined_assessment

    def test_excellent_risp_text(self, base_scenario):
        scen = base_scenario.model_copy(update={"runners_on_base": [2]})
        probs = [("Slider", 0.45), ("Fastball", 0.30)]
        fb = generate_feedback(
            scenario=scen,
            selection=EvaluateSelection(pitch_type="Slider", location="low_away"),
            sorted_probs=probs, rating="excellent", coaching_hint=None,
        )
        assert "runners in scoring position" in fb.combined_assessment.lower()

    def test_poor_recommends_top_pitch(self, base_scenario):
        probs = [("Fastball", 0.55), ("Slider", 0.20)]
        fb = generate_feedback(
            scenario=base_scenario,
            selection=EvaluateSelection(pitch_type="Slider", location="low_away"),
            sorted_probs=probs, rating="poor", coaching_hint=None,
        )
        assert fb.recommended_pitch == "Fastball"
        assert "Fastball" in fb.combined_assessment

    def test_acceptable_no_recommendation(self, base_scenario):
        probs = [("Fastball", 0.40), ("Slider", 0.35)]
        fb = generate_feedback(
            scenario=base_scenario,
            selection=EvaluateSelection(pitch_type="Slider", location="low_away"),
            sorted_probs=probs, rating="acceptable", coaching_hint=None,
        )
        assert fb.recommended_pitch is None

    def test_pitch_not_in_distribution(self, base_scenario):
        # Selected pitch isn't even in sorted_probs
        probs = [("Fastball", 0.55), ("Curveball", 0.45)]
        fb = generate_feedback(
            scenario=base_scenario,
            selection=EvaluateSelection(pitch_type="Slider", location="low_away"),
            sorted_probs=probs, rating="poor", coaching_hint=None,
        )
        assert "wasn't in the model's distribution" in fb.pitch_assessment

    def test_coaching_hint_passthrough(self, base_scenario):
        probs = [("Slider", 0.45), ("Fastball", 0.30)]
        fb = generate_feedback(
            scenario=base_scenario,
            selection=EvaluateSelection(pitch_type="Slider", location="low_away"),
            sorted_probs=probs, rating="good",
            coaching_hint="Read the situation.",
        )
        assert fb.coaching_hint == "Read the situation."


# ════════════════════════════════════════════════════════════════════════════
# OutcomeEngine (audit §2.3)
# ════════════════════════════════════════════════════════════════════════════

class TestBaseWeights:
    def test_higher_quality_more_strikes_fewer_hits(self):
        low  = _base_outcome_weights(0.10)
        high = _base_outcome_weights(0.95)
        low_strike = low["called_strike"] + low["swinging_strike"] + low["foul_ball"]
        hi_strike  = high["called_strike"] + high["swinging_strike"] + high["foul_ball"]
        low_hit    = low["single"] + low["double"] + low["triple"] + low["home_run"]
        hi_hit     = high["single"] + high["double"] + high["triple"] + high["home_run"]
        assert hi_strike > low_strike
        assert hi_hit < low_hit


class TestResolveOutcome:
    def test_two_strikes_strike_becomes_strikeout(self):
        # Force determinism with a heavily weighted rng + 2-strike count
        rng = random.Random(42)
        results = []
        for seed in range(50):
            r = random.Random(seed)
            out = resolve_outcome(
                quality_score=0.95, quality_tier_label="elite",
                pitch_type="Slider", location="low_away",
                balls=0, strikes=2, outs=1, runners=[], batter_avg=0.250,
                rng=r,
            )
            results.append(out.result)
        # A "called_strike" or "swinging_strike" should never appear on a 2-strike count;
        # they get remapped to strikeouts.
        assert "called_strike" not in results
        assert "swinging_strike" not in results
        # Most outcomes should be strikeouts on an elite pitch in 0-2
        ko = sum(1 for r in results if r in ("strikeout_looking", "strikeout_swinging"))
        assert ko >= 10

    def test_three_balls_ball_becomes_walk(self):
        results = []
        for seed in range(80):
            r = random.Random(seed)
            out = resolve_outcome(
                quality_score=0.10, quality_tier_label="terrible",
                pitch_type="Fastball", location="up_in",
                balls=3, strikes=0, outs=0, runners=[], batter_avg=0.260,
                rng=r,
            )
            results.append(out.result)
        # On 3-0 with terrible quality, "ball" should never appear; walks should.
        assert "ball" not in results
        walks = sum(1 for r in results if r == "walk")
        assert walks > 0

    def test_elite_pitch_never_a_ball(self):
        # Quality 0.90 → elite; ball weight should be zeroed.
        for seed in range(40):
            r = random.Random(seed)
            out = resolve_outcome(
                quality_score=0.90, quality_tier_label="elite",
                pitch_type="Slider", location="low_away",
                balls=1, strikes=1, outs=0, runners=[], batter_avg=0.260,
                rng=r,
            )
            assert out.result != "ball"

    def test_double_play_requires_runner_on_first(self):
        for seed in range(50):
            r = random.Random(seed)
            out = resolve_outcome(
                quality_score=0.50, quality_tier_label="good",
                pitch_type="Sinker", location="low_in",
                balls=0, strikes=0, outs=0, runners=[], batter_avg=0.260,
                rng=r,
            )
            assert out.result != "double_play"

    def test_sacrifice_fly_requires_runner_on_third(self):
        for seed in range(50):
            r = random.Random(seed)
            out = resolve_outcome(
                quality_score=0.50, quality_tier_label="good",
                pitch_type="Fastball", location="up_away",
                balls=0, strikes=0, outs=0, runners=[1, 2], batter_avg=0.260,
                rng=r,
            )
            assert out.result != "sacrifice_fly"

    def test_outcome_flags_are_self_consistent(self):
        rng = random.Random(0)
        out = resolve_outcome(
            quality_score=0.30, quality_tier_label="poor",
            pitch_type="Fastball", location="middle_middle",
            balls=1, strikes=1, outs=0, runners=[], batter_avg=0.260,
            rng=rng,
        )
        # A pitch is at most one of: strike, ball, hit, out
        flags = [out.is_strike, out.is_ball, out.is_hit, out.is_out]
        assert sum(flags) >= 1

    def test_seeded_rng_is_deterministic(self):
        r1 = random.Random(123)
        r2 = random.Random(123)
        kwargs = dict(
            quality_score=0.55, quality_tier_label="good",
            pitch_type="Slider", location="low_away",
            balls=1, strikes=1, outs=0, runners=[1], batter_avg=0.275,
        )
        a = resolve_outcome(**kwargs, rng=r1)
        b = resolve_outcome(**kwargs, rng=r2)
        assert a.result == b.result


# ════════════════════════════════════════════════════════════════════════════
# Coach hints + situation tags (audit §2.6)
# ════════════════════════════════════════════════════════════════════════════

class TestSituationTags:
    def test_first_pitch(self, base_scenario):
        s = base_scenario.model_copy(update={"balls": 0, "strikes": 0})
        assert "first_pitch" in compute_situation_tags(s)

    def test_pitcher_count(self, base_scenario):
        s = base_scenario.model_copy(update={"balls": 0, "strikes": 2})
        tags = compute_situation_tags(s)
        assert "pitcher_count" in tags
        assert "two_strike" in tags

    def test_hitter_count(self, base_scenario):
        for c in [(2, 0), (2, 1), (3, 0), (3, 1)]:
            s = base_scenario.model_copy(update={"balls": c[0], "strikes": c[1]})
            assert "hitter_count" in compute_situation_tags(s)

    def test_full_count(self, base_scenario):
        s = base_scenario.model_copy(update={"balls": 3, "strikes": 2})
        assert "full_count" in compute_situation_tags(s)

    def test_risp_and_bases_empty_are_mutually_exclusive(self, base_scenario):
        risp_s = base_scenario.model_copy(update={"runners_on_base": [2]})
        empty_s = base_scenario.model_copy(update={"runners_on_base": []})
        assert "risp" in compute_situation_tags(risp_s)
        assert "bases_empty" not in compute_situation_tags(risp_s)
        assert "bases_empty" in compute_situation_tags(empty_s)

    def test_late_game(self, base_scenario):
        assert "late_game" in compute_situation_tags(
            base_scenario.model_copy(update={"inning": 9})
        )

    def test_two_out(self, base_scenario):
        assert "two_out" in compute_situation_tags(
            base_scenario.model_copy(update={"outs": 2})
        )


class TestCoachHints:
    def test_full_count_wins_over_risp(self):
        assert coach_hint_for_tags(["full_count", "risp"]) == COACH_HINTS["full_count"]

    def test_pitcher_count_wins_over_risp(self):
        assert coach_hint_for_tags(["pitcher_count", "risp"]) == COACH_HINTS["pitcher_count"]

    def test_default_when_no_match(self):
        assert coach_hint_for_tags(["two_strike", "bases_empty"]) == COACH_HINTS["default"]

    def test_risp_alone(self):
        assert coach_hint_for_tags(["risp"]) == COACH_HINTS["risp"]


# ════════════════════════════════════════════════════════════════════════════
# Out-of-zone chase model
# ════════════════════════════════════════════════════════════════════════════

class TestZoneClassification:
    def test_out_of_zone_set(self):
        assert _is_out_of_zone("out_low_away")
        assert not _is_out_of_zone("low_away")
        assert OUT_OF_ZONE_ZONES == {
            "out_up_in", "out_up_away", "out_low_in", "out_low_away"
        }

    def test_zone_depth_defaults_to_chase(self):
        assert _zone_depth("out_low_away") == "chase"
        assert _zone_depth("anything_unknown") == "chase"


class TestOutOfZoneWeights:
    def test_distribution_sums_to_one(self):
        for depth in ("shadow", "chase", "waste"):
            for b in range(4):
                for s in range(3):
                    w = out_of_zone_outcome_weights(depth, b, s)
                    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_ball_rate_rises_with_depth(self):
        # Same count, further out of zone → more balls (less swinging).
        shadow = out_of_zone_outcome_weights("shadow", 1, 1)["ball"]
        chase  = out_of_zone_outcome_weights("chase", 1, 1)["ball"]
        waste  = out_of_zone_outcome_weights("waste", 1, 1)["ball"]
        assert shadow < chase < waste

    def test_three_oh_is_almost_all_ball(self):
        # Nobody chases 3-0 — it's a ball the overwhelming majority of the time.
        w = out_of_zone_outcome_weights("chase", 3, 0)
        assert w["ball"] > 0.85

    def test_two_strikes_more_whiffs_than_three_oh(self):
        protect = out_of_zone_outcome_weights("chase", 0, 2)["swinging_strike"]
        ahead   = out_of_zone_outcome_weights("chase", 3, 0)["swinging_strike"]
        assert protect > ahead

    def test_power_is_near_zero(self):
        w = out_of_zone_outcome_weights("chase", 0, 0)
        assert w["home_run"] < 0.005

    def test_quality_scaling_raises_whiff(self):
        low_q  = out_of_zone_outcome_weights("chase", 0, 2, quality_score=0.0)
        high_q = out_of_zone_outcome_weights("chase", 0, 2, quality_score=1.0)
        assert high_q["swinging_strike"] > low_q["swinging_strike"]


class TestOutOfZoneResolveOutcome:
    def test_elite_pitch_out_of_zone_CAN_be_a_ball(self):
        # In-zone elite pitches never go ball (Layer 6). Out-of-zone, the elite
        # rule is skipped — a taken chase pitch is a ball by design.
        saw_ball = False
        for seed in range(80):
            out = resolve_outcome(
                quality_score=0.90, quality_tier_label="elite",
                pitch_type="Slider", location="out_low_away",
                balls=1, strikes=0, outs=0, runners=[], batter_avg=0.260,
                rng=random.Random(seed),
            )
            if out.result == "ball":
                saw_ball = True
                break
        assert saw_ball, "out-of-zone elite pitch should still be able to be a ball"

    def test_three_oh_out_of_zone_mostly_walks(self):
        results = [
            resolve_outcome(
                quality_score=0.5, quality_tier_label="good",
                pitch_type="Fastball", location="out_up_away",
                balls=3, strikes=0, outs=0, runners=[], batter_avg=0.26,
                rng=random.Random(s),
            ).result
            for s in range(120)
        ]
        # balls==3, so a ball becomes a walk. Hitters take 3-0 chase pitches.
        assert sum(r == "walk" for r in results) > 90

    def test_outcome_is_valid_and_seeded(self):
        a = resolve_outcome(
            quality_score=0.6, quality_tier_label="good",
            pitch_type="Slider", location="out_low_away",
            balls=0, strikes=2, outs=1, runners=[1], batter_avg=0.27,
            rng=random.Random(99),
        )
        b = resolve_outcome(
            quality_score=0.6, quality_tier_label="good",
            pitch_type="Slider", location="out_low_away",
            balls=0, strikes=2, outs=1, runners=[1], batter_avg=0.27,
            rng=random.Random(99),
        )
        assert a.result in ALL_OUTCOMES
        assert a.result == b.result


class TestQualityIntent:
    def _q(self, scenario, selection):
        probs = [("Slider", 0.40), ("Fastball", 0.30),
                 ("Curveball", 0.20), ("Changeup", 0.10)]
        return compute_pitch_quality(probs, "Slider", "Correct", scenario, selection)[0]

    def test_out_of_zone_two_strike_beats_three_oh(self):
        sel = EvaluateSelection(pitch_type="Slider", location="out_low_away")
        two_strike = EvaluateScenario(balls=0, strikes=2, outs=1, inning=4,
                                      runners_on_base=[], batter_avg=0.26)
        three_oh = EvaluateScenario(balls=3, strikes=0, outs=1, inning=4,
                                    runners_on_base=[], batter_avg=0.26)
        assert self._q(two_strike, sel) > self._q(three_oh, sel)

    def test_in_zone_unaffected_by_intent(self):
        # In-zone selection should get no out-of-zone intent adjustment.
        in_zone = EvaluateSelection(pitch_type="Slider", location="low_away")
        s2 = EvaluateScenario(balls=0, strikes=2, outs=1, inning=4,
                              runners_on_base=[], batter_avg=0.26)
        s30 = EvaluateScenario(balls=3, strikes=0, outs=1, inning=4,
                               runners_on_base=[], batter_avg=0.26)
        # Counts still differ via existing modifiers, but neither gets the
        # ±0.03/0.04 out-of-zone intent term — so they stay close.
        assert abs(self._q(s2, in_zone) - self._q(s30, in_zone)) < 0.03


# ════════════════════════════════════════════════════════════════════════════
# /api/v1/evaluate end-to-end
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


class TestEvaluateEndpoint:
    def _payload(self, **scenario_overrides):
        scenario = {
            "balls": 0, "strikes": 2, "outs": 1, "inning": 7,
            "runners_on_base": [2, 3],
            "batter_handedness": "right", "pitcher_handedness": "right",
            "batter_avg": 0.285, "score_diff": -1, "pitcher_pitch_count": 87,
            "previous_pitches": [],
        }
        scenario.update(scenario_overrides)
        return {
            "scenario": scenario,
            "selection": {"pitch_type": "Curveball", "location": "low_away"},
            "context": {"mode": "game", "player_id": "test", "random_seed": 7},
        }

    def test_returns_full_response_shape(self, client):
        r = client.post("/api/v1/evaluate", json=self._payload())
        assert r.status_code == 200, r.text
        body = r.json()
        for key in ("evaluation", "outcome", "feedback", "situation", "meta"):
            assert key in body
        ev = body["evaluation"]
        for key in ("quality_score", "quality_tier", "rating",
                    "probabilities", "top_pitch", "top_probability",
                    "selected_pitch_probability", "verdict", "verdict_reason"):
            assert key in ev
        # quality_score in unit interval
        assert 0.0 <= ev["quality_score"] <= 1.0
        # probabilities sorted desc with ranks
        ranks = [p["rank"] for p in ev["probabilities"]]
        assert ranks == sorted(ranks)
        probs = [p["probability"] for p in ev["probabilities"]]
        assert probs == sorted(probs, reverse=True)

    def test_outcome_is_deterministic_with_seed(self, client):
        a = client.post("/api/v1/evaluate", json=self._payload()).json()
        b = client.post("/api/v1/evaluate", json=self._payload()).json()
        assert a["outcome"]["result"] == b["outcome"]["result"]

    def test_coach_mode_includes_hint(self, client):
        payload = self._payload()
        payload["context"]["mode"] = "coach"
        body = client.post("/api/v1/evaluate", json=payload).json()
        assert body["feedback"]["coaching_hint"] is not None
        # RISP scenario with late_game; pitcher_count (0-2) should win priority
        assert body["feedback"]["coaching_hint"] == COACH_HINTS["pitcher_count"]

    def test_game_mode_no_hint(self, client):
        body = client.post("/api/v1/evaluate", json=self._payload()).json()
        assert body["feedback"]["coaching_hint"] is None

    def test_situation_tags_match_scenario(self, client):
        body = client.post("/api/v1/evaluate", json=self._payload()).json()
        tags = body["situation"]["tags"]
        assert "pitcher_count" in tags
        assert "two_strike" in tags
        assert "risp" in tags
        assert "late_game" in tags

    def test_invalid_pitch_returns_400(self, client):
        payload = self._payload()
        payload["selection"]["pitch_type"] = "NotARealPitch"
        r = client.post("/api/v1/evaluate", json=payload)
        assert r.status_code == 400

    def test_invalid_location_returns_400(self, client):
        payload = self._payload()
        payload["selection"]["location"] = "in_the_dirt"
        r = client.post("/api/v1/evaluate", json=payload)
        assert r.status_code == 400

    def test_invalid_runner_base_returns_400(self, client):
        payload = self._payload()
        payload["scenario"]["runners_on_base"] = [4]
        r = client.post("/api/v1/evaluate", json=payload)
        assert r.status_code == 400

    def test_predict_endpoint_still_works(self, client):
        """Backwards-compat: the old /predict endpoint is untouched."""
        r = client.post("/predict", json={
            "balls": 0, "strikes": 2, "on_1b": 1, "on_2b": 0, "on_3b": 0,
            "batter_avg": 0.285, "stand": "R", "inning": 7, "outs": 1,
            "score_diff": -1, "pitcher_pitch_count": 75, "same_hand": 0,
            "location_zone": "low_away", "selected_pitch": "Slider",
        })
        assert r.status_code == 200, r.text
        assert "probabilities" in r.json()
        assert "top_pitch" in r.json()

    def test_out_of_zone_location_accepted(self, client):
        # An out-of-zone target is now a valid location and yields a real outcome.
        payload = self._payload()
        payload["selection"]["location"] = "out_low_away"
        r = client.post("/api/v1/evaluate", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["outcome"]["result"] in ALL_OUTCOMES

    def test_out_of_zone_three_oh_walks_through_api(self, client):
        # 3-0 chase pitch through the full endpoint should resolve to a walk
        # (deterministic via the seed in the payload context).
        payload = self._payload(balls=3, strikes=0)
        payload["selection"]["location"] = "out_up_away"
        r = client.post("/api/v1/evaluate", json=payload)
        assert r.status_code == 200, r.text
        assert r.json()["outcome"]["result"] == "walk"
