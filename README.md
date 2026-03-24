# Catcher Scenario / Pitching Suggestion API

A REST API service that returns ranked pitching suggestions based on in-game
criteria entered by a catcher or coaching staff.  The service is designed to be
consumed by a separate front-end application — it exposes only JSON endpoints.

---

## Features

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/api/v1/pitch-types` | GET | List all recognised pitch types |
| `/api/v1/pitch-types/{pitch_type}` | GET | Single pitch-type metadata |
| `/api/v1/suggestions` | POST | Get ranked pitching suggestions |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the development server

```bash
uvicorn app.main:app --reload
```

The interactive API docs are available at <http://localhost:8000/docs>.

---

## Example request

```bash
curl -X POST http://localhost:8000/api/v1/suggestions \
  -H "Content-Type: application/json" \
  -d '{
    "balls": 0,
    "strikes": 2,
    "batter_handedness": "right",
    "pitcher_handedness": "right",
    "inning": 7,
    "outs": 2,
    "runners_on_base": [2],
    "previous_pitches": ["four_seam_fastball", "four_seam_fastball"]
  }'
```

### Example response

```json
{
  "balls": 0,
  "strikes": 2,
  "batter_handedness": "right",
  "pitcher_handedness": "right",
  "inning": 7,
  "outs": 2,
  "situation_tags": ["risp", "two_out", "pitcher_count", "late_game"],
  "suggestions": [
    {
      "pitch_type": "slider",
      "location": "down_and_away",
      "confidence": 0.93,
      "rationale": "Pitcher's count — breaking ball to expand the zone and induce a chase. Platoon advantage — breaking ball sweeps away from the batter. Two outs, RISP — go for the strikeout with a breaking ball."
    },
    {
      "pitch_type": "curveball",
      "location": "down_and_away",
      "confidence": 0.93,
      "rationale": "..."
    }
  ]
}
```

---

## Scenario fields

| Field | Type | Default | Description |
|---|---|---|---|
| `balls` | int (0–3) | 0 | Current ball count |
| `strikes` | int (0–2) | 0 | Current strike count |
| `batter_handedness` | `left` / `right` / `switch` | `right` | Batter handedness |
| `pitcher_handedness` | `left` / `right` | `right` | Pitcher handedness |
| `inning` | int (1–20) | 1 | Current inning |
| `outs` | int (0–2) | 0 | Outs in the inning |
| `runners_on_base` | list of 1/2/3 | `[]` | Occupied bases |
| `previous_pitches` | list of pitch-type values | `[]` | Pitches thrown this at-bat |
| `available_pitch_types` | list of pitch-type values | `null` (all) | Pitcher's repertoire |

---

## Suggestion logic

The suggestion engine applies a rule-based scoring strategy:

* **Pitcher's count (0-2, 1-2)** — breaking balls and off-speed to expand the zone
* **Hitter's count (3-0, 3-1)** — fastballs to throw a strike and work back into the count
* **Full count (3-2)** — best available pitch that can land for a strike
* **RISP** — sinkers/two-seamers for ground balls; breaking balls with two outs
* **Platoon advantage** — same-side breaking balls score higher
* **Repeat avoidance** — pitches thrown in the last two pitches are penalised

---

## Running tests

```bash
pytest
```

All 44 tests should pass.

---

## Project structure

```
app/
  __init__.py
  main.py        # FastAPI application and routes
  models.py      # Pydantic request / response models
  services.py    # Suggestion scoring engine
tests/
  __init__.py
  test_api.py    # Pytest test suite
requirements.txt
pytest.ini
README.md
```