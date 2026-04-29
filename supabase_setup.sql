-- ============================================================
--  SUPABASE SCHEMA SETUP
--  Run this in your Supabase SQL editor before anything else
-- ============================================================

-- 1. Raw Statcast pitch data
CREATE TABLE IF NOT EXISTS pitches (
    id                  BIGSERIAL PRIMARY KEY,
    game_pk             BIGINT,
    game_date           DATE,
    pitcher             BIGINT,
    batter              BIGINT,
    player_name         TEXT,          -- batter name
    pitch_type          TEXT,
    pitch_label         TEXT,          -- cleaned label (Fastball, Slider, etc.)
    balls               SMALLINT,
    strikes             SMALLINT,
    on_1b               SMALLINT,
    on_2b               SMALLINT,
    on_3b               SMALLINT,
    inning              SMALLINT,
    outs_when_up        SMALLINT,
    stand               CHAR(1),       -- batter hand L/R
    p_throws            CHAR(1),       -- pitcher hand L/R
    home_score          SMALLINT,
    away_score          SMALLINT,
    release_speed       NUMERIC(5,2),
    at_bat_number       SMALLINT,
    pitch_number        SMALLINT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast training queries
CREATE INDEX IF NOT EXISTS idx_pitches_date     ON pitches (game_date);
CREATE INDEX IF NOT EXISTS idx_pitches_batter   ON pitches (batter);
CREATE INDEX IF NOT EXISTS idx_pitches_pitcher  ON pitches (pitcher);

-- 2. Batter season batting averages
CREATE TABLE IF NOT EXISTS batters (
    id              BIGSERIAL PRIMARY KEY,
    season          SMALLINT NOT NULL,
    name            TEXT NOT NULL,
    batter_avg      NUMERIC(5,3),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(season, name)
);

-- 3. Trained model metadata (store as JSON blob)
CREATE TABLE IF NOT EXISTS model_versions (
    id              BIGSERIAL PRIMARY KEY,
    version         TEXT NOT NULL,
    trained_on      DATE NOT NULL,
    accuracy        NUMERIC(5,4),
    log_loss        NUMERIC(7,4),
    pitch_classes   TEXT[],           -- e.g. {Fastball, Slider, Curveball}
    feature_names   TEXT[],
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
