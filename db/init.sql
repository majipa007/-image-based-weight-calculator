CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS video_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name TEXT NOT NULL,
    top_k INTEGER NOT NULL CHECK (top_k BETWEEN 3 AND 5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS goat_frame_candidates (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES video_runs(run_id) ON DELETE CASCADE,
    goat_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    mask_area INTEGER NOT NULL CHECK (mask_area >= 0),
    crop_image_jpg BYTEA NOT NULL,
    mask_png BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidates_run_goat_area
    ON goat_frame_candidates (run_id, goat_id, mask_area DESC);

CREATE TABLE IF NOT EXISTS goat_results (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES video_runs(run_id) ON DELETE CASCADE,
    goat_id INTEGER NOT NULL,
    weight_proxy_kg DOUBLE PRECISION NOT NULL,
    samples_used INTEGER NOT NULL CHECK (samples_used >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, goat_id)
);
