-- PostgreSQL schema for the AI Model Catalog.
-- Idempotent — safe to run on every startup.

CREATE TABLE IF NOT EXISTS models (
    id                  BIGSERIAL PRIMARY KEY,
    source              TEXT NOT NULL,
    civitai_model_id    BIGINT,
    civitai_version_id  BIGINT UNIQUE,
    hf_repo_id          TEXT,
    name                TEXT NOT NULL,
    version_name        TEXT,
    type                TEXT NOT NULL,
    base_model          TEXT,
    nsfw_level          INTEGER DEFAULT 1,
    description         TEXT,
    tags                JSONB NOT NULL DEFAULT '[]',
    trigger_words       JSONB NOT NULL DEFAULT '[]',
    recommended_weight  FLOAT,
    recommended_cfg     FLOAT,
    recommended_steps   INTEGER,
    recommended_sampler TEXT,
    download_url        TEXT,
    civitai_url         TEXT,
    stats_downloads     BIGINT DEFAULT 0,
    stats_thumbs_up     BIGINT DEFAULT 0,
    stats_thumbs_down   BIGINT DEFAULT 0,
    preview_image_url   TEXT,
    tensorart_model_id  TEXT,
    date_cached         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    date_updated        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Additive migrations — run before indexes so new columns exist before any
-- index that references them. ADD COLUMN IF NOT EXISTS is idempotent: it is
-- a no-op on fresh installs (column already in CREATE TABLE above) and adds
-- the column on existing databases that predate the change.
-- ---------------------------------------------------------------------------
ALTER TABLE models ADD COLUMN IF NOT EXISTS tensorart_model_id TEXT;

CREATE INDEX IF NOT EXISTS idx_base_model   ON models(base_model);
CREATE INDEX IF NOT EXISTS idx_type         ON models(type);
CREATE INDEX IF NOT EXISTS idx_source       ON models(source);
CREATE INDEX IF NOT EXISTS idx_nsfw_level   ON models(nsfw_level);
CREATE INDEX IF NOT EXISTS idx_tags_gin     ON models USING GIN(tags);

-- Unique index for HuggingFace models; NULLs (Civitai rows) are excluded automatically
-- because PostgreSQL treats each NULL as distinct in a non-partial unique index.
CREATE UNIQUE INDEX IF NOT EXISTS idx_hf_repo_id ON models(hf_repo_id);

-- Unique index for TensorArt models; NULLs excluded (same semantics as idx_hf_repo_id)
CREATE UNIQUE INDEX IF NOT EXISTS idx_tensorart_model_id ON models(tensorart_model_id) WHERE tensorart_model_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS base_model_index (
    id               BIGSERIAL PRIMARY KEY,
    base_model_name  TEXT UNIQUE NOT NULL,
    last_crawled     TIMESTAMPTZ,
    total_models     INTEGER DEFAULT 0,
    crawl_complete   BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS cache_requests (
    id               BIGSERIAL PRIMARY KEY,
    base_model_name  TEXT NOT NULL,
    requested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    triggered_by     TEXT
);
