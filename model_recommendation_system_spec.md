# AI Model Catalog & Recommendation System
## Project Specification & Handover Document

**Version:** 1.0  
**Date:** June 2026  
**Author:** Project planning session with Claude (Anthropic)  
**Purpose:** Full specification for a local AI model discovery, cataloging, and recommendation system targeting Flux.1 Dev (and other base models) with NSFW-capable reasoning via local LLM.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Technology Stack](#4-technology-stack)
5. [Component Specifications](#5-component-specifications)
   - [5.1 Database Schema](#51-database-schema)
   - [5.2 Crawler / Cache Manager](#52-crawler--cache-manager)
   - [5.3 Intent Parser (LLM Call 1)](#53-intent-parser-llm-call-1)
   - [5.4 Catalog Query Engine](#54-catalog-query-engine)
   - [5.5 Compatibility Analyst (LLM Call 2)](#55-compatibility-analyst-llm-call-2)
   - [5.6 Output Formatter](#56-output-formatter)
   - [5.7 FastAPI Service Layer](#57-fastapi-service-layer)
6. [Data Sources](#6-data-sources)
7. [LLM Configuration](#7-llm-configuration)
8. [Environment & Configuration](#8-environment--configuration)
9. [Implementation Phases](#9-implementation-phases)
10. [Directory Structure](#10-directory-structure)
11. [SCM & Development Conventions](#11-scm--development-conventions)
12. [Key Design Decisions & Rationale](#12-key-design-decisions--rationale)
13. [Known Constraints & Risks](#13-known-constraints--risks)

---

## 1. Project Overview

This system allows a user to provide a natural language prompt (potentially containing NSFW content) and receive a ranked list of recommended AI image generation models and LoRAs from Civitai and HuggingFace, along with their recommended weights, CFG values, steps, samplers, and trigger words — all reasoned locally without any external censored API.

The system is intentionally scoped to **discovery and recommendation only** in this phase. ComfyUI workflow generation is a deferred future phase and is not in scope here.

### Core User Flow

```
User prompt (natural language, potentially explicit)
        ↓
Intent Parser (local LLM) → structured intent + search tags
        ↓
SQLite cache query → scored candidate models and LoRAs
        ↓
Compatibility Analyst (local LLM) → ranked recommendations with reasoning
        ↓
Output: ranked list with settings, trigger words, download URLs
```

---

## 2. Goals & Non-Goals

### In Scope
- Natural language prompt → ranked model + LoRA recommendations
- Local SQLite cache of Civitai model metadata (expandable by base model on demand)
- Incremental cache updates for newly uploaded models
- Full NSFW content support throughout — no filtering at any layer
- Settings recommendations (CFG, steps, weight, sampler) derived from model metadata
- Trigger word extraction and surfacing
- FastAPI service layer for consumption by external tooling
- CLI interface for interactive use
- HuggingFace as a secondary metadata source

### Out of Scope (Deferred)
- ComfyUI workflow JSON generation
- Model file downloading (URLs are surfaced; downloading is the user's responsibility)
- TensorART and SeaART integration (no stable public API available; revisit if APIs become available)
- Image generation of any kind
- Training or fine-tuning any models

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│              CLI  ──────────────  FastAPI REST               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     Orchestrator                             │
│         Coordinates agents, manages cache checks,            │
│         triggers crawls, sequences LLM calls                 │
└──────┬──────────────────┬──────────────────┬────────────────┘
       │                  │                  │
┌──────▼──────┐  ┌────────▼───────┐  ┌──────▼──────────────┐
│  Agent 1    │  │   Agent 2      │  │     Agent 3          │
│  Intent     │  │   Catalog      │  │  Compatibility       │
│  Parser     │  │   Query        │  │  Analyst             │
│  (LLM)      │  │   (SQL only)   │  │  (LLM)               │
└──────┬──────┘  └────────┬───────┘  └──────┬───────────────┘
       │                  │                  │
       │         ┌────────▼───────┐          │
       │         │  SQLite Cache  │          │
       │         │  (local DB)    │◄─────────┘
       │         └────────┬───────┘
       │                  │
       │         ┌────────▼───────┐
       └────────►│   Civitai API  │
                 │ HuggingFace API│
                 └────────────────┘
```

### Agent Responsibilities

| Agent | LLM Required | Description |
|---|---|---|
| Intent Parser | Yes (Ollama) | Converts natural language prompt to structured intent object |
| Catalog Query | No (pure SQL) | Scores and retrieves candidates from local cache |
| Compatibility Analyst | Yes (Ollama) | Reads model descriptions, reasons over fit and conflicts |
| Crawler | No | Fetches and normalizes Civitai/HuggingFace metadata into cache |
| Output Formatter | No | Renders structured results as human-readable or JSON output |

---

## 4. Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Language | Python 3.11+ | Native to the ComfyUI ecosystem; already available in environment |
| Database | SQLite (via `sqlite3` stdlib) | Zero-ops, local, sufficient for catalog size |
| LLM runtime | Ollama (local) | Uncensored, no API costs, no content filtering |
| LLM model | `dolphin3.0-llama3.1:8b` (primary) | Uncensored fine-tune, supports function calling for structured JSON output |
| LLM fallback | `gemma3:12b` | Higher reasoning quality if VRAM permits when ComfyUI is idle |
| HTTP client | `httpx` (async) | Better async support than `requests` for API pagination |
| API framework | `fastapi` + `uvicorn` | Clean REST interface for external tooling consumption |
| Config management | `python-dotenv` | `.env` file for secrets, never committed to SCM |
| CLI | `argparse` (stdlib) | No extra dependencies for basic CLI |
| Scheduling | `schedule` or `cron` | Incremental cache updates |

### Python Dependencies (`requirements.txt`)

```
httpx>=0.27.0
fastapi>=0.111.0
uvicorn>=0.30.0
python-dotenv>=1.0.0
schedule>=1.2.0
pydantic>=2.0.0
ollama>=0.2.0
```

---

## 5. Component Specifications

### 5.1 Database Schema

**File:** `db/schema.sql`

```sql
CREATE TABLE IF NOT EXISTS models (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source              TEXT NOT NULL,              -- 'civitai' or 'huggingface'
    civitai_model_id    INTEGER,
    civitai_version_id  INTEGER UNIQUE,             -- unique per version
    hf_repo_id          TEXT,                       -- for HuggingFace models
    name                TEXT NOT NULL,
    version_name        TEXT,
    type                TEXT NOT NULL,              -- Checkpoint, LORA, TextualInversion, etc.
    base_model          TEXT,                       -- 'Flux.1 D', 'SDXL 1.0', 'Wan2.2', etc.
    nsfw_level          INTEGER DEFAULT 1,          -- 1=safe 2=suggestive 3=erotica 4=explicit
    description         TEXT,
    tags                TEXT,                       -- JSON array string
    trigger_words       TEXT,                       -- JSON array string
    recommended_weight  REAL,
    recommended_cfg     REAL,
    recommended_steps   INTEGER,
    recommended_sampler TEXT,
    download_url        TEXT,
    civitai_url         TEXT,
    stats_downloads     INTEGER DEFAULT 0,
    stats_thumbs_up     INTEGER DEFAULT 0,
    stats_thumbs_down   INTEGER DEFAULT 0,
    preview_image_url   TEXT,
    date_cached         TEXT NOT NULL,              -- ISO8601
    date_updated        TEXT NOT NULL               -- ISO8601
);

CREATE INDEX IF NOT EXISTS idx_base_model ON models(base_model);
CREATE INDEX IF NOT EXISTS idx_type ON models(type);
CREATE INDEX IF NOT EXISTS idx_source ON models(source);

CREATE TABLE IF NOT EXISTS base_model_index (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    base_model_name  TEXT UNIQUE NOT NULL,
    last_crawled     TEXT,                          -- ISO8601
    total_models     INTEGER DEFAULT 0,
    crawl_complete   INTEGER DEFAULT 0              -- 0=incomplete 1=complete
);

CREATE TABLE IF NOT EXISTS cache_requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    base_model_name  TEXT NOT NULL,
    requested_at     TEXT NOT NULL,                 -- ISO8601
    triggered_by     TEXT                           -- the prompt that triggered this crawl
);
```

**Notes:**
- Tags and trigger words are stored as JSON array strings (e.g. `'["rain", "action", "combat"]'`) and parsed at query time
- `nsfw_level` mirrors Civitai's integer scale — never filter this at write time, only at query time if the user explicitly requests SFW-only results
- `civitai_version_id` is the unique key for Civitai models since a single model can have multiple versions with different base models

---

### 5.2 Crawler / Cache Manager

**File:** `crawler/civitai_crawler.py`

#### Civitai API Endpoint

```
GET https://civitai.com/api/v1/models
    ?limit=100
    &types=Checkpoint,LORA
    &baseModels={base_model}
    &sort=Most Downloaded
    &page={page}
```

Headers:
```
Authorization: Bearer {CIVITAI_API_TOKEN}
Content-Type: application/json
```

**Important:** Do NOT pass `nsfw=false`. Omitting the parameter returns all content including NSFW when authenticated.

#### Crawl Modes

**Full crawl** — called when a base model is not in `base_model_index` or `crawl_complete=0`:
```python
def full_crawl(base_model: str) -> int:
    """Pages through all Civitai results for a base model.
    Returns count of models cached."""
```

**Incremental update** — called on schedule or via `--update` CLI flag:
```python
def incremental_update(base_model: str) -> int:
    """Fetches models newer than the last crawl date.
    Uses sort=Newest and stops when hitting already-cached version IDs."""
```

**On-demand expansion** — called by Orchestrator when a query comes in for an uncached base model:
```python
def ensure_base_model_cached(base_model: str) -> bool:
    """Checks base_model_index. If not cached, triggers full_crawl.
    Returns True when cache is ready."""
```

#### Field Mapping (Civitai API → DB)

| Civitai field | DB column | Notes |
|---|---|---|
| `id` | `civitai_model_id` | Model-level ID |
| `modelVersions[0].id` | `civitai_version_id` | Latest version ID |
| `name` | `name` | |
| `modelVersions[0].name` | `version_name` | |
| `type` | `type` | |
| `modelVersions[0].baseModel` | `base_model` | |
| `nsfwLevel` | `nsfw_level` | |
| `description` | `description` | Strip HTML tags before storing |
| `tags` | `tags` | Store as JSON array string |
| `modelVersions[0].trainedWords` | `trigger_words` | Store as JSON array string |
| `modelVersions[0].downloadUrl` | `download_url` | |
| `stats.downloadCount` | `stats_downloads` | |
| `stats.thumbsUpCount` | `stats_thumbs_up` | |
| `stats.thumbsDownCount` | `stats_thumbs_down` | |
| `modelVersions[0].images[0].url` | `preview_image_url` | First preview image |

#### Pagination

The Civitai API returns a `metadata` object in each response:
```json
{
  "items": [...],
  "metadata": {
    "nextPage": "https://civitai.com/api/v1/models?page=2&...",
    "currentPage": 1,
    "pageSize": 100,
    "totalItems": 4823
  }
}
```

Follow `metadata.nextPage` until it is `null`. Implement exponential backoff on 429 rate limit responses (start at 2s, max 30s).

---

### 5.3 Intent Parser (LLM Call 1)

**File:** `agents/intent_parser.py`

This is the first of two Ollama calls. It converts the user's raw natural language prompt into a structured intent object that drives the catalog query.

#### Ollama Call

```python
import ollama

def parse_intent(user_prompt: str, model: str = "dolphin3.0-llama3.1:8b") -> dict:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": INTENT_PARSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        format="json"
    )
    return json.loads(response["message"]["content"])
```

#### System Prompt

```
You are an AI image generation model selector assistant.
Your job is to analyze a user's image generation prompt and extract structured
information that will be used to search a catalog of AI models and LoRAs.

Extract the following from the user's prompt and return ONLY valid JSON with no
preamble, explanation, or markdown formatting:

{
  "subject_matter": ["list", "of", "main", "subjects"],
  "character_count": 0,
  "environment": ["list", "of", "environment", "descriptors"],
  "mood": ["list", "of", "mood", "descriptors"],
  "style_preference": ["cinematic", "realistic", "anime", "etc"],
  "content_tier": "SFW or NSFW",
  "base_model_preference": "Flux.1 D",
  "lora_types_needed": ["action/pose", "character", "style", "environment", "etc"],
  "search_tags": ["flat", "list", "of", "all", "relevant", "search", "terms"],
  "nsfw_elements": ["specific", "adult", "content", "elements", "if", "any"]
}

Rules:
- search_tags should be comprehensive — include synonyms and related terms
- content_tier must be exactly "SFW" or "NSFW"
- base_model_preference defaults to "Flux.1 D" unless the user specifies otherwise
- nsfw_elements should be specific enough to match relevant LoRA descriptions
- Return ONLY the JSON object, nothing else
```

#### Expected Output Example

Input: `"two characters are swordfighting in the rain"`

```json
{
  "subject_matter": ["sword fighting", "combat", "duel", "action"],
  "character_count": 2,
  "environment": ["rain", "outdoor", "wet", "stormy"],
  "mood": ["dramatic", "intense", "tense"],
  "style_preference": ["cinematic", "realistic"],
  "content_tier": "SFW",
  "base_model_preference": "Flux.1 D",
  "lora_types_needed": ["action/pose", "environment/weather", "character"],
  "search_tags": ["sword", "fighting", "combat", "duel", "rain", "action pose",
                  "two characters", "battle", "weapon", "wet", "stormy", "dynamic"],
  "nsfw_elements": []
}
```

---

### 5.4 Catalog Query Engine

**File:** `agents/catalog_query.py`

Pure SQL — no LLM involved. Fast, deterministic, runs entirely against the local cache.

#### Scoring Formula

Each candidate model receives a relevance score:

```
tag_overlap    = len(model_tags ∩ search_tags) / len(search_tags)
quality_signal = thumbs_up / (thumbs_up + thumbs_down)  [0 if no votes]
download_norm  = log10(downloads + 1) / 6               [normalized, 1M downloads ≈ 1.0]

final_score = (tag_overlap × 0.6) + (quality_signal × 0.25) + (download_norm × 0.15)
```

Tag overlap is weighted highest because it is the primary semantic match signal. Quality and popularity are tie-breakers, not primary drivers.

#### Query Function

```python
def query_catalog(intent: dict, limit_checkpoints: int = 5,
                  limit_loras: int = 15) -> dict:
    """
    Returns:
    {
        "checkpoints": [...],  # top N checkpoints scored and ranked
        "loras": [...]         # top N LoRAs scored and ranked
    }
    Each item includes all DB fields plus a computed relevance_score.
    """
```

#### On-Demand Cache Expansion

Before querying, check if the requested base model is cached:

```python
def ensure_cache_ready(base_model: str) -> None:
    row = db.execute(
        "SELECT crawl_complete FROM base_model_index WHERE base_model_name = ?",
        (base_model,)
    ).fetchone()
    if not row or not row["crawl_complete"]:
        print(f"[Cache] '{base_model}' not in cache. Crawling Civitai now...")
        db.execute(
            "INSERT INTO cache_requests (base_model_name, requested_at, triggered_by) VALUES (?, ?, ?)",
            (base_model, datetime.utcnow().isoformat(), intent.get("original_prompt", ""))
        )
        crawler.full_crawl(base_model)
```

---

### 5.5 Compatibility Analyst (LLM Call 2)

**File:** `agents/compatibility_analyst.py`

This is the most important LLM call. It receives the scored candidates from the catalog query and the original intent, reads model descriptions, and produces the final ranked recommendations with explicit reasoning.

#### System Prompt

```
You are an expert AI image generation model compatibility analyst.
You have deep knowledge of Stable Diffusion, Flux.1, LoRA training, and how
different models interact with each other at inference time.

You will receive:
1. A user's image generation intent (structured JSON)
2. A list of candidate checkpoint models with their metadata
3. A list of candidate LoRA models with their metadata

Your job is to recommend the best combination of one checkpoint and up to 5 LoRAs
that will work well together to achieve the user's intent.

Consider:
- Whether each LoRA's training content genuinely serves the user's prompt
- Whether the LoRAs are likely to conflict with each other (e.g. two strong
  style LoRAs will fight; a character LoRA and an action LoRA stack cleanly)
- Whether the recommended CFG, steps, and weight values from metadata are reliable
- Whether trigger words are required and what they are

Return ONLY valid JSON matching this exact schema, no preamble or explanation:

{
  "recommended_checkpoint": {
    "name": "string",
    "civitai_url": "string",
    "download_url": "string",
    "reason": "string — why this checkpoint suits the intent",
    "settings": {
      "cfg": 3.5,
      "steps": 28,
      "sampler": "dpmpp_2m",
      "scheduler": "sgm_uniform"
    }
  },
  "recommended_loras": [
    {
      "rank": 1,
      "name": "string",
      "civitai_url": "string",
      "download_url": "string",
      "trigger_word": "string or null",
      "weight": 0.8,
      "reason": "string — why this LoRA serves the intent",
      "compatibility_note": "string — how it interacts with other recommended LoRAs",
      "lora_type": "character | action | style | environment | concept"
    }
  ],
  "prompt_additions": ["trigger words and descriptors to add to the generation prompt"],
  "potential_conflicts": ["any tensions or risks in the recommended combination"],
  "confidence": "high | medium | low",
  "reasoning_summary": "2-3 sentence plain English summary of the recommendation"
}
```

#### Call Structure

```python
def analyze_compatibility(intent: dict, candidates: dict,
                          model: str = "dolphin3.0-llama3.1:8b") -> dict:
    context = {
        "user_intent": intent,
        "candidate_checkpoints": candidates["checkpoints"],
        "candidate_loras": candidates["loras"]
    }
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": COMPATIBILITY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context)}
        ],
        format="json"
    )
    return json.loads(response["message"]["content"])
```

---

### 5.6 Output Formatter

**File:** `output/formatter.py`

Renders the compatibility analyst's structured output in one of two modes:

#### Human-readable CLI output

```
═══════════════════════════════════════════════════════════
  MODEL RECOMMENDATIONS FOR: "two characters swordfighting in the rain"
═══════════════════════════════════════════════════════════

RECOMMENDED CHECKPOINT
  Name    : Flux Realism Pro v2.1
  URL     : https://civitai.com/models/12345
  CFG     : 3.5  |  Steps: 28  |  Sampler: dpmpp_2m
  Reason  : Strong photorealistic output with excellent dynamic scene handling

RECOMMENDED LoRAs  (apply in this order)

  #1  Action Combat Poses v3 [weight: 0.85]
      Trigger : actn_combat
      URL     : https://civitai.com/models/23456
      Type    : action/pose
      Reason  : Trained specifically on sword combat reference, covers
                dynamic two-person fight choreography

  #2  Rain & Wet Weather FX [weight: 0.7]
      Trigger : rain_fx
      URL     : https://civitai.com/models/34567
      Type    : environment
      Reason  : Adds rain, wet surfaces, and atmospheric depth

ADD TO YOUR PROMPT
  actn_combat, rain_fx, two figures, dynamic sword fight, heavy rain,
  wet ground reflection, dramatic lighting

CONFIDENCE  : high
SUMMARY     : This combination targets your scene directly — the action
              LoRA handles fight choreography while the weather LoRA adds
              environmental realism. Both stack cleanly with no style conflict.

POTENTIAL CONFLICTS  : none identified
═══════════════════════════════════════════════════════════
```

#### JSON output (for API consumers)

The raw structured output from the Compatibility Analyst, passed through with no transformation.

---

### 5.7 FastAPI Service Layer

**File:** `api/main.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Model Recommendation API", version="1.0.0")

class RecommendRequest(BaseModel):
    prompt: str
    base_model: str = "Flux.1 D"
    max_loras: int = 5
    nsfw_filter: bool = False   # False = return everything; True = SFW only

class UpdateCacheRequest(BaseModel):
    base_model: str

@app.post("/recommend")
async def recommend(request: RecommendRequest) -> dict:
    """
    Primary endpoint. Accepts a natural language prompt,
    returns ranked model and LoRA recommendations with settings.
    """

@app.post("/cache/update")
async def update_cache(request: UpdateCacheRequest) -> dict:
    """
    Triggers an incremental cache update for a specific base model.
    """

@app.post("/cache/crawl")
async def crawl_base_model(request: UpdateCacheRequest) -> dict:
    """
    Triggers a full crawl for a new base model not yet in cache.
    """

@app.get("/cache/status")
async def cache_status() -> dict:
    """
    Returns cache statistics — which base models are indexed,
    model counts, last updated timestamps.
    """
```

**Run with:** `uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload`

---

## 6. Data Sources

### Civitai (Primary)

- **API base URL:** `https://civitai.com/api/v1`
- **Authentication:** `Authorization: Bearer {CIVITAI_API_TOKEN}` header
- **Token:** Free to generate at civitai.com → Account → Settings → API Keys
- **NSFW access:** Requires authenticated requests; do NOT pass `nsfw=false`
- **Rate limits:** Not publicly documented; implement exponential backoff on 429
- **Key endpoints used:**
  - `GET /models` — model search and listing
  - `GET /models/{id}` — single model detail
  - `GET /model-versions/by-hash/{hash}` — lookup by SHA256 (for local file identification)

### HuggingFace (Secondary)

- **API base URL:** `https://huggingface.co/api`
- **Authentication:** `Authorization: Bearer {HF_TOKEN}` (free HuggingFace account)
- **Key endpoints used:**
  - `GET /models` — model search with `filter=` and `search=` params
  - `GET /models/{repo_id}` — model metadata including README card
- **Limitation:** No structured `trainedWords` equivalent — trigger words must be extracted from model card text via the LLM if present

### TensorART / SeaART

Not in scope for this phase. Neither platform has a stable public model catalog API. Revisit if APIs become available. Note that significant model overlap exists between these platforms and Civitai.

---

## 7. LLM Configuration

### Primary Model

```
ollama pull dolphin3.0-llama3.1:8b
```

- Uncensored fine-tune of Llama 3.1 8B
- Supports function calling / structured JSON output via `format="json"`
- Fits within 16GB VRAM
- No content refusals for adult model descriptions or explicit prompts

### Fallback / Higher Quality

```
ollama pull gemma3:12b
```

- Higher reasoning quality for complex compatibility analysis
- Use when ComfyUI is not loaded (cannot share 16GB VRAM simultaneously)
- Does not have the same uncensored fine-tuning as Dolphin — may occasionally refuse explicit content

### VRAM Constraint

**Critical:** The local LLM and ComfyUI cannot run simultaneously on a 16GB card. This system is a **planning tool** — run it when ComfyUI is idle. Do not attempt to load both.

The FastAPI service should check Ollama availability and return a clear error if the model is not loaded, rather than hanging or timing out silently.

### Ollama Configuration

Ensure Ollama is running before starting the service:
```bash
ollama serve
```

Default Ollama API endpoint: `http://localhost:11434` (configure via `OLLAMA_HOST` env var if different).

---

## 8. Environment & Configuration

### `.env` File (never commit to SCM)

```env
# Civitai API
CIVITAI_API_TOKEN=your_civitai_token_here

# HuggingFace API (optional for secondary source)
HF_TOKEN=your_huggingface_token_here

# Database
DB_PATH=./data/model_catalog.db

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_PRIMARY=dolphin3.0-llama3.1:8b
OLLAMA_MODEL_FALLBACK=gemma3:12b

# Service
API_HOST=0.0.0.0
API_PORT=8765

# Cache behavior
DEFAULT_BASE_MODEL=Flux.1 D
CRAWL_PAGE_SIZE=100
RATE_LIMIT_BACKOFF_START=2
RATE_LIMIT_BACKOFF_MAX=30
```

### `.env.example` (commit this to SCM)

Same as above but with placeholder values and comments. This is the file that gets committed — never the actual `.env`.

---

## 9. Implementation Phases

### Phase 1 — Foundation

**Goal:** Working cache with CLI query, no LLM yet.

- [ ] Initialize git repo, `.gitignore`, `.env.example`
- [ ] Set up `requirements.txt` and virtual environment
- [ ] Implement `db/schema.sql` and `db/database.py` (connection management)
- [ ] Implement `crawler/civitai_crawler.py` — full crawl for Flux.1 D
- [ ] Implement basic `agents/catalog_query.py` — tag scoring, returns raw results
- [ ] Implement `cli.py` — simple query interface: `python cli.py "your prompt"`
  - At this stage: outputs raw scored candidates, no LLM reasoning
- [ ] Verify cache data quality manually before adding LLM layer
- [ ] Commit checkpoint

**Acceptance criteria:** Running `python cli.py "two characters swordfighting in the rain"` returns a list of scored Flux.1 D LoRAs and checkpoints from the local cache.

---

### Phase 2 — Intelligence

**Goal:** Full LLM-powered recommendation pipeline.

- [ ] Implement `agents/intent_parser.py` with Ollama integration
- [ ] Implement `agents/compatibility_analyst.py` with Ollama integration
- [ ] Implement `output/formatter.py` — both human-readable and JSON modes
- [ ] Wire all agents together in `orchestrator.py`
- [ ] Update `cli.py` to use full pipeline
- [ ] Add `--json` flag to CLI for machine-readable output
- [ ] Commit checkpoint

**Acceptance criteria:** Running `python cli.py "two characters swordfighting in the rain"` returns a formatted ranked recommendation with reasoning, settings, and trigger words.

---

### Phase 3 — Service & Expansion

**Goal:** FastAPI wrapper and on-demand cache expansion.

- [ ] Implement `api/main.py` with all endpoints
- [ ] Implement on-demand base model expansion in `catalog_query.py`
- [ ] Implement incremental update mode in `civitai_crawler.py`
- [ ] Add `--update` CLI flag for manual incremental updates
- [ ] Add HuggingFace crawler as secondary source (`crawler/hf_crawler.py`)
- [ ] Write basic README for the service
- [ ] Commit checkpoint

**Acceptance criteria:** `POST /recommend` endpoint accepts a prompt and returns JSON recommendations. `POST /cache/crawl` with `base_model: "Wan2.2"` successfully crawls and caches Wan2.2 models.

---

### Phase 4 — Hardening (Future)

- [ ] Retry logic and error handling throughout
- [ ] Logging to file with rotation
- [ ] Rate limit tracking and automatic backoff
- [ ] Cache invalidation strategy for updated models
- [ ] Unit tests for scoring logic and JSON parsing
- [ ] TensorART / SeaART integration (if APIs become available)

---

## 10. Directory Structure

```
model-recommender/
├── .env                        # secrets — NEVER commit
├── .env.example                # template — commit this
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/
│   └── model_catalog.db        # SQLite database — gitignore this
│
├── db/
│   ├── schema.sql
│   └── database.py             # connection management, migrations
│
├── crawler/
│   ├── civitai_crawler.py      # Civitai API crawler
│   └── hf_crawler.py           # HuggingFace crawler (Phase 3)
│
├── agents/
│   ├── intent_parser.py        # LLM Call 1 — prompt → structured intent
│   ├── catalog_query.py        # SQL scoring — no LLM
│   └── compatibility_analyst.py # LLM Call 2 — candidates → recommendations
│
├── output/
│   └── formatter.py            # human-readable and JSON output rendering
│
├── api/
│   └── main.py                 # FastAPI service
│
├── orchestrator.py             # coordinates all agents end-to-end
└── cli.py                      # command-line interface
```

---

## 11. SCM & Development Conventions

### .gitignore

```gitignore
# Secrets
.env

# Database (large binary, not version-controlled)
data/
*.db

# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
```

### Commit Convention

Use conventional commits format for clean history:

```
feat: add Civitai crawler with pagination support
fix: handle 429 rate limit in crawler with exponential backoff
feat: implement intent parser with Ollama dolphin3.0
feat: add FastAPI /recommend endpoint
fix: correct tag overlap scoring for empty tag arrays
chore: add .env.example and update README
refactor: extract scoring formula into catalog_query helpers
```

### Branch Strategy

```
main          — stable, working code only
dev           — integration branch
feature/xxx   — individual feature branches, merge to dev
```

### What to Commit vs. Not

| Commit | Do NOT commit |
|---|---|
| All `.py` source files | `.env` (secrets) |
| `requirements.txt` | `data/*.db` (large binary cache) |
| `db/schema.sql` | `__pycache__/` |
| `.env.example` | `.venv/` |
| `README.md` | `*.log` |
| `.gitignore` | Any model `.safetensors` files |

### Rebuilding the Cache on a Fresh Clone

Because `data/model_catalog.db` is gitignored, a fresh clone needs to rebuild the cache:

```bash
python cli.py --crawl "Flux.1 D"
```

Document this clearly in the project README so it is not forgotten.

---

## 12. Key Design Decisions & Rationale

**Why local LLM over Anthropic API?**
The system processes explicit model descriptions and potentially explicit user prompts throughout. The Anthropic API's content policy is unpredictable on this material — it may refuse or sanitize outputs in ways that make recommendations useless. A local uncensored model has no such constraint.

**Why SQLite over Postgres?**
The catalog is a single-user local tool. SQLite handles millions of rows without issue, requires zero setup, and the database file can be deleted and rebuilt from scratch trivially. Postgres adds operational overhead with no benefit at this scale.

**Why two LLM calls instead of one?**
The intent parsing and compatibility analysis are different cognitive tasks that benefit from clean separation. Combining them into one call produces worse results because the model tries to do database-like reasoning (tag matching) and language reasoning (compatibility judgment) simultaneously. The catalog query between calls also filters the candidate pool significantly, so the second LLM call receives much less noise.

**Why tag scoring over pure semantic embedding search?**
Civitai's tags are already curated community vocabulary — they are the semantic layer. An embedding model adds complexity and another VRAM consumer without meaningfully improving results over tag intersection for this use case. Embeddings would matter more if searching free-form description text, which the LLM already handles in Call 2.

**Why no filtering at crawl time?**
Filtering at crawl time means rebuilding the cache if you change your content preferences. Storing everything and filtering at query time gives full flexibility with one crawl.

**Why templates (deferred phase) over generative workflow JSON?**
ComfyUI workflow JSON is a live graph of node connections with specific type constraints. Generating it from scratch with an LLM produces syntactically plausible but semantically broken graphs. Starting from validated templates and populating them is the only reliable approach.

---

## 13. Known Constraints & Risks

| Constraint | Impact | Mitigation |
|---|---|---|
| 16GB VRAM shared between ComfyUI and Ollama | Cannot run both simultaneously | This is a planning tool — use it when ComfyUI is idle; document clearly |
| Civitai API rate limits undocumented | Crawler may get throttled mid-crawl | Exponential backoff; resume from last page on failure |
| Civitai API may change without notice | Crawler breaks silently | Log raw API responses during crawl for debugging; pin to tested behavior |
| Model descriptions vary wildly in quality | LLM reasoning quality depends on description richness | Surface `confidence: low` in output when descriptions are thin |
| `trainedWords` often empty on Flux LoRAs | Trigger word recommendations may be missing | Fall back to LLM extraction from description; note when unavailable |
| HuggingFace has no structured trigger word field | Requires card text parsing | Use LLM to extract from README in HF crawler; lower confidence flag |
| Dolphin 3.0 8B may produce inconsistent JSON | Parser crashes on malformed output | Wrap all LLM calls in try/except; retry up to 3 times; fall back to raw candidate list |
| TensorART / SeaART have no public catalog API | Cannot index these sources systematically | Deferred; significant overlap with Civitai catalog anyway |

---

*End of specification. Hand this document to your Claude Code assistant and begin with Phase 1 implementation.*
