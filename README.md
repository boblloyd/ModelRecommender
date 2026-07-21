# Model Recommender

A local AI image generation model discovery and recommendation tool. Give it a natural language prompt; it searches a local catalog of CivitAI, HuggingFace, and TensorArt models and uses a local LLM (Ollama) to identify which checkpoint and LoRAs will have the most impact on your specific scene.

Designed to run when ComfyUI is idle — the LLM and ComfyUI share GPU and cannot run simultaneously on 16 GB VRAM. Use this tool to plan your generation, then switch to ComfyUI to execute it.

> **New here?** See [INSTALLATION.md](INSTALLATION.md) to set up the database, Ollama, and the catalog before running any queries.

---

## Quick start

```bash
# Activate your virtual environment (see INSTALLATION.md)
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# Populate the catalog for the base model you use (only needed once)
python cli.py --crawl "Flux.1 D"

# Get recommendations
python cli.py "two knights swordfighting in heavy rain"
```

---

## CLI reference

### Get recommendations

```bash
python cli.py "your prompt here"
```

The full pipeline runs by default:

1. **Intent parser** (Ollama) — extracts structured tags, style, and subject from your prompt
2. **Catalog query** (SQL) — scores all cached models against those tags
3. **Compatibility analyst** (Ollama) — reads model descriptions and ranks by impact on your specific scene; produces trigger words and prompt additions

**Options:**

```bash
# Target a different base model (default: Flux.1 D)
python cli.py "your prompt" --base-model "SDXL 1.0"

# Skip LLM analysis — returns raw tag-scored results immediately (faster)
python cli.py "your prompt" --no-llm

# Machine-readable JSON output (same shape as the /recommend API response)
python cli.py "your prompt" --json

# Combine flags
python cli.py "your prompt" --base-model "SDXL 1.0" --json
```

---

### Populate / update the catalog

The catalog must be populated for a base model before you can query it. This is a one-time operation per model family; use incremental mode after that.

```bash
# Full crawl — fetches all CivitAI models for a base model (takes several minutes)
python cli.py --crawl "Flux.1 D"
python cli.py --crawl "SDXL 1.0"

# Incremental update — only fetches models added since the last crawl (fast)
python cli.py --crawl "Flux.1 D" --mode incremental

# Check what's cached and when it was last crawled
python cli.py --status
```

---

### Import TensorArt models

If you use TensorArt, the [TamperMonkey export script](https://github.com/your-repo/tensor_art_capture.user.js) can capture model data from the TensorArt website. Import the resulting JSON file to add those models to your catalog:

```bash
python cli.py --import-tensorart tensor_art_export.json
```

Imported models appear in recommendation results alongside CivitAI and HuggingFace models.

---

## Reading the output

```
═════════════════════════════════════════════════════════════════
  RECOMMENDATIONS FOR: "two knights swordfighting in heavy rain"
  Phase: 2b — LLM intent + compatibility analysis            ①
  Intent tags: sword, fight, rain, knight, armor, medieval   ②
═════════════════════════════════════════════════════════════════

RECOMMENDED COMBINATION                                        ③
  Flux Realism Pro + Combat Action LoRA
  Use 'actn_combat' at weight 0.8. DPM++ 2M sampler recommended.

ADD TO YOUR PROMPT                                             ④
  actn_combat, sword_fight, heavy rain, wet armor, dramatic lighting

CHECKPOINTS (2 results)

  #1  Flux Realism Pro  (score: 0.842)                        ⑤
       URL   : https://civitai.com/models/99999
       CFG 3.5 | Steps 28 | DPM++ 2M                          ⑥
       Note  : Strong photorealistic output; handles wet surfaces well. ⑦

LoRAs (5 results)

  #1  Combat Action LoRA  [impact: high]  (score: 0.731)      ⑧
       Note  : Trained on medieval sword combat, covers two-person choreography.
       Weight: 0.8                                             ⑨
       Triggers: actn_combat, sword_fight

  #2  Rain Effects LoRA  [impact: medium]  (score: 0.612)
       Note  : Atmospheric rain and wet surfaces; pairs cleanly with most checkpoints.
       Weight: 0.55
       Triggers: rain_fx

  #3  Anime Style LoRA  [impact: low]  [not recommended]  (score: 0.401)  ⑩
       Note  : Anime aesthetic conflicts with the realistic combat scene.
       Triggers: anime_style

═════════════════════════════════════════════════════════════════
```

| # | Field | What it means |
|---|---|---|
| ① | **Phase** | How much intelligence was applied. `1` = raw scores only; `2a` = LLM parsed the prompt but analysis was skipped; `2b` = full pipeline. |
| ② | **Intent tags** | What the LLM extracted from your prompt and used to search the catalog. If these look wrong, rephrase your prompt. |
| ③ | **Recommended combination** | The LLM's top checkpoint + LoRA pick, with brief notes on how to use them together. |
| ④ | **Add to your prompt** | Paste this directly into ComfyUI. Trigger words come first (required for LoRA activation), followed by descriptive keywords the LLM suggests based on model descriptions. |
| ⑤ | **Score** | Composite relevance score (0–1). Weighted 60% tag overlap, 25% community quality rating, 15% download popularity. Used to rank candidates before LLM analysis. |
| ⑥ | **CFG / Steps / Sampler** | Recommended generation settings sourced from the model's CivitAI metadata — not LLM-generated. Use these as your starting point in ComfyUI. |
| ⑦ | **Note** | One sentence from the compatibility analyst explaining why this model does or doesn't fit your prompt, based on its description. |
| ⑧ | **Impact** | How directly the LoRA's description matches your scene. `high` = the description addresses your prompt's core subject; `medium` = relevant but not the main focus; `low` = tangential or no description available. Sort your ComfyUI LoRA stack by impact order. |
| ⑨ | **Weight** | Suggested LoRA strength. Lower (0.4–0.6) for subtle texture or detail LoRAs; higher (0.7–0.9) for strong style or character LoRAs. Absent if the model's description gives no guidance. |
| ⑩ | **Not recommended** | The LoRA was returned by the tag scorer but the compatibility analyst determined it conflicts with your prompt. Shown for transparency; omit it from your generation. |

---

### Understanding the phase

| Phase | Condition | What it means |
|---|---|---|
| `1 — stop-word fallback` | Ollama was unavailable during intent parsing | Tags extracted by simple word splitting. Less accurate; add `--no-llm` to make this explicit. |
| `2a — LLM intent parsed` | Ollama parsed intent but analysis was skipped (`--no-llm`, or analyst unavailable) | Tags are LLM-quality but no compatibility notes or `prompt_additions`. |
| `2b — LLM intent + compatibility analysis` | Full pipeline ran successfully | Most useful output. Use `ADD TO YOUR PROMPT` and per-model notes. |

If Ollama is unavailable, the tool degrades gracefully — you still get scored results, just without the LLM-reasoning layer.

---

## REST API

The same pipeline is available as a REST API. Start the server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload
```

Interactive docs are at **http://localhost:8765/docs**.

Key endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/recommend` | Submit a prompt, get ranked recommendations with all LLM fields |
| `GET` | `/cache/status` | Which base models are cached and their model counts |
| `POST` | `/cache/crawl` | Dispatch a crawl job (dispatches a Kubernetes Job in production) |
| `POST` | `/cache/update` | Incremental update for an already-crawled base model |
| `POST` | `/catalog/import/tensorart` | Upload a TensorArt export JSON file |

Example request:

```bash
curl -X POST http://localhost:8765/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "two knights swordfighting in heavy rain",
    "base_model": "Flux.1 D",
    "max_loras": 5,
    "llm_reasoning": true
  }'
```

The JSON response matches the shape of `python cli.py --json` output — `intent`, `phase`, `checkpoints`, `loras`, `recommended_combination`, `combination_notes`, and `prompt_additions`.

---

## Data sources

| Source | How data gets in | Coverage |
|---|---|---|
| **CivitAI** | `--crawl` CLI or `POST /cache/crawl` API | Checkpoints and LoRAs; trigger words, tags, settings, download stats |
| **HuggingFace** | Same crawl infrastructure | Diffusers-format models; tags and base model info |
| **TensorArt** | TamperMonkey export → `--import-tensorart` | Models you've personally browsed on TensorArt |

All sources are scored and ranked together. There is no source-level preference.

---

## Project structure

```
.
├── agents/
│   ├── intent_parser.py          LLM call 1 — prompt → structured tags, style, subject
│   ├── catalog_query.py          Pure SQL scoring — no LLM
│   └── compatibility_analyst.py  LLM call 2 — descriptions → impact, notes, prompt_additions
├── api/
│   └── main.py                   FastAPI service
├── crawler/
│   ├── civitai_crawler.py        CivitAI paginated crawler
│   ├── hf_crawler.py             HuggingFace crawler
│   └── tensorart_crawler.py      TensorArt Nuxt export parser
├── db/
│   ├── schema.sql                PostgreSQL schema (applied automatically on first run)
│   └── database.py               asyncpg connection pool
├── tests/                        227 tests, 93% coverage — all mocked, no network/DB required
├── cli.py                        Command-line interface
├── docker-compose.yml            Local dev stack (PostgreSQL + API)
├── .env.example                  Environment variable template
├── README.md                     This file
├── INSTALLATION.md               Setup guide
└── DEVELOPER_GUIDE.md            CI, testing, coverage, Kubernetes deployment
```

---

## Further reading

- **[INSTALLATION.md](INSTALLATION.md)** — prerequisites, Python environment, database, Ollama models, first crawl
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** — CI pipeline, running tests, code coverage, Kubernetes deployment
