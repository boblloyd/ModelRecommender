# Installation

This guide covers everything from a fresh machine to a working `python cli.py "your prompt"`.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11+ recommended |
| PostgreSQL | 14+ | Or Docker — see step 3 |
| Ollama | Latest | Must be installed on the host with GPU access |
| CivitAI API token | — | Free; get one at civitai.com → Account → Settings → API Keys |
| HuggingFace token | — | Optional; only needed to crawl HuggingFace models |
| Docker | — | Optional; simplifies PostgreSQL setup |

**VRAM note:** Ollama and ComfyUI share the GPU and cannot run at the same time on a 16 GB card. ModelRecommender is a planning tool — run it when ComfyUI is idle, then close Ollama before launching ComfyUI for generation.

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/ModelRecommender.git
cd ModelRecommender
```

---

## Step 2 — Create a Python virtual environment

```bash
# Create
python -m venv .venv

# Activate (Linux / macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install runtime dependencies
pip install -r requirements.txt
```

To run tests, also install the dev dependencies:

```bash
pip install -r requirements-dev.txt
```

---

## Step 3 — Set up PostgreSQL

Choose one of the options below.

### Option A: Docker Compose (easiest)

```bash
# Starts PostgreSQL on port 5432 with the credentials from docker-compose.yml
docker compose up -d postgres

# Verify it started
docker compose ps
```

### Option B: Local PostgreSQL

If you have PostgreSQL installed:

```bash
# Create the database and user
psql -U postgres -c "CREATE USER recommender WITH PASSWORD 'recommender';"
psql -U postgres -c "CREATE DATABASE model_catalog OWNER recommender;"
```

Adjust the credentials to match what you put in `.env` (step 4).

---

## Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the values:

```env
# Required
CIVITAI_API_TOKEN=your_civitai_api_token_here
DATABASE_URL=postgresql://recommender:recommender@localhost:5432/model_catalog

# Required for LLM features
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_PRIMARY=dolphin3.0-llama3.1:8b
OLLAMA_MODEL_FALLBACK=gemma3:12b

# Optional — only needed to crawl HuggingFace
HF_TOKEN=your_huggingface_token_here
```

**Never commit `.env` to source control.** It is listed in `.gitignore`.

**Finding your CivitAI token:**
1. Log in at [civitai.com](https://civitai.com)
2. Go to Account → Settings → API Keys
3. Create a new key and copy it

---

## Step 5 — Install and configure Ollama

Ollama must be installed directly on the host (not in Docker) so it has GPU access.

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows
# Download the installer from https://ollama.ai
```

Pull the LLM models this tool uses:

```bash
# Primary model — uncensored Llama 3.1 8B, fits in 16 GB alongside other work
ollama pull dolphin3.0-llama3.1:8b

# Fallback model — higher reasoning quality, use when ComfyUI is not loaded
ollama pull gemma3:12b
```

Make sure Ollama is running before using the CLI or API:

```bash
ollama serve
```

You can verify it is reachable:

```bash
curl http://localhost:11434/api/tags
```

---

## Step 6 — Seed the catalog

The database schema is applied automatically on first use. The following command creates the schema and populates it with models for Flux.1 D from CivitAI. This takes several minutes depending on how many models exist in the catalog.

```bash
python cli.py --crawl "Flux.1 D"
```

To add additional base models:

```bash
python cli.py --crawl "SDXL 1.0"
python cli.py --crawl "Pony"
```

Check what is cached:

```bash
python cli.py --status
```

Expected output:

```
Total models in catalog: 4823

  Flux.1 D              COMPLETE    4823 models  last: 2026-07-21 14:30
```

---

## Step 7 — Verify the installation

```bash
python cli.py "two knights swordfighting in heavy rain"
```

You should see a formatted recommendations block. If Ollama is running and the primary model is loaded, the `Phase` line will read `2b — LLM intent + compatibility analysis`. If Ollama is not running, it degrades to `1 — stop-word fallback` and still returns scored results.

---

## Optional: Import TensorArt models

If you use TensorArt and have the [TamperMonkey export script](https://github.com/your-repo/tensor_art_capture.user.js) installed:

1. Browse to a model page on TensorArt in your browser
2. Click the TamperMonkey export button — this captures the model's metadata
3. Export the collected data as JSON
4. Import it:

```bash
python cli.py --import-tensorart tensor_art_export.json
```

Imported models appear in recommendation results alongside CivitAI and HuggingFace models.

---

## Optional: Run the REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload
```

Swagger UI: **http://localhost:8765/docs**
Health check: **http://localhost:8765/health**

### Full stack with Docker Compose

Starts PostgreSQL and the API together:

```bash
docker compose up -d
```

The API will be available at **http://localhost:8765**. Ollama still runs on the host and is reached from Docker via `host.docker.internal:11434` (configured automatically in `docker-compose.yml`).

To run a crawl inside the compose environment:

```bash
CIVITAI_API_TOKEN=your_token CRAWL_BASE_MODEL="Flux.1 D" \
  docker compose --profile crawl run --rm crawler
```

---

## Optional: Kubernetes deployment

For persistent server deployment on Ubuntu with k3s, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#kubernetes-deployment-ubuntu-single-node). The key difference from local dev is that Ollama must be reachable from inside the cluster — set `OLLAMA_HOST` in `k8s/secret.yaml` to your node's LAN IP (e.g. `http://192.168.1.x:11434`), not `localhost`.

---

## Keeping the catalog fresh

The catalog does not update automatically. Run incremental updates periodically to pick up newly published models:

```bash
# Updates only — much faster than a full crawl
python cli.py --crawl "Flux.1 D" --mode incremental
```

A full re-crawl is only needed if you want to reset the catalog or if the model count seems significantly lower than CivitAI shows.

---

## Troubleshooting

**`ERROR: 'Flux.1 D' is not in the cache yet.`**
Run `python cli.py --crawl "Flux.1 D"` first.

**`Phase: 1 — stop-word fallback` (instead of 2b)**
Ollama is not running or the primary model is not loaded. Run `ollama serve` and `ollama pull dolphin3.0-llama3.1:8b`, then retry.

**`asyncpg.exceptions.InvalidPasswordError` or connection refused**
Check that PostgreSQL is running and that `DATABASE_URL` in `.env` matches your database credentials.

**Crawl stalls or returns 429 errors**
CivitAI rate-limits unauthenticated and occasionally authenticated requests. The crawler retries with exponential backoff automatically. If it stalls for more than a few minutes, press Ctrl+C and re-run — the catalog is populated incrementally and the next run picks up where it left off via the incremental mode.

**`RuntimeError: Form data requires "python-multipart"`**
The TensorArt import endpoint requires `python-multipart`. Install it:
```bash
pip install python-multipart
```
It is already listed in `requirements.txt` — this usually means you're running without the virtualenv active.
