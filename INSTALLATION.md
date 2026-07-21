# Installation

Two deployment paths are supported. Choose the one that fits your setup:

| | **Docker Compose** | **Kubernetes (k3s)** |
|---|---|---|
| **Best for** | Single machine, local use | Persistent server, always-on API |
| **PostgreSQL** | Managed by Compose | Deployed as a k8s Deployment |
| **Crawler** | Profile-gated one-shot container | Dispatched as k8s Jobs by the API |
| **API access** | `http://localhost:8765` | `http://<node-ip>:30765` |
| **Web UI access** | `http://localhost:3000` | `http://<node-ip>:30766` |
| **Complexity** | Low | Medium |

**Ollama always runs on the host** — not in Docker or Kubernetes — because it needs direct GPU access. Both paths reach it via a configurable `OLLAMA_HOST` address.

**VRAM note:** Ollama and ComfyUI share the GPU and cannot run simultaneously on a 16 GB card. ModelRecommender is a planning tool — run it when ComfyUI is idle.

---

## Common prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Docker | 24+ | Required for both paths |
| Ollama | Latest | On the host machine, not in a container |
| CivitAI API token | — | Free; civitai.com → Account → Settings → API Keys |
| HuggingFace token | — | Optional; only needed to crawl HuggingFace models |
| k3s / kubectl | Latest | Kubernetes path only |
| Python 3.10+ | — | Only needed to run the CLI locally (see [Running the CLI](#running-the-cli)) |

---

## Ollama setup (required for both paths)

Ollama must be installed on the host machine so it has direct GPU access. The Docker and Kubernetes containers reach it over the network.

**Install Ollama:**

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows
# Download the installer from https://ollama.ai
```

**Pull the required models:**

```bash
# Primary — uncensored Llama 3.1 8B; fits alongside other GPU work in 16 GB
ollama pull dolphin3.0-llama3.1:8b

# Fallback — higher reasoning quality; use when ComfyUI is not loaded
ollama pull gemma3:12b
```

**Start Ollama and verify it is reachable:**

```bash
ollama serve

curl http://localhost:11434/api/tags
# Should return a JSON list of your pulled models
```

---

## Option A: Docker Compose

The quickest path for a single machine. PostgreSQL and the API run in containers; Ollama stays on the host.

### 1. Clone the repository

```bash
git clone https://github.com/your-org/ModelRecommender.git
cd ModelRecommender
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` — at minimum, set these three values:

```env
CIVITAI_API_TOKEN=your_civitai_api_token_here
DATABASE_URL=postgresql://recommender:recommender@localhost:5432/model_catalog
OLLAMA_HOST=http://localhost:11434
```

`DATABASE_URL` uses `localhost` because the CLI connects from the host. The API container uses the internal `postgres` hostname automatically (pre-configured in `docker-compose.yml`).

**Never commit `.env` to source control.** It is listed in `.gitignore`.

### 3. Start the stack

```bash
docker compose up -d
```

This starts PostgreSQL, the API, and the web UI. The API applies the database schema automatically on startup.

Verify:

```bash
docker compose ps
curl http://localhost:8765/health
# {"status":"ok"}
```

- **Web UI**: http://localhost:3000
- **Swagger UI**: http://localhost:8765/docs

### 4. Seed the catalog

Run a one-shot crawler container to populate the catalog. This takes several minutes — CivitAI has thousands of models per base model.

```bash
# Full crawl for Flux.1 D
CIVITAI_API_TOKEN=your_token CRAWL_BASE_MODEL="Flux.1 D" \
  docker compose --profile crawl run --rm crawler

# Add more base models as needed
CIVITAI_API_TOKEN=your_token CRAWL_BASE_MODEL="SDXL 1.0" \
  docker compose --profile crawl run --rm crawler
```

Check what was cached:

```bash
curl http://localhost:8765/cache/status
```

Expected:

```json
{
  "total_models_cached": 4823,
  "base_models": [
    {"base_model_name": "Flux.1 D", "crawl_complete": true, "total_models": 4823}
  ]
}
```

### 5. Get recommendations

**Via the API** (no local Python required):

```bash
curl -s -X POST http://localhost:8765/recommend \
  -H "Content-Type: application/json" \
  -d '{"prompt": "two knights swordfighting in heavy rain", "base_model": "Flux.1 D"}' \
  | python -m json.tool
```

**Via the CLI** (runs inside the API container):

```bash
docker compose exec api python cli.py "two knights swordfighting in heavy rain"

# With flags
docker compose exec api python cli.py "your prompt" --no-llm
docker compose exec api python cli.py "your prompt" --json
docker compose exec api python cli.py --status
```

### 6. Import TensorArt models

Use the API's upload endpoint — no need to copy files into the container:

```bash
curl -X POST http://localhost:8765/catalog/import/tensorart \
  -F "file=@tensor_art_export.json"
```

Or via the Swagger UI at **http://localhost:8765/docs** → `POST /catalog/import/tensorart`.

### 7. Keep the catalog fresh

```bash
# Incremental update — only fetches models added since the last crawl (fast)
CIVITAI_API_TOKEN=your_token CRAWL_BASE_MODEL="Flux.1 D" CRAWL_MODE=incremental \
  docker compose --profile crawl run --rm crawler
```

---

## Option B: Kubernetes (k3s)

For a persistent server deployment on Ubuntu. The API and PostgreSQL run as Deployments; crawlers run as on-demand Jobs dispatched by the API itself.

### 1. Clone and build the image

```bash
git clone https://github.com/your-org/ModelRecommender.git
cd ModelRecommender

docker build -t model-recommender:latest .
```

Import the image into k3s containerd (bypasses a registry):

```bash
docker save model-recommender:latest | sudo k3s ctr images import -
```

### 2. Create the namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 3. Configure secrets

Copy the example and fill in your values:

```bash
cp k8s/secret.example.yaml k8s/secret.yaml
```

Edit `k8s/secret.yaml`:

```yaml
stringData:
  database-url: "postgresql://recommender:your_password@postgres:5432/model_catalog"
  postgres-password: "your_password"
  civitai-api-token: "your_civitai_token"
  hf-token: "your_hf_token"              # leave as CHANGE_ME if not using HuggingFace
  ollama-host: "http://192.168.1.x:11434" # your node's LAN IP — NOT localhost
```

`ollama-host` must be the node's LAN IP address, not `localhost`, because the API pod reaches Ollama over the host network.

Apply the secret:

```bash
kubectl apply -f k8s/secret.yaml
```

**`k8s/secret.yaml` is listed in `.gitignore` and must never be committed.**

### 4. Build and import the UI image

```bash
docker build -t model-recommender-ui:latest ./frontend
docker save model-recommender-ui:latest | sudo k3s ctr images import -
```

### 5. Apply all manifests

Order matters — PostgreSQL must be running before the API starts:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/api/
kubectl apply -f k8s/frontend/
kubectl apply -f k8s/crawler/cronjob.yaml   # scheduled incremental updates
```

Wait for everything to be ready:

```bash
kubectl rollout status deployment/model-recommender-api -n model-recommender
kubectl rollout status deployment/model-recommender-ui  -n model-recommender
```

- **Web UI**: http://\<node-ip\>:30766
- **Swagger UI**: http://\<node-ip\>:30765/docs

### 6. Seed the catalog

Apply the one-shot crawler Job and follow its logs:

```bash
kubectl apply -f k8s/crawler/job.yaml
kubectl logs -n model-recommender -l job-name=civitai-crawl-flux1d -f
```

After the job completes, verify via the API:

```bash
curl http://<node-ip>:30765/cache/status
```

For additional base models, trigger crawls through the API — it dispatches a new Job automatically:

```bash
curl -X POST http://<node-ip>:30765/cache/crawl \
  -H "Content-Type: application/json" \
  -d '{"base_model": "SDXL 1.0", "mode": "full"}'
```

### 7. Get recommendations

Open the **web UI** at http://\<node-ip\>:30766 — enter a prompt, select a base model, and click Get Recommendations.

Alternatively, via curl:

```bash
curl -s -X POST http://<node-ip>:30765/recommend \
  -H "Content-Type: application/json" \
  -d '{"prompt": "two knights swordfighting in heavy rain", "base_model": "Flux.1 D"}' \
  | python -m json.tool
```

Or exec into the API pod for CLI access:

```bash
kubectl exec -n model-recommender deployment/model-recommender-api -- \
  python cli.py "two knights swordfighting in heavy rain"
```

### 8. Import TensorArt models

Use the **Catalog** tab in the web UI, or the upload endpoint directly:

```bash
curl -X POST http://<node-ip>:30765/catalog/import/tensorart \
  -F "file=@tensor_art_export.json"
```

### 9. Keep the catalog fresh

The CronJob at `k8s/crawler/cronjob.yaml` runs incremental updates on a schedule automatically. To trigger one manually:

```bash
kubectl create job --from=cronjob/civitai-incremental-update manual-update \
  -n model-recommender
kubectl logs -n model-recommender -l job-name=manual-update -f
```

### Rebuild and redeploy after code changes

```bash
# API
docker build -t model-recommender:latest .
docker save model-recommender:latest | sudo k3s ctr images import -
kubectl rollout restart deployment/model-recommender-api -n model-recommender

# UI
docker build -t model-recommender-ui:latest ./frontend
docker save model-recommender-ui:latest | sudo k3s ctr images import -
kubectl rollout restart deployment/model-recommender-ui -n model-recommender
```

---

## Running the CLI locally

If you want to run `python cli.py` directly on the host (for development, or when Docker Compose is managing only PostgreSQL), set up a local Python environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -r requirements.txt
```

Ensure `DATABASE_URL` in `.env` points to the running PostgreSQL instance (Docker Compose's on `localhost:5432`, or your remote k8s cluster via a port-forward).

```bash
# Port-forward the k8s PostgreSQL for local CLI access
kubectl port-forward -n model-recommender svc/postgres 5432:5432 &

# Then run the CLI normally
python cli.py "two knights swordfighting in heavy rain"
```

For running tests, also install dev dependencies:

```bash
pip install -r requirements-dev.txt
pytest
```

Tests are fully mocked — no database, Ollama, or network access required to run the suite.

---

## Troubleshooting

**`ERROR: 'Flux.1 D' is not in the cache yet.`**
The catalog hasn't been populated. Run the crawler (step 4 in whichever path you chose).

**`Phase: 1 — stop-word fallback` instead of `2b`**
Ollama is not running, the model is not loaded, or `OLLAMA_HOST` points to the wrong address. Verify with `curl $OLLAMA_HOST/api/tags`. In Kubernetes, make sure `ollama-host` in the Secret uses the node's LAN IP, not `localhost`.

**API returns 409 on `/recommend`**
The requested `base_model` is not yet cached. Call `POST /cache/crawl` first.

**`asyncpg` connection refused or invalid password**
PostgreSQL is not running, or `DATABASE_URL` / `database-url` credentials don't match. Check `docker compose ps` or `kubectl get pods -n model-recommender`.

**Crawl stalls or gets 429 errors**
CivitAI rate-limits requests. The crawler retries automatically with exponential backoff. If it doesn't recover after a few minutes, interrupt and re-run — the catalog is populated incrementally and won't lose progress.

**`RuntimeError: Form data requires "python-multipart"`**
Only occurs when running the API outside Docker/k8s. The package is in `requirements.txt` — activate your virtualenv before running `uvicorn`.

**Image not found in k3s after rebuild**
Re-import after every rebuild: `docker save model-recommender:latest | sudo k3s ctr images import -`. k3s does not pull from the Docker daemon; it reads from its own containerd store.
