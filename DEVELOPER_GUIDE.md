# Developer Guide — Model Recommendation System

## Prerequisites

- Python 3.11+
- Docker (for local development stack)
- kubectl + k3s (for Kubernetes deployment — see deployment section)

---

## CI pipeline (GitHub Actions)

The workflow at [`.github/workflows/test.yml`](.github/workflows/test.yml) runs automatically on every push and pull request to `main` or `dev`.

### What it does

| Step | Tool | Where to see the output |
|---|---|---|
| Run tests + measure coverage | `pytest --cov` | Actions → job log |
| Coverage summary table | `coverage report --format=markdown` | Actions → job → **Summary** tab |
| Pass / fail / skip counts | `EnricoMi/publish-unit-test-result-action` | PR page → **Checks** tab |
| Failing test annotations | same action | PR page → **Files changed** diff |
| Coverage badge + file table | `py-cov-action/python-coverage-comment-action` | PR page → comment thread |
| Full HTML report (downloadable) | `actions/upload-artifact` | Actions → run → **Artifacts** |

### Coverage gate

The pipeline fails if total branch coverage drops below **80%** (`fail_under = 80` in `.coveragerc`).
Reports are still generated and uploaded on failure so you can see exactly what dropped.

Badge colours on PRs:

| Colour | Meaning |
|---|---|
| 🟢 Green | ≥ 80% — pipeline passes |
| 🟡 Orange | 70–79% — pipeline **fails** |
| 🔴 Red | < 70% — pipeline **fails** |

### Required repository permissions

The workflow uses `GITHUB_TOKEN` (automatically provided by GitHub — no setup needed).
The token needs these permissions, which are declared in the workflow file:

```
contents: read        # checkout
pull-requests: write  # post/update coverage comment
checks: write         # publish test results check run
```

For **public repos** these are granted automatically. For **private repos** go to:
`Settings → Actions → General → Workflow permissions` and confirm
*"Read and write permissions"* is selected (or GitHub will prompt you on first run).

---

## Local development setup

The fastest way to run the stack locally is docker-compose, which starts PostgreSQL and the API together.

```bash
# 1. Clone and enter the project
cd /opt/ModelRecommender

# 2. Copy and fill in secrets
cp .env.example .env
# Edit .env — set CIVITAI_API_TOKEN at minimum

# 3. Create a virtual environment for local tooling / tests
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 4. Start the stack
docker compose up -d

# 5. Verify the API is up
curl http://localhost:8765/health
```

The API's Swagger UI is available at `http://localhost:8765/docs`.

### Running the crawler locally (docker-compose)

```bash
# Full crawl for Flux.1 D
CIVITAI_API_TOKEN=your_token CRAWL_BASE_MODEL="Flux.1 D" \
  docker compose --profile crawl run --rm crawler

# Incremental update
CIVITAI_API_TOKEN=your_token CRAWL_BASE_MODEL="Flux.1 D" CRAWL_MODE=incremental \
  docker compose --profile crawl run --rm crawler
```

### Running the CLI directly

```bash
source .venv/bin/activate

# Trigger a crawl (writes directly to the local DB)
python cli.py --crawl "Flux.1 D"

# Query the catalog
python cli.py "two characters swordfighting in the rain"

# JSON output
python cli.py "sword fight rain" --json

# Cache status
python cli.py --status
```

---

## Running the tests

All tests are fully mocked — no database, no network calls, no API keys required.

```bash
# Activate the virtualenv if not already active
source .venv/bin/activate

# Install test dependencies (only needed once)
pip install -r requirements.txt -r requirements-dev.txt

# Run the full suite
pytest

# Verbose output (shows each test name)
pytest -v

# Run a single test file
pytest tests/test_crawler.py -v
pytest tests/test_catalog_query.py -v
pytest tests/test_api.py -v

# Run a single test by name
pytest -v -k "test_full_crawl_follows_next_page_url"

# Stop on first failure
pytest -x

# Show local variable values on failure
pytest -v --tb=short
```

---

## Code coverage

Coverage is measured with [pytest-cov](https://pytest-cov.readthedocs.io/) (backed by [coverage.py](https://coverage.py)).
Configuration lives in `.coveragerc`. **Branch coverage** is enabled, which catches untested conditional
paths in addition to untested lines.

### Running with coverage

```bash
# Terminal report — shows which lines are not covered
pytest --cov

# Terminal report with branch details (verbose)
pytest --cov --cov-report=term-missing

# Generate an HTML report (open htmlcov/index.html in a browser)
pytest --cov --cov-report=html

# Generate both terminal and HTML reports at once
pytest --cov --cov-report=term-missing --cov-report=html

# Generate an XML report (for CI systems such as GitHub Actions, GitLab CI)
pytest --cov --cov-report=xml
```

The HTML report at `htmlcov/index.html` is the most useful for exploring uncovered branches —
click any file to see line-by-line highlighting with branch arcs.

### What is measured

| Included | Excluded |
|---|---|
| `api/` | `output/` — Phase 2 stubs, not yet implemented |
| `agents/` | `*/__init__.py` — empty package markers |
| `crawler/` | `tests/` — test code itself |
| `db/` | |

### Coverage threshold

The build fails if total coverage drops below **70%** (`fail_under = 70` in `.coveragerc`).
Raise this value as the suite grows — a reasonable target for a mature codebase is 85–90%.

```bash
# Check whether the current suite meets the threshold
pytest --cov
# Exit code 2 = threshold not met
```

### Interpreting the terminal report

```
Name                          Stmts   Miss Branch BrPart  Cover   Missing
-------------------------------------------------------------------------
agents/catalog_query.py          38      2     12      1   93.2%   45, 67
api/main.py                      89      8     20      3   88.1%   102-108
crawler/civitai_crawler.py      124     11     34      4   87.6%   178, 201-205
db/database.py                   22      3      4      1   84.1%   23-25
-------------------------------------------------------------------------
TOTAL                           273     24     70      9   89.4%
```

- **Stmts** — executable lines
- **Miss** — lines never executed by any test
- **Branch** — total conditional branches (if/else, while, etc.)
- **BrPart** — branches where only one direction was tested
- **Missing** — line numbers to investigate; `45-48` means lines 45 through 48

### Excluding lines from coverage

Occasionally a line is intentionally untestable (e.g. a defensive `# pragma: no cover` guard).
Use sparingly:

```python
if sys.platform == "win32":  # pragma: no cover
    ...
```

---

## Test coverage

The test suite covers **Phase 1** functionality across three files. Tests are intentionally fast and isolated — the entire suite runs in a few seconds.

### `tests/test_crawler.py` — 22 tests

Covers `crawler/civitai_crawler.py`. All HTTP calls are intercepted by [respx](https://lundberg.github.io/respx/); no real Civitai API traffic is made.

| Area | What is tested |
|---|---|
| `_strip_html` | Tag removal, HTML entity decoding (`&amp;`, `&lt;`), whitespace collapse, None/empty input |
| `_extract_record` | Full field mapping from Civitai API response to DB record; generation params extracted from first image meta; HTML stripped from description; recommended_weight is None (set by LLM in Phase 2); graceful handling of missing versions, missing version ID, empty images, missing stats |
| `full_crawl` | Single-page crawl upserts all records; multi-page crawl follows `nextPage` URL; items with no modelVersions are skipped; `base_model_index` is marked complete after crawl; 429 response triggers exactly one backoff sleep before retry |
| `incremental_update` | Falls back to full crawl when no prior crawl exists; stops pagination immediately when a known `civitai_version_id` is encountered; upserts genuinely new models when no known IDs match |

### `tests/test_catalog_query.py` — 19 tests

Covers `agents/catalog_query.py`. The asyncpg pool is replaced with `mock_pool`/`mock_conn` fixtures.

| Area | What is tested |
|---|---|
| `_score` | Full tag overlap gives 0.6 tag component; partial overlap scales proportionally; trigger words count toward tag overlap; matching is case-insensitive; zero overlap falls back to quality + download signals; empty search tags give zero overlap; quality signal differentiates equal tag scores; no votes gives zero quality; download count contributes positively |
| `query_catalog` | Results split into `checkpoints` and `loras` keys; `LoCon` type grouped with loras (not checkpoints); results sorted by `relevance_score` descending; `limit_loras` parameter respected; `nsfw_max` filter added to SQL when set; every result has a float `relevance_score` field; empty cache returns empty lists; correct `base_model` value passed to SQL |
| `ensure_base_model_cached` | Returns `True` when `crawl_complete = TRUE`; returns `False` when `crawl_complete = FALSE`; returns `False` when base model is not in `base_model_index` at all |

### `tests/test_api.py` — 13 tests

Covers `api/main.py` FastAPI endpoints. The lifespan is patched to inject a mock pool; all agent functions are patched per-test.

| Endpoint | What is tested |
|---|---|
| `GET /health` | Returns 200 `{"status": "ok"}` |
| `GET /cache/status` | Returns `base_models` list and `total_models_cached`; returns empty lists for an unpopulated catalog |
| `POST /recommend` | Returns 409 with informative message when base model is not cached; returns `checkpoints` and `loras` when cached; prompt words are split into search tags with short words filtered; `nsfw_filter: true` propagates `nsfw_max=1` to `query_catalog`; response includes a `phase` note indicating Phase 1 raw results |
| `POST /cache/crawl` | Returns `job_created` status when k8s Job is dispatched successfully; returns `kubernetes_unavailable` gracefully when k8s is not reachable; inserts a `cache_requests` record on every call |
| `POST /cache/update` | Forces `mode` to `incremental` regardless of the value in the request body |

---

## Test fixtures and sample data

**`tests/conftest.py`** contains the shared fixtures used across all test files.

### Civitai API sample data

`CIVITAI_LORA` and `CIVITAI_CHECKPOINT` are the canonical fixtures representing real Civitai API response shapes. They are used directly in crawler tests to verify field mapping.

```
CIVITAI_LORA        — a LORA with tags, trigger words, image meta (CFG/steps/sampler), stats
CIVITAI_CHECKPOINT  — a Checkpoint with no trigger words
CIVITAI_NSFW_LORA   — a LORA with nsfwLevel=8 for filter testing
CIVITAI_NO_VERSIONS — a model with an empty modelVersions array (should be skipped by crawler)
```

**Adding a new source (e.g. HuggingFace in Phase 3):** add a `HF_MODEL` constant to `conftest.py` representing a real HuggingFace API response, then write a `tests/test_hf_crawler.py` following the same pattern as `test_crawler.py`.

### Database fixtures

```python
mock_conn   # AsyncMock of an asyncpg connection with execute/executemany/fetch/fetchrow/fetchval
mock_pool   # MagicMock pool whose acquire() context manager yields mock_conn
make_db_model(**overrides)  # Returns a dict shaped like a row from the models table
```

Use `make_db_model` to build test inputs for scoring and query tests:

```python
model = make_db_model(
    type="Checkpoint",
    tags=["realistic", "cinematic"],
    stats_thumbs_up=1000,
    stats_downloads=500000,
)
```

---

## Adding tests for new functionality

### New crawler field

1. Add the field to `CIVITAI_LORA` in `conftest.py`
2. Add an assertion to `test_extract_record_maps_all_core_fields` in `test_crawler.py`
3. Add a dedicated test for edge cases (missing field, wrong type, etc.)

### New API endpoint

1. Add a test function to `test_api.py`
2. Use the `client` fixture for the HTTP call
3. Patch agent functions with `patch("api.main.function_name", new=AsyncMock(...))` to isolate the endpoint logic from the agents

### New scoring signal (Phase 2+)

1. Update the weight constants in `catalog_query.py`
2. Update `test_score_full_tag_overlap_gives_maximum_tag_component` to reflect the new maximum
3. Add a dedicated test for the new signal's contribution

---

## Project structure

```
.
├── api/
│   └── main.py               FastAPI service; dispatches k8s Jobs for crawls
├── agents/
│   ├── catalog_query.py      Tag-scoring query against PostgreSQL; no LLM
│   ├── intent_parser.py      (Phase 2) Ollama LLM call 1 — prompt → structured intent
│   └── compatibility_analyst.py  (Phase 2) Ollama LLM call 2 — candidates → recommendations
├── crawler/
│   └── civitai_crawler.py    Civitai API crawler; k8s Job entrypoint via python -m
├── db/
│   ├── schema.sql            PostgreSQL schema (idempotent)
│   └── database.py           asyncpg connection pool
├── output/
│   └── formatter.py          (Phase 2) Human-readable and JSON output rendering
├── tests/
│   ├── conftest.py           Shared fixtures and Civitai API sample data
│   ├── test_crawler.py       Crawler unit tests (respx mocks HTTP)
│   ├── test_catalog_query.py Scoring and query tests (asyncpg mocked)
│   └── test_api.py           FastAPI endpoint tests (lifespan + agents mocked)
├── k8s/
│   ├── postgres/             PV, PVC, Deployment, Service, StorageClass
│   ├── api/                  Deployment, NodePort Service, ServiceAccount + RBAC
│   └── crawler/              one-off Job and 6-hourly CronJob
├── Dockerfile                Single image; API by default, crawler via command override
├── docker-compose.yml        Local dev stack
├── requirements.txt          Runtime dependencies
├── requirements-dev.txt      Test-only dependencies (pytest, pytest-asyncio, respx)
└── pytest.ini                asyncio_mode = auto
```

---

## Kubernetes deployment (Ubuntu single-node)

See `k8s/secret.example.yaml` for the full setup sequence. Quick reference:

```bash
# Build and import the image into k3s containerd
docker build -t model-recommender:latest .
docker save model-recommender:latest | sudo k3s ctr images import -

# Apply manifests (order matters)
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml        # copied from secret.example.yaml and filled in
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/api/
kubectl apply -f k8s/crawler/cronjob.yaml

# Seed the initial cache
kubectl apply -f k8s/crawler/job.yaml
kubectl logs -n model-recommender -l job-name=civitai-crawl-flux1d -f

# Access
# Swagger UI:  http://<node-ip>:30765/docs
# Health:      http://<node-ip>:30765/health

# Re-import image after a rebuild
docker build -t model-recommender:latest .
docker save model-recommender:latest | sudo k3s ctr images import -
kubectl rollout restart deployment/model-recommender-api -n model-recommender
```
