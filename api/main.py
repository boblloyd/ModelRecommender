"""
FastAPI service for the Model Recommendation system.

Endpoints:
  GET  /health           — liveness check
  GET  /cache/status     — cached base model stats
  POST /cache/crawl      — trigger a full crawl Job for a base model
  POST /cache/update     — trigger an incremental update Job
  POST /recommend        — score and rank cached models against a prompt

The /cache/crawl and /cache/update endpoints dispatch a Kubernetes Job
(same image, different command) so the crawler pod starts, runs to completion,
and shuts down — leaving the API pod untouched.

Swagger UI is available at /docs.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.catalog_query import ensure_base_model_cached, query_catalog
from db.database import close_pool, get_pool, init_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_pool()
    await init_schema(pool)
    app.state.pool = pool
    yield
    await close_pool()


app = FastAPI(
    title="Model Recommendation API",
    version="1.0.0",
    description=(
        "AI image generation model and LoRA recommendation service. "
        "Queries a local Civitai/HuggingFace catalog and (Phase 2) uses "
        "a local Ollama LLM for compatibility reasoning."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

async def _get_pool(request: Request) -> asyncpg.Pool:
    return request.app.state.pool


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    prompt: str
    base_model: str = "Flux.1 D"
    max_loras: int = 5
    nsfw_filter: bool = False


class CrawlRequest(BaseModel):
    base_model: str
    mode: str = "full"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}


@app.get("/cache/status", tags=["cache"])
async def cache_status(pool: asyncpg.Pool = Depends(_get_pool)) -> dict:
    """Return which base models are cached and their model counts."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT base_model_name, last_crawled, total_models, crawl_complete
            FROM base_model_index
            ORDER BY base_model_name
            """
        )
        total = await conn.fetchval("SELECT COUNT(*) FROM models")
    return {
        "total_models_cached": total,
        "base_models": [dict(r) for r in rows],
    }


@app.post("/cache/crawl", tags=["cache"])
async def crawl(
    req: CrawlRequest,
    pool: asyncpg.Pool = Depends(_get_pool),
) -> dict:
    """
    Dispatch a Kubernetes Job to crawl Civitai for the given base model.

    The Job pod starts, runs the crawl to completion, then exits automatically.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO cache_requests (base_model_name, triggered_by) VALUES ($1, $2)",
            req.base_model, "api",
        )
    return _dispatch_job(req.base_model, req.mode)


@app.post("/cache/update", tags=["cache"])
async def update_cache(
    req: CrawlRequest,
    pool: asyncpg.Pool = Depends(_get_pool),
) -> dict:
    """Dispatch an incremental update Job for an already-crawled base model."""
    req.mode = "incremental"
    return await crawl(req, pool)


@app.post("/recommend", tags=["recommend"])
async def recommend(
    req: RecommendRequest,
    pool: asyncpg.Pool = Depends(_get_pool),
) -> dict[str, Any]:
    """
    Return ranked model and LoRA recommendations for a natural language prompt.

    Phase 1: raw tag-scored results from the catalog.
    Phase 2 (future): LLM compatibility reasoning via local Ollama.
    """
    cached = await ensure_base_model_cached(req.base_model, pool)
    if not cached:
        raise HTTPException(
            status_code=409,
            detail=(
                f"'{req.base_model}' is not in the cache yet. "
                f"POST /cache/crawl with {{\"base_model\": \"{req.base_model}\"}} first."
            ),
        )

    # Phase 1: split the prompt into search tags directly (LLM intent parser is Phase 2)
    search_tags = [w.strip(".,!?;:\"'") for w in req.prompt.lower().split() if len(w) > 2]

    results = await query_catalog(
        search_tags=search_tags,
        pool=pool,
        base_model=req.base_model,
        nsfw_max=1 if req.nsfw_filter else None,
        limit_checkpoints=5,
        limit_loras=req.max_loras,
    )

    return {
        "prompt": req.prompt,
        "base_model": req.base_model,
        "search_tags_used": search_tags,
        "phase": "1 — raw scored results (LLM reasoning added in Phase 2)",
        **results,
    }


# ---------------------------------------------------------------------------
# Kubernetes Job dispatch
# ---------------------------------------------------------------------------

def _dispatch_job(base_model: str, mode: str) -> dict:
    """
    Create a Kubernetes Job that runs the crawler in an ephemeral pod.
    The pod exits when the crawl finishes.  Falls back gracefully when
    running outside of k8s (local dev / docker-compose).
    """
    try:
        from kubernetes import client as k8s
        from kubernetes import config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()

        namespace = os.environ.get("K8S_NAMESPACE", "model-recommender")
        image = os.environ.get("CRAWLER_IMAGE", "model-recommender:latest")
        safe_name = base_model.lower().replace(".", "").replace(" ", "-").replace("_", "-")
        job_name = f"crawl-{safe_name}-{int(time.time())}"

        secret_env = [
            k8s.V1EnvVar(
                name=env_key,
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name="model-recommender-secrets",
                        key=secret_key,
                    )
                ),
            )
            for env_key, secret_key in [
                ("DATABASE_URL", "database-url"),
                ("CIVITAI_API_TOKEN", "civitai-api-token"),
            ]
        ]

        config_env = k8s.V1EnvFromSource(
            config_map_ref=k8s.V1ConfigMapEnvSource(name="model-recommender-config")
        )

        job = k8s.V1Job(
            metadata=k8s.V1ObjectMeta(name=job_name, namespace=namespace),
            spec=k8s.V1JobSpec(
                ttl_seconds_after_finished=3600,
                backoff_limit=3,
                template=k8s.V1PodTemplateSpec(
                    spec=k8s.V1PodSpec(
                        restart_policy="OnFailure",
                        containers=[
                            k8s.V1Container(
                                name="crawler",
                                image=image,
                                image_pull_policy="IfNotPresent",
                                command=[
                                    "python", "-m", "crawler.civitai_crawler",
                                    "--base-model", base_model,
                                    "--mode", mode,
                                ],
                                env=secret_env,
                                env_from=[config_env],
                                resources=k8s.V1ResourceRequirements(
                                    requests={"memory": "128Mi", "cpu": "100m"},
                                    limits={"memory": "512Mi", "cpu": "500m"},
                                ),
                            )
                        ],
                    )
                ),
            ),
        )

        k8s.BatchV1Api().create_namespaced_job(namespace=namespace, body=job)

        return {
            "status": "job_created",
            "job_name": job_name,
            "base_model": base_model,
            "mode": mode,
            "message": f"Crawler pod will start shortly. Track with: kubectl logs -n {namespace} -l job-name={job_name} -f",
        }

    except ImportError:
        return {
            "status": "kubernetes_unavailable",
            "message": (
                "kubernetes Python client not available. "
                "Run the crawler manually:\n"
                f"  python -m crawler.civitai_crawler --base-model '{base_model}' --mode {mode}"
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create crawl Job: {exc}") from exc
