"""
Tests for api/main.py FastAPI endpoints.

The asyncpg pool and all agent functions are mocked so no database
or network access occurs. ASGITransport does not trigger lifespan,
so app.state.pool and the _get_pool dependency override are set directly.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from api.main import _dispatch_job, _get_pool, app
from tests.conftest import make_db_model


@pytest_asyncio.fixture
async def client(mock_pool):
    """
    Test client with the real FastAPI app but a mocked pool.

    ASGITransport does not trigger FastAPI lifespan, so app.state.pool is
    never set by the startup hook. We set it directly and override the
    _get_pool dependency so every endpoint receives mock_pool.
    """
    app.state.pool = mock_pool
    app.dependency_overrides[_get_pool] = lambda: mock_pool
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /cache/status
# ---------------------------------------------------------------------------

async def test_cache_status_returns_base_model_list(client, mock_conn):
    mock_conn.fetch = AsyncMock(return_value=[
        {"base_model_name": "Flux.1 D", "last_crawled": None, "total_models": 23000, "crawl_complete": True}
    ])
    mock_conn.fetchval = AsyncMock(return_value=23000)

    response = await client.get("/cache/status")

    assert response.status_code == 200
    data = response.json()
    assert data["total_models_cached"] == 23000
    assert len(data["base_models"]) == 1
    assert data["base_models"][0]["base_model_name"] == "Flux.1 D"
    assert data["base_models"][0]["crawl_complete"] is True


async def test_cache_status_empty_catalog(client, mock_conn):
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchval = AsyncMock(return_value=0)

    response = await client.get("/cache/status")

    assert response.status_code == 200
    data = response.json()
    assert data["total_models_cached"] == 0
    assert data["base_models"] == []


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------

async def test_recommend_returns_409_when_base_model_not_cached(client):
    with patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=False)):
        response = await client.post(
            "/recommend",
            json={"prompt": "sword fight in rain", "base_model": "Flux.1 D"},
        )

    assert response.status_code == 409
    assert "Flux.1 D" in response.json()["detail"]


async def test_recommend_returns_results_when_cached(client):
    lora = make_db_model(relevance_score=0.72)
    checkpoint = make_db_model(type="Checkpoint", civitai_version_id=11111, relevance_score=0.85)

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.query_catalog", new=AsyncMock(
            return_value={"checkpoints": [checkpoint], "loras": [lora]}
        )),
    ):
        response = await client.post(
            "/recommend",
            json={"prompt": "sword fight in rain", "base_model": "Flux.1 D"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "checkpoints" in data
    assert "loras" in data
    assert len(data["checkpoints"]) == 1
    assert len(data["loras"]) == 1


async def test_recommend_passes_prompt_words_as_search_tags(client):
    captured = {}

    async def capture_catalog(search_tags, pool, **kwargs):
        captured["tags"] = search_tags
        return {"checkpoints": [], "loras": []}

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.query_catalog", side_effect=capture_catalog),
    ):
        await client.post(
            "/recommend",
            json={"prompt": "sword fight in the rain", "base_model": "Flux.1 D"},
        )

    assert "sword" in captured["tags"]
    assert "fight" in captured["tags"]
    assert "rain" in captured["tags"]
    assert "in" not in captured["tags"]
    assert "the" not in captured["tags"]


async def test_recommend_stop_words_filtered_regardless_of_length(client):
    """Stop words like 'with', 'from', 'they' are 4+ chars but must still be dropped."""
    captured = {}

    async def capture_catalog(search_tags, pool, **kwargs):
        captured["tags"] = search_tags
        return {"checkpoints": [], "loras": []}

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.query_catalog", side_effect=capture_catalog),
    ):
        await client.post(
            "/recommend",
            json={"prompt": "dark fantasy character with sword", "base_model": "Flux.1 D"},
        )

    assert "dark" in captured["tags"]
    assert "fantasy" in captured["tags"]
    assert "character" in captured["tags"]
    assert "sword" in captured["tags"]
    assert "with" not in captured["tags"]


async def test_recommend_nsfw_filter_propagated(client):
    captured = {}

    async def capture_catalog(search_tags, pool, **kwargs):
        captured["nsfw_max"] = kwargs.get("nsfw_max")
        return {"checkpoints": [], "loras": []}

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.query_catalog", side_effect=capture_catalog),
    ):
        await client.post(
            "/recommend",
            json={"prompt": "test", "base_model": "Flux.1 D", "nsfw_filter": True},
        )

    assert captured["nsfw_max"] == 1


async def test_recommend_response_includes_phase_note(client):
    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.query_catalog", new=AsyncMock(
            return_value={"checkpoints": [], "loras": []}
        )),
    ):
        response = await client.post(
            "/recommend",
            json={"prompt": "test", "base_model": "Flux.1 D"},
        )

    assert "phase" in response.json()


# ---------------------------------------------------------------------------
# POST /cache/crawl
# ---------------------------------------------------------------------------

async def test_cache_crawl_returns_job_created_when_kubernetes_available(client, mock_conn):
    with patch("api.main._dispatch_job", return_value={
        "status": "job_created",
        "job_name": "crawl-flux1d-1234567890",
        "base_model": "Flux.1 D",
        "mode": "full",
    }):
        response = await client.post(
            "/cache/crawl", json={"base_model": "Flux.1 D"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "job_created"
    assert "job_name" in data


async def test_cache_crawl_graceful_when_kubernetes_unavailable(client, mock_conn):
    with patch("api.main._dispatch_job", return_value={
        "status": "kubernetes_unavailable",
        "message": "Run the crawler manually: python -m crawler.civitai_crawler ...",
    }):
        response = await client.post(
            "/cache/crawl", json={"base_model": "Flux.1 D"}
        )

    assert response.status_code == 200
    assert response.json()["status"] == "kubernetes_unavailable"


async def test_cache_crawl_logs_cache_request(client, mock_conn):
    with patch("api.main._dispatch_job", return_value={"status": "job_created", "job_name": "x"}):
        await client.post("/cache/crawl", json={"base_model": "Flux.1 D"})

    # Should have inserted a cache_request record
    insert_sql = mock_conn.execute.call_args[0][0]
    assert "cache_requests" in insert_sql


# ---------------------------------------------------------------------------
# POST /cache/update
# ---------------------------------------------------------------------------

async def test_cache_update_forces_incremental_mode(client, mock_conn):
    captured = {}

    def capture_dispatch(base_model, mode, source="civitai"):
        captured["mode"] = mode
        return {"status": "job_created", "job_name": "x", "base_model": base_model, "mode": mode}

    with patch("api.main._dispatch_job", side_effect=capture_dispatch):
        await client.post("/cache/update", json={"base_model": "Flux.1 D", "mode": "full"})

    assert captured["mode"] == "incremental"


# ---------------------------------------------------------------------------
# _dispatch_job — Kubernetes job dispatch (direct unit tests)
# ---------------------------------------------------------------------------

def _make_k8s_modules():
    """
    Return a sys.modules patch dict with mocked kubernetes package so
    _dispatch_job's internal `from kubernetes import ...` calls get mocks.
    """
    class ConfigException(Exception):
        pass

    k8s_config = MagicMock()
    k8s_config.ConfigException = ConfigException

    k8s_client = MagicMock()

    kubernetes_mod = MagicMock()
    kubernetes_mod.client = k8s_client
    kubernetes_mod.config = k8s_config

    modules = {
        "kubernetes": kubernetes_mod,
        "kubernetes.client": k8s_client,
        "kubernetes.config": k8s_config,
    }
    return modules, k8s_client, k8s_config, ConfigException


def test_dispatch_job_creates_k8s_job_and_returns_job_created():
    modules, k8s_client, k8s_config, _ = _make_k8s_modules()

    with patch.dict(sys.modules, modules):
        result = _dispatch_job("Flux.1 D", "full")

    assert result["status"] == "job_created"
    assert result["base_model"] == "Flux.1 D"
    assert result["mode"] == "full"
    assert "job_name" in result
    k8s_client.BatchV1Api.return_value.create_namespaced_job.assert_called_once()


def test_dispatch_job_falls_back_to_kube_config_outside_cluster():
    modules, k8s_client, k8s_config, ConfigException = _make_k8s_modules()
    k8s_config.load_incluster_config.side_effect = ConfigException("not in cluster")

    with patch.dict(sys.modules, modules):
        result = _dispatch_job("Flux.1 D", "incremental")

    k8s_config.load_kube_config.assert_called_once()
    assert result["status"] == "job_created"


def test_dispatch_job_raises_500_when_k8s_api_call_fails():
    modules, k8s_client, k8s_config, _ = _make_k8s_modules()
    k8s_client.BatchV1Api.return_value.create_namespaced_job.side_effect = RuntimeError("k8s error")

    with patch.dict(sys.modules, modules):
        with pytest.raises(HTTPException) as exc_info:
            _dispatch_job("Flux.1 D", "full")

    assert exc_info.value.status_code == 500
    assert "Failed to create crawl Job" in exc_info.value.detail
