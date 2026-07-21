"""
Tests for api/main.py FastAPI endpoints.

The asyncpg pool and all agent functions are mocked so no database
or network access occurs. ASGITransport does not trigger lifespan,
so app.state.pool and the _get_pool dependency override are set directly.
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from agents.intent_parser import IntentResult
from api.main import _dispatch_job, _get_pool, app
from tests.conftest import make_db_model, TA_EXPORT_LORA


def _intent(tags=None, style="cinematic", subject="sword fight", llm_used=False) -> IntentResult:
    return IntentResult(
        tags=tags or ["sword", "fight", "rain"],
        style=style,
        subject=subject,
        llm_used=llm_used,
    )


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
        patch("api.main.parse_intent", new=AsyncMock(return_value=_intent())),
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


async def test_recommend_passes_intent_tags_to_catalog(client):
    """Tags from parse_intent are forwarded to query_catalog unchanged."""
    captured = {}

    async def capture_catalog(search_tags, pool, **kwargs):
        captured["tags"] = search_tags
        return {"checkpoints": [], "loras": []}

    intent = _intent(tags=["sword", "fight", "rain"])

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=intent)),
        patch("api.main.query_catalog", side_effect=capture_catalog),
    ):
        await client.post(
            "/recommend",
            json={"prompt": "sword fight in the rain", "base_model": "Flux.1 D"},
        )

    assert captured["tags"] == ["sword", "fight", "rain"]


async def test_recommend_calls_parse_intent_with_user_prompt(client):
    captured_prompt = {}

    async def capture_intent(prompt):
        captured_prompt["value"] = prompt
        return _intent()

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", side_effect=capture_intent),
        patch("api.main.query_catalog", new=AsyncMock(return_value={"checkpoints": [], "loras": []})),
    ):
        await client.post(
            "/recommend",
            json={"prompt": "epic fantasy portrait", "base_model": "Flux.1 D"},
        )

    assert captured_prompt["value"] == "epic fantasy portrait"


async def test_recommend_nsfw_filter_propagated(client):
    captured = {}

    async def capture_catalog(search_tags, pool, **kwargs):
        captured["nsfw_max"] = kwargs.get("nsfw_max")
        return {"checkpoints": [], "loras": []}

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=_intent())),
        patch("api.main.query_catalog", side_effect=capture_catalog),
    ):
        await client.post(
            "/recommend",
            json={"prompt": "test", "base_model": "Flux.1 D", "nsfw_filter": True},
        )

    assert captured["nsfw_max"] == 1


async def test_recommend_response_includes_intent_block(client):
    intent = _intent(tags=["portrait", "anime"], style="anime", subject="warrior", llm_used=True)

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=intent)),
        patch("api.main.query_catalog", new=AsyncMock(return_value={"checkpoints": [], "loras": []})),
        patch("api.main.analyze_compatibility", new=AsyncMock(
            return_value={"checkpoints": [], "loras": []}
        )),
    ):
        response = await client.post(
            "/recommend",
            json={"prompt": "anime warrior", "base_model": "Flux.1 D"},
        )

    data = response.json()
    assert "intent" in data
    assert data["intent"]["tags"] == ["portrait", "anime"]
    assert data["intent"]["style"] == "anime"
    assert data["intent"]["subject"] == "warrior"
    assert data["intent"]["llm_used"] is True


async def test_recommend_phase_is_1_when_llm_falls_back(client):
    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=_intent(llm_used=False))),
        patch("api.main.query_catalog", new=AsyncMock(return_value={"checkpoints": [], "loras": []})),
    ):
        response = await client.post(
            "/recommend",
            json={"prompt": "test", "base_model": "Flux.1 D"},
        )

    assert "1" in response.json()["phase"]


async def test_recommend_phase_is_2a_when_llm_intent_parsed(client):
    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=_intent(llm_used=True))),
        patch("api.main.query_catalog", new=AsyncMock(return_value={"checkpoints": [], "loras": []})),
        patch("api.main.analyze_compatibility", new=AsyncMock(
            return_value={"checkpoints": [], "loras": []}  # no recommended_combination
        )),
    ):
        response = await client.post(
            "/recommend",
            json={"prompt": "test", "base_model": "Flux.1 D"},
        )

    assert "2a" in response.json()["phase"]


async def test_recommend_phase_is_2b_when_full_llm_pipeline_runs(client):
    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=_intent(llm_used=True))),
        patch("api.main.query_catalog", new=AsyncMock(return_value={"checkpoints": [], "loras": []})),
        patch("api.main.analyze_compatibility", new=AsyncMock(return_value={
            "checkpoints": [],
            "loras": [],
            "recommended_combination": "Checkpoint A + LoRA B",
            "combination_notes": "Works well together.",
        })),
    ):
        response = await client.post(
            "/recommend",
            json={"prompt": "test", "base_model": "Flux.1 D"},
        )

    data = response.json()
    assert "2b" in data["phase"]
    assert data["recommended_combination"] == "Checkpoint A + LoRA B"


async def test_recommend_calls_analyze_compatibility_when_llm_used(client):
    mock_analyze = AsyncMock(return_value={"checkpoints": [], "loras": []})

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=_intent(llm_used=True))),
        patch("api.main.query_catalog", new=AsyncMock(return_value={"checkpoints": [], "loras": []})),
        patch("api.main.analyze_compatibility", mock_analyze),
    ):
        await client.post(
            "/recommend",
            json={"prompt": "test", "base_model": "Flux.1 D", "llm_reasoning": True},
        )

    mock_analyze.assert_called_once()


async def test_recommend_skips_analyze_compatibility_when_llm_reasoning_false(client):
    mock_analyze = AsyncMock(return_value={"checkpoints": [], "loras": []})

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=_intent(llm_used=True))),
        patch("api.main.query_catalog", new=AsyncMock(return_value={"checkpoints": [], "loras": []})),
        patch("api.main.analyze_compatibility", mock_analyze),
    ):
        await client.post(
            "/recommend",
            json={"prompt": "test", "base_model": "Flux.1 D", "llm_reasoning": False},
        )

    mock_analyze.assert_not_called()


async def test_recommend_skips_analyze_compatibility_when_llm_not_used(client):
    """analyze_compatibility should not run if parse_intent fell back to stop-words."""
    mock_analyze = AsyncMock(return_value={"checkpoints": [], "loras": []})

    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=_intent(llm_used=False))),
        patch("api.main.query_catalog", new=AsyncMock(return_value={"checkpoints": [], "loras": []})),
        patch("api.main.analyze_compatibility", mock_analyze),
    ):
        await client.post(
            "/recommend",
            json={"prompt": "test", "base_model": "Flux.1 D"},
        )

    mock_analyze.assert_not_called()


async def test_recommend_response_includes_phase_field(client):
    with (
        patch("api.main.ensure_base_model_cached", new=AsyncMock(return_value=True)),
        patch("api.main.parse_intent", new=AsyncMock(return_value=_intent())),
        patch("api.main.query_catalog", new=AsyncMock(return_value={"checkpoints": [], "loras": []})),
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


# ---------------------------------------------------------------------------
# POST /catalog/import/tensorart
# ---------------------------------------------------------------------------

async def test_import_tensorart_returns_count(client):
    payload = json.dumps([TA_EXPORT_LORA]).encode()

    with patch(
        "crawler.tensorart_crawler.ingest_from_export_data",
        new=AsyncMock(return_value=1),
    ):
        response = await client.post(
            "/catalog/import/tensorart",
            files={"file": ("export.json", payload, "application/json")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_imported"] == 1
    assert data["filename"] == "export.json"


async def test_import_tensorart_returns_400_for_invalid_json(client):
    response = await client.post(
        "/catalog/import/tensorart",
        files={"file": ("bad.json", b"not json at all", "application/json")},
    )

    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]


async def test_import_tensorart_returns_400_when_data_is_not_array(client):
    payload = json.dumps({"id": "123"}).encode()

    response = await client.post(
        "/catalog/import/tensorart",
        files={"file": ("bad.json", payload, "application/json")},
    )

    assert response.status_code == 400
    assert "JSON array" in response.json()["detail"]


async def test_import_tensorart_zero_when_all_entries_invalid(client):
    bad_data = [{"id": ""}, {"nuxt": None}]
    payload = json.dumps(bad_data).encode()

    with patch(
        "crawler.tensorart_crawler.ingest_from_export_data",
        new=AsyncMock(return_value=0),
    ):
        response = await client.post(
            "/catalog/import/tensorart",
            files={"file": ("empty.json", payload, "application/json")},
        )

    assert response.status_code == 200
    assert response.json()["models_imported"] == 0
