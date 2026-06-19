"""
Tests for crawler/civitai_crawler.py.

All HTTP calls are intercepted by respx so no real network traffic is made.
The asyncpg pool is replaced with the mock_pool / mock_conn fixtures from conftest.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from crawler.civitai_crawler import (
    _extract_record,
    _strip_html,
    full_crawl,
    incremental_update,
)
from tests.conftest import (
    CIVITAI_CHECKPOINT,
    CIVITAI_LORA,
    CIVITAI_NO_VERSIONS,
    api_page,
    make_db_model,
)

CIVITAI_MODELS_URL = "https://civitai.com/api/v1/models"
PAGE_2_URL = "https://civitai.com/api/v1/models?page=2&limit=100"


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

def test_strip_html_removes_tags():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_decodes_html_entities():
    assert _strip_html("Tom &amp; Jerry") == "Tom & Jerry"
    assert _strip_html("5 &lt; 10") == "5 < 10"


def test_strip_html_collapses_whitespace():
    assert _strip_html("<p>  too   many   spaces  </p>") == "too many spaces"


def test_strip_html_none_returns_none():
    assert _strip_html(None) is None


def test_strip_html_empty_string_returns_none():
    assert _strip_html("") is None


def test_strip_html_only_tags_returns_none():
    assert _strip_html("<br/><br/><p></p>") is None


# ---------------------------------------------------------------------------
# _extract_record — field mapping from Civitai API item → DB record
# ---------------------------------------------------------------------------

def test_extract_record_maps_all_core_fields():
    r = _extract_record(CIVITAI_LORA)
    assert r is not None
    assert r["source"] == "civitai"
    assert r["civitai_model_id"] == 12345
    assert r["civitai_version_id"] == 67890
    assert r["name"] == "Combat Action LoRA"
    assert r["version_name"] == "v1.0"
    assert r["type"] == "LORA"
    assert r["base_model"] == "Flux.1 D"
    assert r["nsfw_level"] == 1
    assert r["tags"] == ["sword", "combat", "action", "fight"]
    assert r["trigger_words"] == ["actn_combat", "sword_fight"]
    assert r["download_url"] == "https://civitai.com/api/download/models/67890"
    assert r["civitai_url"] == "https://civitai.com/models/12345?modelVersionId=67890"
    assert r["stats_downloads"] == 50000
    assert r["stats_thumbs_up"] == 850
    assert r["stats_thumbs_down"] == 12
    assert r["preview_image_url"] == "https://image.civitai.com/lora_preview.jpg"


def test_extract_record_pulls_generation_params_from_first_image_meta():
    r = _extract_record(CIVITAI_LORA)
    assert r is not None
    assert r["recommended_cfg"] == 3.5
    assert r["recommended_steps"] == 28
    assert r["recommended_sampler"] == "DPM++ 2M"


def test_extract_record_recommended_weight_is_none():
    # Weight is not available from the Civitai API — set by LLM in Phase 2
    r = _extract_record(CIVITAI_LORA)
    assert r is not None
    assert r["recommended_weight"] is None


def test_extract_record_strips_html_from_description():
    r = _extract_record(CIVITAI_LORA)
    assert r is not None
    assert "<" not in r["description"]
    assert "combat" in r["description"]
    assert "sword fighting" in r["description"]


def test_extract_record_no_versions_returns_none():
    assert _extract_record(CIVITAI_NO_VERSIONS) is None


def test_extract_record_empty_versions_list_returns_none():
    assert _extract_record({**CIVITAI_LORA, "modelVersions": []}) is None


def test_extract_record_version_missing_id_returns_none():
    item = {**CIVITAI_LORA, "modelVersions": [{"name": "v1.0", "baseModel": "Flux.1 D"}]}
    assert _extract_record(item) is None


def test_extract_record_no_images_gives_null_preview_and_params():
    version = {**CIVITAI_LORA["modelVersions"][0], "images": []}
    item = {**CIVITAI_LORA, "modelVersions": [version]}
    r = _extract_record(item)
    assert r is not None
    assert r["preview_image_url"] is None
    assert r["recommended_cfg"] is None
    assert r["recommended_steps"] is None
    assert r["recommended_sampler"] is None


def test_extract_record_missing_stats_defaults_to_zero():
    r = _extract_record({**CIVITAI_LORA, "stats": {}})
    assert r is not None
    assert r["stats_downloads"] == 0
    assert r["stats_thumbs_up"] == 0
    assert r["stats_thumbs_down"] == 0


def test_extract_record_empty_trained_words():
    version = {**CIVITAI_LORA["modelVersions"][0], "trainedWords": []}
    r = _extract_record({**CIVITAI_LORA, "modelVersions": [version]})
    assert r is not None
    assert r["trigger_words"] == []


def test_extract_record_checkpoint_type():
    r = _extract_record(CIVITAI_CHECKPOINT)
    assert r is not None
    assert r["type"] == "Checkpoint"
    assert r["civitai_model_id"] == 99999
    assert r["civitai_version_id"] == 11111


# ---------------------------------------------------------------------------
# full_crawl
# ---------------------------------------------------------------------------

async def test_full_crawl_single_page_upserts_all_records(respx_mock, mock_pool, mock_conn):
    respx_mock.get(CIVITAI_MODELS_URL).mock(
        return_value=httpx.Response(200, json=api_page([CIVITAI_LORA, CIVITAI_CHECKPOINT]))
    )

    count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 2
    mock_conn.executemany.assert_called_once()


async def test_full_crawl_follows_next_page_url(respx_mock, mock_pool, mock_conn):
    # Two separate routes won't work: CIVITAI_MODELS_URL (no query params) also
    # matches PAGE_2_URL requests, causing an infinite loop. Use side_effect list
    # so the single route returns responses in call order regardless of URL.
    respx_mock.get(CIVITAI_MODELS_URL).mock(side_effect=[
        httpx.Response(200, json=api_page([CIVITAI_LORA], next_url=PAGE_2_URL)),
        httpx.Response(200, json=api_page([CIVITAI_CHECKPOINT])),
    ])

    count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 2
    assert mock_conn.executemany.call_count == 2


async def test_full_crawl_skips_items_with_no_versions(respx_mock, mock_pool, mock_conn):
    respx_mock.get(CIVITAI_MODELS_URL).mock(
        return_value=httpx.Response(200, json=api_page([CIVITAI_NO_VERSIONS]))
    )

    count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 0
    mock_conn.executemany.assert_not_called()


async def test_full_crawl_marks_base_model_index_complete(respx_mock, mock_pool, mock_conn):
    respx_mock.get(CIVITAI_MODELS_URL).mock(
        return_value=httpx.Response(200, json=api_page([CIVITAI_LORA]))
    )

    await full_crawl("Flux.1 D", mock_pool)

    # The final execute call should set crawl_complete = TRUE
    final_sql = mock_conn.execute.call_args_list[-1][0][0]
    assert "crawl_complete" in final_sql
    assert "TRUE" in final_sql


async def test_full_crawl_retries_on_429_rate_limit(respx_mock, mock_pool, mock_conn):
    respx_mock.get(CIVITAI_MODELS_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=api_page([CIVITAI_LORA])),
        ]
    )

    with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
        count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 1
    mock_sleep.assert_called_once()  # exactly one backoff sleep


# ---------------------------------------------------------------------------
# incremental_update
# ---------------------------------------------------------------------------

async def test_incremental_update_falls_back_when_no_prior_crawl(respx_mock, mock_pool, mock_conn):
    mock_conn.fetchrow = AsyncMock(return_value=None)

    with patch("crawler.civitai_crawler.full_crawl", new=AsyncMock(return_value=42)) as mock_full:
        count = await incremental_update("Flux.1 D", mock_pool)

    mock_full.assert_called_once_with("Flux.1 D", mock_pool)
    assert count == 42


async def test_incremental_update_stops_when_known_version_id_found(respx_mock, mock_pool, mock_conn):
    mock_conn.fetchrow = AsyncMock(
        return_value={"last_crawled": datetime.now(timezone.utc)}
    )
    # The LORA's version_id is already in the cache
    mock_conn.fetch = AsyncMock(
        return_value=[{"civitai_version_id": 67890}]
    )
    respx_mock.get(CIVITAI_MODELS_URL).mock(
        return_value=httpx.Response(200, json=api_page([CIVITAI_LORA]))
    )

    count = await incremental_update("Flux.1 D", mock_pool)

    assert count == 0
    mock_conn.executemany.assert_not_called()


async def test_incremental_update_upserts_genuinely_new_models(respx_mock, mock_pool, mock_conn):
    mock_conn.fetchrow = AsyncMock(
        return_value={"last_crawled": datetime.now(timezone.utc)}
    )
    # No existing IDs in cache
    mock_conn.fetch = AsyncMock(return_value=[])
    respx_mock.get(CIVITAI_MODELS_URL).mock(
        return_value=httpx.Response(200, json=api_page([CIVITAI_LORA]))
    )

    count = await incremental_update("Flux.1 D", mock_pool)

    assert count == 1
    mock_conn.executemany.assert_called_once()
