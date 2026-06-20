"""
Tests for crawler/hf_crawler.py.

All HTTP calls are intercepted by respx so no real network traffic is made.
The asyncpg pool is replaced with the mock_pool / mock_conn fixtures from conftest.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from crawler.hf_crawler import (
    _base_model_matches,
    _extract_record,
    _infer_type,
    _parse_next_url,
    full_crawl,
    incremental_update,
)
from tests.conftest import (
    HF_CHECKPOINT,
    HF_LORA,
    HF_NO_TYPE,
    HF_WRONG_BASE_MODEL,
)

HF_MODELS_URL = "https://huggingface.co/api/models"
PAGE_2_URL = "https://huggingface.co/api/models?cursor=page2cursor"


# ---------------------------------------------------------------------------
# _parse_next_url
# ---------------------------------------------------------------------------

def test_parse_next_url_extracts_url_from_link_header():
    header = f'<{PAGE_2_URL}>; rel="next"'
    assert _parse_next_url(header) == PAGE_2_URL


def test_parse_next_url_returns_none_for_missing_header():
    assert _parse_next_url(None) is None


def test_parse_next_url_returns_none_when_no_next_rel():
    header = '<https://huggingface.co/api/models?cursor=prev>; rel="prev"'
    assert _parse_next_url(header) is None


def test_parse_next_url_handles_multiple_links():
    header = (
        '<https://huggingface.co/api/models?cursor=prev>; rel="prev", '
        f'<{PAGE_2_URL}>; rel="next"'
    )
    assert _parse_next_url(header) == PAGE_2_URL


# ---------------------------------------------------------------------------
# _infer_type
# ---------------------------------------------------------------------------

def test_infer_type_lora_from_lora_tag():
    assert _infer_type(["lora", "flux-dev", "diffusers"]) == "LORA"


def test_infer_type_checkpoint_from_diffusers_tag():
    assert _infer_type(["diffusers", "flux-dev", "text-to-image"]) == "Checkpoint"


def test_infer_type_lora_takes_priority_over_diffusers():
    assert _infer_type(["lora", "diffusers"]) == "LORA"


def test_infer_type_returns_none_for_unknown_tags():
    assert _infer_type(["transformers", "text-encoders"]) is None


def test_infer_type_case_insensitive():
    assert _infer_type(["LoRA", "flux-dev"]) == "LORA"


# ---------------------------------------------------------------------------
# _base_model_matches
# ---------------------------------------------------------------------------

def test_base_model_matches_when_card_data_contains_expected_id():
    assert _base_model_matches(HF_LORA, "Flux.1 D") is True


def test_base_model_does_not_match_wrong_base_model():
    assert _base_model_matches(HF_WRONG_BASE_MODEL, "Flux.1 D") is False


def test_base_model_matches_when_card_data_is_a_list():
    item = {
        **HF_LORA,
        "cardData": {"base_model": ["black-forest-labs/FLUX.1-dev", "some-other/model"]},
    }
    assert _base_model_matches(item, "Flux.1 D") is True


def test_base_model_matches_returns_true_for_unknown_base_model():
    assert _base_model_matches(HF_LORA, "Unknown Model") is True


def test_base_model_matches_missing_card_data_returns_false():
    item = {**HF_LORA, "cardData": None}
    assert _base_model_matches(item, "Flux.1 D") is False


# ---------------------------------------------------------------------------
# _extract_record
# ---------------------------------------------------------------------------

def test_extract_record_maps_lora_fields():
    r = _extract_record(HF_LORA, "Flux.1 D")
    assert r is not None
    assert r["source"] == "huggingface"
    assert r["hf_repo_id"] == "XLabs-AI/flux-RealismLora"
    assert r["civitai_model_id"] is None
    assert r["civitai_version_id"] is None
    assert r["type"] == "LORA"
    assert r["base_model"] == "Flux.1 D"
    assert r["stats_downloads"] == 85000
    assert r["stats_thumbs_up"] == 2300
    assert r["stats_thumbs_down"] == 0
    assert r["trigger_words"] == []
    assert r["recommended_weight"] is None


def test_extract_record_maps_checkpoint_fields():
    r = _extract_record(HF_CHECKPOINT, "Flux.1 D")
    assert r is not None
    assert r["type"] == "Checkpoint"
    assert r["hf_repo_id"] == "Freepik/flux.1-lite-8B-alpha"


def test_extract_record_derives_name_from_repo_id():
    r = _extract_record(HF_LORA, "Flux.1 D")
    assert r is not None
    assert r["name"] == "flux RealismLora"


def test_extract_record_strips_arxiv_tags():
    item = {**HF_LORA, "tags": ["lora", "arxiv:2401.00001", "flux-dev"]}
    r = _extract_record(item, "Flux.1 D")
    assert r is not None
    assert not any(t.startswith("arxiv:") for t in r["tags"])
    assert "lora" in r["tags"]


def test_extract_record_returns_none_for_missing_repo_id():
    item = {k: v for k, v in HF_LORA.items() if k not in ("id", "modelId")}
    assert _extract_record(item, "Flux.1 D") is None


def test_extract_record_returns_none_for_unknown_type():
    assert _extract_record(HF_NO_TYPE, "Flux.1 D") is None


def test_extract_record_returns_none_when_base_model_does_not_match():
    assert _extract_record(HF_WRONG_BASE_MODEL, "Flux.1 D") is None


def test_extract_record_sets_download_url_to_hf_repo_page():
    r = _extract_record(HF_LORA, "Flux.1 D")
    assert r is not None
    assert r["download_url"] == "https://huggingface.co/XLabs-AI/flux-RealismLora"


def test_extract_record_civitai_url_is_none():
    r = _extract_record(HF_LORA, "Flux.1 D")
    assert r is not None
    assert r["civitai_url"] is None


# ---------------------------------------------------------------------------
# full_crawl
# ---------------------------------------------------------------------------

async def test_full_crawl_single_page_upserts_all_records(respx_mock, mock_pool, mock_conn):
    respx_mock.get(HF_MODELS_URL).mock(
        return_value=httpx.Response(200, json=[HF_LORA, HF_CHECKPOINT])
    )

    count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 2
    mock_conn.executemany.assert_called_once()


async def test_full_crawl_follows_link_header_pagination(respx_mock, mock_pool, mock_conn):
    respx_mock.get(HF_MODELS_URL).mock(side_effect=[
        httpx.Response(200, json=[HF_LORA], headers={"Link": f'<{PAGE_2_URL}>; rel="next"'}),
        httpx.Response(200, json=[HF_CHECKPOINT]),
    ])

    count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 2
    assert mock_conn.executemany.call_count == 2


async def test_full_crawl_skips_wrong_base_model(respx_mock, mock_pool, mock_conn):
    respx_mock.get(HF_MODELS_URL).mock(
        return_value=httpx.Response(200, json=[HF_WRONG_BASE_MODEL])
    )

    count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 0
    mock_conn.executemany.assert_not_called()


async def test_full_crawl_skips_unknown_type(respx_mock, mock_pool, mock_conn):
    respx_mock.get(HF_MODELS_URL).mock(
        return_value=httpx.Response(200, json=[HF_NO_TYPE])
    )

    count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 0
    mock_conn.executemany.assert_not_called()


async def test_full_crawl_marks_base_model_index_complete(respx_mock, mock_pool, mock_conn):
    respx_mock.get(HF_MODELS_URL).mock(
        return_value=httpx.Response(200, json=[HF_LORA])
    )

    await full_crawl("Flux.1 D", mock_pool)

    final_sql = mock_conn.execute.call_args_list[-1][0][0]
    assert "crawl_complete" in final_sql
    assert "TRUE" in final_sql


async def test_full_crawl_retries_on_429(respx_mock, mock_pool, mock_conn):
    respx_mock.get(HF_MODELS_URL).mock(side_effect=[
        httpx.Response(429),
        httpx.Response(200, json=[HF_LORA]),
    ])

    with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
        count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 1
    mock_sleep.assert_called_once()


async def test_full_crawl_retries_on_request_error(respx_mock, mock_pool, mock_conn):
    respx_mock.get(HF_MODELS_URL).mock(side_effect=[
        httpx.ConnectError("connection refused"),
        httpx.Response(200, json=[HF_LORA]),
    ])

    with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
        count = await full_crawl("Flux.1 D", mock_pool)

    assert count == 1
    mock_sleep.assert_called_once()


async def test_full_crawl_sends_auth_header_when_token_set(respx_mock, mock_pool, mock_conn):
    respx_mock.get(HF_MODELS_URL).mock(
        return_value=httpx.Response(200, json=[])
    )

    with patch.dict("os.environ", {"HF_API_TOKEN": "hf-test-token"}):
        await full_crawl("Flux.1 D", mock_pool)

    assert respx_mock.calls[0].request.headers["Authorization"] == "Bearer hf-test-token"


async def test_full_crawl_unknown_base_model_uses_no_filter(respx_mock, mock_pool, mock_conn):
    """A base model not in BASE_MODEL_HF_TAGS crawls without a filter tag."""
    respx_mock.get(HF_MODELS_URL).mock(
        return_value=httpx.Response(200, json=[])
    )

    await full_crawl("Unknown Model XL", mock_pool)

    request = respx_mock.calls[0].request
    assert "filter" not in str(request.url)


# ---------------------------------------------------------------------------
# incremental_update
# ---------------------------------------------------------------------------

async def test_incremental_update_delegates_to_full_crawl(mock_pool, mock_conn):
    with patch("crawler.hf_crawler.full_crawl", new=AsyncMock(return_value=17)) as mock_full:
        count = await incremental_update("Flux.1 D", mock_pool)

    mock_full.assert_called_once_with("Flux.1 D", mock_pool)
    assert count == 17
