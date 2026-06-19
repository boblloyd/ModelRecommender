"""
Tests for agents/catalog_query.py.

No HTTP calls. The asyncpg pool is replaced with the mock_pool / mock_conn
fixtures from conftest so no real database is required.
"""

import pytest
from unittest.mock import AsyncMock

from agents.catalog_query import _score, ensure_base_model_cached, query_catalog
from tests.conftest import make_db_model


# ---------------------------------------------------------------------------
# _score — relevance scoring formula
# ---------------------------------------------------------------------------

def test_score_full_tag_overlap_gives_maximum_tag_component():
    model = make_db_model(
        tags=["sword", "combat"],
        trigger_words=[],
        stats_thumbs_up=0, stats_thumbs_down=0, stats_downloads=0,
    )
    # 2/2 overlap × 0.6 = 0.6; quality=0; downloads=0
    assert _score(model, ["sword", "combat"]) == pytest.approx(0.6, abs=0.001)


def test_score_partial_tag_overlap_scales_proportionally():
    model = make_db_model(
        tags=["sword", "combat"],
        trigger_words=[],
        stats_thumbs_up=0, stats_thumbs_down=0, stats_downloads=0,
    )
    # 1/2 search tags matched → overlap=0.5 → 0.5×0.6=0.3
    assert _score(model, ["sword", "fire"]) == pytest.approx(0.3, abs=0.001)


def test_score_trigger_words_count_toward_tag_overlap():
    model = make_db_model(
        tags=[],
        trigger_words=["sword", "combat"],
        stats_thumbs_up=0, stats_thumbs_down=0, stats_downloads=0,
    )
    # trigger_words union with tags for matching
    assert _score(model, ["sword", "combat"]) == pytest.approx(0.6, abs=0.001)


def test_score_tag_matching_is_case_insensitive():
    model = make_db_model(
        tags=["Sword", "COMBAT"],
        trigger_words=[],
        stats_thumbs_up=0, stats_thumbs_down=0, stats_downloads=0,
    )
    assert _score(model, ["sword", "combat"]) == pytest.approx(0.6, abs=0.001)


def test_score_no_tag_overlap_falls_back_to_quality_and_downloads():
    model = make_db_model(
        tags=["fire"],
        trigger_words=[],
        stats_thumbs_up=0, stats_thumbs_down=0, stats_downloads=0,
    )
    assert _score(model, ["water", "ocean"]) == pytest.approx(0.0, abs=0.001)


def test_score_empty_search_tags_gives_zero_overlap_component():
    model = make_db_model(
        tags=["sword"],
        trigger_words=[],
        stats_thumbs_up=100, stats_thumbs_down=0, stats_downloads=0,
    )
    score = _score(model, [])
    # tag_overlap=0, quality=1.0×0.25=0.25, downloads≈0
    assert score == pytest.approx(0.25, abs=0.01)


def test_score_quality_signal_differentiates_equal_tag_overlap():
    high_quality = make_db_model(
        tags=["sword"], trigger_words=[],
        stats_thumbs_up=100, stats_thumbs_down=0, stats_downloads=0,
    )
    low_quality = make_db_model(
        tags=["sword"], trigger_words=[],
        stats_thumbs_up=10, stats_thumbs_down=90, stats_downloads=0,
    )
    assert _score(high_quality, ["sword"]) > _score(low_quality, ["sword"])


def test_score_no_votes_gives_zero_quality():
    model = make_db_model(
        tags=[], trigger_words=[],
        stats_thumbs_up=0, stats_thumbs_down=0, stats_downloads=0,
    )
    assert _score(model, []) == 0.0


def test_score_download_count_contributes_to_final_score():
    many_downloads = make_db_model(
        tags=[], trigger_words=[],
        stats_thumbs_up=0, stats_thumbs_down=0, stats_downloads=1_000_000,
    )
    few_downloads = make_db_model(
        tags=[], trigger_words=[],
        stats_thumbs_up=0, stats_thumbs_down=0, stats_downloads=10,
    )
    assert _score(many_downloads, []) > _score(few_downloads, [])


# ---------------------------------------------------------------------------
# query_catalog
# ---------------------------------------------------------------------------

async def test_query_catalog_splits_results_by_type(mock_pool, mock_conn):
    lora = make_db_model(type="LORA")
    checkpoint = make_db_model(type="Checkpoint", civitai_version_id=11111, name="Checkpoint A")
    mock_conn.fetch = AsyncMock(return_value=[lora, checkpoint])

    result = await query_catalog(["sword"], mock_pool)

    assert len(result["checkpoints"]) == 1
    assert len(result["loras"]) == 1
    assert result["checkpoints"][0]["type"] == "Checkpoint"
    assert result["loras"][0]["type"] == "LORA"


async def test_query_catalog_locon_grouped_with_loras(mock_pool, mock_conn):
    locon = make_db_model(type="LoCon")
    mock_conn.fetch = AsyncMock(return_value=[locon])

    result = await query_catalog(["sword"], mock_pool)

    assert len(result["loras"]) == 1
    assert result["loras"][0]["type"] == "LoCon"
    assert len(result["checkpoints"]) == 0


async def test_query_catalog_results_ordered_by_score_descending(mock_pool, mock_conn):
    high = make_db_model(tags=["sword", "combat", "action"], civitai_version_id=1)
    low = make_db_model(tags=["forest", "nature"], civitai_version_id=2)
    mock_conn.fetch = AsyncMock(return_value=[low, high])  # low first in DB order

    result = await query_catalog(["sword", "combat", "action"], mock_pool)

    scores = [m["relevance_score"] for m in result["loras"]]
    assert scores == sorted(scores, reverse=True)
    assert result["loras"][0]["civitai_version_id"] == 1  # high-score model is first


async def test_query_catalog_respects_limit_loras(mock_pool, mock_conn):
    models = [make_db_model(civitai_version_id=i, name=f"LoRA {i}") for i in range(20)]
    mock_conn.fetch = AsyncMock(return_value=models)

    result = await query_catalog(["sword"], mock_pool, limit_loras=5)

    assert len(result["loras"]) == 5


async def test_query_catalog_nsfw_filter_adds_condition_to_sql(mock_pool, mock_conn):
    mock_conn.fetch = AsyncMock(return_value=[])

    await query_catalog(["sword"], mock_pool, nsfw_max=1)

    sql = mock_conn.fetch.call_args[0][0]
    assert "nsfw_level" in sql


async def test_query_catalog_each_result_has_relevance_score(mock_pool, mock_conn):
    mock_conn.fetch = AsyncMock(return_value=[make_db_model()])

    result = await query_catalog(["sword"], mock_pool)

    assert "relevance_score" in result["loras"][0]
    assert isinstance(result["loras"][0]["relevance_score"], float)


async def test_query_catalog_empty_cache_returns_empty_lists(mock_pool, mock_conn):
    mock_conn.fetch = AsyncMock(return_value=[])

    result = await query_catalog(["sword"], mock_pool)

    assert result["checkpoints"] == []
    assert result["loras"] == []


async def test_query_catalog_passes_base_model_to_sql(mock_pool, mock_conn):
    mock_conn.fetch = AsyncMock(return_value=[])

    await query_catalog(["sword"], mock_pool, base_model="SDXL 1.0")

    args = mock_conn.fetch.call_args[0]
    assert "SDXL 1.0" in args


# ---------------------------------------------------------------------------
# ensure_base_model_cached
# ---------------------------------------------------------------------------

async def test_ensure_cached_true_when_crawl_complete(mock_pool, mock_conn):
    mock_conn.fetchrow = AsyncMock(return_value={"crawl_complete": True})
    assert await ensure_base_model_cached("Flux.1 D", mock_pool) is True


async def test_ensure_cached_false_when_crawl_incomplete(mock_pool, mock_conn):
    mock_conn.fetchrow = AsyncMock(return_value={"crawl_complete": False})
    assert await ensure_base_model_cached("Flux.1 D", mock_pool) is False


async def test_ensure_cached_false_when_base_model_not_in_index(mock_pool, mock_conn):
    mock_conn.fetchrow = AsyncMock(return_value=None)
    assert await ensure_base_model_cached("SDXL 1.0", mock_pool) is False
