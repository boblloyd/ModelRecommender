"""
Tests for crawler/tensorart_crawler.py.

All DB calls use the mock_pool / mock_conn fixtures from conftest.
No real network traffic or filesystem I/O in the test suite.
"""

import json
import pytest

from crawler.tensorart_crawler import (
    _extract_nuxt,
    _extract_record,
    _map_to_record,
    _normalize_base_model,
    _resolve,
    ingest_from_export_data,
    ingest_from_export,
)
from tests.conftest import (
    TA_EXPORT_LORA,
    TA_EXPORT_MISSING_ID,
    TA_EXPORT_MISSING_NUXT,
    TA_EXPORT_NO_NAME,
    TA_NUXT_LORA,
    TA_NUXT_NO_NAME,
)


# ---------------------------------------------------------------------------
# _resolve — Nuxt devalue integer-indirection
# ---------------------------------------------------------------------------

def test_resolve_non_integer_returned_as_is():
    assert _resolve([], "hello") == "hello"


def test_resolve_integer_within_bounds_dereferences():
    raw = ["a", "b", "target"]
    assert _resolve(raw, 2) == "target"


def test_resolve_out_of_bounds_returns_index():
    raw = ["only_one"]
    assert _resolve(raw, 99) == 99


def test_resolve_item_is_int_returns_int():
    raw = [42]
    assert _resolve(raw, 0) == 42


def test_resolve_item_is_list_recurses():
    raw = [[1, 2], "a", "b"]
    result = _resolve(raw, 0)
    assert result == ["a", "b"]


def test_resolve_item_is_dict_recurses():
    raw = [{"key": 1}, "value"]
    result = _resolve(raw, 0)
    assert result == {"key": "value"}


def test_resolve_depth_limit_prevents_stack_overflow():
    """A chain of nested dicts 15 deep should not raise RecursionError."""
    # Each entry is a dict whose sole value points to the next index
    raw = [{"k": i + 1} for i in range(15)] + ["leaf"]
    result = _resolve(raw, 0)
    # The chain stops at depth 12; top-level result is a dict (not None)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _normalize_base_model
# ---------------------------------------------------------------------------

def test_normalize_flux_dev_dotted():
    assert _normalize_base_model("FLUX.1-dev") == "Flux.1 D"


def test_normalize_flux_dev_lowercased():
    assert _normalize_base_model("flux.1-dev") == "Flux.1 D"


def test_normalize_flux_dev_spaced():
    assert _normalize_base_model("flux.1 dev") == "Flux.1 D"


def test_normalize_sdxl():
    assert _normalize_base_model("SDXL") == "SDXL 1.0"


def test_normalize_sd15():
    assert _normalize_base_model("SD1.5") == "SD 1.5"


def test_normalize_unknown_returns_none():
    assert _normalize_base_model("SomeUnknownModel3") is None


def test_normalize_empty_returns_none():
    assert _normalize_base_model("") is None


def test_normalize_none_returns_none():
    assert _normalize_base_model(None) is None


def test_normalize_strips_whitespace():
    assert _normalize_base_model("  flux.1-dev  ") == "Flux.1 D"


# ---------------------------------------------------------------------------
# _extract_nuxt
# ---------------------------------------------------------------------------

def test_extract_nuxt_gets_name():
    info = _extract_nuxt(TA_NUXT_LORA)
    assert info["name"] == "TensorArt Combat LoRA"


def test_extract_nuxt_gets_base_model():
    info = _extract_nuxt(TA_NUXT_LORA)
    assert info["base_model"] == "FLUX.1-dev"


def test_extract_nuxt_gets_trigger_words():
    info = _extract_nuxt(TA_NUXT_LORA)
    assert "ta_combat" in info["trained_words"]
    assert "action" in info["trained_words"]


def test_extract_nuxt_gets_tags():
    info = _extract_nuxt(TA_NUXT_LORA)
    assert "combat" in info["tags"]
    assert "action" in info["tags"]


def test_extract_nuxt_gets_stats():
    info = _extract_nuxt(TA_NUXT_LORA)
    assert info["download_count"] == 5000
    assert info["like_count"] == 150


def test_extract_nuxt_gets_cover_url():
    info = _extract_nuxt(TA_NUXT_LORA)
    assert "tensorartassets.com" in info["cover_url"]


def test_extract_nuxt_gets_description():
    info = _extract_nuxt(TA_NUXT_LORA)
    assert "combat" in info["description"].lower()


def test_extract_nuxt_empty_array_returns_defaults():
    info = _extract_nuxt([])
    assert info["name"] == ""
    assert info["trained_words"] == []
    assert info["tags"] == []


def test_extract_nuxt_no_cover_returns_empty_string():
    info = _extract_nuxt(TA_NUXT_NO_NAME)
    assert info["cover_url"] == ""


def test_extract_nuxt_trained_words_as_single_string():
    """trainedWords can be a plain string instead of a list."""
    raw = [{"trainedWords": "single_trigger", "baseModel": "FLUX.1-dev"}]
    info = _extract_nuxt(raw)
    assert info["trained_words"] == ["single_trigger"]


def test_extract_nuxt_standalone_user_generated_tags_with_int_refs():
    """Standalone tag objects where name/type are integer index references."""
    raw = [
        # index 0: standalone tag object (name=1, type=2)
        {"name": 1, "type": 2},
        "combat",
        "USER_GENERATED",
    ]
    info = _extract_nuxt(raw)
    assert "combat" in info["tags"]


def test_extract_nuxt_cover_url_as_direct_string():
    """cover field is a plain string, not a list."""
    raw = [{"cover": "https://tensorartassets.com/model_showcase/img.jpg"}]
    info = _extract_nuxt(raw)
    assert "tensorartassets.com" in info["cover_url"]


def test_extract_nuxt_cover_url_cdn_fallback():
    """If no cover object found, fall back to any CDN string in the raw array."""
    raw = [
        "some irrelevant string",
        "https://tensorartassets.com/model_showcase/fallback.jpg",
    ]
    info = _extract_nuxt(raw)
    assert "fallback.jpg" in info["cover_url"]


def test_extract_nuxt_cdn_fallback_skips_video():
    """Video files should not be used as cover URLs."""
    raw = ["https://tensorartassets.com/model_showcase/preview.mp4"]
    info = _extract_nuxt(raw)
    assert info["cover_url"] == ""


def test_extract_nuxt_strips_html_from_description():
    raw = [
        {
            "name": "My LoRA",
            "description": "<p>A <strong>great</strong> LoRA</p>",
            "relatedTags": [],
            "statisticInfo": {"downloadCount": 0, "likeCount": 0},
        }
    ]
    info = _extract_nuxt(raw)
    assert "<" not in info["description"]
    assert "great" in info["description"]


def test_extract_nuxt_uses_statisticsinfo_alternate_spelling():
    """statisticsInfo (plural) is an alternate key name for stats."""
    raw = [
        {
            "name": "LoRA",
            "relatedTags": [],
            "statisticsInfo": {"downloadCount": 999, "likeCount": 77},
        }
    ]
    info = _extract_nuxt(raw)
    assert info["download_count"] == 999
    assert info["like_count"] == 77


def test_extract_nuxt_cover_url_from_dict_in_list():
    """First item in cover list can be a dict with a 'url' key."""
    raw = [
        {
            "coverShowcases": [
                {"url": "https://tensorartassets.com/model_showcase/dict_cover.jpg"}
            ]
        }
    ]
    info = _extract_nuxt(raw)
    assert "dict_cover.jpg" in info["cover_url"]


# ---------------------------------------------------------------------------
# _map_to_record
# ---------------------------------------------------------------------------

def _sample_info(**overrides) -> dict:
    base = {
        "name": "Combat LoRA",
        "description": "A great LoRA",
        "base_model": "FLUX.1-dev",
        "trained_words": ["ta_combat"],
        "cover_url": "https://tensorartassets.com/preview.jpg",
        "nsfw_level": 0,
        "tags": ["combat", "action"],
        "download_count": 1000,
        "like_count": 50,
    }
    base.update(overrides)
    return base


def test_map_to_record_source_is_tensorart():
    r = _map_to_record("111", _sample_info())
    assert r["source"] == "tensorart"


def test_map_to_record_sets_tensorart_model_id():
    r = _map_to_record("111222", _sample_info())
    assert r["tensorart_model_id"] == "111222"


def test_map_to_record_normalizes_base_model():
    r = _map_to_record("111", _sample_info(base_model="FLUX.1-dev"))
    assert r["base_model"] == "Flux.1 D"


def test_map_to_record_keeps_unknown_base_model_raw():
    r = _map_to_record("111", _sample_info(base_model="SomeFutureModel"))
    assert r["base_model"] == "SomeFutureModel"


def test_map_to_record_none_base_model_when_empty():
    r = _map_to_record("111", _sample_info(base_model=""))
    assert r["base_model"] is None


def test_map_to_record_download_url_points_to_tensorart():
    r = _map_to_record("999888", _sample_info())
    assert r["download_url"] == "https://tensor.art/models/999888"


def test_map_to_record_civitai_fields_are_none():
    r = _map_to_record("111", _sample_info())
    assert r["civitai_model_id"] is None
    assert r["civitai_version_id"] is None
    assert r["civitai_url"] is None
    assert r["hf_repo_id"] is None


def test_map_to_record_type_is_lora():
    r = _map_to_record("111", _sample_info())
    assert r["type"] == "LORA"


def test_map_to_record_maps_stats():
    r = _map_to_record("111", _sample_info(download_count=5000, like_count=150))
    assert r["stats_downloads"] == 5000
    assert r["stats_thumbs_up"] == 150
    assert r["stats_thumbs_down"] == 0


def test_map_to_record_returns_none_when_no_name_or_words():
    r = _map_to_record("111", _sample_info(name="", trained_words=[]))
    assert r is None


def test_map_to_record_nsfw_defaults_to_1_when_zero():
    r = _map_to_record("111", _sample_info(nsfw_level=0))
    assert r["nsfw_level"] == 1


def test_map_to_record_preserves_nonzero_nsfw():
    r = _map_to_record("111", _sample_info(nsfw_level=4))
    assert r["nsfw_level"] == 4


# ---------------------------------------------------------------------------
# _extract_record
# ---------------------------------------------------------------------------

def test_extract_record_parses_full_entry():
    r = _extract_record(TA_EXPORT_LORA)
    assert r is not None
    assert r["tensorart_model_id"] == "111222333"
    assert r["name"] == "TensorArt Combat LoRA"
    assert r["base_model"] == "Flux.1 D"


def test_extract_record_returns_none_for_missing_nuxt():
    assert _extract_record(TA_EXPORT_MISSING_NUXT) is None


def test_extract_record_returns_none_for_missing_id():
    assert _extract_record(TA_EXPORT_MISSING_ID) is None


def test_extract_record_returns_none_when_no_name_or_words():
    assert _extract_record(TA_EXPORT_NO_NAME) is None


def test_extract_record_trigger_words_present():
    r = _extract_record(TA_EXPORT_LORA)
    assert r is not None
    assert "ta_combat" in r["trigger_words"]


def test_extract_record_tags_present():
    r = _extract_record(TA_EXPORT_LORA)
    assert r is not None
    assert "combat" in r["tags"]


# ---------------------------------------------------------------------------
# ingest_from_export_data
# ---------------------------------------------------------------------------

async def test_ingest_upserts_valid_records(mock_pool, mock_conn):
    count = await ingest_from_export_data([TA_EXPORT_LORA], mock_pool)
    assert count == 1
    mock_conn.executemany.assert_called_once()


async def test_ingest_skips_invalid_entries(mock_pool, mock_conn):
    data = [TA_EXPORT_MISSING_NUXT, TA_EXPORT_MISSING_ID, TA_EXPORT_NO_NAME]
    count = await ingest_from_export_data(data, mock_pool)
    assert count == 0
    mock_conn.executemany.assert_not_called()


async def test_ingest_mixed_batch(mock_pool, mock_conn):
    data = [TA_EXPORT_LORA, TA_EXPORT_MISSING_NUXT]
    count = await ingest_from_export_data(data, mock_pool)
    assert count == 1


async def test_ingest_empty_list(mock_pool, mock_conn):
    count = await ingest_from_export_data([], mock_pool)
    assert count == 0
    mock_conn.executemany.assert_not_called()


# ---------------------------------------------------------------------------
# ingest_from_export (file I/O)
# ---------------------------------------------------------------------------

async def test_ingest_from_export_reads_file(tmp_path, mock_pool, mock_conn):
    export = tmp_path / "export.json"
    export.write_text(json.dumps([TA_EXPORT_LORA]), encoding="utf-8")

    count = await ingest_from_export(str(export), mock_pool)
    assert count == 1


async def test_ingest_from_export_raises_on_non_array(tmp_path, mock_pool):
    export = tmp_path / "bad.json"
    export.write_text(json.dumps({"id": "123"}), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON array"):
        await ingest_from_export(str(export), mock_pool)
