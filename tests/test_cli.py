"""
Tests for the CLI's LLM-integrated query path (cli._run_query).

All external calls — get_pool, parse_intent, query_catalog, analyze_compatibility —
are mocked.  No database, no Ollama, no network.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_db_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool():
    pool = MagicMock()
    pool.close = AsyncMock()
    return pool


def _make_intent(llm_used=True, tags=None, style="cinematic", subject="portrait"):
    return SimpleNamespace(
        tags=tags or ["portrait", "cinematic", "dramatic"],
        style=style,
        subject=subject,
        llm_used=llm_used,
    )


def _catalog_results():
    lora = {
        **make_db_model(id=2, type="LORA"),
        "relevance_score": 0.731,
        "compatibility_note": None,
        "recommended": True,
        "recommended_weight": None,
        "impact": None,
    }
    return {"checkpoints": [], "loras": [lora]}


def _analyzed_results():
    lora = {
        **make_db_model(id=2, type="LORA", trigger_words=["actn_combat", "sword_fight"]),
        "relevance_score": 0.731,
        "compatibility_note": "Trained specifically on medieval combat with realistic sword physics.",
        "recommended": True,
        "recommended_weight": 0.8,
        "impact": "high",
    }
    return {
        "checkpoints": [],
        "loras": [lora],
        "recommended_combination": "Flux Realism Pro + Combat Action LoRA",
        "combination_notes": "Use 'actn_combat' at weight 0.8.",
        "prompt_additions": ["actn_combat", "sword_fight", "dramatic lighting"],
    }


def _patches(
    cached=True,
    intent=None,
    catalog=None,
    analyzed=None,
):
    """Return a dict of patch kwargs, allowing per-test overrides."""
    return {
        "agents.catalog_query.ensure_base_model_cached": AsyncMock(return_value=cached),
        "agents.intent_parser.parse_intent": AsyncMock(return_value=intent or _make_intent()),
        "agents.catalog_query.query_catalog": AsyncMock(return_value=catalog or _catalog_results()),
        "agents.compatibility_analyst.analyze_compatibility": AsyncMock(
            return_value=analyzed or _analyzed_results()
        ),
        "db.database.get_pool": AsyncMock(return_value=_make_pool()),
    }


def _apply_patches(patch_map):
    """Return a list of patch context managers from a name→mock dict."""
    return [patch(name, new=mock) for name, mock in patch_map.items()]


# ---------------------------------------------------------------------------
# Intent parsing — replaces naive word splitting
# ---------------------------------------------------------------------------

async def test_run_query_calls_parse_intent(capsys):
    from cli import _run_query

    pm = _patches()
    with patch("agents.catalog_query.ensure_base_model_cached", new=pm["agents.catalog_query.ensure_base_model_cached"]), \
         patch("agents.intent_parser.parse_intent", new=pm["agents.intent_parser.parse_intent"]) as mock_intent, \
         patch("agents.catalog_query.query_catalog", new=pm["agents.catalog_query.query_catalog"]), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=pm["agents.compatibility_analyst.analyze_compatibility"]), \
         patch("db.database.get_pool", new=pm["db.database.get_pool"]):
        await _run_query("a cinematic portrait", "Flux.1 D", False, True)

    mock_intent.assert_called_once_with("a cinematic portrait")


async def test_run_query_uses_intent_tags_for_catalog_query(capsys):
    from cli import _run_query

    intent = _make_intent(tags=["sword", "rain", "cinematic"])
    mock_query = AsyncMock(return_value=_catalog_results())

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=intent)), \
         patch("agents.catalog_query.query_catalog", new=mock_query), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=_analyzed_results())), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("swords in the rain", "Flux.1 D", False, True)

    call_kwargs = mock_query.call_args.kwargs
    assert call_kwargs["search_tags"] == ["sword", "rain", "cinematic"]


# ---------------------------------------------------------------------------
# analyze_compatibility gating
# ---------------------------------------------------------------------------

async def test_run_query_calls_analyze_when_llm_used(capsys):
    from cli import _run_query

    mock_analyst = AsyncMock(return_value=_analyzed_results())

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent(llm_used=True))), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=mock_analyst), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("prompt", "Flux.1 D", False, True)

    mock_analyst.assert_called_once()


async def test_run_query_skips_analyze_when_no_llm_flag(capsys):
    from cli import _run_query

    mock_analyst = AsyncMock(return_value=_analyzed_results())

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent(llm_used=True))), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=mock_analyst), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("prompt", "Flux.1 D", False, llm_reasoning=False)

    mock_analyst.assert_not_called()


async def test_run_query_skips_analyze_when_intent_not_llm(capsys):
    from cli import _run_query

    mock_analyst = AsyncMock(return_value=_analyzed_results())

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent(llm_used=False))), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=mock_analyst), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("prompt", "Flux.1 D", False, True)

    mock_analyst.assert_not_called()


async def test_run_query_passes_prompt_style_subject_to_analyst(capsys):
    from cli import _run_query

    intent = _make_intent(llm_used=True, style="fantasy", subject="dragon")
    mock_analyst = AsyncMock(return_value=_analyzed_results())

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=intent)), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=mock_analyst), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("a fantasy dragon", "Flux.1 D", False, True)

    call_kwargs = mock_analyst.call_args.kwargs
    assert call_kwargs["prompt"] == "a fantasy dragon"
    assert call_kwargs["style"] == "fantasy"
    assert call_kwargs["subject"] == "dragon"


# ---------------------------------------------------------------------------
# Cache miss → sys.exit
# ---------------------------------------------------------------------------

async def test_run_query_exits_when_base_model_not_cached():
    from cli import _run_query

    mock_intent = AsyncMock(return_value=_make_intent())

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=False)), \
         patch("agents.intent_parser.parse_intent", new=mock_intent), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        with pytest.raises(SystemExit) as exc:
            await _run_query("any prompt", "Flux.1 D", False, True)

    assert exc.value.code == 1
    mock_intent.assert_not_called()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

async def test_run_query_json_includes_intent_and_phase(capsys):
    from cli import _run_query

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent())), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=_analyzed_results())), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("cinematic portrait", "Flux.1 D", True, True)

    output = json.loads(capsys.readouterr().out)
    assert output["prompt"] == "cinematic portrait"
    assert output["base_model"] == "Flux.1 D"
    assert output["intent"]["llm_used"] is True
    assert "tags" in output["intent"]
    assert output["phase"] == "2b — LLM intent + compatibility analysis"
    assert output["prompt_additions"] == ["actn_combat", "sword_fight", "dramatic lighting"]


async def test_run_query_json_phase_2b_requires_recommendation(capsys):
    """Phase is '2a' when analyze_compatibility returns no recommended_combination."""
    from cli import _run_query

    analyzed = {**_catalog_results(), "recommended_combination": None,
                "combination_notes": None, "prompt_additions": []}

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent())), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=analyzed)), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("prompt", "Flux.1 D", True, True)

    output = json.loads(capsys.readouterr().out)
    assert output["phase"] == "2a — LLM intent parsed"


async def test_run_query_json_phase_1_when_no_llm_intent(capsys):
    from cli import _run_query

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent(llm_used=False))), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock()), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("prompt", "Flux.1 D", True, True)

    output = json.loads(capsys.readouterr().out)
    assert output["phase"] == "1 — stop-word fallback"


async def test_run_query_json_no_llm_flag_returns_raw_results(capsys):
    """With --no-llm the JSON should contain raw catalog results and phase 2a."""
    from cli import _run_query

    mock_analyst = AsyncMock(return_value=_analyzed_results())

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent())), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=mock_analyst), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("prompt", "Flux.1 D", True, llm_reasoning=False)

    mock_analyst.assert_not_called()
    output = json.loads(capsys.readouterr().out)
    assert output["phase"] == "2a — LLM intent parsed"
    assert "prompt_additions" not in output  # raw results have no prompt_additions


# ---------------------------------------------------------------------------
# Terminal output — new fields visible in printed text
# ---------------------------------------------------------------------------

async def test_run_query_print_shows_prompt_additions(capsys):
    from cli import _run_query

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent())), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=_analyzed_results())), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("sword fight in rain", "Flux.1 D", False, True)

    out = capsys.readouterr().out
    assert "actn_combat" in out
    assert "dramatic lighting" in out


async def test_run_query_print_shows_recommended_combination(capsys):
    from cli import _run_query

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent())), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=_analyzed_results())), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("sword fight in rain", "Flux.1 D", False, True)

    out = capsys.readouterr().out
    assert "Flux Realism Pro + Combat Action LoRA" in out
    assert "actn_combat" in out  # appears in combination_notes


async def test_run_query_print_shows_impact_and_compatibility_note(capsys):
    from cli import _run_query

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent())), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=_analyzed_results())), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("sword fight", "Flux.1 D", False, True)

    out = capsys.readouterr().out
    assert "high" in out
    assert "medieval combat" in out  # from compatibility_note


async def test_run_query_print_shows_not_recommended_tag(capsys):
    from cli import _run_query

    analyzed = _analyzed_results()
    analyzed["loras"][0]["recommended"] = False

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent())), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=analyzed)), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("prompt", "Flux.1 D", False, True)

    out = capsys.readouterr().out
    assert "not recommended" in out


async def test_run_query_print_shows_phase(capsys):
    from cli import _run_query

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent())), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=_analyzed_results())), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("prompt", "Flux.1 D", False, True)

    out = capsys.readouterr().out
    assert "2b" in out


async def test_run_query_print_shows_intent_tags(capsys):
    from cli import _run_query

    intent = _make_intent(tags=["sword", "rain", "cinematic"])

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=intent)), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=_analyzed_results())), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("swords in the rain", "Flux.1 D", False, True)

    out = capsys.readouterr().out
    assert "sword" in out
    assert "rain" in out


async def test_run_query_print_no_prompt_additions_section_when_empty(capsys):
    from cli import _run_query

    analyzed = _analyzed_results()
    analyzed["prompt_additions"] = []

    with patch("agents.catalog_query.ensure_base_model_cached", new=AsyncMock(return_value=True)), \
         patch("agents.intent_parser.parse_intent", new=AsyncMock(return_value=_make_intent())), \
         patch("agents.catalog_query.query_catalog", new=AsyncMock(return_value=_catalog_results())), \
         patch("agents.compatibility_analyst.analyze_compatibility", new=AsyncMock(return_value=analyzed)), \
         patch("db.database.get_pool", new=AsyncMock(return_value=_make_pool())):
        await _run_query("prompt", "Flux.1 D", False, True)

    out = capsys.readouterr().out
    assert "ADD TO YOUR PROMPT" not in out
