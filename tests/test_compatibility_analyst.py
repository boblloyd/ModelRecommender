"""
Tests for agents/compatibility_analyst.py.

Ollama is always mocked — no real LLM calls are made.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import ollama
import pytest

from agents.compatibility_analyst import analyze_compatibility
from tests.conftest import make_db_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_resp(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.response = json.dumps(data)
    return resp


def _mock_client(*side_effects):
    client = MagicMock()
    client.generate = AsyncMock(side_effect=list(side_effects))
    return client


def _analyst_payload(
    checkpoints=None,
    loras=None,
    combination="Best Checkpoint + Style LoRA",
    notes="Use trigger word 'style_key' at weight 0.7.",
) -> dict:
    return {
        "checkpoints": checkpoints or [],
        "loras": loras or [],
        "recommended_combination": combination,
        "combination_notes": notes,
    }


def _cp(**overrides):
    return make_db_model(type="Checkpoint", civitai_version_id=11111, **overrides)


def _lora(**overrides):
    return make_db_model(type="LORA", **overrides)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_analyze_compatibility_merges_note_into_each_candidate():
    checkpoint = _cp(id=1)
    lora = _lora(id=2)
    response = _analyst_payload(
        checkpoints=[{"id": 1, "compatibility_note": "Great realism match", "recommended": True}],
        loras=[{"id": 2, "compatibility_note": "Matches style well", "recommended": True, "recommended_weight": 0.7}],
    )
    client = _mock_client(_ollama_resp(response))

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility(
            "cinematic portrait", "cinematic", "portrait",
            {"checkpoints": [checkpoint], "loras": [lora]},
        )

    assert output["checkpoints"][0]["compatibility_note"] == "Great realism match"
    assert output["checkpoints"][0]["recommended"] is True
    assert output["loras"][0]["compatibility_note"] == "Matches style well"
    assert output["loras"][0]["recommended_weight"] == 0.7


async def test_analyze_compatibility_adds_recommended_combination_to_output():
    checkpoint = _cp(id=1)
    lora = _lora(id=2)
    response = _analyst_payload(
        checkpoints=[{"id": 1, "compatibility_note": "Good", "recommended": True}],
        loras=[{"id": 2, "compatibility_note": "Good", "recommended": True}],
        combination="Flux Realism Pro + Combat LoRA",
        notes="Use trigger word 'actn_combat' at weight 0.7.",
    )
    client = _mock_client(_ollama_resp(response))

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility(
            "a cinematic sword fight", "cinematic", "sword fight",
            {"checkpoints": [checkpoint], "loras": [lora]},
        )

    assert output["recommended_combination"] == "Flux Realism Pro + Combat LoRA"
    assert "actn_combat" in output["combination_notes"]


async def test_analyze_compatibility_marks_poor_matches_as_not_recommended():
    lora = _lora(id=1)
    response = _analyst_payload(
        loras=[{"id": 1, "compatibility_note": "Wrong style for this prompt", "recommended": False}],
    )
    client = _mock_client(_ollama_resp(response))

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility(
            "anime portrait", "anime", "portrait",
            {"checkpoints": [], "loras": [lora]},
        )

    assert output["loras"][0]["recommended"] is False


async def test_analyze_compatibility_default_recommended_is_true_when_note_absent():
    """Models not mentioned by the LLM default to recommended=True."""
    lora = _lora(id=1)
    response = _analyst_payload(loras=[])  # no note for id=1
    client = _mock_client(_ollama_resp(response))

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility(
            "prompt", "", "",
            {"checkpoints": [], "loras": [lora]},
        )

    assert output["loras"][0]["recommended"] is True


# ---------------------------------------------------------------------------
# Fallback — primary fails, fallback model succeeds
# ---------------------------------------------------------------------------

async def test_analyze_compatibility_tries_fallback_model_when_primary_errors():
    lora = _lora(id=1)
    response = _analyst_payload(
        loras=[{"id": 1, "compatibility_note": "Good", "recommended": True}],
    )
    client = _mock_client(
        ollama.ResponseError("primary model not found"),
        _ollama_resp(response),
    )

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility(
            "prompt", "", "",
            {"checkpoints": [], "loras": [lora]},
        )

    assert output["loras"][0]["compatibility_note"] == "Good"
    assert client.generate.call_count == 2


async def test_analyze_compatibility_tries_fallback_when_primary_returns_invalid_json():
    lora = _lora(id=1)
    bad = MagicMock()
    bad.response = "not valid json {"
    good = _ollama_resp(_analyst_payload(
        loras=[{"id": 1, "compatibility_note": "Works", "recommended": True}],
    ))
    client = _mock_client(bad, good)

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility(
            "prompt", "", "",
            {"checkpoints": [], "loras": [lora]},
        )

    assert output["loras"][0]["compatibility_note"] == "Works"


# ---------------------------------------------------------------------------
# Both models fail → return original results unchanged
# ---------------------------------------------------------------------------

async def test_analyze_compatibility_returns_original_results_when_both_models_fail():
    original = {"checkpoints": [], "loras": [_lora(id=1)]}
    client = _mock_client(
        ollama.ResponseError("primary failed"),
        ollama.ResponseError("fallback failed"),
    )

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility("prompt", "", "", original)

    assert output is original


async def test_analyze_compatibility_returns_original_when_both_models_return_bad_json():
    original = {"checkpoints": [], "loras": [_lora(id=1)]}
    bad = MagicMock()
    bad.response = "not json"
    client = _mock_client(bad, bad)

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility("prompt", "", "", original)

    assert output is original


async def test_analyze_compatibility_returns_original_when_ollama_unreachable():
    original = {"checkpoints": [], "loras": [_lora(id=1)]}

    with patch(
        "agents.compatibility_analyst.ollama.AsyncClient",
        side_effect=Exception("connection refused"),
    ):
        output = await analyze_compatibility("prompt", "", "", original)

    assert output is original


# ---------------------------------------------------------------------------
# Short-circuit — no candidates
# ---------------------------------------------------------------------------

async def test_analyze_compatibility_skips_llm_when_no_candidates():
    with patch("agents.compatibility_analyst.ollama.AsyncClient") as MockClient:
        output = await analyze_compatibility(
            "prompt", "", "",
            {"checkpoints": [], "loras": []},
        )

    MockClient.assert_not_called()
    assert output == {"checkpoints": [], "loras": []}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

async def test_analyze_compatibility_ignores_notes_for_unknown_model_id():
    lora = _lora(id=1)
    response = _analyst_payload(
        loras=[{"id": 999, "compatibility_note": "Ghost note", "recommended": False}],
    )
    client = _mock_client(_ollama_resp(response))

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility(
            "prompt", "", "",
            {"checkpoints": [], "loras": [lora]},
        )

    # id=1 lora is unaffected by the note intended for id=999
    assert output["loras"][0]["compatibility_note"] is None
    assert output["loras"][0]["recommended"] is True


async def test_analyze_compatibility_returns_original_when_response_lacks_expected_keys():
    """Response with no 'checkpoints' or 'loras' key is treated as unusable."""
    original = {"checkpoints": [], "loras": [_lora(id=1)]}
    bad_structure = _ollama_resp({"recommended_combination": "something else"})
    client = _mock_client(bad_structure, bad_structure)

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        output = await analyze_compatibility("prompt", "", "", original)

    assert output is original


async def test_analyze_compatibility_passes_compact_payload_to_llm():
    """Verify only trimmed fields are sent (not the full DB record)."""
    lora = _lora(id=1, description="x" * 1000, tags=[f"tag{i}" for i in range(50)])
    response = _analyst_payload(loras=[{"id": 1, "compatibility_note": "ok", "recommended": True}])
    client = _mock_client(_ollama_resp(response))

    with patch("agents.compatibility_analyst.ollama.AsyncClient", return_value=client):
        await analyze_compatibility("prompt", "", "", {"checkpoints": [], "loras": [lora]})

    # The prompt sent to the LLM should not contain the full 1000-char description
    call_prompt = client.generate.call_args.kwargs["prompt"]
    parsed = json.loads(call_prompt)
    sent_lora = parsed["loras"][0]
    assert len(sent_lora.get("description", "")) <= 400
    assert len(sent_lora.get("tags", [])) <= 15
