"""
Tests for agents/intent_parser.py.

Ollama is always mocked — no real LLM calls are made.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import ollama
import pytest

from agents.intent_parser import IntentResult, _fallback, parse_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_resp(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.response = json.dumps(data)
    return resp


def _mock_client(*side_effects):
    """Return a mock AsyncClient whose generate() returns each side_effect in order."""
    client = MagicMock()
    client.generate = AsyncMock(side_effect=list(side_effects))
    return client


_GOOD_TAGS = ["portrait", "anime", "warrior"]
_GOOD_RESP = {"tags": _GOOD_TAGS, "style": "anime art", "subject": "warrior portrait"}


# ---------------------------------------------------------------------------
# _fallback — stop-word filtering
# ---------------------------------------------------------------------------

def test_fallback_keeps_content_words():
    result = _fallback("epic fantasy warrior portrait")
    for word in ("epic", "fantasy", "warrior", "portrait"):
        assert word in result.tags
    assert result.llm_used is False


def test_fallback_removes_stop_words():
    result = _fallback("a portrait of the warrior in the rain")
    for word in ("portrait", "warrior", "rain"):
        assert word in result.tags
    for stop in ("a", "of", "the", "in"):
        assert stop not in result.tags


def test_fallback_filters_four_letter_stop_words():
    """'with', 'from', 'they' are 4+ chars but must still be dropped."""
    result = _fallback("dark fantasy character with sword from myth")
    for word in ("dark", "fantasy", "character", "sword", "myth"):
        assert word in result.tags
    for stop in ("with", "from"):
        assert stop not in result.tags


def test_fallback_strips_punctuation():
    result = _fallback("swords, daggers, and shields!")
    assert "swords" in result.tags
    assert "daggers" in result.tags
    assert "shields" in result.tags


def test_fallback_result_is_not_llm_used():
    assert _fallback("some prompt").llm_used is False


# ---------------------------------------------------------------------------
# parse_intent — happy path
# ---------------------------------------------------------------------------

async def test_parse_intent_returns_llm_tags_on_success():
    client = _mock_client(_ollama_resp(_GOOD_RESP))

    with patch("agents.intent_parser.ollama.AsyncClient", return_value=client):
        result = await parse_intent("anime warrior portrait")

    assert result.llm_used is True
    assert "portrait" in result.tags
    assert "anime" in result.tags
    assert result.style == "anime art"
    assert result.subject == "warrior portrait"


async def test_parse_intent_normalises_tags_to_lowercase():
    resp = _ollama_resp({"tags": ["Portrait", "ANIME", "Warrior"], "style": "", "subject": ""})
    client = _mock_client(resp)

    with patch("agents.intent_parser.ollama.AsyncClient", return_value=client):
        result = await parse_intent("some prompt")

    assert all(t == t.lower() for t in result.tags)


async def test_parse_intent_caps_tags_at_20():
    resp = _ollama_resp({"tags": [f"tag{i}" for i in range(30)], "style": "", "subject": ""})
    client = _mock_client(resp)

    with patch("agents.intent_parser.ollama.AsyncClient", return_value=client):
        result = await parse_intent("some prompt")

    assert len(result.tags) <= 20


async def test_parse_intent_uses_env_vars_for_host_and_primary_model():
    client = _mock_client(_ollama_resp(_GOOD_RESP))

    with (
        patch("agents.intent_parser.ollama.AsyncClient", return_value=client) as MockClient,
        patch.dict("os.environ", {
            "OLLAMA_HOST": "http://192.168.1.50:11434",
            "OLLAMA_MODEL_PRIMARY": "custom-model:7b",
        }),
    ):
        await parse_intent("test prompt")

    MockClient.assert_called_once_with(host="http://192.168.1.50:11434")
    assert client.generate.call_args.kwargs["model"] == "custom-model:7b"


# ---------------------------------------------------------------------------
# parse_intent — primary fails, fallback succeeds
# ---------------------------------------------------------------------------

async def test_parse_intent_tries_fallback_model_when_primary_errors():
    fallback_resp = _ollama_resp({"tags": ["fantasy", "warrior"], "style": "fantasy", "subject": "warrior"})
    client = _mock_client(
        ollama.ResponseError("primary model not found"),
        fallback_resp,
    )

    with patch("agents.intent_parser.ollama.AsyncClient", return_value=client):
        result = await parse_intent("fantasy warrior")

    assert result.llm_used is True
    assert "fantasy" in result.tags
    assert client.generate.call_count == 2


async def test_parse_intent_tries_fallback_when_primary_returns_invalid_json():
    bad = MagicMock()
    bad.response = "not json {"
    fallback_resp = _ollama_resp({"tags": ["fantasy"], "style": "", "subject": ""})
    client = _mock_client(bad, fallback_resp)

    with patch("agents.intent_parser.ollama.AsyncClient", return_value=client):
        result = await parse_intent("fantasy prompt")

    assert result.llm_used is True
    assert "fantasy" in result.tags


async def test_parse_intent_tries_fallback_when_primary_returns_empty_tags():
    empty = _ollama_resp({"tags": [], "style": "", "subject": ""})
    fallback_resp = _ollama_resp({"tags": ["portrait"], "style": "", "subject": ""})
    client = _mock_client(empty, fallback_resp)

    with patch("agents.intent_parser.ollama.AsyncClient", return_value=client):
        result = await parse_intent("fantasy warrior portrait")

    assert result.llm_used is True
    assert "portrait" in result.tags


# ---------------------------------------------------------------------------
# parse_intent — both models fail → stop-word fallback
# ---------------------------------------------------------------------------

async def test_parse_intent_falls_back_to_stop_words_when_both_models_fail():
    client = _mock_client(
        ollama.ResponseError("primary failed"),
        ollama.ResponseError("fallback failed"),
    )

    with patch("agents.intent_parser.ollama.AsyncClient", return_value=client):
        result = await parse_intent("fantasy warrior portrait")

    assert result.llm_used is False
    assert "fantasy" in result.tags
    assert "warrior" in result.tags


async def test_parse_intent_falls_back_when_both_models_return_invalid_json():
    bad = MagicMock()
    bad.response = "not json"
    client = _mock_client(bad, bad)

    with patch("agents.intent_parser.ollama.AsyncClient", return_value=client):
        result = await parse_intent("fantasy warrior")

    assert result.llm_used is False


async def test_parse_intent_falls_back_when_ollama_unreachable():
    with patch(
        "agents.intent_parser.ollama.AsyncClient",
        side_effect=Exception("connection refused"),
    ):
        result = await parse_intent("fantasy warrior")

    assert result.llm_used is False
    assert "fantasy" in result.tags
