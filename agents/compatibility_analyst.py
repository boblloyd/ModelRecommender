"""
Compatibility Analyst — evaluates top candidate models against the user's intent
using a local Ollama LLM.

Adds a compatibility_note and recommended flag to each candidate, plus a
recommended_combination summary.  Returns results unchanged if Ollama is
unavailable or returns unparseable output.
"""

import json
import logging
import os
from typing import Any

import ollama

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert in AI image generation models (Stable Diffusion, Flux) assessing \
candidate checkpoints and LoRAs for a user's creative prompt.

You will receive a JSON object with:
  intent     — the user's prompt, detected style, and subject
  checkpoints — candidate base model checkpoints
  loras       — candidate LoRA adapters

Evaluate each candidate and return ONLY valid JSON with exactly this structure:
{
  "checkpoints": [
    {
      "id": <int>,
      "compatibility_note": "<one sentence>",
      "recommended": <true|false>
    }
  ],
  "loras": [
    {
      "id": <int>,
      "compatibility_note": "<one sentence>",
      "recommended": <true|false>,
      "recommended_weight": <float 0.0–1.0 or null>
    }
  ],
  "recommended_combination": "<checkpoint name> + <lora name(s)>",
  "combination_notes": "<1–2 sentences: why this combination works and any trigger word or weight tips>"
}

Guidelines:
- compatibility_note: one concise sentence on fit or mismatch with the user's intent
- recommended_weight: suggest lower (0.4–0.6) for subtle detail/texture LoRAs, higher (0.7–0.9) for strong style LoRAs; null if uncertain
- Set recommended=false for candidates that clearly don't match the intent
- combination_notes must mention trigger words if any LoRA requires them\
"""


def _compact(model: dict) -> dict:
    """Reduce a full model record to the fields the LLM needs, saving tokens."""
    return {
        "id": model["id"],
        "name": model["name"],
        "type": model["type"],
        "tags": (model.get("tags") or [])[:15],
        "trigger_words": (model.get("trigger_words") or [])[:10],
        "description": (model.get("description") or "")[:400],
    }


def _merge(candidates: list[dict], notes: list[dict]) -> list[dict]:
    note_map = {n["id"]: n for n in notes if isinstance(n.get("id"), int)}
    for m in candidates:
        note = note_map.get(m["id"], {})
        m["compatibility_note"] = note.get("compatibility_note")
        m["recommended"] = note.get("recommended", True)
        if "recommended_weight" in note:
            m["recommended_weight"] = note["recommended_weight"]
    return candidates


async def _try_analysis(
    client: ollama.AsyncClient,
    model: str,
    payload: dict,
) -> dict | None:
    try:
        resp = await client.generate(
            model=model,
            system=_SYSTEM_PROMPT,
            prompt=json.dumps(payload),
            format="json",
            options={"temperature": 0.2, "num_predict": 1024},
        )
        data = json.loads(resp.response)
        if "checkpoints" not in data and "loras" not in data:
            log.warning("Compatibility analyst response missing expected keys from %s", model)
            return None
        return data
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        log.warning("Compatibility analyst got bad JSON from model %s: %s", model, exc)
        return None
    except ollama.ResponseError as exc:
        log.warning("Ollama model %s error: %s", model, exc)
        return None


async def analyze_compatibility(
    prompt: str,
    style: str,
    subject: str,
    results: dict[str, Any],
) -> dict[str, Any]:
    """
    Annotate catalog results with LLM compatibility notes.

    Merges compatibility_note, recommended, and recommended_weight into each
    candidate dict, and adds recommended_combination and combination_notes at
    the top level.  Returns results unchanged on any failure.
    """
    checkpoints: list[dict] = results.get("checkpoints", [])
    loras: list[dict] = results.get("loras", [])

    if not checkpoints and not loras:
        return results

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    primary = os.environ.get("OLLAMA_MODEL_PRIMARY", "dolphin3.0-llama3.1:8b")
    fallback_model = os.environ.get("OLLAMA_MODEL_FALLBACK", "gemma3:12b")

    payload = {
        "intent": {"prompt": prompt, "style": style, "subject": subject},
        "checkpoints": [_compact(m) for m in checkpoints],
        "loras": [_compact(m) for m in loras],
    }

    data: dict | None = None
    try:
        client = ollama.AsyncClient(host=host)
        data = await _try_analysis(client, primary, payload)
        if data is None:
            log.info("Primary model failed — trying fallback for compatibility analysis.")
            data = await _try_analysis(client, fallback_model, payload)
    except Exception as exc:
        log.warning("Ollama unreachable for compatibility analysis: %s", exc)

    if data is None:
        log.info("Compatibility analyst unavailable — returning raw catalog results.")
        return results

    return {
        "checkpoints": _merge(checkpoints, data.get("checkpoints") or []),
        "loras": _merge(loras, data.get("loras") or []),
        "recommended_combination": data.get("recommended_combination"),
        "combination_notes": data.get("combination_notes"),
    }
