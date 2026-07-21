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
You are an expert AI image generation pipeline architect who selects the best \
checkpoint and LoRA combination for a user's creative prompt.

You will receive a JSON object with:
  intent      — the user's original prompt text, plus detected style and subject
  checkpoints — candidate base model checkpoints with names, tags, and descriptions
  loras       — candidate LoRA adapters with names, tags, trigger words, and descriptions

How to evaluate:
1. Read model DESCRIPTIONS carefully. Descriptions reveal what a model was actually
   trained on and what it excels at. A LoRA described as "trained on rainy cityscapes
   with neon reflections" is high-impact for a cyberpunk rain scene; its tags alone
   would not tell you this.
2. For each checkpoint, assess whether its description indicates strong handling of
   the requested style, subject, or aesthetic.
3. For each LoRA, judge impact by how directly its description matches the prompt:
   - high: description directly addresses the prompt's core subject, style, or environment
   - medium: description is relevant but not the main focus of the prompt
   - low: description is tangentially related, generic, or the LoRA has no description
4. Build prompt_additions: list trigger words for each recommended LoRA first (required
   for activation), then add descriptive keywords the model descriptions suggest would
   improve results for this specific prompt.

Return ONLY valid JSON with exactly this structure:
{
  "checkpoints": [
    {
      "id": <int>,
      "compatibility_note": "<one sentence: why this fits or doesn't, based on description>",
      "recommended": <true|false>
    }
  ],
  "loras": [
    {
      "id": <int>,
      "compatibility_note": "<one sentence: what this LoRA contributes based on its description>",
      "recommended": <true|false>,
      "recommended_weight": <float 0.4–0.9 or null>,
      "impact": "<high|medium|low>"
    }
  ],
  "recommended_combination": "<checkpoint name> + <top LoRA name(s)>",
  "combination_notes": "<1–2 sentences: why this combination works and any weight tips>",
  "prompt_additions": ["<trigger_word_or_keyword>", ...]
}

Rules:
- Base compatibility_note on the description content, not just the model name or tags
- A LoRA with no description defaults to impact=low unless trigger words strongly match
- recommended_weight: 0.4–0.6 for subtle texture/detail LoRAs; 0.7–0.9 for strong
  style or character LoRAs; null if the description gives no guidance
- prompt_additions: trigger words first (recommended LoRAs in priority order),
  then style keywords their descriptions suggest for this specific prompt\
"""


def _compact(model: dict) -> dict:
    """Reduce a full model record to the fields the LLM needs, saving tokens."""
    result = {
        "id": model["id"],
        "name": model["name"],
        "type": model["type"],
        "tags": (model.get("tags") or [])[:15],
        "trigger_words": (model.get("trigger_words") or [])[:10],
        "description": (model.get("description") or "")[:600],
    }
    if model.get("type") == "Checkpoint":
        settings: dict = {}
        if model.get("recommended_cfg"):
            settings["cfg"] = model["recommended_cfg"]
        if model.get("recommended_steps"):
            settings["steps"] = model["recommended_steps"]
        if model.get("recommended_sampler"):
            settings["sampler"] = model["recommended_sampler"]
        if settings:
            result["settings"] = settings
    return result


def _merge(candidates: list[dict], notes: list[dict]) -> list[dict]:
    note_map = {n["id"]: n for n in notes if isinstance(n.get("id"), int)}
    for m in candidates:
        note = note_map.get(m["id"], {})
        m["compatibility_note"] = note.get("compatibility_note")
        m["recommended"] = note.get("recommended", True)
        if "recommended_weight" in note:
            m["recommended_weight"] = note["recommended_weight"]
        if "impact" in note:
            m["impact"] = note["impact"]
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
            options={"temperature": 0.2, "num_predict": 2048},
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
        "prompt_additions": data.get("prompt_additions") or [],
    }
