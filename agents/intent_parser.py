"""
Intent Parser — converts a natural language prompt into structured search tags
using a local Ollama LLM.

Falls back to simple stop-word filtering if Ollama is unavailable or returns
unparseable output.
"""

import json
import logging
import os
from dataclasses import dataclass, field

import ollama

log = logging.getLogger(__name__)

# Used only when Ollama is unavailable.
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "not", "nor",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those",
    "as", "so", "if", "do", "did", "has", "have", "had",
    "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "his", "her", "their",
    "into", "onto", "upon", "up", "out", "no", "go",
    "want", "like", "need", "make", "create", "generate",
    "image", "picture", "photo",
})

_SYSTEM_PROMPT = """\
You are a search query extractor for an AI image generation model catalog.
The catalog contains checkpoint models and LoRA adapters for Stable Diffusion and Flux.

Given the user's creative prompt, extract terms that would match relevant models in the catalog.

Return ONLY valid JSON with exactly this structure:
{
  "tags": ["tag1", "tag2", ...],
  "style": "brief art style summary",
  "subject": "brief subject or theme summary"
}

Rules for tags:
- Up to 20 terms; single words or short 2-word phrases only
- Include content tags (what is depicted: "portrait", "landscape", "fantasy", "sci-fi")
- Include style tags (how it looks: "anime", "realistic", "painterly", "cinematic", "watercolor")
- Include mood/atmosphere tags ("dark", "vibrant", "ethereal", "gritty") where relevant
- Include adult content tags if the prompt implies it — this catalog includes NSFW models
- Omit articles, prepositions, conjunctions, and generic words like "image" or "generate"\
"""


@dataclass
class IntentResult:
    tags: list[str] = field(default_factory=list)
    style: str = ""
    subject: str = ""
    llm_used: bool = False


def _fallback(prompt: str) -> IntentResult:
    tags = [
        word
        for w in prompt.lower().split()
        if (word := w.strip(".,!?;:\"'()")) and word not in _STOP_WORDS
    ]
    return IntentResult(tags=tags, llm_used=False)


async def _try_model(client: ollama.AsyncClient, model: str, prompt: str) -> IntentResult | None:
    try:
        resp = await client.generate(
            model=model,
            system=_SYSTEM_PROMPT,
            prompt=f"User prompt: {prompt}",
            format="json",
            options={"temperature": 0.1, "num_predict": 300},
        )
        data = json.loads(resp.response)
        tags = [str(t).lower().strip() for t in (data.get("tags") or []) if t]
        if not tags:
            log.warning("Intent parser returned empty tags from model %s", model)
            return None
        return IntentResult(
            tags=tags[:20],
            style=str(data.get("style") or ""),
            subject=str(data.get("subject") or ""),
            llm_used=True,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        log.warning("Intent parser got bad JSON from model %s: %s", model, exc)
        return None
    except ollama.ResponseError as exc:
        log.warning("Ollama model %s error: %s", model, exc)
        return None


async def parse_intent(prompt: str) -> IntentResult:
    """
    Extract structured search tags from a natural language prompt.

    Tries the primary Ollama model, then the fallback model, then
    returns a basic stop-word-filtered result if both fail.
    """
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    primary = os.environ.get("OLLAMA_MODEL_PRIMARY", "dolphin3.0-llama3.1:8b")
    fallback_model = os.environ.get("OLLAMA_MODEL_FALLBACK", "gemma3:12b")

    try:
        client = ollama.AsyncClient(host=host)

        result = await _try_model(client, primary, prompt)
        if result is not None:
            return result

        log.info("Primary model %s failed — trying fallback %s", primary, fallback_model)
        result = await _try_model(client, fallback_model, prompt)
        if result is not None:
            return result

    except Exception as exc:
        log.warning("Ollama unreachable at %s: %s", host, exc)

    log.info("Intent parser falling back to stop-word filtering.")
    return _fallback(prompt)
