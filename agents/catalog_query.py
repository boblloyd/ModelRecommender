"""
Catalog Query Agent — pure SQL, no LLM.

Scores candidate models from the local PostgreSQL cache using:
  tag_overlap   × 0.60  (primary semantic signal)
  quality_score × 0.25  (thumbs up ratio)
  download_norm × 0.15  (log-normalised download count, 1M ≈ 1.0)
"""

import math
from typing import Any

import asyncpg


def _score(model: dict, search_tags: list[str]) -> float:
    model_tags = {t.lower() for t in (model["tags"] or [])}
    trigger_words = {t.lower() for t in (model["trigger_words"] or [])}
    all_tags = model_tags | trigger_words

    if search_tags:
        search_set = {t.lower() for t in search_tags}
        tag_overlap = len(all_tags & search_set) / len(search_set)
    else:
        tag_overlap = 0.0

    up = model["stats_thumbs_up"] or 0
    down = model["stats_thumbs_down"] or 0
    quality = up / (up + down) if (up + down) > 0 else 0.0

    downloads = model["stats_downloads"] or 0
    download_norm = math.log10(downloads + 1) / 6

    return round((tag_overlap * 0.6) + (quality * 0.25) + (download_norm * 0.15), 4)


async def query_catalog(
    search_tags: list[str],
    pool: asyncpg.Pool,
    base_model: str = "Flux.1 D",
    nsfw_max: int | None = None,
    limit_checkpoints: int = 5,
    limit_loras: int = 15,
) -> dict[str, Any]:
    """
    Return scored and ranked checkpoints and LoRAs for the given search tags.

    Results are split by type and sorted by relevance_score descending.
    """
    conditions: list[str] = ["base_model = $1"]
    params: list[Any] = [base_model]

    if nsfw_max is not None:
        conditions.append(f"nsfw_level <= ${len(params) + 1}")
        params.append(nsfw_max)

    where = " AND ".join(conditions)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT
                id, source, civitai_model_id, civitai_version_id,
                name, version_name, type, base_model, nsfw_level,
                description, tags, trigger_words,
                recommended_weight, recommended_cfg,
                recommended_steps, recommended_sampler,
                download_url, civitai_url,
                stats_downloads, stats_thumbs_up, stats_thumbs_down,
                preview_image_url,
                date_cached, date_updated
            FROM models
            WHERE {where}
            """,
            *params,
        )

    models = [dict(r) for r in rows]
    for m in models:
        m["relevance_score"] = _score(m, search_tags)

    checkpoints = sorted(
        [m for m in models if m["type"] == "Checkpoint"],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )[:limit_checkpoints]

    loras = sorted(
        [m for m in models if m["type"] in ("LORA", "LoCon")],
        key=lambda x: x["relevance_score"],
        reverse=True,
    )[:limit_loras]

    return {"checkpoints": checkpoints, "loras": loras}


async def ensure_base_model_cached(base_model: str, pool: asyncpg.Pool) -> bool:
    """Return True if a completed crawl exists for this base model."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT crawl_complete FROM base_model_index WHERE base_model_name = $1",
            base_model,
        )
    return bool(row and row["crawl_complete"])
