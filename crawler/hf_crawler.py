"""
HuggingFace Hub model catalog crawler.

Crawls HuggingFace for LoRA and Checkpoint models compatible with a given
base model and stores them in the same PostgreSQL catalog as Civitai models.

Pagination uses the HTTP Link response header (cursor-based), not a JSON body
field, so _fetch_page returns (data, link_header).

Usage:
  python -m crawler.hf_crawler --base-model "Flux.1 D" --mode full
  python -m crawler.hf_crawler --base-model "Flux.1 D" --mode incremental
"""

import argparse
import asyncio
import logging
import os
import re
import sys

import asyncpg
import httpx

from db.database import get_pool, init_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api"
PAGE_SIZE = int(os.environ.get("CRAWL_PAGE_SIZE", "100"))
BACKOFF_START = float(os.environ.get("RATE_LIMIT_BACKOFF_START", "2"))
BACKOFF_MAX = float(os.environ.get("RATE_LIMIT_BACKOFF_MAX", "30"))

# HuggingFace tag used in API filter requests for each base model
BASE_MODEL_HF_TAGS = {
    "Flux.1 D": "base_model:black-forest-labs/FLUX.1-dev",
    "SDXL 1.0": "base_model:stabilityai/stable-diffusion-xl-base-1.0",
    "SD 1.5": "base_model:runwayml/stable-diffusion-v1-5",
}

# cardData.base_model values that indicate compatibility with each base model
BASE_MODEL_HF_REPO_IDS: dict[str, list[str]] = {
    "Flux.1 D": ["black-forest-labs/FLUX.1-dev", "black-forest-labs/flux-dev"],
    "SDXL 1.0": ["stabilityai/stable-diffusion-xl-base-1.0"],
    "SD 1.5": ["runwayml/stable-diffusion-v1-5"],
}


def _parse_next_url(link_header: str | None) -> str | None:
    """Extract the URL with rel="next" from an HTTP Link header."""
    if not link_header:
        return None
    m = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return m.group(1) if m else None


def _infer_type(tags: list[str]) -> str | None:
    """
    Determine model type from HuggingFace tags.

    HuggingFace has no explicit type field — LoRAs are identified by the
    'lora' tag; everything else with diffusers support is treated as a Checkpoint.
    """
    tag_set = {t.lower() for t in tags}
    if "lora" in tag_set:
        return "LORA"
    if "diffusers" in tag_set:
        return "Checkpoint"
    return None


def _base_model_matches(item: dict, base_model: str) -> bool:
    """
    Return True if the model's cardData.base_model is compatible with
    the target base model.  Unknown base models pass through unfiltered.
    """
    expected_ids = BASE_MODEL_HF_REPO_IDS.get(base_model, [])
    if not expected_ids:
        return True

    card_data = item.get("cardData") or {}
    raw = card_data.get("base_model") or ""
    # cardData.base_model can be a string or a list
    if isinstance(raw, list):
        candidates = [str(x).lower() for x in raw]
    else:
        candidates = [str(raw).lower()]

    return any(
        expected.lower() in candidate
        for expected in expected_ids
        for candidate in candidates
    )


def _extract_record(item: dict, base_model: str) -> dict | None:
    """Map a HuggingFace API model object to a DB record dict."""
    repo_id = item.get("id") or item.get("modelId")
    if not repo_id:
        return None

    tags = [t for t in (item.get("tags") or []) if not t.startswith("arxiv:")]
    model_type = _infer_type(tags)
    if model_type is None:
        return None

    if not _base_model_matches(item, base_model):
        return None

    return {
        "source": "huggingface",
        "civitai_model_id": None,
        "civitai_version_id": None,
        "hf_repo_id": repo_id,
        "name": repo_id.split("/")[-1].replace("-", " ").replace("_", " "),
        "version_name": None,
        "type": model_type,
        "base_model": base_model,
        "nsfw_level": 1,
        "description": None,
        "tags": tags,
        "trigger_words": [],
        "recommended_weight": None,
        "recommended_cfg": None,
        "recommended_steps": None,
        "recommended_sampler": None,
        "download_url": f"https://huggingface.co/{repo_id}",
        "civitai_url": None,
        "stats_downloads": item.get("downloads") or 0,
        "stats_thumbs_up": item.get("likes") or 0,
        "stats_thumbs_down": 0,
        "preview_image_url": None,
    }


async def _upsert_batch(pool: asyncpg.Pool, records: list[dict]) -> int:
    if not records:
        return 0

    sql = """
        INSERT INTO models (
            source, civitai_model_id, civitai_version_id, hf_repo_id,
            name, version_name, type, base_model, nsfw_level,
            description, tags, trigger_words,
            recommended_weight, recommended_cfg, recommended_steps, recommended_sampler,
            download_url, civitai_url,
            stats_downloads, stats_thumbs_up, stats_thumbs_down,
            preview_image_url,
            date_cached, date_updated
        ) VALUES (
            $1,  $2,  $3,  $4,
            $5,  $6,  $7,  $8,  $9,
            $10, $11, $12,
            $13, $14, $15, $16,
            $17, $18,
            $19, $20, $21,
            $22,
            NOW(), NOW()
        )
        ON CONFLICT (hf_repo_id) DO UPDATE SET
            name                = EXCLUDED.name,
            type                = EXCLUDED.type,
            base_model          = EXCLUDED.base_model,
            nsfw_level          = EXCLUDED.nsfw_level,
            tags                = EXCLUDED.tags,
            download_url        = EXCLUDED.download_url,
            stats_downloads     = EXCLUDED.stats_downloads,
            stats_thumbs_up     = EXCLUDED.stats_thumbs_up,
            date_updated        = NOW()
    """

    rows = [
        (
            r["source"], r["civitai_model_id"], r["civitai_version_id"], r["hf_repo_id"],
            r["name"], r["version_name"], r["type"], r["base_model"], r["nsfw_level"],
            r["description"], r["tags"], r["trigger_words"],
            r["recommended_weight"], r["recommended_cfg"],
            r["recommended_steps"], r["recommended_sampler"],
            r["download_url"], r["civitai_url"],
            r["stats_downloads"], r["stats_thumbs_up"], r["stats_thumbs_down"],
            r["preview_image_url"],
        )
        for r in records
    ]

    async with pool.acquire() as conn:
        await conn.executemany(sql, rows)
    return len(records)


async def _fetch_page(
    client: httpx.AsyncClient,
    url: str,
    token: str | None,
    params: dict | None = None,
) -> tuple[list, str | None]:
    """
    Fetch one page from the HuggingFace API.

    Returns (items_list, link_header).  Pagination URL lives in the Link
    response header, not in the JSON body.
    """
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    backoff = BACKOFF_START
    while True:
        try:
            resp = await client.get(url, headers=headers, params=params, timeout=30.0)
        except httpx.RequestError as exc:
            log.warning("Request error: %s — retrying in %.0fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue

        if resp.status_code == 429:
            log.warning("Rate limited — backing off %.0fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue

        resp.raise_for_status()
        return resp.json(), resp.headers.get("Link")


async def full_crawl(base_model: str, pool: asyncpg.Pool) -> int:
    """Page through HuggingFace models for a base model and cache them."""
    token = os.environ.get("HF_API_TOKEN")
    total = 0

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO base_model_index (base_model_name, crawl_complete)
            VALUES ($1, FALSE)
            ON CONFLICT (base_model_name) DO UPDATE SET crawl_complete = FALSE
            """,
            base_model,
        )

    hf_tag = BASE_MODEL_HF_TAGS.get(base_model)
    params: dict = {
        "full": "true",
        "limit": PAGE_SIZE,
        "sort": "downloads",
        "direction": -1,
    }
    if hf_tag:
        params["filter"] = hf_tag

    base_url = f"{HF_API_BASE}/models"
    next_url: str | None = None

    async with httpx.AsyncClient() as client:
        page = 1
        while True:
            log.info("Page %d — '%s' (HuggingFace) …", page, base_model)
            if next_url:
                items, link_header = await _fetch_page(client, next_url, token)
            else:
                items, link_header = await _fetch_page(client, base_url, token, params=params)

            records = [
                r
                for item in items
                if (r := _extract_record(item, base_model)) is not None
            ]
            inserted = await _upsert_batch(pool, records)
            total += inserted
            log.info("  → %d records upserted (running total: %d)", inserted, total)

            next_url = _parse_next_url(link_header)
            page += 1
            if not next_url:
                break

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE base_model_index
            SET crawl_complete = TRUE,
                last_crawled   = NOW(),
                total_models   = $2
            WHERE base_model_name = $1
            """,
            base_model,
            total,
        )

    log.info("HuggingFace crawl complete for '%s': %d models cached.", base_model, total)
    return total


async def incremental_update(base_model: str, pool: asyncpg.Pool) -> int:
    """
    Fetch only new HuggingFace models since the last crawl.

    HuggingFace's cursor-based pagination doesn't support reliable stop-on-known-ID
    semantics, so this delegates to a full crawl.  The upsert is idempotent, so
    re-crawling existing models just refreshes their download/like counts.
    """
    log.info("HuggingFace incremental update for '%s' — running full crawl (idempotent).", base_model)
    return await full_crawl(base_model, pool)


async def _main() -> None:
    parser = argparse.ArgumentParser(description="HuggingFace model catalog crawler.")
    parser.add_argument("--base-model", required=True, help="Base model to crawl, e.g. 'Flux.1 D'")
    parser.add_argument(
        "--mode", choices=["full", "incremental"], default="full",
        help="Crawl mode (default: full)",
    )
    args = parser.parse_args()

    pool = await get_pool()
    await init_schema(pool)

    try:
        if args.mode == "full":
            await full_crawl(args.base_model, pool)
        else:
            await incremental_update(args.base_model, pool)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(_main())
