"""
Civitai model catalog crawler.

Runs as a standalone process — designed as a Kubernetes Job entrypoint.
Exits with code 0 on success, non-zero on failure, so k8s Job tracking works correctly.

Usage:
  python -m crawler.civitai_crawler --base-model "Flux.1 D" --mode full
  python -m crawler.civitai_crawler --base-model "Flux.1 D" --mode incremental
"""

import argparse
import asyncio
import html
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

CIVITAI_API_BASE = "https://civitai.com/api/v1"
CRAWL_TYPES = ["Checkpoint", "LORA", "LoCon"]
PAGE_SIZE = int(os.environ.get("CRAWL_PAGE_SIZE", "100"))
BACKOFF_START = float(os.environ.get("RATE_LIMIT_BACKOFF_START", "2"))
BACKOFF_MAX = float(os.environ.get("RATE_LIMIT_BACKOFF_MAX", "30"))


def _strip_html(raw: str | None) -> str | None:
    if not raw:
        return None
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return " ".join(text.split()) or None


def _civitai_url(model_id: int, version_id: int) -> str:
    return f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"


def _extract_record(item: dict) -> dict | None:
    versions = item.get("modelVersions") or []
    if not versions:
        return None
    version = versions[0]
    version_id = version.get("id")
    model_id = item.get("id")
    if not version_id or not model_id:
        return None

    images = version.get("images") or []
    first_image = images[0] if images else {}
    image_meta = first_image.get("meta") or {}
    preview_url = first_image.get("url")

    stats = item.get("stats") or {}

    return {
        "source": "civitai",
        "civitai_model_id": model_id,
        "civitai_version_id": version_id,
        "name": item.get("name", ""),
        "version_name": version.get("name"),
        "type": item.get("type", ""),
        "base_model": version.get("baseModel"),
        "nsfw_level": item.get("nsfwLevel", 1),
        "description": _strip_html(item.get("description")),
        "tags": item.get("tags") or [],
        "trigger_words": version.get("trainedWords") or [],
        "recommended_weight": None,
        "recommended_cfg": image_meta.get("cfgScale"),
        "recommended_steps": image_meta.get("steps"),
        "recommended_sampler": image_meta.get("sampler"),
        "download_url": version.get("downloadUrl"),
        "civitai_url": _civitai_url(model_id, version_id),
        "stats_downloads": stats.get("downloadCount", 0),
        "stats_thumbs_up": stats.get("thumbsUpCount", 0),
        "stats_thumbs_down": stats.get("thumbsDownCount", 0),
        "preview_image_url": preview_url,
    }


async def _upsert_batch(pool: asyncpg.Pool, records: list[dict]) -> int:
    if not records:
        return 0

    sql = """
        INSERT INTO models (
            source, civitai_model_id, civitai_version_id,
            name, version_name, type, base_model, nsfw_level,
            description, tags, trigger_words,
            recommended_weight, recommended_cfg, recommended_steps, recommended_sampler,
            download_url, civitai_url,
            stats_downloads, stats_thumbs_up, stats_thumbs_down,
            preview_image_url,
            date_cached, date_updated
        ) VALUES (
            $1,  $2,  $3,
            $4,  $5,  $6,  $7,  $8,
            $9,  $10, $11,
            $12, $13, $14, $15,
            $16, $17,
            $18, $19, $20,
            $21,
            NOW(), NOW()
        )
        ON CONFLICT (civitai_version_id) DO UPDATE SET
            name                = EXCLUDED.name,
            version_name        = EXCLUDED.version_name,
            type                = EXCLUDED.type,
            base_model          = EXCLUDED.base_model,
            nsfw_level          = EXCLUDED.nsfw_level,
            description         = EXCLUDED.description,
            tags                = EXCLUDED.tags,
            trigger_words       = EXCLUDED.trigger_words,
            recommended_cfg     = EXCLUDED.recommended_cfg,
            recommended_steps   = EXCLUDED.recommended_steps,
            recommended_sampler = EXCLUDED.recommended_sampler,
            download_url        = EXCLUDED.download_url,
            civitai_url         = EXCLUDED.civitai_url,
            stats_downloads     = EXCLUDED.stats_downloads,
            stats_thumbs_up     = EXCLUDED.stats_thumbs_up,
            stats_thumbs_down   = EXCLUDED.stats_thumbs_down,
            preview_image_url   = EXCLUDED.preview_image_url,
            date_updated        = NOW()
    """

    rows = [
        (
            r["source"], r["civitai_model_id"], r["civitai_version_id"],
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
) -> dict:
    headers = {"Content-Type": "application/json"}
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
        return resp.json()


async def full_crawl(base_model: str, pool: asyncpg.Pool) -> int:
    """Page through all Civitai results for a base model and cache them."""
    token = os.environ.get("CIVITAI_API_TOKEN")
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

    initial_params = {
        "limit": PAGE_SIZE,
        "types": CRAWL_TYPES,
        "baseModels": base_model,
        "sort": "Most Downloaded",
    }
    base_url = f"{CIVITAI_API_BASE}/models"
    next_url: str | None = None

    async with httpx.AsyncClient() as client:
        page = 1
        while True:
            log.info("Page %d — '%s' …", page, base_model)
            if next_url:
                data = await _fetch_page(client, next_url, token)
            else:
                data = await _fetch_page(client, base_url, token, params=initial_params)

            records = [
                r
                for item in data.get("items", [])
                if (r := _extract_record(item)) is not None
            ]
            inserted = await _upsert_batch(pool, records)
            total += inserted
            log.info("  → %d records upserted (running total: %d)", inserted, total)

            next_url = data.get("metadata", {}).get("nextPage")
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

    log.info("Full crawl complete for '%s': %d models cached.", base_model, total)
    return total


async def incremental_update(base_model: str, pool: asyncpg.Pool) -> int:
    """Fetch only models newer than the last crawl, stopping when known IDs reappear."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT last_crawled FROM base_model_index WHERE base_model_name = $1",
            base_model,
        )

    if not row or not row["last_crawled"]:
        log.info("No prior crawl found for '%s' — falling back to full crawl.", base_model)
        return await full_crawl(base_model, pool)

    async with pool.acquire() as conn:
        existing_ids: set[int] = {
            r["civitai_version_id"]
            for r in await conn.fetch(
                "SELECT civitai_version_id FROM models WHERE base_model = $1",
                base_model,
            )
        }

    log.info(
        "Incremental update for '%s' — %d models already cached.",
        base_model, len(existing_ids),
    )

    token = os.environ.get("CIVITAI_API_TOKEN")
    total = 0

    initial_params = {
        "limit": PAGE_SIZE,
        "types": CRAWL_TYPES,
        "baseModels": base_model,
        "sort": "Newest",
    }
    base_url = f"{CIVITAI_API_BASE}/models"
    next_url: str | None = None

    async with httpx.AsyncClient() as client:
        page = 1
        stop = False
        while not stop:
            log.info("Incremental page %d …", page)
            if next_url:
                data = await _fetch_page(client, next_url, token)
            else:
                data = await _fetch_page(client, base_url, token, params=initial_params)

            records: list[dict] = []
            for item in data.get("items", []):
                r = _extract_record(item)
                if r is None:
                    continue
                if r["civitai_version_id"] in existing_ids:
                    log.info("Hit known model on page %d — stopping.", page)
                    stop = True
                    break
                records.append(r)

            next_url = data.get("metadata", {}).get("nextPage")
            if not next_url:
                stop = True

            if records:
                inserted = await _upsert_batch(pool, records)
                total += inserted
                log.info("  → %d new records", inserted)

            url = data.get("metadata", {}).get("nextPage")
            page += 1

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE base_model_index
            SET last_crawled = NOW(),
                total_models = (SELECT COUNT(*) FROM models WHERE base_model = $1)
            WHERE base_model_name = $1
            """,
            base_model,
        )

    log.info("Incremental update complete for '%s': %d new models.", base_model, total)
    return total


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Civitai model catalog crawler.")
    parser.add_argument(
        "--base-model", required=True,
        help="Base model to crawl, e.g. 'Flux.1 D'",
    )
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
