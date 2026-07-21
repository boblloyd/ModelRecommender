"""
TensorArt model catalog importer.

Reads the JSON export produced by the tensor_art_capture.user.js TamperMonkey
script and upserts model records into the PostgreSQL catalog alongside CivitAI
and HuggingFace data.

Export format (from "Export JSON" button in the userscript):
    [{id: "123456", nuxt: [...flat Nuxt devalue array...]}, ...]

Usage:
  python -m crawler.tensorart_crawler --from-export tensor_art_export_1234.json
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import asyncpg

from db.database import get_pool, init_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

TA_URL_BASE = "https://tensor.art/models"

# Map TensorArt base_model strings → our canonical catalog names.
# Keys are lowercased; values match what CivitAI uses so existing
# base_model_index entries cover TensorArt models too.
_BASE_MODEL_MAP: dict[str, str] = {
    # Flux variants
    "flux.1-dev": "Flux.1 D",
    "flux.1 dev": "Flux.1 D",
    "flux1-dev": "Flux.1 D",
    "flux1 dev": "Flux.1 D",
    "flux.1-schnell": "Flux.1 S",
    "flux.1 schnell": "Flux.1 S",
    "flux1-schnell": "Flux.1 S",
    # SDXL variants
    "sdxl": "SDXL 1.0",
    "sdxl 1.0": "SDXL 1.0",
    "stable diffusion xl": "SDXL 1.0",
    "stable diffusion xl base 1.0": "SDXL 1.0",
    # SD 1.5 variants
    "sd1.5": "SD 1.5",
    "sd 1.5": "SD 1.5",
    "stable diffusion 1.5": "SD 1.5",
    "stable diffusion v1.5": "SD 1.5",
    # Wan variants
    "wan2.1": "Wan2.1",
    "wan 2.1": "Wan2.1",
    "wan2.2": "Wan2.2",
    "wan 2.2": "Wan2.2",
}


def _normalize_base_model(raw: str | None) -> str | None:
    """Map a TensorArt base model string to our canonical name, or None if unknown."""
    if not raw:
        return None
    return _BASE_MODEL_MAP.get(raw.lower().strip())


# ---------------------------------------------------------------------------
# Nuxt devalue parser (adapted from tensor_art_metadata.py)
# ---------------------------------------------------------------------------

def _resolve(raw: list, v, _depth: int = 0):
    """
    Recursively resolve a value from the Nuxt flat array.
    Integer values are treated as index references into raw[].
    Non-integer values (strings, lists, dicts) are returned as-is or
    recursively resolved when reached via an index.
    """
    if _depth > 12:
        return None
    if not isinstance(v, int):
        return v
    if v < 0 or v >= len(raw):
        return v
    item = raw[v]
    if isinstance(item, (str, bool, float, type(None))):
        return item
    if isinstance(item, int):
        return item
    if isinstance(item, list):
        return [_resolve(raw, x, _depth + 1) for x in item]
    if isinstance(item, dict):
        return {k: _resolve(raw, vi, _depth + 1) for k, vi in item.items()}
    return item


def _extract_nuxt(raw: list) -> dict:
    """
    Walk the Nuxt devalue flat array and extract the model fields we care about.
    Returns a normalised dict; empty strings / empty lists mean "not found".
    """
    out: dict = {
        "name": "",
        "description": "",
        "base_model": "",
        "trained_words": [],
        "cover_url": "",
        "nsfw_level": 0,
        "tags": [],
        "download_count": 0,
        "like_count": 0,
    }

    for i, v in enumerate(raw):
        if not isinstance(v, dict):
            continue
        keys = set(v.keys())

        # Model version object: has trigger words + baseModel
        if ("triggerWords" in keys or "trainedWords" in keys) and "baseModel" in keys:
            obj = _resolve(raw, i)
            if not isinstance(obj, dict):
                continue
            if not out["trained_words"]:
                tw = obj.get("triggerWords") or obj.get("trainedWords") or []
                if isinstance(tw, list):
                    out["trained_words"] = [t for t in tw if isinstance(t, str) and t.strip()]
                elif isinstance(tw, str) and tw.strip():
                    out["trained_words"] = [tw.strip()]
            if not out["base_model"]:
                bm = obj.get("baseModel") or obj.get("baseModelDisplayName") or ""
                if isinstance(bm, str):
                    out["base_model"] = bm
            nsfw = obj.get("nsfwLevel")
            if isinstance(nsfw, int) and nsfw:
                out["nsfw_level"] = nsfw

        # Project object: has relatedTags — this holds the real model name and stats
        if "relatedTags" in keys:
            obj = _resolve(raw, i)
            if not isinstance(obj, dict):
                continue
            if not out["name"]:
                n = obj.get("name") or ""
                if isinstance(n, str):
                    out["name"] = n
            if not out["description"]:
                d = obj.get("description") or ""
                if isinstance(d, str):
                    out["description"] = re.sub(r"<[^>]+>", "", d).strip()
            stats = obj.get("statisticsInfo") or obj.get("statisticInfo") or {}
            if isinstance(stats, dict) and not out["download_count"]:
                out["download_count"] = int(stats.get("downloadCount") or 0)
                out["like_count"] = int(stats.get("likeCount") or 0)
            for tag_ref in (obj.get("relatedTags") or []):
                tag = tag_ref if isinstance(tag_ref, dict) else {}
                tag_name = tag.get("name") or ""
                if isinstance(tag_name, str) and tag_name and tag_name not in out["tags"]:
                    out["tags"].append(tag_name)

        # Cover image
        if keys & {"coverShowcases", "covers", "showcaseImageUrls", "cover", "showcaseImages"}:
            obj = _resolve(raw, i)
            if not isinstance(obj, dict):
                continue
            if not out["cover_url"]:
                for field in ("coverShowcases", "covers", "showcaseImageUrls", "showcaseImages", "cover"):
                    val = obj.get(field)
                    if isinstance(val, list) and val:
                        first = val[0]
                        url = first if isinstance(first, str) else (
                            first.get("url") or first.get("imageUrl", "")
                            if isinstance(first, dict) else ""
                        )
                        if url and "tensorartassets" in url:
                            out["cover_url"] = url
                            break
                    elif isinstance(val, str) and "tensorartassets" in val:
                        out["cover_url"] = val
                        break

        # Standalone tag objects (USER_GENERATED type)
        if "name" in keys and "type" in keys and len(keys) <= 4 and not (keys - {"id", "name", "type", "icon"}):
            type_val = raw[v["type"]] if isinstance(v["type"], int) else v["type"]
            name_val = raw[v["name"]] if isinstance(v["name"], int) else v["name"]
            if isinstance(name_val, str) and isinstance(type_val, str) and type_val == "USER_GENERATED":
                if name_val and name_val not in out["tags"]:
                    out["tags"].append(name_val)

    # Cover URL fallback: any CDN string on a model showcase path
    if not out["cover_url"]:
        for s in raw:
            if isinstance(s, str) and "tensorartassets.com" in s and "model_showcase" in s:
                if not re.search(r"\.(mp4|webm|gif)$", s, re.I):
                    out["cover_url"] = s
                    break

    return out


# ---------------------------------------------------------------------------
# Record extraction and mapping
# ---------------------------------------------------------------------------

def _map_to_record(model_id: str, info: dict) -> dict | None:
    """Map extracted TensorArt info dict to a catalog record dict."""
    if not info["name"] and not info["trained_words"]:
        return None

    canonical_base = _normalize_base_model(info["base_model"])
    # Fall back to the raw string if we don't recognise it — better than losing the model
    base_model = canonical_base or (info["base_model"] or None)

    return {
        "source": "tensorart",
        "tensorart_model_id": model_id,
        "civitai_model_id": None,
        "civitai_version_id": None,
        "hf_repo_id": None,
        "name": info["name"] or f"TensorArt #{model_id}",
        "version_name": None,
        "type": "LORA",
        "base_model": base_model,
        "nsfw_level": info["nsfw_level"] or 1,
        "description": info["description"] or None,
        "tags": info["tags"],
        "trigger_words": info["trained_words"],
        "recommended_weight": None,
        "recommended_cfg": None,
        "recommended_steps": None,
        "recommended_sampler": None,
        "download_url": f"{TA_URL_BASE}/{model_id}",
        "civitai_url": None,
        "stats_downloads": info["download_count"],
        "stats_thumbs_up": info["like_count"],
        "stats_thumbs_down": 0,
        "preview_image_url": info["cover_url"] or None,
    }


def _extract_record(entry: dict) -> dict | None:
    """Parse one export entry {id, nuxt} → catalog record dict, or None if unusable."""
    model_id = str(entry.get("id") or "").strip()
    nuxt = entry.get("nuxt")
    if not model_id or not isinstance(nuxt, list):
        return None
    info = _extract_nuxt(nuxt)
    return _map_to_record(model_id, info)


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------

async def _upsert_batch(pool: asyncpg.Pool, records: list[dict]) -> int:
    if not records:
        return 0

    sql = """
        INSERT INTO models (
            source, tensorart_model_id,
            civitai_model_id, civitai_version_id, hf_repo_id,
            name, version_name, type, base_model, nsfw_level,
            description, tags, trigger_words,
            recommended_weight, recommended_cfg, recommended_steps, recommended_sampler,
            download_url, civitai_url,
            stats_downloads, stats_thumbs_up, stats_thumbs_down,
            preview_image_url,
            date_cached, date_updated
        ) VALUES (
            $1,  $2,
            $3,  $4,  $5,
            $6,  $7,  $8,  $9,  $10,
            $11, $12, $13,
            $14, $15, $16, $17,
            $18, $19,
            $20, $21, $22,
            $23,
            NOW(), NOW()
        )
        ON CONFLICT (tensorart_model_id) WHERE tensorart_model_id IS NOT NULL DO UPDATE SET
            name                = EXCLUDED.name,
            type                = EXCLUDED.type,
            base_model          = EXCLUDED.base_model,
            nsfw_level          = EXCLUDED.nsfw_level,
            description         = EXCLUDED.description,
            tags                = EXCLUDED.tags,
            trigger_words       = EXCLUDED.trigger_words,
            download_url        = EXCLUDED.download_url,
            stats_downloads     = EXCLUDED.stats_downloads,
            stats_thumbs_up     = EXCLUDED.stats_thumbs_up,
            preview_image_url   = EXCLUDED.preview_image_url,
            date_updated        = NOW()
    """

    rows = [
        (
            r["source"], r["tensorart_model_id"],
            r["civitai_model_id"], r["civitai_version_id"], r["hf_repo_id"],
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_from_export_data(data: list[dict], pool: asyncpg.Pool) -> int:
    """
    Ingest a list of TamperMonkey export entries into the catalog.

    Each entry must be {id: str, nuxt: list}.
    Returns the count of models upserted.
    """
    records: list[dict] = []
    skipped = 0
    for entry in data:
        r = _extract_record(entry)
        if r is None:
            skipped += 1
            continue
        records.append(r)

    log.info(
        "Parsed %d usable records from %d entries (%d skipped/empty)",
        len(records), len(data), skipped,
    )
    count = await _upsert_batch(pool, records)
    log.info("Upserted %d TensorArt models into catalog.", count)
    return count


async def ingest_from_export(export_path: str, pool: asyncpg.Pool) -> int:
    """Read a TamperMonkey export JSON file and upsert all records into the catalog."""
    raw = Path(export_path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Export file must contain a JSON array, got {type(data).__name__}")
    return await ingest_from_export_data(data, pool)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(description="Import TensorArt models from TamperMonkey export.")
    parser.add_argument(
        "--from-export", required=True, metavar="FILE",
        help="JSON export file produced by tensor_art_capture.user.js",
    )
    args = parser.parse_args()

    pool = await get_pool()
    await init_schema(pool)
    try:
        count = await ingest_from_export(args.from_export, pool)
        print(f"\nDone. {count} TensorArt models imported.")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(_main())
