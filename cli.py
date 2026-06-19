#!/usr/bin/env python3
"""
Model Recommendation CLI — Phase 1 (raw scored results, no LLM).

Usage:
  python cli.py "two characters swordfighting in the rain"
  python cli.py "your prompt" --base-model "SDXL 1.0" --json
  python cli.py --crawl "Flux.1 D"
  python cli.py --crawl "Flux.1 D" --mode incremental
  python cli.py --status
"""

import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------

async def _run_crawl(base_model: str, mode: str) -> None:
    from crawler.civitai_crawler import full_crawl, incremental_update
    from db.database import get_pool, init_schema

    pool = await get_pool()
    await init_schema(pool)
    try:
        if mode == "full":
            count = await full_crawl(base_model, pool)
        else:
            count = await incremental_update(base_model, pool)
    finally:
        await pool.close()

    print(f"\nDone. {count} models cached for '{base_model}'.")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

async def _run_status() -> None:
    from db.database import get_pool

    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT base_model_name, last_crawled, total_models, crawl_complete
                FROM base_model_index ORDER BY base_model_name
                """
            )
            total = await conn.fetchval("SELECT COUNT(*) FROM models")
    finally:
        await pool.close()

    print(f"\nTotal models in catalog: {total}\n")
    if not rows:
        print("  No base models indexed yet.")
        print("  Run: python cli.py --crawl \"Flux.1 D\"")
        return

    for r in rows:
        status = "COMPLETE" if r["crawl_complete"] else "INCOMPLETE"
        crawled = r["last_crawled"].strftime("%Y-%m-%d %H:%M") if r["last_crawled"] else "never"
        print(f"  {r['base_model_name']:20s}  {status:10s}  {r['total_models']:6d} models  last: {crawled}")
    print()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

async def _run_query(prompt: str, base_model: str, as_json: bool) -> None:
    from agents.catalog_query import ensure_base_model_cached, query_catalog
    from db.database import get_pool

    pool = await get_pool()
    try:
        cached = await ensure_base_model_cached(base_model, pool)
        if not cached:
            print(f"ERROR: '{base_model}' is not in the cache yet.")
            print(f"  Run: python cli.py --crawl \"{base_model}\"")
            sys.exit(1)

        search_tags = [w.strip(".,!?;:\"'") for w in prompt.lower().split() if len(w) > 2]
        results = await query_catalog(
            search_tags=search_tags,
            pool=pool,
            base_model=base_model,
        )
    finally:
        await pool.close()

    if as_json:
        print(json.dumps(results, indent=2, default=str))
        return

    _print_results(prompt, search_tags, results)


def _print_results(prompt: str, search_tags: list[str], results: dict) -> None:
    W = 65
    SEP = "═" * W

    print(f"\n{SEP}")
    print(f'  RECOMMENDATIONS FOR: "{prompt}"')
    print(f"  Tags used: {', '.join(search_tags[:10])}")
    print(SEP)

    checkpoints = results.get("checkpoints", [])
    loras = results.get("loras", [])

    print(f"\nCHECKPOINTS ({len(checkpoints)} results)\n")
    if not checkpoints:
        print("  None found — try a broader prompt or crawl more base models.")
    for i, m in enumerate(checkpoints, 1):
        _print_model(i, m)

    print(f"\nLoRAs ({len(loras)} results)\n")
    if not loras:
        print("  None found.")
    for i, m in enumerate(loras, 1):
        triggers = ", ".join(m.get("trigger_words") or []) or "—"
        _print_model(i, m, extra=f"  Triggers: {triggers}")

    print(f"\n{SEP}")
    print("  NOTE: Phase 1 — raw tag-scored results. LLM reasoning in Phase 2.\n")


def _print_model(rank: int, m: dict, extra: str = "") -> None:
    score = m.get("relevance_score", 0)
    print(f"  #{rank}  {m['name']}  (score: {score:.3f})")
    if m.get("civitai_url"):
        print(f"       URL  : {m['civitai_url']}")
    cfg = m.get("recommended_cfg")
    steps = m.get("recommended_steps")
    sampler = m.get("recommended_sampler")
    if cfg or steps:
        parts = []
        if cfg:
            parts.append(f"CFG {cfg}")
        if steps:
            parts.append(f"Steps {steps}")
        if sampler:
            parts.append(sampler)
        print(f"       {'  |  '.join(parts)}")
    if extra:
        print(f"      {extra}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model Recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("prompt", nargs="?", help="Natural language image generation prompt")
    parser.add_argument("--crawl", metavar="BASE_MODEL",
                        help="Trigger a Civitai crawl for a base model")
    parser.add_argument("--mode", choices=["full", "incremental"], default="full",
                        help="Crawl mode — used with --crawl (default: full)")
    parser.add_argument("--base-model", default="Flux.1 D",
                        help="Base model filter for recommendations (default: Flux.1 D)")
    parser.add_argument("--status", action="store_true",
                        help="Show cache status for all indexed base models")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Output raw JSON")
    args = parser.parse_args()

    if args.crawl:
        asyncio.run(_run_crawl(args.crawl, args.mode))
    elif args.status:
        asyncio.run(_run_status())
    elif args.prompt:
        asyncio.run(_run_query(args.prompt, args.base_model, args.as_json))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
