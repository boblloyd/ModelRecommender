#!/usr/bin/env python3
"""
Model Recommendation CLI.

Usage:
  python cli.py "two characters swordfighting in the rain"
  python cli.py "your prompt" --base-model "SDXL 1.0" --json
  python cli.py "your prompt" --no-llm        # raw tag-scored results, no LLM
  python cli.py --crawl "Flux.1 D"
  python cli.py --crawl "Flux.1 D" --mode incremental
  python cli.py --import-tensorart tensor_art_export_1234.json
  python cli.py --status
"""

import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# TensorArt import
# ---------------------------------------------------------------------------

async def _run_import_tensorart(export_path: str) -> None:
    from crawler.tensorart_crawler import ingest_from_export
    from db.database import get_pool, init_schema

    pool = await get_pool()
    await init_schema(pool)
    try:
        count = await ingest_from_export(export_path, pool)
    finally:
        await pool.close()

    print(f"\nDone. {count} TensorArt models imported into catalog.")


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

async def _run_query(prompt: str, base_model: str, as_json: bool, llm_reasoning: bool = True) -> None:
    from agents.catalog_query import ensure_base_model_cached, query_catalog
    from agents.compatibility_analyst import analyze_compatibility
    from agents.intent_parser import parse_intent
    from db.database import get_pool

    pool = await get_pool()
    try:
        cached = await ensure_base_model_cached(base_model, pool)
        if not cached:
            print(f"ERROR: '{base_model}' is not in the cache yet.")
            print(f"  Run: python cli.py --crawl \"{base_model}\"")
            sys.exit(1)

        intent = await parse_intent(prompt)

        results = await query_catalog(
            search_tags=intent.tags,
            pool=pool,
            base_model=base_model,
        )

        phase = "2a — LLM intent parsed" if intent.llm_used else "1 — stop-word fallback"

        if llm_reasoning and intent.llm_used:
            results = await analyze_compatibility(
                prompt=prompt,
                style=intent.style,
                subject=intent.subject,
                results=results,
            )
            if results.get("recommended_combination"):
                phase = "2b — LLM intent + compatibility analysis"
    finally:
        await pool.close()

    if as_json:
        print(json.dumps({
            "prompt": prompt,
            "base_model": base_model,
            "intent": {
                "tags": intent.tags,
                "style": intent.style,
                "subject": intent.subject,
                "llm_used": intent.llm_used,
            },
            "phase": phase,
            **results,
        }, indent=2, default=str))
        return

    _print_results(prompt, intent, phase, results)


def _print_results(prompt: str, intent, phase: str, results: dict) -> None:
    W = 65
    SEP = "═" * W

    print(f"\n{SEP}")
    print(f'  RECOMMENDATIONS FOR: "{prompt}"')
    print(f"  Phase: {phase}")
    if intent.tags:
        print(f"  Intent tags: {', '.join(intent.tags[:12])}")
    print(SEP)

    combo = results.get("recommended_combination")
    combo_notes = results.get("combination_notes")
    if combo:
        print(f"\nRECOMMENDED COMBINATION")
        print(f"  {combo}")
        if combo_notes:
            print(f"  {combo_notes}")

    additions = results.get("prompt_additions") or []
    if additions:
        print(f"\nADD TO YOUR PROMPT")
        print(f"  {', '.join(additions)}")

    checkpoints = results.get("checkpoints", [])
    print(f"\nCHECKPOINTS ({len(checkpoints)} results)\n")
    if not checkpoints:
        print("  None found — try a broader prompt or crawl more base models.")
    for i, m in enumerate(checkpoints, 1):
        _print_model(i, m)

    loras = results.get("loras", [])
    print(f"\nLoRAs ({len(loras)} results)\n")
    if not loras:
        print("  None found.")
    for i, m in enumerate(loras, 1):
        triggers = ", ".join(m.get("trigger_words") or []) or "—"
        _print_model(i, m, extra=f"  Triggers: {triggers}")

    print(f"\n{SEP}\n")


def _print_model(rank: int, m: dict, extra: str = "") -> None:
    score = m.get("relevance_score", 0)
    impact = m.get("impact")
    recommended = m.get("recommended", True)

    impact_tag = f"  [impact: {impact}]" if impact else ""
    not_rec_tag = "  [not recommended]" if not recommended else ""
    print(f"  #{rank}  {m['name']}{impact_tag}{not_rec_tag}  (score: {score:.3f})")

    if m.get("civitai_url"):
        print(f"       URL   : {m['civitai_url']}")

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
        print(f"       {' | '.join(parts)}")

    note = m.get("compatibility_note")
    if note:
        print(f"       Note  : {note}")

    weight = m.get("recommended_weight")
    if weight is not None:
        print(f"       Weight: {weight}")

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
    parser.add_argument("--import-tensorart", metavar="FILE",
                        help="Import a TensorArt TamperMonkey export JSON file into the catalog")
    parser.add_argument("--status", action="store_true",
                        help="Show cache status for all indexed base models")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Output raw JSON")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM compatibility analysis; return raw tag-scored results (faster)")
    args = parser.parse_args()

    if args.crawl:
        asyncio.run(_run_crawl(args.crawl, args.mode))
    elif args.import_tensorart:
        asyncio.run(_run_import_tensorart(args.import_tensorart))
    elif args.status:
        asyncio.run(_run_status())
    elif args.prompt:
        asyncio.run(_run_query(args.prompt, args.base_model, args.as_json, not args.no_llm))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
