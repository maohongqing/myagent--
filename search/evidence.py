from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .browser import Screenshotter
from .models import QuoteLocationItem, TextChunk
from .reporting import markdown_escape, write_quote_location_report
from .utils import clean_page_text, clean_text, is_public_http_url, progress, safe_error


def capture_chunk_shots(
    chunks: list[TextChunk],
    run_dir: Path,
    *,
    enabled: bool,
    max_shots: int,
) -> dict[str, dict[str, str | None]]:
    shot_map: dict[str, dict[str, str | None]] = {}
    if not enabled or max_shots <= 0:
        return shot_map
    with Screenshotter(run_dir, full_page=False) as screenshotter:
        for index, chunk in enumerate(chunks[:max_shots], start=1):
            path, error = screenshotter.capture_chunk_position(
                chunk.url,
                chunk.start_char,
                chunk.end_char,
                f"chunk_{index:04d}_{chunk.chunk_id}",
            )
            chunk.chunk_shot_path = path
            chunk.chunk_shot_error = error
            chunk.chunk_shot_paths = [path] if path else None
            shot_map[chunk.chunk_id] = {"path": path, "paths": chunk.chunk_shot_paths, "error": error}
    for chunk in chunks[max_shots:]:
        chunk.chunk_shot_error = "Chunk screenshot skipped: max evidence shot limit reached."
        chunk.chunk_shot_paths = None
        shot_map[chunk.chunk_id] = {"path": None, "paths": None, "error": chunk.chunk_shot_error}
    return shot_map

def chunk_shot_map_from_chunks(chunks: list[TextChunk]) -> dict[str, dict[str, Any]]:
    return {
        chunk.chunk_id: {
            "path": chunk.chunk_shot_path,
            "paths": chunk.chunk_shot_paths or ([chunk.chunk_shot_path] if chunk.chunk_shot_path else None),
            "error": chunk.chunk_shot_error,
        }
        for chunk in chunks
    }

def capture_quote_shots_for_extraction(
    extraction: dict[str, Any],
    run_dir: Path,
    *,
    min_score: float,
) -> dict[str, dict[str, str | None]]:
    facts = [fact for fact in extraction.get("facts") or [] if fact.get("id")]
    shot_map: dict[str, dict[str, str | None]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        quote = clean_page_text(fact.get("evidence_quote") or "", max_chars=1000)
        url = clean_text(fact.get("url") or "", max_chars=2000)
        fact_id = str(fact.get("id") or "")
        if not quote:
            shot_map[fact_id] = {"path": None, "error": "Evidence quote is empty."}
            continue
        if not is_public_http_url(url):
            shot_map[fact_id] = {"path": None, "error": "URL is not a public http/https URL."}
            continue
        grouped.setdefault(url, []).append(fact)

    with Screenshotter(run_dir, full_page=False) as screenshotter:
        for url, url_facts in grouped.items():
            progress(f"Quote shots page: {url} ({len(url_facts)} quote(s))")
            page, dom_nodes, page_error = screenshotter.load_quote_dom_page(url)
            if page_error or page is None:
                for fact in url_facts:
                    shot_map[str(fact.get("id") or "")] = {"path": None, "error": page_error or "Page could not be loaded."}
                continue
            try:
                for index, fact in enumerate(url_facts, start=1):
                    fact_id = str(fact.get("id") or "")
                    match, path, error = screenshotter.locate_quote_on_page(
                        page,
                        dom_nodes,
                        url,
                        clean_page_text(fact.get("evidence_quote") or "", max_chars=1000),
                        f"quote_{fact_id or index}",
                        min_score=min_score,
                    )
                    shot_map[fact_id] = {"path": path, "error": error if not path else None}
                    progress(
                        f"Quote shot {len(shot_map)}/{len(facts)} {fact_id}: "
                        f"{'matched' if match else 'failed'}"
                    )
            finally:
                try:
                    page.close()
                except Exception:
                    pass
    return shot_map

def apply_chunk_shots_to_extraction(
    extraction: dict[str, Any],
    chunk_shot_map: dict[str, dict[str, str | None]],
) -> None:
    for fact in extraction.get("facts") or []:
        shot = chunk_shot_map.get(str(fact.get("chunk_id") or ""))
        if not shot:
            fact["evidence_shot_path"] = None
            fact["chunk_shot_paths"] = None
            fact["screenshot_error"] = None
            continue
        fact["evidence_shot_path"] = shot.get("path")
        paths = shot.get("paths")
        fact["chunk_shot_paths"] = paths if isinstance(paths, list) else ([shot.get("path")] if shot.get("path") else None)
        fact["screenshot_error"] = shot.get("error")

    by_fact_id = {fact.get("id"): fact for fact in extraction.get("facts") or []}
    for evidence in extraction.get("evidence") or []:
        fact = by_fact_id.get(evidence.get("fact_id"))
        if fact:
            evidence["evidence_shot_path"] = fact.get("evidence_shot_path")
            evidence["chunk_shot_paths"] = fact.get("chunk_shot_paths")
            evidence["screenshot_error"] = fact.get("screenshot_error")

def apply_quote_shots_to_extraction(
    extraction: dict[str, Any],
    quote_shot_map: dict[str, dict[str, str | None]],
) -> None:
    for fact in extraction.get("facts") or []:
        shot = quote_shot_map.get(str(fact.get("id") or ""))
        fact["quote_shot_path"] = shot.get("path") if shot else None
        fact["quote_shot_error"] = shot.get("error") if shot else None

    by_fact_id = {fact.get("id"): fact for fact in extraction.get("facts") or []}
    for evidence in extraction.get("evidence") or []:
        fact = by_fact_id.get(evidence.get("fact_id"))
        if fact:
            evidence["quote_shot_path"] = fact.get("quote_shot_path")
            evidence["quote_shot_error"] = fact.get("quote_shot_error")

def quote_location_item_from_mapping(item: dict[str, Any], index: int) -> QuoteLocationItem:
    quote = clean_page_text(item.get("evidence_quote") or item.get("quote") or "", max_chars=1000)
    return QuoteLocationItem(
        source_id=clean_text(
            item.get("source_id") or item.get("source_result_id") or item.get("chunk_id") or "",
            max_chars=200,
        )
        or None,
        fact_id=clean_text(item.get("fact_id") or item.get("id") or f"quote_{index:04d}", max_chars=200) or None,
        url=clean_text(item.get("url") or "", max_chars=2000),
        quote=quote,
    )

def load_quote_location_items(extraction_path: Path, *, max_items: int) -> tuple[list[QuoteLocationItem], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    try:
        payload = json.loads(extraction_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [{"stage": "load_extraction", "error": safe_error(exc)}]
    if not isinstance(payload, dict):
        return [], [{"stage": "load_extraction", "error": "Extraction JSON root is not an object."}]

    raw_items = payload.get("evidence") or []
    if not raw_items:
        raw_items = payload.get("facts") or []
    if not isinstance(raw_items, list):
        return [], [{"stage": "load_extraction", "error": "Extraction evidence/facts is not a list."}]

    items: list[QuoteLocationItem] = []
    for index, raw in enumerate(raw_items, start=1):
        if len(items) >= max_items:
            break
        if not isinstance(raw, dict):
            errors.append({"stage": "load_item", "index": index, "error": "Extraction item is not an object."})
            continue
        item = quote_location_item_from_mapping(raw, index)
        items.append(item)
    return items, errors

def quote_location_result_base(item: QuoteLocationItem) -> dict[str, Any]:
    return {
        "source_id": item.source_id,
        "fact_id": item.fact_id,
        "url": item.url,
        "quote": item.quote,
        "match_status": "invalid_input",
        "match_method": None,
        "score": 0.0,
        "dom_start": None,
        "dom_end": None,
        "page_top": None,
        "page_bottom": None,
        "scroll_y": None,
        "screenshot_path": None,
        "error": None,
    }

def locate_quotes_from_extraction(
    extraction_path: Path,
    run_dir: Path,
    *,
    max_items: int,
    min_score: float,
) -> dict[str, Any]:
    items, load_errors = load_quote_location_items(extraction_path, max_items=max_items)
    results: list[dict[str, Any]] = []
    if load_errors:
        for error in load_errors:
            progress(f"Quote locator load warning: {error.get('error')}")

    grouped: dict[str, list[QuoteLocationItem]] = {}
    for item in items:
        result = quote_location_result_base(item)
        if not item.url:
            result["error"] = "URL is empty."
            results.append(result)
            continue
        if not item.quote:
            result["error"] = "Evidence quote is empty."
            results.append(result)
            continue
        if not is_public_http_url(item.url):
            result["error"] = "URL is not a public http/https URL."
            results.append(result)
            continue
        grouped.setdefault(item.url, []).append(item)

    with Screenshotter(run_dir, full_page=False) as screenshotter:
        for url, url_items in grouped.items():
            progress(f"Quote locator page: {url} ({len(url_items)} quote(s))")
            page, dom_nodes, page_error = screenshotter.load_quote_dom_page(url)
            if page_error or page is None:
                for item in url_items:
                    result = quote_location_result_base(item)
                    result["match_status"] = "page_error"
                    result["error"] = page_error or "Page could not be loaded."
                    results.append(result)
                    progress(f"Quote locator {len(results)}/{len(items)}: page_error score=0.0")
                continue
            try:
                for index, item in enumerate(url_items, start=1):
                    result = quote_location_result_base(item)
                    prefix = f"quote_{len(results) + 1:04d}_{item.fact_id or index}"
                    match, shot_path, error = screenshotter.locate_quote_on_page(
                        page,
                        dom_nodes,
                        item.url,
                        item.quote,
                        prefix,
                        min_score=min_score,
                    )
                    if match:
                        result.update(
                            {
                                "match_status": "matched",
                                "match_method": match.method,
                                "score": match.score,
                                "dom_start": match.dom_start,
                                "dom_end": match.dom_end,
                                "page_top": match.page_top,
                                "page_bottom": match.page_bottom,
                                "scroll_y": max(0, int((match.page_top or 0) - (screenshotter.viewport_height * 0.35)))
                                if match.page_top is not None
                                else None,
                                "screenshot_path": shot_path,
                                "error": error,
                            }
                        )
                    else:
                        result["match_status"] = "not_found"
                        result["error"] = error or "Evidence quote was not found in visible DOM text."
                    results.append(result)
                    progress(
                        f"Quote locator {len(results)}/{len(items)}: "
                        f"{result['match_status']} {result.get('match_method') or ''} score={result.get('score')}"
                    )
            finally:
                try:
                    page.close()
                except Exception:
                    pass

    payload = {
        "source_extraction_path": str(extraction_path),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_dir": str(run_dir),
        "max_items": max_items,
        "min_score": min_score,
        "item_count": len(items),
        "matched_count": sum(1 for item in results if item.get("match_status") == "matched"),
        "errors": load_errors,
        "locations": results,
    }
    (run_dir / "quote_locations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_quote_location_report(run_dir, payload)
    return payload
