from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import SearchPageScreenshot, SearchResult, TextChunk
from .utils import clean_text


def chunk_metadata(chunk: TextChunk) -> dict[str, Any]:
    payload = asdict(chunk)
    payload["text_preview"] = clean_text(chunk.text, max_chars=500)
    payload.pop("text", None)
    return payload

def write_chunks_progress(
    run_dir: Path,
    chunks: list[TextChunk],
    chunk_extractions: list[dict[str, Any]] | None = None,
) -> Path:
    extraction_dir = run_dir / "extractions"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = extraction_dir / "chunks.json"
    chunks_payload = {
        "chunks": [chunk_metadata(chunk) for chunk in chunks],
        "chunk_extractions": chunk_extractions or [],
    }
    chunks_path.write_text(json.dumps(chunks_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks_path

def write_extraction_progress(run_dir: Path, extraction: dict[str, Any]) -> Path:
    extraction_dir = run_dir / "extractions"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = extraction_dir / "extraction.json"
    extraction_path.write_text(json.dumps(extraction, ensure_ascii=False, indent=2), encoding="utf-8")
    return extraction_path

def write_extraction_files(
    run_dir: Path,
    chunks: list[TextChunk],
    chunk_extractions: list[dict[str, Any]],
    extraction: dict[str, Any],
) -> tuple[Path, Path]:
    chunks_path = write_chunks_progress(run_dir, chunks, chunk_extractions)
    extraction_path = write_extraction_progress(run_dir, extraction)
    return chunks_path, extraction_path

def write_quote_location_report(run_dir: Path, payload: dict[str, Any]) -> Path:
    locations = payload.get("locations") or []
    lines: list[str] = [
        "# Quote Location Report",
        "",
        f"- Created at: {payload.get('created_at')}",
        f"- Source extraction: {payload.get('source_extraction_path')}",
        f"- Items: {payload.get('item_count')}",
        f"- Matched: {payload.get('matched_count')}",
        f"- JSON: [quote_locations.json](quote_locations.json)",
        "",
        "## Results",
        "",
    ]
    if not locations:
        lines.append("No quote locations were processed.")
    for index, item in enumerate(locations, start=1):
        quote = markdown_escape(clean_text(item.get("quote") or "", max_chars=180))
        status = item.get("match_status") or "unknown"
        method = item.get("match_method") or "-"
        score = item.get("score")
        lines.append(f"### {index}. {status} ({method}, {score})")
        if item.get("url"):
            lines.append(f"- URL: {item.get('url')}")
        if item.get("fact_id"):
            lines.append(f"- Fact: {item.get('fact_id')}")
        if quote:
            lines.append(f"- Quote: {quote}")
        if item.get("dom_start") is not None:
            lines.append(f"- DOM range: {item.get('dom_start')}-{item.get('dom_end')}")
        if item.get("page_top") is not None:
            lines.append(f"- Page position: {item.get('page_top')}-{item.get('page_bottom')}, scroll_y={item.get('scroll_y')}")
        if item.get("screenshot_path"):
            lines.append(f"![quote screenshot]({item.get('screenshot_path')})")
        if item.get("error"):
            lines.append(f"- Error: {item.get('error')}")
        lines.append("")
    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def write_json(
    run_dir: Path,
    query: str,
    methods: list[str],
    max_results: int,
    screenshots_mode: str,
    full_page: bool,
    content_mode: str,
    content_max_chars: int,
    search_page_screenshots: list[SearchPageScreenshot],
    results: list[SearchResult],
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "query": query,
        "methods": methods,
        "max_results": max_results,
        "screenshots": screenshots_mode,
        "full_page": full_page,
        "content": content_mode,
        "content_max_chars": content_max_chars,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_dir": str(run_dir),
        "search_page_screenshots": [asdict(item) for item in search_page_screenshots],
        "results": [asdict(item) for item in results],
    }
    if extra:
        payload.update(extra)
    path = run_dir / "results.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def markdown_escape(text: str) -> str:
    return text.replace("[", "\\[").replace("]", "\\]")

def write_report(
    run_dir: Path,
    query: str,
    methods: list[str],
    search_page_screenshots: list[SearchPageScreenshot],
    results: list[SearchResult],
    extraction: dict[str, Any] | None = None,
) -> Path:
    lines: list[str] = [
        f"# Search Report: {query}",
        "",
        f"- Created at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Methods: {', '.join(methods)}",
        f"- Result rows: {len(results)}",
        "",
    ]

    if search_page_screenshots:
        lines.extend(["## Search Page Screenshots", ""])
        for shot in search_page_screenshots:
            lines.append(f"### {shot.method}")
            lines.append(f"- URL: {shot.url}")
            if shot.screenshot_path:
                lines.append(f"![{shot.method} search page]({shot.screenshot_path})")
            if shot.error:
                lines.append(f"- Error: {shot.error}")
            lines.append("")

    for method in methods:
        method_results = [result for result in results if result.method == method]
        lines.extend([f"## {method}", ""])
        if not method_results:
            lines.extend(["No results.", ""])
            continue
        for result in method_results:
            rank = result.rank if result.rank is not None else "-"
            title = markdown_escape(result.title or "Untitled")
            if result.url:
                lines.append(f"### {rank}. [{title}]({result.url})")
            else:
                lines.append(f"### {rank}. {title}")
            if result.snippet:
                lines.append(f"- Snippet: {result.snippet}")
            if result.access_status:
                status_details = result.access_status
                if result.http_status is not None:
                    status_details = f"{status_details} (HTTP {result.http_status})"
                lines.append(f"- Access: {status_details}")
            if result.final_url and result.final_url != result.url:
                lines.append(f"- Final URL: {result.final_url}")
            if result.block_reason:
                lines.append(f"- Block reason: {result.block_reason}")
            if result.robots_warning:
                lines.append(f"- Robots warning: {result.robots_warning}")
            if result.discovery_type:
                lines.append(f"- Discovery: {result.discovery_type}, depth={result.depth}")
            if result.source_url:
                lines.append(f"- Source URL: {result.source_url}")
            if result.link_relevance is not None:
                lines.append(f"- Link relevance: {result.link_relevance}")
            if result.link_reason:
                lines.append(f"- Link reason: {result.link_reason}")
            if result.content_path:
                lines.append(f"- Page text: [{result.content_path}]({result.content_path}) ({result.content_chars} chars)")
            if result.content_source:
                lines.append(f"- Content source: {result.content_source}")
            if result.content_preview:
                lines.append(f"> {result.content_preview}")
            if result.content_error:
                lines.append(f"- Content error: {result.content_error}")
            if result.error:
                lines.append(f"- Error: {result.error}")
            if result.screenshot_path:
                alt = markdown_escape(result.title or result.method)
                lines.append(f"![{alt}]({result.screenshot_path})")
            if result.screenshot_error:
                lines.append(f"- Screenshot error: {result.screenshot_error}")
            lines.append("")

    if extraction is not None:
        lines.extend(["## LLM Extraction", ""])
        extraction_path = "extractions/extraction.json"
        lines.append(f"- JSON: [{extraction_path}]({extraction_path})")
        if extraction.get("summary"):
            lines.append(f"- Summary: {extraction.get('summary')}")
        errors = extraction.get("errors") or []
        if errors:
            lines.append(f"- Errors: {len(errors)}")
            for error in errors[:3]:
                lines.append(f"  - {error.get('stage', 'error')}: {error.get('error')}")
        facts = sorted(extraction.get("facts") or [], key=lambda item: item.get("confidence") or 0, reverse=True)
        if facts:
            lines.extend(["", "### Top Facts", ""])
            for fact in facts[:8]:
                field = fact.get("field") or "other"
                claim = fact.get("claim") or ""
                confidence = fact.get("confidence")
                lines.append(f"- **{field}** ({confidence}): {claim}")
                if fact.get("evidence_source"):
                    lines.append(f"  - Source: {fact.get('evidence_source')}")
                if fact.get("evidence_quote"):
                    lines.append(f"  - Evidence: {fact.get('evidence_quote')}")
                if fact.get("evidence_shot_path"):
                    lines.append(f"  - Chunk shot: ![]({fact.get('evidence_shot_path')})")
                if fact.get("quote_shot_path"):
                    lines.append(f"  - Quote shot: ![]({fact.get('quote_shot_path')})")
                if fact.get("screenshot_mismatch_warning"):
                    lines.append(f"  - Screenshot warning: {fact.get('screenshot_mismatch_warning')}")
                if fact.get("screenshot_error"):
                    lines.append(f"  - Chunk screenshot error: {fact.get('screenshot_error')}")
                if fact.get("quote_shot_error"):
                    lines.append(f"  - Quote screenshot error: {fact.get('quote_shot_error')}")
        lines.append("")

    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
