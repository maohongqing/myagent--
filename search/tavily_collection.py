from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from .models import SearchResult
from .utils import canonical_url_key, clean_page_text, clean_text, filename_for_url, is_public_http_url, safe_error


TAVILY_TOOLS = {"search", "extract", "map", "crawl"}
TAVILY_CONTENT_SOURCES = {"tavily_extract", "tavily_crawl"}


def parse_tavily_tools(value: str) -> list[str]:
    tools = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not tools:
        raise argparse.ArgumentTypeError("At least one Tavily tool is required.")
    unknown = [tool for tool in tools if tool not in TAVILY_TOOLS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown Tavily tool(s): {', '.join(unknown)}. Supported: {', '.join(sorted(TAVILY_TOOLS))}"
        )
    deduped: list[str] = []
    for tool in tools:
        if tool not in deduped:
            deduped.append(tool)
    return deduped


def parse_seed_urls(value: str) -> list[str]:
    seeds: list[str] = []
    for item in value.split(","):
        url = item.strip()
        if not url or not is_public_http_url(url):
            continue
        if url not in seeds:
            seeds.append(url)
    return seeds


def normalize_url_key(url: str) -> str:
    return canonical_url_key(url)


def site_root_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    root = urlunparse((parsed.scheme, parsed.netloc, "/", "", "", ""))
    return root if is_public_http_url(root) else None


def tavily_client() -> tuple[Any | None, str | None]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None, "TAVILY_API_KEY is not set."
    try:
        from tavily import TavilyClient  # type: ignore
    except ImportError as exc:
        return None, f"tavily-python is not installed: {safe_error(exc)}"
    try:
        return TavilyClient(api_key=api_key), None
    except Exception as exc:
        return None, safe_error(exc)


def next_tavily_rank(results: list[SearchResult]) -> int:
    ranks = [result.rank for result in results if result.method == "tavily" and result.rank is not None]
    return max(ranks, default=0) + 1


def results_by_url(results: list[SearchResult]) -> dict[str, list[SearchResult]]:
    grouped: dict[str, list[SearchResult]] = {}
    for result in results:
        if result.url:
            grouped.setdefault(normalize_url_key(result.url), []).append(result)
    return grouped


def ensure_tavily_result(
    results: list[SearchResult],
    grouped: dict[str, list[SearchResult]],
    *,
    query: str,
    url: str,
    title: str,
    snippet: str = "",
    source_url: str | None = None,
    discovery_type: str | None = None,
    depth: int = 0,
) -> SearchResult:
    key = normalize_url_key(url)
    existing = grouped.get(key)
    if existing:
        result = existing[0]
        if discovery_type and not result.discovery_type:
            result.discovery_type = discovery_type
        if source_url and not result.source_url:
            result.source_url = source_url
        return result

    result = SearchResult(
        method="tavily",
        query=query,
        rank=next_tavily_rank(results),
        title=clean_text(title or url, max_chars=500),
        url=url,
        snippet=clean_text(snippet),
        source_url=source_url,
        discovery_type=discovery_type,
        depth=depth,
    )
    results.append(result)
    grouped.setdefault(key, []).append(result)
    return result


def content_from_item(item: dict[str, Any]) -> str:
    for key in ["raw_content", "content", "markdown", "text"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def url_from_item(item: Any) -> str:
    if isinstance(item, str):
        return clean_text(item, max_chars=2000)
    if isinstance(item, dict):
        return clean_text(item.get("url") or item.get("link") or item.get("href") or "", max_chars=2000)
    return ""


def title_from_item(item: Any, url: str, fallback: str) -> str:
    if isinstance(item, dict):
        return clean_text(item.get("title") or item.get("name") or url or fallback, max_chars=500)
    return clean_text(url or fallback, max_chars=500)


def response_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ["results", "urls", "links", "pages"]:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def response_failures(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    failures = payload.get("failed_results") or payload.get("failures") or payload.get("errors") or []
    return failures if isinstance(failures, list) else []


def write_tavily_content(
    run_dir: Path,
    *,
    url: str,
    prefix: str,
    text: str,
    max_chars: int,
) -> tuple[str | None, int, str | None]:
    cleaned = clean_page_text(text, max_chars=max_chars)
    if not cleaned:
        return None, 0, None
    content_dir = run_dir / "contents"
    content_dir.mkdir(parents=True, exist_ok=True)
    target = content_dir / filename_for_url(prefix, url, extension="txt")
    target.write_text(cleaned, encoding="utf-8")
    return target.relative_to(run_dir).as_posix(), len(cleaned), clean_text(cleaned, max_chars=700)


def apply_tavily_content(
    result: SearchResult,
    run_dir: Path,
    *,
    url: str,
    prefix: str,
    text: str,
    max_chars: int,
    source: str,
) -> bool:
    path, chars, preview = write_tavily_content(run_dir, url=url, prefix=prefix, text=text, max_chars=max_chars)
    if not path:
        return False
    result.content_path = path
    result.content_chars = chars
    result.content_preview = preview
    result.content_error = None
    result.content_source = source
    result.access_status = result.access_status or "success"
    return True


def tavily_seed_urls(results: list[SearchResult], explicit_seed_urls: list[str], seed_results: int) -> list[str]:
    seeds: list[str] = []
    for url in explicit_seed_urls:
        if url not in seeds:
            seeds.append(url)
    for result in results:
        if seed_results <= 0:
            break
        if result.rank is None or not result.url:
            continue
        root = site_root_url(result.url)
        if root and root not in seeds:
            seeds.append(root)
            seed_results -= 1
    return seeds


def collect_tavily_pages(
    results: list[SearchResult],
    run_dir: Path,
    *,
    query: str,
    methods: list[str],
    tools: list[str],
    content_max_chars: int,
    seed_urls: list[str],
    seed_results: int,
    map_limit: int,
    crawl_limit: int,
    max_depth: int,
    max_breadth: int,
    extract_limit: int,
    extract_depth: str,
    tavily_format: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": "tavily" in methods,
        "tools": tools,
        "extract_depth": extract_depth,
        "format": tavily_format,
        "seed_urls": [],
        "seed_results": seed_results,
        "map_limit": map_limit,
        "crawl_limit": crawl_limit,
        "max_depth": max_depth,
        "max_breadth": max_breadth,
        "extract_limit": extract_limit,
        "map_added": 0,
        "crawl_pages": 0,
        "crawl_content_success": 0,
        "extract_success": 0,
        "extract_failed": 0,
        "errors": [],
    }
    active_tools = [tool for tool in tools if tool in {"extract", "map", "crawl"}]
    if "tavily" not in methods or not active_tools:
        return payload

    client, client_error = tavily_client()
    if client_error or client is None:
        payload["errors"].append({"stage": "tavily_client", "error": client_error or "Tavily client unavailable."})
        return payload

    grouped = results_by_url(results)
    explicit_seed_urls = parse_seed_urls(",".join(seed_urls))
    for seed in explicit_seed_urls:
        ensure_tavily_result(
            results,
            grouped,
            query=query,
            url=seed,
            title=f"Tavily seed: {seed}",
            source_url=seed,
            discovery_type="tavily_seed",
        )

    seeds = tavily_seed_urls(results, explicit_seed_urls, seed_results)
    payload["seed_urls"] = seeds
    map_urls: list[str] = []

    if "map" in active_tools:
        for seed in seeds:
            try:
                response = client.map(
                    seed,
                    max_depth=max_depth,
                    max_breadth=max_breadth,
                    limit=map_limit,
                    timeout=150,
                    include_usage=True,
                )
            except Exception as exc:
                payload["errors"].append({"stage": "tavily_map", "url": seed, "error": safe_error(exc)})
                continue
            for item in response_items(response)[:map_limit]:
                url = url_from_item(item)
                if not url or not is_public_http_url(url):
                    continue
                map_urls.append(url)
                before = len(results)
                ensure_tavily_result(
                    results,
                    grouped,
                    query=query,
                    url=url,
                    title=title_from_item(item, url, "Tavily map result"),
                    source_url=seed,
                    discovery_type="tavily_map",
                    depth=max_depth,
                )
                if len(results) > before:
                    payload["map_added"] += 1

    if "crawl" in active_tools:
        for seed in seeds:
            try:
                response = client.crawl(
                    seed,
                    max_depth=max_depth,
                    max_breadth=max_breadth,
                    limit=crawl_limit,
                    extract_depth=extract_depth,
                    format=tavily_format,
                    timeout=150,
                    include_usage=True,
                )
            except Exception as exc:
                payload["errors"].append({"stage": "tavily_crawl", "url": seed, "error": safe_error(exc)})
                continue
            for index, item in enumerate(response_items(response)[:crawl_limit], start=1):
                url = url_from_item(item)
                if not url or not is_public_http_url(url):
                    continue
                result = ensure_tavily_result(
                    results,
                    grouped,
                    query=query,
                    url=url,
                    title=title_from_item(item, url, f"Tavily crawl result {index}"),
                    snippet=clean_text(item.get("content") if isinstance(item, dict) else "", max_chars=700),
                    source_url=seed,
                    discovery_type="tavily_crawl",
                    depth=max_depth,
                )
                payload["crawl_pages"] += 1
                if isinstance(item, dict):
                    text = content_from_item(item)
                    if text and apply_tavily_content(
                        result,
                        run_dir,
                        url=url,
                        prefix=f"tavily_crawl_{result.rank or index}",
                        text=text,
                        max_chars=content_max_chars,
                        source="tavily_crawl",
                    ):
                        payload["crawl_content_success"] += 1
            for failure in response_failures(response):
                payload["errors"].append({"stage": "tavily_crawl", "url": failure.get("url"), "error": failure.get("error")})

    if "extract" in active_tools:
        candidate_urls: list[str] = []
        for result in results:
            if result.rank is not None and result.url:
                candidate_urls.append(result.url)
        candidate_urls.extend(map_urls)
        candidate_urls.extend(explicit_seed_urls)
        deduped_candidates: list[str] = []
        seen: set[str] = set()
        for url in candidate_urls:
            if not is_public_http_url(url):
                continue
            key = normalize_url_key(url)
            if key in seen:
                continue
            seen.add(key)
            matching = grouped.get(key) or []
            if any(result.content_path for result in matching):
                continue
            deduped_candidates.append(url)
            if len(deduped_candidates) >= extract_limit:
                break

        if deduped_candidates:
            try:
                response = client.extract(
                    deduped_candidates,
                    extract_depth=extract_depth,
                    format=tavily_format,
                    timeout=30,
                    include_usage=True,
                    query=query,
                )
            except Exception as exc:
                payload["extract_failed"] += len(deduped_candidates)
                payload["errors"].append({"stage": "tavily_extract", "error": safe_error(exc), "urls": deduped_candidates})
            else:
                for index, item in enumerate(response_items(response), start=1):
                    if not isinstance(item, dict):
                        continue
                    url = url_from_item(item)
                    if not url or not is_public_http_url(url):
                        continue
                    result = ensure_tavily_result(
                        results,
                        grouped,
                        query=query,
                        url=url,
                        title=title_from_item(item, url, f"Tavily extract result {index}"),
                        source_url=url,
                        discovery_type="tavily_extract",
                    )
                    text = content_from_item(item)
                    if text and apply_tavily_content(
                        result,
                        run_dir,
                        url=url,
                        prefix=f"tavily_extract_{result.rank or index}",
                        text=text,
                        max_chars=content_max_chars,
                        source="tavily_extract",
                    ):
                        payload["extract_success"] += 1
                    else:
                        payload["extract_failed"] += 1
                        if not result.content_error:
                            result.content_error = "Tavily extract returned no content."
                for failure in response_failures(response):
                    payload["extract_failed"] += 1
                    payload["errors"].append(
                        {"stage": "tavily_extract", "url": failure.get("url"), "error": failure.get("error")}
                    )

    return payload
