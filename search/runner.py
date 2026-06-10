from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .browser import Screenshotter
from .chunking import chunk_dom_text, chunk_text, chunk_viewports
from .cli import (
    DEEP_BACKEND_CHOICES,
    append_unique_methods,
    capture_search_pages,
    make_run_dir,
    make_seed_results,
    process_result_pages,
    resolve_deep_backend,
    should_run_link_expansion,
)
from .expansion import expand_results_with_links
from .models import SearchPageScreenshot, SearchResult, TextChunk
from .providers import collect_results
from .reporting import write_chunks_progress, write_json, write_report
from .tavily_collection import collect_tavily_pages
from .utils import DEFAULT_METHODS, SCREENSHOT_MODES, SUPPORTED_METHODS, canonical_url_key, load_default_env_files


@dataclass
class SearchRunOptions:
    methods: list[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    max_results: int = 20
    start_urls: list[str] = field(default_factory=list)
    exclude_url_keys: set[str] = field(default_factory=set)
    deep_backend: str = "none"
    deep_title_prefix: str = "Seed page"
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs")
    tavily_tools: list[str] | None = None
    tavily_search_depth: str = "basic"
    tavily_extract_depth: str = "basic"
    tavily_format: str = "markdown"
    tavily_seed_urls: list[str] = field(default_factory=list)
    tavily_seed_results: int = 3
    tavily_map_limit: int = 10
    tavily_crawl_limit: int = 5
    tavily_max_depth: int = 1
    tavily_max_breadth: int = 10
    tavily_extract_limit: int = 10
    content: str = "result_pages"
    content_max_chars: int = 12000
    screenshots: str = "both"
    full_page: bool = False
    browser_disable_proxy: bool = True
    collect_chunks: bool = True
    chunk_mode: str = "text"
    chunk_size: int = 4000
    chunk_overlap: int = 400
    viewport_max_scrolls: int = 3
    viewport_overlap: float = 0.5
    viewport_text_max_chars: int = 4000
    viewport_text_overlap_chars: int = 400
    viewport_height: int = 1440
    evidence_screenshots: bool = True
    max_evidence_shots: int = 20
    link_scope: str = "same_domain"
    link_allowlist: list[str] = field(default_factory=list)
    max_link_depth: int = 1
    max_linked_pages_per_result: int = 3
    max_candidate_links: int = 30
    link_filter: str = "llm"


@dataclass
class SearchRunResult:
    query: str
    run_dir: Path
    methods: list[str]
    results: list[SearchResult]
    chunks: list[TextChunk]
    search_page_screenshots: list[SearchPageScreenshot]
    screenshots_mode: str
    deep_backend: str
    errors: list[dict[str, Any]]
    json_path: Path
    report_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "run_dir": str(self.run_dir),
            "methods": self.methods,
            "results": [asdict(item) for item in self.results],
            "chunks": [asdict(item) for item in self.chunks],
            "search_page_screenshots": [asdict(item) for item in self.search_page_screenshots],
            "screenshots_mode": self.screenshots_mode,
            "deep_backend": self.deep_backend,
            "errors": self.errors,
            "json_path": str(self.json_path),
            "report_path": str(self.report_path),
            "metadata": self.metadata,
        }


def run_search(query: str, options: SearchRunOptions | None = None) -> SearchRunResult:
    """Run search collection logic without invoking the search package LLM extraction prompts."""
    load_default_env_files()
    options = options or SearchRunOptions()
    _validate_options(query, options)

    run_dir = make_run_dir(Path(options.output_dir))
    methods = _dedupe_methods(options.methods)
    start_urls = _dedupe_public_urls(options.start_urls)
    has_start_urls = bool(start_urls)
    deep_backend = options.deep_backend
    if has_start_urls and deep_backend == "none":
        deep_backend = "local_relevant_links"
    local_follow_mode, use_tavily_deep = resolve_deep_backend(deep_backend)

    results: list[SearchResult] = []
    skipped_duplicates: list[dict[str, Any]] = []
    if has_start_urls:
        results.extend(make_seed_results(query, start_urls, options.deep_title_prefix))

    search_methods = methods if (not has_start_urls or methods) else []
    if search_methods:
        results.extend(
            collect_results(
                query,
                search_methods,
                options.max_results,
                tavily_search_depth=options.tavily_search_depth,
            )
        )
    results = _filter_unique_results(results, options.exclude_url_keys, skipped_duplicates, stage="initial_results")

    report_methods = list(search_methods)
    if has_start_urls:
        report_methods.insert(0, "seed_url")

    tavily_methods = list(search_methods)
    if use_tavily_deep:
        append_unique_methods(tavily_methods, ["tavily"])

    errors: list[dict[str, Any]] = []
    tavily_payload: dict[str, Any] = {"enabled": False, "tools": options.tavily_tools or ["search"], "errors": []}

    def run_tavily_collection() -> None:
        nonlocal tavily_payload
        if not use_tavily_deep:
            return
        tavily_tools = options.tavily_tools or ["map", "crawl", "extract"]
        tavily_payload = collect_tavily_pages(
            results,
            run_dir,
            query=query,
            methods=tavily_methods,
            tools=tavily_tools,
            content_max_chars=options.content_max_chars,
            seed_urls=list(options.tavily_seed_urls) + (start_urls if has_start_urls else []),
            seed_results=options.tavily_seed_results,
            map_limit=options.tavily_map_limit,
            crawl_limit=options.tavily_crawl_limit,
            max_depth=options.tavily_max_depth,
            max_breadth=options.tavily_max_breadth,
            extract_limit=options.tavily_extract_limit,
            extract_depth=options.tavily_extract_depth,
            tavily_format=options.tavily_format,
        )

    tavily_before_expansion = use_tavily_deep and not should_run_link_expansion(local_follow_mode)
    if tavily_before_expansion:
        run_tavily_collection()
        results = _filter_unique_results(results, options.exclude_url_keys, skipped_duplicates, stage="tavily_before_expansion")

    if should_run_link_expansion(local_follow_mode):
        expanded_results, link_payload = expand_results_with_links(
            results,
            run_dir,
            query=query,
            follow_pages=local_follow_mode,
            link_scope=options.link_scope,
            link_allowlist=options.link_allowlist,
            max_link_depth=options.max_link_depth,
            max_linked_pages_per_result=options.max_linked_pages_per_result,
            max_candidate_links=options.max_candidate_links,
            link_filter=options.link_filter,
            viewport_height=options.viewport_height,
        )
        if expanded_results:
            results.extend(expanded_results)
        errors.extend(link_payload.get("errors") or [])
        results = _filter_unique_results(results, options.exclude_url_keys, skipped_duplicates, stage="link_expansion")

    if use_tavily_deep and not tavily_before_expansion:
        run_tavily_collection()
        results = _filter_unique_results(results, options.exclude_url_keys, skipped_duplicates, stage="tavily_after_expansion")
    errors.extend(tavily_payload.get("errors") or [])
    results = _filter_unique_results(results, options.exclude_url_keys, skipped_duplicates, stage="before_processing")

    search_page_screenshots: list[SearchPageScreenshot] = []
    needs_result_page_processing = options.screenshots in {"result_pages", "both"} or options.content == "result_pages"
    if options.screenshots != "none" or options.content != "none":
        with Screenshotter(run_dir, options.full_page, disable_proxy=options.browser_disable_proxy) as screenshotter:
            if options.screenshots in {"search_page", "both"}:
                search_page_screenshots = capture_search_pages(query, report_methods, screenshotter)
            if needs_result_page_processing:
                process_result_pages(
                    results,
                    screenshotter,
                    capture_screenshots=options.screenshots in {"result_pages", "both"},
                    extract_content=options.content == "result_pages",
                    content_max_chars=options.content_max_chars,
                )

    chunks: list[TextChunk] = []
    if options.collect_chunks:
        chunking_errors: list[dict[str, Any]] = []
        if options.chunk_mode == "viewport":
            chunks, chunking_errors = chunk_viewports(
                results,
                run_dir,
                max_scrolls=options.viewport_max_scrolls,
                overlap_ratio=options.viewport_overlap,
                text_max_chars=options.viewport_text_max_chars,
                text_overlap_chars=options.viewport_text_overlap_chars,
                viewport_height=options.viewport_height,
                capture_screenshots=options.evidence_screenshots,
                max_screenshots=options.max_evidence_shots,
            )
        else:
            file_chunks = chunk_text(
                [
                    result
                    for result in results
                    if result.content_source in {"tavily_extract", "tavily_crawl"} and result.content_path
                ],
                run_dir,
                chunk_size=options.chunk_size,
                chunk_overlap=options.chunk_overlap,
            )
            dom_results = [
                result
                for result in results
                if not (result.content_source in {"tavily_extract", "tavily_crawl"} and result.content_path)
            ]
            dom_chunks, chunking_errors = chunk_dom_text(
                dom_results,
                run_dir,
                chunk_size=options.chunk_size,
                chunk_overlap=options.chunk_overlap,
                viewport_height=options.viewport_height,
                capture_screenshots=options.evidence_screenshots,
                start_sequence=len(file_chunks) + 1,
            )
            chunks = file_chunks + dom_chunks
        errors.extend(chunking_errors)
        write_chunks_progress(run_dir, chunks)

    json_path = write_json(
        run_dir=run_dir,
        query=query,
        methods=report_methods,
        max_results=options.max_results,
        screenshots_mode=options.screenshots,
        full_page=options.full_page,
        content_mode=options.content,
        content_max_chars=options.content_max_chars,
        search_page_screenshots=search_page_screenshots,
        results=results,
        extra={
            "start_urls": start_urls,
            "deep_backend": deep_backend,
            "seed_url_count": len(start_urls),
            "link_scope": options.link_scope,
            "max_link_depth": options.max_link_depth,
            "max_linked_pages_per_result": options.max_linked_pages_per_result,
            "max_candidate_links": options.max_candidate_links,
            "link_filter": options.link_filter,
            "tavily_collection": tavily_payload,
            "chunk_count": len(chunks),
            "runner_mode": "collection_only",
            "browser_disable_proxy": options.browser_disable_proxy,
            "skipped_duplicates": skipped_duplicates,
            "skipped_duplicate_count": len(skipped_duplicates),
            "seen_url_keys": sorted(key for key in _result_url_keys(results) if key),
        },
    )
    report_path = write_report(
        run_dir=run_dir,
        query=query,
        methods=report_methods,
        search_page_screenshots=search_page_screenshots,
        results=results,
        extraction=None,
    )
    return SearchRunResult(
        query=query,
        run_dir=run_dir,
        methods=report_methods,
        results=results,
        chunks=chunks,
        search_page_screenshots=search_page_screenshots,
        screenshots_mode=options.screenshots,
        deep_backend=deep_backend,
        errors=errors,
        json_path=json_path,
        report_path=report_path,
        metadata={
            "tavily_collection": tavily_payload,
            "runner_mode": "collection_only",
            "browser_disable_proxy": options.browser_disable_proxy,
            "skipped_duplicates": skipped_duplicates,
            "skipped_duplicate_count": len(skipped_duplicates),
            "seen_url_keys": sorted(key for key in _result_url_keys(results) if key),
        },
    )


def _validate_options(query: str, options: SearchRunOptions) -> None:
    if not query.strip():
        raise ValueError("query is required")
    if options.max_results < 1:
        raise ValueError("max_results must be at least 1")
    if options.deep_backend not in DEEP_BACKEND_CHOICES:
        raise ValueError(f"Unsupported deep_backend: {options.deep_backend}")
    if options.screenshots not in SCREENSHOT_MODES:
        raise ValueError(f"Unsupported screenshots mode: {options.screenshots}")
    if options.content not in {"none", "result_pages"}:
        raise ValueError(f"Unsupported content mode: {options.content}")
    if options.chunk_mode not in {"text", "viewport"}:
        raise ValueError(f"Unsupported chunk_mode: {options.chunk_mode}")
    if options.chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if options.chunk_overlap < 0 or options.chunk_overlap >= options.chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    unknown = [method for method in options.methods if method not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported search method(s): {', '.join(unknown)}")


def _dedupe_methods(methods: list[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for method in methods or DEFAULT_METHODS:
        if method in SUPPORTED_METHODS and method not in seen:
            seen.add(method)
            items.append(method)
    return items or list(DEFAULT_METHODS)


def _dedupe_public_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for url in urls:
        cleaned = str(url or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            items.append(cleaned)
    return items


def _result_url_key(result: SearchResult) -> str:
    return canonical_url_key(result.final_url or result.url)


def _result_url_keys(results: list[SearchResult]) -> set[str]:
    return {_result_url_key(result) for result in results if _result_url_key(result)}


def _filter_unique_results(
    results: list[SearchResult],
    exclude_url_keys: set[str],
    skipped_duplicates: list[dict[str, Any]],
    *,
    stage: str,
) -> list[SearchResult]:
    seen = set(exclude_url_keys)
    filtered: list[SearchResult] = []
    for result in results:
        key = _result_url_key(result)
        if not key or result.error:
            filtered.append(result)
            continue
        if key in seen:
            skipped_duplicates.append(
                {
                    "stage": stage,
                    "method": result.method,
                    "query": result.query,
                    "rank": result.rank,
                    "url": result.url,
                    "final_url": result.final_url,
                    "title": result.title,
                    "url_key": key,
                }
            )
            continue
        seen.add(key)
        filtered.append(result)
    return filtered
