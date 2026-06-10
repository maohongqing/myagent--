from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .browser import Screenshotter
from .chunking import chunk_dom_text, chunk_text, chunk_viewports
from .evidence import (
    apply_chunk_shots_to_extraction,
    apply_quote_shots_to_extraction,
    capture_quote_shots_for_extraction,
    chunk_shot_map_from_chunks,
    locate_quotes_from_extraction,
)
from .expansion import expand_results_with_links
from .llm import extract_chunks_with_llm, merge_extractions, run_image_understanding_for_chunks
from .models import PageData, SearchPageScreenshot, SearchResult
from .ocr import run_ocr_for_chunks
from .providers import collect_results, duckduckgo_search_url
from .reporting import write_chunks_progress, write_extraction_files, write_json, write_report
from .tavily_collection import collect_tavily_pages, parse_tavily_tools
from .utils import DEFAULT_METHODS, SCREENSHOT_MODES, is_public_http_url, load_default_env_files, parse_methods, progress


DEEP_BACKEND_CHOICES = [
    "none",
    "local_pagination",
    "local_relevant_links",
    "local_all_links",
    "tavily",
    "local_relevant_links_tavily",
]

LOCAL_FOLLOW_BY_DEEP_BACKEND = {
    "none": "none",
    "local_pagination": "pagination",
    "local_relevant_links": "relevant_links",
    "local_all_links": "both",
    "tavily": "none",
    "local_relevant_links_tavily": "relevant_links",
}
TAVILY_DEEP_BACKENDS = {"tavily", "local_relevant_links_tavily"}


def capture_search_pages(
    query: str,
    methods: list[str],
    screenshotter: Screenshotter,
) -> list[SearchPageScreenshot]:
    screenshots: list[SearchPageScreenshot] = []
    for method in methods:
        if method not in {"duckduckgo_text", "duckduckgo_news"}:
            continue
        url = duckduckgo_search_url(query, method)
        path, error = screenshotter.capture(url, f"search_page_{method}")
        screenshots.append(SearchPageScreenshot(method=method, url=url, screenshot_path=path, error=error))
    return screenshots

def apply_page_data(result: SearchResult, data: PageData) -> None:
    if data.content_path or not result.content_path:
        result.content_path = data.content_path
        result.content_chars = data.content_chars
        result.content_preview = data.content_preview
        result.content_error = data.content_error
        result.content_source = data.content_source
    elif data.content_error and not result.content_error:
        result.content_error = data.content_error
    result.screenshot_path = data.screenshot_path
    result.screenshot_error = data.screenshot_error
    result.access_status = data.access_status
    result.http_status = data.http_status
    result.final_url = data.final_url
    result.block_reason = data.block_reason
    result.robots_warning = data.robots_warning
    result.access_diagnostics = data.access_diagnostics

def process_result_pages(
    results: list[SearchResult],
    screenshotter: Screenshotter,
    *,
    capture_screenshots: bool,
    extract_content: bool,
    content_max_chars: int,
) -> None:
    cache: dict[tuple[str, bool], PageData] = {}
    for result in results:
        if result.rank is None or not result.url:
            continue
        needs_content = extract_content and not result.content_path
        cache_key = (result.url, needs_content)
        if cache_key not in cache:
            cache[cache_key] = screenshotter.process_result_page(
                result.url,
                f"result_{result.method}_{result.rank}",
                capture_screenshot=capture_screenshots,
                extract_content=needs_content,
                content_max_chars=content_max_chars,
            )
        apply_page_data(result, cache[cache_key])


def option_was_provided(argv: list[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def parse_start_urls(value: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for item in value.split(","):
        url = item.strip()
        if not url or not is_public_http_url(url) or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def make_seed_results(query: str, start_urls: list[str], title_prefix: str) -> list[SearchResult]:
    prefix = title_prefix.strip() or "Seed page"
    return [
        SearchResult(
            method="seed_url",
            query=query,
            rank=index,
            title=f"{prefix}: {url}",
            url=url,
            snippet="",
            source_url=url,
            discovery_type="seed_url",
            depth=0,
        )
        for index, url in enumerate(start_urls, start=1)
    ]


def normalize_tavily_tools(value: Any) -> list[str]:
    if isinstance(value, str):
        return parse_tavily_tools(value)
    return list(value)


def append_unique_methods(methods: list[str], additions: list[str]) -> list[str]:
    for method in additions:
        if method not in methods:
            methods.append(method)
    return methods


def resolve_deep_backend(deep_backend: str) -> tuple[str, bool]:
    return LOCAL_FOLLOW_BY_DEEP_BACKEND[deep_backend], deep_backend in TAVILY_DEEP_BACKENDS


def should_run_link_expansion(local_follow_mode: str) -> bool:
    return local_follow_mode != "none"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run web searches with multiple providers and save text results plus screenshots.",
    )
    parser.add_argument("--query", help="Search query, for example: \"飞书 竞品分析\"")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        type=parse_methods,
        help="Comma-separated methods: duckduckgo_text, duckduckgo_news, tavily, or all.",
    )
    parser.add_argument("--max-results", type=int, default=5, help="Maximum results per method.")
    parser.add_argument(
        "--start-urls",
        default="",
        help="Comma-separated public HTTP(S) URLs used as deep-search seed pages.",
    )
    parser.add_argument(
        "--deep-backend",
        choices=DEEP_BACKEND_CHOICES,
        default="none",
        help="Deep search mode: none, local_pagination, local_relevant_links, local_all_links, tavily, or local_relevant_links_tavily.",
    )
    parser.add_argument(
        "--deep-title-prefix",
        default="Seed page",
        help="Title prefix used for seed URL rows in reports.",
    )
    parser.add_argument(
        "--tavily-tools",
        default="search",
        type=parse_tavily_tools,
        help="Comma-separated Tavily tools: search, extract, map, crawl. Deep search defaults to map,crawl,extract unless explicitly set.",
    )
    parser.add_argument(
        "--tavily-search-depth",
        choices=["basic", "advanced", "fast", "ultra-fast", "auto"],
        default="basic",
        help="Tavily Search depth. auto enables Tavily auto_parameters.",
    )
    parser.add_argument(
        "--tavily-extract-depth",
        choices=["basic", "advanced"],
        default="basic",
        help="Tavily Extract/Crawl extraction depth.",
    )
    parser.add_argument(
        "--tavily-format",
        choices=["markdown", "text"],
        default="markdown",
        help="Tavily Extract/Crawl content format.",
    )
    parser.add_argument(
        "--tavily-seed-urls",
        default="",
        help="Comma-separated public seed URLs for Tavily Map/Crawl.",
    )
    parser.add_argument(
        "--tavily-seed-results",
        type=int,
        default=3,
        help="Number of current result site roots used as Tavily Map/Crawl seeds.",
    )
    parser.add_argument("--tavily-map-limit", type=int, default=10, help="Maximum Tavily Map URLs per seed.")
    parser.add_argument("--tavily-crawl-limit", type=int, default=5, help="Maximum Tavily Crawl pages per seed.")
    parser.add_argument("--tavily-max-depth", type=int, default=1, help="Tavily Map/Crawl maximum depth.")
    parser.add_argument("--tavily-max-breadth", type=int, default=10, help="Tavily Map/Crawl maximum breadth.")
    parser.add_argument("--tavily-extract-limit", type=int, default=10, help="Maximum URLs sent to Tavily Extract.")
    parser.add_argument(
        "--content",
        choices=["none", "result_pages"],
        default="result_pages",
        help="Extract visible page text from result pages, or disable with none.",
    )
    parser.add_argument(
        "--content-max-chars",
        type=int,
        default=12000,
        help="Maximum visible text characters saved per result page.",
    )
    parser.add_argument(
        "--screenshots",
        choices=sorted(SCREENSHOT_MODES),
        default="both",
        help="Screenshot mode: search_page, result_pages, both, or none.",
    )
    parser.add_argument("--full-page", action="store_true", help="Capture full-page screenshots.")
    parser.add_argument("--extract", action="store_true", help="Run LLM extraction over result page text chunks.")
    parser.add_argument(
        "--extract-schema",
        choices=["competitor"],
        default="competitor",
        help="Extraction schema. The first version supports competitor only.",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=["text", "viewport"],
        default="text",
        help="Extraction chunking mode: text splits visible DOM text by characters and captures chunk ranges; viewport uses scroll viewport chunks.",
    )
    parser.add_argument("--chunk-size", type=int, default=4000, help="Characters per extraction chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=400, help="Overlapping characters between chunks.")
    parser.add_argument(
        "--viewport-max-scrolls",
        type=int,
        default=3,
        help="Maximum downward scroll steps per result page for viewport chunks. Use 0 to continue until the page bottom.",
    )
    parser.add_argument(
        "--viewport-overlap",
        type=float,
        default=0.5,
        help="Viewport chunk screenshot overlap ratio from 0 to 0.95. Default 0.5 means half-page overlap.",
    )
    parser.add_argument(
        "--viewport-text-max-chars",
        type=int,
        default=4000,
        help="Maximum continuous DOM text characters included in each viewport chunk before splitting.",
    )
    parser.add_argument(
        "--viewport-text-overlap-chars",
        type=int,
        default=400,
        help="DOM text characters overlapped around adjacent viewport chunks.",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=1440,
        help="Browser viewport height used for viewport chunk screenshots. Default 1440 captures twice the previous 720px height.",
    )
    parser.add_argument(
        "--max-evidence-shots",
        type=int,
        default=20,
        help="Maximum chunk evidence screenshots in viewport mode. Text mode captures all chunk and quote shots when evidence screenshots are enabled.",
    )
    parser.add_argument(
        "--evidence-screenshots",
        dest="evidence_screenshots",
        action="store_true",
        default=True,
        help="Capture chunkshot and quoteshot evidence images. In text mode, chunkshots use DOM text boundaries.",
    )
    parser.add_argument(
        "--no-evidence-screenshots",
        dest="evidence_screenshots",
        action="store_false",
        help="Disable chunk-level evidence screenshots.",
    )
    parser.add_argument(
        "--ocr",
        dest="ocr",
        action="store_true",
        default=False,
        help="Enable OCR fallback over chunk screenshots.",
    )
    parser.add_argument("--no-ocr", dest="ocr", action="store_false", help="Disable OCR fallback.")
    parser.add_argument(
        "--ocr-engine",
        choices=["tesseract", "paddle"],
        default="tesseract",
        help="OCR engine to use: tesseract or paddle.",
    )
    parser.add_argument(
        "--ocr-lang",
        default=None,
        help="OCR language. Defaults to chi_sim+eng for tesseract, ch for paddle.",
    )
    parser.add_argument(
        "--ocr-min-extra-chars",
        type=int,
        default=80,
        help="Minimum OCR-only text characters required before OCR text is appended to a chunk.",
    )
    parser.add_argument(
        "--image-understanding",
        dest="image_understanding",
        action="store_true",
        default=False,
        help="Use an OpenAI-compatible vision model to summarize non-text visual information in chunk screenshots.",
    )
    parser.add_argument(
        "--no-image-understanding",
        dest="image_understanding",
        action="store_false",
        help="Disable vision-model image understanding.",
    )
    parser.add_argument(
        "--locate-extraction",
        type=Path,
        default=None,
        help="Locate evidence_quote items from an existing extraction.json and save quote_locations.json plus highlighted screenshots.",
    )
    parser.add_argument(
        "--locate-max-items",
        type=int,
        default=50,
        help="Maximum evidence/fact quote items to locate when using --locate-extraction.",
    )
    parser.add_argument(
        "--locate-min-score",
        type=float,
        default=0.72,
        help="Minimum fuzzy match score for quote location when exact matching fails.",
    )
    parser.add_argument(
        "--link-scope",
        choices=["same_domain", "allow_cross_domain", "allowlist"],
        default="same_domain",
        help="Limit followed links to same domain, allow all domains, or an explicit allowlist.",
    )
    parser.add_argument(
        "--link-allowlist",
        default="",
        help="Comma-separated host allowlist used when --link-scope allowlist is selected.",
    )
    parser.add_argument("--max-link-depth", type=int, default=1, help="Maximum link-following depth.")
    parser.add_argument(
        "--max-linked-pages-per-result",
        type=int,
        default=3,
        help="Maximum followed pages added for each source result.",
    )
    parser.add_argument(
        "--max-candidate-links",
        type=int,
        default=30,
        help="Maximum candidate links inspected on each source page.",
    )
    parser.add_argument(
        "--link-filter",
        choices=["llm", "heuristic", "none"],
        default="llm",
        help="How relevant_links are selected. llm uses the configured OpenAI-compatible model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Base output directory. A timestamped run directory will be created inside it.",
    )
    return parser

def make_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / timestamp
    counter = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def main(argv: list[str] | None = None) -> int:
    load_default_env_files()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    methods_explicit = option_was_provided(raw_argv, "--methods")
    deep_backend_explicit = option_was_provided(raw_argv, "--deep-backend")
    tavily_tools_explicit = option_was_provided(raw_argv, "--tavily-tools")
    if args.locate_max_items < 1:
        parser.error("--locate-max-items must be at least 1.")
    if args.locate_min_score < 0 or args.locate_min_score > 1:
        parser.error("--locate-min-score must be between 0 and 1.")
    if args.locate_extraction is not None:
        if not args.locate_extraction.exists():
            parser.error("--locate-extraction path does not exist.")
        run_dir = make_run_dir(args.output_dir)
        payload = locate_quotes_from_extraction(
            args.locate_extraction,
            run_dir,
            max_items=args.locate_max_items,
            min_score=args.locate_min_score,
        )
        print(f"Located {payload.get('matched_count')} of {payload.get('item_count')} quote item(s).")
        print(f"JSON: {run_dir / 'quote_locations.json'}")
        print(f"Report: {run_dir / 'report.md'}")
        return 0

    start_urls = parse_start_urls(args.start_urls)
    if args.start_urls.strip() and not start_urls:
        parser.error("--start-urls did not contain any valid public HTTP(S) URL.")
    has_start_urls = bool(start_urls)
    if has_start_urls and not deep_backend_explicit:
        args.deep_backend = "local_relevant_links"
    local_follow_mode, use_tavily_deep = resolve_deep_backend(args.deep_backend)

    if not args.query:
        parser.error("--query is required unless --locate-extraction is used.")
    if args.max_results < 1:
        parser.error("--max-results must be at least 1.")
    if args.tavily_seed_results < 0:
        parser.error("--tavily-seed-results must be at least 0.")
    if args.tavily_map_limit < 1:
        parser.error("--tavily-map-limit must be at least 1.")
    if args.tavily_crawl_limit < 1:
        parser.error("--tavily-crawl-limit must be at least 1.")
    if args.tavily_max_depth < 0:
        parser.error("--tavily-max-depth must be at least 0.")
    if args.tavily_max_breadth < 1:
        parser.error("--tavily-max-breadth must be at least 1.")
    if args.tavily_extract_limit < 1:
        parser.error("--tavily-extract-limit must be at least 1.")
    if args.content_max_chars < 1:
        parser.error("--content-max-chars must be at least 1.")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1.")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap must be at least 0.")
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size.")
    if args.viewport_max_scrolls < 0:
        parser.error("--viewport-max-scrolls must be at least 0.")
    if args.viewport_overlap < 0 or args.viewport_overlap >= 1:
        parser.error("--viewport-overlap must be between 0 and 1.")
    if args.viewport_text_max_chars < 1:
        parser.error("--viewport-text-max-chars must be at least 1.")
    if args.viewport_text_overlap_chars < 0:
        parser.error("--viewport-text-overlap-chars must be at least 0.")
    if args.viewport_height < 200:
        parser.error("--viewport-height must be at least 200.")
    if args.max_evidence_shots < 0:
        parser.error("--max-evidence-shots must be at least 0.")
    if args.ocr_min_extra_chars < 0:
        parser.error("--ocr-min-extra-chars must be at least 0.")
    if args.max_link_depth < 0:
        parser.error("--max-link-depth must be at least 0.")
    if args.max_linked_pages_per_result < 0:
        parser.error("--max-linked-pages-per-result must be at least 0.")
    if args.max_candidate_links < 1:
        parser.error("--max-candidate-links must be at least 1.")
    if args.link_scope == "allowlist" and not args.link_allowlist.strip():
        parser.error("--link-allowlist is required when --link-scope allowlist is used.")
    ocr_lang = args.ocr_lang or ("ch" if args.ocr_engine == "paddle" else "chi_sim+eng")

    run_dir = make_run_dir(args.output_dir)
    tavily_tools = normalize_tavily_tools(args.tavily_tools)
    if use_tavily_deep and not tavily_tools_explicit:
        tavily_tools = ["map", "crawl", "extract"]
    explicit_tavily_seed_urls = [item.strip() for item in args.tavily_seed_urls.split(",") if item.strip()]
    deep_tavily_seed_urls = start_urls if has_start_urls and use_tavily_deep else []

    results: list[SearchResult] = []
    if has_start_urls:
        results.extend(make_seed_results(args.query, start_urls, args.deep_title_prefix))

    search_methods = args.methods if (not has_start_urls or methods_explicit) else []
    if search_methods:
        results.extend(
            collect_results(
                args.query,
                search_methods,
                args.max_results,
                tavily_search_depth=args.tavily_search_depth,
            )
        )

    report_methods = list(search_methods)
    if has_start_urls:
        report_methods.insert(0, "seed_url")

    tavily_methods = list(search_methods)
    if use_tavily_deep:
        append_unique_methods(tavily_methods, ["tavily"])
    run_tavily = use_tavily_deep

    tavily_payload: dict[str, Any] = {
        "enabled": False,
        "tools": tavily_tools,
        "errors": [],
    }

    def run_tavily_collection() -> None:
        nonlocal tavily_payload
        if not run_tavily:
            return
        tavily_payload = collect_tavily_pages(
            results,
            run_dir,
            query=args.query,
            methods=tavily_methods,
            tools=tavily_tools,
            content_max_chars=args.content_max_chars,
            seed_urls=explicit_tavily_seed_urls + deep_tavily_seed_urls,
            seed_results=args.tavily_seed_results,
            map_limit=args.tavily_map_limit,
            crawl_limit=args.tavily_crawl_limit,
            max_depth=args.tavily_max_depth,
            max_breadth=args.tavily_max_breadth,
            extract_limit=args.tavily_extract_limit,
            extract_depth=args.tavily_extract_depth,
            tavily_format=args.tavily_format,
        )

    run_tavily_before_expansion = use_tavily_deep and not should_run_link_expansion(local_follow_mode)
    if run_tavily_before_expansion:
        run_tavily_collection()

    if should_run_link_expansion(local_follow_mode):
        expanded_results, link_payload = expand_results_with_links(
            results,
            run_dir,
            query=args.query,
            follow_pages=local_follow_mode,
            link_scope=args.link_scope,
            link_allowlist=[item.strip() for item in args.link_allowlist.split(",") if item.strip()],
            max_link_depth=args.max_link_depth,
            max_linked_pages_per_result=args.max_linked_pages_per_result,
            max_candidate_links=args.max_candidate_links,
            link_filter=args.link_filter,
            viewport_height=args.viewport_height,
        )
        if expanded_results:
            progress(f"Link expansion added {len(expanded_results)} page(s).")
            results.extend(expanded_results)
        elif link_payload.get("errors"):
            progress(f"Link expansion completed with {len(link_payload.get('errors') or [])} error(s).")
    if use_tavily_deep and not run_tavily_before_expansion:
        run_tavily_collection()
    search_page_screenshots: list[SearchPageScreenshot] = []
    extraction: dict[str, Any] | None = None

    needs_result_page_processing = (
        args.screenshots in {"result_pages", "both"}
        or args.content == "result_pages"
    )

    if args.screenshots != "none" or args.content != "none":
        with Screenshotter(run_dir, args.full_page) as screenshotter:
            if args.screenshots in {"search_page", "both"}:
                search_page_screenshots = capture_search_pages(args.query, report_methods, screenshotter)
            if needs_result_page_processing:
                process_result_pages(
                    results,
                    screenshotter,
                    capture_screenshots=args.screenshots in {"result_pages", "both"},
                    extract_content=args.content == "result_pages",
                    content_max_chars=args.content_max_chars,
                )

    if args.extract:
        chunking_errors: list[dict[str, Any]] = []
        if args.chunk_mode == "viewport":
            chunks, chunking_errors = chunk_viewports(
                results,
                run_dir,
                max_scrolls=args.viewport_max_scrolls,
                overlap_ratio=args.viewport_overlap,
                text_max_chars=args.viewport_text_max_chars,
                text_overlap_chars=args.viewport_text_overlap_chars,
                viewport_height=args.viewport_height,
                capture_screenshots=args.evidence_screenshots,
                max_screenshots=args.max_evidence_shots,
            )
        else:
            file_chunks = chunk_text(
                [
                    result
                    for result in results
                    if result.content_source in {"tavily_extract", "tavily_crawl"} and result.content_path
                ],
                run_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            dom_results = [
                result
                for result in results
                if not (result.content_source in {"tavily_extract", "tavily_crawl"} and result.content_path)
            ]
            dom_chunks, chunking_errors = chunk_dom_text(
                dom_results,
                run_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                viewport_height=args.viewport_height,
                capture_screenshots=args.evidence_screenshots,
                start_sequence=len(file_chunks) + 1,
            )
            chunks = file_chunks + dom_chunks
        write_chunks_progress(run_dir, chunks)
        if args.ocr:
            chunking_errors.extend(
                run_ocr_for_chunks(
                    chunks,
                    run_dir,
                    enabled=True,
                    engine=args.ocr_engine,
                    lang=ocr_lang,
                    min_extra_chars=args.ocr_min_extra_chars,
                )
            )
        if args.image_understanding:
            progress("Image understanding start")
            chunking_errors.extend(
                run_image_understanding_for_chunks(
                    chunks,
                    run_dir,
                    enabled=True,
                )
            )
            write_chunks_progress(run_dir, chunks)
            progress("Image understanding done")
        chunk_extractions, extraction_errors = extract_chunks_with_llm(
            chunks,
            run_dir,
            query=args.query,
            initial_errors=chunking_errors,
        )
        extraction_errors.extend(chunking_errors)
        if not chunks:
            extraction_errors.append(
                {
                    "stage": "chunking",
                    "error": "No result page text chunks were available. Ensure pages are reachable, or use --content result_pages with --chunk-mode text.",
                }
            )
        extraction = merge_extractions(args.query, chunks, chunk_extractions, extraction_errors)
        if args.evidence_screenshots and args.chunk_mode == "text":
            apply_chunk_shots_to_extraction(extraction, chunk_shot_map_from_chunks(chunks))
        elif args.chunk_mode == "viewport":
            apply_chunk_shots_to_extraction(extraction, chunk_shot_map_from_chunks(chunks))
        if args.evidence_screenshots:
            quote_shot_map = capture_quote_shots_for_extraction(
                extraction,
                run_dir,
                min_score=args.locate_min_score,
            )
            apply_quote_shots_to_extraction(extraction, quote_shot_map)
        write_extraction_files(run_dir, chunks, chunk_extractions, extraction)

    json_path = write_json(
        run_dir=run_dir,
        query=args.query,
        methods=report_methods,
        max_results=args.max_results,
        screenshots_mode=args.screenshots,
        full_page=args.full_page,
        content_mode=args.content,
        content_max_chars=args.content_max_chars,
        search_page_screenshots=search_page_screenshots,
        results=results,
        extra={
            "start_urls": start_urls,
            "deep_backend": args.deep_backend,
            "seed_url_count": len(start_urls),
            "link_scope": args.link_scope,
            "max_link_depth": args.max_link_depth,
            "max_linked_pages_per_result": args.max_linked_pages_per_result,
            "max_candidate_links": args.max_candidate_links,
            "link_filter": args.link_filter,
            "tavily_collection": tavily_payload,
        },
    )
    report_path = write_report(
        run_dir=run_dir,
        query=args.query,
        methods=report_methods,
        search_page_screenshots=search_page_screenshots,
        results=results,
        extraction=extraction,
    )

    print(f"Saved {len(results)} result row(s).")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")
    return 0
