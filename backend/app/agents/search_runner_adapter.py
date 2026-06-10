from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

try:
    from myagent.search.chunking import source_result_id
    from myagent.search.runner import SearchRunOptions, SearchRunResult, run_search
    from myagent.search.utils import DEFAULT_METHODS, SUPPORTED_METHODS, canonical_url_key
except ModuleNotFoundError:
    SearchRunResult = Any  # type: ignore[assignment]
    DEFAULT_METHODS = ["duckduckgo"]
    SUPPORTED_METHODS = set(DEFAULT_METHODS)

    class SearchRunOptions:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def source_result_id(result) -> str:
        return str(getattr(result, "id", "") or getattr(result, "url", "") or getattr(result, "source_url", ""))

    def canonical_url_key(url: str) -> str:
        return (url or "").strip().lower()

    def run_search(*_args, **_kwargs):
        raise RuntimeError("Missing optional search package. Install or include the top-level 'search' package to run web collection.")

from myagent.backend.app.config import Settings
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CreateTaskRequest,
    EvidenceSignal,
    RawSource,
    SearchPlan,
    SearchQuery,
    SourceProviderResult,
)
from myagent.backend.app.agents.collector import clean_source_text


SUPPORTED_DEEP_BACKENDS = [
    "none",
    "local_pagination",
    "local_relevant_links",
    "local_all_links",
    "tavily",
    "local_relevant_links_tavily",
]
DEFAULT_RESULTS_PER_QUERY_PER_SOURCE = 5
WEB_CHUNK_SIZE = 3000
WEB_CHUNK_OVERLAP = 300


def normalize_search_methods(methods: list[str] | None) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for method in methods or DEFAULT_METHODS:
        if method in SUPPORTED_METHODS and method not in seen:
            seen.add(method)
            items.append(method)
    return items or list(DEFAULT_METHODS)


def normalize_deep_backend(value: str | None) -> str:
    return value if value in SUPPORTED_DEEP_BACKENDS else "none"


async def collect_candidate_sources_with_search_runner(
    queries: list[str],
    request: CreateTaskRequest,
    settings: Settings,
    *,
    task_id: str,
    max_sources: int = 50,
    seen_url_keys: set[str] | None = None,
    should_cancel=None,
    on_sources: Callable[[list[RawSource]], Awaitable[None]] | None = None,
) -> tuple[list[RawSource], list[dict[str, Any]]]:
    sources: list[RawSource] = []
    errors: list[dict[str, Any]] = []
    seen = seen_url_keys if seen_url_keys is not None else set()
    query_items = [SearchQuery(query=query, priority=3, rationale="Candidate discovery") for query in queries[: settings.max_queries_per_generation]]
    seen_lock = asyncio.Lock()
    source_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, settings.search_query_concurrency))

    async def run_one(query: SearchQuery) -> None:
        nonlocal sources
        if should_cancel:
            should_cancel()
        async with source_lock:
            if len(sources) >= max_sources:
                return
        async with semaphore:
            async with seen_lock:
                seen_snapshot = set(seen)
            run, run_error = await _run_query(
                query.query,
                request,
                settings,
                task_id=task_id,
                purpose="candidate_discovery",
                max_results=DEFAULT_RESULTS_PER_QUERY_PER_SOURCE,
                seen_url_keys=seen_snapshot,
            )
        if should_cancel:
            should_cancel()
        if run_error:
            errors.append(run_error)
            return
        assert run is not None
        run_sources = _sources_from_run(run, request, query=query, purpose="candidate_discovery")
        async with seen_lock:
            _update_seen_url_keys(seen, run)
        if on_sources and run_sources:
            try:
                await on_sources(run_sources)
            except Exception as exc:
                errors.append(
                    {
                        "stage": "candidate_source_callback",
                        "query": query.query,
                        "purpose": "candidate_discovery",
                        "error": str(exc),
                    }
                )
        async with source_lock:
            errors.extend(run.errors)
            remaining = max(0, max_sources - len(sources))
            if remaining:
                sources.extend(run_sources[:remaining])

    await asyncio.gather(*(run_one(query) for query in query_items))
    return sources[:max_sources], errors


async def collect_candidate_existence_sources_with_search_runner(
    candidate_name: str,
    request: CreateTaskRequest,
    settings: Settings,
    *,
    task_id: str,
    max_sources: int = 3,
    should_cancel=None,
) -> tuple[list[RawSource], list[dict[str, Any]]]:
    if should_cancel:
        should_cancel()
    query = SearchQuery(query=candidate_name, priority=1, rationale="Candidate existence QA")
    run, run_error = await _run_query(
        candidate_name,
        request,
        settings,
        task_id=task_id,
        purpose="candidate_existence_qa",
        max_results=max(1, max_sources),
    )
    if should_cancel:
        should_cancel()
    if run_error:
        return [], [run_error]
    assert run is not None
    sources = _sources_from_run(run, request, query=query, purpose="candidate_existence_qa")
    errors = list(run.errors)
    return sources[:max_sources], errors


async def collect_evidence_with_search_runner(
    search_plan: SearchPlan,
    request: CreateTaskRequest,
    settings: Settings,
    *,
    task_id: str,
    benchmark: AnalysisBenchmark,
    competitors: list[CandidateCompetitor],
    max_sources: int = 80,
    seen_url_keys: set[str] | None = None,
    should_cancel=None,
    on_sources: Callable[[list[RawSource]], Awaitable[None]] | None = None,
) -> SourceProviderResult:
    sources: list[RawSource] = []
    errors: list[str] = []
    seen = seen_url_keys if seen_url_keys is not None else set()
    queries = sorted(search_plan.queries, key=lambda item: item.priority)
    seed_urls = [url for url in request.urls if url.startswith(("http://", "https://"))]
    seen_lock = asyncio.Lock()
    source_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, settings.search_query_concurrency))

    async def run_one(query: SearchQuery) -> None:
        nonlocal sources
        if should_cancel:
            should_cancel()
        async with source_lock:
            if len(sources) >= max_sources:
                return
        async with semaphore:
            async with seen_lock:
                seen_snapshot = set(seen)
            run, run_error = await _run_query(
                query.query,
                request,
                settings,
                task_id=task_id,
                purpose="targeted_collection",
                max_results=DEFAULT_RESULTS_PER_QUERY_PER_SOURCE,
                start_urls=seed_urls,
                seen_url_keys=seen_snapshot,
            )
        if should_cancel:
            should_cancel()
        if run_error:
            errors.append(str(run_error.get("error") or run_error))
            return
        assert run is not None
        async with seen_lock:
            _update_seen_url_keys(seen, run)
        run_sources = _sources_from_run(run, request, query=query, purpose="targeted_collection")
        if on_sources and run_sources:
            try:
                await on_sources(run_sources)
            except Exception as exc:
                errors.append(str(exc))
        async with source_lock:
            errors.extend(str(item.get("error") or item) for item in run.errors[:12])
            remaining = max(0, max_sources - len(sources))
            if remaining:
                sources.extend(run_sources[:remaining])

    await asyncio.gather(*(run_one(query) for query in queries[: settings.max_queries_per_generation]))

    signals = _signals_from_sources(sources, search_plan.queries, competitors)
    return SourceProviderResult(
        provider="search",
        sources=sources[:max_sources],
        signals=signals,
        cache_hit=False,
        query="; ".join(item.query for item in queries[:8]),
        errors=errors,
        metadata={
            "runner": "myagent.search.runner",
            "search_methods": normalize_search_methods(list(request.search_methods)),
            "deep_backend": normalize_deep_backend(request.deep_search_backend),
            "target_product": benchmark.target_product,
            "seen_url_key_count": len(seen),
        },
    )


async def _run_query(
    query: str,
    request: CreateTaskRequest,
    settings: Settings,
    *,
    task_id: str,
    purpose: str,
    max_results: int,
    start_urls: list[str] | None = None,
    seen_url_keys: set[str] | None = None,
) -> tuple[SearchRunResult | None, dict[str, Any] | None]:
    options = _options_for_request(
        request,
        settings,
        task_id=task_id,
        purpose=purpose,
        max_results=max_results,
        start_urls=start_urls or [],
        seen_url_keys=seen_url_keys or set(),
    )
    try:
        return await asyncio.to_thread(run_search, query, options), None
    except Exception as exc:
        return None, {
            "stage": "search_runner",
            "query": query,
            "purpose": purpose,
            "error": str(exc),
        }


def _options_for_request(
    request: CreateTaskRequest,
    settings: Settings,
    *,
    task_id: str,
    purpose: str,
    max_results: int,
    start_urls: list[str] | None = None,
    seen_url_keys: set[str] | None = None,
) -> SearchRunOptions:
    data_dir = Path(settings.sqlite_path).parent
    screenshots = "none"
    capped_max_results = max_results
    if purpose != "candidate_existence_qa":
        capped_max_results = min(max_results, DEFAULT_RESULTS_PER_QUERY_PER_SOURCE)
    return SearchRunOptions(
        methods=normalize_search_methods(list(request.search_methods)),
        deep_backend=normalize_deep_backend(request.deep_search_backend),
        max_results=max(1, capped_max_results),
        start_urls=start_urls or [],
        exclude_url_keys=set(seen_url_keys or set()),
        output_dir=data_dir / "search_runs" / task_id / purpose,
        content="result_pages",
        screenshots=screenshots,
        browser_disable_proxy=settings.search_browser_disable_proxy,
        collect_chunks=True,
        chunk_mode="text",
        chunk_size=WEB_CHUNK_SIZE,
        chunk_overlap=WEB_CHUNK_OVERLAP,
        evidence_screenshots=False,
        max_evidence_shots=0,
        content_max_chars=12000,
        tavily_search_depth="basic",
        tavily_extract_depth="basic",
        link_filter="heuristic",
    )


def _sources_from_run(
    run: SearchRunResult,
    request: CreateTaskRequest,
    *,
    query: SearchQuery,
    purpose: str,
) -> list[RawSource]:
    data_dir = _data_dir_for_run(run.run_dir)
    by_source_id = {source_result_id(result): result for result in run.results}
    sources: list[RawSource] = []
    if run.chunks:
        for chunk in run.chunks:
            result = by_source_id.get(chunk.source_result_id)
            screenshot_path = None
            title = chunk.title or (result.title if result else "Untitled source")
            metadata = _base_metadata(run, request, query, purpose, data_dir)
            metadata.update(
                {
                    "method": chunk.method,
                    "rank": chunk.rank,
                    "source_result_id": chunk.source_result_id,
                    "chunk_id": chunk.chunk_id,
                    "content_path": _relative_data_path(run.run_dir, chunk.content_path, data_dir),
                    "screenshot_path": screenshot_path,
                    "chunk_shot_path": None,
                    "chunk_shot_paths": [],
                    "evidence_screenshot_path": None,
                    "image_path": None,
                    "image_alt": f"{title} evidence screenshot",
                    "source_url": result.source_url if result else None,
                    "discovery_type": result.discovery_type if result else None,
                    "tool": query.tool,
                    "competitor": query.competitor,
                    "expected_fields": query.expected_fields,
                }
            )
            for split_index, split_text in enumerate(_split_web_chunk_text(_clean_run_content(chunk.text or "", request)), start=1):
                split_metadata = dict(metadata)
                split_chunk_id = _split_chunk_id(chunk.chunk_id, split_index)
                split_metadata["chunk_id"] = split_chunk_id
                split_metadata["parent_chunk_id"] = chunk.chunk_id
                split_metadata["chunk_split_index"] = split_index
                sources.append(
                    RawSource(
                        url=chunk.url,
                        title=f"{title} [{split_chunk_id}]",
                        content=split_text,
                        source_type="search",
                        metadata=split_metadata,
                    )
                )
    else:
        for result in run.results:
            content = _clean_run_content(_content_for_result(run.run_dir, result), request)
            screenshot_path = None
            metadata = _base_metadata(run, request, query, purpose, data_dir)
            metadata.update(
                {
                    "method": result.method,
                    "rank": result.rank,
                    "source_result_id": source_result_id(result),
                    "content_path": _relative_data_path(run.run_dir, result.content_path, data_dir),
                    "screenshot_path": screenshot_path,
                    "chunk_shot_path": None,
                    "chunk_shot_paths": [],
                    "evidence_screenshot_path": None,
                    "image_path": None,
                    "image_alt": f"{result.title or 'source'} screenshot",
                    "source_url": result.source_url,
                    "discovery_type": result.discovery_type,
                    "tool": query.tool,
                    "competitor": query.competitor,
                    "expected_fields": query.expected_fields,
                }
            )
            fallback_chunk_id = source_result_id(result) or "result"
            for split_index, split_text in enumerate(_split_web_chunk_text(content), start=1):
                split_metadata = dict(metadata)
                split_chunk_id = _split_chunk_id(fallback_chunk_id, split_index)
                split_metadata["chunk_id"] = split_chunk_id
                split_metadata["parent_chunk_id"] = fallback_chunk_id
                split_metadata["chunk_split_index"] = split_index
                sources.append(
                    RawSource(
                        url=result.url,
                        title=f"{result.title or 'Untitled source'} [{split_chunk_id}]",
                        content=split_text,
                        source_type="search",
                        metadata=split_metadata,
                    )
                )
    return sources


def _base_metadata(
    run: SearchRunResult,
    request: CreateTaskRequest,
    query: SearchQuery,
    purpose: str,
    data_dir: Path,
) -> dict[str, Any]:
    return {
        "provider": "search_runner",
        "runner": "myagent.search.runner",
        "purpose": purpose,
        "query": query.query,
        "query_rationale": query.rationale,
        "search_methods": normalize_search_methods(list(request.search_methods)),
        "deep_backend": run.deep_backend,
        "run_dir": _relative_data_path(run.run_dir, "", data_dir),
        "search_run_json": _relative_data_path(run.run_dir, run.json_path, data_dir),
        "search_run_report": _relative_data_path(run.run_dir, run.report_path, data_dir),
        "screenshots_mode": run.screenshots_mode,
        "skipped_duplicate_count": run.metadata.get("skipped_duplicate_count", 0),
    }


def _update_seen_url_keys(seen: set[str], run: SearchRunResult) -> None:
    for key in run.metadata.get("seen_url_keys") or []:
        if key:
            seen.add(str(key))
    for result in run.results:
        key = canonical_url_key(result.final_url or result.url)
        if key:
            seen.add(key)


def _signals_from_sources(
    sources: list[RawSource],
    queries: list[SearchQuery],
    competitors: list[CandidateCompetitor],
) -> list[EvidenceSignal]:
    by_query = {query.query: query for query in queries}
    competitor_names = [item.name for item in competitors]
    signals: list[EvidenceSignal] = []
    for source in sources:
        query = by_query.get(str(source.metadata.get("query") or ""))
        competitor = source.metadata.get("competitor") or _match_competitor(source, competitor_names)
        expected = query.expected_fields if query else source.metadata.get("expected_fields") or []
        signal_type = _signal_type(expected)
        signals.append(
            EvidenceSignal(
                provider="search",
                signal_type=signal_type,
                competitor=competitor,
                value=(source.content or source.title)[:800],
                confidence=0.6 if source.content else 0.4,
                metadata={
                    "source_url": source.url,
                    "source_title": source.title,
                    "source_id": source.id,
                    "chunk_id": source.metadata.get("chunk_id"),
                    "query": source.metadata.get("query"),
                    "image_path": source.metadata.get("image_path"),
                },
            )
        )
    return signals


def _signal_type(expected_fields: list[str] | Any) -> str:
    values = {str(item).lower() for item in (expected_fields or [])}
    for candidate in ["pricing", "growth", "funding", "hiring", "technology", "risk", "review", "feature", "market", "positioning"]:
        if candidate in values:
            return candidate
    return "other"


def _match_competitor(source: RawSource, names: list[str]) -> str | None:
    haystack = f"{source.title} {source.content}".lower()
    for name in names:
        if name and name.lower() in haystack:
            return name
    return None


def _content_for_result(run_dir: Path, result) -> str:
    parts = [result.snippet or "", result.content_preview or ""]
    if result.content_path:
        path = run_dir / result.content_path
        try:
            parts.append(path.read_text(encoding="utf-8")[:8000])
        except OSError:
            pass
    text = "\n".join(part for part in parts if part).strip()
    return text or result.title or "No source content available."


def _clean_run_content(content: str, request: CreateTaskRequest) -> str:
    if not request.enable_readability_extraction:
        return content
    return clean_source_text(content, use_readability=True)


def _split_web_chunk_text(content: str) -> list[str]:
    text = (content or "").strip()
    if not text:
        return []
    if len(text) <= WEB_CHUNK_SIZE:
        return [text]

    chunks: list[str] = []
    step = WEB_CHUNK_SIZE - WEB_CHUNK_OVERLAP
    start = 0
    while start < len(text):
        end = min(len(text), start + WEB_CHUNK_SIZE)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def _split_chunk_id(chunk_id: str, split_index: int) -> str:
    base = str(chunk_id or "chunk")
    return base if split_index == 1 else f"{base}_part_{split_index:02d}"


def _data_dir_for_run(run_dir: Path) -> Path:
    for parent in [run_dir, *run_dir.parents]:
        if parent.name == "data":
            return parent
    return run_dir.parent


def _relative_data_path(run_dir: Path, value: str | Path | None, data_dir: Path) -> str | None:
    if value is None:
        return None
    raw = str(value)
    if not raw:
        path = run_dir
    else:
        path = Path(raw)
        if not path.is_absolute():
            path = run_dir / path
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return str(path)
