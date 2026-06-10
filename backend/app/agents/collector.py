from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

from myagent.backend.app.config import Settings
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CollectionPlan,
    CompetitorInput,
    CreateTaskRequest,
    RawSource,
)


def _clean_text(value: str, max_chars: int = 12000) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", value, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _extract_readable_text(value: str, max_chars: int = 12000) -> str:
    try:
        import trafilatura
    except ImportError:
        return ""
    try:
        extracted = trafilatura.extract(value, include_comments=False, include_tables=False) or ""
    except Exception:
        return ""
    text = re.sub(r"\s+", " ", extracted).strip()
    return text[:max_chars] if len(text) >= 300 else ""


def clean_source_text(value: str, *, max_chars: int = 12000, use_readability: bool = False) -> str:
    if use_readability:
        readable = _extract_readable_text(value, max_chars=max_chars)
        if readable:
            return readable
    return _clean_text(value, max_chars=max_chars)


def _title_from_html(html: str, fallback: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if not match:
        return fallback
    return re.sub(r"\s+", " ", match.group(1)).strip() or fallback


def _normalize_url(url: str) -> str:
    cleaned = (url or "").strip()
    if cleaned and not cleaned.startswith(("http://", "https://", "manual://")):
        return f"https://{cleaned}"
    return cleaned


def _is_public_http_url(url: str) -> bool:
    cleaned = _normalize_url(url)
    if not cleaned.startswith(("http://", "https://")):
        return False
    parsed = urlparse(cleaned)
    host = (parsed.hostname or "").lower()
    if not host or host == "localhost" or host.endswith(".local"):
        return False
    return not re.match(r"^(127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.)", host)


async def fetch_url(url: str, *, use_readability: bool = False) -> RawSource:
    normalized = _normalize_url(url)
    timeout = httpx.Timeout(15.0, connect=8.0)
    headers = {
        "User-Agent": "CompetitorAnalysisAgent/0.1 (+https://localhost)",
        "Accept": "text/html,text/plain,application/json;q=0.8,*/*;q=0.5",
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = await client.get(normalized)
        response.raise_for_status()
    title = _title_from_html(response.text, urlparse(str(response.url)).netloc or normalized)
    return RawSource(
        url=str(response.url),
        title=title,
        content=clean_source_text(response.text, use_readability=use_readability),
        source_type="url",
        metadata={"status_code": response.status_code, "readability_extraction": bool(use_readability)},
    )


async def tavily_search(query: str, settings: Settings) -> list[RawSource]:
    if not settings.tavily_api_key:
        return []
    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        return []

    client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    response = await client.search(
        query=query,
        search_depth="advanced",
        max_results=settings.tavily_max_results,
        include_answer=False,
        include_raw_content=True,
    )
    results = response.get("results", [])
    sources: list[RawSource] = []
    for result in results:
        content = result.get("raw_content") or result.get("content") or result.get("snippet") or ""
        if not content:
            continue
        sources.append(
            RawSource(
                url=result.get("url", ""),
                title=result.get("title") or result.get("url", "Search result"),
                content=_clean_text(content),
                source_type="search",
                metadata={"score": result.get("score"), "query": query, "provider": "tavily"},
            )
        )
    return sources


async def duckduckgo_search(query: str, settings: Settings) -> list[RawSource]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return []

    try:
        results = list(DDGS().text(query, max_results=settings.duckduckgo_max_results))
    except Exception as exc:
        digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
        return [
            RawSource(
                url=f"manual://duckduckgo-error/{digest}",
                title=f"DuckDuckGo search failed: {query}",
                content=f"DuckDuckGo search failed for query {query}: {exc}",
                source_type="manual",
                metadata={"query": query, "error": str(exc), "provider": "duckduckgo"},
            )
        ]

    sources: list[RawSource] = []
    for result in results:
        url = result.get("href") or result.get("url") or ""
        title = result.get("title") or url or "DuckDuckGo result"
        body = result.get("body") or result.get("snippet") or ""
        if not url and not body:
            continue
        fallback_digest = hashlib.sha1((query + title).encode("utf-8")).hexdigest()[:10]
        sources.append(
            RawSource(
                url=url or f"manual://duckduckgo/{fallback_digest}",
                title=title,
                content=_clean_text(body or title),
                source_type="search",
                metadata={"query": query, "provider": "duckduckgo"},
            )
        )
    return sources


def _manual_source(url: str, title: str, content: str, metadata: dict | None = None) -> RawSource:
    return RawSource(
        url=url,
        title=title,
        content=_clean_text(content),
        source_type="manual",
        metadata=metadata or {},
    )


def _competitor_manual_sources(profile: CompetitorInput) -> list[RawSource]:
    sources: list[RawSource] = []
    if profile.website:
        website = _normalize_url(profile.website)
        sources.append(
            _manual_source(
                website,
                f"{profile.name} official website",
                f"{profile.name} official website: {website}",
                {"competitor": profile.name, "field": "website"},
            )
        )
    if profile.website_copy:
        sources.append(
            _manual_source(
                f"manual://competitor/{profile.name}/website-copy",
                f"{profile.name} website copy",
                profile.website_copy,
                {"competitor": profile.name, "field": "website_copy"},
            )
        )
    if profile.sales_notes:
        sources.append(
            _manual_source(
                f"manual://competitor/{profile.name}/sales-notes",
                f"{profile.name} sales notes",
                profile.sales_notes,
                {"competitor": profile.name, "field": "sales_notes"},
            )
        )
    if profile.win_loss_notes:
        sources.append(
            _manual_source(
                f"manual://competitor/{profile.name}/win-loss-notes",
                f"{profile.name} win/loss notes",
                profile.win_loss_notes,
                {"competitor": profile.name, "field": "win_loss_notes"},
            )
        )
    if profile.tags:
        sources.append(
            _manual_source(
                f"manual://competitor/{profile.name}/tags",
                f"{profile.name} tags",
                ", ".join(profile.tags),
                {"competitor": profile.name, "field": "tags"},
            )
        )
    return sources


def _search_queries(request: CreateTaskRequest) -> list[str]:
    context = " ".join(
        item
        for item in [
            request.market or "",
            request.product_category or "",
            request.geography or "",
            request.company_size or "",
        ]
        if item
    )
    queries = [
        f"{request.target_product} 竞品分析 定价 功能 商业模式 {context}".strip(),
        f"{request.target_product} competitors pricing features business model {context}".strip(),
    ]
    for competitor in request.competitors:
        queries.extend(
            [
                f"{competitor} 官网 official website",
                f"{competitor} pricing plans",
                f"{competitor} features product docs help",
                f"{competitor} blog news changelog release notes",
                f"{request.target_product} vs {competitor} 功能 定价 用户 {context}".strip(),
                f"{competitor} pricing features target users business model {context}".strip(),
            ]
        )
    return [query for query in queries if query]


def _candidate_screenshot_urls(request: CreateTaskRequest, sources: list[RawSource]) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(url: str, label: str) -> None:
        normalized = _normalize_url(url)
        if normalized and normalized not in seen and _is_public_http_url(normalized):
            seen.add(normalized)
            urls.append((normalized, label))

    for profile in request.competitor_profiles:
        if profile.website:
            add(profile.website, f"{profile.name} official website")

    pricing_pattern = re.compile(r"(pricing|plans|price|套餐|价格)", re.I)
    for source in sources:
        if pricing_pattern.search(source.url) or pricing_pattern.search(source.title):
            add(source.url, f"{source.title} screenshot")
    return urls[: max(1, len(request.competitors) * 2)]


async def capture_screenshot(url: str, label: str, task_id: str, settings: Settings) -> RawSource:
    output_dir = Path(settings.sqlite_path).parent / "screenshots" / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    host = re.sub(r"[^\w.\-]+", "_", parsed.netloc or "page")[:80]
    filename = f"{host}_{hashlib.sha256(url.encode('utf-8')).hexdigest()[:10]}.png"
    path = output_dir / filename
    relative_path = f"screenshots/{task_id}/{filename}"

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed. Run `pip install playwright`.") from exc

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    user_agent="CompetitorAnalysisAgent/0.1 (+https://localhost)",
                    locale="zh-CN",
                )
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                await page.wait_for_timeout(800)
                await page.screenshot(path=str(path), full_page=False)
            finally:
                await browser.close()
    except Exception as exc:
        raise RuntimeError(f"Screenshot failed for {url}: {exc}") from exc

    return RawSource(
        url=url,
        title=f"Screenshot: {label}",
        content=f"Visual screenshot evidence for {label}. Image path: {relative_path}",
        source_type="screenshot",
        metadata={"image_path": relative_path, "image_alt": label},
    )


async def collect_sources(
    request: CreateTaskRequest,
    settings: Settings,
    *,
    task_id: str = "task",
) -> list[RawSource]:
    sources: list[RawSource] = []
    seen_source_keys: set[str] = set()

    def add_source(source: RawSource) -> None:
        key = "|".join(
            [
                source.source_type,
                str(source.metadata.get("provider") or ""),
                str(source.metadata.get("field") or ""),
                source.url or source.id,
            ]
        )
        if key and key not in seen_source_keys:
            seen_source_keys.add(key)
            sources.append(source)

    for profile in request.competitor_profiles:
        for source in _competitor_manual_sources(profile):
            add_source(source)

    for query in _search_queries(request):
        for source in await tavily_search(query, settings):
            add_source(source)
        for source in await duckduckgo_search(query, settings):
            add_source(source)

    urls = list(request.urls)
    for profile in request.competitor_profiles:
        if profile.website:
            urls.append(_normalize_url(profile.website))

    for url in urls:
        try:
            source = await fetch_url(url, use_readability=request.enable_readability_extraction)
        except Exception as exc:
            source = RawSource(
                url=url,
                title=f"Fetch failed: {url}",
                content=f"URL fetch failed: {exc}",
                source_type="url",
                metadata={"error": str(exc)},
            )
        add_source(source)

    if request.enable_screenshots:
        for url, label in _candidate_screenshot_urls(request, sources):
            try:
                add_source(await capture_screenshot(url, label, task_id, settings))
            except Exception as exc:
                add_source(
                    RawSource(
                        url=url,
                        title=f"Screenshot unavailable: {label}",
                        content=f"Screenshot unavailable for {url}: {exc}",
                        source_type="manual",
                        metadata={"error": str(exc), "field": "screenshot", "url": url},
                    )
                )

    if not sources:
        fallback_content = (
            f"用户请求分析目标产品 {request.target_product}，竞品包括 "
            f"{', '.join(request.competitors) or '未指定'}。"
            "当前未配置可用搜索或未提供可抓取 URL，因此只能生成低置信度框架分析。"
        )
        sources.append(
            RawSource(
                url="manual://task-input",
                title="Task input fallback",
                content=fallback_content,
                source_type="manual",
            )
        )

    return sources[:40]


async def collect_candidate_sources(
    queries: list[str],
    settings: Settings,
    *,
    max_sources: int = 50,
) -> list[RawSource]:
    sources: list[RawSource] = []
    seen: set[str] = set()
    for query in queries:
        for source in await tavily_search(query, settings):
            key = f"{source.source_type}|{source.url}|{source.title}"
            if key not in seen:
                seen.add(key)
                sources.append(source)
        for source in await duckduckgo_search(query, settings):
            key = f"{source.source_type}|{source.url}|{source.title}"
            if key not in seen:
                seen.add(key)
                sources.append(source)
        if len(sources) >= max_sources:
            break
    return sources[:max_sources]


async def collect_sources_for_plan(
    request: CreateTaskRequest,
    settings: Settings,
    collection_plan: CollectionPlan,
    *,
    task_id: str = "task",
) -> list[RawSource]:
    sources: list[RawSource] = []
    seen: set[str] = set()

    def add(source: RawSource) -> None:
        key = "|".join(
            [
                source.source_type,
                str(source.metadata.get("provider") or ""),
                str(source.metadata.get("tool") or ""),
                str(source.metadata.get("competitor") or ""),
                source.url or source.id,
            ]
        )
        if key not in seen:
            seen.add(key)
            sources.append(source)

    for profile in request.competitor_profiles:
        for source in _competitor_manual_sources(profile):
            add(source)

    for instruction in collection_plan.instructions:
        for query in instruction.queries:
            for source in await tavily_search(query, settings):
                source.metadata.update(
                    {
                        "tool": instruction.tool,
                        "competitor": instruction.competitor,
                        "extraction_targets": instruction.extraction_targets,
                    }
                )
                add(source)
            for source in await duckduckgo_search(query, settings):
                source.metadata.update(
                    {
                        "tool": instruction.tool,
                        "competitor": instruction.competitor,
                        "extraction_targets": instruction.extraction_targets,
                    }
                )
                add(source)
            if len(sources) >= 60:
                break
        if len(sources) >= 60:
            break

    for url in request.urls:
        try:
            add(await fetch_url(url, use_readability=request.enable_readability_extraction))
        except Exception as exc:
            add(
                RawSource(
                    url=url,
                    title=f"Fetch failed: {url}",
                    content=f"URL fetch failed: {exc}",
                    source_type="url",
                    metadata={"error": str(exc)},
                )
            )

    if request.enable_screenshots:
        for url, label in _candidate_screenshot_urls(request, sources):
            try:
                add(await capture_screenshot(url, label, task_id, settings))
            except Exception as exc:
                add(
                    RawSource(
                        url=url,
                        title=f"Screenshot unavailable: {label}",
                        content=f"Screenshot unavailable for {url}: {exc}",
                        source_type="manual",
                        metadata={"error": str(exc), "field": "screenshot", "url": url},
                    )
                )

    if not sources:
        description = request.raw_description or request.notes or request.target_product
        sources.append(
            RawSource(
                url="manual://task-input",
                title="Task input fallback",
                content=(
                    f"用户请求分析 {request.target_product or '未命名产品'}。"
                    f"原始描述：{description or '未提供'}。"
                    "当前未配置可用搜索结果或外部链接，因此生成低置信度框架分析。"
                ),
                source_type="manual",
            )
        )

    return sources[:60]
