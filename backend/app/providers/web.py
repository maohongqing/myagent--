from __future__ import annotations

from urllib.parse import urlparse

from myagent.backend.app.agents import collector
from myagent.backend.app.config import Settings
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    EvidenceSignal,
    RawSource,
    SearchPlan,
    SearchQuery,
    SourceProviderResult,
    ToolPlan,
)
from myagent.backend.app.providers.base import SourceProvider, dedupe_sources


class WebProvider(SourceProvider):
    name = "web"

    def __init__(self, settings: Settings):
        self.settings = settings

    async def search_candidates(
        self,
        queries: list[str],
        benchmark: AnalysisBenchmark,
        *,
        max_sources: int = 50,
    ) -> SourceProviderResult:
        return SourceProviderResult(provider=self.name, sources=[], signals=[], cache_hit=False)

    async def search_evidence(
        self,
        search_plan: SearchPlan,
        benchmark: AnalysisBenchmark,
        tool_plan: ToolPlan,
        competitors: list[CandidateCompetitor],
        *,
        max_sources: int = 60,
    ) -> SourceProviderResult:
        urls = _urls_from_search_plan(search_plan)
        return await self.fetch_detail(urls, max_sources=max_sources)

    async def fetch_detail(
        self,
        urls: list[str],
        *,
        task_id: str = "task",
        max_sources: int = 20,
    ) -> SourceProviderResult:
        sources: list[RawSource] = []
        errors: list[str] = []
        for url in urls:
            if not _is_public_url(url):
                continue
            try:
                source = await collector.fetch_url(url)
                source.metadata["provider"] = self.name
                sources.append(source)
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                sources.append(
                    RawSource(
                        url=url,
                        title=f"Fetch failed: {url}",
                        content=f"URL fetch failed: {exc}",
                        source_type="url",
                        metadata={"provider": self.name, "error": str(exc)},
                    )
                )
            if len(sources) >= max_sources:
                break
        sources = dedupe_sources(sources)
        signals = await self.extract_signals(sources)
        return SourceProviderResult(provider=self.name, sources=sources, signals=signals, cache_hit=False, errors=errors)

    async def extract_signals(
        self,
        sources: list[RawSource],
        search_queries: list[SearchQuery] | None = None,
    ) -> list[EvidenceSignal]:
        return [
            EvidenceSignal(
                provider=self.name,
                signal_type="other",
                competitor=source.metadata.get("competitor"),
                value=source.content[:500],
                confidence=0.7 if not source.metadata.get("error") else 0.25,
                metadata={"url": source.url, "title": source.title},
            )
            for source in sources
        ]


def _urls_from_search_plan(search_plan: SearchPlan) -> list[str]:
    urls: list[str] = []
    for query in search_plan.queries:
        stripped = query.query.strip()
        if stripped.startswith(("http://", "https://")):
            urls.append(stripped)
    return urls


def _is_public_url(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host or host == "localhost" or host.endswith(".local"):
        return False
    return not (host.startswith("127.") or host.startswith("10.") or host.startswith("192.168."))
