from __future__ import annotations

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


class SearchProvider(SourceProvider):
    name = "search"

    def __init__(self, settings: Settings):
        self.settings = settings

    async def search_candidates(
        self,
        queries: list[str],
        benchmark: AnalysisBenchmark,
        *,
        max_sources: int = 50,
    ) -> SourceProviderResult:
        sources = await self._search_queries(queries, max_sources=max_sources)
        signals = await self.extract_signals(sources)
        return SourceProviderResult(provider=self.name, sources=sources, signals=signals, cache_hit=False)

    async def search_evidence(
        self,
        search_plan: SearchPlan,
        benchmark: AnalysisBenchmark,
        tool_plan: ToolPlan,
        competitors: list[CandidateCompetitor],
        *,
        max_sources: int = 60,
    ) -> SourceProviderResult:
        ordered = sorted(search_plan.queries, key=lambda item: item.priority)
        sources = await self._search_queries([item.query for item in ordered], max_sources=max_sources)
        by_query = {item.query: item for item in ordered}
        for source in sources:
            query = str(source.metadata.get("query") or "")
            plan_item = by_query.get(query)
            if plan_item:
                source.metadata.update(
                    {
                        "tool": plan_item.tool,
                        "competitor": plan_item.competitor,
                        "target_site_types": plan_item.target_site_types,
                        "expected_fields": plan_item.expected_fields,
                    }
                )
        signals = await self.extract_signals(sources, ordered)
        return SourceProviderResult(provider=self.name, sources=sources, signals=signals, cache_hit=False)

    async def fetch_detail(
        self,
        urls: list[str],
        *,
        task_id: str = "task",
        max_sources: int = 20,
    ) -> SourceProviderResult:
        return SourceProviderResult(provider=self.name, sources=[], signals=[], cache_hit=False)

    async def extract_signals(
        self,
        sources: list[RawSource],
        search_queries: list[SearchQuery] | None = None,
    ) -> list[EvidenceSignal]:
        signals: list[EvidenceSignal] = []
        for source in sources:
            signal_type = _signal_type_from_source(source)
            signals.append(
                EvidenceSignal(
                    provider=self.name,
                    signal_type=signal_type,
                    competitor=source.metadata.get("competitor"),
                    value=source.content[:500],
                    confidence=0.62 if source.source_type == "search" else 0.5,
                    metadata={
                        "query": source.metadata.get("query"),
                        "tool": source.metadata.get("tool"),
                        "url": source.url,
                    },
                )
            )
        return signals

    async def _search_queries(self, queries: list[str], *, max_sources: int) -> list[RawSource]:
        sources: list[RawSource] = []
        for query in queries:
            for source in await collector.tavily_search(query, self.settings):
                sources.append(source)
            for source in await collector.duckduckgo_search(query, self.settings):
                sources.append(source)
            if len(sources) >= max_sources:
                break
        return dedupe_sources(sources)[:max_sources]


def _signal_type_from_source(source: RawSource):
    text = f"{source.title} {source.url} {source.content}".lower()
    if any(word in text for word in ["pricing", "price", "plan", "定价", "价格"]):
        return "pricing"
    if any(word in text for word in ["review", "rating", "comment", "评价", "差评"]):
        return "review"
    if any(word in text for word in ["funding", "融资", "series a", "vc"]):
        return "funding"
    if any(word in text for word in ["hiring", "job", "招聘"]):
        return "hiring"
    if any(word in text for word in ["feature", "release", "changelog", "功能", "更新"]):
        return "feature"
    if any(word in text for word in ["ai", "llm", "model", "技术", "算法"]):
        return "technology"
    if any(word in text for word in ["growth", "download", "mau", "增长", "下载"]):
        return "growth"
    return "other"
