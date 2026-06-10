from __future__ import annotations

import json
import re
from pathlib import Path

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


class SourceCacheProvider(SourceProvider):
    name = "cache"

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path

    async def search_candidates(
        self,
        queries: list[str],
        benchmark: AnalysisBenchmark,
        *,
        max_sources: int = 50,
    ) -> SourceProviderResult:
        sources = self._match_cached_sources(queries, benchmark.target_product)[:max_sources]
        signals = await self.extract_signals(sources)
        return SourceProviderResult(provider=self.name, sources=sources, signals=signals, cache_hit=bool(sources))

    async def search_evidence(
        self,
        search_plan: SearchPlan,
        benchmark: AnalysisBenchmark,
        tool_plan: ToolPlan,
        competitors: list[CandidateCompetitor],
        *,
        max_sources: int = 60,
    ) -> SourceProviderResult:
        queries = [item.query for item in search_plan.queries]
        keywords = [benchmark.target_product, *[item.name for item in competitors], *queries]
        sources = self._match_cached_sources(keywords, benchmark.target_product)[:max_sources]
        signals = await self.extract_signals(sources, search_plan.queries)
        return SourceProviderResult(provider=self.name, sources=sources, signals=signals, cache_hit=bool(sources))

    async def fetch_detail(
        self,
        urls: list[str],
        *,
        task_id: str = "task",
        max_sources: int = 20,
    ) -> SourceProviderResult:
        cached = self._read_sources()
        wanted = {url.strip().lower() for url in urls if url.strip()}
        sources = [source for source in cached if source.url.lower() in wanted][:max_sources]
        signals = await self.extract_signals(sources)
        return SourceProviderResult(provider=self.name, sources=sources, signals=signals, cache_hit=bool(sources))

    async def extract_signals(
        self,
        sources: list[RawSource],
        search_queries: list[SearchQuery] | None = None,
    ) -> list[EvidenceSignal]:
        signals: list[EvidenceSignal] = []
        for source in sources:
            signal_type = str(source.metadata.get("signal_type") or "other")
            if signal_type not in {
                "positioning",
                "pricing",
                "feature",
                "review",
                "growth",
                "funding",
                "hiring",
                "technology",
                "risk",
                "market",
                "other",
            }:
                signal_type = "other"
            signals.append(
                EvidenceSignal(
                    provider=self.name,
                    signal_type=signal_type,  # type: ignore[arg-type]
                    competitor=source.metadata.get("competitor"),
                    value=source.content[:500],
                    confidence=float(source.metadata.get("confidence") or 0.55),
                    metadata={"source_url": source.url, "source_title": source.title},
                )
            )
        return signals

    def remember_sources(self, sources: list[RawSource]) -> None:
        if not sources:
            return
        existing = self._read_sources()
        merged = dedupe_sources([*existing, *sources])
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.model_dump(mode="json") for item in merged[-1000:]]
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_sources(self) -> list[RawSource]:
        if not self.cache_path.exists():
            return []
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        sources: list[RawSource] = []
        for item in payload:
            try:
                sources.append(RawSource.model_validate(item))
            except Exception:
                continue
        return sources

    def _match_cached_sources(self, keywords: list[str], fallback_keyword: str) -> list[RawSource]:
        cached = self._read_sources()
        tokens = _tokens(" ".join([*keywords, fallback_keyword]))
        if not tokens:
            return []
        matched: list[RawSource] = []
        for source in cached:
            haystack = f"{source.title} {source.url} {source.content}".lower()
            if any(token in haystack for token in tokens):
                matched.append(source)
        return dedupe_sources(matched)


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}", text)]
