from __future__ import annotations

from abc import ABC, abstractmethod

from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    EvidenceSignal,
    RawSource,
    SearchPlan,
    SearchQuery,
    SourceProviderName,
    SourceProviderResult,
    ToolPlan,
)


class SourceProvider(ABC):
    name: SourceProviderName

    @abstractmethod
    async def search_candidates(
        self,
        queries: list[str],
        benchmark: AnalysisBenchmark,
        *,
        max_sources: int = 50,
    ) -> SourceProviderResult:
        raise NotImplementedError

    @abstractmethod
    async def search_evidence(
        self,
        search_plan: SearchPlan,
        benchmark: AnalysisBenchmark,
        tool_plan: ToolPlan,
        competitors: list[CandidateCompetitor],
        *,
        max_sources: int = 60,
    ) -> SourceProviderResult:
        raise NotImplementedError

    @abstractmethod
    async def fetch_detail(
        self,
        urls: list[str],
        *,
        task_id: str = "task",
        max_sources: int = 20,
    ) -> SourceProviderResult:
        raise NotImplementedError

    @abstractmethod
    async def extract_signals(
        self,
        sources: list[RawSource],
        search_queries: list[SearchQuery] | None = None,
    ) -> list[EvidenceSignal]:
        raise NotImplementedError


def source_key(source: RawSource) -> str:
    return "|".join(
        [
            source.source_type,
            str(source.metadata.get("provider") or ""),
            str(source.metadata.get("tool") or ""),
            str(source.metadata.get("competitor") or ""),
            str(source.metadata.get("source_result_id") or ""),
            str(source.metadata.get("chunk_id") or ""),
            source.url or source.id,
            source.title,
        ]
    ).lower()


def dedupe_sources(sources: list[RawSource]) -> list[RawSource]:
    seen: set[str] = set()
    unique: list[RawSource] = []
    for source in sources:
        key = source_key(source)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique
