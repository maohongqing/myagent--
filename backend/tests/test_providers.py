import pytest

from myagent.backend.app.config import Settings
from myagent.backend.app.models import AnalysisBenchmark, RawSource, SearchPlan, SearchQuery, ToolPlan
from myagent.backend.app.providers.cache import SourceCacheProvider
from myagent.backend.app.providers.search import SearchProvider


@pytest.mark.asyncio
async def test_cache_provider_hits_local_source(tmp_path):
    provider = SourceCacheProvider(tmp_path / "source_cache.json")
    provider.remember_sources(
        [
            RawSource(
                url="https://betasuite.example/pricing",
                title="BetaSuite pricing",
                content="BetaSuite pricing includes a free plan.",
                source_type="url",
                metadata={"competitor": "BetaSuite", "signal_type": "pricing"},
            )
        ]
    )
    benchmark = AnalysisBenchmark(
        target_product="Acme Workspace",
        raw_description="Collaboration product",
        lifecycle_stage="concept",
        analysis_purpose="decision_support",
        analysis_goal="Analyze competitors.",
    )
    plan = SearchPlan(queries=[SearchQuery(query="BetaSuite pricing", expected_fields=["pricing"])])
    result = await provider.search_evidence(
        plan,
        benchmark,
        ToolPlan(purpose="decision_support", focus="market", selected_tools=["SWOT分析"]),
        [],
    )

    assert result.cache_hit is True
    assert result.sources[0].title == "BetaSuite pricing"
    assert result.signals[0].signal_type == "pricing"


@pytest.mark.asyncio
async def test_search_provider_uses_search_fallback(monkeypatch, tmp_path):
    from myagent.backend.app.agents import collector

    async def fake_tavily(query: str, settings: Settings):
        return []

    async def fake_duckduckgo(query: str, settings: Settings):
        return [
            RawSource(
                url="https://example.com/result",
                title="BetaSuite features",
                content="BetaSuite release notes mention workflow automation.",
                source_type="search",
                metadata={"provider": "duckduckgo", "query": query},
            )
        ]

    monkeypatch.setattr(collector, "tavily_search", fake_tavily)
    monkeypatch.setattr(collector, "duckduckgo_search", fake_duckduckgo)

    provider = SearchProvider(Settings(openai_api_key=None, tavily_api_key=None, sqlite_path=tmp_path / "app.db"))
    benchmark = AnalysisBenchmark(
        target_product="Acme Workspace",
        raw_description="Collaboration product",
        lifecycle_stage="development",
        analysis_purpose="learning",
        analysis_goal="Learn features.",
    )
    plan = SearchPlan(queries=[SearchQuery(query="BetaSuite release notes", tool="功能拆解", expected_fields=["feature"])])
    result = await provider.search_evidence(
        plan,
        benchmark,
        ToolPlan(purpose="learning", focus="features", selected_tools=["功能拆解"]),
        [],
    )

    assert result.sources
    assert result.signals[0].signal_type == "feature"
