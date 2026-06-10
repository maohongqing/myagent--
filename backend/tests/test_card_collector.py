import json

import pytest

from myagent.backend.app.agents.card_collector import (
    CardFieldPatch,
    CardGapQueryOutput,
    CardPatchOutput,
    _competitor_gap_queries,
    _normalize_card_qa_result,
    _clear_error_fields,
    build_card_report_references,
    collect_detail_cards,
)
from myagent.backend.app.agents.chunk_filter import filter_relevant_sources
from myagent.backend.app.agents.strategy import scoring_profile_for
from myagent.backend.app.config import Settings
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CardQAResult,
    CompetitorDetailCard,
    CreateTaskRequest,
    DetailQueryPlan,
    IndustryCard,
    IndustryQuery,
    QueryTemplate,
    RawSource,
    ResearchField,
    SearchQuery,
    SelectedCompetitorSet,
    SourceProviderResult,
    SourceRef,
)


def _benchmark() -> AnalysisBenchmark:
    return AnalysisBenchmark(
        target_product="Acme Workspace",
        raw_description="Collaboration product",
        lifecycle_stage="concept",
        analysis_purpose="decision_support",
        analysis_goal="Analyze competitors.",
    )


def _selected_set(*names: str) -> SelectedCompetitorSet:
    competitors = [CandidateCompetitor(name=name, competitor_type="head_direct", score=0.9) for name in names]
    return SelectedCompetitorSet(
        total_budget=len(competitors),
        scoring_profile=scoring_profile_for("decision_support"),
        candidates=competitors,
        selected=competitors,
        allocation={"head_direct": len(competitors)},
    )


def _query_plan() -> DetailQueryPlan:
    return DetailQueryPlan(
        competitor_query_templates=[
            QueryTemplate(template="{competitor} pricing features", target_fields=["pricing", "core_features"], priority=1),
            QueryTemplate(template="{competitor} business model growth", target_fields=["business_model", "growth_signals"], priority=2),
            QueryTemplate(template="{competitor} reviews risks", target_fields=["user_feedback", "risks"], priority=3),
            QueryTemplate(template="{competitor} extra query", target_fields=["recent_updates"], priority=4),
        ],
        industry_queries=[
            IndustryQuery(query="collaboration market size", target_fields=["market_situation.market_size"], priority=1),
            IndustryQuery(query="collaboration growth trend", target_fields=["market_situation.growth_trend"], priority=2),
            IndustryQuery(query="collaboration regulation technology", target_fields=["external_environment.technology_trends"], priority=3),
            IndustryQuery(query="collaboration extra industry", target_fields=["basic_info.main_users"], priority=4),
        ],
    )


def test_card_references_use_stable_ids_and_anchors():
    card = CompetitorDetailCard(
        competitor="BetaSuite",
        positioning=ResearchField(
            value="BetaSuite targets async product teams.",
            confidence=0.82,
            source_refs=[
                SourceRef(
                    source_id="raw_1",
                    chunk_id="chunk_1",
                    url="https://example.com/about",
                    title="About BetaSuite",
                    quote="targets async product teams",
                    evidence_screenshot_path="screenshots/raw_1.png",
                )
            ],
        ),
    )
    industry = IndustryCard()
    industry.market_situation.market_size = ResearchField(
        value="The collaboration market is large and growing.",
        confidence=0.76,
    )

    references = build_card_report_references([card], industry)
    by_id = {item.id: item for item in references}

    assert "card.competitor.betasuite.positioning" in by_id
    assert by_id["card.competitor.betasuite.positioning"].anchor == "ref-card-competitor-betasuite-positioning"
    assert by_id["card.competitor.betasuite.positioning"].payload["source_refs"][0]["evidence_screenshot_path"] == "screenshots/raw_1.png"
    assert "card.industry.market_situation.market_size" in by_id
    assert by_id["card.industry.market_situation.market_size"].anchor == "ref-card-industry-market-situation-market-size"


def test_chunk_filter_skips_ad_chunks_and_keeps_field_evidence():
    ad = RawSource(url="https://example.com/ad", title="Ads", content="登录 注册 隐私政策 用户协议 热门推荐 相关推荐 广告 推广", source_type="search")
    useful = RawSource(url="https://example.com/growth", title="BetaSuite report", content="BetaSuite 2024 revenue growth slowed and user complaints increased after a product update.", source_type="search")

    kept, stats = filter_relevant_sources(
        [ad, useful],
        object_name="BetaSuite",
        expected_fields=["growth_signals", "user_feedback"],
        min_chars=20,
    )

    assert kept == [useful]
    assert stats.kept_count == 1
    assert stats.skipped_reasons


def test_card_qa_optional_fields_do_not_trigger_retry():
    qa = CardQAResult(
        passed=False,
        needs_retry=True,
        missing_fields=["recent_updates"],
        error_fields=["channels"],
        reference_issues=["optional weak"],
    )

    normalized = _normalize_card_qa_result(qa, "competitor")

    assert normalized.passed is True
    assert normalized.needs_retry is False
    assert normalized.missing_fields == []
    assert normalized.error_fields == []
    assert any("Optional fields" in issue for issue in normalized.reference_issues)


def test_error_fields_can_be_cleared_before_replace_mode():
    card = CompetitorDetailCard(
        competitor="BetaSuite",
        growth_signals=ResearchField(value="Unsupported global growth claim.", confidence=0.7),
    )

    _clear_error_fields(card, ["growth_signals"], allowed=["growth_signals"])

    assert card.growth_signals.value == ""


def test_competitor_gap_queries_are_field_specific():
    competitor = CandidateCompetitor(name="BetaSuite")

    queries = _competitor_gap_queries(competitor, ["growth_signals", "risks"], 1)

    assert queries[0].query == "BetaSuite 财报 收入 用户 增长 下滑 2023 2024"
    assert queries[1].query == "BetaSuite 投诉 风险 监管 争议 售后 问题"


@pytest.mark.asyncio
async def test_collect_detail_cards_expands_three_templates_per_competitor_and_three_industry_queries(tmp_path):
    searched_queries: list[str] = []

    async def fake_structured(schema, messages):
        if schema is DetailQueryPlan:
            return _query_plan()
        if schema is CardPatchOutput:
            return CardPatchOutput()
        if schema is CardQAResult:
            return CardQAResult(passed=True, needs_retry=False)
        if schema is CardGapQueryOutput:
            return CardGapQueryOutput()
        raise AssertionError(f"Unexpected schema {schema}")

    async def fake_search(search_plan, request, settings, *, task_id, benchmark, competitors, max_sources=12, seen_url_keys=None, should_cancel=None):
        searched_queries.extend(query.query for query in search_plan.queries)
        query = search_plan.queries[0]
        source = RawSource(
            id=f"raw_{len(searched_queries)}",
            url=f"https://example.com/{len(searched_queries)}",
            title=query.query,
            content=f"{query.query} pricing features business model growth reviews risks market size regulation technology",
            source_type="search",
            metadata={"competitor": query.competitor or "", "chunk_id": f"chunk_{len(searched_queries)}"},
        )
        return SourceProviderResult(provider="search", sources=[source], signals=[], cache_hit=False)

    await collect_detail_cards(
        model_available=True,
        structured_call=fake_structured,
        search_collect=fake_search,
        benchmark=_benchmark(),
        selected_set=_selected_set("BetaSuite", "GammaDesk"),
        request=CreateTaskRequest(target_product="Acme Workspace", market="collaboration", product_category="collaboration software"),
        settings=Settings(sqlite_path=tmp_path / "app.db", json_storage_dir=tmp_path / "json"),
        task_id="task_cards",
        existing_sources=[],
    )

    assert set(searched_queries) == {
        "BetaSuite pricing features",
        "BetaSuite business model growth",
        "BetaSuite reviews risks",
        "GammaDesk pricing features",
        "GammaDesk business model growth",
        "GammaDesk reviews risks",
        "collaboration market size",
        "collaboration growth trend",
        "collaboration regulation technology",
    }


@pytest.mark.asyncio
async def test_collect_detail_cards_patches_chunks_in_order_with_value_only_snapshot(tmp_path):
    patch_payloads: list[dict] = []
    search_calls = 0

    async def fake_structured(schema, messages):
        if schema is DetailQueryPlan:
            return _query_plan()
        if schema is CardQAResult:
            return CardQAResult(passed=True, needs_retry=False)
        if schema is CardGapQueryOutput:
            return CardGapQueryOutput()
        if schema is CardPatchOutput:
            payload = json.loads(messages[-1][1])
            if "competitor" not in payload:
                return CardPatchOutput()
            patch_payloads.append(payload)
            current_values = payload["current_card_values"]
            assert all(isinstance(value, str) for value in current_values.values())
            assert "confidence" not in json.dumps(current_values)
            assert "source_refs" not in json.dumps(current_values)
            if len(patch_payloads) == 1:
                assert current_values == {}
                return CardPatchOutput(
                    fields={"pricing": CardFieldPatch(value="BetaSuite has paid subscriptions.", confidence=0.88)},
                    missing_fields=["positioning"],
                )
            if len(patch_payloads) == 2:
                assert current_values == {"pricing": "BetaSuite has paid subscriptions."}
                return CardPatchOutput(
                    fields={"positioning": CardFieldPatch(value="BetaSuite targets collaboration teams.", confidence=0.84)},
                    missing_fields=[],
                )
            return CardPatchOutput()
        raise AssertionError(f"Unexpected schema {schema}")

    async def fake_search(search_plan, request, settings, *, task_id, benchmark, competitors, max_sources=12, seen_url_keys=None, should_cancel=None):
        nonlocal search_calls
        search_calls += 1
        query = search_plan.queries[0]
        if query.competitor and search_calls == 1:
            return SourceProviderResult(
                provider="search",
                sources=[
                    RawSource(
                        id="raw_1",
                        url="https://example.com/pricing",
                        title="Pricing",
                        content="BetaSuite has paid subscriptions.",
                        source_type="search",
                        metadata={"competitor": query.competitor, "chunk_id": "chunk_1"},
                    ),
                    RawSource(
                        id="raw_2",
                        url="https://example.com/about",
                        title="About",
                        content="BetaSuite targets collaboration teams.",
                        source_type="search",
                        metadata={"competitor": query.competitor, "chunk_id": "chunk_2"},
                    ),
                ],
                signals=[],
                cache_hit=False,
            )
        return SourceProviderResult(provider="search", sources=[], signals=[], cache_hit=False)

    result = await collect_detail_cards(
        model_available=True,
        structured_call=fake_structured,
        search_collect=fake_search,
        benchmark=_benchmark(),
        selected_set=_selected_set("BetaSuite"),
        request=CreateTaskRequest(target_product="Acme Workspace", market="collaboration", product_category="collaboration software"),
        settings=Settings(sqlite_path=tmp_path / "app.db", json_storage_dir=tmp_path / "json"),
        task_id="task_patch_order",
        existing_sources=[],
    )

    assert [payload["chunk"]["chunk_id"] for payload in patch_payloads[:2]] == ["chunk_1", "chunk_2"]
    clean_card = result.competitor_detail_cards_clean[0]
    assert clean_card.fields["pricing"] == "BetaSuite has paid subscriptions."
    assert clean_card.fields["positioning"] == "BetaSuite targets collaboration teams."


@pytest.mark.asyncio
async def test_collect_detail_cards_qa_uses_value_and_quote_then_limits_gap_rounds(tmp_path):
    qa_payloads: list[dict] = []
    gap_payloads: list[dict] = []

    async def fake_structured(schema, messages):
        if schema is DetailQueryPlan:
            return _query_plan()
        if schema is CardPatchOutput:
            return CardPatchOutput(
                fields={"pricing": CardFieldPatch(value="BetaSuite has paid subscriptions.", confidence=0.83)},
                missing_fields=["risks"],
            )
        if schema is CardQAResult:
            payload = json.loads(messages[-1][1])
            if payload["object_type"] == "industry":
                return CardQAResult(passed=True, needs_retry=False)
            qa_payloads.append(payload)
            for field_payload in payload["card"]["fields"].values():
                assert set(field_payload) == {"value", "source_refs"}
                for ref in field_payload["source_refs"]:
                    assert set(ref) == {"quote", "url"}
            return CardQAResult(passed=False, missing_fields=["risks"], needs_retry=True, reason="Need risks.")
        if schema is CardGapQueryOutput:
            payload = json.loads(messages[-1][1])
            gap_payloads.append(payload)
            return CardGapQueryOutput(
                queries=[SearchQuery(query="risk complaints", expected_fields=payload["missing_fields"], priority=1)]
            )
        raise AssertionError(f"Unexpected schema {schema}")

    async def fake_search(search_plan, request, settings, *, task_id, benchmark, competitors, max_sources=12, seen_url_keys=None, should_cancel=None):
        query = search_plan.queries[0]
        return SourceProviderResult(
            provider="search",
            sources=[
                RawSource(
                    id=f"raw_{len(qa_payloads)}_{len(gap_payloads)}",
                    url="https://example.com/source",
                    title="Source",
                    content="BetaSuite pricing and risk complaints.",
                    source_type="search",
                    metadata={"competitor": query.competitor or "", "chunk_id": "chunk"},
                )
            ],
            signals=[],
            cache_hit=False,
        )

    await collect_detail_cards(
        model_available=True,
        structured_call=fake_structured,
        search_collect=fake_search,
        benchmark=_benchmark(),
        selected_set=_selected_set("BetaSuite"),
        request=CreateTaskRequest(target_product="Acme Workspace", market="collaboration", product_category="collaboration software"),
        settings=Settings(
            sqlite_path=tmp_path / "app.db",
            json_storage_dir=tmp_path / "json",
            card_search_iterations_per_object=3,
        ),
        task_id="task_qa_gap",
        existing_sources=[],
    )

    assert len(gap_payloads) == 3
    assert all(payload["missing_fields"] == ["risks"] for payload in gap_payloads)
    assert len(qa_payloads) == 4
