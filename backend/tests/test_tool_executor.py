import pytest

from myagent.backend.app.agents import tool_executor as tool_executor_module
from myagent.backend.app.agents.tool_executor import execute_tool, execute_tools, tool_results_to_sandboxes
from myagent.backend.app.agents.tool_registry import get_tool_spec
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CompetitorDetailCardClean,
    IndustryCardClean,
    ReportReference,
    SelectedCompetitorSet,
    ToolClaim,
    ToolExecutionResult,
    ToolPlan,
)


def _benchmark() -> AnalysisBenchmark:
    return AnalysisBenchmark(
        target_product="Acme Workspace",
        raw_description="Collaboration product",
        lifecycle_stage="concept",
        analysis_purpose="decision_support",
        analysis_goal="Analyze competitors.",
    )


def _competitor_card(name: str = "BetaSuite") -> CompetitorDetailCardClean:
    slug = name.lower()
    fields = {
        "positioning": f"{name} targets collaboration teams.",
        "target_users": f"{name} serves product and operations teams.",
        "core_scenarios": f"{name} supports review and approval workflows.",
        "core_features": f"{name} offers docs, chat, and workflow automation.",
        "key_parameters": f"{name} supports 50 integrations.",
        "pricing": f"{name} offers a free plan and paid subscriptions.",
        "business_model": f"{name} uses seat-based SaaS pricing.",
        "channels": f"{name} sells through self-serve and partners.",
        "growth_signals": f"{name} is hiring sales roles.",
        "funding_or_org_signals": f"{name} recently raised a Series B.",
        "technology_or_product_features": f"{name} ships AI workflow features.",
        "risks": f"{name} has enterprise procurement risk.",
        "differentiation": f"{name} differentiates on automation depth.",
        "user_feedback": f"{name} users complain about setup complexity.",
        "recent_updates": f"{name} launched mobile approvals.",
        "unrelated_field": f"{name} should not be sent to tools.",
    }
    return CompetitorDetailCardClean(
        competitor=name,
        fields=fields,
        reference_ids={key: f"card.competitor.{slug}.{key}" for key in fields},
    )


def _industry_card() -> IndustryCardClean:
    return IndustryCardClean(
        fields={
            "market_situation.market_size": "The collaboration market is growing.",
            "external_environment.technology_trends": "AI workflow adoption is rising.",
        },
        reference_ids={
            "market_situation.market_size": "card.industry.market_situation.market_size",
            "external_environment.technology_trends": "card.industry.external_environment.technology_trends",
        },
    )


def _references(*reference_ids: str) -> list[ReportReference]:
    return [
        ReportReference(
            id=reference_id,
            title=reference_id,
            reference_type="clean_card",
            anchor=f"ref-{reference_id.replace('.', '-')}",
            summary=f"Summary for {reference_id} with https://example.com/raw and screenshots/raw.png",
            payload={"field": reference_id},
        )
        for reference_id in reference_ids
    ]


@pytest.mark.asyncio
async def test_tool_executor_fallback_outputs_claims_with_references():
    spec = get_tool_spec("swot")
    benchmark = _benchmark()
    card = CompetitorDetailCardClean(
        competitor="BetaSuite",
        fields={"pricing": "BetaSuite offers a free plan and paid subscriptions."},
        reference_ids={"pricing": "card.competitor.betasuite.pricing"},
    )
    reference_id = card.reference_ids["pricing"]

    result = await execute_tool(
        None,
        spec,
        benchmark,
        [CandidateCompetitor(name="BetaSuite")],
        scope_type="competitor",
        competitor_name="BetaSuite",
        input_reference_ids=[reference_id],
        clean_context={"competitor_detail_card": card.model_dump(mode="json")},
    )

    assert result.generated_by == "fallback"
    assert result.claims
    assert result.claims[0].reference_ids == [reference_id]
    assert result.claims[0].evidence_ids == []
    assert result.evidence_ids == []
    assert result.data["strengths"]


def test_tool_results_to_sandboxes_preserves_claims():
    result = ToolExecutionResult(
        tool_name="SWOT",
        canonical_tool_name="swot",
        schema_name="swot_schema",
        input_reference_ids=["card.competitor.betasuite.pricing"],
        claims=[
            ToolClaim(
                title="Supported claim",
                claim="Claim with a card reference.",
                reference_ids=["card.competitor.betasuite.pricing"],
            )
        ],
        evidence_ids=[],
    )

    sandboxes = tool_results_to_sandboxes([result])

    assert sandboxes[0].claims[0].reference_ids == ["card.competitor.betasuite.pricing"]
    assert sandboxes[0].claims[0].evidence_ids == []
    assert sandboxes[0].result is not None


def test_tool_jobs_trims_single_competitor_clean_card_fields():
    spec = get_tool_spec("feature_breakdown")
    card = _competitor_card()

    jobs = tool_executor_module._tool_jobs(spec, [CandidateCompetitor(name="BetaSuite")], [card], None, [])

    clean_card = jobs[0]["clean_context"]["competitor_detail_card"]
    expected_fields = {
        "core_features",
        "key_parameters",
        "recent_updates",
        "technology_or_product_features",
        "user_feedback",
    }
    assert set(clean_card["fields"]) == expected_fields
    assert set(clean_card["reference_ids"]) == expected_fields
    assert set(jobs[0]["input_reference_ids"]) == {card.reference_ids[field] for field in expected_fields}
    assert "unrelated_field" not in clean_card["fields"]


def test_tool_jobs_trims_comparison_clean_cards_fields():
    spec = get_tool_spec("comparison")
    cards = [_competitor_card("BetaSuite"), _competitor_card("GammaDesk")]

    jobs = tool_executor_module._tool_jobs(
        spec,
        [CandidateCompetitor(name="BetaSuite"), CandidateCompetitor(name="GammaDesk")],
        cards,
        None,
        [],
    )

    clean_cards = jobs[0]["clean_context"]["competitor_detail_cards"]
    expected_fields = {
        "positioning",
        "core_features",
        "pricing",
        "business_model",
        "growth_signals",
        "risks",
        "differentiation",
    }
    assert all(set(card["fields"]) == expected_fields for card in clean_cards)
    assert all("unrelated_field" not in card["fields"] for card in clean_cards)
    assert len(jobs[0]["input_reference_ids"]) == len(cards) * len(expected_fields)


def test_tool_jobs_trims_industry_tool_competitor_summaries():
    spec = get_tool_spec("pest_analysis")
    card = _competitor_card()
    industry = _industry_card()

    jobs = tool_executor_module._tool_jobs(
        spec,
        [CandidateCompetitor(name="BetaSuite")],
        [card],
        industry,
        [],
    )

    clean_context = jobs[0]["clean_context"]
    expected_competitor_fields = {
        "growth_signals",
        "risks",
        "technology_or_product_features",
        "funding_or_org_signals",
    }
    assert set(clean_context["industry_card"]["fields"]) == set(industry.fields)
    assert set(clean_context["competitor_summaries"][0]["fields"]) == expected_competitor_fields
    assert set(clean_context["competitor_summaries"][0]["reference_ids"]) == expected_competitor_fields
    assert set(jobs[0]["input_reference_ids"]) == {
        *industry.reference_ids.values(),
        *[card.reference_ids[field] for field in expected_competitor_fields],
    }


def test_tool_jobs_unknown_tool_uses_default_clean_card_fields():
    spec = get_tool_spec("custom_method")
    card = _competitor_card()

    jobs = tool_executor_module._tool_jobs(spec, [CandidateCompetitor(name="BetaSuite")], [card], None, [])

    clean_card = jobs[0]["clean_context"]["competitor_detail_cards"][0]
    expected_fields = {
        "positioning",
        "core_features",
        "pricing",
        "business_model",
        "growth_signals",
        "risks",
    }
    assert set(clean_card["fields"]) == expected_fields
    assert set(clean_card["reference_ids"]) == expected_fields
    assert len(jobs[0]["input_reference_ids"]) == len(expected_fields)


@pytest.mark.asyncio
async def test_execute_tools_uses_fixed_clean_card_scopes():
    benchmark = _benchmark()
    competitors = [
        CandidateCompetitor(name="BetaSuite", score=0.9),
        CandidateCompetitor(name="GammaDesk", score=0.8),
    ]
    selected_set = SelectedCompetitorSet(
        scoring_profile={
            "purpose": "decision_support",
            "alpha_heat": 0.25,
            "beta_growth": 0.25,
            "gamma_similarity": 0.35,
            "delta_risk": 0.15,
            "type_weights": {},
        },
        candidates=competitors,
        selected=competitors,
        allocation={"head_direct": 1, "challenger": 1},
    )
    cards = [_competitor_card("BetaSuite"), _competitor_card("GammaDesk")]
    industry = _industry_card()

    results = await execute_tools(
        None,
        benchmark,
        selected_set,
        ToolPlan(
            purpose="decision_support",
            focus="decision",
            selected_tools=["swot", "pest_analysis", "strategy_canvas"],
        ),
        competitor_detail_cards_clean=cards,
        industry_card_clean=industry,
        report_references=_references(
            *[reference_id for card in cards for reference_id in card.reference_ids.values()],
            *industry.reference_ids.values(),
        ),
    )

    competitor_results = [item for item in results if item.scope_type == "competitor"]
    industry_results = [item for item in results if item.scope_type == "industry"]
    comparison_results = [item for item in results if item.scope_type == "comparison"]
    assert len(competitor_results) == 2
    assert len(industry_results) == 1
    assert len(comparison_results) == 1
    assert {item.competitor_name for item in competitor_results} == {"BetaSuite", "GammaDesk"}
    assert all(item.input_reference_ids for item in results)
    assert all(item.claims[0].reference_ids for item in results)
    assert all(not item.claims[0].evidence_ids for item in results)
    assert all(not item.evidence_ids for item in results)


@pytest.mark.asyncio
async def test_tool_executor_prompt_omits_old_collection_payload_and_paths(monkeypatch):
    captured: dict[str, str] = {}
    reference_id = "card.competitor.betasuite.pricing"

    async def fake_structured(_model, _schema, messages):
        captured["prompt"] = "\n".join(content for _role, content in messages)
        return ToolExecutionResult(
            tool_name="SWOT",
            canonical_tool_name="swot",
            schema_name="swot_schema",
            input_reference_ids=[reference_id],
            claims=[
                ToolClaim(
                    title="Supported",
                    claim="Supported by the clean card.",
                    reference_ids=[reference_id],
                    evidence_ids=["legacy_raw_id"],
                )
            ],
            evidence_ids=["legacy_raw_id"],
            generated_by="llm",
        )

    monkeypatch.setattr(tool_executor_module, "retry_structured_ainvoke", fake_structured)
    spec = get_tool_spec("swot")
    card = CompetitorDetailCardClean(
        competitor="BetaSuite",
        fields={"pricing": "Free plan. See https://example.com/pricing and screenshots/raw.png"},
        reference_ids={"pricing": reference_id},
    )

    result = await execute_tool(
        object(),
        spec,
        _benchmark(),
        [CandidateCompetitor(name="BetaSuite")],
        scope_type="competitor",
        competitor_name="BetaSuite",
        input_reference_ids=[reference_id],
        clean_context={"competitor_detail_card": card.model_dump(mode="json")},
        available_references=_references(reference_id),
    )

    prompt = captured["prompt"].lower()
    assert result.generated_by == "llm"
    assert result.claims[0].reference_ids == [reference_id]
    assert result.claims[0].evidence_ids == []
    assert result.evidence_ids == []
    assert "search_plan" not in prompt
    assert "evidence" not in prompt
    assert "signals" not in prompt
    assert "https://example.com" not in prompt
    assert "screenshots/raw.png" not in prompt


@pytest.mark.asyncio
async def test_tool_executor_falls_back_when_structured_mock_returns_string(monkeypatch):
    async def fake_structured(_model, _schema, _messages):
        return "not a ToolExecutionResult"

    monkeypatch.setattr(tool_executor_module, "retry_structured_ainvoke", fake_structured)
    spec = get_tool_spec("swot")
    card = _competitor_card()
    reference_id = card.reference_ids["pricing"]

    result = await execute_tool(
        object(),
        spec,
        _benchmark(),
        [CandidateCompetitor(name="BetaSuite")],
        scope_type="competitor",
        competitor_name="BetaSuite",
        input_reference_ids=[reference_id],
        clean_context={"competitor_detail_card": card.model_dump(mode="json")},
    )

    assert isinstance(result, ToolExecutionResult)
    assert result.generated_by == "fallback"
    assert result.model_dump(mode="json")["tool_name"] == spec.name
    assert result.claims[0].reference_ids == [reference_id]
