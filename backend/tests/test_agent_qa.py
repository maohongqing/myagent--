from __future__ import annotations

import pytest

from myagent.backend.app.agents.workflow import AgentWorkflow
from myagent.backend.app.agents import workflow as workflow_module
from myagent.backend.app.agents.state_tree import selected_set_to_phase_2
from myagent.backend.app.events import EventBus
from myagent.backend.app.models import AnalysisBenchmark, AnalysisFinding, CandidateCompetitor, CompetitorBriefProfile, CompetitorReport, QAResult, RankFactors, ResearchField, RevisionRequest, SourceRef


@pytest.fixture
def agent_workflow(app_settings, memory_storage):
    return AgentWorkflow(app_settings, memory_storage, EventBus(memory_storage))


def _phase_2_state(selected_count: int) -> dict:
    type_cycle = ["head_direct", "challenger", "indirect_cross"]
    selected_competitors = [
        {
            "competitor_name": f"Competitor {index}",
            "competitor_type": type_cycle[(index - 1) % len(type_cycle)],
            "selection_reason": "Selected from direct market overlap and clear positioning evidence.",
            "rank_score": 0.9,
            "source_refs": [{"url": "https://example.com", "title": "Example", "quote": "Evidence"}],
            "source_urls": ["https://example.com"],
            "source_titles": ["Example"],
        }
        for index in range(1, selected_count + 1)
    ]
    return {
        "task_id": f"task_phase2_qa_{selected_count}",
        "state_tree": {
            "phase_1_intent": {
                "product_lifecycle": "growth",
                "analysis_purpose": "decision_support",
                "core_problem_to_solve": "Decide which collaboration competitors to benchmark.",
                "specific_goal": "Select a balanced direct competitor lineup.",
            },
            "phase_2_targets": {
                "selected_competitors": selected_competitors,
                "selection_logic": "Use exactly three competitors with objective selection reasons.",
            },
            "quality_gates": {},
        },
        "revision_count": 0,
        "max_revision_loops": 1,
    }


@pytest.mark.asyncio
async def test_qa_agent_appends_passing_result(agent_workflow, memory_storage, sample_request):
    from conftest import make_report

    report = make_report("task_qa_pass", sample_request)
    state = {
        "task_id": "task_qa_pass",
        "report": report.model_dump(mode="json"),
        "qa_history": [],
        "revision_count": 0,
        "max_revision_loops": 1,
    }

    updates = await agent_workflow.qa_agent(state)

    updated_report = CompetitorReport.model_validate(updates["report"])
    assert updated_report.qa_history
    assert updated_report.qa_history[-1].passed is True
    assert "revision_count" not in updates
    assert memory_storage.list_events("task_qa_pass")[-1].status == "completed"


@pytest.mark.asyncio
async def test_phase_2_qa_checkpoint_passes_for_required_competitor_count(agent_workflow, memory_storage):
    state = _phase_2_state(selected_count=3)

    updates = await agent_workflow._qa_checkpoint_1(state)

    checkpoint = updates["state_tree"]["quality_gates"]["checkpoint_1"]
    assert checkpoint["inspection_status"] == "pass"
    assert checkpoint["issues_found"] == []
    assert checkpoint["feedback"] is None
    assert updates["workflow_phase"] == "phase_3_strategy"
    assert updates["revision_count"] == 0
    assert memory_storage.list_events("task_phase2_qa_3")[-1].status == "completed"


@pytest.mark.asyncio
async def test_phase_2_qa_checkpoint_renormalizes_duplicate_aliases(agent_workflow, memory_storage, sample_request):
    benchmark = AnalysisBenchmark(
        target_product="Acme Video",
        raw_description="Video platform analysis",
        lifecycle_stage="launched",
        analysis_purpose="decision_support",
        analysis_goal="Select competitors.",
    )
    def sourced_candidate(name: str, competitor_type: str, score: float) -> CandidateCompetitor:
        ref = SourceRef(url=f"https://example.com/{name}", title=f"{name} source", quote=f"{name} evidence")
        return CandidateCompetitor(
            name=name,
            competitor_type=competitor_type,
            source_refs=[ref],
            source_urls=[ref.url],
            source_titles=[ref.title],
            rank_factors=RankFactors(heat=score, growth=score, similarity=score),
        )

    candidates = [
        sourced_candidate("B站", "head_direct", 0.9),
        sourced_candidate("哔哩哔哩（B站）", "head_direct", 0.85),
        sourced_candidate("Slack", "challenger", 0.8),
        sourced_candidate("Notion", "indirect_cross", 0.75),
    ]
    selected_set = workflow_module._select_top_competitors_by_category(benchmark, candidates)
    state = {
        "task_id": "task_phase2_qa_aliases",
        "request": sample_request.model_dump(mode="json"),
        "benchmark": benchmark.model_dump(mode="json"),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "selected_competitor_set": selected_set.model_dump(mode="json"),
        "competitor_brief_profiles": [],
        "state_tree": {
            "phase_1_intent": {"analysis_purpose": "decision_support"},
            "phase_2_targets": selected_set_to_phase_2(selected_set),
            "quality_gates": {},
        },
        "revision_count": 0,
        "max_revision_loops": 1,
    }

    updates = await agent_workflow._qa_checkpoint_1(state)

    selected_names = [item["name"] for item in updates["selected_competitor_set"]["selected"]]
    checkpoint_names = [
        item["competitor_name"]
        for item in updates["state_tree"]["phase_2_targets"]["selected_competitors"]
    ]
    assert selected_names == ["B站", "Slack", "Notion"]
    assert checkpoint_names == selected_names
    assert updates["workflow_phase"] == "phase_3_strategy"
    assert updates["state_tree"]["quality_gates"]["checkpoint_1"]["inspection_status"] == "pass"
    assert updates["candidate_qa_change_log"]["merged_candidates"]


@pytest.mark.asyncio
async def test_phase_2_qa_checkpoint_renormalizes_app_suffix_aliases(agent_workflow, memory_storage, sample_request):
    benchmark = AnalysisBenchmark(
        target_product="Figure resale",
        raw_description="Figure resale platform analysis",
        lifecycle_stage="launched",
        analysis_purpose="decision_support",
        analysis_goal="Select competitors.",
    )

    def sourced_candidate(name: str, competitor_type: str, score: float) -> CandidateCompetitor:
        ref = SourceRef(url=f"https://example.com/{name}", title=f"{name} source", quote=f"{name} evidence")
        return CandidateCompetitor(
            name=name,
            competitor_type=competitor_type,
            source_refs=[ref],
            source_urls=[ref.url],
            source_titles=[ref.title],
            rank_factors=RankFactors(heat=score, growth=score, similarity=score),
        )

    candidates = [
        sourced_candidate("闲鱼", "head_direct", 0.95),
        sourced_candidate("闲鱼App", "indirect_cross", 0.9),
        sourced_candidate("得物", "challenger", 0.85),
        sourced_candidate("淘宝", "indirect_cross", 0.8),
    ]
    selected_set = workflow_module._select_top_competitors_by_category(benchmark, candidates)
    state = {
        "task_id": "task_phase2_qa_app_aliases",
        "request": sample_request.model_dump(mode="json"),
        "benchmark": benchmark.model_dump(mode="json"),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "selected_competitor_set": selected_set.model_dump(mode="json"),
        "competitor_brief_profiles": [],
        "state_tree": {
            "phase_1_intent": {"analysis_purpose": "decision_support"},
            "phase_2_targets": selected_set_to_phase_2(selected_set),
            "quality_gates": {},
        },
        "revision_count": 0,
        "max_revision_loops": 1,
    }

    updates = await agent_workflow._qa_checkpoint_1(state)

    selected_names = [item["name"] for item in updates["selected_competitor_set"]["selected"]]
    checkpoint_names = [
        item["competitor_name"]
        for item in updates["state_tree"]["phase_2_targets"]["selected_competitors"]
    ]
    assert "闲鱼App" not in selected_names
    assert selected_names.count("闲鱼") == 1
    assert checkpoint_names == selected_names
    assert updates["state_tree"]["quality_gates"]["checkpoint_1"]["inspection_status"] == "pass"
    assert any(
        item["from_name"] == "闲鱼App" and item["to_name"] == "闲鱼"
        for item in updates["candidate_qa_change_log"]["merged_candidates"]
    )


@pytest.mark.asyncio
async def test_phase_2_qa_checkpoint_rejects_wrong_selection_count(agent_workflow, memory_storage):
    state = _phase_2_state(selected_count=2)

    updates = await agent_workflow._qa_checkpoint_1(state)

    checkpoint = updates["state_tree"]["quality_gates"]["checkpoint_1"]
    assert checkpoint["inspection_status"] == "reject"
    assert any("Expected exactly 3" in issue for issue in checkpoint["issues_found"])
    assert updates["workflow_phase"] == "phase_2_analyze"
    assert updates["revision_count"] == 1
    assert memory_storage.list_events("task_phase2_qa_2")[-1].status == "revision"


def test_market_growth_profile_qa_flags_strong_fact_without_refs(agent_workflow):
    candidate = CandidateCompetitor(name="BetaSuite", description="A collaboration product.")
    profile = CompetitorBriefProfile(
        competitor="BetaSuite",
        market_heat=ResearchField(value="BetaSuite has 10 million users and strong market popularity.", confidence=0.6),
        development_status=ResearchField(value="BetaSuite raised Series B funding and grew 200%.", confidence=0.6),
    )

    qa = agent_workflow._rule_market_growth_qa(candidate, profile)

    assert qa.needs_rework is True
    assert qa.has_missing_fields is True
    assert qa.has_suspicious_fields is True
    assert qa.rework_queries_needed is True


@pytest.mark.asyncio
async def test_qa_agent_requests_revision_for_invalid_report(agent_workflow, memory_storage, sample_request):
    from conftest import make_report

    report = make_report("task_qa_revision", sample_request)
    report.findings = [
        AnalysisFinding(
            category="strategy",
            title="Unsupported finding",
            conclusion="This finding has no evidence and should trigger QA revision.",
            evidence_ids=[],
        )
    ]
    report.deep_dive_markdown = "# incomplete"
    state = {
        "task_id": "task_qa_revision",
        "report": report.model_dump(mode="json"),
        "qa_history": [],
        "revision_count": 0,
        "max_revision_loops": 2,
    }

    updates = await agent_workflow.qa_agent(state)

    updated_report = CompetitorReport.model_validate(updates["report"])
    assert updates["revision_count"] == 1
    assert updated_report.qa_history[-1].passed is False
    assert updated_report.qa_history[-1].revision_request is not None
    assert memory_storage.list_events("task_qa_revision")[-1].status == "revision"


@pytest.mark.asyncio
async def test_qa_agent_finishes_with_risk_notice_after_revision_budget(agent_workflow, sample_request):
    from conftest import make_report

    report = make_report("task_qa_budget", sample_request)
    report.competitors = []
    report.deep_dive_markdown = ""
    state = {
        "task_id": "task_qa_budget",
        "report": report.model_dump(mode="json"),
        "qa_history": [],
        "revision_count": 1,
        "max_revision_loops": 1,
    }

    updates = await agent_workflow.qa_agent(state)

    updated_report = CompetitorReport.model_validate(updates["report"])
    assert updated_report.qa_history[-1].passed is False
    assert updated_report.qa_history[-1].revision_request is None
    assert updated_report.risk_notice


def test_route_after_qa_returns_target_agent_or_end(agent_workflow):
    revision = QAResult(
        passed=False,
        score=0.4,
        revision_request=RevisionRequest(
            target_agent="writer_agent",
            reason="missing sections",
            missing_fields=["deep_dive_markdown"],
        ),
    )
    passed = QAResult(passed=True, score=0.95)

    assert agent_workflow.route_after_qa({"qa_history": [revision.model_dump(mode="json")]}) == "writer_agent"
    assert agent_workflow.route_after_qa({"qa_history": [passed.model_dump(mode="json")]}) == "end"
    assert agent_workflow.route_after_qa({"qa_history": []}) == "end"
