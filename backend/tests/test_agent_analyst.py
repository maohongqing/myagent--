from __future__ import annotations

import pytest

from myagent.backend.app.agents.workflow import AgentWorkflow
from myagent.backend.app.events import EventBus
from myagent.backend.app.models import CompetitorReport


@pytest.fixture
def agent_workflow(app_settings, memory_storage):
    return AgentWorkflow(app_settings, memory_storage, EventBus(memory_storage))


@pytest.mark.asyncio
async def test_analyst_agent_builds_report_from_tool_results_without_model(agent_workflow, sample_request):
    fixture = make_fixture_report("task_analyst", sample_request)
    state = {
        "task_id": "task_analyst",
        "request": sample_request.model_dump(mode="json"),
        "workflow_phase": "phase_4_analysis",
        "benchmark": fixture.analysis_benchmark.model_dump(mode="json"),
        "selected_competitor_set": fixture.selected_competitor_set.model_dump(mode="json"),
        "tool_plan": fixture.tool_plan.model_dump(mode="json"),
        "collection_plan": fixture.collection_plan.model_dump(mode="json"),
        "tool_results": [item.model_dump(mode="json") for item in fixture.tool_results],
        "tool_sandboxes": [item.model_dump(mode="json") for item in fixture.tool_sandboxes],
    }

    updates = await agent_workflow.analyst_agent(state)

    report = CompetitorReport.model_validate(updates["analysis_report"])
    assert report.task_id == "task_analyst"
    assert report.evidence == []
    assert report.report_references == []
    assert report.source_provider_results == []
    assert report.tool_results


@pytest.mark.asyncio
async def test_analyst_agent_passthrough_does_not_call_model(monkeypatch, agent_workflow, sample_request):
    fixture = make_fixture_report("task_analyst_model", sample_request)
    fixture.tool_results[0].reference_id = None
    fixture.tool_results[0].input_reference_ids = ["card.competitor.betasuite.pricing"]
    fixture.tool_results[0].evidence_ids = ["ev_beta"]
    fixture.tool_results[0].claims[0].evidence_ids = ["ev_beta"]
    fixture.tool_results[0].claims[0].reference_ids = ["card.competitor.betasuite.pricing"]
    agent_workflow.model = object()

    async def fake_llm_structured(task_id, node, schema, messages, *, llm_lane="main"):
        raise AssertionError("analyst_agent should not call final summary LLM")

    monkeypatch.setattr(agent_workflow, "_llm_structured", fake_llm_structured)
    state = {
        "task_id": "task_analyst_model",
        "request": sample_request.model_dump(mode="json"),
        "workflow_phase": "phase_4_analysis",
        "benchmark": fixture.analysis_benchmark.model_dump(mode="json"),
        "selected_competitor_set": fixture.selected_competitor_set.model_dump(mode="json"),
        "tool_plan": fixture.tool_plan.model_dump(mode="json"),
        "collection_plan": fixture.collection_plan.model_dump(mode="json"),
        "raw_sources": [{"content": "raw source text must not be passed"}],
        "evidence": [{"id": "ev_beta", "excerpt": "raw evidence must not be passed"}],
        "report_references": [{"id": "ref_beta", "summary": "reference must not be passed"}],
        "source_provider_results": [{"provider": "cache", "sources": [{"content": "provider source must not be passed"}]}],
        "tool_results": [item.model_dump(mode="json") for item in fixture.tool_results],
        "tool_sandboxes": [item.model_dump(mode="json") for item in fixture.tool_sandboxes],
    }

    updates = await agent_workflow.analyst_agent(state)

    report = CompetitorReport.model_validate(updates["analysis_report"])
    assert report.task_id == "task_analyst_model"
    assert report.target_product == sample_request.target_product
    assert report.evidence == []
    assert report.report_references == []
    assert report.source_provider_results == []
    assert report.tool_results
    assert report.tool_results[0].input_reference_ids == ["card.competitor.betasuite.pricing"]
    assert report.tool_results[0].claims[0].evidence_ids == ["ev_beta"]


@pytest.mark.asyncio
async def test_analyst_agent_preserves_tool_report_without_model_fallback(monkeypatch, agent_workflow, sample_request):
    fixture = make_fixture_report("task_analyst_error", sample_request)
    agent_workflow.model = object()

    async def failing_llm_structured(*_args, **_kwargs):
        raise AssertionError("analyst_agent should not call final summary LLM")

    monkeypatch.setattr(agent_workflow, "_llm_structured", failing_llm_structured)
    state = {
        "task_id": "task_analyst_error",
        "request": sample_request.model_dump(mode="json"),
        "workflow_phase": "phase_4_analysis",
        "benchmark": fixture.analysis_benchmark.model_dump(mode="json"),
        "selected_competitor_set": fixture.selected_competitor_set.model_dump(mode="json"),
        "tool_plan": fixture.tool_plan.model_dump(mode="json"),
        "collection_plan": fixture.collection_plan.model_dump(mode="json"),
        "tool_results": [item.model_dump(mode="json") for item in fixture.tool_results],
        "tool_sandboxes": [item.model_dump(mode="json") for item in fixture.tool_sandboxes],
    }

    updates = await agent_workflow.analyst_agent(state)

    report = CompetitorReport.model_validate(updates["analysis_report"])
    assert report.evidence == []
    assert report.report_references == []
    assert report.tool_results
    assert report.risk_notice is None


def make_fixture_report(task_id, request):
    from conftest import make_report

    report = make_report(task_id=task_id, request=request)
    report.deep_dive_markdown = ""
    return report
