from __future__ import annotations

import pytest

from myagent.backend.app.agents.deep_report import build_deep_dive_markdown, has_required_deep_dive_sections
from myagent.backend.app.agents.workflow import AgentWorkflow
from myagent.backend.app.events import EventBus
from myagent.backend.app.models import CompetitorReport, QAResult


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeDeepDiveModel:
    def __init__(self, content: str | Exception):
        self.content = content
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        if isinstance(self.content, Exception):
            raise self.content
        return FakeMessage(self.content)


@pytest.fixture
def agent_workflow(app_settings, memory_storage):
    return AgentWorkflow(app_settings, memory_storage, EventBus(memory_storage))


@pytest.mark.asyncio
async def test_writer_agent_builds_deep_dive_without_model(agent_workflow, sample_request):
    from conftest import make_report

    analysis = make_report("task_writer", sample_request)
    analysis.deep_dive_markdown = ""
    state = {
        "task_id": "task_writer",
        "request": sample_request.model_dump(mode="json"),
        "analysis_report": analysis.model_dump(mode="json"),
    }

    updates = await agent_workflow.writer_agent(state)

    report = CompetitorReport.model_validate(updates["report"])
    assert report.task_id == "task_writer"
    assert has_required_deep_dive_sections(report.deep_dive_markdown)
    assert report.evidence == analysis.evidence


@pytest.mark.asyncio
async def test_writer_agent_uses_chapter_model_and_preserves_evidence_and_qa_history(
    agent_workflow, sample_request
):
    from conftest import make_report

    analysis = make_report("task_writer_model", sample_request)
    analysis.qa_history = [QAResult(passed=True, score=0.95)]
    model_markdown = build_deep_dive_markdown(analysis, sample_request)
    agent_workflow.model = FakeDeepDiveModel(model_markdown)
    state = {
        "task_id": "task_writer_model",
        "request": sample_request.model_dump(mode="json"),
        "analysis_report": analysis.model_dump(mode="json"),
    }

    updates = await agent_workflow.writer_agent(state)

    report = CompetitorReport.model_validate(updates["report"])
    assert agent_workflow.model.messages is not None
    assert report.task_id == "task_writer_model"
    assert report.target_product == sample_request.target_product
    assert report.evidence == analysis.evidence
    assert report.qa_history == analysis.qa_history
    assert has_required_deep_dive_sections(report.deep_dive_markdown)
    assert report.report_chapters
    assert "#ref-" in report.deep_dive_markdown
    assert "引用索引" in report.deep_dive_markdown
    assert report.report_references


@pytest.mark.asyncio
async def test_writer_agent_falls_back_when_chapter_model_fails(agent_workflow, sample_request):
    from conftest import make_report

    analysis = make_report("task_writer_error", sample_request)
    analysis.deep_dive_markdown = ""
    agent_workflow.model = FakeDeepDiveModel(RuntimeError("deep dive unavailable"))
    state = {
        "task_id": "task_writer_error",
        "request": sample_request.model_dump(mode="json"),
        "analysis_report": analysis.model_dump(mode="json"),
    }

    updates = await agent_workflow.writer_agent(state)

    report = CompetitorReport.model_validate(updates["report"])
    assert "deep dive unavailable" in report.risk_notice
    assert has_required_deep_dive_sections(report.deep_dive_markdown)


@pytest.mark.asyncio
async def test_build_deep_dive_markdown_uses_chapter_model(agent_workflow, sample_request):
    from conftest import make_report

    report = make_report("task_deep_model", sample_request)
    valid_markdown = build_deep_dive_markdown(report, sample_request)
    agent_workflow.model = FakeDeepDiveModel(valid_markdown)

    markdown = await agent_workflow._build_deep_dive_markdown("task_deep_model", sample_request, report)

    assert has_required_deep_dive_sections(markdown)
    assert agent_workflow.model.messages is not None


@pytest.mark.asyncio
async def test_build_deep_dive_markdown_falls_back_for_invalid_or_failed_model(agent_workflow, sample_request):
    from conftest import make_report

    report = make_report("task_deep_invalid", sample_request)
    agent_workflow.model = FakeDeepDiveModel("# Incomplete")

    invalid_markdown = await agent_workflow._build_deep_dive_markdown("task_deep_invalid", sample_request, report)

    assert has_required_deep_dive_sections(invalid_markdown)

    agent_workflow.model = FakeDeepDiveModel(RuntimeError("markdown generation failed"))
    failed_markdown = await agent_workflow._build_deep_dive_markdown("task_deep_invalid", sample_request, report)

    assert has_required_deep_dive_sections(failed_markdown)
    assert "markdown generation failed" in report.risk_notice
