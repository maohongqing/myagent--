import pytest

from myagent.backend.app.agents.survey import (
    analyze_survey_responses,
    append_survey_analysis_to_markdown,
    build_fallback_survey_design,
)
from myagent.backend.app.models import CreateTaskRequest


@pytest.mark.asyncio
async def test_fallback_survey_design_uses_competitors(sample_request):
    design = build_fallback_survey_design("task_survey", sample_request)

    assert design.task_id == "task_survey"
    assert len(design.questions) >= 6
    assert any("BetaSuite" in question.options for question in design.questions)


@pytest.mark.asyncio
async def test_analyze_survey_responses_from_csv(sample_request):
    design = build_fallback_survey_design("task_survey_csv", sample_request)
    raw = "platform,reason,feedback\nBetaSuite,价格、售后,价格透明但售后需要提升\nBetaSuite,价格,希望交易更安全\n"

    result = await analyze_survey_responses(None, "task_survey_csv", sample_request, design, raw)

    assert result.sample_size == 2
    assert result.key_findings
    assert "用户问卷补充分析" in result.appendix_markdown
    assert result.limitations


@pytest.mark.asyncio
async def test_analyze_survey_responses_rejects_empty(sample_request):
    with pytest.raises(ValueError):
        await analyze_survey_responses(None, "task_empty", sample_request, None, "   ")


def test_append_survey_analysis_replaces_existing_section():
    first = "## 原报告\n\n内容\n\n## 用户问卷补充分析\n\n旧内容\n"
    updated = append_survey_analysis_to_markdown(first, "## 用户问卷补充分析\n\n新内容\n")

    assert "旧内容" not in updated
    assert "新内容" in updated
    assert updated.count("## 用户问卷补充分析") == 1
