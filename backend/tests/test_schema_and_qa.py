from myagent.backend.app.agents.deep_report import DEEP_DIVE_SECTIONS
from myagent.backend.app.agents.fallbacks import build_fallback_report
from myagent.backend.app.agents.prompt_bridge import candidate_existence_qa_system_prompt
from myagent.backend.app.agents.qa import validate_report
from myagent.backend.app.models import AnalysisFinding, CandidateQAResult, CompetitorReport, CreateTaskRequest, EvidenceItem


def test_create_task_request_accepts_extended_fields():
    request = CreateTaskRequest(
        target_product="Feishu",
        raw_description="Feishu is a collaboration product. We want decision support for market positioning.",
        lifecycle_stage="concept",
        analysis_purpose="decision_support",
        analysis_goal="Evaluate market positioning.",
        our_product="Feishu",
        market="collaboration",
        product_category="Collaboration",
        geography="China",
        company_size="enterprise teams",
        competitors=["DingTalk"],
        competitor_profiles=[
            {
                "name": "Notion",
                "category": "indirect",
                "website": "https://www.notion.so",
                "website_copy": "Connected workspace",
                "sales_notes": "Often mentioned in knowledge-base scenarios.",
                "win_loss_notes": "Strong templates and docs experience.",
                "tags": ["docs", "wiki"],
            }
        ],
        enable_screenshots=True,
    )

    assert request.raw_description
    assert request.lifecycle_stage == "concept"
    assert request.analysis_purpose == "decision_support"
    assert request.competitors == ["DingTalk", "Notion"]
    assert request.competitor_profiles[0].website == "https://www.notion.so"


def test_candidate_qa_uses_backend_competitor_type_enum():
    result = CandidateQAResult(exists=True, corrected_competitor_type="head_direct")

    assert result.corrected_competitor_type == "head_direct"
    assert "head_direct" in candidate_existence_qa_system_prompt
    assert "indirect_cross" in candidate_existence_qa_system_prompt
    assert "不要输出 direct、indirect、substitute" in candidate_existence_qa_system_prompt


def test_fallback_report_has_traceable_findings():
    request = CreateTaskRequest(
        target_product="Feishu",
        raw_description="Feishu needs decision support for collaboration positioning.",
        lifecycle_stage="concept",
        analysis_purpose="decision_support",
        competitors=["DingTalk"],
        urls=[],
    )
    report = build_fallback_report("task_test", request, sources=[])

    assert report.task_id == "task_test"
    assert report.findings
    assert report.deep_dive_markdown
    for section in DEEP_DIVE_SECTIONS:
        assert f"## {section}" in report.deep_dive_markdown
    assert all(finding.evidence_ids for finding in report.findings)


def test_qa_rejects_missing_evidence():
    evidence = EvidenceItem(url="https://example.com", title="Example", excerpt="Example source")
    report = CompetitorReport(
        task_id="task_test",
        target_product="Feishu",
        competitors=[],
        executive_summary="Too short",
        findings=[
            AnalysisFinding(
                category="strategy",
                title="Unsupported conclusion",
                conclusion="This conclusion has no source support.",
                evidence_ids=[],
            )
        ],
        evidence=[evidence],
        deep_dive_markdown="## 一、封面与综述\nIncomplete",
    )

    result = validate_report(report, revision_count=0, max_revisions=2)

    assert result.passed is False
    assert result.revision_request is not None
    assert result.revision_request.target_agent in {
        "candidate_discovery_agent",
        "targeted_collector_agent",
        "source_collector_agent",
        "tool_executor_agent",
        "writer_agent",
    }


def test_qa_passes_traceable_report(sample_request):
    from conftest import make_report

    report = make_report("task_test", sample_request)
    result = validate_report(report, revision_count=0, max_revisions=2)

    assert result.passed is True
    assert result.revision_request is None


def test_qa_rejects_incomplete_deep_dive_markdown(sample_request):
    from conftest import make_report

    report = make_report("task_test", sample_request)
    report.deep_dive_markdown = "## 一、封面与综述\nOnly one section"

    result = validate_report(report, revision_count=0, max_revisions=2)

    assert result.passed is False
    assert any(issue.field == "deep_dive_markdown" for issue in result.issues)


def test_qa_rejects_tool_claim_without_reference(sample_request):
    from conftest import make_report

    report = make_report("task_test", sample_request)
    report.tool_results[0].claims[0].reference_ids = []
    report.tool_results[0].input_reference_ids = []

    result = validate_report(report, revision_count=0, max_revisions=2)

    assert result.passed is False
    assert any(issue.field.startswith("tool_") for issue in result.issues)
    assert result.revision_request is not None
    assert result.revision_request.target_agent == "tool_executor_agent"
