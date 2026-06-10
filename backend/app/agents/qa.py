from __future__ import annotations

import re

from myagent.backend.app.agents.deep_report import has_required_deep_dive_sections
from myagent.backend.app.models import CompetitorReport, QAResult, QAIssue, RevisionRequest


def validate_report(report: CompetitorReport, revision_count: int, max_revisions: int) -> QAResult:
    issues: list[QAIssue] = []
    evidence_ids = {item.id for item in report.evidence}
    reference_ids = {item.id for item in report.report_references}

    if not report.analysis_benchmark:
        issues.append(QAIssue(severity="high", field="analysis_benchmark", message="缺少锁定后的分析基准。"))
    else:
        benchmark = report.analysis_benchmark
        if not benchmark.target_product or not benchmark.lifecycle_stage or not benchmark.analysis_purpose or not benchmark.analysis_goal:
            issues.append(QAIssue(severity="high", field="analysis_benchmark", message="分析基准核心槽位不完整。"))
        if not benchmark.locked:
            issues.append(QAIssue(severity="medium", field="analysis_benchmark.locked", message="分析基准未锁定。"))

    if not report.selected_competitor_set:
        issues.append(QAIssue(severity="high", field="selected_competitor_set", message="缺少候选池评分和竞品选择结果。"))
    else:
        selected_names = {item.name for item in report.selected_competitor_set.selected}
        profile_names = {profile.name for profile in report.competitors}
        if not selected_names:
            issues.append(QAIssue(severity="high", field="selected_competitor_set.selected", message="没有最终入选竞品。"))
        if selected_names and not selected_names.issubset(profile_names):
            issues.append(QAIssue(severity="high", field="competitors", message="报告竞品列表与评分入选名单不一致。"))

    if not report.tool_plan:
        issues.append(QAIssue(severity="high", field="tool_plan", message="缺少工具链规划。"))
    elif report.analysis_benchmark and report.tool_plan.purpose != report.analysis_benchmark.analysis_purpose:
        issues.append(QAIssue(severity="high", field="tool_plan.purpose", message="工具链目的与分析基准不一致。"))
    elif report.tool_plan and not report.tool_plan.selected_tools:
        issues.append(QAIssue(severity="high", field="tool_plan.selected_tools", message="没有选定分析工具。"))

    if not report.collection_plan or not report.collection_plan.instructions:
        issues.append(QAIssue(severity="medium", field="collection_plan", message="缺少按工具生成的数据采集清单。"))

    if report.selected_competitor_set and not report.competitor_detail_cards_clean:
        issues.append(QAIssue(severity="high", field="competitor_detail_cards_clean", message="缺少清洗后的竞品资料卡。"))
    if report.selected_competitor_set and not report.industry_card_clean:
        issues.append(QAIssue(severity="medium", field="industry_card_clean", message="缺少清洗后的行业资料卡。"))
    if (report.competitor_detail_cards_clean or report.industry_card_clean) and not report.report_references:
        issues.append(QAIssue(severity="high", field="report_references", message="资料卡缺少可追溯的字段引用。"))
    if not report.source_provider_results:
        issues.append(QAIssue(severity="medium", field="source_provider_results", message="缺少来源 Provider 的采集结果记录。"))
    if report.tool_plan and not report.tool_results:
        issues.append(QAIssue(severity="high", field="tool_results", message="缺少每个分析工具的大模型结构化执行结果。"))
    if report.tool_results:
        for result in report.tool_results:
            if not result.claims:
                issues.append(QAIssue(severity="high", field=f"tool_results.{result.tool_name}", message="工具执行结果缺少 claims。"))
            invalid_input_refs = [item for item in result.input_reference_ids if item not in reference_ids]
            if invalid_input_refs:
                issues.append(QAIssue(severity="high", field=f"tool_results.{result.tool_name}.input_reference_ids", message="工具执行结果引用了不存在的资料卡字段。"))
            for claim in result.claims:
                if not claim.reference_ids:
                    issues.append(QAIssue(severity="high", field=f"tool_claim.{result.tool_name}.{claim.title}", message="工具结论缺少资料卡字段引用。"))
                invalid_claim_refs = [item for item in claim.reference_ids if item not in reference_ids]
                if invalid_claim_refs:
                    issues.append(QAIssue(severity="high", field=f"tool_claim.{result.tool_name}.{claim.title}", message="工具结论引用了不存在的资料卡字段。"))

    if not report.competitors:
        issues.append(QAIssue(severity="high", field="competitors", message="缺少竞品画像。"))
    if len(report.executive_summary.strip()) < 40:
        issues.append(QAIssue(severity="medium", field="executive_summary", message="摘要过短。"))
    if len(report.findings) < 3:
        issues.append(QAIssue(severity="medium", field="findings", message="关键发现少于 3 条。"))
    if not report.recommendations:
        issues.append(QAIssue(severity="high", field="recommendations", message="缺少行动建议。"))
    if not report.deep_dive_markdown.strip():
        issues.append(QAIssue(severity="high", field="deep_dive_markdown", message="缺少章节级 Markdown 报告。"))
    elif not has_required_deep_dive_sections(report.deep_dive_markdown):
        issues.append(QAIssue(severity="high", field="deep_dive_markdown", message="章节级报告缺少分析背景、竞品选择、分析维度、关键发现、总结建议或附录。"))

    for profile in report.competitors:
        if not profile.evidence_ids:
            issues.append(
                QAIssue(severity="medium", field=f"profile.{profile.name}", message="竞品画像缺少证据引用。")
            )
        invalid = [item for item in profile.evidence_ids if item not in evidence_ids]
        if invalid:
            issues.append(
                QAIssue(severity="high", field=f"profile.{profile.name}.evidence_ids", message="存在无效证据 ID。")
            )
        if profile.score is None and report.selected_competitor_set:
            issues.append(
                QAIssue(severity="medium", field=f"profile.{profile.name}.score", message="入选竞品缺少评分。")
            )
        if not profile.selection_reason and report.selected_competitor_set:
            issues.append(
                QAIssue(severity="medium", field=f"profile.{profile.name}.selection_reason", message="入选竞品缺少选择理由。")
            )

    for finding in report.findings:
        if not finding.evidence_ids:
            issues.append(QAIssue(severity="high", field=f"finding.{finding.id}", message="结论缺少证据引用。"))
        invalid = [item for item in finding.evidence_ids if item not in evidence_ids]
        if invalid:
            issues.append(QAIssue(severity="high", field=f"finding.{finding.id}.evidence_ids", message="结论引用了不存在的证据。"))

    if report.deep_dive_markdown and evidence_ids:
        cited = [evidence_id for evidence_id in evidence_ids if evidence_id in report.deep_dive_markdown]
        if not cited:
            issues.append(QAIssue(severity="medium", field="deep_dive_markdown.sources", message="深度报告没有引用有效证据 ID。"))
        if "选择逻辑" not in report.deep_dive_markdown and "组合逻辑" not in report.deep_dive_markdown:
            issues.append(QAIssue(severity="medium", field="deep_dive_markdown.selection", message="报告未解释竞品选择组合逻辑。"))
        if not re.search(r"- \[ \] ", report.deep_dive_markdown):
            issues.append(QAIssue(severity="medium", field="deep_dive_markdown.todo", message="报告缺少行动 To-do 清单。"))

    recommendations_supported = False
    if report.recommendations and report.deep_dive_markdown:
        recommendations_supported = any(evidence_id in report.deep_dive_markdown for evidence_id in evidence_ids)
    if report.recommendations and not recommendations_supported:
        issues.append(QAIssue(severity="medium", field="recommendations.evidence", message="行动建议缺少前文证据支撑。"))

    source_coverage = 0.0
    if report.findings:
        with_evidence = sum(1 for finding in report.findings if finding.evidence_ids)
        source_coverage = with_evidence / len(report.findings)
    structural_checks = [
        report.analysis_benchmark is not None,
        report.selected_competitor_set is not None,
        report.tool_plan is not None,
        bool(report.collection_plan and report.collection_plan.instructions),
        bool(report.competitor_detail_cards_clean and report.industry_card_clean and report.report_references),
        bool(report.tool_results),
        bool(report.recommendations),
        bool(report.deep_dive_markdown and has_required_deep_dive_sections(report.deep_dive_markdown)),
    ]
    structural_score = sum(1 for item in structural_checks if item) / len(structural_checks)
    completeness = 1.0 - min(len(issues) * 0.08, 0.75)
    score = round(max(0.0, min(1.0, (source_coverage * 0.45) + (structural_score * 0.35) + (completeness * 0.20))), 2)

    high_or_medium = [issue for issue in issues if issue.severity in {"high", "medium"}]
    passed = not high_or_medium and score >= 0.72

    revision_request = None
    if not passed and revision_count < max_revisions:
        if any(issue.field.startswith("selected_competitor_set") or "评分" in issue.message for issue in high_or_medium):
            target = "candidate_discovery_agent"
        elif any(issue.field.startswith("tool_results") or issue.field.startswith("tool_claim") for issue in high_or_medium):
            target = "tool_executor_agent"
        elif any(issue.field.startswith("competitor_detail_cards") or issue.field.startswith("industry_card") or issue.field.startswith("report_references") for issue in high_or_medium):
            target = "source_collector_agent"
        elif any(issue.field.startswith("source_provider_results") for issue in high_or_medium):
            target = "source_collector_agent"
        elif any("证据" in issue.message or "来源" in issue.message or "采集" in issue.message for issue in high_or_medium):
            target = "targeted_collector_agent"
        elif any(issue.field.startswith("tool_plan") or issue.field.startswith("collection_plan") for issue in high_or_medium):
            target = "writer_agent"
        else:
            target = "writer_agent"
        revision_request = RevisionRequest(
            target_agent=target,
            reason="; ".join(issue.message for issue in high_or_medium[:4]),
            missing_fields=[issue.field for issue in high_or_medium[:8]],
            suggested_queries=[
                f"{report.target_product} 竞品 定价 功能 用户",
                f"{report.target_product} competitors business model funding",
            ],
        )

    return QAResult(passed=passed, score=score, issues=issues, revision_request=revision_request)
