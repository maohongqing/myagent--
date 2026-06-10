from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from myagent.backend.app.agents.candidate_selector import (
    generate_candidate_pool_from_chunks_with_llm_or_rules,
    is_invalid_candidate_name,
)
from myagent.backend.app.agents.card_collector import collect_detail_cards
from myagent.backend.app.agents.deep_report import build_deep_dive_markdown, has_required_deep_dive_sections
from myagent.backend.app.agents.fallbacks import build_fallback_report, evidence_from_sources
from myagent.backend.app.agents.llm import (
    QA_LANE,
    configure_llm_runtime,
    friendly_llm_error,
    get_chat_model,
    retry_ainvoke,
    retry_structured_ainvoke,
)
from myagent.backend.app.agents.prompt_bridge import (
    REPORT_BACKGROUND_PROMPT,
    REPORT_COMPETITOR_SELECTION_PROMPT,
    REPORT_FINAL_UNIFY_PROMPT,
    REPORT_KEY_FINDINGS_PROMPT,
    REPORT_RECOMMENDATIONS_PROMPT,
    REPORT_TITLE_PROMPT,
    REPORT_TOOL_SECTION_PROMPT,
    analysis_intent_system_prompt,
    analysis_intent_user_instruction,
    analyst_system_prompt,
    brief_query_template_instruction,
    brief_query_template_system_prompt,
    candidate_existence_qa_instruction,
    candidate_existence_qa_system_prompt,
    phase_1_plan_intent,
    phase_2_analyst_intent,
    phase_2_plan_intent,
    phase_2_qa_intent,
    phase_3_plan_intent,
    phase_4_qa_intent,
    candidate_query_generation_prompt,
    competitor_relevance_score_system_prompt,
    plan_system_prompt,
    qa_agent_prompt_7,
    qa_system_prompt,
    writer_system_prompt,
)
from myagent.backend.app.agents.qa import validate_report
from myagent.backend.app.agents.search_runner_adapter import (
    collect_candidate_existence_sources_with_search_runner,
    collect_candidate_sources_with_search_runner,
    collect_evidence_with_search_runner,
)
from myagent.backend.app.agents.state_tree import (
    benchmark_to_phase_1,
    candidate_pool_payload,
    create_initial_state_tree,
    merge_phase_3_strategy,
    merge_updated_nodes,
    render_prompt,
    report_to_phase_4,
    selected_set_to_phase_2,
    sources_to_sandbox,
    tool_plan_to_phase_3,
)
from myagent.backend.app.agents.strategy import (
    AVAILABLE_CANVAS_TOOLS,
    AVAILABLE_METHODS,
    apply_selection_to_profiles,
    build_candidate_queries,
    build_candidates_from_sources,
    build_collection_plan,
    build_tool_plan,
    infer_benchmark,
    score_and_select_candidates,
)
from myagent.backend.app.agents.survey import generate_survey_design
from myagent.backend.app.agents.tool_executor import execute_tools, tool_results_to_sandboxes
from myagent.backend.app.config import Settings
from myagent.backend.app.events import EventBus
from myagent.backend.app.exporters import report_to_markdown
from myagent.backend.app.knowledge import retrieve_knowledge_context
from myagent.backend.app.models import (
    AgentRunEvent,
    AnalysisBenchmark,
    AnalysisFinding,
    AnalysisIntentJSON,
    CardQAResult,
    CandidateCompetitor,
    CandidateGapRequest,
    CandidateQAChangeLog,
    CandidateQAResult,
    ClarificationRequest,
    CollectionPlan,
    CompetitorBriefProfile,
    CompetitorDetailCard,
    CompetitorDetailCardClean,
    CompetitorRelevanceScore,
    CompetitorReport,
    CreateTaskRequest,
    DecisionOption,
    DecisionRequest,
    DetailQueryPlan,
    EvidenceItem,
    EvidenceSignal,
    ExtractedEvidenceItem,
    FieldExtractionResult,
    IndustryCard,
    IndustryCardClean,
    KnowledgeReuseSummary,
    QAResult,
    QAIssue,
    RankFactors,
    RawSource,
    ReportChapterDraft,
    ReportReference,
    ResearchField,
    SearchQuery,
    SearchPlan,
    SourceRef,
    SWOT,
    SelectedCompetitorSet,
    SourceProviderResult,
    ToolPlan,
    ToolExecutionResult,
    ToolSandbox,
    SurveyDesign,
    utc_now,
)
from myagent.backend.app.providers import dedupe_sources
from myagent.backend.app.storage import Storage

logger = logging.getLogger("competitor_agent")


async def structured_ainvoke(model, schema, messages, *, lane: str = "main"):
    return await retry_structured_ainvoke(model, schema, messages, lane=lane)


class ClarificationNeeded(Exception):
    def __init__(self, clarification: ClarificationRequest):
        super().__init__(clarification.question)
        self.clarification = clarification


class TaskCancelled(Exception):
    pass


class LLMDecisionNeeded(Exception):
    def __init__(self, decision: DecisionRequest):
        super().__init__(decision.question)
        self.decision = decision


class AnalystOutput(CompetitorReport):
    pass


class WriterOutput(CompetitorReport):
    pass


REPORT_SECTION_TITLES = [
    "一、报告名称与报告日期",
    "二、分析背景",
    "三、竞品选择",
    "四、展示自身竞品画布",
    "五、关键发现",
    "六、分析维度",
    "七、总结建议",
    "八、附录",
]


def _tool_result_reference_id(result: ToolExecutionResult, index: int) -> str:
    return (result.reference_id or "").strip() or f"tool_result_{index}"


def _sanitize_tool_results_for_analysis(tool_results: list[ToolExecutionResult]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for result_index, result in enumerate(tool_results, start=1):
        result_id = _tool_result_reference_id(result, result_index)
        sanitized_claims: list[dict[str, Any]] = []
        for claim_index, claim in enumerate(result.claims, start=1):
            sanitized_claims.append(
                {
                    "id": f"{result_id}.claim_{claim_index}",
                    "category": claim.category,
                    "title": claim.title,
                    "claim": claim.claim,
                    "confidence": claim.confidence,
                    "reasoning": claim.reasoning,
                }
            )
        sanitized.append(
            {
                "id": result_id,
                "tool_name": result.tool_name,
                "canonical_tool_name": result.canonical_tool_name,
                "scope_type": result.scope_type,
                "competitor_name": result.competitor_name,
                "reference_id": result_id,
                "claims": sanitized_claims,
                "data": result.data,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            }
        )
    return sanitized


def _has_own_product_info(request: CreateTaskRequest, benchmark: AnalysisBenchmark | None = None) -> bool:
    own = (request.our_product or "").strip()
    target = (request.target_product or (benchmark.target_product if benchmark else "") or "").strip()
    if not own:
        return False
    return own != target and len(own) >= 8


def _renumber_report_sections(markdown_sections: list[tuple[str, str]]) -> str:
    chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    output: list[str] = []
    report_title = ""
    for index, (_, markdown) in enumerate(markdown_sections):
        title = markdown_sections[index][0]
        number = chinese_numbers[index] if index < len(chinese_numbers) else str(index + 1)
        text = markdown.strip()
        if index == 0:
            title_match = re.match(r"^(#\s+.+?)(?:\r?\n|$)", text)
            if title_match:
                report_title = title_match.group(1).strip()
                text = text[title_match.end() :].strip()
        text = re.sub(r"^##\s*.*?(?:\r?\n|$)", "", text, count=1).strip()
        text = f"## {number}、{title}" + (f"\n\n{text}" if text else "")
        output.append(text)
    parts = [report_title] if report_title else []
    parts.extend(item for item in output if item)
    return "\n\n".join(parts).strip() + "\n"


def _report_context_payload(
    *,
    request: CreateTaskRequest,
    report: CompetitorReport,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark = report.analysis_benchmark
    selected_set = report.selected_competitor_set
    payload: dict[str, Any] = {
        "任务ID": report.task_id,
        "目标产品": benchmark.target_product if benchmark else report.target_product,
        "报告日期": utc_now().date().isoformat(),
        "用户请求": {
            "目标产品": request.target_product,
            "自身产品": request.our_product,
            "市场": request.market,
            "产品类别": request.product_category,
            "补充说明": _strip_prompt_links(request.notes or ""),
        },
        "阶段一分析基准": benchmark.model_dump(mode="json") if benchmark else None,
        "分析意图": report.analysis_intent.model_dump(mode="json") if report.analysis_intent else None,
        "竞品选择结果": _selected_set_prompt_payload(selected_set) if selected_set else None,
        "工具计划": report.tool_plan.model_dump(mode="json") if report.tool_plan else None,
        "采集计划": report.collection_plan.model_dump(mode="json") if report.collection_plan else None,
    }
    if extra:
        payload.update(extra)
    return _compact_for_llm(payload, max_text=900, max_list=20)


def _candidate_selection_payload(report: CompetitorReport) -> dict[str, Any]:
    selected_set = report.selected_competitor_set
    return _compact_for_llm(
        {
            "竞品候选池与最终选择": _selected_set_prompt_payload(selected_set) if selected_set else None,
            "候选竞品简介": [item.model_dump(mode="json") for item in report.competitor_brief_profiles],
            "清洗后竞品资料卡": [item.model_dump(mode="json") for item in report.competitor_detail_cards_clean],
            "最终竞品画像": [item.model_dump(mode="json") for item in report.competitors],
        },
        max_text=800,
        max_list=30,
    )


def _group_tool_results_by_tool(tool_results: list[ToolExecutionResult]) -> list[tuple[str, list[ToolExecutionResult]]]:
    groups: dict[str, list[ToolExecutionResult]] = defaultdict(list)
    labels: dict[str, str] = {}
    for result in tool_results:
        key = result.canonical_tool_name or result.tool_name or "analysis_tool"
        labels.setdefault(key, result.tool_name or key)
        groups[key].append(result)
    return [(labels[key], groups[key]) for key in groups]


def _fallback_tool_section(tool_name: str, tool_results: list[ToolExecutionResult]) -> str:
    subjects = [result.competitor_name or result.scope_type for result in tool_results]
    strongest_claims: list[str] = []
    for result in tool_results:
        for claim in result.claims[:2]:
            if claim.claim:
                strongest_claims.append(claim.claim.strip())
    overview = "；".join(strongest_claims[:3]) or "当前工具结果已形成结构化判断，但仍需要结合更多资料进一步验证。"

    lines = [
        f"### {tool_name}",
        "",
        f"本节基于 {tool_name} 的工具结果，对 {', '.join(subjects) if subjects else '相关对象'} 进行横向整理。整体来看，{overview}",
        "",
        "从竞品分析角度看，这组结果的价值不在于罗列单个功能或事实，而在于帮助判断不同竞品在定位、能力、用户痛点和市场机会上的差异。若多个对象在同一维度上呈现相似结论，说明该维度更可能是行业共性门槛；若某个对象呈现明显差异，则可作为后续产品定位、MVP 功能取舍或风险规避的重点参考。",
        "",
    ]
    for result in tool_results:
        subject = result.competitor_name or result.scope_type
        claims = [claim.claim.strip() for claim in result.claims[:3] if claim.claim.strip()]
        body = "；".join(claims) or result.reasoning or "暂无结构化结论。"
        lines.extend(
            [
                f"**{subject} 的主要表现**：{body}",
                "",
            ]
        )

    lines.extend(["| 对象 | 结论摘要 | 置信度 |", "| --- | --- | ---: |"])
    for result in tool_results:
        subject = result.competitor_name or result.scope_type
        claims = "；".join(claim.claim for claim in result.claims[:3]) or result.reasoning or "暂无结构化结论。"
        lines.append(f"| {subject} | {claims[:500]} | {result.confidence:.2f} |")
    lines.extend(
        [
            "",
            f"**小结**：{tool_name} 的结果表明，本轮分析已经覆盖该维度下的主要竞品差异。后续写入产品策略时，应优先使用上文中与目标用户痛点、交易信任、差异化定位和增长空间直接相关的结论，表格仅作为过程数据索引。",
        ]
    )
    return "\n".join(lines)


def _valid_web_url(url: str | None) -> bool:
    if not url:
        return False
    normalized = str(url).strip().lower()
    return normalized.startswith("http://") or normalized.startswith("https://")


def _appendix_web_sources(report: CompetitorReport, limit: int = 80) -> list[EvidenceItem]:
    seen: set[str] = set()
    sources: list[EvidenceItem] = []
    for evidence in report.evidence:
        if not _valid_web_url(evidence.url):
            continue
        key = str(evidence.url).strip().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        sources.append(evidence)
        if len(sources) >= limit:
            break
    return sources


def _build_appendix_markdown(report: CompetitorReport, request: CreateTaskRequest, section_title: str) -> str:
    lines = [
        f"## {section_title}",
        "",
        "### 附录说明",
        "本附录用于呈现报告写作过程中使用到的结构化信息、工具结果和资料索引，不额外调用大模型生成。",
        "",
        "### 输入与分析基准",
        f"- 目标产品：{report.analysis_benchmark.target_product if report.analysis_benchmark else report.target_product}",
        f"- 分析目的：{report.analysis_benchmark.analysis_purpose if report.analysis_benchmark else request.analysis_purpose}",
        f"- 产品阶段：{report.analysis_benchmark.lifecycle_stage if report.analysis_benchmark else request.lifecycle_stage}",
        "",
        "### 最终竞品",
    ]
    for profile in report.competitors:
        lines.append(f"- {profile.name}：{profile.positioning}；类型={profile.competitor_type or '待分类'}；分数={profile.score if profile.score is not None else '待评分'}")
    lines.extend(["", "### 工具执行结果索引"])
    for index, result in enumerate(report.tool_results, start=1):
        result_id = _tool_result_reference_id(result, index)
        lines.append(
            f"- {result_id}：{result.tool_name}；对象={result.competitor_name or result.scope_type}；claims={len(result.claims)}；confidence={result.confidence:.2f}"
        )
    web_sources = _appendix_web_sources(report)
    if web_sources:
        lines.extend(
            [
                "",
                "### 网页来源 URL 索引",
                "| 序号 | 关联对象 | 关联字段 | 网页标题 | URL | 摘录 |",
                "| ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for index, evidence in enumerate(web_sources, start=1):
            title = (evidence.title or evidence.url or "网页来源").replace("|", "\\|")
            url = str(evidence.url).strip()
            competitor = (evidence.competitor or "未标注").replace("|", "\\|")
            field_name = (evidence.field_name or ",".join(evidence.related_fields[:3]) or "未标注").replace("|", "\\|")
            excerpt = _detail_text(evidence.excerpt or "", 180).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {index} | {competitor} | {field_name} | {title} | {url} | {excerpt} |")
        if len(report.evidence) > len(web_sources):
            lines.append(f"\n> 注：附录已展示前 {len(web_sources)} 条去重网页 URL；完整证据可通过报告接口的 evidence 字段查看。")
    if report.report_references:
        lines.extend(["", "### 资料卡引用索引"])
        for ref in report.report_references:
            lines.append(f"- {ref.id}：{ref.title}")
    return "\n".join(lines)


def _fallback_background_section(report: CompetitorReport, request: CreateTaskRequest) -> str:
    benchmark = report.analysis_benchmark
    tool_plan = report.tool_plan
    return "\n".join(
        [
            "## 二、分析背景",
            "",
            f"### 1. 分析目标",
            f"本次竞品分析围绕“{benchmark.target_product if benchmark else report.target_product}”展开，分析目的为“{benchmark.analysis_goal if benchmark else request.analysis_goal}”。",
            f"产品当前阶段为“{benchmark.lifecycle_stage if benchmark else request.lifecycle_stage or 'concept'}”，核心任务是判断市场空间、竞品压力和可切入方向。",
            "",
            "### 2. 分析维度",
            f"本次选择的分析工具包括：{', '.join(tool_plan.selected_tools) if tool_plan else '待补充'}。",
            "",
            "### 3. 维度选择理由",
            tool_plan.rationale if tool_plan else "当前工具维度由系统根据分析目的和竞品类型自动选择。",
        ]
    )


def _fallback_competitor_selection_section(report: CompetitorReport) -> str:
    lines = [
        "## 三、竞品选择",
        "",
        "### 1. 最终选择竞品",
        "| 竞品 | 类型 | 分数 | 选择理由 |",
        "| --- | --- | ---: | --- |",
    ]
    for profile in report.competitors:
        lines.append(
            f"| {profile.name} | {profile.competitor_type or '待分类'} | {profile.score if profile.score is not None else '待评分'} | {profile.selection_reason or '基于候选池评分进入分析名单'} |"
        )
    lines.extend(
        [
            "",
            "### 2. 组合逻辑",
            "- 决策支持：优先选择 2 个直接竞品 + 1 个间接竞品。",
            "- 学习借鉴：优先选择 1 个直接竞品 + 2 个间接竞品。",
            "- 市场预警：优先选择 1 个直接竞品 + 1 个间接竞品 + 1 个替代品。",
        ]
    )
    return "\n".join(lines)


def _fallback_key_findings_section(report: CompetitorReport) -> str:
    lines = ["## 关键发现", ""]
    findings = _findings_from_tool_results(report.tool_results)
    if not findings:
        return "## 关键发现\n\n- 当前工具结果不足以稳定提炼关键发现，需要补充更多竞品资料和工具分析。"
    for item in findings:
        lines.append(f"- **{item.title}**：{item.conclusion}")
    return "\n".join(lines)


def _fallback_recommendations_section(report: CompetitorReport) -> str:
    return "\n".join(
        [
            "## 总结建议",
            "",
            "- [ ] 优先验证目标用户对交易信任、品相标准和售后保障的真实需求，形成 MVP 的核心验证指标。",
            "- [ ] 围绕最终入选竞品的短板设计差异化功能清单，并为每项功能绑定用户痛点和验证方法。",
            "- [ ] 补充关键竞品的价格、交易规则、用户反馈和近期增长证据，降低后续立项判断的不确定性。",
        ]
    )


def _valid_chapter_markdown(markdown: str, title: str) -> bool:
    if not markdown.strip():
        return False
    if title == "报告名称与报告日期":
        return markdown.lstrip().startswith("#") and "竞品分析报告" in markdown[:200]
    return markdown.lstrip().startswith("##")


def _split_findings_and_recommendations(
    markdown: str,
    fallback_findings: str,
    fallback_recommendations: str,
) -> tuple[str, str]:
    match = re.search(r"(?m)^##\s*(?:[一二三四五六七八九十]+[、.．]\s*)?总结建议\b", markdown)
    if not match:
        return fallback_findings, fallback_recommendations
    findings = markdown[: match.start()].strip()
    recommendations = markdown[match.start() :].strip()
    if not findings or not recommendations:
        return fallback_findings, fallback_recommendations
    return findings, recommendations


def _derive_report_summary_fields(report: CompetitorReport, markdown: str) -> None:
    if not report.executive_summary or len(report.executive_summary.strip()) < 40:
        report.executive_summary = _plain_text_excerpt(markdown, 500)
    if not report.findings:
        report.findings = _findings_from_tool_results(report.tool_results)
    if not report.recommendations:
        report.recommendations = _recommendations_from_markdown(markdown)


def _plain_text_excerpt(markdown: str, limit: int) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", markdown)
    text = re.sub(r"[#>*_\-\[\]\(\)`|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _findings_from_tool_results(tool_results: list[ToolExecutionResult]) -> list[AnalysisFinding]:
    findings: list[AnalysisFinding] = []
    for result in tool_results:
        for claim in result.claims[:1]:
            if len(findings) >= 3:
                return findings
            findings.append(
                AnalysisFinding(
                    category="strategy",
                    title=claim.title[:120] or result.tool_name,
                    conclusion=claim.claim[:1000],
                    evidence_ids=list(claim.evidence_ids or result.evidence_ids),
                    confidence=claim.confidence,
                    impact="medium",
                )
            )
    return findings


def _recommendations_from_markdown(markdown: str) -> list[str]:
    candidates = []
    in_recommendation = False
    for line in markdown.splitlines():
        if "总结建议" in line or "行动计划" in line:
            in_recommendation = True
            continue
        if in_recommendation and re.match(r"\s*[-*]\s+", line):
            candidates.append(re.sub(r"^\s*[-*]\s+", "", line).strip())
        if len(candidates) >= 6:
            break
    return candidates or ["基于本次竞品分析结果，优先围绕差异化定位、核心交易信任机制和用户验证闭环推进下一轮产品设计。"]


class StateTreePatch(BaseModel):
    requires_clarification: bool = False
    clarification_question: str | None = None
    updated_nodes: dict[str, Any] = Field(default_factory=dict)


class QAInspectionOutput(BaseModel):
    inspection_status: Literal["pass", "reject"] = "pass"
    issues_found: list[str] = Field(default_factory=list)
    feedback_to_analyzer: str | None = None
    feedback_to_planner_or_analyzer: str | None = None


class MarketGrowthDimensionQA(BaseModel):
    is_complete: bool = True
    is_evidence_supported: bool = True
    is_consistent: bool = True
    issues: list[str] = Field(default_factory=list)


class MarketGrowthQAOutput(BaseModel):
    product_name: str = ""
    is_accurate: bool = True
    has_missing_fields: bool = False
    has_suspicious_fields: bool = False
    needs_rework: bool = False
    dimension_qa: dict[str, MarketGrowthDimensionQA] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    suspicious_fields: list[Any] = Field(default_factory=list)
    rework_queries_needed: bool = False
    rework_focus: list[str] = Field(default_factory=list)
    qa_reason: str = ""


class CandidateSearchQueryItem(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    intent: str = Field(default="", max_length=300)
    expected_discovery_types: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(default="", max_length=500)


class CandidateSearchQueryOutput(BaseModel):
    queries: list[CandidateSearchQueryItem] = Field(default_factory=list, max_length=5)


class CandidateExistenceQAOutput(CandidateQAResult):
    pass


class WorkflowState(TypedDict, total=False):
    task_id: str
    request: dict[str, Any]
    analysis_intent: dict[str, Any]
    benchmark: dict[str, Any]
    clarification: dict[str, Any]
    candidate_sources: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    selected_competitor_set: dict[str, Any]
    tool_plan: dict[str, Any]
    collection_plan: dict[str, Any]
    search_plan: dict[str, Any]
    source_provider_results: list[dict[str, Any]]
    raw_sources: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    extracted_evidence: list[dict[str, Any]]
    field_extraction_results: list[dict[str, Any]]
    competitor_brief_profiles: list[dict[str, Any]]
    competitor_detail_cards: list[dict[str, Any]]
    industry_card: dict[str, Any] | None
    competitor_detail_cards_clean: list[dict[str, Any]]
    industry_card_clean: dict[str, Any] | None
    report_references: list[dict[str, Any]]
    report_chapters: list[dict[str, Any]]
    knowledge_context: dict[str, Any]
    reused_competitor_cards: list[dict[str, Any]]
    reused_industry_cards: list[dict[str, Any]]
    knowledge_sources: list[str]
    evidence_signals: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tool_sandboxes: list[dict[str, Any]]
    structured_report: dict[str, Any]
    analysis_report: dict[str, Any]
    report: dict[str, Any]
    qa_history: list[dict[str, Any]]
    revision_count: int
    max_revision_loops: int
    error: str | None
    state_tree: dict[str, Any]
    workflow_phase: str
    phase2_supplement_count: int
    qa_checkpoint: str
    candidate_search_iterations: int
    candidate_gap_request: dict[str, Any] | None
    candidate_qa_change_log: dict[str, Any]
    candidate_query_history: list[str]
    collection_retry_count: int
    collection_gap_request: list[str]
    phase_4_synthesis_step: str | None
    risk_notice: str | None
    candidate_pool_mode: str
    candidate_pool_notes: list[str]
    searched_url_keys: list[str]
    candidate_seen_url_keys: list[str]
    detail_seen_url_keys: list[str]
    industry_seen_url_keys: list[str]
    llm_failure_policy: str


def _short(text: str, size: int = 240) -> str:
    return text if len(text) <= size else f"{text[:size]}..."


def _merge_knowledge_candidates(
    candidates: list[CandidateCompetitor], knowledge_cards: list[dict[str, Any]]
) -> list[CandidateCompetitor]:
    seen = {candidate.name.strip().lower() for candidate in candidates}
    merged = list(candidates)
    for item in knowledge_cards:
        name = str(item.get("name") or item.get("canonical_name") or "").strip()
        if not name or name.lower() in seen:
            continue
        card = item.get("card") if isinstance(item.get("card"), dict) else {}
        fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
        tags = [str(tag) for tag in item.get("tags", []) if str(tag).strip()]
        merged.append(
            CandidateCompetitor(
                name=name,
                description=str(fields.get("positioning") or fields.get("category") or "Knowledge base competitor card."),
                website=str(fields.get("website") or "") or None,
                competitor_type=card.get("competitor_type") or "challenger",
                source_urls=[],
                source_titles=[],
                tags=["knowledge_base", *tags],
                rank_factors=RankFactors(heat=0.45, growth=0.35, similarity=0.65, risk=0.2),
                score=0.85,
                selection_reason="来自知识库资料卡的复用候选。",
            )
        )
        seen.add(name.lower())
    return merged


def _estimate_tokens(value: Any) -> int:
    text = str(value)
    return max(1, len(text) // 4)


def _compact_for_llm(value: Any, *, max_text: int = 900, max_list: int = 12, depth: int = 0) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_prompt_link_key(key):
                continue
            key_limit = max_list
            text_limit = max_text
            if key in {"evidence", "raw_sources", "source_provider_results", "tool_results", "tool_sandboxes"}:
                key_limit = min(max_list, 8)
                text_limit = min(max_text, 600)
            compacted[key] = _compact_for_llm(item, max_text=text_limit, max_list=key_limit, depth=depth + 1)
        return compacted
    if isinstance(value, list):
        visible = value[:max_list]
        items = [_compact_for_llm(item, max_text=max_text, max_list=max_list, depth=depth + 1) for item in visible]
        omitted = len(value) - len(visible)
        if omitted > 0:
            items.append({"_omitted_count": omitted})
        return items
    if isinstance(value, str):
        cleaned = _strip_prompt_links(value)
        return cleaned if len(cleaned) <= max_text else f"{cleaned[:max_text]}...[truncated]"
    return value


def _json_for_llm(value: Any, *, max_chars: int = 60000, max_text: int = 900, max_list: int = 12) -> str:
    compacted = _compact_for_llm(value, max_text=max_text, max_list=max_list)
    text = json.dumps(compacted, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    smaller = _compact_for_llm(value, max_text=360, max_list=6)
    smaller_text = json.dumps(smaller, ensure_ascii=False, default=str)
    if len(smaller_text) <= max_chars:
        return smaller_text
    return smaller_text[:max_chars] + "...[truncated_for_model]"


def _selected_set_prompt_payload(selected_set: SelectedCompetitorSet) -> dict[str, Any]:
    return {
        "total_budget": selected_set.total_budget,
        "candidate_count": len(selected_set.candidates),
        "selected_count": len(selected_set.selected),
        "scoring_profile": selected_set.scoring_profile.model_dump(mode="json"),
        "allocation": selected_set.allocation,
        "fallback_notes": selected_set.fallback_notes[:5],
        "gap_request": selected_set.gap_request.model_dump(mode="json") if selected_set.gap_request else None,
        "selected": [
            {
                "name": competitor.name,
                "description": _detail_text(_strip_prompt_links(competitor.description), 360),
                "competitor_type": competitor.competitor_type,
                "tags": competitor.tags[:8],
                "score": competitor.score,
                "selection_reason": _detail_text(_strip_prompt_links(competitor.selection_reason), 360),
                "rank_factors": competitor.rank_factors.model_dump(mode="json"),
                "risk_factors": competitor.risk_factors.model_dump(mode="json"),
                "source_titles": [_detail_text(_strip_prompt_links(title), 140) for title in competitor.source_titles[:5]],
                "omitted_source_title_count": max(0, len(competitor.source_titles) - 5),
            }
            for competitor in selected_set.selected
        ],
    }


def _is_prompt_link_key(key: Any) -> bool:
    return str(key).lower() in {
        "url",
        "urls",
        "website",
        "source_url",
        "source_urls",
        "matched_urls",
        "image_path",
        "image_alt",
        "screenshot_path",
        "chunk_shot_path",
        "chunk_shot_paths",
        "screenshot",
        "screenshots",
        "search_run_json",
        "search_run_report",
        "run_dir",
        "content_path",
        "report_path",
        "results_json_path",
    }


def _strip_prompt_links(value: str) -> str:
    cleaned = re.sub(r"\b[a-z][a-z0-9+.-]*://\S+|www\.\S+", "[link_removed]", value)
    cleaned = re.sub(r"\b(?:search_runs|task_runs|screenshots|extractions)/\S+", "[path_removed]", cleaned)
    return cleaned


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "step"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


_KNOWN_ALIAS_GROUPS: dict[str, set[str]] = {
    "bilibili": {"b站", "bilibili", "哔哩哔哩", "嗶哩嗶哩"},
}
_KNOWN_ALIAS_BY_KEY: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _KNOWN_ALIAS_GROUPS.items()
    for alias in aliases
}
_GENERIC_ALIAS_SUFFIXES = ("小程序", "客户端", "应用", "官网", "平台")


def _basic_candidate_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _strip_generic_candidate_suffix(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for suffix in _GENERIC_ALIAS_SUFFIXES:
        if text.endswith(suffix):
            stem = text[: -len(suffix)].strip(" \t\r\n-_")
            if len(_basic_candidate_key(stem)) >= 2:
                return stem
    match = re.match(r"^(.+?)[\s_-]*app$", text, flags=re.IGNORECASE)
    if match:
        stem = match.group(1).strip(" \t\r\n-_")
        if any(ord(ch) > 127 for ch in stem) and len(_basic_candidate_key(stem)) >= 2:
            return stem
    return text


def _candidate_alias_keys(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    normalized = (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("【", "(")
        .replace("】", ")")
        .replace("［", "(")
        .replace("］", ")")
    )
    parts = [normalized]
    parts.extend(match.strip() for match in re.findall(r"\(([^()]+)\)", normalized) if match.strip())
    no_parenthetical = re.sub(r"\([^()]*\)", "", normalized).strip()
    if no_parenthetical:
        parts.append(no_parenthetical)
    stripped_parts = [_strip_generic_candidate_suffix(part) for part in parts]
    stripped_keys = [_basic_candidate_key(part) for part in stripped_parts if part]
    keys = [_basic_candidate_key(part) for part in parts]
    keys = [key for key in keys if len(key) >= 2]
    stripped_keys = [key for key in stripped_keys if len(key) >= 2]
    all_keys = _dedupe([*stripped_keys, *keys])
    canonical_hits = [_KNOWN_ALIAS_BY_KEY[key] for key in all_keys if key in _KNOWN_ALIAS_BY_KEY]
    return _dedupe([*canonical_hits, *all_keys])


def _candidate_key(value: str) -> str:
    keys = _candidate_alias_keys(value)
    return keys[0] if keys else ""


def _display_name_rank(name: str) -> tuple[int, int, int, int, str]:
    cleaned = str(name or "").strip()
    key = _basic_candidate_key(cleaned)
    known_alias = 0 if key in _KNOWN_ALIAS_BY_KEY else 1
    bracket_penalty = 1 if re.search(r"[()（）【】［］]", cleaned) else 0
    suffix_penalty = 1 if _strip_generic_candidate_suffix(cleaned) != cleaned else 0
    return known_alias, suffix_penalty, bracket_penalty, len(cleaned), cleaned.lower()


def _preferred_candidate_name(names: list[str]) -> str:
    cleaned = _dedupe([name for name in names if str(name or "").strip()])
    if not cleaned:
        return ""
    return sorted(cleaned, key=_display_name_rank)[0]


def _record_candidate_merge(
    change_log: CandidateQAChangeLog,
    from_name: str,
    to_name: str,
    reason: str,
) -> None:
    if not from_name or not to_name or from_name == to_name:
        return
    key = (from_name, to_name, reason)
    existing = {
        (str(item.get("from_name") or ""), str(item.get("to_name") or ""), str(item.get("reason") or ""))
        for item in change_log.merged_candidates
    }
    if key not in existing:
        change_log.merged_candidates.append({"from_name": from_name, "to_name": to_name, "reason": reason})


def _merge_candidate_records(existing: CandidateCompetitor, candidate: CandidateCompetitor) -> CandidateCompetitor:
    preferred_name = _preferred_candidate_name([existing.name, candidate.name]) or existing.name
    existing.name = preferred_name
    existing.score = max(existing.score, candidate.score)
    existing.source_urls = _dedupe([*existing.source_urls, *candidate.source_urls])
    existing.source_titles = _dedupe([*existing.source_titles, *candidate.source_titles])
    existing.tags = _dedupe([*existing.tags, *candidate.tags, "alias_canonicalized"])
    existing.source_refs = _merge_source_refs(existing.source_refs, candidate.source_refs)
    if len(candidate.description) > len(existing.description):
        existing.description = candidate.description
    if candidate.selection_reason and len(candidate.selection_reason) > len(existing.selection_reason):
        existing.selection_reason = candidate.selection_reason
    if candidate.website and not existing.website:
        existing.website = candidate.website
    return existing


def _resolve_alias_canonical_name(
    names: set[str],
    directed_targets: dict[str, str],
    candidates_by_name: dict[str, CandidateCompetitor],
) -> str:
    target_names = {target for target in directed_targets.values() if target in names}
    if target_names:
        return _preferred_candidate_name(list(target_names))
    return _preferred_candidate_name(list(names)) or next(iter(names))


def _candidate_alias_components(
    candidates: list[CandidateCompetitor],
    alias_decisions: dict[str, str],
) -> dict[str, str]:
    names_by_key: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for key in _candidate_alias_keys(candidate.name):
            names_by_key[key].add(candidate.name)
    for source, target in alias_decisions.items():
        if source and target:
            names_by_key[_candidate_key(source) or source].update([source, target])
            names_by_key[_candidate_key(target) or target].update([source, target])

    parent: dict[str, str] = {}

    def find(name: str) -> str:
        parent.setdefault(name, name)
        if parent[name] != name:
            parent[name] = find(parent[name])
        return parent[name]

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for names in names_by_key.values():
        ordered = [name for name in names if name]
        if not ordered:
            continue
        first = ordered[0]
        for name in ordered[1:]:
            union(first, name)
    for source, target in alias_decisions.items():
        if source and target:
            union(source, target)
    for candidate in candidates:
        find(candidate.name)

    grouped: dict[str, set[str]] = defaultdict(set)
    for name in list(parent):
        grouped[find(name)].add(name)

    candidates_by_name = {candidate.name: candidate for candidate in candidates}
    canonical_by_name: dict[str, str] = {}
    for names in grouped.values():
        directed_targets = {
            source: target
            for source, target in alias_decisions.items()
            if source in names and target in names
        }
        canonical = _resolve_alias_canonical_name(names, directed_targets, candidates_by_name)
        for name in names:
            canonical_by_name[name] = canonical
    return canonical_by_name


def _apply_candidate_alias_decisions(
    candidates: list[CandidateCompetitor],
    alias_decisions: dict[str, str],
    change_log: CandidateQAChangeLog | None = None,
) -> list[CandidateCompetitor]:
    if not candidates:
        return []
    canonical_by_name = _candidate_alias_components(candidates, alias_decisions)
    renamed: list[CandidateCompetitor] = []
    for candidate in candidates:
        canonical_name = canonical_by_name.get(candidate.name, candidate.name)
        if canonical_name != candidate.name and change_log is not None:
            reason = (
                "QA merge target canonicalization."
                if candidate.name in alias_decisions
                else "Deterministic alias key canonicalization."
            )
            _record_candidate_merge(change_log, candidate.name, canonical_name, reason)
        renamed.append(candidate.model_copy(update={"name": canonical_name}, deep=True))
    return _merge_candidate_list([], renamed, change_log)


def _merge_candidate_list(
    previous: list[CandidateCompetitor],
    current: list[CandidateCompetitor],
    change_log: CandidateQAChangeLog | None = None,
) -> list[CandidateCompetitor]:
    merged: dict[str, CandidateCompetitor] = {}
    for candidate in [*previous, *current]:
        key = _candidate_key(candidate.name)
        if not key:
            continue
        existing = merged.get(key)
        if not existing:
            merged[key] = candidate.model_copy(deep=True)
            continue
        before_name = existing.name
        _merge_candidate_records(existing, candidate)
        if change_log is not None:
            from_name = candidate.name if candidate.name != existing.name else before_name
            to_name = existing.name
            if from_name != to_name:
                _record_candidate_merge(
                    change_log,
                    from_name,
                    to_name,
                    "Deterministic alias key canonicalization.",
                )
    return list(merged.values())[:50]


def _merge_source_refs(existing: list[SourceRef], additions: list[SourceRef]) -> list[SourceRef]:
    merged = list(existing)
    seen = {f"{item.source_id or ''}|{item.chunk_id or ''}|{item.url}|{item.quote[:80]}" for item in merged}
    for item in additions:
        key = f"{item.source_id or ''}|{item.chunk_id or ''}|{item.url}|{item.quote[:80]}"
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged[:12]


def _candidate_name_lookup(candidates: list[CandidateCompetitor]) -> dict[str, CandidateCompetitor]:
    lookup: dict[str, CandidateCompetitor] = {}
    for candidate in candidates:
        for key in _candidate_alias_keys(candidate.name):
            lookup.setdefault(key, candidate)
    return lookup


def _canonical_name_lookup_from_candidates(candidates: list[CandidateCompetitor]) -> dict[str, str]:
    return {key: candidate.name for key, candidate in _candidate_name_lookup(candidates).items()}


def _canonicalize_name_value(name: str, lookup: dict[str, str]) -> str:
    for key in _candidate_alias_keys(name):
        if key in lookup:
            return lookup[key]
    return name


def _alias_duplicate_groups(names: list[str]) -> list[list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        key = _candidate_key(name)
        if key:
            grouped[key].append(name)
    duplicates: list[list[str]] = []
    for group in grouped.values():
        unique = _dedupe(group)
        if len(unique) > 1:
            duplicates.append(unique)
    return duplicates


def _sync_brief_profile_names(
    profiles: list[CompetitorBriefProfile],
    candidates: list[CandidateCompetitor],
) -> list[CompetitorBriefProfile]:
    lookup = _canonical_name_lookup_from_candidates(candidates)
    merged: dict[str, CompetitorBriefProfile] = {}
    for profile in profiles:
        name = _canonicalize_name_value(profile.competitor, lookup)
        updated = profile.model_copy(update={"competitor": name}, deep=True)
        key = _candidate_key(name)
        if not key:
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = updated
            continue
        existing.market_heat.source_refs = _merge_source_refs(existing.market_heat.source_refs, updated.market_heat.source_refs)
        existing.development_status.source_refs = _merge_source_refs(existing.development_status.source_refs, updated.development_status.source_refs)
        existing.source_refs = _merge_source_refs(existing.source_refs, updated.source_refs)
        existing.qa_passed = existing.qa_passed or updated.qa_passed
        existing.missing_fields = _dedupe([*existing.missing_fields, *updated.missing_fields])
        if len(updated.market_heat.value) > len(existing.market_heat.value):
            existing.market_heat.value = updated.market_heat.value
            existing.market_heat.confidence = max(existing.market_heat.confidence, updated.market_heat.confidence)
        if len(updated.development_status.value) > len(existing.development_status.value):
            existing.development_status.value = updated.development_status.value
            existing.development_status.confidence = max(existing.development_status.confidence, updated.development_status.confidence)
        existing.qa_result = existing.qa_result or updated.qa_result
    return list(merged.values())


def _sync_detail_card_names(
    cards: list[CompetitorDetailCard],
    candidates: list[CandidateCompetitor],
) -> list[CompetitorDetailCard]:
    lookup = _canonical_name_lookup_from_candidates(candidates)
    merged: dict[str, CompetitorDetailCard] = {}
    for card in cards:
        name = _canonicalize_name_value(card.competitor, lookup)
        updated = card.model_copy(update={"competitor": name}, deep=True)
        updated.name.value = _canonicalize_name_value(updated.name.value or name, lookup)
        key = _candidate_key(name)
        if key and key not in merged:
            merged[key] = updated
    return list(merged.values())


def _sync_clean_card_names(
    cards: list[CompetitorDetailCardClean],
    candidates: list[CandidateCompetitor],
) -> list[CompetitorDetailCardClean]:
    lookup = _canonical_name_lookup_from_candidates(candidates)
    merged: dict[str, CompetitorDetailCardClean] = {}
    for card in cards:
        name = _canonicalize_name_value(card.competitor, lookup)
        updated = card.model_copy(update={"competitor": name}, deep=True)
        key = _candidate_key(name)
        if key and key not in merged:
            merged[key] = updated
    return list(merged.values())


def _canonicalize_report_competitor_aliases(report: CompetitorReport) -> None:
    selected_set = report.selected_competitor_set
    if not selected_set:
        return
    canonical_candidates = _merge_candidate_list([], [*selected_set.candidates, *selected_set.selected])
    if not canonical_candidates:
        return
    if report.analysis_benchmark:
        report.selected_competitor_set = _select_top_competitors_by_category(report.analysis_benchmark, canonical_candidates)
        canonical_candidates = report.selected_competitor_set.candidates
    else:
        selected = _merge_candidate_list([], selected_set.selected)
        report.selected_competitor_set = selected_set.model_copy(
            update={"candidates": canonical_candidates, "selected": selected},
            deep=True,
        )
    lookup = _canonical_name_lookup_from_candidates(canonical_candidates)

    profile_by_key: dict[str, Any] = {}
    for profile in report.competitors:
        name = _canonicalize_name_value(profile.name, lookup)
        updated = profile.model_copy(update={"name": name}, deep=True)
        key = _candidate_key(name)
        if key and key not in profile_by_key:
            profile_by_key[key] = updated
    report.competitors = list(profile_by_key.values())
    report.competitor_brief_profiles = _sync_brief_profile_names(report.competitor_brief_profiles, canonical_candidates)
    report.competitor_detail_cards = _sync_detail_card_names(report.competitor_detail_cards, canonical_candidates)
    report.competitor_detail_cards_clean = _sync_clean_card_names(report.competitor_detail_cards_clean, canonical_candidates)

    for query in (report.search_plan.queries if report.search_plan else []):
        if query.competitor:
            query.competitor = _canonicalize_name_value(query.competitor, lookup)
    for instruction in (report.collection_plan.instructions if report.collection_plan else []):
        if instruction.competitor:
            instruction.competitor = _canonicalize_name_value(instruction.competitor, lookup)
    for item in report.evidence:
        if item.competitor:
            item.competitor = _canonicalize_name_value(item.competitor, lookup)
    for item in report.field_extraction_results:
        item.competitor = _canonicalize_name_value(item.competitor, lookup)
        for fact in item.facts:
            fact.competitor = _canonicalize_name_value(fact.competitor, lookup)
    for item in report.tool_results:
        if item.competitor_name:
            item.competitor_name = _canonicalize_name_value(item.competitor_name, lookup)
    for item in report.tool_sandboxes:
        if item.competitor_name:
            item.competitor_name = _canonicalize_name_value(item.competitor_name, lookup)
    for cell in report.comparison_matrix:
        cell.competitor = _canonicalize_name_value(cell.competitor, lookup)


def _resolve_candidate_merge_target(
    candidate: CandidateCompetitor,
    output: CandidateQAResult,
    candidates: list[CandidateCompetitor],
) -> tuple[str, str | None]:
    lookup = _candidate_name_lookup([item for item in candidates if item.name != candidate.name])
    merge_target = (output.merge_target or "").strip()
    if merge_target:
        target = next((lookup[key] for key in _candidate_alias_keys(merge_target) if key in lookup), None)
        if target:
            return target.name, None
        canonical_name = (output.canonical_name or "").strip()
        if canonical_name:
            return canonical_name, f"merge_target not found in candidate pool: {merge_target}"
        return candidate.name, f"merge_target not found in candidate pool: {merge_target}"
    canonical_name = (output.canonical_name or "").strip()
    if canonical_name:
        return canonical_name, None
    return candidate.name, None



def _fallback_analysis_intent(request: CreateTaskRequest, benchmark: AnalysisBenchmark) -> AnalysisIntentJSON:
    context = " / ".join(
        item
        for item in [
            request.market or "",
            request.product_category or "",
            request.geography or "",
            request.company_size or "",
        ]
        if item
    )
    return AnalysisIntentJSON(
        industry=request.product_category or request.market or benchmark.target_product,
        analysis_purpose=benchmark.analysis_purpose,
        goal=request.analysis_goal or benchmark.analysis_goal,
        target=benchmark.target_product,
        problems=[request.notes or benchmark.analysis_goal] if (request.notes or benchmark.analysis_goal) else [],
        challenges=[context] if context else [],
        has_own_product=bool(request.our_product or request.target_product),
        own_product=request.our_product or request.target_product or "",
        target_product=benchmark.target_product,
        confidence=benchmark.confidence,
        source_summary=request.raw_description or request.notes or benchmark.raw_description,
    )


def state_safe_intent_payload(request: CreateTaskRequest, benchmark: AnalysisBenchmark) -> dict[str, Any]:
    return {
        "industry": request.product_category or request.market or benchmark.target_product,
        "analysis_purpose": benchmark.analysis_purpose,
        "goal": request.analysis_goal or benchmark.analysis_goal,
        "target_product": benchmark.target_product,
        "our_product": request.our_product or request.target_product or "",
        "market": request.market or "",
        "geography": request.geography or "",
        "notes": _strip_prompt_links(request.notes or request.raw_description or ""),
    }


def _sources_for_competitor_name(name: str, sources: list[RawSource]) -> list[RawSource]:
    key = name.lower()
    matched = [
        source
        for source in sources
        if key
        and (
            key in str(source.metadata.get("competitor") or "").lower()
            or key in source.title.lower()
            or key in source.content.lower()
            or key in source.url.lower()
        )
    ]
    return matched or sources[:3]


def _source_ref_from_raw_source(source: RawSource, *, quote: str) -> SourceRef:
    return SourceRef(
        source_id=source.id,
        chunk_id=source.metadata.get("chunk_id"),
        url=source.url,
        title=source.title,
        quote=_detail_text(quote or source.content, 900),
        screenshot_path=source.metadata.get("screenshot_path"),
        evidence_screenshot_path=source.metadata.get("evidence_screenshot_path") or source.metadata.get("quote_shot_path"),
        chunk_shot_paths=[],
    )


def _brief_field_text(candidate: CandidateCompetitor, sources: list[RawSource], keywords: list[str]) -> str:
    for source in sources:
        text = f"{source.title}. {source.content}"
        lower = text.lower()
        if any(keyword.lower() in lower for keyword in keywords):
            return _detail_text(text, 700)
    if sources:
        return _detail_text(f"{sources[0].title}. {sources[0].content}", 700)
    return candidate.description


def _brief_profile_clean(profile: CompetitorBriefProfile) -> dict[str, Any]:
    return {
        "competitor": profile.competitor,
        "market_heat": profile.market_heat.model_dump(mode="json"),
        "development_status": profile.development_status.model_dump(mode="json"),
        "qa_passed": profile.qa_passed,
        "missing_fields": profile.missing_fields,
        "qa_result": profile.qa_result,
    }


def _select_top_competitors_by_category(
    benchmark: AnalysisBenchmark,
    candidates: list[CandidateCompetitor],
) -> SelectedCompetitorSet:
    selectable_candidates = [
        candidate
        for candidate in candidates
        if not is_invalid_candidate_name(candidate.name, benchmark)
        and "not_suitable_for_deep_analysis" not in set(candidate.tags)
        and "fallback_placeholder" not in set(candidate.tags)
    ]
    if not selectable_candidates:
        empty_set = score_and_select_candidates(benchmark, [], budget=3)
        empty_set.gap_request = CandidateGapRequest(
            missing_competitor_types=["head_direct", "indirect_cross", "potential_substitute"],
            missing_factors=["heat", "growth", "similarity", "strict_source_refs"],
            suggested_query_focus=[
                f"{benchmark.target_product} competitors alternatives market heat growth",
                f"{benchmark.target_product} indirect competitors substitute products",
            ],
            rationale="No selectable competitors remained after QA and validity filtering.",
        )
        return empty_set

    ranked_set = score_and_select_candidates(benchmark, selectable_candidates, budget=max(3, len(selectable_candidates)))
    ranked = [candidate for candidate in ranked_set.candidates if candidate.score > 0.05]
    selected: list[CandidateCompetitor] = []
    used: set[str] = set()

    def count_direct() -> int:
        return sum(1 for item in selected if item.competitor_type in {"head_direct", "challenger"})

    def count_type(competitor_type: str) -> int:
        return sum(1 for item in selected if item.competitor_type == competitor_type)

    def take(types: set[str], count: int) -> None:
        for candidate in ranked:
            if len(selected) >= 3:
                break
            key = _candidate_key(candidate.name)
            if key in used or candidate.competitor_type not in types:
                continue
            selected.append(candidate)
            used.add(key)
            count -= 1
            if count <= 0:
                break

    if benchmark.analysis_purpose == "decision_support":
        allocation = {"direct": 2, "indirect_cross": 1}
        take({"head_direct"}, 2)
        if count_direct() < 2:
            take({"challenger"}, 2 - count_direct())
        if count_type("indirect_cross") < 1:
            take({"indirect_cross"}, 1)
    elif benchmark.analysis_purpose == "learning":
        allocation = {"direct": 1, "indirect_cross": 2}
        take({"head_direct"}, 1)
        if count_direct() < 1:
            take({"challenger"}, 1)
        if count_type("indirect_cross") < 2:
            take({"indirect_cross"}, 2 - count_type("indirect_cross"))
    else:
        allocation = {"direct": 1, "indirect_cross": 1, "potential_substitute": 1}
        take({"head_direct"}, 1)
        if count_direct() < 1:
            take({"challenger"}, 1)
        if count_type("indirect_cross") < 1:
            take({"indirect_cross"}, 1)
        if count_type("potential_substitute") < 1:
            take({"potential_substitute"}, 1)

    for candidate in ranked:
        if len(selected) >= 3:
            break
        key = _candidate_key(candidate.name)
        if key not in used:
            selected.append(candidate)
            used.add(key)

    missing_types = _missing_phase2_selection_types(benchmark, selected)
    weak_sources = [item.name for item in ranked if not item.source_refs][:5]
    gap_request = None
    if len(selected) < 3 or missing_types:
        gap_request = CandidateGapRequest(
            missing_competitor_types=missing_types,
            missing_factors=["heat", "growth", "similarity"] + (["strict_source_refs"] if weak_sources else []),
            suggested_query_focus=_dedupe(
                [
                    f"{benchmark.target_product} competitors alternatives market heat growth",
                    f"{benchmark.target_product} direct competitors indirect competitors substitute products",
                    *[f"{benchmark.target_product} {item.name} users funding growth popularity" for item in selected],
                ]
            ),
            rationale=(
                f"Selected {len(selected)} of 3 competitors. "
                f"Missing types: {', '.join(missing_types) or 'none'}. "
                f"Weak source_refs: {', '.join(weak_sources) or 'none'}."
            ),
        )
    return ranked_set.model_copy(
        update={
            "total_budget": 3,
            "selected": selected[:3],
            "allocation": allocation,
            "fallback_notes": _dedupe(
                [
                    *ranked_set.fallback_notes,
                    "Selected 3 competitors by purpose-specific rule after per-competitor QA and scoring.",
                ]
            ),
            "gap_request": gap_request,
        }
    )


def _missing_phase2_selection_types(
    benchmark: AnalysisBenchmark,
    selected: list[CandidateCompetitor],
) -> list[Literal["head_direct", "challenger", "indirect_cross", "potential_substitute"]]:
    direct_count = sum(1 for item in selected if item.competitor_type in {"head_direct", "challenger"})
    indirect_count = sum(1 for item in selected if item.competitor_type == "indirect_cross")
    substitute_count = sum(1 for item in selected if item.competitor_type == "potential_substitute")
    missing: list[Literal["head_direct", "challenger", "indirect_cross", "potential_substitute"]] = []
    if benchmark.analysis_purpose == "decision_support":
        if direct_count < 2:
            missing.append("head_direct")
        if indirect_count < 1:
            missing.append("indirect_cross")
    elif benchmark.analysis_purpose == "learning":
        if direct_count < 1:
            missing.append("head_direct")
        if indirect_count < 2:
            missing.append("indirect_cross")
    else:
        if direct_count < 1:
            missing.append("head_direct")
        if indirect_count < 1:
            missing.append("indirect_cross")
        if substitute_count < 1:
            missing.append("potential_substitute")
    return missing


def _research_field_from_sources(
    sources: list[RawSource],
    keywords: list[str],
    *,
    fallback: str = "",
    max_refs: int = 3,
) -> ResearchField:
    matched: list[RawSource] = []
    for source in sources:
        text = f"{source.title} {source.content}".lower()
        if any(keyword.lower() in text for keyword in keywords):
            matched.append(source)
    if not matched:
        matched = sources[:1]
    refs = [_source_ref_from_raw_source(source, quote=_detail_text(source.content, 700)) for source in matched[:max_refs]]
    if matched:
        value = _detail_text(" ".join(f"{source.title}: {source.content}" for source in matched[:2]), 1000)
    else:
        value = fallback
    return ResearchField(value=value or fallback, confidence=0.68 if refs else 0.25, source_refs=refs)


def _build_competitor_detail_cards(
    competitors: list[CandidateCompetitor],
    sources: list[RawSource],
) -> list[CompetitorDetailCard]:
    cards: list[CompetitorDetailCard] = []
    for competitor in competitors:
        competitor_sources = _sources_for_competitor_name(competitor.name, sources)
        name_field = ResearchField(value=competitor.name, confidence=0.9, source_refs=[])
        category_field = ResearchField(value=competitor.competitor_type, confidence=0.8, source_refs=[])
        website_field = ResearchField(value=competitor.website or (competitor.source_urls[0] if competitor.source_urls else ""), confidence=0.65, source_refs=[])
        cards.append(
            CompetitorDetailCard(
                competitor=competitor.name,
                competitor_type=competitor.competitor_type,
                name=name_field,
                category=category_field,
                website=website_field,
                positioning=_research_field_from_sources(competitor_sources, ["position", "定位", "about", "product"], fallback=competitor.description),
                target_users=_research_field_from_sources(competitor_sources, ["target user", "customer", "用户", "客户", "audience"]),
                core_scenarios=_research_field_from_sources(competitor_sources, ["scenario", "use case", "场景", "案例"]),
                core_features=_research_field_from_sources(competitor_sources, ["feature", "功能", "capability", "release"]),
                key_parameters=_research_field_from_sources(competitor_sources, ["parameter", "spec", "参数", "配置", "limit"]),
                pricing=_research_field_from_sources(competitor_sources, ["pricing", "price", "subscription", "定价", "价格"]),
                business_model=_research_field_from_sources(competitor_sources, ["business model", "revenue", "商业模式", "收入"]),
                channels=_research_field_from_sources(competitor_sources, ["channel", "app store", "website", "渠道", "分发"]),
                growth_signals=_research_field_from_sources(competitor_sources, ["growth", "download", "MAU", "增长", "下载", "用户量"]),
                funding_or_org_signals=_research_field_from_sources(competitor_sources, ["funding", "financing", "hiring", "融资", "招聘", "团队"]),
                technology_or_product_features=_research_field_from_sources(competitor_sources, ["technology", "AI", "API", "技术", "模型"]),
                differentiation=_research_field_from_sources(competitor_sources, ["differentiation", "advantage", "特色", "优势", "差异"]),
                user_feedback=_research_field_from_sources(competitor_sources, ["review", "rating", "feedback", "评价", "评论"]),
                risks=_research_field_from_sources(competitor_sources, ["risk", "complaint", "regulation", "风险", "投诉"]),
                recent_updates=_research_field_from_sources(competitor_sources, ["news", "release", "update", "新闻", "更新", "动态"]),
                key_evidence=_research_field_from_sources(competitor_sources, [competitor.name], fallback=competitor.description, max_refs=5),
            )
        )
    return cards


def _build_industry_card(
    benchmark: AnalysisBenchmark,
    request: CreateTaskRequest,
    sources: list[RawSource],
) -> IndustryCard:
    industry_name = request.product_category or request.market or benchmark.target_product
    return IndustryCard(
        basic_info={
            "industry_name": ResearchField(value=industry_name, confidence=0.78),
            "industry_scope": _research_field_from_sources(sources, ["industry", "market", "行业", "市场", "范围"], fallback=request.market or benchmark.raw_description),
            "main_users": _research_field_from_sources(sources, ["user", "customer", "用户", "客户", "人群"]),
            "core_scenarios": _research_field_from_sources(sources, ["scenario", "use case", "场景", "用途"]),
            "buyer_behavior_or_preferences": _research_field_from_sources(sources, ["buyer", "consumer", "preference", "采购", "消费", "偏好"]),
        },
        market_situation={
            "market_size": _research_field_from_sources(sources, ["market size", "TAM", "规模", "市场规模"]),
            "growth_trend": _research_field_from_sources(sources, ["growth", "CAGR", "增长", "趋势"]),
            "willingness_to_pay": _research_field_from_sources(sources, ["willingness to pay", "pricing", "付费", "价格", "客单价"]),
            "cost_pressure": _research_field_from_sources(sources, ["cost", "expense", "成本", "压力"]),
            "investment_environment": _research_field_from_sources(sources, ["funding", "investment", "融资", "投融资", "投资"]),
        },
        competition_situation={
            "competition_intensity": _research_field_from_sources(sources, ["competition", "competitor", "竞争", "竞品"]),
            "entry_barriers": _research_field_from_sources(sources, ["barrier", "moat", "门槛", "壁垒"]),
            "substitutes": _research_field_from_sources(sources, ["substitute", "alternative", "替代", "替代方案"]),
            "user_switching_cost": _research_field_from_sources(sources, ["switching cost", "迁移", "转换成本", "留存"]),
        },
        external_environment={
            "regulation_or_laws": _research_field_from_sources(sources, ["regulation", "law", "policy", "监管", "法律", "政策"]),
            "technology_trends": _research_field_from_sources(sources, ["technology", "AI", "trend", "技术", "趋势"]),
            "key_resource_dependencies": _research_field_from_sources(sources, ["resource", "dependency", "supply", "资源", "依赖"]),
        },
    )


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", str(value or "").strip()).strip("-").lower()
    return cleaned or "item"


def _reference_anchor(reference_id: str) -> str:
    return "ref-" + re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", reference_id).strip("-").lower()


def _iter_research_fields(prefix: str, value: Any):
    if isinstance(value, ResearchField):
        yield prefix, value
        return
    if isinstance(value, BaseModel):
        for key, item in value.model_dump().items():
            # Re-read through attributes to preserve ResearchField instances.
            yield from _iter_research_fields(f"{prefix}.{key}" if prefix else key, getattr(value, key))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_research_fields(f"{prefix}.{key}" if prefix else str(key), item)


def _build_report_references(report: CompetitorReport) -> list[ReportReference]:
    references: list[ReportReference] = []
    seen: set[str] = set()

    def source_ref_payload(field: ResearchField) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for ref in field.source_refs:
            url = str(ref.url or "").strip()
            if url:
                key = url.rstrip("/")
                if key in seen_urls:
                    continue
                seen_urls.add(key)
            refs.append(
                {
                    "source_id": ref.source_id,
                    "chunk_id": ref.chunk_id,
                    "url": url,
                    "title": ref.title,
                    "quote": _detail_text(ref.quote, 500),
                    "screenshot_path": ref.screenshot_path,
                    "evidence_screenshot_path": ref.evidence_screenshot_path,
                    "chunk_shot_paths": ref.chunk_shot_paths,
                }
            )
        return refs

    def summary_with_sources(value: str, field: ResearchField) -> str:
        summary = _detail_text(value, 800)
        urls = [
            str(ref.url).strip()
            for ref in field.source_refs
            if _valid_web_url(ref.url)
        ]
        deduped_urls = list(dict.fromkeys(url.rstrip("/") for url in urls))
        if not deduped_urls:
            return summary
        source_lines = "\n".join(f"- {url}" for url in deduped_urls[:5])
        return f"{summary}\n\n网页来源：\n{source_lines}"

    def add(reference_id: str, title: str, reference_type: Literal["clean_card", "tool_result"], summary: str, payload: dict[str, Any]) -> None:
        if reference_id in seen or not summary.strip():
            return
        seen.add(reference_id)
        references.append(
            ReportReference(
                id=reference_id,
                title=title,
                reference_type=reference_type,
                anchor=_reference_anchor(reference_id),
                summary=_detail_text(summary, 1000),
                payload=payload,
            )
        )

    for card in report.competitor_detail_cards:
        competitor_slug = _slug(card.competitor)
        for field_name, field in _iter_research_fields("", card):
            if not isinstance(field, ResearchField) or not field.value.strip():
                continue
            safe_field = field_name.replace(".", "_")
            reference_id = f"card.competitor.{competitor_slug}.{safe_field}"
            add(
                reference_id,
                f"{card.competitor} {safe_field}",
                "clean_card",
                summary_with_sources(field.value, field),
                {
                    "competitor": card.competitor,
                    "field": field_name,
                    "confidence": field.confidence,
                    "source_refs": source_ref_payload(field),
                },
            )

    if report.industry_card:
        for field_name, field in _iter_research_fields("", report.industry_card):
            if not isinstance(field, ResearchField) or not field.value.strip():
                continue
            safe_field = field_name.replace(".", "_")
            reference_id = f"card.industry.{safe_field}"
            add(
                reference_id,
                f"行业资料 {safe_field}",
                "clean_card",
                summary_with_sources(field.value, field),
                {
                    "field": field_name,
                    "confidence": field.confidence,
                    "source_refs": source_ref_payload(field),
                },
            )

    for tool_index, result in enumerate(report.tool_results, start=1):
        tool_slug = _slug(result.canonical_tool_name or result.tool_name or f"tool-{tool_index}")
        for claim_index, claim in enumerate(result.claims, start=1):
            scope_slug = _slug(result.competitor_name or result.scope_type or "comparison")
            category_slug = _slug(claim.category or "claim")
            reference_id = f"tool.{tool_slug}.{scope_slug}.{category_slug}.{claim_index}"
            add(
                reference_id,
                claim.title or f"{result.tool_name} 结论 {claim_index}",
                "tool_result",
                claim.claim,
                {
                    "tool_name": result.tool_name,
                    "canonical_tool_name": result.canonical_tool_name,
                    "scope_type": result.scope_type,
                    "competitor_name": result.competitor_name,
                    "claim_category": claim.category,
                    "confidence": claim.confidence,
                    "evidence_ids": claim.evidence_ids,
                    "reference_ids": claim.reference_ids,
                    "input_reference_ids": result.input_reference_ids,
                },
            )
    return references


def _ensure_markdown_reference_links(markdown: str, report: CompetitorReport) -> str:
    text = (markdown or "").strip()
    references = report.report_references or _build_report_references(report)
    if not references:
        return text

    source_entries: list[tuple[int, ReportReference, dict[str, Any], str]] = []
    ref_source_badges: dict[str, list[str]] = defaultdict(list)
    seen_sources: set[tuple[str, str]] = set()
    for ref in references:
        if ref.reference_type != "clean_card" or not isinstance(ref.payload, dict):
            continue
        source_refs = ref.payload.get("source_refs")
        if not isinstance(source_refs, list):
            continue
        for source in source_refs:
            if not isinstance(source, dict):
                continue
            url = str(source.get("url") or "").strip()
            if not _valid_web_url(url):
                continue
            key = (ref.id, url.rstrip("/"))
            if key in seen_sources:
                continue
            seen_sources.add(key)
            number = len(source_entries) + 1
            source_entries.append((number, ref, source, url))
            if len(ref_source_badges[ref.anchor]) < 3:
                ref_source_badges[ref.anchor].append(f"[源{number}](#source-{number})")
            if len(source_entries) >= 80:
                break
        if len(source_entries) >= 80:
            break

    for anchor, badges in ref_source_badges.items():
        badge_text = " ".join(badges)
        pattern = re.compile(rf"(\[[^\]]+\]\(#{re.escape(anchor)}\))(?!\s*\[源\d+\])")
        text = pattern.sub(rf"\1 {badge_text}", text)

    visible_refs = references[:12]
    if "#ref-" not in text:
        links = "\n".join(
            f"- [{ref.title}](#{ref.anchor})"
            + (f" {' '.join(ref_source_badges.get(ref.anchor, []))}" if ref_source_badges.get(ref.anchor) else "")
            for ref in visible_refs[:6]
        )
        text = f"{text}\n\n## 本章引用\n{links}".strip()

    if "引用索引" not in text:
        lines = ["", "## 引用索引"]
        for ref in visible_refs:
            source_refs = ref.payload.get("source_refs") if isinstance(ref.payload, dict) else None
            source_lines: list[str] = []
            if isinstance(source_refs, list):
                for source in source_refs[:5]:
                    if not isinstance(source, dict):
                        continue
                    url = str(source.get("url") or "").strip()
                    if not _valid_web_url(url):
                        continue
                    title = str(source.get("title") or url).strip()
                    source_lines.append(f"- [{title}]({url})")
            lines.extend(
                [
                    "",
                    f'<a id="{ref.anchor}"></a>',
                    f"### {ref.title}",
                    ref.summary,
                ]
            )
            if source_lines:
                lines.extend(["", "**网页来源**：", *source_lines])
        text = f"{text}\n" + "\n".join(lines)
    if "资料卡网页来源索引" not in text and source_entries:
        source_rows: list[str] = []
        for number, ref, source, url in source_entries:
            title = str(source.get("title") or url).strip().replace("|", "\\|")
            quote = _detail_text(source.get("quote") or "", 160).replace("|", "\\|").replace("\n", " ")
            source_rows.append(
                f'| <a id="source-{number}"></a>源{number} | [{ref.title}](#{ref.anchor}) | [{title}]({url}) | {quote} |'
            )
        if source_rows:
            text = (
                f"{text.rstrip()}\n\n"
                "## 资料卡网页来源索引\n"
                "| 来源编号 | 资料卡字段 | 网页来源 | 支撑摘录 |\n"
                "| --- | --- | --- | --- |\n"
                + "\n".join(source_rows)
            )
    return text.strip() + "\n"


DETAIL_ITEM_LIMIT = 20
DETAIL_TEXT_LIMIT = 500


def _detail_text(value: Any, size: int = DETAIL_TEXT_LIMIT) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= size else f"{text[:size]}..."


def _detail_items(items: list[Any], mapper, limit: int = DETAIL_ITEM_LIMIT) -> dict[str, Any]:
    visible = items[:limit]
    return {
        "items": [mapper(item) for item in visible],
        "total": len(items),
        "omitted_count": max(0, len(items) - len(visible)),
    }


def _event_display_details(
    node: str,
    status: str,
    message: str | None,
    output_summary: str | None,
    details: dict[str, Any] | None,
    token_payload: Any | None,
) -> dict[str, Any]:
    enriched = dict(details or {})
    stage_key = str(enriched.get("ui_stage_key") or _infer_ui_stage_key(node, message, output_summary, enriched))
    if stage_key:
        enriched["ui_stage_key"] = stage_key
        enriched.setdefault("ui_title", _ui_stage_title(stage_key, node))
    if status in {"completed", "revision", "failed"} and not enriched.get("display_markdown"):
        markdown = _display_markdown_for_event(stage_key, enriched, token_payload, output_summary or message or "")
        if markdown:
            enriched["display_markdown"] = markdown
    return enriched


def _infer_ui_stage_key(node: str, message: str | None, output_summary: str | None, details: dict[str, Any]) -> str:
    kind = str(details.get("kind") or "")
    kind_map = {
        "state_tree_phase_1": "phase_1_intent",
        "candidate_search": "phase_2_candidate_search",
        "target_selection": "phase_2_target_selection",
        "qa_checkpoint_1": "qa_checkpoint_1",
        "strategy_planning": "phase_3_strategy",
        "search_plan": "phase_4_search_plan",
        "source_collection": "phase_4_source_collection",
        "tool_execution": "phase_4_tool_execution",
        "analysis": "phase_4_analysis",
        "writer": "final_report",
        "qa_checkpoint_2": "qa_checkpoint_2",
        "qa": "final_qa",
        "clarification": "clarification",
    }
    if kind in kind_map:
        return kind_map[kind]
    text = f"{message or ''} {output_summary or ''}"
    if "阶段一" in text:
        return "phase_1_intent"
    if "全网泛搜" in text or "候选竞品池" in text:
        return "phase_2_candidate_search"
    if "语义判断候选" in text or "锁定竞品阵型" in text:
        return "phase_2_target_selection"
    if "质检点 1" in text:
        return "qa_checkpoint_1"
    if "阶段三" in text or "分析框架" in text:
        return "phase_3_strategy"
    if node == "search_planner_agent":
        return "phase_4_search_plan"
    if node == "source_collector_agent":
        return "phase_4_source_collection"
    if node == "tool_executor_agent":
        return "phase_4_tool_execution"
    if node == "analyst_agent":
        return "phase_4_analysis"
    if node == "writer_agent":
        return "final_report"
    if "质检点 2" in text:
        return "qa_checkpoint_2"
    return node


def _ui_stage_title(stage_key: str, node: str) -> str:
    if stage_key == "phase_4_analysis":
        return "阶段二：逐个质检热度/增速信息，逐个打分并锁定3个目标竞品"
    return {
        "phase_1_intent": "阶段一：分析目的",
        "phase_2_candidate_search": "阶段二：泛搜索候选竞品",
        "phase_2_target_selection": "阶段二：判断候选竞品",
        "qa_checkpoint_1": "质检点一：竞品阵型",
        "phase_3_strategy": "阶段三：分析策略",
        "phase_4_search_plan": "阶段四：定向搜索计划",
        "phase_4_source_collection": "阶段四：定向搜索资料",
        "phase_4_tool_execution": "阶段四：分析工具执行",
        "phase_4_analysis": "阶段四：分析结论",
        "qa_checkpoint_2": "质检点二：全局审查",
        "final_report": "最终报告",
        "final_qa": "质量检查",
        "clarification": "信息澄清",
    }.get(stage_key, node)


def _display_markdown_for_event(
    stage_key: str,
    details: dict[str, Any],
    token_payload: Any | None,
    fallback: str,
) -> str:
    payload = _jsonable(token_payload) if token_payload is not None else {}
    if stage_key == "phase_1_intent":
        phase_1_payload = payload.get("state_tree", {}).get("phase_1_intent") if isinstance(payload.get("state_tree"), dict) else None
        return _display_phase_1(
            details.get("benchmark") or payload.get("phase_1_intent") or phase_1_payload or payload,
            details.get("analysis_intent") or payload.get("analysis_intent"),
        )
    if stage_key == "phase_2_candidate_search":
        return _display_candidate_search(payload, fallback)
    if stage_key == "phase_2_target_selection":
        return _display_target_selection(payload, fallback)
    if stage_key == "phase_3_strategy":
        return _display_strategy(payload, fallback)
    if stage_key == "phase_4_search_plan":
        return _display_search_plan(payload, fallback)
    if stage_key == "phase_4_source_collection":
        return _display_source_collection(payload, fallback)
    if stage_key == "phase_4_tool_execution":
        return _display_tool_execution(payload, fallback)
    if stage_key in {"phase_4_analysis", "final_report"}:
        return _display_report(details.get("report") or payload, fallback)
    if stage_key in {"qa_checkpoint_1", "qa_checkpoint_2", "final_qa"}:
        return _display_qa(details, payload, fallback)
    return fallback.strip()


def _display_phase_1(data: Any, analysis_intent: Any | None = None) -> str:
    item = data if isinstance(data, dict) else {}
    intent = analysis_intent if isinstance(analysis_intent, dict) else item.get("analysis_intent")
    intent = intent if isinstance(intent, dict) else {}
    specific_goal = (
        intent.get("goal")
        or intent.get("target")
        or item.get("analysis_goal")
        or item.get("specific_goal")
        or item.get("core_problem_to_solve")
        or "支撑竞品分析与产品决策"
    )
    problems = _as_list(intent.get("problems"))
    core_problem = "；".join(str(problem).strip() for problem in problems if str(problem).strip()) or specific_goal
    return "\n".join(
        [
            "## 分析目的",
            f"- 目标产品：{intent.get('target_product') or item.get('target_product') or item.get('product_name') or '未命名产品'}",
            f"- 产品阶段：{_lifecycle_label(str(item.get('lifecycle_stage') or item.get('product_lifecycle') or ''))}",
            f"- 分析目的：{_purpose_label(str(item.get('analysis_purpose') or ''))}",
            f"- 分析目标：{specific_goal}",
            f"- 核心问题：{core_problem}",
            f"- 置信度：{intent.get('confidence') if intent.get('confidence') is not None else item.get('confidence') if item.get('confidence') is not None else '未标注'}",
        ]
    )


def _display_candidate_search(payload: Any, fallback: str) -> str:
    item = payload if isinstance(payload, dict) else {}
    candidates = _as_list(item.get("candidates"))
    queries = _as_list(item.get("queries"))
    qa_change_log = item.get("candidate_qa_change_log") if isinstance(item.get("candidate_qa_change_log"), dict) else {}
    lines = [
        "## 泛搜索结果",
        f"- 搜索问题数：{len(queries)}",
        f"- 候选竞品数：{len(candidates)}",
    ]
    if queries:
        lines.extend(["", "## 搜索方向"])
        lines.extend(f"- {query}" for query in queries[:8])
    lines.extend(["", "## 候选竞品"])
    if not candidates:
        lines.append(fallback or "未识别出候选竞品。")
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        lines.extend(_candidate_lines(index, candidate))
    if qa_change_log:
        for title, key in [
            ("QA removed", "removed_candidates"),
            ("QA merged", "merged_candidates"),
            ("Type corrections", "type_corrections"),
            ("Unverified kept", "unverified_candidates"),
        ]:
            records = _as_list(qa_change_log.get(key))
            if not records:
                continue
            lines.extend(["", f"## {title}"])
            for record in records[:12]:
                if isinstance(record, dict):
                    lines.append(f"- {_clean_display_text(record, 320)}")
    return "\n".join(lines).strip()


def _display_target_selection(payload: Any, fallback: str) -> str:
    item = payload if isinstance(payload, dict) else {}
    selected = _as_list(item.get("selected"))
    lines = ["## 确定的 5 个竞品"]
    if not selected:
        lines.append(fallback or "未锁定最终竞品。")
    for index, candidate in enumerate(selected, start=1):
        if not isinstance(candidate, dict):
            continue
        lines.extend(_candidate_lines(index, candidate))
    notes = _as_list(item.get("fallback_notes"))
    if notes:
        lines.extend(["", "## 选择说明"])
        lines.extend(f"- {_clean_display_text(note, 260)}" for note in notes)
    return "\n".join(lines).strip()


def _display_strategy(payload: Any, fallback: str) -> str:
    item = payload if isinstance(payload, dict) else {}
    lines = ["## 分析策略"]
    tools = _as_list(item.get("selected_tools") or item.get("selected_analysis_units") or item.get("selected_frameworks"))
    if tools:
        lines.append("- 选定工具：" + "、".join(str(tool) for tool in tools))
    if item.get("analysis_focus"):
        lines.append(f"- 分析重点：{item.get('analysis_focus')}")
    if item.get("selection_rationale"):
        lines.append(f"- 选择理由：{_clean_display_text(item.get('selection_rationale'), 420)}")
    if len(lines) == 1:
        lines.append(fallback or "分析策略已生成。")
    return "\n".join(lines).strip()


def _display_search_plan(payload: Any, fallback: str) -> str:
    item = payload if isinstance(payload, dict) else {}
    queries = _as_list(item.get("queries"))
    lines = ["## 定向搜索计划", f"- Query 数量：{len(queries)}"]
    for index, query in enumerate(queries[:20], start=1):
        if isinstance(query, dict):
            target = query.get("competitor") or "综合检索"
            fields = "、".join(str(value) for value in _as_list(query.get("expected_fields")))
            lines.append(f"{index}. {target}：{query.get('query') or ''}")
            if fields:
                lines.append(f"   - 关注字段：{fields}")
        else:
            lines.append(f"{index}. {query}")
    if not queries:
        lines.append(fallback or "未生成搜索计划。")
    return "\n".join(lines).strip()


def _display_source_collection(payload: Any, fallback: str) -> str:
    item = payload if isinstance(payload, dict) else {}
    sources = _as_list(item.get("sources"))
    competitor_cards = _as_list(item.get("competitor_detail_cards_clean"))
    industry_card = item.get("industry_card_clean") if isinstance(item.get("industry_card_clean"), dict) else None
    lines = ["## 本阶段采集完成", f"- 采集来源数：{len(sources)}", f"- 竞品资料卡数：{len(competitor_cards)}"]
    if industry_card:
        lines.append("- 行业资料卡：已生成")
    if competitor_cards:
        lines.extend(["", "## 竞品资料卡"])
        for card in competitor_cards:
            if not isinstance(card, dict):
                continue
            name = card.get("competitor") or "未命名竞品"
            lines.extend(["", f"### {name}"])
            fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
            for field_name, value in list(fields.items())[:12]:
                if str(value).strip():
                    lines.append(f"- {field_name}：{_clean_display_text(value, 220)}")
    if industry_card:
        lines.extend(["", "## 行业资料卡"])
        fields = industry_card.get("fields") if isinstance(industry_card.get("fields"), dict) else {}
        for field_name, value in list(fields.items())[:14]:
            if str(value).strip():
                lines.append(f"- {field_name}：{_clean_display_text(value, 220)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        if not isinstance(source, dict):
            continue
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        competitor = metadata.get("competitor") or "未归类来源"
        grouped[str(competitor)].append(source)
    if grouped:
        lines.extend(["", "## 来源摘要"])
    for competitor, items in grouped.items():
        lines.extend(["", f"### {competitor}"])
        for source in items[:5]:
            lines.append(f"- {source.get('title') or '未命名来源'}")
            if source.get("url"):
                lines.append(f"  - 页面地址：{source.get('url')}")
            if source.get("content"):
                lines.append(f"  - 摘要：{_clean_display_text(source.get('content'), 320)}")
    if not sources:
        lines.append(fallback or "未采集到可用来源。")
    return "\n".join(lines).strip()


def _display_tool_execution(payload: Any, fallback: str) -> str:
    results = payload if isinstance(payload, list) else _as_list(payload.get("results") if isinstance(payload, dict) else [])
    lines = ["## 工具执行结果", f"- 完成工具数：{len(results)}"]
    for result in results[:12]:
        if not isinstance(result, dict):
            continue
        lines.extend(["", f"## {result.get('tool_name') or '未命名工具'}"])
        claims = _as_list(result.get("claims"))
        for claim in claims[:5]:
            if isinstance(claim, dict):
                lines.append(f"- {claim.get('title') or claim.get('category') or '结论'}：{_clean_display_text(claim.get('claim'), 320)}")
    if not results:
        lines.append(fallback or "未生成工具结论。")
    return "\n".join(lines).strip()


def _display_report(report: Any, fallback: str) -> str:
    item = report if isinstance(report, dict) else {}
    lines = ["## 阶段结论"]
    if item.get("executive_summary"):
        lines.append(f"- 摘要：{_clean_display_text(item.get('executive_summary'), 500)}")
    competitors = _as_list(item.get("competitors"))
    if competitors:
        lines.extend(["", "## 竞品画像"])
        for index, competitor in enumerate(competitors[:8], start=1):
            if isinstance(competitor, dict):
                lines.append(f"{index}. {competitor.get('name') or '未命名竞品'}：{_clean_display_text(competitor.get('positioning') or competitor.get('selection_reason'), 260)}")
    findings = _as_list(item.get("findings"))
    if findings:
        lines.extend(["", "## 关键发现"])
        for finding in findings[:8]:
            if isinstance(finding, dict):
                lines.append(f"- {finding.get('title') or '发现'}：{_clean_display_text(finding.get('conclusion'), 320)}")
    if len(lines) == 1:
        lines.append(fallback or "本阶段已完成。")
    return "\n".join(lines).strip()


def _display_qa(details: dict[str, Any], payload: Any, fallback: str) -> str:
    qa = details.get("qa_result")
    if qa is None and isinstance(payload, dict):
        qa = payload
    if not isinstance(qa, dict):
        qa = {}
    passed = details.get("passed")
    if passed is None and isinstance(qa, dict):
        passed = qa.get("passed") or qa.get("inspection_status") == "pass"
    issues = _as_list(details.get("issues") or (qa.get("issues") if isinstance(qa, dict) else []))
    lines = ["## 质检结果", f"- 状态：{'通过' if passed else '需要关注'}"]
    if issues:
        lines.extend(["", "## 问题与提示"])
        for issue in issues:
            if isinstance(issue, dict):
                lines.append(f"- {_clean_display_text(issue.get('message') or issue, 260)}")
            else:
                lines.append(f"- {_clean_display_text(issue, 260)}")
    elif fallback:
        lines.append(f"- {fallback}")
    return "\n".join(lines).strip()


def _candidate_lines(index: int, candidate: dict[str, Any]) -> list[str]:
    lines = [f"{index}. {candidate.get('name') or '未命名竞品'}"]
    if candidate.get("competitor_type"):
        lines.append(f"   - 分类：{_competitor_type_label(str(candidate.get('competitor_type')))}")
    if candidate.get("score") is not None:
        lines.append(f"   - 评分：{candidate.get('score')}")
    if candidate.get("description"):
        lines.append(f"   - 简介：{_clean_display_text(candidate.get('description'), 360)}")
    if candidate.get("selection_reason"):
        lines.append(f"   - 理由：{_clean_display_text(candidate.get('selection_reason'), 360)}")
    return lines


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_display_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else f"{text[:limit]}..."


def _candidate_note_display_text(value: Any, limit: int) -> str:
    text = _clean_display_text(value, limit * 2)
    replacements = {
        "chunk analysis": "candidate extraction",
        "chunk 分析": "候选抽取",
        "chunk 鍒嗘瀽": "候选抽取",
        "chunk 调用": "source processing call",
        "chunk 璋冪敤": "source processing call",
        "chunk 大模型": "source processing model",
        "chunk 澶фā鍨?": "source processing model",
        "chunk": "source segment",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text if len(text) <= limit else f"{text[:limit]}..."


def _competitor_type_label(value: str) -> str:
    return {
        "head_direct": "头部直接竞品",
        "challenger": "追赶型竞品",
        "indirect_cross": "间接或跨界竞品",
        "potential_substitute": "潜在替代品",
    }.get(value, value)


def _lifecycle_label(value: str) -> str:
    return {"concept": "概念期", "development": "研发期", "launched": "已上线"}.get(value, value or "未标注")


def _purpose_label(value: str) -> str:
    return {
        "decision_support": "决策支持",
        "learning": "学习借鉴",
        "market_warning": "市场预警",
    }.get(value, value or "未标注")


def _benchmark_detail(benchmark: AnalysisBenchmark) -> dict[str, Any]:
    return {
        "target_product": benchmark.target_product,
        "lifecycle_stage": benchmark.lifecycle_stage,
        "analysis_purpose": benchmark.analysis_purpose,
        "analysis_goal": benchmark.analysis_goal,
        "confidence": benchmark.confidence,
        "inferred_from": benchmark.inferred_from,
        "locked": benchmark.locked,
    }


def _has_valid_phase_1_intent(state_tree: dict[str, Any]) -> bool:
    intent = state_tree.get("phase_1_intent")
    if not isinstance(intent, dict):
        return False
    lifecycle = intent.get("product_lifecycle")
    purpose = intent.get("analysis_purpose")
    goal = intent.get("specific_goal") or intent.get("core_problem_to_solve")
    return lifecycle in {"concept", "development", "launched"} and purpose in {
        "decision_support",
        "learning",
        "market_warning",
    } and bool(str(goal or "").strip())


def _clarification_detail(clarification: ClarificationRequest) -> dict[str, Any]:
    return {
        "missing_slots": clarification.missing_slots,
        "question": clarification.question,
        "options": [option.model_dump(mode="json") for option in clarification.options],
        "context_summary": clarification.context_summary,
    }


def _source_detail(source: RawSource) -> dict[str, Any]:
    return {
        "id": source.id,
        "title": source.title,
        "url": source.url,
        "source_type": source.source_type,
        "provider": source.metadata.get("provider"),
        "query": source.metadata.get("query"),
        "excerpt": _detail_text(source.content),
    }


def _candidate_detail(candidate: CandidateCompetitor) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "description": _detail_text(candidate.description),
        "website": candidate.website,
        "competitor_type": candidate.competitor_type,
        "score": candidate.score,
        "selection_reason": _detail_text(candidate.selection_reason),
        "source_titles": candidate.source_titles[:5],
        "source_urls": candidate.source_urls[:5],
        "source_ref_count": len(candidate.source_refs),
        "tags": candidate.tags,
        "rank_factors": candidate.rank_factors.model_dump(mode="json"),
        "weight_assignment": candidate.weight_assignment,
        "factor_scores": candidate.factor_scores,
        "scoring_reason": candidate.scoring_reason,
    }


def _candidate_search_progress_markdown(queries: list[str], candidates: list[CandidateCompetitor], notes: list[str] | None = None) -> str:
    payload = {
        "queries": queries,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "notes": notes or [],
    }
    lines = [_display_candidate_search(payload, "正在识别候选竞品。")]
    if notes:
        lines.extend(["", "## 处理提示"])
        lines.extend(f"- {_candidate_note_display_text(note, 260)}" for note in notes[:8])
    return "\n".join(lines).strip()


def _candidate_existence_candidate_payload(candidate: CandidateCompetitor) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "description": _strip_prompt_links(candidate.description),
        "competitor_type": candidate.competitor_type,
        "tags": candidate.tags,
        "score": candidate.score,
        "selection_reason": _strip_prompt_links(candidate.selection_reason),
    }


def _candidate_existence_source_payload(source: RawSource) -> dict[str, str]:
    return {
        "source_id": source.id,
        "chunk_id": str(source.metadata.get("chunk_id") or ""),
        "title": _strip_prompt_links(source.title),
        "content_excerpt": _strip_prompt_links(_detail_text(source.content, 1200)),
    }


def _collection_instruction_detail(instruction) -> dict[str, Any]:
    return {
        "tool": instruction.tool,
        "competitor": instruction.competitor,
        "queries": instruction.queries[:5],
        "source_types": instruction.source_types,
        "extraction_targets": instruction.extraction_targets,
    }


def _search_query_detail(query) -> dict[str, Any]:
    return {
        "query": query.query,
        "tool": query.tool,
        "competitor": query.competitor,
        "priority": query.priority,
        "target_site_types": query.target_site_types,
        "expected_fields": query.expected_fields,
        "rationale": _detail_text(query.rationale, 260),
    }


def _evidence_detail(evidence: EvidenceItem) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "title": evidence.title,
        "url": evidence.url,
        "source_type": evidence.source_type,
        "excerpt": _detail_text(evidence.excerpt),
        "credibility": evidence.credibility,
        "related_fields": evidence.related_fields,
    }


def _evidence_prompt_line(evidence: EvidenceItem) -> str:
    source_refs = ", ".join(
        f"{ref.source_id or 'source'}:{ref.chunk_id or 'chunk'}"
        for ref in evidence.source_refs[:4]
    )
    field = evidence.field_name or "/".join(evidence.related_fields[:2])
    return (
        f"{evidence.id} | kind={evidence.evidence_kind} | competitor={evidence.competitor or ''} | "
        f"tool={evidence.tool or ''} | field={field} | source_refs={source_refs} | content={_strip_prompt_links(_detail_text(evidence.excerpt, 900))}"
    )


def _signal_detail(signal: EvidenceSignal) -> dict[str, Any]:
    return {
        "id": signal.id,
        "provider": signal.provider,
        "signal_type": signal.signal_type,
        "competitor": signal.competitor,
        "value": _detail_text(signal.value),
        "confidence": signal.confidence,
        "evidence_ids": signal.evidence_ids,
    }


def _provider_result_detail(result: SourceProviderResult) -> dict[str, Any]:
    return {
        "provider": result.provider,
        "cache_hit": result.cache_hit,
        "query": result.query,
        "source_count": len(result.sources),
        "signal_count": len(result.signals),
        "errors": [_detail_text(error, 220) for error in result.errors[:5]],
        "omitted_error_count": max(0, len(result.errors) - 5),
    }


def _tool_claim_detail(claim) -> dict[str, Any]:
    return {
        "category": claim.category,
        "title": claim.title,
        "claim": _detail_text(claim.claim),
        "evidence_ids": claim.evidence_ids,
        "confidence": claim.confidence,
        "reasoning": _detail_text(claim.reasoning, 320),
    }


def _tool_result_detail(result: ToolExecutionResult) -> dict[str, Any]:
    return {
        "tool_name": result.tool_name,
        "canonical_tool_name": result.canonical_tool_name,
        "confidence": result.confidence,
        "generated_by": result.generated_by,
        "evidence_ids": result.evidence_ids,
        "reasoning": _detail_text(result.reasoning),
        "claims": _detail_items(result.claims, _tool_claim_detail),
    }


def _profile_detail(profile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "competitor_type": profile.competitor_type,
        "score": profile.score,
        "positioning": _detail_text(profile.positioning),
        "selection_reason": _detail_text(profile.selection_reason),
        "core_features": profile.core_features[:8],
        "evidence_ids": profile.evidence_ids,
    }


def _finding_detail(finding) -> dict[str, Any]:
    return {
        "id": finding.id,
        "category": finding.category,
        "title": finding.title,
        "conclusion": _detail_text(finding.conclusion),
        "confidence": finding.confidence,
        "impact": finding.impact,
        "evidence_ids": finding.evidence_ids,
    }


def _report_detail(report: CompetitorReport) -> dict[str, Any]:
    return {
        "target_product": report.target_product,
        "executive_summary": _detail_text(report.executive_summary),
        "competitors": _detail_items(report.competitors, _profile_detail),
        "findings": _detail_items(report.findings, _finding_detail),
        "recommendations": _detail_items(report.recommendations, lambda item: _detail_text(item, 260)),
        "evidence_count": len(report.evidence),
        "risk_notice": _detail_text(report.risk_notice),
    }


def _retarget_report_evidence_ids(report: CompetitorReport, evidence: list[EvidenceItem]) -> None:
    valid_ids = {item.id for item in evidence}
    extracted_ids = [item.id for item in evidence if item.evidence_kind == "extracted_fact"]
    fallback_ids = (extracted_ids or [item.id for item in evidence])[:3]

    def clean(ids: list[str]) -> list[str]:
        kept = [item for item in ids if item in valid_ids]
        return kept or fallback_ids

    for profile in report.competitors:
        profile.evidence_ids = clean(profile.evidence_ids)
    for finding in report.findings:
        finding.evidence_ids = clean(finding.evidence_ids)
    for cell in report.comparison_matrix:
        cell.evidence_ids = clean(cell.evidence_ids)
    for result in report.tool_results:
        result.evidence_ids = clean(result.evidence_ids)
        for claim in result.claims:
            claim.evidence_ids = clean(claim.evidence_ids)
    for sandbox in report.tool_sandboxes:
        sandbox.evidence_ids = clean(sandbox.evidence_ids)
        for claim in sandbox.claims:
            claim.evidence_ids = clean(claim.evidence_ids)


def _qa_detail(qa_result: QAResult) -> dict[str, Any]:
    return {
        "passed": qa_result.passed,
        "score": qa_result.score,
        "issues": _detail_items(
            qa_result.issues,
            lambda issue: {
                "severity": issue.severity,
                "field": issue.field,
                "message": issue.message,
            },
        ),
        "revision_request": (
            qa_result.revision_request.model_dump(mode="json")
            if qa_result.revision_request
            else None
        ),
    }


def _append_checkpoint_qa_history(
    state: WorkflowState,
    *,
    passed: bool,
    issues: list[str],
    field: str,
) -> list[dict[str, Any]]:
    qa_history = [QAResult.model_validate(item) for item in state.get("qa_history", [])]
    qa_history.append(
        QAResult(
            passed=passed,
            score=1.0 if passed and not issues else max(0.2, 0.88 - 0.12 * len(issues)),
            issues=[
                QAIssue(
                    severity="medium" if passed else "high",
                    field=field,
                    message=issue,
                )
                for issue in issues
            ],
        )
    )
    return [item.model_dump(mode="json") for item in qa_history]


def _attach_evidence_ids_to_provider_signals(
    provider_results: list[SourceProviderResult],
    sources: list[RawSource],
    evidence: list[EvidenceItem],
) -> list[SourceProviderResult]:
    by_url_title = {(source.url, source.title): item.id for source, item in zip(sources, evidence)}
    by_url = {source.url: item.id for source, item in zip(sources, evidence)}
    updated_results: list[SourceProviderResult] = []
    for result in provider_results:
        signals: list[EvidenceSignal] = []
        for signal in result.signals:
            if signal.evidence_ids:
                signals.append(signal)
                continue
            source_url = str(signal.metadata.get("source_url") or signal.metadata.get("url") or "")
            source_title = str(signal.metadata.get("source_title") or signal.metadata.get("title") or "")
            evidence_id = by_url_title.get((source_url, source_title)) or by_url.get(source_url)
            signals.append(signal.model_copy(update={"evidence_ids": [evidence_id] if evidence_id else []}))
        updated_results.append(result.model_copy(update={"signals": signals}))
    return updated_results


def _merge_search_plans(seed: SearchPlan, field_plan: SearchPlan) -> SearchPlan:
    seen: set[str] = set()
    queries: list[SearchQuery] = []
    for query in [*seed.queries, *field_plan.queries]:
        key = f"{query.query.lower()}|{query.tool or ''}|{query.competitor or ''}"
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    missing = _dedupe([*seed.missing_fields, *field_plan.missing_fields])
    return seed.model_copy(
        update={
            "queries": queries,
            "missing_fields": missing,
            "rationale": (
                f"{seed.rationale}\n{field_plan.rationale}".strip()
                or "Seed and field-level search plans were merged."
            ),
        }
    )


class AgentWorkflow:
    def __init__(self, settings: Settings, storage: Storage, event_bus: EventBus):
        self.settings = settings
        self.storage = storage
        self.event_bus = event_bus
        configure_llm_runtime(
            max_concurrency=settings.openai_max_concurrency,
            main_max_concurrency=settings.effective_openai_main_max_concurrency,
            qa_max_concurrency=settings.effective_openai_qa_max_concurrency,
            retry_max_concurrency=settings.effective_openai_retry_max_concurrency,
        )
        self.model = get_chat_model(settings)
        self.graph = self._build_graph()

    def _check_cancelled(self, task_id: str) -> None:
        task = self.storage.get_task(task_id)
        if task and task.status in {"cancelling", "cancelled"}:
            raise TaskCancelled()

    async def _llm_structured(
        self,
        task_id: str,
        node: str,
        schema: type[BaseModel],
        messages: list[tuple[str, str]],
        *,
        llm_lane: str = "main",
    ):
        self._check_cancelled(task_id)
        if not self.model:
            raise self._llm_decision(task_id, node, "No LLM model is configured.")
        started = time.perf_counter()
        call_path = self._save_llm_call_started(task_id, node, "structured", messages, schema=schema)
        try:
            try:
                result = await structured_ainvoke(self.model, schema, messages, lane=llm_lane)
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc) or "lane" not in str(exc):
                    raise
                result = await structured_ainvoke(self.model, schema, messages)
            self._save_llm_call_finished(task_id, call_path, output=result, started_at=started)
            return result
        except Exception as exc:
            self._save_llm_call_finished(
                task_id,
                call_path,
                error={"error": friendly_llm_error(exc), "raw_error": str(exc)},
                started_at=started,
            )
            task = self.storage.get_task(task_id)
            if task is None:
                raise
            raise self._llm_decision(task_id, node, friendly_llm_error(exc)) from exc

    async def _llm_text(self, task_id: str, node: str, messages: list[tuple[str, str]], *, llm_lane: str = "main"):
        self._check_cancelled(task_id)
        if not self.model:
            raise self._llm_decision(task_id, node, "No LLM model is configured.")
        started = time.perf_counter()
        call_path = self._save_llm_call_started(task_id, node, "text", messages)
        try:
            result = await retry_ainvoke(self.model, messages, lane=llm_lane)
            content = getattr(result, "content", result)
            self._save_llm_call_finished(task_id, call_path, output={"content": content}, started_at=started)
            return result
        except Exception as exc:
            self._save_llm_call_finished(
                task_id,
                call_path,
                error={"error": friendly_llm_error(exc), "raw_error": str(exc)},
                started_at=started,
            )
            task = self.storage.get_task(task_id)
            if task is None:
                raise
            raise self._llm_decision(task_id, node, friendly_llm_error(exc)) from exc

    def _llm_decision(self, task_id: str, node: str, reason: str) -> LLMDecisionNeeded:
        decision = DecisionRequest(
            node=node,
            question="大模型连续调用失败，是否继续当前任务？",
            reason=reason,
            context_summary=f"任务 {task_id} 在 {node} 调用大模型失败。可重试模型、使用规则降级继续，或中止任务。",
            options=[
                DecisionOption(label="重新运行并重试模型", value="retry_llm", description="检查模型配置或网络后，从任务入口重新运行并再次调用模型。"),
                DecisionOption(label="规则继续", value="continue_without_llm", description="关闭本次任务的大模型调用，使用规则和已收集来源继续。"),
                DecisionOption(label="中止任务", value="cancel", description="停止当前任务并保留已收集来源。"),
            ],
        )
        return LLMDecisionNeeded(decision)

    async def _generate_analysis_intent(
        self,
        task_id: str,
        request: CreateTaskRequest,
        benchmark: AnalysisBenchmark,
    ) -> AnalysisIntentJSON:
        fallback = _fallback_analysis_intent(request, benchmark)
        if not self.model or request.llm_failure_policy == "continue_without_llm":
            return fallback
        payload = {
            "request": request.model_dump(mode="json"),
            "benchmark": benchmark.model_dump(mode="json"),
            "instruction": analysis_intent_user_instruction,
        }
        try:
            return await self._llm_structured(
                task_id,
                "plan_agent",
                AnalysisIntentJSON,
                [
                    ("system", analysis_intent_system_prompt),
                    ("user", json.dumps(payload, ensure_ascii=False, indent=2)),
                ],
            )
        except LLMDecisionNeeded:
            raise
        except Exception as exc:
            logger.info("plan_agent analysis intent fallback for task %s: %s", task_id, exc)
            return fallback

    async def plan_agent(self, state: WorkflowState) -> WorkflowState:
        phase = state.get("workflow_phase", "phase_1_intent")
        if phase == "phase_3_strategy":
            return await self._plan_phase_3_strategy(state)
        return await self._plan_phase_1_intent(state)

    async def search_agent(self, state: WorkflowState) -> WorkflowState:
        phase = state.get("workflow_phase", "phase_2_search")
        if phase == "phase_4_collect":
            return await self._search_phase_4_collect(state)
        return await self._search_phase_2_candidates(state)

    async def _emit_candidate_search_progress(
        self,
        task_id: str,
        queries: list[str],
        candidates: list[CandidateCompetitor],
        *,
        notes: list[str] | None = None,
        started_at: float | None = None,
    ) -> None:
        await self._emit(
            task_id,
            "search_agent",
            "running",
            output_summary=f"已实时识别 {len(candidates)} 个候选竞品。",
            started_at=started_at,
            token_payload={"queries": queries, "candidates": candidates},
            details={
                "kind": "candidate_search",
                "progress": True,
                "candidates": _detail_items(candidates, _candidate_detail),
                "display_markdown": _candidate_search_progress_markdown(queries, candidates, notes),
            },
        )

    def _save_candidate_search_checkpoint(self, task_id: str, state: WorkflowState, **updates: Any) -> None:
        checkpoint = dict(state)
        checkpoint.update(updates)
        checkpoint["workflow_phase"] = "phase_2_search"
        self._save_checkpoint(task_id, checkpoint, "search_agent")

    async def _qa_verify_candidate_existence(
        self,
        task_id: str,
        request: CreateTaskRequest,
        candidates: list[CandidateCompetitor],
    ) -> tuple[list[CandidateCompetitor], list[str], list[RawSource]]:
        self._last_candidate_qa_change_log = CandidateQAChangeLog()
        if not candidates:
            return [], [], []
        valid_candidates: list[CandidateCompetitor] = []
        validation_notes: list[str] = []
        change_log = CandidateQAChangeLog()
        for candidate in candidates:
            if is_invalid_candidate_name(candidate.name, request.target_product):
                reason = "Removed by rule-based QA because the candidate name is malformed or generic."
                validation_notes.append(f"{candidate.name}: {reason}")
                change_log.removed_candidates.append(
                    {"name": candidate.name, "reason": reason, "source": "rule"}
                )
                continue
            valid_candidates.append(candidate)
        candidates = valid_candidates
        if not candidates:
            change_log.qa_notes = _dedupe(validation_notes)[:12]
            self._last_candidate_qa_change_log = change_log
            return [], change_log.qa_notes, []
        if not self.model or request.llm_failure_policy == "continue_without_llm":
            notes = [*validation_notes, "Candidate existence QA skipped because LLM is unavailable or disabled."]
            kept = []
            for candidate in candidates:
                kept.append(candidate)
            change_log.qa_notes = _dedupe(notes)[:12]
            kept = self._merge_candidates([], kept, change_log)
            self._last_candidate_qa_change_log = change_log
            return kept, change_log.qa_notes, []

        semaphore = asyncio.Semaphore(max(1, int(self.settings.candidate_qa_concurrency)))

        async def verify_one(index: int, candidate: CandidateCompetitor):
            async with semaphore:
                self._check_cancelled(task_id)
                try:
                    sources, errors = await collect_candidate_existence_sources_with_search_runner(
                        candidate.name,
                        request,
                        self.settings,
                        task_id=task_id,
                        max_sources=6,
                        should_cancel=lambda: self._check_cancelled(task_id),
                    )
                    item_notes: list[str] = []
                    if errors:
                        item_notes.append(f"{candidate.name}: existence search returned {len(errors)} error(s).")
                    usable_sources = [source for source in sources if (source.content or "").strip()]
                    if not usable_sources:
                        attempted_count = max(len(sources), 6)
                        reason = (
                            "Removed because existence search could not fetch usable page content "
                            f"from {attempted_count} results."
                        )
                        item_notes.append(f"{candidate.name}: {reason}")
                        return index, None, item_notes, sources, {
                            "removed": {"name": candidate.name, "reason": reason, "source": "search"},
                        }
                    qa_sources = usable_sources[:3]
                    output = await self._llm_structured(
                        task_id,
                        "qa_agent",
                        CandidateExistenceQAOutput,
                        [
                            (
                                "system",
                                candidate_existence_qa_system_prompt
                                + "\nCurrent candidates for alias/merge checks: "
                                + json.dumps(
                                    [
                                        _candidate_existence_candidate_payload(item)
                                        for item in candidates
                                        if item.name != candidate.name
                                    ],
                                    ensure_ascii=False,
                                ),
                            ),
                            (
                                "user",
                                json.dumps(
                                    {
                                        "target_product": request.target_product,
                                        "candidate": _candidate_existence_candidate_payload(candidate),
                                        "web_results": [_candidate_existence_source_payload(source) for source in qa_sources],
                                        "instruction": candidate_existence_qa_instruction,
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                            ),
                        ],
                        llm_lane=QA_LANE,
                    )
                    if output.exists:
                        candidate_urls = _dedupe([*candidate.source_urls, *output.matched_urls, *[source.url for source in qa_sources]])
                        matched = set(output.matched_urls or candidate_urls)
                        source_refs = list(candidate.source_refs)
                        for source in qa_sources:
                            if output.matched_urls and source.url not in matched:
                                continue
                            source_refs.append(
                                SourceRef(
                                    source_id=source.id,
                                    chunk_id=source.metadata.get("chunk_id"),
                                    url=source.url,
                                    title=source.title,
                                    quote=_detail_text(source.content, 500),
                                    evidence_screenshot_path=source.metadata.get("evidence_screenshot_path") or source.metadata.get("quote_shot_path"),
                                )
                            )
                        updated_name, merge_warning = _resolve_candidate_merge_target(candidate, output, candidates)
                        updated_name = updated_name.strip() or candidate.name
                        if merge_warning:
                            item_notes.append(f"{candidate.name}: {merge_warning}.")
                        if is_invalid_candidate_name(updated_name, request.target_product):
                            reason = "Canonical name returned by QA is malformed or generic."
                            item_notes.append(f"{candidate.name}: removed by QA ({reason}).")
                            return index, None, item_notes, sources, {
                                "removed": {"name": candidate.name, "reason": reason, "source": "llm"},
                            }
                        changes: dict[str, Any] = {}
                        if updated_name != candidate.name:
                            changes["alias_decision"] = {
                                "from_name": candidate.name,
                                "to_name": updated_name,
                                "reason": output.reason or "QA canonicalized or merged an alias.",
                            }
                        if output.corrected_competitor_type and output.corrected_competitor_type != candidate.competitor_type:
                            changes["type_correction"] = {
                                "name": updated_name,
                                "from_type": candidate.competitor_type,
                                "to_type": output.corrected_competitor_type,
                                "reason": output.reason or "QA corrected competitor type.",
                            }
                        updated = candidate.model_copy(
                            update={
                                "name": updated_name,
                                "competitor_type": output.corrected_competitor_type or candidate.competitor_type,
                                "source_urls": candidate_urls[:12],
                                "source_titles": _dedupe([*candidate.source_titles, *[source.title for source in qa_sources]])[:12],
                                "source_refs": source_refs[:12],
                                "tags": _dedupe(
                                    [
                                        *candidate.tags,
                                        "qa_verified",
                                        *(["qa_merged_alias"] if output.merge_target else []),
                                    ]
                                ),
                                "score": max(candidate.score, round(output.confidence, 4)),
                                "selection_reason": candidate.selection_reason or output.reason,
                            }
                        )
                        return index, updated, item_notes, qa_sources, changes
                    reason = output.reason or "existence not confirmed"
                    item_notes.append(f"{candidate.name}: removed by QA ({reason}).")
                    return index, None, item_notes, qa_sources, {
                        "removed": {"name": candidate.name, "reason": reason, "source": "llm"},
                    }
                except Exception as exc:
                    reason = f"Removed because existence QA failed after retries: {friendly_llm_error(exc)}"
                    if is_invalid_candidate_name(candidate.name, request.target_product):
                        reason = "Removed by rule-based QA after QA failure."
                        return index, None, [f"{candidate.name}: {reason}"], [], {
                            "removed": {"name": candidate.name, "reason": reason, "source": "rule"},
                        }
                    return index, None, [f"{candidate.name}: {reason}"], [], {
                        "removed": {"name": candidate.name, "reason": reason, "source": "llm"},
                    }

        results = await asyncio.gather(*(verify_one(index, candidate) for index, candidate in enumerate(candidates)))
        verified: list[CandidateCompetitor] = []
        notes: list[str] = []
        qa_sources: list[RawSource] = []
        alias_decisions: dict[str, str] = {}
        for _index, candidate, item_notes, sources, changes in sorted(results, key=lambda item: item[0]):
            if candidate is not None:
                verified.append(candidate)
            notes.extend(item_notes)
            qa_sources.extend(sources)
            if changes.get("removed"):
                change_log.removed_candidates.append(changes["removed"])
            if changes.get("alias_decision"):
                decision = changes["alias_decision"]
                from_name = str(decision.get("from_name") or "")
                to_name = str(decision.get("to_name") or "")
                if from_name and to_name:
                    alias_decisions[from_name] = to_name
            if changes.get("type_correction"):
                change_log.type_corrections.append(changes["type_correction"])
            if changes.get("unverified"):
                change_log.unverified_candidates.append(changes["unverified"])
        change_log.qa_notes = _dedupe([*validation_notes, *notes])[:12]
        verified = _apply_candidate_alias_decisions(verified, alias_decisions, change_log)
        self._last_candidate_qa_change_log = change_log
        return verified, change_log.qa_notes, qa_sources

    async def analyst_agent(self, state: WorkflowState) -> WorkflowState:
        phase = state.get("workflow_phase", "phase_2_analyze")
        if phase == "phase_4_synthesis":
            return await self._analyst_phase_4_synthesis(state)
        if "tool_results" in state and "phase_2" not in phase:
            return await self._legacy_analyst_agent(state)
        if "structured_report" in state and "benchmark" not in state:
            return await self._legacy_analyst_agent(state)
        if "structured_report" in state and "phase_2" not in phase and "phase_4" not in phase:
            return await self._legacy_analyst_agent(state)
        return await self._analyst_phase_2_scoring(state)

    async def qa_agent(self, state: WorkflowState) -> WorkflowState:
        if state.get("workflow_phase") in {"phase_2_search", "phase_4_collect"} and not state.get("qa_checkpoint"):
            return {}
        checkpoint = state.get("qa_checkpoint")
        if checkpoint == "checkpoint_1":
            return await self._qa_checkpoint_1(state)
        if checkpoint == "checkpoint_2":
            return await self._qa_checkpoint_2(state)
        return await self._legacy_qa_agent(state)

    async def writer_agent(self, state: WorkflowState) -> WorkflowState:
        if "state_tree" in state and "analysis_report" not in state:
            return await self._writer_final_report(state)
        return await self._legacy_writer_agent(state)

    async def _plan_phase_1_intent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        state_tree = state.get("state_tree") or create_initial_state_tree(request)
        started = time.perf_counter()
        await self._emit(task_id, "plan_agent", "running", "阶段一：解析意图、生命周期与分析目的。")

        benchmark, clarification = infer_benchmark(request)
        if self.model and request.llm_failure_policy != "continue_without_llm":
            try:
                prompt = render_prompt(
                    phase_1_plan_intent,
                    phase_input=request.model_dump(mode="json"),
                )
                patch = await self._llm_structured(
                    task_id,
                    "plan_agent",
                    StateTreePatch,
                    [("system", plan_system_prompt), ("user", prompt)],
                )
                state_tree = merge_updated_nodes(state_tree, patch.model_dump(mode="json"))
            except LLMDecisionNeeded:
                raise
            except Exception as exc:
                logger.info("plan_agent phase 1 model fallback for task %s: %s", task_id, exc)

        if benchmark:
            request.target_product = request.target_product or benchmark.target_product
            analysis_intent = await self._generate_analysis_intent(task_id, request, benchmark)
            if not _has_valid_phase_1_intent(state_tree):
                state_tree = merge_updated_nodes(
                    state_tree,
                    {"updated_nodes": {"phase_1_intent": benchmark_to_phase_1(benchmark)}},
                )
            state_tree.setdefault("phase_1_intent", {})
            state_tree["phase_1_intent"]["analysis_intent"] = analysis_intent.model_dump(mode="json")
            state_tree["phase_1_intent"].update(
                {
                    "product_lifecycle": benchmark.lifecycle_stage,
                    "analysis_purpose": analysis_intent.analysis_purpose,
                    "specific_goal": analysis_intent.goal or analysis_intent.target or benchmark.analysis_goal,
                    "core_problem_to_solve": "；".join(analysis_intent.problems)
                    or analysis_intent.goal
                    or benchmark.analysis_goal,
                    "target_product": analysis_intent.target_product or benchmark.target_product,
                    "confidence": analysis_intent.confidence,
                }
            )
            state_tree["global_metadata"]["status"] = "phase_1_completed"
            state_tree["global_metadata"]["product_name"] = benchmark.target_product
            await self._emit(
                task_id,
                "plan_agent",
                "completed",
                output_summary=f"阶段一完成：{benchmark.target_product} / {benchmark.lifecycle_stage} / {benchmark.analysis_purpose}。",
                started_at=started,
                token_payload={"state_tree": state_tree, "analysis_intent": analysis_intent},
                details={
                    "kind": "state_tree_phase_1",
                    "benchmark": _benchmark_detail(benchmark),
                    "analysis_intent": analysis_intent.model_dump(mode="json"),
                },
            )
            return {
                "request": request.model_dump(mode="json"),
                "analysis_intent": analysis_intent.model_dump(mode="json"),
                "benchmark": benchmark.model_dump(mode="json"),
                "state_tree": state_tree,
                "workflow_phase": "phase_2_search",
                "phase2_supplement_count": 0,
            }

        assert clarification is not None
        await self._emit(
            task_id,
            "plan_agent",
            "waiting",
            message=clarification.question,
            output_summary="阶段一信息不足，等待用户澄清。",
            started_at=started,
            token_payload=clarification,
            details={"kind": "clarification", "clarification": _clarification_detail(clarification)},
        )
        raise ClarificationNeeded(clarification)

    async def _search_phase_2_candidates(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        self._check_cancelled(task_id)
        request = CreateTaskRequest.model_validate(state["request"]) if state.get("request") else CreateTaskRequest(target_product="unknown")
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        target_count = max(1, min(30, int(request.candidate_target_count)))
        iteration = int(state.get("candidate_search_iterations", 0))
        previous_sources = [RawSource.model_validate(item) for item in state.get("candidate_sources", [])]
        previous_candidates = [CandidateCompetitor.model_validate(item) for item in state.get("candidates", [])]
        resume_stage = str(state.get("candidate_search_stage") or "")
        started = time.perf_counter()
        await self._emit(task_id, "search_agent", "running", "阶段二：全网泛搜并建立候选竞品池。")

        resume_after_search = resume_stage in {"search_complete", "chunk_analysis_complete"} and bool(previous_sources)
        if resume_after_search:
            queries = _dedupe([str(item) for item in state.get("candidate_active_queries", [])])
            if not queries:
                queries = _dedupe([str(item) for item in state.get("candidate_query_history", [])])[-self.settings.max_queries_per_generation :]
            if not queries:
                queries = build_candidate_queries(benchmark, request)[: self.settings.max_queries_per_generation]
            query_mode = str(state.get("candidate_query_mode") or "checkpoint_resume")
        else:
            queries, query_mode = await self._generate_candidate_search_queries(state, benchmark, request)
        seen_url_keys = set(str(item) for item in state.get("candidate_seen_url_keys", []))
        candidate_tasks: list[asyncio.Task] = []
        chunk_model = None if request.llm_failure_policy == "continue_without_llm" else self.model
        progress_candidates = list(previous_candidates)
        progress_sources = list(previous_sources)
        progress_lock = asyncio.Lock()

        async def analyze_sources_as_they_arrive(items: list[RawSource]) -> None:
            nonlocal progress_sources
            async with progress_lock:
                progress_sources = dedupe_sources([*progress_sources, *items])[:90]
                source_snapshot = list(progress_sources)
                candidate_snapshot = list(progress_candidates)
                self._save_candidate_search_checkpoint(
                    task_id,
                    state,
                    candidate_search_stage="search_progress",
                    candidate_active_queries=list(queries),
                    candidate_query_mode=query_mode,
                    candidate_sources=[source.model_dump(mode="json") for source in source_snapshot],
                    candidates=[candidate.model_dump(mode="json") for candidate in candidate_snapshot],
                    candidate_seen_url_keys=sorted(seen_url_keys),
                )
            if not chunk_model:
                return
            async def run_and_emit():
                nonlocal progress_candidates
                async with progress_lock:
                    existing_names = [candidate.name for candidate in progress_candidates]
                result = await generate_candidate_pool_from_chunks_with_llm_or_rules(
                    chunk_model,
                    benchmark,
                    request,
                    items,
                    max_candidates=target_count,
                    task_output_dir=self.settings.task_output_dir,
                    task_id=task_id,
                    existing_candidate_names=existing_names,
                )
                chunk_candidates, notes, _mode = result
                if chunk_candidates:
                    async with progress_lock:
                        progress_candidates = self._merge_candidates(progress_candidates, chunk_candidates)
                        self._save_candidate_search_checkpoint(
                            task_id,
                            state,
                            candidate_search_stage="chunk_analysis_progress",
                            candidate_active_queries=list(queries),
                            candidate_query_mode=query_mode,
                            candidate_sources=[source.model_dump(mode="json") for source in progress_sources],
                            candidates=[candidate.model_dump(mode="json") for candidate in progress_candidates],
                            candidate_seen_url_keys=sorted(seen_url_keys),
                        )
                        await self._emit_candidate_search_progress(
                            task_id,
                            queries,
                            progress_candidates,
                            notes=notes,
                            started_at=started,
                        )
                return result

            candidate_tasks.append(asyncio.create_task(run_and_emit()))

        search_errors = list(state.get("candidate_search_errors") or [])
        if resume_after_search:
            sources = dedupe_sources(previous_sources)[:90]
        else:
            new_sources, search_errors = await collect_candidate_sources_with_search_runner(
                queries,
                request,
                self.settings,
                task_id=task_id,
                max_sources=50,
                seen_url_keys=seen_url_keys,
                should_cancel=lambda: self._check_cancelled(task_id),
                on_sources=analyze_sources_as_they_arrive,
            )
            sources = dedupe_sources([*previous_sources, *new_sources])[:90]
            query_history_after_search = _dedupe([*state.get("candidate_query_history", []), *queries])
            self._save_candidate_search_checkpoint(
                task_id,
                state,
                candidate_search_stage="search_complete",
                candidate_active_queries=list(queries),
                candidate_query_mode=query_mode,
                candidate_sources=[source.model_dump(mode="json") for source in sources],
                candidates=[candidate.model_dump(mode="json") for candidate in progress_candidates],
                candidate_query_history=query_history_after_search,
                candidate_search_errors=search_errors[:12],
                candidate_seen_url_keys=sorted(seen_url_keys),
            )
        if resume_stage == "chunk_analysis_complete" and previous_candidates:
            candidates = previous_candidates
            candidate_pool_notes = list(state.get("candidate_pool_notes") or [])
            candidate_pool_mode = str(state.get("candidate_pool_mode") or "checkpoint_resume")
        else:
            try:
                if candidate_tasks:
                    task_results = await asyncio.gather(*candidate_tasks)
                    task_candidates = [candidate for chunk_candidates, _notes, _mode in task_results for candidate in chunk_candidates]
                    task_notes = [note for _chunk_candidates, notes, _mode in task_results for note in notes]
                    candidates = self._merge_candidates(previous_candidates, task_candidates)
                    candidate_pool_notes = _dedupe(task_notes)[:8]
                    candidate_pool_mode = "llm_candidate_chunk_stream"
                else:
                    candidates, candidate_pool_notes, candidate_pool_mode = await generate_candidate_pool_from_chunks_with_llm_or_rules(
                        chunk_model,
                        benchmark,
                    request,
                    sources,
                    max_candidates=target_count,
                        task_output_dir=self.settings.task_output_dir,
                        task_id=task_id,
                        existing_candidate_names=[candidate.name for candidate in previous_candidates],
                    )
            except Exception as exc:
                raise self._llm_decision(task_id, "search_agent", friendly_llm_error(exc)) from exc
            candidates = self._merge_candidates(previous_candidates, candidates)
            self._save_candidate_search_checkpoint(
                task_id,
                state,
                candidate_search_stage="chunk_analysis_complete",
                candidate_active_queries=list(queries),
                candidate_query_mode=query_mode,
                candidate_sources=[source.model_dump(mode="json") for source in sources],
                candidates=[candidate.model_dump(mode="json") for candidate in candidates],
                candidate_pool_mode=candidate_pool_mode,
                candidate_pool_notes=candidate_pool_notes,
                candidate_query_history=_dedupe([*state.get("candidate_query_history", []), *queries]),
                candidate_search_errors=search_errors[:12],
                candidate_seen_url_keys=sorted(seen_url_keys),
            )
        candidates, qa_notes, qa_sources = await self._qa_verify_candidate_existence(task_id, request, candidates)
        qa_change_log = getattr(self, "_last_candidate_qa_change_log", CandidateQAChangeLog())
        if qa_notes:
            candidate_pool_notes = _dedupe([*candidate_pool_notes, *qa_notes])[:12]
        if qa_sources:
            sources = dedupe_sources([*sources, *qa_sources])[:90]
        await self._emit_candidate_search_progress(
            task_id,
            queries,
            candidates,
            notes=candidate_pool_notes,
            started_at=started,
        )
        query_history = _dedupe([*state.get("candidate_query_history", []), *queries])
        await self._emit(
            task_id,
            "search_agent",
            "completed",
            output_summary=f"基于网页来源生成 {len(candidates)} 个候选竞品，来源 {len(sources)} 条。",
            started_at=started,
            token_payload={"queries": queries, "query_mode": query_mode, "candidates": candidates},
            details={
                "kind": "candidate_search",
                "iteration": iteration + 1,
                "query_mode": query_mode,
                "candidate_pool_mode": candidate_pool_mode,
                "candidate_pool_notes": candidate_pool_notes,
                "search_errors": search_errors[:8],
                "queries": _detail_items(list(queries), lambda query: query),
                "sources": _detail_items(sources, _source_detail),
                "candidates": _detail_items(candidates, _candidate_detail),
                "candidate_qa_change_log": qa_change_log.model_dump(mode="json"),
            },
        )
        return {
            "candidate_sources": [source.model_dump(mode="json") for source in sources],
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            "candidate_pool_mode": candidate_pool_mode,
            "candidate_pool_notes": candidate_pool_notes,
            "candidate_search_iterations": iteration + 1,
            "candidate_query_history": query_history,
            "candidate_gap_request": None,
            "candidate_qa_change_log": qa_change_log.model_dump(mode="json"),
            "candidate_seen_url_keys": sorted(seen_url_keys),
            "workflow_phase": "phase_2_analyze",
        }

    async def _generate_candidate_search_queries(
        self,
        state: WorkflowState,
        benchmark: AnalysisBenchmark,
        request: CreateTaskRequest,
    ) -> tuple[list[str], str]:
        fallback = build_candidate_queries(benchmark, request)
        if request.llm_failure_policy == "continue_without_llm":
            return fallback[: self.settings.max_queries_per_generation], "rule_fallback_llm_disabled"
        if not self.model:
            if self.storage.get_task(state["task_id"]):
                raise self._llm_decision(state["task_id"], "search_agent", "No LLM model is configured.")
            return fallback[: self.settings.max_queries_per_generation], "rule_fallback_no_llm"
        payload = {
            "analysis_intent": benchmark.model_dump(mode="json"),
            "request_context": request.model_dump(mode="json"),
            "current_candidates": state.get("candidates", [])[:30],
            "query_history": state.get("candidate_query_history", []),
            "candidate_gap_request": state.get("candidate_gap_request"),
        }
        try:
            output = await self._llm_structured(
                state["task_id"],
                "search_agent",
                CandidateSearchQueryOutput,
                [
                    ("system", candidate_query_generation_prompt),
                    ("user", json.dumps(payload, ensure_ascii=False, indent=2)),
                ],
            )
            queries = _dedupe([item.query for item in output.queries if item.query.strip()])
            history = {item.lower() for item in state.get("candidate_query_history", [])}
            fresh_queries = [item for item in queries if item.lower() not in history]
            if fresh_queries:
                return fresh_queries[: self.settings.max_queries_per_generation], "llm_generated_queries"
            if queries:
                return queries[: self.settings.max_queries_per_generation], "llm_generated_queries_reused"
        except LLMDecisionNeeded:
            raise
        except Exception as exc:
            logger.info("search_agent candidate query LLM fallback: %s", exc)
        return fallback[: self.settings.max_queries_per_generation], "rule_fallback_after_llm_error"

    def _merge_candidates(
        self,
        previous: list[CandidateCompetitor],
        current: list[CandidateCompetitor],
        change_log: CandidateQAChangeLog | None = None,
    ) -> list[CandidateCompetitor]:
        return _apply_candidate_alias_decisions([*previous, *current], {}, change_log)

    def _renormalize_phase2_candidate_state(
        self,
        state: WorkflowState,
        state_tree: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        if not state.get("benchmark") or not state.get("candidates"):
            return state_tree, {}, []
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        change_log = CandidateQAChangeLog.model_validate(state.get("candidate_qa_change_log") or {})
        candidates = [CandidateCompetitor.model_validate(item) for item in state.get("candidates", [])]
        selected_set = (
            SelectedCompetitorSet.model_validate(state["selected_competitor_set"])
            if state.get("selected_competitor_set")
            else None
        )
        names = [candidate.name for candidate in candidates]
        if selected_set:
            names.extend(candidate.name for candidate in selected_set.selected)
        duplicate_groups = _alias_duplicate_groups(names)
        canonical_candidates = self._merge_candidates([], candidates, change_log)
        changed = bool(duplicate_groups) or len(canonical_candidates) != len(candidates)
        if not changed:
            return state_tree, {}, []

        selected_set = _select_top_competitors_by_category(benchmark, canonical_candidates)
        brief_profiles = _sync_brief_profile_names(
            [CompetitorBriefProfile.model_validate(item) for item in state.get("competitor_brief_profiles", [])],
            canonical_candidates,
        )
        state_tree = merge_updated_nodes(
            state_tree,
            {"updated_nodes": {"phase_2_targets": selected_set_to_phase_2(selected_set)}},
        )
        notes = [
            "Candidate aliases were re-normalized before checkpoint QA: "
            + "; ".join(" / ".join(group) for group in duplicate_groups)
        ]
        change_log.qa_notes = _dedupe([*change_log.qa_notes, *notes])[:12]
        updates: dict[str, Any] = {
            "candidates": [candidate.model_dump(mode="json") for candidate in canonical_candidates],
            "selected_competitor_set": selected_set.model_dump(mode="json"),
            "candidate_qa_change_log": change_log.model_dump(mode="json"),
        }
        if brief_profiles:
            updates["competitor_brief_profiles"] = [profile.model_dump(mode="json") for profile in brief_profiles]
        return state_tree, updates, notes

    def _candidate_gap_from_exception(
        self,
        benchmark: AnalysisBenchmark,
        candidates: list[CandidateCompetitor],
        exc: Exception,
    ) -> CandidateGapRequest:
        return CandidateGapRequest(
            missing_competitor_types=self._missing_candidate_types([]),
            missing_factors=["selection_failure", "heat", "growth", "similarity", "strict_source_refs"],
            suggested_query_focus=[
                f"{benchmark.target_product} alternatives competitors",
                f"{benchmark.target_product} market map emerging competitors",
                f"{benchmark.target_product} funding growth downloads reviews",
            ],
            rationale=f"Analyst selection failed after {len(candidates)} candidates were generated: {friendly_llm_error(exc)}",
        )

    def _candidate_gap_from_selection(
        self,
        benchmark: AnalysisBenchmark,
        selected_set: SelectedCompetitorSet,
        candidates: list[CandidateCompetitor],
    ) -> CandidateGapRequest:
        weak_sources = [item.name for item in candidates if not item.source_refs][:8]
        focus = [
            f"{benchmark.target_product} alternatives competitors",
            f"{benchmark.target_product} competitor comparison recent funding growth",
        ]
        for item in selected_set.selected[:5]:
            focus.append(f"{benchmark.target_product} {item.name} alternative similar product")
        return CandidateGapRequest(
            missing_competitor_types=_missing_phase2_selection_types(benchmark, selected_set.selected),
            missing_factors=["heat", "growth", "similarity"] + (["strict_source_refs"] if weak_sources else []),
            suggested_query_focus=_dedupe(focus),
            rationale=(
                f"Only {len(selected_set.selected)} competitors were selected. "
                f"Candidates with weak source_refs: {', '.join(weak_sources) or 'none'}."
            ),
        )

    def _missing_candidate_types(self, selected: list[CandidateCompetitor]) -> list[Literal["head_direct", "challenger", "indirect_cross", "potential_substitute"]]:
        expected: list[Literal["head_direct", "challenger", "indirect_cross", "potential_substitute"]] = [
            "head_direct",
            "challenger",
            "indirect_cross",
            "potential_substitute",
        ]
        present = {item.competitor_type for item in selected}
        return [item for item in expected if item not in present]

    async def _collect_competitor_brief_profiles(
        self,
        task_id: str,
        request: CreateTaskRequest,
        benchmark: AnalysisBenchmark,
        candidates: list[CandidateCompetitor],
        sources: list[RawSource],
    ) -> list[CompetitorBriefProfile]:
        all_sources = list(sources)
        if self.model and request.llm_failure_policy != "continue_without_llm" and candidates:
            fallback_queries = [
                f"{candidate.name} popularity growth trend market traction"
                for candidate in candidates
            ]
            try:
                plan = await self._llm_structured(
                    task_id,
                    "analyst_agent",
                    DetailQueryPlan,
                        [
                        ("system", brief_query_template_system_prompt),
                        (
                            "user",
                            json.dumps(
                                {
                                    "analysis_intent": state_safe_intent_payload(request, benchmark),
                                    "competitors": [candidate.name for candidate in candidates],
                                    "required_fields": ["market_heat", "development_status"],
                                    "instruction": brief_query_template_instruction,
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                        ),
                    ],
                )
                templates = [item.template for item in plan.competitor_query_templates if item.template.strip()]
                queries = []
                for candidate in candidates:
                    for template in (templates or ["{competitor} popularity growth trend market traction"])[:2]:
                        queries.append(
                            template
                            .replace("{{product_name}}", candidate.name)
                            .replace("{competitor}", candidate.name)
                        )
                if queries:
                    fresh_sources, _errors = await collect_candidate_sources_with_search_runner(
                        _dedupe(queries),
                        request,
                        self.settings,
                        task_id=task_id,
                        max_sources=max(5, min(80, len(queries) * 5)),
                        seen_url_keys=set(),
                        should_cancel=lambda: self._check_cancelled(task_id),
                    )
                    all_sources = dedupe_sources([*all_sources, *fresh_sources])[:160]
            except LLMDecisionNeeded:
                raise
            except Exception as exc:
                logger.info("brief profile query/search fallback for task %s: %s", task_id, exc)

        profiles: list[CompetitorBriefProfile] = []
        for candidate in candidates:
            matched = _sources_for_competitor_name(candidate.name, all_sources)
            refs = [_source_ref_from_raw_source(source, quote=_detail_text(source.content, 500)) for source in matched[:5]]
            market_text = _brief_field_text(candidate, matched, ["market", "popular", "download", "user", "review", "榜", "用户", "热度"])
            growth_text = _brief_field_text(candidate, matched, ["growth", "funding", "news", "release", "hiring", "增长", "融资", "更新"])
            if not market_text:
                market_text = candidate.description or "公开来源中暂未识别出明确市场热度信号。"
            if not growth_text:
                growth_text = candidate.selection_reason or "公开来源中暂未识别出明确发展情况信号。"
            missing = []
            if not refs:
                missing.extend(["market_heat", "development_status"])
            profiles.append(
                CompetitorBriefProfile(
                    competitor=candidate.name,
                    market_heat=ResearchField(value=market_text, confidence=0.68 if refs else 0.28, source_refs=refs[:3]),
                    development_status=ResearchField(value=growth_text, confidence=0.68 if refs else 0.28, source_refs=refs[:3]),
                    source_refs=refs,
                    qa_passed=bool(refs),
                    missing_fields=_dedupe(missing),
                )
            )
        return profiles

    def _market_growth_info_for_qa(self, candidate: CandidateCompetitor, profile: CompetitorBriefProfile) -> dict[str, Any]:
        return {
            "product_name": candidate.name,
            "market_popularity": {
                "summary": profile.market_heat.value,
                "facts": [profile.market_heat.value] if profile.market_heat.value else [],
                "updated": None,
                "main_evidence_refs": [ref.model_dump(mode="json") for ref in profile.market_heat.source_refs],
                "data_gaps": [item for item in profile.missing_fields if item == "market_heat"],
            },
            "growth_factor": {
                "summary": profile.development_status.value,
                "facts": [profile.development_status.value] if profile.development_status.value else [],
                "updated": None,
                "main_evidence_refs": [ref.model_dump(mode="json") for ref in profile.development_status.source_refs],
                "data_gaps": [item for item in profile.missing_fields if item == "development_status"],
            },
            "update_notes": "Generated from brief profile evidence before phase-2 scoring.",
        }

    def _rule_market_growth_qa(
        self,
        candidate: CandidateCompetitor,
        profile: CompetitorBriefProfile,
    ) -> MarketGrowthQAOutput:
        missing_fields = list(profile.missing_fields)
        if not profile.market_heat.value.strip():
            missing_fields.append("market_popularity.summary")
        if not profile.development_status.value.strip():
            missing_fields.append("growth_factor.summary")
        if not profile.market_heat.source_refs:
            missing_fields.append("market_popularity.main_evidence_refs")
        if not profile.development_status.source_refs:
            missing_fields.append("growth_factor.main_evidence_refs")

        suspicious: list[str] = []
        strong_fact_pattern = r"\d|融资|排名|MAU|DAU|用户|营收|增长|下载|估值|轮|million|billion|users|revenue|growth|funding"
        if re.search(strong_fact_pattern, profile.market_heat.value, re.I) and not profile.market_heat.source_refs:
            suspicious.append("market_popularity contains strong facts without evidence refs.")
        if re.search(strong_fact_pattern, profile.development_status.value, re.I) and not profile.development_status.source_refs:
            suspicious.append("growth_factor contains strong facts without evidence refs.")

        market_supported = bool(profile.market_heat.source_refs)
        growth_supported = bool(profile.development_status.source_refs)
        needs_rework = len(missing_fields) >= 3 or bool(suspicious)
        return MarketGrowthQAOutput(
            product_name=candidate.name,
            is_accurate=not suspicious,
            has_missing_fields=bool(missing_fields),
            has_suspicious_fields=bool(suspicious),
            needs_rework=needs_rework,
            dimension_qa={
                "market_popularity": MarketGrowthDimensionQA(
                    is_complete=bool(profile.market_heat.value.strip()),
                    is_evidence_supported=market_supported,
                    is_consistent=True,
                    issues=[item for item in missing_fields if item.startswith("market_popularity")]
                    + [item for item in suspicious if item.startswith("market_popularity")],
                ),
                "growth_factor": MarketGrowthDimensionQA(
                    is_complete=bool(profile.development_status.value.strip()),
                    is_evidence_supported=growth_supported,
                    is_consistent=True,
                    issues=[item for item in missing_fields if item.startswith("growth_factor")]
                    + [item for item in suspicious if item.startswith("growth_factor")],
                ),
            },
            missing_fields=_dedupe(missing_fields),
            suspicious_fields=suspicious,
            rework_queries_needed=needs_rework,
            rework_focus=_dedupe(["market_heat", "growth_factor"] if needs_rework else []),
            qa_reason=(
                "市场热度与增速资料存在缺失或无引用强事实，建议补充。"
                if needs_rework
                else "市场热度与增速资料可支持阶段二打分。"
            ),
        )

    async def _qa_market_growth_profiles(
        self,
        task_id: str,
        request: CreateTaskRequest,
        benchmark: AnalysisBenchmark,
        candidates: list[CandidateCompetitor],
        brief_profiles: list[CompetitorBriefProfile],
    ) -> list[CompetitorBriefProfile]:
        by_name = {_candidate_key(candidate.name): candidate for candidate in candidates}
        checked: list[CompetitorBriefProfile] = []
        for profile in brief_profiles:
            candidate = by_name.get(_candidate_key(profile.competitor))
            if candidate is None:
                checked.append(profile)
                continue
            qa_output = self._rule_market_growth_qa(candidate, profile)
            if self.model and request.llm_failure_policy != "continue_without_llm":
                try:
                    product_info = {
                        "name": candidate.name,
                        "canonical_name": candidate.name,
                        "competitor_type": candidate.competitor_type,
                        "description": _strip_prompt_links(candidate.description),
                        "source_refs": [ref.model_dump(mode="json") for ref in candidate.source_refs[:5]],
                    }
                    market_growth_info = self._market_growth_info_for_qa(candidate, profile)
                    qa_output = await self._llm_structured(
                        task_id,
                        "qa_agent",
                        MarketGrowthQAOutput,
                        [
                            ("system", qa_system_prompt),
                            (
                                "user",
                                qa_agent_prompt_7.format(
                                    product_info=json.dumps(product_info, ensure_ascii=False, indent=2),
                                    market_growth_info=json.dumps(market_growth_info, ensure_ascii=False, indent=2),
                                ),
                            ),
                        ],
                        llm_lane=QA_LANE,
                    )
                except LLMDecisionNeeded:
                    raise
                except Exception as exc:
                    logger.info("market/growth QA fallback for %s: %s", candidate.name, exc)
            missing = _dedupe([*profile.missing_fields, *qa_output.missing_fields])
            checked.append(
                profile.model_copy(
                    update={
                        "qa_passed": not qa_output.needs_rework,
                        "missing_fields": missing,
                        "qa_result": qa_output.model_dump(mode="json"),
                    }
                )
            )
        return checked

    def _rule_competitor_relevance_score(
        self,
        benchmark: AnalysisBenchmark,
        candidate: CandidateCompetitor,
        profile: CompetitorBriefProfile | None,
    ) -> CompetitorRelevanceScore:
        scoring = score_and_select_candidates(benchmark, [candidate.model_copy(deep=True)], budget=1).scoring_profile
        type_weight = scoring.type_weights.get(candidate.competitor_type, 1.0)
        factors = candidate.rank_factors
        heat = max(factors.heat, profile.market_heat.confidence if profile else 0.0)
        growth = max(factors.growth, profile.development_status.confidence if profile else 0.0)
        similarity = factors.similarity
        base = scoring.alpha_heat * heat + scoring.beta_growth * growth + scoring.gamma_similarity * similarity
        total_score = round(min(1.0, type_weight * base), 4)
        weight_assignment = {
            "W_type": round(type_weight, 4),
            "alpha_heat": round(scoring.alpha_heat, 4),
            "beta_growth": round(scoring.beta_growth, 4),
            "gamma_similarity": round(scoring.gamma_similarity, 4),
        }
        factor_scores = {
            "f_heat": round(heat, 4),
            "f_growth": round(growth, 4),
            "f_similarity": round(similarity, 4),
            "f_risk": round(candidate.rank_factors.risk, 4),
        }
        reason = {
            "summary": f"{candidate.name} 在当前模式下综合分 {total_score:.2f}，按三因子公式计算，风险因子未参与总分。",
            "details": {
                "heat_analysis": (profile.market_heat.value if profile else candidate.description) or "未识别到明确热度信息。",
                "growth_analysis": (profile.development_status.value if profile else candidate.selection_reason) or "未识别到明确增速信息。",
                "similarity_analysis": f"相似度因子来自目标产品描述与竞品简介的文本重合度，当前得分 {similarity:.2f}。",
                "risk_analysis": "风险因子仅作为兼容字段保留，不参与阶段二评分公式。",
            },
        }
        return CompetitorRelevanceScore(
            competitor=candidate.name,
            total_score=total_score,
            category_score=heat,
            detailed_reason=reason["summary"],
            suitable_for_deep_analysis=not profile.qa_result.get("needs_rework", False) if profile else True,
            weight_assignment=weight_assignment,
            factor_scores=factor_scores,
            scoring_reason=reason,
        )

    def _normalize_phase2_score_output(
        self,
        fallback: CompetitorRelevanceScore,
        output: CompetitorRelevanceScore,
    ) -> CompetitorRelevanceScore:
        weights = output.weight_assignment or fallback.weight_assignment
        factors = output.factor_scores or fallback.factor_scores
        w_type = float(weights.get("W_type", fallback.weight_assignment.get("W_type", 1.0)) or 1.0)
        alpha = float(weights.get("alpha_heat", fallback.weight_assignment.get("alpha_heat", 0.0)) or 0.0)
        beta = float(weights.get("beta_growth", fallback.weight_assignment.get("beta_growth", 0.0)) or 0.0)
        gamma = float(weights.get("gamma_similarity", fallback.weight_assignment.get("gamma_similarity", 0.0)) or 0.0)
        heat = float(factors.get("f_heat", fallback.factor_scores.get("f_heat", 0.0)) or 0.0)
        growth = float(factors.get("f_growth", fallback.factor_scores.get("f_growth", 0.0)) or 0.0)
        similarity = float(factors.get("f_similarity", fallback.factor_scores.get("f_similarity", 0.0)) or 0.0)
        risk = float(factors.get("f_risk", fallback.factor_scores.get("f_risk", 0.0)) or 0.0)
        total_score = round(min(1.0, max(0.0, w_type * (alpha * heat + beta * growth + gamma * similarity))), 4)
        reason = output.scoring_reason or fallback.scoring_reason
        if not output.scoring_reason:
            reason = fallback.scoring_reason
        return output.model_copy(
            update={
                "total_score": total_score,
                "category_score": round(min(1.0, max(0.0, heat)), 4),
                "weight_assignment": {
                    "W_type": round(w_type, 4),
                    "alpha_heat": round(alpha, 4),
                    "beta_growth": round(beta, 4),
                    "gamma_similarity": round(gamma, 4),
                },
                "factor_scores": {
                    "f_heat": round(min(1.0, max(0.0, heat)), 4),
                    "f_growth": round(min(1.0, max(0.0, growth)), 4),
                    "f_similarity": round(min(1.0, max(0.0, similarity)), 4),
                    "f_risk": round(min(1.0, max(0.0, risk)), 4),
                },
                "scoring_reason": reason,
                "detailed_reason": output.detailed_reason or fallback.detailed_reason,
            }
        )

    async def _score_candidates_with_brief_profiles(
        self,
        task_id: str,
        request: CreateTaskRequest,
        benchmark: AnalysisBenchmark,
        candidates: list[CandidateCompetitor],
        brief_profiles: list[CompetitorBriefProfile],
    ) -> list[CandidateCompetitor]:
        by_name = {_candidate_key(profile.competitor): profile for profile in brief_profiles}
        scored: list[CandidateCompetitor] = []
        for candidate in candidates:
            profile = by_name.get(_candidate_key(candidate.name))
            fallback_score_output = self._rule_competitor_relevance_score(benchmark, candidate, profile)
            score_output = fallback_score_output
            if self.model and request.llm_failure_policy != "continue_without_llm" and profile:
                try:
                    score_output = await self._llm_structured(
                        task_id,
                        "analyst_agent",
                        CompetitorRelevanceScore,
                        [
                            ("system", competitor_relevance_score_system_prompt),
                            (
                                "user",
                                json.dumps(
                                    {
                                        "analysis_intent": state_safe_intent_payload(request, benchmark),
                                        "competitor": candidate.model_dump(mode="json"),
                                        "brief_profile_clean": _brief_profile_clean(profile),
                                        "formula": "Score = W_type * (alpha_heat*f_heat + beta_growth*f_growth + gamma_similarity*f_similarity)",
                                        "risk_rule": "f_risk is compatibility-only and must not affect total_score.",
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                            ),
                        ],
                    )
                except LLMDecisionNeeded:
                    raise
                except Exception as exc:
                    logger.info("candidate relevance score fallback for %s: %s", candidate.name, exc)
            score_output = self._normalize_phase2_score_output(fallback_score_output, score_output)
            rank_factors = candidate.rank_factors.model_copy(
                update={
                    "heat": round(score_output.factor_scores.get("f_heat", score_output.category_score), 4),
                    "growth": round(score_output.factor_scores.get("f_growth", score_output.total_score), 4),
                    "similarity": round(score_output.factor_scores.get("f_similarity", candidate.rank_factors.similarity), 4),
                }
            )
            scored.append(
                candidate.model_copy(
                    update={
                        "score": round(score_output.total_score, 4),
                        "rank_factors": rank_factors,
                        "selection_reason": score_output.detailed_reason or candidate.selection_reason,
                        "weight_assignment": score_output.weight_assignment,
                        "factor_scores": score_output.factor_scores,
                        "scoring_reason": score_output.scoring_reason,
                        "tags": _dedupe(
                            [
                                *candidate.tags,
                                "brief_scored",
                                *(["brief_qa_rework"] if profile and profile.qa_result.get("needs_rework") else []),
                                *(["not_suitable_for_deep_analysis"] if not score_output.suitable_for_deep_analysis else []),
                            ]
                        ),
                    }
                )
            )
        return scored

    async def _analyst_phase_2_scoring(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        candidates = [CandidateCompetitor.model_validate(item) for item in state.get("candidates", [])]
        candidates = self._merge_candidates([], candidates)
        sources = [RawSource.model_validate(item) for item in state.get("candidate_sources", [])]
        state_tree = state.get("state_tree") or {}
        started = time.perf_counter()
        await self._emit(task_id, "analyst_agent", "running", "阶段二：逐个质检热度/增速信息、逐个打分并锁定 3 个目标竞品。")

        brief_profiles = [CompetitorBriefProfile.model_validate(item) for item in state.get("competitor_brief_profiles", [])]
        brief_profiles = _sync_brief_profile_names(brief_profiles, candidates)
        if not brief_profiles:
            brief_profiles = await self._collect_competitor_brief_profiles(task_id, request, benchmark, candidates, sources)
        brief_profiles = await self._qa_market_growth_profiles(task_id, request, benchmark, candidates, brief_profiles)
        qa_passed_names = {_candidate_key(profile.competitor) for profile in brief_profiles if profile.qa_passed}
        if len(qa_passed_names) >= 3:
            score_candidates = [
                candidate
                for candidate in candidates
                if _candidate_key(candidate.name) in qa_passed_names
            ]
        else:
            score_candidates = candidates
        candidates = await self._score_candidates_with_brief_profiles(task_id, request, benchmark, candidates, brief_profiles)
        scored_by_name = {_candidate_key(candidate.name): candidate for candidate in candidates}
        score_candidates = [scored_by_name[_candidate_key(candidate.name)] for candidate in score_candidates if _candidate_key(candidate.name) in scored_by_name]
        selected_set = _select_top_competitors_by_category(benchmark, score_candidates)

        if (len(selected_set.selected) < 3 or selected_set.gap_request) and int(state.get("candidate_search_iterations", 0)) < 3:
            gap_request = selected_set.gap_request or self._candidate_gap_from_selection(benchmark, selected_set, candidates)
            await self._emit(
                task_id,
                "analyst_agent",
                "revision",
                output_summary=f"Only {len(selected_set.selected)} reliable competitors selected; pushing a gap request back to search_agent.",
                started_at=started,
                token_payload={"selected_set": selected_set, "brief_profiles": brief_profiles},
                details={
                    "kind": "target_selection_gap",
                    "selected_count": len(selected_set.selected),
                    "gap_request": gap_request.model_dump(mode="json"),
                    "brief_profiles": _detail_items(brief_profiles, lambda item: item.model_dump(mode="json")),
                },
            )
            return {
                "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                "competitor_brief_profiles": [item.model_dump(mode="json") for item in brief_profiles],
                "selected_competitor_set": selected_set.model_dump(mode="json"),
                "candidate_gap_request": gap_request.model_dump(mode="json"),
                "workflow_phase": "phase_2_search",
            }

        if len(selected_set.selected) < 3:
            selected_set.fallback_notes = _dedupe(
                [*selected_set.fallback_notes, "Candidate search retry limit reached; final fewer-than-three selection released with gap_request."]
            )

        state_tree = merge_updated_nodes(
            state_tree,
            {"updated_nodes": {"phase_2_targets": selected_set_to_phase_2(selected_set)}},
        )
        state_tree["global_metadata"]["status"] = "phase_2_targets_locked"
        await self._emit(
            task_id,
            "analyst_agent",
            "completed",
            output_summary="锁定竞品阵型：" + "、".join(item.name for item in selected_set.selected),
            started_at=started,
            token_payload={"selected_set": selected_set, "brief_profiles": brief_profiles},
            details={
                "kind": "target_selection",
                "selection_rule": "purpose_specific_three_competitors_without_llm_selection",
                "selected": _detail_items(selected_set.selected, _candidate_detail),
                "candidates": _detail_items(selected_set.candidates, _candidate_detail),
                "brief_profiles": _detail_items(brief_profiles, lambda item: item.model_dump(mode="json")),
                "state_tree_phase": "phase_2_targets",
            },
        )
        return {
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            "competitor_brief_profiles": [item.model_dump(mode="json") for item in brief_profiles],
            "selected_competitor_set": selected_set.model_dump(mode="json"),
            "candidate_gap_request": selected_set.gap_request.model_dump(mode="json") if selected_set.gap_request else None,
            "state_tree": state_tree,
            "qa_checkpoint": "checkpoint_1",
        }

    def _ensure_candidate_budget(
        self,
        benchmark: AnalysisBenchmark,
        candidates: list[CandidateCompetitor],
        budget: int,
    ) -> list[CandidateCompetitor]:
        if len(candidates) >= budget:
            return candidates
        padded = list(candidates)
        type_cycle: list[Literal["head_direct", "challenger", "indirect_cross", "potential_substitute"]] = [
            "head_direct",
            "challenger",
            "indirect_cross",
            "potential_substitute",
            "challenger",
        ]
        existing = {item.name for item in padded}
        index = 1
        while len(padded) < budget:
            name = f"{benchmark.target_product} 待补充竞品 {index}"
            index += 1
            if name in existing:
                continue
            competitor_type = type_cycle[(len(padded)) % len(type_cycle)]
            padded.append(
                CandidateCompetitor(
                    name=name,
                    description="外部数据不足时生成的低置信度占位候选，需要采集 Agent 后续补齐真实公开来源。",
                    competitor_type=competitor_type,
                    tags=["fallback_placeholder"],
                    selection_reason="候选池不足时生成的低置信度占位候选，后续报告会标注低置信度。",
                )
            )
            existing.add(name)
        return padded

    async def _qa_checkpoint_1(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        state_tree = state["state_tree"]
        started = time.perf_counter()
        await self._emit(task_id, "qa_agent", "running", "质检点 1：审查竞品阵型与分析目的是否匹配。")

        inspection = QAInspectionOutput(inspection_status="pass")
        request = CreateTaskRequest.model_validate(state["request"]) if state.get("request") else CreateTaskRequest(target_product="unknown")
        normalization_updates: dict[str, Any] = {}
        normalization_notes: list[str] = []
        state_tree, normalization_updates, normalization_notes = self._renormalize_phase2_candidate_state(state, state_tree)
        if self.model and request.llm_failure_policy != "continue_without_llm":
            try:
                prompt = render_prompt(
                    phase_2_qa_intent,
                    phase_1_intent=state_tree.get("phase_1_intent", {}),
                    phase_2_targets=state_tree.get("phase_2_targets", {}),
                )
                inspection = await self._llm_structured(
                    task_id,
                    "qa_agent",
                    QAInspectionOutput,
                    [("system", qa_system_prompt), ("user", prompt)],
                    llm_lane=QA_LANE,
                )
            except LLMDecisionNeeded:
                raise
            except Exception as exc:
                logger.info("qa_agent checkpoint 1 model fallback for task %s: %s", task_id, exc)

        selected = state_tree.get("phase_2_targets", {}).get("selected_competitors", [])
        issues = list(inspection.issues_found)
        if len(selected) != 3:
            issues.append(f"Expected exactly 3 selected competitors, got {len(selected)}.")
        for item in selected:
            if not str(item.get("selection_reason") or "").strip():
                issues.append(f"{item.get('competitor_name') or 'unknown'} missing selection_reason.")
            if not item.get("source_refs") and not item.get("source_urls") and not item.get("source_titles"):
                issues.append(f"{item.get('competitor_name') or 'unknown'} missing source_refs/source_urls/source_titles.")
        passed = inspection.inspection_status == "pass" and not issues
        revision_count = int(state.get("revision_count", 0))
        max_revision_loops = int(state.get("max_revision_loops", self.settings.max_revision_loops))
        if not passed and revision_count >= max_revision_loops:
            issues.append("Revision budget reached; checkpoint is released with risk notice.")
            passed = True
        state_tree["quality_gates"]["checkpoint_1"] = {
            "inspection_status": "pass" if passed else "reject",
            "issues_found": issues,
            "feedback": inspection.feedback_to_analyzer,
            "normalization_notes": normalization_notes,
        }
        await self._emit(
            task_id,
            "qa_agent",
            "completed" if passed else "revision",
            output_summary="质检点 1 通过。" if passed else "质检点 1 未通过，回到分析 Agent。",
            started_at=started,
            token_payload=state_tree["quality_gates"]["checkpoint_1"],
            details={"kind": "qa_checkpoint_1", "passed": passed, "issues": issues},
        )
        return {
            **normalization_updates,
            "state_tree": state_tree,
            "workflow_phase": (
                "survey_design"
                if passed and not state.get("survey_design") and state.get("enable_survey_design_agent")
                else ("phase_3_strategy" if passed else "phase_2_analyze")
            ),
            "qa_checkpoint": None,
            "qa_history": _append_checkpoint_qa_history(
                state,
                passed=passed,
                issues=issues,
                field="checkpoint_1",
            ),
            "revision_count": revision_count if passed else revision_count + 1,
        }

    async def _plan_phase_3_strategy(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        selected_set = SelectedCompetitorSet.model_validate(state["selected_competitor_set"])
        state_tree = state["state_tree"]
        started = time.perf_counter()
        await self._emit(task_id, "plan_agent", "running", "阶段三：动态选择分析框架并下发采集清单。")

        tool_plan = build_tool_plan(benchmark, selected_set, request)
        collection_plan = build_collection_plan(benchmark, selected_set, tool_plan)
        llm_phase_3_strategy: dict[str, Any] | None = None
        if self.model and request.llm_failure_policy != "continue_without_llm":
            try:
                prompt = render_prompt(
                    phase_3_plan_intent,
                    phase_1_intent=state_tree.get("phase_1_intent", {}),
                    phase_2_targets=state_tree.get("phase_2_targets", {}),
                    available_canvas_tools=AVAILABLE_CANVAS_TOOLS,
                    available_methods=AVAILABLE_METHODS,
                    max_canvas_tools=2,
                    max_methods=4,
                )
                patch = await self._llm_structured(
                    task_id,
                    "plan_agent",
                    StateTreePatch,
                    [("system", plan_system_prompt), ("user", prompt)],
                )
                llm_phase_3_strategy = patch.model_dump(mode="json").get("updated_nodes", {}).get("phase_3_strategy")
                state_tree = merge_updated_nodes(state_tree, patch.model_dump(mode="json"))
            except LLMDecisionNeeded:
                raise
            except Exception as exc:
                logger.info("plan_agent phase 3 model fallback for task %s: %s", task_id, exc)
        merged_phase_3 = merge_phase_3_strategy(
            llm_phase_3_strategy,
            tool_plan,
            collection_plan,
            max_canvas_tools=2,
            max_methods=5,
        )
        merged_tools = merged_phase_3.get("selected_tools") or []
        merged_methods = merged_phase_3.get("selected_methods") or []
        merged_tool_plan = tool_plan.model_copy(
            update={
                "focus": merged_phase_3.get("analysis_focus") or tool_plan.focus,
                "frameworks": merged_tools,
                "methods": merged_methods,
                "selected_tools": [*merged_tools, *merged_methods],
                "rationale": merged_phase_3.get("selection_rationale") or tool_plan.rationale,
            }
        )
        collection_plan = build_collection_plan(benchmark, selected_set, merged_tool_plan)
        merged_phase_3 = merge_phase_3_strategy(
            llm_phase_3_strategy,
            merged_tool_plan,
            collection_plan,
            max_canvas_tools=2,
            max_methods=5,
        )
        state_tree = merge_updated_nodes(
            state_tree,
            {"updated_nodes": {"phase_3_strategy": merged_phase_3}},
        )
        state_tree["global_metadata"]["status"] = "phase_3_strategy_ready"
        await self._emit(
            task_id,
            "plan_agent",
            "completed",
            output_summary=f"选定分析工具：{', '.join(merged_tool_plan.selected_tools)}。",
            started_at=started,
            token_payload=state_tree.get("phase_3_strategy"),
            details={"kind": "strategy_planning", "state_tree_phase": "phase_3_strategy"},
        )
        return {
            "tool_plan": merged_tool_plan.model_dump(mode="json"),
            "collection_plan": collection_plan.model_dump(mode="json"),
            "state_tree": state_tree,
            "workflow_phase": "phase_4_collect",
            "detail_seen_url_keys": [],
            "industry_seen_url_keys": [],
        }

    async def survey_design_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        selected_set = SelectedCompetitorSet.model_validate(state["selected_competitor_set"])
        started = time.perf_counter()
        await self._emit(task_id, "survey_design_agent", "running", "根据已锁定竞品生成用户问卷设计。")
        design = generate_survey_design(
            self.model,
            task_id,
            request,
            benchmark,
            selected_set,
            {
                "phase_1_intent": state.get("state_tree", {}).get("phase_1_intent", {}),
                "phase_2_targets": state.get("state_tree", {}).get("phase_2_targets", {}),
            },
        )
        if hasattr(design, "__await__"):
            design = await design
        assert isinstance(design, SurveyDesign)
        self.storage.save_survey_design(design)
        self._save_survey_design_snapshot(task_id, design)
        await self._emit(
            task_id,
            "survey_design_agent",
            "completed",
            output_summary=f"已生成 {len(design.questions)} 道题的用户问卷设计。",
            started_at=started,
            token_payload=design,
            details={
                "kind": "survey_design",
                "survey_design": design.model_dump(mode="json"),
                "display_markdown": _survey_design_markdown(design),
                "ui_stage_key": "survey_design",
                "ui_title": "问卷设计",
            },
        )
        return {
            "survey_design": design.model_dump(mode="json"),
            "workflow_phase": "phase_3_strategy",
        }

    async def _search_phase_4_collect(self, state: WorkflowState) -> WorkflowState:
        collector_updates = await self.source_collector_agent(state)
        state.update(collector_updates)
        state_tree = state["state_tree"]
        sources = [RawSource.model_validate(item) for item in state.get("raw_sources", [])]
        evidence = [EvidenceItem.model_validate(item) for item in state.get("evidence", [])]
        state_tree = merge_updated_nodes(
            state_tree,
            {"updated_nodes": {"raw_data_sandbox": sources_to_sandbox(sources, evidence)}},
        )
        state_tree["global_metadata"]["status"] = "raw_data_collected"
        return {
            **collector_updates,
            "state_tree": state_tree,
            "workflow_phase": "phase_4_synthesis",
        }

    def _missing_collection_evidence(self, state: WorkflowState) -> list[str]:
        if state.get("field_extraction_results"):
            return []
        if not state.get("selected_competitor_set"):
            return []
        selected_set = SelectedCompetitorSet.model_validate(state["selected_competitor_set"])
        sources = [RawSource.model_validate(item) for item in state.get("raw_sources", [])]
        missing: list[str] = []
        for competitor in selected_set.selected:
            if not self._has_collection_evidence_for(competitor.name, sources):
                missing.append(competitor.name)
        return missing

    def _has_collection_evidence_for(self, name: str, sources: list[RawSource]) -> bool:
        key = name.lower()
        for source in sources:
            text = f"{source.metadata.get('competitor') or ''} {source.title} {source.content}".lower()
            has_match = key in text
            if not has_match:
                continue
            has_body = bool((source.content or "").strip())
            has_shot = bool(
                source.metadata.get("evidence_screenshot_path")
                or source.metadata.get("quote_shot_path")
                or source.metadata.get("screenshot_path")
                or source.metadata.get("image_path")
            )
            if has_body or has_shot:
                return True
        return False

    async def _analyst_phase_4_synthesis(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        phase_step = state.get("phase_4_synthesis_step")
        if not state.get("tool_results") or phase_step == "tool_executor_agent":
            state["phase_4_synthesis_step"] = "tool_executor_agent"
            self._save_checkpoint(task_id, state, "analyst_agent")
            tool_updates = await self.tool_executor_agent(state)
            state.update(tool_updates)
            state["phase_4_synthesis_step"] = "analyst_agent"
            self._save_checkpoint(task_id, state, "analyst_agent")
        else:
            tool_updates = {}

        if not state.get("analysis_report") or state.get("phase_4_synthesis_step") == "analyst_agent":
            state["phase_4_synthesis_step"] = "analyst_agent"
            self._save_checkpoint(task_id, state, "analyst_agent")
            analysis_updates = await self._legacy_analyst_agent(state)
            state.update(analysis_updates)
        else:
            analysis_updates = {}

        state_tree = state["state_tree"]
        report = CompetitorReport.model_validate(state["analysis_report"])
        state_tree = merge_updated_nodes(
            state_tree,
            {"updated_nodes": {"phase_4_synthesis": report_to_phase_4(report)}},
        )
        state_tree["global_metadata"]["status"] = "phase_4_synthesis_completed"
        state.pop("phase_4_synthesis_step", None)
        return {
            **tool_updates,
            **analysis_updates,
            "state_tree": state_tree,
            "qa_checkpoint": "checkpoint_2",
            "phase_4_synthesis_step": None,
        }

    async def _qa_checkpoint_2(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        state_tree = state["state_tree"]
        request = CreateTaskRequest.model_validate(state["request"])
        started = time.perf_counter()
        await self._emit(task_id, "qa_agent", "running", "质检点 2：全局一致性和防幻觉审查。")

        inspection = QAInspectionOutput(inspection_status="pass")
        if self.model and request.llm_failure_policy != "continue_without_llm":
            try:
                prompt = render_prompt(
                    phase_4_qa_intent,
                    complete_state_tree_json=_compact_for_llm(state_tree, max_text=700, max_list=10),
                )
                inspection = await self._llm_structured(
                    task_id,
                    "qa_agent",
                    QAInspectionOutput,
                    [("system", qa_system_prompt), ("user", prompt)],
                    llm_lane=QA_LANE,
                )
            except LLMDecisionNeeded:
                raise
            except Exception as exc:
                logger.info("qa_agent checkpoint 2 model fallback for task %s: %s", task_id, exc)

        issues = list(inspection.issues_found)
        dynamic_analyses = state_tree.get("phase_4_synthesis", {}).get("dynamic_analyses", [])
        conclusion = state_tree.get("phase_4_synthesis", {}).get("conclusion", {})
        expected_units = state_tree.get("phase_3_strategy", {}).get("selected_analysis_units") or []
        produced_units = [
            item.get("analysis_unit_name") or item.get("framework_name")
            for item in dynamic_analyses
            if isinstance(item, dict)
        ]
        if not dynamic_analyses:
            issues.append("phase_4_synthesis.dynamic_analyses is empty.")
        missing_units = [unit for unit in expected_units if unit not in produced_units]
        if missing_units:
            issues.append(f"phase_4_synthesis.dynamic_analyses missing units: {', '.join(missing_units)}.")
        if not conclusion.get("competitive_strategy"):
            issues.append("phase_4_synthesis.conclusion.competitive_strategy is empty.")
        passed = inspection.inspection_status == "pass" and not issues
        revision_count = int(state.get("revision_count", 0))
        max_revision_loops = int(state.get("max_revision_loops", self.settings.max_revision_loops))
        if not passed and revision_count >= max_revision_loops:
            issues.append("Revision budget reached; checkpoint is released with risk notice.")
            passed = True
        state_tree["quality_gates"]["checkpoint_2"] = {
            "inspection_status": "pass" if passed else "reject",
            "issues_found": issues,
            "feedback": inspection.feedback_to_planner_or_analyzer,
        }
        await self._emit(
            task_id,
            "qa_agent",
            "completed" if passed else "revision",
            output_summary="质检点 2 通过。" if passed else "质检点 2 未通过，回到分析 Agent。",
            started_at=started,
            token_payload=state_tree["quality_gates"]["checkpoint_2"],
            details={"kind": "qa_checkpoint_2", "passed": passed, "issues": issues},
        )
        return {
            "state_tree": state_tree,
            "workflow_phase": "writer_final" if passed else "phase_4_synthesis",
            "qa_checkpoint": None,
            "qa_history": _append_checkpoint_qa_history(
                state,
                passed=passed,
                issues=issues,
                field="checkpoint_2",
            ),
            "revision_count": revision_count if passed else revision_count + 1,
        }

    async def _writer_final_report(self, state: WorkflowState) -> WorkflowState:
        writer_updates = await self._legacy_writer_agent(state)
        report = CompetitorReport.model_validate(writer_updates["report"])
        if state.get("qa_history"):
            report.qa_history = [QAResult.model_validate(item) for item in state.get("qa_history", [])]
            writer_updates["report"] = report.model_dump(mode="json")
        state_tree = state["state_tree"]
        report.report_references = report.report_references or _build_report_references(report)
        report.deep_dive_markdown = _ensure_markdown_reference_links(report.deep_dive_markdown, report)
        if not report.report_chapters:
            report.report_chapters = [
                ReportChapterDraft(
                    title="最终报告",
                    markdown=report.deep_dive_markdown,
                    reference_ids=[ref.id for ref in report.report_references],
                )
            ]
        writer_updates["report"] = report.model_dump(mode="json")
        state_tree["final_report_markdown"] = report.deep_dive_markdown
        state_tree["global_metadata"]["status"] = "completed"
        return {**writer_updates, "state_tree": state_tree}

    def _ensure_planning_context(
        self,
        state: WorkflowState,
        request: CreateTaskRequest,
        sources: list[RawSource] | None = None,
    ) -> tuple[AnalysisBenchmark, SelectedCompetitorSet, ToolPlan, CollectionPlan]:
        benchmark = (
            AnalysisBenchmark.model_validate(state["benchmark"])
            if state.get("benchmark")
            else None
        )
        if benchmark is None:
            inferred, _ = infer_benchmark(request)
            benchmark = inferred or AnalysisBenchmark(
                target_product=request.target_product or request.our_product or "未命名产品",
                raw_description=request.raw_description or request.notes or request.target_product,
                lifecycle_stage=request.lifecycle_stage or "concept",
                analysis_purpose=request.analysis_purpose or "decision_support",
                analysis_goal=request.analysis_goal or "支撑竞品分析与产品决策",
                confidence=0.45,
                inferred_from=["compatibility_fallback"],
                locked=True,
            )

        if state.get("selected_competitor_set"):
            selected_set = SelectedCompetitorSet.model_validate(state["selected_competitor_set"])
        else:
            candidate_sources = sources or []
            candidates = _merge_knowledge_candidates(
                build_candidates_from_sources(benchmark, request, candidate_sources),
                state.get("reused_competitor_cards", []),
            )
            selected_set = _select_top_competitors_by_category(benchmark, candidates)

        tool_plan = (
            ToolPlan.model_validate(state["tool_plan"])
            if state.get("tool_plan")
            else build_tool_plan(benchmark, selected_set, request)
        )
        collection_plan = (
            CollectionPlan.model_validate(state["collection_plan"])
            if state.get("collection_plan")
            else build_collection_plan(benchmark, selected_set, tool_plan)
        )
        return benchmark, selected_set, tool_plan, collection_plan

    async def _emit(
        self,
        task_id: str,
        node: str,
        status: Literal["queued", "running", "waiting", "completed", "failed", "revision"],
        message: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        started_at: float | None = None,
        error: str | None = None,
        token_payload: Any | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        duration_ms = int((time.perf_counter() - started_at) * 1000) if started_at else None
        event_details = _event_display_details(node, status, message, output_summary, details, token_payload)
        event = AgentRunEvent(
            task_id=task_id,
            node=node,
            status=status,
            message=message,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=duration_ms,
            error=error,
            token_estimate=_estimate_tokens(token_payload) if token_payload is not None else None,
            details=event_details,
        )
        self._save_step_snapshot(event, token_payload=token_payload)
        await self.event_bus.publish(event)

    def _save_step_snapshot(self, event: AgentRunEvent, *, token_payload: Any | None = None) -> None:
        step_dir = self.settings.task_output_dir / event.task_id / "steps"
        try:
            index = sum(1 for path in step_dir.glob("*_event.json")) + 1 if step_dir.exists() else 1
            prefix = f"{index:03d}_{_safe_filename(event.node)}_{event.status}"
            event_payload = event.model_dump(mode="json")
            event_payload["output_file"] = f"{prefix}_output.json" if token_payload is not None else None
            display_markdown = event.details.get("display_markdown")
            event_payload["output_markdown_file"] = f"{prefix}_output.md" if display_markdown else None
            _write_json_file(step_dir / f"{prefix}_event.json", event_payload)
            if token_payload is not None:
                _write_json_file(step_dir / f"{prefix}_output.json", token_payload)
            if display_markdown:
                (step_dir / f"{prefix}_output.md").write_text(str(display_markdown).strip() + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save step snapshot for task %s node %s: %s", event.task_id, event.node, exc)

    def _save_llm_call_started(
        self,
        task_id: str,
        node: str,
        call_type: str,
        messages: list[tuple[str, str]],
        *,
        schema: type[BaseModel] | None = None,
        prompt_template: str | None = None,
    ) -> Path | None:
        call_dir = self.settings.task_output_dir / task_id / "llm_calls"
        try:
            index = sum(1 for path in call_dir.glob("*.json")) + 1 if call_dir.exists() else 1
            prefix = f"{index:03d}_{_safe_filename(node)}_{call_type}"
            system_messages = [content for role, content in messages if role == "system"]
            user_messages = [content for role, content in messages if role == "user"]
            payload: dict[str, Any] = {
                "task_id": task_id,
                "node": node,
                "call_type": call_type,
                "schema": schema.__name__ if schema is not None else None,
                "prompt_template": prompt_template,
                "system": system_messages[0] if system_messages else None,
                "user": "\n\n".join(user_messages),
                "messages": [{"role": role, "content": content} for role, content in messages],
                "output": None,
                "error": None,
                "duration_ms": None,
                "created_at": time.time(),
            }
            path = call_dir / f"{prefix}.json"
            _write_json_file(path, payload)
            return path
        except OSError as exc:
            logger.warning("Could not save LLM request snapshot for task %s node %s: %s", task_id, node, exc)
            return None

    def _save_llm_call_finished(
        self,
        task_id: str,
        path: Path | None,
        *,
        output: Any | None = None,
        error: Any | None = None,
        started_at: float | None = None,
    ) -> None:
        if path is None:
            return
        try:
            payload: dict[str, Any] = {}
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
            payload["output"] = _jsonable(output) if error is None else None
            payload["error"] = _jsonable(error)
            payload["duration_ms"] = int((time.perf_counter() - started_at) * 1000) if started_at else None
            payload["finished_at"] = time.time()
            _write_json_file(path, payload)
        except OSError as exc:
            logger.warning("Could not save LLM result snapshot for task %s call %s: %s", task_id, path, exc)
        except json.JSONDecodeError as exc:
            logger.warning("Could not update LLM snapshot for task %s call %s: %s", task_id, path, exc)

    def _save_llm_request_snapshot(
        self,
        task_id: str,
        node: str,
        call_type: str,
        messages: list[tuple[str, str]],
        *,
        schema: type[BaseModel] | None = None,
    ) -> str:
        path = self._save_llm_call_started(task_id, node, call_type, messages, schema=schema)
        return path.stem if path is not None else f"000_{_safe_filename(node)}_{call_type}"

    def _save_llm_result_snapshot(self, task_id: str, prefix: str, suffix: str, payload: Any) -> None:
        call_dir = self.settings.task_output_dir / task_id / "llm_calls"
        path = call_dir / f"{prefix}.json"
        if suffix == "output":
            self._save_llm_call_finished(task_id, path, output=payload)
        else:
            self._save_llm_call_finished(task_id, path, error=payload)

    def _save_final_report_snapshot(self, task_id: str, report: CompetitorReport) -> None:
        try:
            report_dir = self.settings.task_output_dir / task_id / "final"
            _write_json_file(report_dir / "report.json", report)
            (report_dir / "report.md").write_text(report_to_markdown(report), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save final report snapshot for task %s: %s", task_id, exc)

    def _save_survey_design_snapshot(self, task_id: str, design: SurveyDesign) -> None:
        try:
            survey_dir = self.settings.task_output_dir / task_id / "survey"
            _write_json_file(survey_dir / "design.json", design)
        except OSError as exc:
            logger.warning("Could not save survey design snapshot for task %s: %s", task_id, exc)

    def _checkpoint_path(self, task_id: str) -> Path:
        return self.settings.task_output_dir / task_id / "checkpoint.json"

    def _save_checkpoint(self, task_id: str, state: WorkflowState, next_node: str) -> None:
        payload = {
            "version": 1,
            "task_id": task_id,
            "next_node": next_node,
            "updated_at": time.time(),
            "state": _jsonable(state),
        }
        try:
            _write_json_file(self._checkpoint_path(task_id), payload)
        except OSError as exc:
            logger.warning("Could not save workflow checkpoint for task %s: %s", task_id, exc)

    def _load_checkpoint(self, task_id: str) -> tuple[WorkflowState, str] | None:
        path = self._checkpoint_path(task_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load workflow checkpoint for task %s: %s", task_id, exc)
            return None
        state = payload.get("state")
        next_node = payload.get("next_node")
        if not isinstance(state, dict) or next_node not in {"plan_agent", "search_agent", "analyst_agent", "qa_agent", "writer_agent"}:
            return None
        return state, str(next_node)

    def _initial_state(self, task_id: str, request: CreateTaskRequest, max_revision_loops: int) -> WorkflowState:
        initial: WorkflowState = {
            "task_id": task_id,
            "request": request.model_dump(mode="json"),
            "revision_count": 0,
            "max_revision_loops": max_revision_loops,
            "qa_history": [],
            "state_tree": create_initial_state_tree(request),
            "workflow_phase": "phase_1_intent",
            "enable_survey_design_agent": True,
            "phase2_supplement_count": 0,
            "candidate_search_iterations": 0,
            "candidate_query_history": [],
            "collection_retry_count": 0,
            "candidate_seen_url_keys": [],
            "detail_seen_url_keys": [],
            "industry_seen_url_keys": [],
            "competitor_brief_profiles": [],
            "competitor_detail_cards": [],
            "industry_card": None,
            "competitor_detail_cards_clean": [],
            "industry_card_clean": None,
            "report_references": [],
            "report_chapters": [],
            "knowledge_context": {"competitor_cards": [], "industry_cards": [], "source_ids": [], "notes": []},
            "reused_competitor_cards": [],
            "reused_industry_cards": [],
            "knowledge_sources": [],
            "candidate_qa_change_log": CandidateQAChangeLog().model_dump(mode="json"),
        }
        knowledge = retrieve_knowledge_context(self.storage, request)
        if knowledge.competitor_cards or knowledge.industry_cards:
            state_tree = dict(initial["state_tree"])
            state_tree["knowledge_base"] = knowledge.model_dump(mode="json")
            initial.update(
                {
                    "knowledge_context": knowledge.model_dump(mode="json"),
                    "reused_competitor_cards": [card.model_dump(mode="json") for card in knowledge.competitor_cards],
                    "reused_industry_cards": [card.model_dump(mode="json") for card in knowledge.industry_cards],
                    "knowledge_sources": knowledge.source_ids,
                    "state_tree": state_tree,
                }
            )
        return initial

    def _prepare_resumed_state(self, state: WorkflowState, task_id: str, request: CreateTaskRequest, max_revision_loops: int) -> WorkflowState:
        state["task_id"] = task_id
        state["request"] = request.model_dump(mode="json")
        state["max_revision_loops"] = max_revision_loops
        state.setdefault("revision_count", 0)
        state.setdefault("qa_history", [])
        state.setdefault("state_tree", create_initial_state_tree(request))
        state.setdefault("workflow_phase", "phase_1_intent")
        state.setdefault("candidate_search_iterations", 0)
        state.setdefault("candidate_query_history", [])
        state.setdefault("collection_retry_count", 0)
        state.setdefault("candidate_seen_url_keys", [])
        state.setdefault("detail_seen_url_keys", [])
        state.setdefault("industry_seen_url_keys", [])
        state.setdefault("competitor_brief_profiles", [])
        state.setdefault("competitor_detail_cards", [])
        state.setdefault("industry_card", None)
        state.setdefault("competitor_detail_cards_clean", [])
        state.setdefault("industry_card_clean", None)
        state.setdefault("report_references", [])
        state.setdefault("report_chapters", [])
        state.setdefault("knowledge_context", {"competitor_cards": [], "industry_cards": [], "source_ids": [], "notes": []})
        state.setdefault("reused_competitor_cards", [])
        state.setdefault("reused_industry_cards", [])
        state.setdefault("knowledge_sources", [])
        state.setdefault("candidate_qa_change_log", CandidateQAChangeLog().model_dump(mode="json"))
        return state

    async def perception_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        started = time.perf_counter()
        await self._emit(task_id, "perception_agent", "running", "解析产品描述、阶段、分析目的和目标。")

        benchmark, clarification = infer_benchmark(request)
        if benchmark:
            request.target_product = request.target_product or benchmark.target_product
            await self._emit(
                task_id,
                "perception_agent",
                "completed",
                output_summary=(
                    f"锁定分析基准：{benchmark.target_product} / {benchmark.lifecycle_stage} / "
                    f"{benchmark.analysis_purpose}。"
                ),
                started_at=started,
                token_payload=benchmark,
                details={
                    "kind": "analysis_benchmark",
                    "benchmark": _benchmark_detail(benchmark),
                },
            )
            return {
                "request": request.model_dump(mode="json"),
                "benchmark": benchmark.model_dump(mode="json"),
            }

        assert clarification is not None
        await self._emit(
            task_id,
            "perception_agent",
            "completed",
            output_summary=f"需要补充槽位：{', '.join(clarification.missing_slots)}。",
            started_at=started,
            token_payload=clarification,
            details={
                "kind": "clarification",
                "clarification": _clarification_detail(clarification),
            },
        )
        return {"clarification": clarification.model_dump(mode="json")}

    async def review_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        started = time.perf_counter()
        await self._emit(task_id, "review_agent", "running", "质检最低可行上下文。")
        if "benchmark" in state:
            benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
            await self._emit(
                task_id,
                "review_agent",
                "completed",
                output_summary=f"上下文达标，进入寻源定标。置信度 {benchmark.confidence:.2f}。",
                started_at=started,
                token_payload=benchmark,
                details={
                    "kind": "context_review",
                    "benchmark": _benchmark_detail(benchmark),
                },
            )
            return {}

        clarification = ClarificationRequest.model_validate(state["clarification"])
        await self._emit(
            task_id,
            "review_agent",
            "waiting",
            message=clarification.question,
            output_summary="任务已暂停，等待用户澄清。",
            started_at=started,
            token_payload=clarification,
            details={
                "kind": "clarification",
                "clarification": _clarification_detail(clarification),
            },
        )
        raise ClarificationNeeded(clarification)

    async def candidate_discovery_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        started = time.perf_counter()
        await self._emit(task_id, "candidate_discovery_agent", "running", "全网泛搜并召回候选竞品池。")

        queries = build_candidate_queries(benchmark, request)
        sources, _search_errors = await collect_candidate_sources_with_search_runner(
            queries,
            request,
            self.settings,
            task_id=task_id,
            max_sources=50,
        )
        candidates = _merge_knowledge_candidates(
            build_candidates_from_sources(benchmark, request, sources),
            state.get("reused_competitor_cards", []),
        )
        await self._emit(
            task_id,
            "candidate_discovery_agent",
            "completed",
            output_summary=f"召回 {len(candidates)} 个候选竞品，来源 {len(sources)} 条。",
            started_at=started,
            token_payload={"queries": queries, "candidates": candidates},
            details={
                "kind": "candidate_discovery",
                "queries": _detail_items(list(queries), lambda query: query),
                "sources": _detail_items(sources, _source_detail),
                "candidates": _detail_items(candidates, _candidate_detail),
            },
        )
        return {
            "candidate_sources": [source.model_dump(mode="json") for source in sources],
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        }

    async def ranking_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        candidates = [CandidateCompetitor.model_validate(item) for item in state.get("candidates", [])]
        started = time.perf_counter()
        await self._emit(task_id, "ranking_agent", "running", "按目的动态加权，筛选最终深度分析竞品。")

        selected_set = _select_top_competitors_by_category(benchmark, candidates)
        await self._emit(
            task_id,
            "ranking_agent",
            "completed",
            output_summary=(
                "锁定竞品阵型："
                + "、".join(f"{item.name}({item.score:.2f})" for item in selected_set.selected)
            ),
            started_at=started,
            token_payload=selected_set,
            details={
                "kind": "ranking",
                "scoring_profile": selected_set.scoring_profile.model_dump(mode="json"),
                "selected": _detail_items(selected_set.selected, _candidate_detail),
                "candidates": _detail_items(selected_set.candidates, _candidate_detail),
                "allocation": selected_set.allocation,
                "fallback_notes": selected_set.fallback_notes,
            },
        )
        return {"selected_competitor_set": selected_set.model_dump(mode="json")}

    async def planner_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        selected_set = SelectedCompetitorSet.model_validate(state["selected_competitor_set"])
        started = time.perf_counter()
        await self._emit(task_id, "planner_agent", "running", "根据分析目的选择工具链并生成采集清单。")

        tool_plan = build_tool_plan(benchmark, selected_set, request)
        collection_plan = build_collection_plan(benchmark, selected_set, tool_plan)
        await self._emit(
            task_id,
            "planner_agent",
            "completed",
            output_summary=f"选择工具：{', '.join(tool_plan.selected_tools)}。",
            started_at=started,
            token_payload={"tool_plan": tool_plan, "collection_plan": collection_plan},
            details={
                "kind": "tool_planning",
                "tool_plan": {
                    "purpose": tool_plan.purpose,
                    "focus": tool_plan.focus,
                    "frameworks": tool_plan.frameworks,
                    "methods": tool_plan.methods,
                    "selected_tools": tool_plan.selected_tools,
                    "rationale": tool_plan.rationale,
                },
                "collection_plan": {
                    "instructions": _detail_items(collection_plan.instructions, _collection_instruction_detail),
                    "provider_interfaces": collection_plan.provider_interfaces,
                    "notes": collection_plan.notes,
                },
            },
        )
        return {
            "tool_plan": tool_plan.model_dump(mode="json"),
            "collection_plan": collection_plan.model_dump(mode="json"),
        }

    async def search_planner_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        self._check_cancelled(task_id)
        started = time.perf_counter()
        await self._emit(task_id, "search_planner_agent", "running", "搜索规划步骤已下线，资料卡采集器会直接生成 query。")
        search_plan = SearchPlan(queries=[], rationale="搜索规划步骤已下线；资料卡采集器直接生成并执行 query。")
        await self._emit(
            task_id,
            "search_planner_agent",
            "completed",
            output_summary="搜索规划步骤已下线，未生成独立 search_plan。",
            started_at=started,
            token_payload=search_plan,
            details={
                "kind": "search_plan",
                "queries": _detail_items(search_plan.queries, _search_query_detail),
                "provider_order": search_plan.provider_order,
                "missing_fields": search_plan.missing_fields,
                "rationale": search_plan.rationale,
            },
        )
        return {"search_plan": search_plan.model_dump(mode="json")}

    async def source_collector_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        self._check_cancelled(task_id)
        request = CreateTaskRequest.model_validate(state["request"])
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        selected_set = SelectedCompetitorSet.model_validate(state["selected_competitor_set"])
        started = time.perf_counter()
        await self._emit(task_id, "source_collector_agent", "running", "按资料卡 query 逐条搜索、逐 chunk 填写资料卡。")

        sources: list[RawSource] = []
        provider_results: list[SourceProviderResult] = []

        async def save_card_progress(card_progress: Any) -> None:
            progress_sources = dedupe_sources(card_progress.sources)[:260]
            progress_state: WorkflowState = dict(state)
            if progress_sources:
                progress_evidence = evidence_from_sources(progress_sources)
                progress_provider_results = _attach_evidence_ids_to_provider_signals(
                    list(card_progress.provider_results),
                    progress_sources,
                    progress_evidence,
                )
                progress_state.update(
                    {
                        "raw_sources": [source.model_dump(mode="json") for source in progress_sources],
                        "evidence": [item.model_dump(mode="json") for item in progress_evidence],
                        "evidence_signals": [
                            signal.model_dump(mode="json")
                            for result in progress_provider_results
                            for signal in result.signals
                        ],
                        "source_provider_results": [
                            item.model_dump(mode="json") for item in progress_provider_results
                        ],
                    }
                )
            progress_state.update(
                {
                    "workflow_phase": "phase_4_collect",
                    "competitor_detail_cards": [
                        item.model_dump(mode="json") for item in card_progress.competitor_detail_cards
                    ],
                    "industry_card": (
                        card_progress.industry_card.model_dump(mode="json")
                        if card_progress.industry_card
                        else None
                    ),
                    "competitor_detail_cards_clean": [
                        item.model_dump(mode="json") for item in card_progress.competitor_detail_cards_clean
                    ],
                    "industry_card_clean": (
                        card_progress.industry_card_clean.model_dump(mode="json")
                        if card_progress.industry_card_clean
                        else None
                    ),
                    "report_references": [
                        item.model_dump(mode="json") for item in card_progress.report_references
                    ],
                    "object_progress": card_progress.object_progress,
                    "detail_seen_url_keys": card_progress.detail_seen_url_keys,
                    "industry_seen_url_keys": card_progress.industry_seen_url_keys,
                }
            )
            if card_progress.risk_notices:
                progress_state["risk_notice"] = (
                    f"{progress_state.get('risk_notice') or ''}\n"
                    + "\n".join(card_progress.risk_notices)
                ).strip()
            self._save_checkpoint(task_id, progress_state, "source_collector_agent")

        card_result = await collect_detail_cards(
            model_available=bool(self.model and request.llm_failure_policy != "continue_without_llm"),
            structured_call=lambda schema, messages: self._llm_structured(
                task_id,
                "source_collector_agent",
                schema,
                messages,
            ),
            search_collect=collect_evidence_with_search_runner,
            benchmark=benchmark,
            selected_set=selected_set,
            request=request,
            settings=self.settings,
            task_id=task_id,
            existing_sources=sources,
            detail_seen_url_keys=set(str(item) for item in state.get("detail_seen_url_keys", [])),
            industry_seen_url_keys=set(str(item) for item in state.get("industry_seen_url_keys", [])),
            should_cancel=lambda: self._check_cancelled(task_id),
            progress_callback=save_card_progress,
        )
        sources = dedupe_sources(card_result.sources)[:260]
        if not sources:
            description = request.raw_description or request.notes or request.target_product
            sources = [
                RawSource(
                    url="manual://task-input",
                    title="Task input fallback",
                    content=(
                        f"User asked to analyze {benchmark.target_product}. "
                        f"Raw description: {description or 'not provided'}. "
                        "No search provider or URL fetch returned usable evidence."
                    ),
                    source_type="manual",
                    metadata={"provider": "fallback"},
                )
            ]
        provider_results.extend(card_result.provider_results)
        source_evidence = evidence_from_sources(sources)
        provider_results = _attach_evidence_ids_to_provider_signals(provider_results, sources, source_evidence)
        signals = [signal for result in provider_results for signal in result.signals]
        evidence = source_evidence
        competitor_detail_cards = card_result.competitor_detail_cards
        industry_card = card_result.industry_card or IndustryCard()
        competitor_detail_cards_clean = card_result.competitor_detail_cards_clean
        industry_card_clean = card_result.industry_card_clean
        report_references = card_result.report_references
        object_progress = card_result.object_progress
        if card_result.risk_notices:
            state["risk_notice"] = (
                f"{state.get('risk_notice') or ''}\n" + "\n".join(card_result.risk_notices)
            ).strip()
        await self._emit(
            task_id,
            "source_collector_agent",
            "completed",
            output_summary="本阶段采集完成。",
            started_at=started,
            token_payload={
                "sources": sources,
                "signals": signals,
                "competitor_detail_cards_clean": competitor_detail_cards_clean,
                "industry_card_clean": industry_card_clean,
                "object_progress": object_progress,
            },
            details={
                "kind": "source_collection",
                "providers": _detail_items(provider_results, _provider_result_detail),
                "sources": _detail_items(sources, _source_detail),
                "evidence": _detail_items(evidence, _evidence_detail),
                "signals": _detail_items(signals, _signal_detail),
                "competitor_detail_cards": _detail_items(competitor_detail_cards, lambda item: item.model_dump(mode="json")),
                "industry_card": industry_card.model_dump(mode="json"),
                "competitor_detail_cards_clean": _detail_items(competitor_detail_cards_clean, lambda item: item.model_dump(mode="json")),
                "industry_card_clean": industry_card_clean.model_dump(mode="json") if industry_card_clean else None,
                "report_references": _detail_items(report_references, lambda item: item.model_dump(mode="json")),
                "object_progress": object_progress,
            },
        )
        return {
            "raw_sources": [source.model_dump(mode="json") for source in sources],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "extracted_evidence": [],
            "field_extraction_results": [],
            "evidence_signals": [item.model_dump(mode="json") for item in signals],
            "source_provider_results": [item.model_dump(mode="json") for item in provider_results],
            "search_plan": None,
            "competitor_detail_cards": [item.model_dump(mode="json") for item in competitor_detail_cards],
            "industry_card": industry_card.model_dump(mode="json"),
            "competitor_detail_cards_clean": [item.model_dump(mode="json") for item in competitor_detail_cards_clean],
            "industry_card_clean": industry_card_clean.model_dump(mode="json") if industry_card_clean else None,
            "object_progress": object_progress,
            "report_references": [item.model_dump(mode="json") for item in report_references],
            "collection_gap_request": state.get("collection_gap_request", []),
            "risk_notice": state.get("risk_notice"),
            "detail_seen_url_keys": card_result.detail_seen_url_keys,
            "industry_seen_url_keys": card_result.industry_seen_url_keys,
        }

    async def tool_executor_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        self._check_cancelled(task_id)
        request = CreateTaskRequest.model_validate(state["request"])
        benchmark = AnalysisBenchmark.model_validate(state["benchmark"])
        selected_set = SelectedCompetitorSet.model_validate(state["selected_competitor_set"])
        tool_plan = ToolPlan.model_validate(state["tool_plan"])
        competitor_detail_cards_clean = [
            CompetitorDetailCardClean.model_validate(item)
            for item in state.get("competitor_detail_cards_clean", [])
        ]
        industry_card_clean = (
            IndustryCardClean.model_validate(state["industry_card_clean"])
            if state.get("industry_card_clean")
            else None
        )
        report_references = [
            ReportReference.model_validate(item) for item in state.get("report_references", [])
        ]
        started = time.perf_counter()
        await self._emit(task_id, "tool_executor_agent", "running", "逐个执行已选分析工具，由大模型输出带证据的结构化结果。")

        try:
            tool_results = await execute_tools(
                None if request.llm_failure_policy == "continue_without_llm" else self.model,
                benchmark,
                selected_set,
                tool_plan,
                competitor_detail_cards_clean=competitor_detail_cards_clean,
                industry_card_clean=industry_card_clean,
                report_references=report_references,
                concurrency=self.settings.tool_execution_concurrency,
            )
        except Exception as exc:
            raise self._llm_decision(task_id, "tool_executor_agent", friendly_llm_error(exc)) from exc
        sandboxes = tool_results_to_sandboxes(tool_results)
        await self._emit(
            task_id,
            "tool_executor_agent",
            "completed",
            output_summary=f"完成 {len(tool_results)} 个工具执行结果，每条工具结论要求引用资料卡字段。",
            started_at=started,
            token_payload=tool_results,
            details={
                "kind": "tool_execution",
                "results": _detail_items(tool_results, _tool_result_detail),
            },
        )
        return {
            "tool_results": [item.model_dump(mode="json") for item in tool_results],
            "tool_sandboxes": [item.model_dump(mode="json") for item in sandboxes],
        }

    async def targeted_collector_agent(self, state: WorkflowState) -> WorkflowState:
        collector_updates = await self.source_collector_agent(state)
        return collector_updates
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        collection_plan = CollectionPlan.model_validate(state["collection_plan"])
        selected_set = SelectedCompetitorSet.model_validate(state["selected_competitor_set"])
        started = time.perf_counter()
        await self._emit(task_id, "targeted_collector_agent", "running", "按工具清单定向采集资料。")

        sources = await collect_sources_for_plan(request, self.settings, collection_plan, task_id=task_id)
        evidence = evidence_from_sources(sources)
        tool_plan = ToolPlan.model_validate(state["tool_plan"])
        sandboxes = build_tool_sandboxes(tool_plan, selected_set, [item.id for item in evidence])
        await self._emit(
            task_id,
            "targeted_collector_agent",
            "completed",
            output_summary=f"采集 {len(sources)} 条来源，生成 {len(sandboxes)} 个工具沙盒。",
            started_at=started,
            token_payload=sources,
        )
        return {
            "raw_sources": [source.model_dump(mode="json") for source in sources],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "tool_sandboxes": [item.model_dump(mode="json") for item in sandboxes],
        }


    async def _legacy_analyst_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        benchmark, selected_set, tool_plan, collection_plan = self._ensure_planning_context(state, request)
        structured = (
            CompetitorReport.model_validate(state["structured_report"])
            if state.get("structured_report")
            else build_fallback_report(
                task_id,
                request,
                [],
                risk_notice=state.get("risk_notice"),
                benchmark=benchmark,
                selected_competitor_set=selected_set,
                tool_plan=tool_plan,
                collection_plan=collection_plan,
                search_plan=SearchPlan.model_validate(state["search_plan"]) if state.get("search_plan") else None,
                source_provider_results=[],
                tool_results=state.get("tool_results", []),
                tool_sandboxes=state.get("tool_sandboxes", []),
            )
        )
        structured.evidence = []
        structured.report_references = []
        structured.source_provider_results = []
        structured.field_extraction_results = []
        structured.analysis_benchmark = benchmark
        structured.selected_competitor_set = selected_set
        structured.tool_plan = tool_plan
        structured.collection_plan = collection_plan
        if state.get("search_plan"):
            structured.search_plan = SearchPlan.model_validate(state["search_plan"])
        structured.tool_results = [
            ToolExecutionResult.model_validate(item) for item in state.get("tool_results", [])
        ]
        structured.tool_sandboxes = [
            ToolSandbox.model_validate(item) for item in state.get("tool_sandboxes", [])
        ]
        if state.get("analysis_intent"):
            structured.analysis_intent = AnalysisIntentJSON.model_validate(state["analysis_intent"])
        structured.competitor_brief_profiles = [
            CompetitorBriefProfile.model_validate(item) for item in state.get("competitor_brief_profiles", [])
        ]
        structured.competitor_detail_cards = [
            CompetitorDetailCard.model_validate(item) for item in state.get("competitor_detail_cards", [])
        ]
        structured.industry_card = IndustryCard.model_validate(state["industry_card"]) if state.get("industry_card") else None
        structured.competitor_detail_cards_clean = [
            CompetitorDetailCardClean.model_validate(item)
            for item in state.get("competitor_detail_cards_clean", [])
        ]
        structured.industry_card_clean = (
            IndustryCardClean.model_validate(state["industry_card_clean"])
            if state.get("industry_card_clean")
            else None
        )
        if state.get("object_progress") and not structured.object_progress:
            structured.object_progress = list(state.get("object_progress", []))
        started = time.perf_counter()
        await self._emit(task_id, "analyst_agent", "running", "整理工具结果并透传给章节级报告撰写。")

        try:
            report = structured

            report.analysis_benchmark = structured.analysis_benchmark
            report.selected_competitor_set = structured.selected_competitor_set
            report.tool_plan = structured.tool_plan
            report.collection_plan = structured.collection_plan
            report.search_plan = structured.search_plan
            report.source_provider_results = []
            report.tool_results = structured.tool_results
            report.tool_sandboxes = structured.tool_sandboxes
            report.field_extraction_results = []
            report.analysis_intent = structured.analysis_intent
            report.competitor_brief_profiles = structured.competitor_brief_profiles
            report.competitor_detail_cards = structured.competitor_detail_cards
            report.industry_card = structured.industry_card
            report.competitor_detail_cards_clean = structured.competitor_detail_cards_clean
            report.industry_card_clean = structured.industry_card_clean
            report.report_references = []
            report.object_progress = structured.object_progress
            report.evidence = []
            _canonicalize_report_competitor_aliases(report)
            report.competitors = apply_selection_to_profiles(report.competitors, report.selected_competitor_set)

            await self._emit(
                task_id,
                "analyst_agent",
                "completed",
                output_summary=f"已整理 {len(report.tool_results)} 个工具结果，等待章节级写作。",
                started_at=started,
                token_payload=report,
                details={
                    "kind": "analysis_passthrough",
                    "report": _report_detail(report),
                    "comparison_count": len(report.comparison_matrix),
                },
            )
            return {"analysis_report": report.model_dump(mode="json")}
        except LLMDecisionNeeded:
            raise
        except Exception as exc:
            logger.exception("analyst_agent fallback for task %s", task_id)
            structured.risk_notice = f"分析模型调用失败，保留结构化降级报告：{friendly_llm_error(exc)}"
            await self._emit(
                task_id,
                "analyst_agent",
                "completed",
                output_summary="分析模型失败，保留结构化报告继续流转。",
                started_at=started,
                error=str(exc),
                details={
                    "kind": "analysis",
                    "report": _report_detail(structured),
                    "fallback": True,
                },
            )
            return {"analysis_report": structured.model_dump(mode="json")}

    async def _legacy_writer_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        request = CreateTaskRequest.model_validate(state["request"])
        analysis = CompetitorReport.model_validate(state["analysis_report"])
        started = time.perf_counter()
        await self._emit(task_id, "writer_agent", "running", "按章节撰写最终竞品分析报告。")

        try:
            report = analysis
            _canonicalize_report_competitor_aliases(report)
            report.competitors = apply_selection_to_profiles(report.competitors, report.selected_competitor_set)
            if state.get("evidence"):
                report.evidence = [EvidenceItem.model_validate(item) for item in state.get("evidence", [])]
            if state.get("source_provider_results"):
                report.source_provider_results = [
                    SourceProviderResult.model_validate(item)
                    for item in state.get("source_provider_results", [])
                ]
            if report.evidence:
                _retarget_report_evidence_ids(report, report.evidence)
            report.report_references = report.report_references or _build_report_references(report)
            report.deep_dive_markdown, report.report_chapters = await self._build_chaptered_report_markdown(
                task_id,
                request,
                report,
            )
            report.deep_dive_markdown = _ensure_markdown_reference_links(report.deep_dive_markdown, report)
            _derive_report_summary_fields(report, report.deep_dive_markdown)

            await self._emit(
                task_id,
                "writer_agent",
                "completed",
                output_summary="章节级最终报告已生成。",
                started_at=started,
                token_payload=report,
                details={
                    "kind": "writer",
                    "target_product": report.target_product,
                    "deep_dive_markdown_length": len(report.deep_dive_markdown or ""),
                    "report": _report_detail(report),
                },
            )
            return {"report": report.model_dump(mode="json")}
        except LLMDecisionNeeded:
            raise
        except Exception as exc:
            logger.exception("writer_agent fallback for task %s", task_id)
            analysis.risk_notice = f"章节级报告撰写失败，使用规则模板生成最终报告：{friendly_llm_error(exc)}"
            analysis.report_references = analysis.report_references or _build_report_references(analysis)
            if not analysis.deep_dive_markdown:
                analysis.deep_dive_markdown = build_deep_dive_markdown(analysis, request)
            analysis.deep_dive_markdown = _ensure_markdown_reference_links(analysis.deep_dive_markdown, analysis)
            analysis.report_chapters = [
                ReportChapterDraft(
                    title="最终报告",
                    markdown=analysis.deep_dive_markdown,
                    reference_ids=[ref.id for ref in analysis.report_references],
                )
            ]
            await self._emit(
                task_id,
                "writer_agent",
                "completed",
                output_summary="报告撰写失败，使用分析报告继续质检。",
                started_at=started,
                error=str(exc),
                details={
                    "kind": "writer",
                    "target_product": analysis.target_product,
                    "deep_dive_markdown_length": len(analysis.deep_dive_markdown or ""),
                    "report": _report_detail(analysis),
                    "fallback": True,
                },
            )
            return {"report": analysis.model_dump(mode="json")}

    async def _build_deep_dive_markdown(
        self,
        task_id: str,
        request: CreateTaskRequest,
        report: CompetitorReport,
    ) -> str:
        markdown, _ = await self._build_chaptered_report_markdown(task_id, request, report)
        return markdown

    async def _build_chaptered_report_markdown(
        self,
        task_id: str,
        request: CreateTaskRequest,
        report: CompetitorReport,
    ) -> tuple[str, list[ReportChapterDraft]]:
        use_llm = bool(self.model and request.llm_failure_policy != "continue_without_llm")
        section_pairs: list[tuple[str, str]] = []
        chapters: list[ReportChapterDraft] = []

        async def render(title: str, prompt_template: str, context: dict[str, Any], fallback: str) -> str:
            if not use_llm:
                return fallback.strip()
            try:
                prompt = prompt_template.format(context_json=_json_for_llm(context, max_chars=18000, max_text=800, max_list=25))
                response = await self._llm_text(task_id, "writer_agent", [("system", writer_system_prompt), ("user", prompt)])
                content = getattr(response, "content", "") or ""
                if isinstance(content, list):
                    content = "\n".join(str(item) for item in content)
                text = str(content).strip()
                if not _valid_chapter_markdown(text, title):
                    return fallback.strip()
                return text
            except LLMDecisionNeeded:
                raise
            except Exception as exc:
                report.risk_notice = (
                    f"{report.risk_notice or ''}\n章节“{title}”生成失败，已使用规则模板：{friendly_llm_error(exc)}"
                ).strip()
                return fallback.strip()

        target = report.analysis_benchmark.target_product if report.analysis_benchmark else report.target_product
        title_context = _report_context_payload(request=request, report=report)
        title_md = await render(
            "报告名称与报告日期",
            REPORT_TITLE_PROMPT,
            title_context,
            f"# {target}竞品分析报告\n\n- 报告日期：{utc_now().date().isoformat()}\n- 分析目的：{request.analysis_purpose}\n- 产品阶段：{request.lifecycle_stage or 'concept'}",
        )
        section_pairs.append(("报告名称与报告日期", title_md))

        background_md = await render(
            "分析背景",
            REPORT_BACKGROUND_PROMPT,
            _report_context_payload(request=request, report=report),
            _fallback_background_section(report, request),
        )
        section_pairs.append(("分析背景", background_md))

        competitor_md = await render(
            "竞品选择",
            REPORT_COMPETITOR_SELECTION_PROMPT,
            _report_context_payload(
                request=request,
                report=report,
                extra={"竞品选择材料": _candidate_selection_payload(report)},
            ),
            _fallback_competitor_selection_section(report),
        )
        section_pairs.append(("竞品选择", competitor_md))

        include_own_canvas = _has_own_product_info(request, report.analysis_benchmark)
        if include_own_canvas:
            section_pairs.append(
                (
                    "展示自身竞品画布",
                    "## 四、展示自身竞品画布\n\n本次已识别到用户输入中的自身产品信息，但当前版本仅预留自身竞品画布接口，暂不调用大模型生成该章节。",
                )
            )

        tool_section_parts: list[str] = []
        for tool_name, grouped_results in _group_tool_results_by_tool(report.tool_results):
            context = _compact_for_llm(
                {
                    "工具名称": tool_name,
                    "分析基准": report.analysis_benchmark.model_dump(mode="json") if report.analysis_benchmark else None,
                    "最终竞品": [profile.model_dump(mode="json") for profile in report.competitors],
                    "工具结果": _sanitize_tool_results_for_analysis(grouped_results),
                },
                max_text=900,
                max_list=30,
            )
            fallback = _fallback_tool_section(tool_name, grouped_results)
            if use_llm:
                try:
                    prompt = REPORT_TOOL_SECTION_PROMPT.format(
                        context_json=_json_for_llm(context, max_chars=20000, max_text=900, max_list=30)
                    )
                    response = await self._llm_text(task_id, "writer_agent", [("system", writer_system_prompt), ("user", prompt)])
                    content = getattr(response, "content", "") or ""
                    if isinstance(content, list):
                        content = "\n".join(str(item) for item in content)
                    section_text = str(content).strip() or fallback
                    if not section_text.startswith("###"):
                        section_text = fallback
                except LLMDecisionNeeded:
                    raise
                except Exception as exc:
                    report.risk_notice = (
                        f"{report.risk_notice or ''}\n工具小节“{tool_name}”生成失败，已使用规则模板：{friendly_llm_error(exc)}"
                    ).strip()
                    section_text = fallback
            else:
                section_text = fallback
            tool_section_parts.append(section_text.strip())

        analysis_sections = "\n\n".join(tool_section_parts).strip()
        analysis_md = f"## {'五' if include_own_canvas else '四'}、分析维度\n\n{analysis_sections}" if analysis_sections else f"## {'五' if include_own_canvas else '四'}、分析维度\n\n本次没有可用的工具执行结果。"

        key_findings_md = await self._render_summary_from_analysis(
            task_id,
            request,
            report,
            REPORT_KEY_FINDINGS_PROMPT,
            analysis_sections,
            _fallback_key_findings_section(report),
            "关键发现",
            use_llm,
        )
        recommendations_md = await self._render_summary_from_analysis(
            task_id,
            request,
            report,
            REPORT_RECOMMENDATIONS_PROMPT,
            analysis_sections,
            _fallback_recommendations_section(report),
            "总结建议",
            use_llm,
        )
        key_findings_md, recommendations_md = await self._unify_findings_and_recommendations(
            task_id,
            report,
            key_findings_md,
            recommendations_md,
            use_llm,
        )
        section_pairs.insert(4 if include_own_canvas else 3, ("分析维度", analysis_md))
        section_pairs.append(("关键发现", key_findings_md))
        section_pairs.append(("总结建议", recommendations_md))

        appendix_title = "附录"
        section_pairs.append((appendix_title, _build_appendix_markdown(report, request, "附录")))

        markdown = _renumber_report_sections(section_pairs)
        for title, section_markdown in section_pairs:
            chapters.append(
                ReportChapterDraft(
                    title=title,
                    markdown=section_markdown.strip(),
                    reference_ids=[ref.id for ref in report.report_references],
                )
            )
        return markdown, chapters

    async def _render_summary_from_analysis(
        self,
        task_id: str,
        request: CreateTaskRequest,
        report: CompetitorReport,
        prompt_template: str,
        analysis_sections_markdown: str,
        fallback: str,
        title: str,
        use_llm: bool,
    ) -> str:
        if not use_llm:
            return fallback
        try:
            prompt = prompt_template.format(
                analysis_sections_markdown=analysis_sections_markdown[:24000],
            )
            response = await self._llm_text(task_id, "writer_agent", [("system", writer_system_prompt), ("user", prompt)])
            content = getattr(response, "content", "") or ""
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            return str(content).strip() or fallback
        except LLMDecisionNeeded:
            raise
        except Exception as exc:
            report.risk_notice = (
                f"{report.risk_notice or ''}\n章节“{title}”生成失败，已使用规则模板：{friendly_llm_error(exc)}"
            ).strip()
            return fallback

    async def _unify_findings_and_recommendations(
        self,
        task_id: str,
        report: CompetitorReport,
        key_findings_markdown: str,
        recommendations_markdown: str,
        use_llm: bool,
    ) -> tuple[str, str]:
        fallback = f"{key_findings_markdown.strip()}\n\n{recommendations_markdown.strip()}"
        if not use_llm:
            return key_findings_markdown, recommendations_markdown
        try:
            prompt = REPORT_FINAL_UNIFY_PROMPT.format(
                key_findings_markdown=key_findings_markdown[:12000],
                recommendations_markdown=recommendations_markdown[:12000],
            )
            response = await self._llm_text(task_id, "writer_agent", [("system", writer_system_prompt), ("user", prompt)])
            content = getattr(response, "content", "") or ""
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            unified = str(content).strip() or fallback
            return _split_findings_and_recommendations(unified, key_findings_markdown, recommendations_markdown)
        except LLMDecisionNeeded:
            raise
        except Exception as exc:
            report.risk_notice = (
                f"{report.risk_notice or ''}\n关键发现与总结建议统一润色失败，已使用未润色版本：{friendly_llm_error(exc)}"
            ).strip()
            return key_findings_markdown, recommendations_markdown

    async def _legacy_qa_agent(self, state: WorkflowState) -> WorkflowState:
        task_id = state["task_id"]
        report = CompetitorReport.model_validate(state["report"])
        revision_count = int(state.get("revision_count", 0))
        max_revision_loops = int(state.get("max_revision_loops", self.settings.max_revision_loops))
        started = time.perf_counter()
        await self._emit(task_id, "qa_agent", "running", "检查槽位、定标逻辑、证据引用和报告结构。")

        qa_result = validate_report(report, revision_count, max_revision_loops)
        qa_history = [QAResult.model_validate(item) for item in state.get("qa_history", [])]
        qa_history.append(qa_result)
        report.qa_history = qa_history

        updates: WorkflowState = {
            "qa_history": [item.model_dump(mode="json") for item in qa_history],
            "report": report.model_dump(mode="json"),
        }
        if qa_result.revision_request:
            updates["revision_count"] = revision_count + 1
            await self._emit(
                task_id,
                "qa_agent",
                "revision",
                message=qa_result.revision_request.reason,
                output_summary=f"质检未通过，打回 {qa_result.revision_request.target_agent}。",
                started_at=started,
                token_payload=qa_result,
                details={
                    "kind": "qa",
                    "qa_result": _qa_detail(qa_result),
                    "revision_count": revision_count + 1,
                    "max_revision_loops": max_revision_loops,
                },
            )
        else:
            if not qa_result.passed:
                report.risk_notice = (
                    "报告未完全通过质检，但已达到最大返工次数。请优先补充来源后重新分析。"
                )
                updates["report"] = report.model_dump(mode="json")
            await self._emit(
                task_id,
                "qa_agent",
                "completed",
                output_summary=f"质检{'通过' if qa_result.passed else '带风险结束'}，得分 {qa_result.score}。",
                started_at=started,
                token_payload=qa_result,
                details={
                    "kind": "qa",
                    "qa_result": _qa_detail(qa_result),
                    "revision_count": revision_count,
                    "max_revision_loops": max_revision_loops,
                },
            )
        return updates

    def route_after_qa(self, state: WorkflowState) -> str:
        checkpoint = state.get("qa_checkpoint")
        if checkpoint:
            return "qa_agent"
        phase = state.get("workflow_phase")
        if phase == "phase_2_search":
            return "search_agent"
        if phase == "phase_4_collect":
            return "search_agent"
        if phase == "phase_2_analyze":
            return "analyst_agent"
        if phase == "survey_design":
            return "survey_design_agent"
        if phase == "phase_3_strategy":
            return "plan_agent"
        if phase == "phase_4_synthesis":
            return "analyst_agent"
        if phase == "writer_final":
            return "writer_agent"
        qa_history = [QAResult.model_validate(item) for item in state.get("qa_history", [])]
        if not qa_history:
            return "end"
        latest = qa_history[-1]
        if latest.revision_request:
            return latest.revision_request.target_agent
        return "end"

    def _build_graph(self):
        workflow = StateGraph(WorkflowState)
        workflow.add_node("plan_agent", self.plan_agent)
        workflow.add_node("search_agent", self.search_agent)
        workflow.add_node("analyst_agent", self.analyst_agent)
        workflow.add_node("survey_design_agent", self.survey_design_agent)
        workflow.add_node("qa_agent", self.qa_agent)
        workflow.add_node("writer_agent", self.writer_agent)

        workflow.set_entry_point("plan_agent")
        workflow.add_edge("plan_agent", "search_agent")
        workflow.add_edge("search_agent", "analyst_agent")
        workflow.add_edge("analyst_agent", "qa_agent")
        workflow.add_conditional_edges(
            "qa_agent",
            self.route_after_qa,
            {
                "plan_agent": "plan_agent",
                "search_agent": "search_agent",
                "analyst_agent": "analyst_agent",
                "survey_design_agent": "survey_design_agent",
                "writer_agent": "writer_agent",
                "end": END,
            },
        )
        workflow.add_edge("writer_agent", END)
        return workflow.compile()

    async def run(self, task_id: str, request: CreateTaskRequest, max_revision_loops: int) -> CompetitorReport:
        loaded = self._load_checkpoint(task_id)
        if loaded:
            result, next_node = loaded
            result = self._prepare_resumed_state(result, task_id, request, max_revision_loops)
        else:
            result = self._initial_state(task_id, request, max_revision_loops)
            next_node = "plan_agent"

        recursion_limit = 32
        for _ in range(recursion_limit):
            self._check_cancelled(task_id)
            if next_node == "end":
                break
            self._save_checkpoint(task_id, result, next_node)
            if next_node == "plan_agent":
                updates = await self.plan_agent(result)
                next_node = "search_agent"
            elif next_node == "search_agent":
                updates = await self.search_agent(result)
                next_node = "analyst_agent"
            elif next_node == "analyst_agent":
                updates = await self.analyst_agent(result)
                next_node = "qa_agent"
            elif next_node == "qa_agent":
                updates = await self.qa_agent(result)
                result.update(updates)
                next_node = self.route_after_qa(result)
                self._save_checkpoint(task_id, result, next_node)
                continue
            elif next_node == "survey_design_agent":
                updates = await self.survey_design_agent(result)
                next_node = "plan_agent"
            elif next_node == "writer_agent":
                updates = await self.writer_agent(result)
                next_node = "end"
            else:
                raise RuntimeError(f"Unknown workflow checkpoint node: {next_node}")
            result.update(updates)
            self._save_checkpoint(task_id, result, next_node)
        else:
            raise RuntimeError("Workflow recursion limit reached.")

        if "report" not in result:
            raise RuntimeError("Workflow finished without a report.")
        report = CompetitorReport.model_validate(result["report"])
        if not report.qa_history:
            report.qa_history = _qa_history_from_state_tree(result.get("state_tree", {}))
        knowledge_context = result.get("knowledge_context")
        if isinstance(knowledge_context, dict):
            report.knowledge_reuse_summary = KnowledgeReuseSummary.model_validate(knowledge_context)
            report.knowledge_card_ids = [
                str(item["id"])
                for item in [
                    *result.get("reused_competitor_cards", []),
                    *result.get("reused_industry_cards", []),
                ]
                if isinstance(item, dict) and item.get("id")
            ]
            report.knowledge_source_ids = [str(item) for item in result.get("knowledge_sources", [])]
        survey_design_payload = result.get("survey_design")
        if isinstance(survey_design_payload, dict):
            report.survey_design = SurveyDesign.model_validate(survey_design_payload)
        self.storage.save_report(report)
        self._save_final_report_snapshot(task_id, report)
        await asyncio.sleep(0)
        return report


def _qa_history_from_state_tree(state_tree: dict[str, Any]) -> list[QAResult]:
    gates = state_tree.get("quality_gates") if isinstance(state_tree, dict) else None
    if not isinstance(gates, dict):
        return []
    history: list[QAResult] = []
    for name, gate in gates.items():
        if not isinstance(gate, dict):
            continue
        issues = [str(item) for item in gate.get("issues_found") or []]
        passed = gate.get("inspection_status") == "pass"
        history.append(
            QAResult(
                passed=passed,
                score=1.0 if passed and not issues else max(0.2, 0.88 - 0.12 * len(issues)),
                issues=[
                    QAIssue(
                        severity="medium" if passed else "high",
                        field=str(name),
                        message=issue,
                    )
                    for issue in issues
                ],
            )
        )
    return history


def _survey_design_markdown(design: SurveyDesign) -> str:
    lines = [
        f"## {design.title}",
        "",
        f"- 目标受访者：{design.target_respondents}",
        f"- 建议样本量：{design.suggested_sample_size}",
        f"- 投放渠道：{'、'.join(design.distribution_channels) or '待确认'}",
        "",
        "### 研究目标",
        *[f"- {item}" for item in design.research_goals],
        "",
        "### 问卷题目",
    ]
    for index, question in enumerate(design.questions, start=1):
        options = f" 选项：{' / '.join(question.options)}" if question.options else ""
        lines.append(f"{index}. [{question.type}] {question.title}（{question.dimension}）{options}")
    return "\n".join(lines).strip()
