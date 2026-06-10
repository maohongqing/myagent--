from __future__ import annotations

from myagent.backend.app.agents.deep_report import build_deep_dive_markdown
from myagent.backend.app.agents.strategy import apply_selection_to_profiles
from myagent.backend.app.models import (
    AnalysisBenchmark,
    AnalysisFinding,
    CollectionPlan,
    ComparisonCell,
    CompetitorProfile,
    CompetitorReport,
    CreateTaskRequest,
    EvidenceItem,
    RawSource,
    SearchPlan,
    SourceRef,
    SWOT,
    SelectedCompetitorSet,
    SourceProviderResult,
    ToolPlan,
    ToolExecutionResult,
    ToolSandbox,
)


def evidence_from_sources(sources: list[RawSource]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for source in sources:
        excerpt = source.content[:700] if source.content else "No source content available."
        credibility = 0.75 if source.source_type in {"search", "url"} and not source.metadata.get("error") else 0.35
        if source.source_type == "screenshot":
            credibility = 0.8
        related = ["positioning", "features", "pricing"]
        if source.metadata.get("tool"):
            related.append(str(source.metadata["tool"]))
        image_path = (
            source.metadata.get("evidence_screenshot_path")
            or source.metadata.get("quote_shot_path")
            or source.metadata.get("screenshot_path")
            or source.metadata.get("image_path")
        )
        evidence.append(
            EvidenceItem(
                url=source.url,
                title=source.title,
                source_type=source.source_type,
                excerpt=excerpt,
                credibility=credibility,
                related_fields=related,
                image_path=image_path,
                image_alt=source.metadata.get("image_alt"),
                evidence_kind="source_chunk",
                competitor=source.metadata.get("competitor"),
                tool=source.metadata.get("tool"),
                source_refs=[
                    SourceRef(
                        source_id=source.id,
                        chunk_id=source.metadata.get("chunk_id"),
                        url=source.url,
                        title=source.title,
                        quote=excerpt,
                        screenshot_path=source.metadata.get("screenshot_path"),
                        evidence_screenshot_path=source.metadata.get("evidence_screenshot_path") or source.metadata.get("quote_shot_path"),
                        chunk_shot_paths=[],
                    )
                ],
            )
        )
    return evidence


def build_fallback_report(
    task_id: str,
    request: CreateTaskRequest,
    sources: list[RawSource],
    qa_history=None,
    risk_notice: str | None = None,
    benchmark: AnalysisBenchmark | None = None,
    selected_competitor_set: SelectedCompetitorSet | None = None,
    tool_plan: ToolPlan | None = None,
    collection_plan: CollectionPlan | None = None,
    search_plan: SearchPlan | None = None,
    source_provider_results: list[SourceProviderResult | dict] | None = None,
    tool_results: list[ToolExecutionResult | dict] | None = None,
    tool_sandboxes: list[ToolSandbox | dict] | None = None,
) -> CompetitorReport:
    target = benchmark.target_product if benchmark else (request.target_product or "未命名产品")
    if not sources:
        sources = [
            RawSource(
                url="manual://task-input",
                title="Task input fallback",
                content=(
                    f"用户请求分析目标产品 {target}，竞品包括 {', '.join(request.competitors) or '未指定'}。"
                    f"原始描述：{request.raw_description or request.notes or '无'}。"
                    "当前未配置可用搜索或未提供可抓取 URL，因此只能生成低置信度框架分析。"
                ),
                source_type="manual",
            )
        ]
    evidence = evidence_from_sources(sources)
    first_evidence = [evidence[0].id] if evidence else []

    competitor_names = [item.name for item in selected_competitor_set.selected] if selected_competitor_set else request.competitors
    if not competitor_names:
        competitor_names = [target]

    profiles = []
    selected_by_name = {
        item.name: item for item in selected_competitor_set.selected
    } if selected_competitor_set else {}
    for name in competitor_names:
        candidate = selected_by_name.get(name)
        profiles.append(
            CompetitorProfile(
                name=name,
                website=candidate.website if candidate else None,
                positioning=(
                    candidate.description[:240]
                    if candidate and candidate.description
                    else "需要结合公开资料进一步确认的竞品定位。"
                ),
                target_users=["公开资料未充分确认"],
                core_features=["核心功能待从来源中抽取确认"],
                pricing="Unknown",
                business_model="Unknown",
                channels=["公开网页", "搜索结果"],
                growth_signals=["需要持续监测新闻、官网更新和价格页变化"],
                evidence_ids=first_evidence,
                competitor_type=candidate.competitor_type if candidate else None,
                score=candidate.score if candidate else None,
                selection_reason=candidate.selection_reason if candidate else "",
                rank_factors=candidate.rank_factors if candidate else None,
            )
        )

    findings = [
        AnalysisFinding(
            category="strategy",
            title="竞品选择已从用户指定转向目标导向定标",
            conclusion=(
                "本次流程先根据产品阶段、分析目的和分析目标确定分析基准，再通过候选池评分筛选竞品。"
                "若外部来源较少，竞品画像仍需在后续数据补充后提高置信度。"
            ),
            evidence_ids=first_evidence,
            confidence=0.58 if first_evidence else 0.25,
            impact="high",
        ),
        AnalysisFinding(
            category="opportunity",
            title="工具链决定后续采集重点",
            conclusion=(
                f"当前选择的工具链为 {', '.join(tool_plan.selected_tools) if tool_plan else '待确认'}，"
                "报告中的深度拆解会优先围绕这些工具需要的数据展开。"
            ),
            evidence_ids=first_evidence,
            confidence=0.55,
            impact="medium",
        ),
        AnalysisFinding(
            category="risk",
            title="数据覆盖度仍是结论质量的主要约束",
            conclusion=(
                "现阶段优先复用搜索结果和用户资料，应用商店、招聘、融资和行业数据库接口已预留，"
                "但未接入前对增长和颠覆风险的判断应标注为低到中等置信度。"
            ),
            evidence_ids=first_evidence,
            confidence=0.5,
            impact="medium",
        ),
    ]

    matrix = [
        ComparisonCell(
            competitor=profile.name,
            dimension="选择逻辑",
            value=profile.selection_reason or profile.positioning,
            evidence_ids=profile.evidence_ids,
        )
        for profile in profiles
    ]

    report = CompetitorReport(
        task_id=task_id,
        target_product=target,
        competitors=apply_selection_to_profiles(profiles, selected_competitor_set),
        executive_summary=(
            f"本报告围绕 {target} 展开，采用四阶段流程：先澄清分析基准，再召回候选竞品并动态评分，"
            "随后按分析目的选择工具链和采集清单，最后合成五板块商业报告。"
        ),
        findings=findings,
        comparison_matrix=matrix,
        swot=SWOT(
            strengths=["流程能够保留竞品选择逻辑、评分因素和证据链"],
            weaknesses=["当前搜索来源可能不足以支撑所有市场规模、下载量和融资判断"],
            opportunities=["可继续接入应用商店、行业数据库、招聘和融资数据源"],
            threats=["若澄清信息过少，分析目的误判会影响竞品配比和工具选择"],
        ),
        recommendations=[
            "优先补充每个入选竞品的官网、定价页、更新日志和用户评论来源。",
            "对分数最高的竞品做一次人工复核，确认其类型标签和入选理由是否符合业务直觉。",
            "下一轮接入应用商店和融资/招聘数据后，重新计算增长因子与风险因子。",
        ],
        evidence=evidence,
        qa_history=qa_history or [],
        risk_notice=risk_notice,
        analysis_benchmark=benchmark,
        selected_competitor_set=selected_competitor_set,
        tool_plan=tool_plan,
        collection_plan=collection_plan,
        search_plan=search_plan,
        source_provider_results=[
            item if isinstance(item, SourceProviderResult) else SourceProviderResult.model_validate(item)
            for item in (source_provider_results or [])
        ],
        tool_results=[
            item if isinstance(item, ToolExecutionResult) else ToolExecutionResult.model_validate(item)
            for item in (tool_results or [])
        ],
        tool_sandboxes=[
            item if isinstance(item, ToolSandbox) else ToolSandbox.model_validate(item)
            for item in (tool_sandboxes or [])
        ],
    )
    report.deep_dive_markdown = build_deep_dive_markdown(report, request)
    return report
