from __future__ import annotations

import re

from myagent.backend.app.agents.strategy import PURPOSE_LABELS, STAGE_LABELS, TYPE_LABELS
from myagent.backend.app.models import CompetitorReport, CreateTaskRequest


DEEP_DIVE_SECTIONS = [
    "一、报告名称与报告日期",
    "二、分析背景",
    "三、竞品选择",
    "四、分析维度",
    "五、关键发现",
    "六、总结建议",
    "七、附录",
]


def _join(items: list[str], fallback: str = "待核实") -> str:
    return "、".join(item for item in items if item) or fallback


def _source_refs(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "待补充来源"


def _score(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "待评分"


def build_deep_dive_markdown(
    report: CompetitorReport,
    request: CreateTaskRequest | None = None,
) -> str:
    benchmark = report.analysis_benchmark
    selected_set = report.selected_competitor_set
    tool_plan = report.tool_plan
    collection_plan = report.collection_plan
    target = benchmark.target_product if benchmark else (
        request.our_product if request and request.our_product else report.target_product
    )
    purpose = PURPOSE_LABELS.get(benchmark.analysis_purpose, benchmark.analysis_purpose) if benchmark else "待确认"
    stage = STAGE_LABELS.get(benchmark.lifecycle_stage, benchmark.lifecycle_stage) if benchmark else "待确认"
    goal = benchmark.analysis_goal if benchmark else (request.analysis_goal if request else "待确认")

    lines: list[str] = [
        f"# {target}竞品分析报告",
        "",
        "## 一、报告名称与报告日期",
        f"- **报告日期**：{report.generated_at.date().isoformat()}",
        f"- **目标产品**：{target}",
        f"- **产品阶段**：{stage}",
        f"- **分析目的**：{purpose}",
        "",
        "## 二、分析背景",
        "",
        "### 1. 分析目标",
        f"本次竞品分析围绕“{target}”展开，目标是为“{goal}”提供竞品证据、差异化判断和产品决策参考。",
        "",
        "### 2. 将从哪些维度进行分析",
        f"本次采用的分析工具包括：{_join(tool_plan.selected_tools if tool_plan else [])}。",
        "",
        "### 3. 选择维度的理由",
        tool_plan.rationale if tool_plan else "系统根据产品阶段、分析目的和竞品类型自动选择分析维度。",
        "",
    ]
    if report.risk_notice:
        lines.extend([f"> 风险提示：{report.risk_notice}", ""])

    lines.extend(["## 三、竞品选择", ""])
    if selected_set:
        scoring = selected_set.scoring_profile
        lines.extend(
            [
                f"- **候选池规模**：{len(selected_set.candidates)} 个候选，最终选择 {len(selected_set.selected)} 个分析对象。",
                (
                    f"- **评分权重**：热度 {scoring.alpha_heat:.2f}、增速 {scoring.beta_growth:.2f}、"
                    f"相似度 {scoring.gamma_similarity:.2f}、风险 {scoring.delta_risk:.2f}。"
                ),
                f"- **组合逻辑**：{', '.join(f'{TYPE_LABELS.get(k, k)} {v} 个' for k, v in selected_set.allocation.items())}",
                "",
            ]
        )
        for note in selected_set.fallback_notes:
            lines.append(f"- **降级说明**：{note}")
        if selected_set.fallback_notes:
            lines.append("")
    else:
        lines.extend(["- 尚未生成候选池评分结果。", ""])

    lines.extend(
        [
            "| 竞品 | 类型 | 分数 | 选择理由 | 关键因子 |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for profile in report.competitors:
        factors = profile.rank_factors
        factor_text = (
            f"热度 {factors.heat:.2f} / 增长 {factors.growth:.2f} / "
            f"相似度 {factors.similarity:.2f} / 风险 {factors.risk:.2f}"
            if factors
            else "待补充"
        )
        lines.append(
            "| "
            f"{profile.name} | "
            f"{TYPE_LABELS.get(profile.competitor_type or 'challenger', profile.competitor_type or '待分类')} | "
            f"{_score(profile.score)} | "
            f"{profile.selection_reason or '基于公开资料与用户输入进入分析名单'} | "
            f"{factor_text} |"
        )
    lines.append("")

    lines.extend(["## 四、分析维度", ""])
    if tool_plan:
        lines.extend(
            [
                f"- **工具链**：{_join(tool_plan.selected_tools)}",
                f"- **选择理由**：{tool_plan.rationale}",
                f"- **采集接口预留**：{_join(collection_plan.provider_interfaces if collection_plan else [])}",
                "",
            ]
        )
    for title, body in _analysis_dimensions(report):
        lines.extend([f"### {title}", body, ""])

    lines.extend(["## 五、关键发现", ""])
    findings = report.findings[:5] if report.findings else []
    if findings:
        for index, finding in enumerate(findings, start=1):
            lines.extend(
                [
                    f"### 发现 {index}：{finding.title}",
                    f"- **结论**：{finding.conclusion}",
                    f"- **影响**：{finding.impact}；**置信度**：{finding.confidence:.2f}",
                    f"- **证据**：{_source_refs(finding.evidence_ids)}",
                    "",
                ]
            )
    else:
        lines.extend(["- 当前工具结果不足以稳定提炼关键发现，需要补充更多竞品资料和工具分析。", ""])

    lines.extend(
        [
            "## 六、总结建议",
            "",
            "### 1. 分析结论",
            f"- 当前应围绕“{goal}”判断竞品行动优先级，避免把所有竞品按同一把尺子比较。",
            "",
            "### 2. 行动计划",
        ]
    )
    if report.recommendations:
        for item in report.recommendations:
            lines.append(f"- [ ] {item}")
    else:
        lines.extend(
            [
                "- [ ] 补充官网、定价页、更新日志、用户评价和近期新闻后重新评分。",
                "- [ ] 将竞品短板转化为 MVP 功能验证点，并为每个验证点绑定用户访谈或原型测试。",
            ]
        )

    lines.extend(["", "## 七、附录", "", "### 附件与来源"])
    for item in report.evidence:
        image = f"；截图：{item.image_path}" if item.image_path else ""
        lines.append(f"- [{item.id}] {item.title}: {item.url}{image}")
        lines.append(f"  - 摘录：{item.excerpt[:240]}")
    if report.tool_sandboxes:
        lines.extend(["", "### 工具沙盒"])
        for sandbox in report.tool_sandboxes:
            lines.append(f"- **{sandbox.tool}**：schema={sandbox.schema_name}，证据={_source_refs(sandbox.evidence_ids)}")
    if report.tool_results:
        lines.extend(["", "### 工具执行结果"])
        for index, result in enumerate(report.tool_results, start=1):
            lines.append(
                f"- **tool_result_{index} / {result.tool_name}**：schema={result.schema_name}，"
                f"claims={len(result.claims)}，confidence={result.confidence:.2f}，evidence={_source_refs(result.evidence_ids)}"
            )
            for claim in result.claims[:3]:
                lines.append(f"  - {claim.title}: {claim.claim[:180]}（证据：{_source_refs(claim.evidence_ids)}）")

    return "\n".join(line for line in lines if line is not None).strip() + "\n"


def has_required_deep_dive_sections(markdown: str) -> bool:
    required_patterns = [
        r"#\s+.+竞品分析报告",
        r"##\s+[一二三四五六七八九十]+、分析背景",
        r"##\s+[一二三四五六七八九十]+、竞品选择",
        r"##\s+[一二三四五六七八九十]+、分析维度",
        r"##\s+[一二三四五六七八九十]+、关键发现",
        r"##\s+[一二三四五六七八九十]+、总结建议",
        r"##\s+[一二三四五六七八九十]+、附录",
    ]
    return all(re.search(pattern, markdown) for pattern in required_patterns)


def _analysis_dimensions(report: CompetitorReport) -> list[tuple[str, str]]:
    competitors = report.competitors[:5]
    rows = []
    for profile in competitors:
        rows.append(
            f"| {profile.name} | {profile.positioning} | {_join(profile.core_features)} | "
            f"{profile.business_model or 'Unknown'} | {_source_refs(profile.evidence_ids)} |"
        )
    dimension_1 = "\n".join(
        [
            "| 竞品 | 定位 | 核心功能 | 商业模式 | 证据 |",
            "| --- | --- | --- | --- | --- |",
            *rows,
        ]
    )

    matrix_rows = []
    for cell in report.comparison_matrix[:12]:
        matrix_rows.append(f"| {cell.competitor} | {cell.dimension} | {cell.value} | {_source_refs(cell.evidence_ids)} |")
    if not matrix_rows:
        matrix_rows.append("| 待补充 | 对比维度 | 需要更多来源支撑 | 待补充 |")
    dimension_2 = "\n".join(
        [
            "| 对象 | 维度 | 观察 | 证据 |",
            "| --- | --- | --- | --- |",
            *matrix_rows,
        ]
    )

    swot = report.swot
    dimension_3 = "\n".join(
        [
            f"- **优势**：{_join(swot.strengths, '待补充')}",
            f"- **劣势**：{_join(swot.weaknesses, '待补充')}",
            f"- **机会**：{_join(swot.opportunities, '待补充')}",
            f"- **威胁**：{_join(swot.threats, '待补充')}",
        ]
    )
    tool_rows = []
    for result in report.tool_results:
        top_claims = "<br>".join(
            f"{claim.title}: {claim.claim[:120]} ({_source_refs(claim.evidence_ids)})"
            for claim in result.claims[:4]
        )
        tool_rows.append(
            f"| {result.tool_name} | {result.schema_name} | {result.confidence:.2f} | "
            f"{top_claims or '暂无工具结论'} | {_source_refs(result.evidence_ids)} |"
        )
    if tool_rows:
        dimension_3 = "\n".join(
            [
                dimension_3,
                "",
                "| 工具 | Schema | 置信度 | 关键结论 | 证据 |",
                "| --- | --- | ---: | --- | --- |",
                *tool_rows,
            ]
        )
    return [
        ("定位、功能与商业模式", dimension_1),
        ("横向比较矩阵", dimension_2),
        ("SWOT 与工具结论", dimension_3),
    ]
