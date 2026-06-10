from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from myagent.backend.app.models import (
    AnalysisBenchmark,
    AnalysisPurpose,
    CandidateCompetitor,
    ClarificationOption,
    ClarificationRequest,
    CollectionInstruction,
    CollectionPlan,
    CompetitorInput,
    CompetitorProfile,
    CompetitorType,
    CreateTaskRequest,
    LifecycleStage,
    RankFactors,
    RawSource,
    RiskFactors,
    ScoringProfile,
    SelectedCompetitorSet,
    SourceRef,
    ToolPlan,
    ToolSandbox,
)


PURPOSE_LABELS: dict[AnalysisPurpose, str] = {
    "decision_support": "决策支持",
    "learning": "学习借鉴",
    "market_warning": "市场预警",
}

STAGE_LABELS: dict[LifecycleStage, str] = {
    "concept": "概念期",
    "development": "研发期",
    "launched": "已上线运营期",
}

TYPE_LABELS: dict[CompetitorType, str] = {
    "head_direct": "头部直接竞品",
    "challenger": "追赶型竞品",
    "indirect_cross": "间接/跨界竞品",
    "potential_substitute": "潜在替代品",
}

TYPE_WEIGHTS: dict[AnalysisPurpose, dict[str, float]] = {
    "decision_support": {
        "head_direct": 1.5,
        "challenger": 1.0,
        "indirect_cross": 0.8,
        "potential_substitute": 0.5,
    },
    "learning": {
        "head_direct": 1.2,
        "challenger": 0.8,
        "indirect_cross": 1.5,
        "potential_substitute": 0.2,
    },
    "market_warning": {
        "head_direct": 0.5,
        "challenger": 1.5,
        "indirect_cross": 1.0,
        "potential_substitute": 2.0,
    },
}

AVAILABLE_CANVAS_TOOLS: list[str] = ["精益画布", "战略画布", "竞品画布"]
AVAILABLE_METHODS: list[str] = [
    "比较法",
    "矩阵分析法",
    "竞品跟踪矩阵",
    "功能拆解",
    "探索需求",
    "PEST分析",
    "波特五力模型",
    "SWOT分析",
    "加减乘除(ERRAC)",
]


def infer_benchmark(request: CreateTaskRequest) -> tuple[AnalysisBenchmark | None, ClarificationRequest | None]:
    text = _combined_input(request)
    target_product = request.target_product or _infer_product_name(text)
    stage = request.lifecycle_stage or _infer_stage(text)
    purpose = request.analysis_purpose or _infer_purpose(text)
    goal = request.analysis_goal or _infer_goal(text, purpose)

    missing: list[str] = []
    if not target_product:
        missing.append("target_product")
    if not stage:
        missing.append("lifecycle_stage")
    if not purpose:
        missing.append("analysis_purpose")
    if not goal:
        missing.append("analysis_goal")

    if target_product and request.competitors and missing:
        stage = stage or "concept"
        purpose = purpose or "decision_support"
        goal = goal or "支撑竞品分析与产品定位决策"
        missing = [item for item in missing if item == "target_product"]

    if missing:
        return None, build_clarification_request(missing, request, target_product, stage, purpose)

    confidence = 0.55
    confidence += 0.15 if request.lifecycle_stage else 0.05
    confidence += 0.15 if request.analysis_purpose else 0.05
    confidence += 0.10 if request.analysis_goal else 0.03
    confidence += 0.10 if request.target_product else 0.02
    benchmark = AnalysisBenchmark(
        target_product=target_product,
        raw_description=text,
        lifecycle_stage=stage,
        analysis_purpose=purpose,
        analysis_goal=goal,
        confidence=min(confidence, 0.95),
        inferred_from=[item for item in ["raw_description", "structured_fields", "clarification_answers"] if text],
        locked=True,
    )
    return benchmark, None


def merge_clarification_answer(request: CreateTaskRequest, clarification: ClarificationRequest, answer: str) -> CreateTaskRequest:
    updates: dict[str, str] = {}
    for option in clarification.options:
        if option.value == answer or option.label == answer:
            updates.update(option.updates)
            break

    merged = request.model_copy(deep=True)
    merged.clarification_answers = [*merged.clarification_answers, answer]
    if merged.raw_description:
        merged.raw_description = f"{merged.raw_description}\n澄清补充：{answer}"
    else:
        merged.raw_description = f"澄清补充：{answer}"

    if "target_product" in updates and not merged.target_product:
        merged.target_product = updates["target_product"]
    if "lifecycle_stage" in updates:
        merged.lifecycle_stage = updates["lifecycle_stage"]  # type: ignore[assignment]
    if "analysis_purpose" in updates:
        merged.analysis_purpose = updates["analysis_purpose"]  # type: ignore[assignment]
    if "analysis_goal" in updates:
        merged.analysis_goal = updates["analysis_goal"]

    lower = answer.lower()
    if not merged.lifecycle_stage:
        merged.lifecycle_stage = _infer_stage(answer)
    if not merged.analysis_purpose:
        merged.analysis_purpose = _infer_purpose(answer)
    if not merged.analysis_goal and len(answer) >= 8:
        merged.analysis_goal = answer
    if not merged.target_product:
        merged.target_product = _infer_product_name(answer) or merged.target_product
    if not merged.analysis_purpose:
        if any(word in lower for word in ["市场", "空间", "商业", "定位", "大盘"]):
            merged.analysis_purpose = "decision_support"
        elif any(word in lower for word in ["学习", "借鉴", "体验", "功能"]):
            merged.analysis_purpose = "learning"
        elif any(word in lower for word in ["预警", "风险", "威胁", "黑马"]):
            merged.analysis_purpose = "market_warning"
    return CreateTaskRequest.model_validate(merged.model_dump(mode="json"))


def build_clarification_request(
    missing: list[str],
    request: CreateTaskRequest,
    target_product: str,
    stage: LifecycleStage | None,
    purpose: AnalysisPurpose | None,
) -> ClarificationRequest:
    product = target_product or request.target_product or "这个产品"
    if "lifecycle_stage" in missing:
        return ClarificationRequest(
            missing_slots=missing,
            context_summary=_combined_input(request)[:500],
            question=(
                f"为了让竞品分析更贴合当前决策，请确认 {product} 现在更接近哪个阶段？"
            ),
            options=[
                ClarificationOption(
                    label="刚有初步想法",
                    value="concept",
                    description="重点分析市场空间、定位机会和主要对手。",
                    updates={"lifecycle_stage": "concept", "analysis_goal": "验证市场空间与产品定位"},
                ),
                ClarificationOption(
                    label="准备设计开发",
                    value="development",
                    description="重点分析功能差异化、体验路径和MVP优先级。",
                    updates={"lifecycle_stage": "development", "analysis_goal": "找到功能差异化与MVP优先级"},
                ),
                ClarificationOption(
                    label="已上线运营",
                    value="launched",
                    description="重点分析增长、留存、商业化和竞对异动。",
                    updates={"lifecycle_stage": "launched", "analysis_goal": "优化增长、运营与商业化策略"},
                ),
            ],
        )
    if "analysis_purpose" in missing or "analysis_goal" in missing:
        return ClarificationRequest(
            missing_slots=missing,
            context_summary=_combined_input(request)[:500],
            question=f"这次分析 {product}，你最希望它帮助你完成哪类判断？",
            options=[
                ClarificationOption(
                    label="决策支持",
                    value="decision_support",
                    description="摸清大盘、判断是否进入或如何定位。",
                    updates={"analysis_purpose": "decision_support", "analysis_goal": "支撑立项、战略调整或市场定位决策"},
                ),
                ClarificationOption(
                    label="学习借鉴",
                    value="learning",
                    description="拆解优秀体验、功能路径和跨界做法。",
                    updates={"analysis_purpose": "learning", "analysis_goal": "借鉴竞品体验与功能创新做法"},
                ),
                ClarificationOption(
                    label="市场预警",
                    value="market_warning",
                    description="识别黑马、跨界威胁和潜在替代品。",
                    updates={"analysis_purpose": "market_warning", "analysis_goal": "识别潜在威胁与竞对异动"},
                ),
            ],
        )
    return ClarificationRequest(
        missing_slots=missing,
        context_summary=_combined_input(request)[:500],
        question="请补充产品名称或一句话产品描述，方便我锁定分析对象。",
        options=[
            ClarificationOption(
                label="补充产品描述",
                value="describe_product",
                description="说明产品是什么、面向谁、想解决什么问题。",
                updates={},
            )
        ],
    )


def build_candidate_queries(benchmark: AnalysisBenchmark, request: CreateTaskRequest) -> list[str]:
    context = " ".join(
        item
        for item in [
            benchmark.target_product,
            request.market or "",
            request.product_category or "",
            request.geography or "",
            benchmark.analysis_goal,
        ]
        if item
    )
    purpose_label = PURPOSE_LABELS[benchmark.analysis_purpose]
    queries = [
        f"{context} 竞品 替代品 头部产品 黑马",
        f"{context} competitors alternatives market map",
        f"{benchmark.target_product} 类似产品 排名 对比 {purpose_label}",
        f"{context} 融资 增长 用户评价 发布 更新",
    ]
    for competitor in request.competitors:
        queries.append(f"{benchmark.target_product} vs {competitor} 竞品 分析")
    return _dedupe(queries)


def build_candidate_queries(benchmark: AnalysisBenchmark, request: CreateTaskRequest) -> list[str]:
    context = " ".join(
        item
        for item in [
            benchmark.target_product,
            request.market or "",
            request.product_category or "",
            request.geography or "",
            benchmark.analysis_goal,
        ]
        if item
    )
    purpose_label = PURPOSE_LABELS[benchmark.analysis_purpose]
    queries = [
        f"{context} 竞品 替代品 头部产品 黑马",
        f"{context} 竞品地图 替代方案 产品榜单",
        f"{benchmark.target_product} 类似产品 排名 对比 {purpose_label}",
        f"{context} 融资 增长 用户评价 发布 更新",
    ]
    for competitor in request.competitors:
        queries.append(f"{benchmark.target_product} vs {competitor} 竞品 分析")
    return _dedupe(queries)[:5]


def build_candidates_from_sources(
    benchmark: AnalysisBenchmark,
    request: CreateTaskRequest,
    sources: list[RawSource],
) -> list[CandidateCompetitor]:
    buckets: dict[str, CandidateCompetitor] = {}
    for profile in request.competitor_profiles:
        candidate = _candidate_from_profile(profile, benchmark)
        buckets[_key(candidate.name)] = candidate
    for competitor in request.competitors:
        key = _key(competitor)
        if key not in buckets:
            buckets[key] = CandidateCompetitor(
                name=competitor,
                description=f"用户指定的候选竞品：{competitor}",
                competitor_type="head_direct",
                tags=["user_specified"],
                selection_reason="用户输入中明确指定，需要进入候选池参与评分。",
            )

    for source in sources:
        names = _extract_candidate_names(source, benchmark)
        for name in names:
            if not _is_valid_candidate_name(name, benchmark):
                continue
            key = _key(name)
            existing = buckets.get(key)
            if not existing:
                existing = CandidateCompetitor(name=name)
                buckets[key] = existing
            _merge_source_into_candidate(existing, source, benchmark)

    if not buckets:
        buckets[_key(benchmark.target_product)] = CandidateCompetitor(
            name=benchmark.target_product,
            description="未召回到明确竞品，使用目标产品作为低置信度占位候选。",
            competitor_type="head_direct",
            tags=["fallback"],
        )

    candidates = _filter_candidate_pool(list(buckets.values()), benchmark)
    for candidate in candidates:
        _classify_candidate(candidate, benchmark, request)
        candidate.rank_factors = _rank_factors(candidate, benchmark)
        candidate.risk_factors = _risk_factors(candidate)
    return candidates[:50]


def score_and_select_candidates(
    benchmark: AnalysisBenchmark,
    candidates: list[CandidateCompetitor],
    budget: int = 5,
) -> SelectedCompetitorSet:
    scoring = scoring_profile_for(benchmark.analysis_purpose)
    for candidate in candidates:
        type_weight = scoring.type_weights[candidate.competitor_type]
        factors = candidate.rank_factors
        base = (
            scoring.alpha_heat * factors.heat
            + scoring.beta_growth * factors.growth
            + scoring.gamma_similarity * factors.similarity
        )
        candidate.score = round(min(1.0, type_weight * base), 4)
        candidate.weight_assignment = {
            "W_type": round(type_weight, 4),
            "alpha_heat": round(scoring.alpha_heat, 4),
            "beta_growth": round(scoring.beta_growth, 4),
            "gamma_similarity": round(scoring.gamma_similarity, 4),
        }
        candidate.factor_scores = {
            "f_heat": round(factors.heat, 4),
            "f_growth": round(factors.growth, 4),
            "f_similarity": round(factors.similarity, 4),
            "f_risk": round(factors.risk, 4),
        }
        candidate.selection_reason = _selection_reason(candidate, scoring)
        candidate.scoring_reason = {
            "summary": candidate.selection_reason,
            "details": {
                "heat_analysis": f"市场热度因子得分 {factors.heat:.2f}，来自候选来源数量、热度关键词和公开资料信号。",
                "growth_analysis": f"增速因子得分 {factors.growth:.2f}，来自增长、融资、发布、招聘或更新相关信号。",
                "similarity_analysis": f"相似度因子得分 {factors.similarity:.2f}，来自目标产品描述与竞品简介的文本/语义重合。",
                "risk_analysis": "风险因子仅作为兼容字段保留，不参与阶段二评分公式。",
            },
        }

    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    allocation = allocation_for(benchmark.analysis_purpose, budget)
    selected: list[CandidateCompetitor] = []
    used: set[str] = set()

    for competitor_type, count in allocation.items():
        type_ranked = [item for item in ranked if item.competitor_type == competitor_type and _key(item.name) not in used]
        for item in type_ranked[:count]:
            selected.append(item)
            used.add(_key(item.name))

    fallback_notes: list[str] = []
    if len(selected) < budget:
        fallback_notes.append(
            f"候选池中部分类型数量不足，剩余名额按综合得分递补到 {budget} 个。"
        )
        for item in ranked:
            if len(selected) >= budget:
                break
            if _key(item.name) not in used:
                selected.append(item)
                used.add(_key(item.name))

    if len(ranked) < budget:
        fallback_notes.append(f"实际候选数只有 {len(ranked)} 个，未达到默认深度分析预算 {budget}。")

    return SelectedCompetitorSet(
        total_budget=budget,
        scoring_profile=scoring,
        candidates=ranked,
        selected=selected[:budget],
        allocation=allocation,
        fallback_notes=fallback_notes,
    )


def scoring_profile_for(purpose: AnalysisPurpose) -> ScoringProfile:
    if purpose == "decision_support":
        return ScoringProfile(
            purpose=purpose,
            alpha_heat=0.35,
            beta_growth=0.2,
            gamma_similarity=0.35,
            delta_risk=0.0,
            type_weights=TYPE_WEIGHTS[purpose],
        )
    if purpose == "learning":
        return ScoringProfile(
            purpose=purpose,
            alpha_heat=0.2,
            beta_growth=0.25,
            gamma_similarity=0.3,
            delta_risk=0.0,
            type_weights=TYPE_WEIGHTS[purpose],
        )
    return ScoringProfile(
        purpose=purpose,
        alpha_heat=0.15,
        beta_growth=0.45,
        gamma_similarity=0.2,
        delta_risk=0.0,
        type_weights=TYPE_WEIGHTS[purpose],
    )


def allocation_for(purpose: AnalysisPurpose, budget: int = 5) -> dict[str, int]:
    if purpose == "decision_support":
        allocation = {"head_direct": 2, "indirect_cross": 1}
    elif purpose == "learning":
        allocation = {"head_direct": 1, "indirect_cross": 2}
    else:
        allocation = {"head_direct": 1, "indirect_cross": 1, "potential_substitute": 1}
    total = sum(allocation.values())
    if total != budget:
        allocation["challenger"] = allocation.get("challenger", 0) + (budget - total)
    return allocation


def build_tool_plan(benchmark: AnalysisBenchmark, selected_set: SelectedCompetitorSet, request: CreateTaskRequest) -> ToolPlan:
    matrix = {
        "decision_support": ToolPlan(
            purpose="decision_support",
            focus="宏观环境、商业模式、自身资源与市场空缺的匹配度",
            frameworks=["精益画布", "战略画布"],
            methods=["PEST分析", "SWOT分析"],
            selected_tools=["精益画布", "战略画布", "PEST分析", "SWOT分析"],
            rationale="用于支撑立项或战略调整，优先看商业模式、定位差异、宏观环境和自身机会。",
        ),
        "learning": ToolPlan(
            purpose="learning",
            focus="微观体验、核心路径、用户痛点、跨界创新点",
            frameworks=["竞品画布"],
            methods=["功能拆解", "探索需求", "加减乘除(ERRAC)"],
            selected_tools=["竞品画布", "功能拆解", "探索需求", "加减乘除(ERRAC)"],
            rationale="用于寻找体验和功能创新灵感，优先拆解路径、评价反馈和跨界做法。",
        ),
        "market_warning": ToolPlan(
            purpose="market_warning",
            focus="竞对异动、外部环境变化、资源重组、相对优势的衰退",
            frameworks=[],
            methods=["竞品跟踪矩阵", "矩阵分析法", "PEST分析"],
            selected_tools=["竞品跟踪矩阵", "矩阵分析法", "PEST分析"],
            rationale="用于识别黑马、外部风险和替代威胁，优先跟踪增长、融资、招聘、技术路线与宏观环境变化。",
        ),
    }
    plan = matrix[benchmark.analysis_purpose]
    names = "、".join(item.name for item in selected_set.selected[:3])
    if request.product_category and benchmark.analysis_purpose == "learning":
        plan.rationale += f" 当前品类为 {request.product_category}，会把 {names} 的核心路径和用户抱怨作为重点。"
    return plan


def build_collection_plan(
    benchmark: AnalysisBenchmark,
    selected_set: SelectedCompetitorSet,
    tool_plan: ToolPlan,
) -> CollectionPlan:
    instructions: list[CollectionInstruction] = []
    for competitor in selected_set.selected:
        for tool in tool_plan.selected_tools:
            queries = _queries_for_tool(tool, benchmark, competitor)
            instructions.append(
                CollectionInstruction(
                    tool=tool,
                    competitor=competitor.name,
                    queries=queries,
                    source_types=_source_types_for_tool(tool),
                    extraction_targets=_extraction_targets_for_tool(tool),
                )
            )
    return CollectionPlan(
        instructions=instructions,
        notes=[
            "当前版本使用 SearchProvider 复用 Tavily/DuckDuckGo/URL 采集。",
            "AppStoreProvider、IndustryDatabaseProvider、HiringSignalProvider、FinancingNewsProvider 已作为接口名预留。",
        ],
    )


def build_tool_sandboxes(
    tool_plan: ToolPlan,
    selected_set: SelectedCompetitorSet,
    evidence_ids: list[str],
) -> list[ToolSandbox]:
    competitor_names = [item.name for item in selected_set.selected]
    sandboxes: list[ToolSandbox] = []
    for tool in tool_plan.selected_tools:
        if tool == "SWOT分析":
            data = {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}
            schema = "swot_schema"
        elif tool == "PEST分析":
            data = {"political": [], "economic": [], "social": [], "technological": []}
            schema = "pest_analysis_schema"
        elif tool == "功能拆解":
            data = {"competitors": {name: {"level_1": [], "level_2": [], "level_3": []} for name in competitor_names}}
            schema = "feature_tree_schema"
        elif tool == "竞品跟踪矩阵":
            data = {"timeline": {name: [] for name in competitor_names}}
            schema = "tracking_matrix_schema"
        else:
            data = {"competitors": competitor_names, "observations": []}
            schema = "generic_tool_schema"
        sandboxes.append(ToolSandbox(tool=tool, schema_name=schema, data=data, evidence_ids=evidence_ids[:8]))
    return sandboxes


def apply_selection_to_profiles(
    profiles: list[CompetitorProfile],
    selected_set: SelectedCompetitorSet | None,
) -> list[CompetitorProfile]:
    if not selected_set:
        return profiles
    selected_by_name = {_key(item.name): item for item in selected_set.selected}
    ordered: list[CompetitorProfile] = []
    existing_by_name = {_key(profile.name): profile for profile in profiles}
    for candidate in selected_set.selected:
        profile = existing_by_name.get(_key(candidate.name))
        if profile is None:
            profile = CompetitorProfile(name=candidate.name)
        profile.website = profile.website or candidate.website
        profile.competitor_type = candidate.competitor_type
        profile.score = candidate.score
        profile.selection_reason = candidate.selection_reason
        profile.rank_factors = candidate.rank_factors
        if profile.positioning == "Unknown":
            profile.positioning = candidate.description or f"{TYPE_LABELS[candidate.competitor_type]}，需结合公开资料继续核实。"
        ordered.append(profile)
    for profile in profiles:
        if _key(profile.name) not in selected_by_name:
            ordered.append(profile)
    return ordered


def _combined_input(request: CreateTaskRequest) -> str:
    parts = [
        request.raw_description or "",
        request.target_product,
        request.our_product or "",
        request.market or "",
        request.product_category or "",
        request.analysis_goal or "",
        request.notes or "",
        " ".join(request.clarification_answers),
    ]
    return "\n".join(part for part in parts if part).strip()


def _infer_product_name(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"(?:想做|开发|设计|分析一下|分析|研究|对比)\s*([A-Za-z0-9\u4e00-\u9fff·\-\s]{2,40}?)(?:，|,|。|；|;|$)",
        r"产品(?:叫|名称为|是)\s*([A-Za-z0-9\u4e00-\u9fff·\-\s]{2,40}?)(?:，|,|。|；|;|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).strip()
            name = re.sub(r"^(一个|一款|这个|关于)", "", name).strip()
            if name:
                return name[:80]
    first_line = text.strip().splitlines()[0]
    if len(first_line) <= 40:
        return first_line.strip(" ：:，,。")
    return ""


def _infer_stage(text: str) -> LifecycleStage | None:
    if not text:
        return None
    if re.search(r"想法|概念|立项|规划|战略|准备做|想做|验证|市场空间", text):
        return "concept"
    if re.search(r"研发|开发|设计|MVP|原型|准备上线|功能差异", text, re.I):
        return "development"
    if re.search(r"上线|运营|增长|留存|转化|版本|商业化|用户反馈", text):
        return "launched"
    return None


def _infer_purpose(text: str) -> AnalysisPurpose | None:
    if not text:
        return None
    if re.search(r"决策|立项|市场空间|定位|战略|商业模式|进入|大盘", text):
        return "decision_support"
    if re.search(r"学习|借鉴|体验|功能|路径|差异化|灵感|优化", text):
        return "learning"
    if re.search(r"预警|风险|威胁|黑马|替代|融资|异动|超车", text):
        return "market_warning"
    return None


def _infer_goal(text: str, purpose: AnalysisPurpose | None) -> str:
    if text and len(text) >= 12:
        sentences = re.split(r"[。！？!?;\n]", text)
        for sentence in sentences:
            cleaned = sentence.strip()
            if 8 <= len(cleaned) <= 120 and re.search(r"分析|判断|了解|找到|优化|预警|借鉴|决策", cleaned):
                return cleaned
    if purpose == "decision_support":
        return "支撑产品立项、战略调整与市场定位判断"
    if purpose == "learning":
        return "借鉴竞品功能体验与跨界创新做法"
    if purpose == "market_warning":
        return "识别潜在威胁、黑马竞品与替代风险"
    return ""


def _candidate_from_profile(profile: CompetitorInput, benchmark: AnalysisBenchmark) -> CandidateCompetitor:
    category_map = {
        "direct": "head_direct",
        "head_direct": "head_direct",
        "challenger": "challenger",
        "indirect": "indirect_cross",
        "indirect_cross": "indirect_cross",
        "cross": "indirect_cross",
        "substitute": "potential_substitute",
        "potential_substitute": "potential_substitute",
    }
    description = " ".join(
        item
        for item in [profile.website_copy, profile.sales_notes, profile.win_loss_notes, ", ".join(profile.tags)]
        if item
    )
    return CandidateCompetitor(
        name=profile.name,
        description=description or f"用户提供的 {profile.name} 资料",
        website=profile.website or None,
        competitor_type=category_map.get(profile.category, "head_direct"),  # type: ignore[arg-type]
        source_urls=[profile.website] if profile.website else [],
        tags=["user_specified", *profile.tags],
        selection_reason="用户在结构化竞品资料中提供，作为高优先级候选。",
    )


def _extract_candidate_names(source: RawSource, benchmark: AnalysisBenchmark) -> list[str]:
    title = source.title or ""
    content = source.content or ""
    text = f"{title}\n{content[:1000]}"
    names: list[str] = []
    names.extend(_split_possible_product_names(title, benchmark))
    title_name = re.split(r"[-_|:：—]", title)[0].strip()
    if _looks_like_name(title_name, benchmark):
        names.append(title_name)

    patterns = [
        r"([A-Z][A-Za-z0-9][A-Za-z0-9 .\-]{1,28})\s+(?:is|offers|announces|launches)",
        r"([\u4e00-\u9fffA-Za-z0-9·]{2,24})(?:是一款|是一个|发布|推出|完成融资|获得融资)",
        r"(?:竞品|替代品|类似产品|包括|例如|如)[:：]?\s*([\u4e00-\u9fffA-Za-z0-9·、,，\s-]{2,120})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            fragment = match.group(1).strip()
            for name in re.split(r"[、,，/]|以及|和|与", fragment):
                cleaned = name.strip(" .。；;:：")
                if _looks_like_name(cleaned, benchmark):
                    names.append(cleaned)
    return _dedupe([name for name in names if _is_valid_candidate_name(name, benchmark)])[:8]


def _looks_like_name(value: str, benchmark: AnalysisBenchmark) -> bool:
    if not value or len(value) < 2 or len(value) > 40:
        return False
    if benchmark.target_product and value == benchmark.target_product:
        return False
    bad_words = {
        "竞品分析",
        "产品分析",
        "官方网站",
        "价格",
        "功能",
        "用户",
        "市场",
        "搜索结果",
        "Mock search",
        "DuckDuckGo result",
        "DuckDuckGo",
        "search",
        "map",
        "competitors",
        "alternatives",
        "market",
    }
    if value in bad_words:
        return False
    if re.search(r"https?://|manual://|failed|error", value, re.I):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", value))


def _split_possible_product_names(text: str, benchmark: AnalysisBenchmark) -> list[str]:
    if not text:
        return []
    normalized = re.sub(r"(?i)vs\.?|对比|比较|竞品分析|全面对比|深度对比|选择指南", "、", text)
    normalized = re.sub(r"[|:：—\-_/（）()【】\\[\\]]", "、", normalized)
    names: list[str] = []
    for part in re.split(r"[、,，;；\s]+", normalized):
        cleaned = _normalize_candidate_name(part.strip(" .。！？!?…"))
        if _looks_like_name(cleaned, benchmark) and _is_valid_candidate_name(cleaned, benchmark):
            names.append(cleaned)
    return names


def _filter_candidate_pool(
    candidates: list[CandidateCompetitor],
    benchmark: AnalysisBenchmark,
) -> list[CandidateCompetitor]:
    filtered: list[CandidateCompetitor] = []
    seen: set[str] = set()
    for candidate in candidates:
        if "user_specified" not in candidate.tags and not _is_valid_candidate_name(candidate.name, benchmark):
            continue
        key = _key(candidate.name)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(candidate)
    return filtered


def _is_valid_candidate_name(value: str, benchmark: AnalysisBenchmark) -> bool:
    name = _normalize_candidate_name(re.sub(r"\s+", " ", value or "").strip(" .。！？!?;；:：…"))
    if not _looks_like_name(name, benchmark):
        return False

    target_key = _key(benchmark.target_product)
    name_key = _key(name)
    if target_key and (name_key in {target_key, f"{target_key}feishu", "feishu"} or target_key in name_key):
        return False

    short_name_allowlist = {"钉钉", "企微", "飞连", "蓝湖", "语雀", "Zoom", "Slack", "Notion", "Trello", "Asana"}
    if len(name) <= 2 and name not in short_name_allowlist:
        return False
    if len(name) > 28 and not re.search(r"[A-Za-z]{2,}", name):
        return False

    invalid_exact = {
        "全新",
        "根据艾瑞咨询",
        "如何选择",
        "怎么选择",
        "哪款好",
        "排行榜",
        "排名",
        "何选择最适合自己的那一款",
        "官网",
        "首页",
        "下载",
        "博客园",
        "大发明家2",
        "先进团队",
        "先用",
        "与选择指南",
        "场景与生态选择指南",
        "DuckDuckGo",
        "search",
        "map",
        "competitors",
        "alternatives",
        "market",
        "替代品",
        "头部产品",
        "类似产品",
        "学习借鉴",
        "用户评价",
        "功能体验",
        "企业协同办公",
        "协作办公套件",
        "借鉴竞品功能体验与跨界创新做法",
        "跨界创新做法 融资 增长 用户评价 发布 更新",
    }
    if name in invalid_exact:
        return False

    article_patterns = [
        r"揭秘|盘点|推荐|排行榜|排名|哪款|怎么选|如何选择|何选择|最适合|全面对比|深度解析|知识整理|系统工程|选择指南|生态选择|场景与生态",
        r"\d+\s*款|\d+\s*大|\d+\s*个",
        r"根据.+咨询|研究报告|市场报告|白皮书",
        r"下载|官网|博客园|知乎|头条|文章|作者|团队|先进团队|先用",
        r"需求|目的|功能|体验|用户|评价|评论|融资|增长|发布|更新|创新做法",
    ]
    if any(re.search(pattern, name, re.I) for pattern in article_patterns):
        return False

    if re.search(r"vs|/|…|\.{2,}|对比", name, re.I):
        return False

    generic_terms = {
        "企业",
        "协同",
        "协作",
        "办公",
        "软件",
        "产品",
        "工具",
        "平台",
        "系统",
        "方案",
        "竞品",
        "市场",
        "替代",
        "头部",
    }
    tokens = set(_tokens(name))
    if tokens and tokens <= generic_terms:
        return False

    return True


def _normalize_candidate_name(value: str) -> str:
    aliases = {
        "企微": "企业微信",
        "飞书深度": "飞书",
    }
    cleaned = re.sub(r"^(?:和|与|及|以及)", "", value.strip())
    cleaned = re.sub(r"(?:电脑版|网页版|客户端下载|下载)$", "", cleaned)
    return aliases.get(cleaned, cleaned)


def _merge_source_into_candidate(candidate: CandidateCompetitor, source: RawSource, benchmark: AnalysisBenchmark) -> None:
    if source.url and source.url not in candidate.source_urls:
        candidate.source_urls.append(source.url)
    if source.title and source.title not in candidate.source_titles:
        candidate.source_titles.append(source.title)
    ref_key = f"{source.id}|{source.metadata.get('chunk_id') or ''}|{source.url}"
    existing_ref_keys = {
        f"{item.source_id or ''}|{item.chunk_id or ''}|{item.url}"
        for item in candidate.source_refs
    }
    if source.url and ref_key not in existing_ref_keys:
        chunk_paths = source.metadata.get("chunk_shot_paths") or []
        if isinstance(chunk_paths, str):
            chunk_paths = [chunk_paths]
        candidate.source_refs.append(
            SourceRef(
                source_id=source.id,
                chunk_id=source.metadata.get("chunk_id"),
                url=source.url,
                title=source.title,
                quote=source.content[:500],
                screenshot_path=source.metadata.get("screenshot_path"),
                chunk_shot_paths=[str(item) for item in chunk_paths if item],
            )
        )
    content = source.content[:500]
    if content and len(content) > len(candidate.description):
        candidate.description = content
    provider = source.metadata.get("provider")
    if provider and str(provider) not in candidate.tags:
        candidate.tags.append(str(provider))
    query = str(source.metadata.get("query") or "")
    if query and benchmark.target_product.lower() in query.lower():
        candidate.tags.append("query_related")


def _classify_candidate(candidate: CandidateCompetitor, benchmark: AnalysisBenchmark, request: CreateTaskRequest) -> None:
    if "user_specified" in candidate.tags:
        return
    text = f"{candidate.name} {candidate.description} {' '.join(candidate.source_titles)}".lower()
    target_terms = set(_tokens(benchmark.target_product + " " + (request.product_category or "") + " " + (request.market or "")))
    candidate_terms = set(_tokens(text))
    overlap = len(target_terms & candidate_terms)
    if re.search(r"替代|alternative|substitute|开源|永久免费|跨界|ai|大模型", text, re.I):
        candidate.competitor_type = "potential_substitute"
    elif re.search(r"融资|增长|黑马|新锐|launch|announces|funding|series", text, re.I):
        candidate.competitor_type = "challenger"
    elif overlap >= 2 or re.search(r"竞品|competitor|vs|对比", text, re.I):
        candidate.competitor_type = "head_direct"
    else:
        candidate.competitor_type = "indirect_cross"


def _rank_factors(candidate: CandidateCompetitor, benchmark: AnalysisBenchmark) -> RankFactors:
    text = f"{candidate.name} {candidate.description} {' '.join(candidate.source_titles)} {' '.join(candidate.tags)}"
    lower = text.lower()
    source_count = len(candidate.source_urls)
    heat = min(1.0, 0.18 + source_count * 0.12)
    if re.search(r"头部|领先|leader|popular|top|排名|mau|下载|用户", lower, re.I):
        heat += 0.25
    growth = 0.15
    if re.search(r"融资|增长|飙升|新增|launch|announces|funding|series|招聘|版本|更新", lower, re.I):
        growth += 0.45
    similarity = SequenceMatcher(None, benchmark.raw_description.lower(), text.lower()).ratio()
    if benchmark.target_product and benchmark.target_product.lower() in lower:
        similarity += 0.25
    if "query_related" in candidate.tags:
        similarity += 0.15
    risk = 0.1
    if re.search(r"字节|腾讯|阿里|百度|BAT|Google|Microsoft|OpenAI|大厂|AI|大模型|免费|开源|替代", text, re.I):
        risk += 0.55
    return RankFactors(
        heat=round(min(heat, 1.0), 3),
        growth=round(min(growth, 1.0), 3),
        similarity=round(min(similarity, 1.0), 3),
        risk=round(min(risk, 1.0), 3),
    )


def _risk_factors(candidate: CandidateCompetitor) -> RiskFactors:
    text = f"{candidate.name} {candidate.description} {' '.join(candidate.source_titles)}"
    capital = 0.8 if re.search(r"字节|腾讯|阿里|百度|Google|Microsoft|OpenAI|融资|VC|A轮|B轮", text, re.I) else 0.2
    tech = 0.9 if re.search(r"AI|大模型|生成式|LLM|自动化|算法", text, re.I) else 0.2
    model = 0.8 if re.search(r"免费|开源|低价|订阅|freemium|open source", text, re.I) else 0.2
    team = 0.7 if re.search(r"创始人|连续创业|前.*负责人|团队", text, re.I) else 0.2
    return RiskFactors(capital=capital, tech=tech, model=model, team=team)


def _selection_reason(candidate: CandidateCompetitor, scoring: ScoringProfile) -> str:
    factors = candidate.rank_factors
    dominant = max(
        [
            ("热度", factors.heat * scoring.alpha_heat),
            ("增长", factors.growth * scoring.beta_growth),
            ("相似度", factors.similarity * scoring.gamma_similarity),
        ],
        key=lambda item: item[1],
    )[0]
    return (
        f"{TYPE_LABELS[candidate.competitor_type]}；综合分 {candidate.score:.2f}，"
        f"主要由{dominant}因子拉动；风险因子仅保留兼容，不参与阶段二总分。"
    )


def _queries_for_tool(tool: str, benchmark: AnalysisBenchmark, competitor: CandidateCompetitor) -> list[str]:
    base = competitor.name
    if tool == "PEST分析":
        return [
            f"{benchmark.target_product} {base} 政策 监管 行业 趋势 技术",
            f"{base} market trend regulation economy social technology",
            f"{benchmark.target_product} 行业报告 政策 经济 社会 技术 PEST",
        ]
    if tool in {"波特五力模型", "精益画布", "战略画布", "SWOT分析"}:
        return [
            f"{base} 商业模式 定价 渠道 用户 市场",
            f"{base} competitors pricing business model target users",
            f"{benchmark.target_product} {base} 对比 定位 差异",
        ]
    if tool in {"功能拆解", "探索需求", "加减乘除(ERRAC)", "竞品画布"}:
        return [
            f"{base} 功能 更新日志 release notes help docs",
            f"{base} 用户评价 差评 痛点",
            f"{base} app store reviews changelog features",
        ]
    return [
        f"{base} 近期 融资 招聘 新闻 发布 更新",
        f"{base} funding hiring release notes news",
        f"{base} AI 技术 战略 竞品异动",
    ]


def _source_types_for_tool(tool: str) -> list[str]:
    if tool in {"功能拆解", "探索需求", "加减乘除(ERRAC)"}:
        return ["官网", "帮助文档", "更新日志", "应用商店评论"]
    if tool == "PEST分析":
        return ["行业报告", "政策监管", "新闻", "技术趋势报告"]
    if tool in {"竞品跟踪矩阵", "矩阵分析法", "比较法"}:
        return ["新闻", "公众号/博客", "招聘", "融资数据库"]
    return ["官网", "定价页", "新闻", "行业报告"]


def _extraction_targets_for_tool(tool: str) -> list[str]:
    mapping = {
        "SWOT分析": ["优势", "劣势", "机会", "威胁"],
        "PEST分析": ["政治/监管因素", "经济因素", "社会因素", "技术因素", "外部机会", "外部风险"],
        "功能拆解": ["一级功能", "二级功能", "核心路径", "更新变化"],
        "探索需求": ["低星评价", "用户抱怨", "未满足需求"],
        "竞品跟踪矩阵": ["版本时间线", "营销动作", "招聘变化", "融资动作"],
        "战略画布": ["竞争要素", "价值曲线", "差异化定位"],
    }
    return mapping.get(tool, ["定位", "功能", "商业模式", "增长信号"])


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}", text)]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _key(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower())
