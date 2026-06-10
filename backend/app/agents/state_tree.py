from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from myagent.backend.app.agents.prompt_bridge import new_state_tree
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CollectionPlan,
    CompetitorReport,
    CreateTaskRequest,
    EvidenceItem,
    RawSource,
    SelectedCompetitorSet,
    ToolPlan,
)


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_initial_state_tree(request: CreateTaskRequest) -> dict[str, Any]:
    tree = new_state_tree()
    now = utc_iso()
    tree["global_metadata"].update(
        {
            "status": "init",
            "product_name": request.target_product or request.our_product or None,
            "created_at": now,
            "updated_at": now,
        }
    )
    return tree


def merge_updated_nodes(state_tree: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(state_tree)
    nodes = payload.get("updated_nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, dict):
        return updated
    for key, value in nodes.items():
        if key == "phase_2_competitors":
            key = "phase_2_targets"
        if key in updated and isinstance(updated[key], dict) and isinstance(value, dict):
            updated[key] = _deep_merge(updated[key], value)
        else:
            updated[key] = value
    updated.setdefault("global_metadata", {})["updated_at"] = utc_iso()
    return updated


def render_prompt(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", _to_json(value))
    return rendered


def benchmark_to_phase_1(benchmark: AnalysisBenchmark) -> dict[str, Any]:
    return {
        "product_lifecycle": benchmark.lifecycle_stage,
        "analysis_purpose": benchmark.analysis_purpose,
        "core_problem_to_solve": benchmark.raw_description or benchmark.analysis_goal,
        "specific_goal": benchmark.analysis_goal,
    }


def selected_set_to_phase_2(selected_set: SelectedCompetitorSet) -> dict[str, Any]:
    return {
        "selected_competitors": [
            {
                "competitor_name": item.name,
                "version": None,
                "competitor_type": item.competitor_type,
                "selection_reason": item.selection_reason,
                "rank_score": item.score,
                "source_refs": [ref.model_dump(mode="json") for ref in item.source_refs],
                "source_urls": item.source_urls,
                "source_titles": item.source_titles,
                "weight_assignment": item.weight_assignment,
                "factor_scores": item.factor_scores,
                "scoring_reason": item.scoring_reason,
            }
            for item in selected_set.selected
        ],
        "selection_logic": "; ".join(selected_set.fallback_notes) or "Selected by dynamic scoring profile.",
    }


def tool_plan_to_phase_3(tool_plan: ToolPlan, collection_plan: CollectionPlan) -> dict[str, Any]:
    directives: list[str] = []
    structured_directives: list[dict[str, Any]] = []
    for instruction in collection_plan.instructions:
        target = instruction.competitor or "all competitors"
        fields = ", ".join(instruction.extraction_targets)
        queries = "; ".join(instruction.queries[:3])
        directives.append(f"{instruction.tool} | {target} | fields: {fields} | queries: {queries}")
        structured_directives.append(
            {
                "analysis_unit": instruction.tool,
                "competitor": instruction.competitor,
                "queries": instruction.queries,
                "source_types": instruction.source_types,
                "extraction_targets": instruction.extraction_targets,
            }
        )
    selected_analysis_units = list(dict.fromkeys([*tool_plan.frameworks, *tool_plan.methods, *tool_plan.selected_tools]))
    return {
        "analysis_purpose": tool_plan.purpose,
        "analysis_focus": tool_plan.focus,
        "analysis_dimensions": [tool_plan.focus, *tool_plan.frameworks, *tool_plan.methods],
        "selected_tools": tool_plan.frameworks,
        "selected_methods": tool_plan.methods,
        "selected_analysis_units": selected_analysis_units,
        "selected_frameworks": selected_analysis_units,
        "selection_rationale": tool_plan.rationale,
        "collection_channels": collection_plan.provider_interfaces,
        "data_collection_spec": {
            "crawler_directives": directives,
            "structured_crawler_directives": structured_directives,
            "api_endpoints": [],
        },
    }


def sources_to_sandbox(sources: list[RawSource], evidence: list[EvidenceItem]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        items.append(
            {
                "id": evidence[index].id if index < len(evidence) else source.id,
                "title": source.title,
                "url": source.url,
                "content": source.content[:1600],
                "source_type": source.source_type,
                "metadata": source.metadata,
            }
        )
    news = [item for item in items if _matches_any(item, ["news", "funding", "融资", "新闻"])]
    reviews = [item for item in items if _matches_any(item, ["review", "app_store", "评价", "差评"])]
    features = [item for item in items if _matches_any(item, ["feature", "help", "docs", "功能", "更新"])]
    reports = [item for item in items if item not in news and item not in reviews and item not in features]
    return {
        "competitor_news_feed": news[:40],
        "user_reviews_scrape": reviews[:40],
        "feature_ocr_texts": features[:40],
        "market_reports": reports[:40],
    }


def report_to_phase_4(report: CompetitorReport) -> dict[str, Any]:
    dynamic: list[dict[str, Any]] = []
    if report.tool_results:
        for result in report.tool_results:
            dynamic.append(
                {
                    "framework_name": result.tool_name,
                    "analysis_unit_name": result.tool_name,
                    "analysis_unit_key": result.canonical_tool_name,
                    "analysis_unit_type": _analysis_unit_type(result.tool_name),
                    "framework_content": {
                        "claims": [claim.model_dump(mode="json") for claim in result.claims],
                        "data": result.data,
                        "evidence_ids": result.evidence_ids,
                        "confidence": result.confidence,
                    },
                }
            )
    elif report.swot:
        dynamic.append(
            {
                "framework_name": "SWOT分析",
                "framework_content": report.swot.model_dump(mode="json"),
                "analysis_unit_name": "SWOT分析",
                "analysis_unit_key": "swot",
                "analysis_unit_type": "method",
            }
        )
    return {
        "dynamic_analyses": dynamic,
        "conclusion": {
            "competitive_strategy": report.executive_summary,
            "actionable_suggestions": [
                {
                    "suggestion": item,
                    "priority": "medium",
                    "rationale": "Derived from phase_4_synthesis and cited evidence.",
                }
                for item in report.recommendations[:5]
            ],
        },
    }


def candidate_pool_payload(candidates: list[CandidateCompetitor]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in candidates]


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _matches_any(item: dict[str, Any], terms: list[str]) -> bool:
    text = _to_json(item).lower()
    return any(term.lower() in text for term in terms)


def merge_phase_3_strategy(
    llm_phase_3: dict[str, Any] | None,
    rule_tool_plan: ToolPlan,
    collection_plan: CollectionPlan,
    *,
    max_canvas_tools: int = 2,
    max_methods: int = 5,
) -> dict[str, Any]:
    llm_phase_3 = llm_phase_3 or {}
    llm_tools = _normalize_tool_list(llm_phase_3.get("selected_tools"))
    llm_methods = _normalize_tool_list(llm_phase_3.get("selected_methods"))
    rule_tools = _normalize_tool_list(rule_tool_plan.frameworks)
    rule_methods = _normalize_tool_list(rule_tool_plan.methods)

    merged_tools = _merge_limited_lists(llm_tools, rule_tools, max_canvas_tools)
    merged_methods = _merge_limited_lists(llm_methods, rule_methods, max_methods)
    merged_tool_plan = ToolPlan(
        purpose=rule_tool_plan.purpose,
        focus=llm_phase_3.get("analysis_focus") or rule_tool_plan.focus,
        frameworks=merged_tools,
        methods=merged_methods,
        selected_tools=[*merged_tools, *merged_methods],
        rationale=_merge_text_parts(llm_phase_3.get("selection_rationale"), rule_tool_plan.rationale),
    )
    return tool_plan_to_phase_3(merged_tool_plan, collection_plan)


def _normalize_tool_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _merge_limited_lists(primary: list[str], secondary: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*primary, *secondary]:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _merge_text_parts(primary: Any, secondary: Any) -> str:
    parts: list[str] = []
    for item in [primary, secondary]:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned and cleaned not in parts:
                parts.append(cleaned)
    return " ".join(parts)


def _analysis_unit_type(name: str) -> str:
    if name in {"精益画布", "战略画布", "竞品画布"}:
        return "tool"
    return "method"
