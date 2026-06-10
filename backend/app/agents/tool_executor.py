from __future__ import annotations

import json
import re
import asyncio
from typing import Any

from myagent.backend.app.agents.llm import retry_structured_ainvoke
from myagent.backend.app.agents.prompt_bridge import tool_executor_system_prompt, tool_executor_user_prompt
from myagent.backend.app.agents.tool_registry import ToolSpec, get_tool_spec, tool_specs_for_plan
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CompetitorDetailCardClean,
    IndustryCardClean,
    ReportReference,
    SelectedCompetitorSet,
    ToolClaim,
    ToolExecutionInput,
    ToolExecutionResult,
    ToolPlan,
    ToolSandbox,
)


DEFAULT_COMPETITOR_CARD_FIELDS = [
    "positioning",
    "core_features",
    "pricing",
    "business_model",
    "growth_signals",
    "risks",
]


COMPETITOR_CARD_FIELDS_BY_TOOL: dict[str, list[str]] = {
    "lean_canvas": [
        "positioning",
        "target_users",
        "core_scenarios",
        "pricing",
        "business_model",
        "channels",
        "growth_signals",
        "risks",
    ],
    "strategy_canvas": [
        "positioning",
        "differentiation",
        "pricing",
        "core_features",
        "key_parameters",
        "business_model",
    ],
    "swot": [
        "positioning",
        "core_features",
        "pricing",
        "business_model",
        "growth_signals",
        "risks",
        "differentiation",
        "user_feedback",
    ],
    "porter_five_forces": [
        "pricing",
        "business_model",
        "growth_signals",
        "funding_or_org_signals",
        "risks",
    ],
    "pest_analysis": [
        "growth_signals",
        "risks",
        "technology_or_product_features",
        "funding_or_org_signals",
    ],
    "competitor_canvas": [
        "positioning",
        "target_users",
        "core_scenarios",
        "core_features",
        "user_feedback",
        "differentiation",
    ],
    "feature_breakdown": [
        "core_features",
        "key_parameters",
        "recent_updates",
        "technology_or_product_features",
        "user_feedback",
    ],
    "needs_exploration": [
        "user_feedback",
        "core_scenarios",
        "core_features",
        "risks",
    ],
    "errac": [
        "core_features",
        "differentiation",
        "user_feedback",
        "pricing",
        "risks",
    ],
    "tracking_matrix": [
        "recent_updates",
        "growth_signals",
        "funding_or_org_signals",
        "technology_or_product_features",
        "risks",
    ],
    "matrix_analysis": [
        "positioning",
        "pricing",
        "growth_signals",
        "risks",
        "differentiation",
        "business_model",
    ],
    "comparison": [
        "positioning",
        "core_features",
        "pricing",
        "business_model",
        "growth_signals",
        "risks",
        "differentiation",
    ],
}


async def execute_tools(
    model,
    benchmark: AnalysisBenchmark,
    selected_set: SelectedCompetitorSet,
    tool_plan: ToolPlan,
    *,
    competitor_detail_cards_clean: list[CompetitorDetailCardClean] | None = None,
    industry_card_clean: IndustryCardClean | None = None,
    report_references: list[ReportReference] | None = None,
    concurrency: int = 3,
) -> list[ToolExecutionResult]:
    references = report_references or []
    jobs: list[tuple[ToolSpec, dict[str, Any]]] = []
    for spec in tool_specs_for_plan(tool_plan):
        for job in _tool_jobs(
            spec,
            selected_set.selected,
            competitor_detail_cards_clean or [],
            industry_card_clean,
            references,
        ):
            jobs.append((spec, job))

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def run_one(index: int, spec: ToolSpec, job: dict[str, Any]) -> tuple[int, ToolExecutionResult]:
        async with semaphore:
            result = await execute_tool(
                model,
                spec,
                benchmark,
                job["competitors"],
                scope_type=job["scope_type"],
                competitor_name=job.get("competitor_name"),
                input_reference_ids=job["input_reference_ids"],
                clean_context=job["clean_context"],
                available_references=references,
            )
            return index, result

    completed = await asyncio.gather(*(run_one(index, spec, job) for index, (spec, job) in enumerate(jobs)))
    return [result for _index, result in sorted(completed, key=lambda item: item[0])]


async def execute_tool(
    model,
    spec: ToolSpec,
    benchmark: AnalysisBenchmark,
    competitors: list[CandidateCompetitor],
    *,
    scope_type: str = "comparison",
    competitor_name: str | None = None,
    input_reference_ids: list[str] | None = None,
    clean_context: dict[str, Any] | None = None,
    available_references: list[ReportReference] | None = None,
) -> ToolExecutionResult:
    input_reference_ids = input_reference_ids or []
    clean_context = clean_context or {}
    fallback = build_fallback_tool_result(
        spec,
        benchmark,
        competitors,
        scope_type=scope_type,
        competitor_name=competitor_name,
        input_reference_ids=input_reference_ids,
        clean_context=clean_context,
    )
    if not model:
        return fallback
    try:
        payload = ToolExecutionInput(
            tool_name=spec.name,
            benchmark=benchmark,
            competitors=competitors,
            context={
                "input_reference_ids": input_reference_ids,
                "clean_cards": clean_context,
                "available_references": [
                    {
                        "id": ref.id,
                        "title": ref.title,
                        "type": ref.reference_type,
                        "summary": _strip_prompt_links(ref.summary[:800]),
                    }
                    for ref in (available_references or [])
                    if ref.id in set(input_reference_ids)
                ],
            },
        )
        prompt_payload = json.dumps(_tool_execution_prompt_payload(payload), ensure_ascii=False, default=str)
        tool_spec_payload = json.dumps(_tool_spec_prompt_payload(spec), ensure_ascii=False, default=str)
        messages = [
            ("system", tool_executor_system_prompt.format(tool_prompt=_reference_only_tool_prompt(spec))),
            (
                "user",
                tool_executor_user_prompt.format(
                    tool_spec=tool_spec_payload,
                    tool_execution_input=prompt_payload,
                ),
            ),
        ]
        result = await retry_structured_ainvoke(model, ToolExecutionResult, messages)
        if not isinstance(result, ToolExecutionResult):
            raise TypeError(f"Tool executor expected ToolExecutionResult, got {type(result).__name__}")
        return _sanitize_tool_result(result, spec, fallback, input_reference_ids=input_reference_ids)
    except Exception:
        return fallback


def build_fallback_tool_result(
    spec: ToolSpec,
    benchmark: AnalysisBenchmark,
    competitors: list[CandidateCompetitor],
    *,
    scope_type: str = "comparison",
    competitor_name: str | None = None,
    input_reference_ids: list[str] | None = None,
    clean_context: dict[str, Any] | None = None,
) -> ToolExecutionResult:
    input_reference_ids = input_reference_ids or []
    clean_context = clean_context or {}
    claims = _claims_from_references(spec, benchmark, input_reference_ids, clean_context)
    data = _data_for_spec(spec, competitors, claims)
    return ToolExecutionResult(
        tool_name=spec.name,
        canonical_tool_name=spec.canonical_name,
        schema_name=spec.output_schema_name,
        scope_type=scope_type,  # type: ignore[arg-type]
        competitor_name=competitor_name,
        input_reference_ids=input_reference_ids,
        claims=claims,
        data=data,
        evidence_ids=[],
        confidence=0.42 if input_reference_ids else 0.2,
        reasoning="未配置大模型或工具执行失败，已基于资料卡字段引用生成低置信度兜底结果。",
        generated_by="fallback",
    )


def tool_results_to_sandboxes(results: list[ToolExecutionResult]) -> list[ToolSandbox]:
    return [
        ToolSandbox(
            tool=result.tool_name,
            schema_name=result.schema_name,
            scope_type=result.scope_type,
            competitor_name=result.competitor_name,
            input_reference_ids=result.input_reference_ids,
            data=result.data,
            evidence_ids=result.evidence_ids,
            claims=result.claims,
            result=result,
        )
        for result in results
    ]


def _sanitize_tool_result(
    result: ToolExecutionResult,
    spec: ToolSpec,
    fallback: ToolExecutionResult,
    *,
    input_reference_ids: list[str] | None = None,
) -> ToolExecutionResult:
    input_reference_ids = input_reference_ids or []
    valid_reference_ids = set(input_reference_ids)
    cleaned_claims: list[ToolClaim] = []
    for claim in result.claims:
        refs = [item for item in claim.reference_ids if item in valid_reference_ids]
        if refs:
            cleaned_claims.append(claim.model_copy(update={"reference_ids": refs, "evidence_ids": []}))
    if not cleaned_claims:
        return fallback
    return result.model_copy(
        update={
            "tool_name": result.tool_name or spec.name,
            "canonical_tool_name": result.canonical_tool_name or spec.canonical_name,
            "schema_name": result.schema_name or spec.output_schema_name,
            "scope_type": result.scope_type or fallback.scope_type,
            "competitor_name": result.competitor_name or fallback.competitor_name,
            "input_reference_ids": input_reference_ids,
            "claims": cleaned_claims,
            "evidence_ids": [],
            "generated_by": "llm",
        }
    )


def _tool_jobs(
    spec: ToolSpec,
    competitors: list[CandidateCompetitor],
    competitor_cards: list[CompetitorDetailCardClean],
    industry_card: IndustryCardClean | None,
    references: list[ReportReference],
) -> list[dict[str, Any]]:
    canonical = spec.canonical_name
    single_competitor_tools = {
        "lean_canvas",
        "competitor_canvas",
        "swot",
        "feature_breakdown",
        "needs_exploration",
        "errac",
    }
    industry_tools = {"pest_analysis", "porter_five_forces"}
    comparison_tools = {"strategy_canvas", "matrix_analysis", "comparison", "tracking_matrix"}
    card_by_name = {_name_key(card.competitor): card for card in competitor_cards}
    if canonical in single_competitor_tools:
        jobs: list[dict[str, Any]] = []
        fields = _fields_for_tool(spec)
        for competitor in competitors:
            card = card_by_name.get(_name_key(competitor.name))
            clean_card = _trim_competitor_card(card, fields) if card else None
            reference_ids = _reference_ids_from_clean_card(clean_card)
            jobs.append(
                {
                    "scope_type": "competitor",
                    "competitor_name": competitor.name,
                    "competitors": [competitor],
                    "input_reference_ids": reference_ids,
                    "clean_context": {
                        "competitor_detail_card": clean_card,
                    },
                }
            )
        return jobs
    if canonical in industry_tools:
        competitor_fields = _fields_for_tool(spec)
        clean_industry_card = _trim_industry_card(industry_card)
        competitor_summaries = [
            _competitor_summary(card, competitor_fields)
            for card in competitor_cards
        ]
        reference_ids = _reference_ids_from_clean_card(clean_industry_card)
        for card in competitor_cards:
            reference_ids.extend(_reference_ids_from_clean_card(_trim_competitor_card(card, competitor_fields)))
        return [
            {
                "scope_type": "industry",
                "competitor_name": None,
                "competitors": competitors,
                "input_reference_ids": _dedupe_strings(reference_ids),
                "clean_context": {
                    "industry_card": clean_industry_card,
                    "competitor_summaries": competitor_summaries,
                },
            }
        ]
    fields = _fields_for_tool(spec)
    clean_cards = [_trim_competitor_card(card, fields) for card in competitor_cards]
    reference_ids = []
    for card in clean_cards:
        reference_ids.extend(_reference_ids_from_clean_card(card))
    if canonical not in comparison_tools:
        reference_ids.extend(ref.id for ref in references if ref.reference_type == "clean_card")
    return [
        {
            "scope_type": "comparison",
            "competitor_name": None,
            "competitors": competitors,
            "input_reference_ids": _dedupe_strings(list(reference_ids)),
            "clean_context": {
                "competitor_detail_cards": clean_cards,
            },
        }
    ]


def _fields_for_tool(spec: ToolSpec, *, industry: bool = False) -> list[str]:
    if industry:
        return []
    return COMPETITOR_CARD_FIELDS_BY_TOOL.get(spec.canonical_name, DEFAULT_COMPETITOR_CARD_FIELDS)


def _trim_competitor_card(card: CompetitorDetailCardClean, fields: list[str]) -> dict[str, Any]:
    preferred = set(fields)
    return {
        "competitor": card.competitor,
        "competitor_type": card.competitor_type,
        "fields": {key: value for key, value in card.fields.items() if key in preferred},
        "reference_ids": {key: value for key, value in card.reference_ids.items() if key in preferred},
    }


def _trim_industry_card(card: IndustryCardClean | None) -> dict[str, Any] | None:
    if card is None:
        return None
    return card.model_dump(mode="json")


def _reference_ids_from_clean_card(card_dict: dict[str, Any] | None) -> list[str]:
    if not isinstance(card_dict, dict):
        return []
    refs = card_dict.get("reference_ids")
    if not isinstance(refs, dict):
        return []
    return _dedupe_strings(refs.values())


def _competitor_summary(card: CompetitorDetailCardClean, fields: list[str] | None = None) -> dict[str, Any]:
    preferred = fields or DEFAULT_COMPETITOR_CARD_FIELDS
    return {
        "competitor": card.competitor,
        "competitor_type": card.competitor_type,
        "fields": {key: card.fields[key] for key in preferred if key in card.fields},
        "reference_ids": {key: card.reference_ids[key] for key in preferred if key in card.reference_ids},
    }


def _claims_from_references(
    spec: ToolSpec,
    benchmark: AnalysisBenchmark,
    reference_ids: list[str],
    clean_context: dict[str, Any],
) -> list[ToolClaim]:
    claims: list[ToolClaim] = []
    allowed_reference_ids = set(reference_ids)
    summaries = _reference_summary_items(clean_context)
    for index, item in enumerate(summaries[:6], start=1):
        ref_id = item.get("reference_id")
        text = item.get("text") or "资料卡字段可用，但内容较少。"
        if not ref_id:
            continue
        if allowed_reference_ids and ref_id not in allowed_reference_ids:
            continue
        claims.append(
            ToolClaim(
                category=_category_for_spec(spec, index),
                title=f"{spec.name} 资料卡观察 {index}",
                claim=f"资料卡显示，{benchmark.target_product} 的 {spec.canonical_name} 分析可参考该信号：{str(text)[:260]}",
                reference_ids=[str(ref_id)],
                confidence=0.42,
                reasoning="兜底结论引用了传入工具的清洗资料卡字段。",
            )
        )
    if claims:
        return claims
    return [
        ToolClaim(
            category="data_gap",
            title=f"{spec.name} 缺少可用资料卡引用",
            claim="本工具没有拿到可用的资料卡字段引用，因此只能标记为数据缺口。",
            reference_ids=reference_ids[:1],
            confidence=0.12,
            reasoning="需要重新执行资料卡采集，补齐可引用字段。",
        )
    ]


def _reference_summary_items(clean_context: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    def add_card(card: dict[str, Any] | None) -> None:
        if not isinstance(card, dict):
            return
        fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
        refs = card.get("reference_ids") if isinstance(card.get("reference_ids"), dict) else {}
        for field_name, text in fields.items():
            ref_id = refs.get(field_name)
            if ref_id and str(text).strip():
                items.append({"reference_id": str(ref_id), "text": str(text)})

    add_card(clean_context.get("competitor_detail_card"))
    add_card(clean_context.get("industry_card"))
    for card in clean_context.get("competitor_detail_cards") or []:
        add_card(card)
    for card in clean_context.get("competitor_summaries") or []:
        add_card(card)
    return items


def _data_for_spec(
    spec: ToolSpec,
    competitors: list[CandidateCompetitor],
    claims: list[ToolClaim],
) -> dict:
    canonical = get_tool_spec(spec.name).canonical_name
    reference_ids = _dedupe_strings(ref for claim in claims for ref in claim.reference_ids)
    if canonical == "swot":
        buckets: dict[str, list[dict]] = {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}
        mapping = ["strengths", "weaknesses", "opportunities", "threats"]
        for index, claim in enumerate(claims):
            buckets[mapping[index % 4]].append(claim.model_dump(mode="json"))
        return buckets
    if canonical == "feature_breakdown":
        return {
            "competitors": {
                competitor.name: {
                    "level_1": [claim.claim for claim in claims[:2]],
                    "level_2": [],
                    "level_3": [],
                    "reference_ids": reference_ids,
                }
                for competitor in competitors
            }
        }
    if canonical == "tracking_matrix":
        return {"timeline": {}, "reference_ids": reference_ids}
    return {
        "competitors": [item.name for item in competitors],
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "reference_ids": reference_ids,
        "evidence_ids": [],
    }


def _category_for_spec(spec: ToolSpec, index: int) -> str:
    if spec.canonical_name == "swot":
        return ["strength", "weakness", "opportunity", "threat"][(index - 1) % 4]
    if spec.canonical_name == "feature_breakdown":
        return "feature"
    if spec.canonical_name == "tracking_matrix":
        return "market_signal"
    return "observation"


def _tool_execution_prompt_payload(payload: ToolExecutionInput) -> dict[str, Any]:
    return {
        "tool_name": payload.tool_name,
        "benchmark": payload.benchmark.model_dump(mode="json"),
        "competitors": [
            {
                "name": competitor.name,
                "description": _strip_prompt_links(competitor.description),
                "competitor_type": competitor.competitor_type,
                "tags": competitor.tags,
                "score": competitor.score,
                "selection_reason": _strip_prompt_links(competitor.selection_reason),
            }
            for competitor in payload.competitors
        ],
        "context": _strip_prompt_links_recursive(payload.context),
    }


def _tool_spec_prompt_payload(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "canonical_name": spec.canonical_name,
        "description": _strip_prompt_links(spec.description),
        "output_schema_name": spec.output_schema_name,
    }


def _reference_only_tool_prompt(spec: ToolSpec) -> str:
    text = _strip_prompt_links(spec.prompt)
    replacements = {
        "evidence_ids": "reference_ids",
        "Evidence IDs": "reference_ids",
        "evidence IDs": "reference_ids",
        "evidence id": "reference_id",
        "Evidence": "资料卡引用",
        "evidence": "资料卡引用",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _strip_prompt_links_recursive(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_prompt_links(value)
    if isinstance(value, list):
        return [_strip_prompt_links_recursive(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_prompt_links_recursive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _strip_prompt_links_recursive(item) for key, item in value.items()}
    return value


def _strip_prompt_links(value: str) -> str:
    text = re.sub(r"\b[a-z][a-z0-9+.-]*://\S+|www\.\S+", "[link_removed]", str(value or ""))
    return re.sub(r"\b(?:search_runs|task_runs|screenshots|extractions)/\S+", "[path_removed]", text)


def _name_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _dedupe_strings(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
