from __future__ import annotations

import json
import re
import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from myagent.backend.app.agents.chunk_filter import filter_relevant_sources
from myagent.backend.app.agents.prompt_bridge import (
    card_gap_query_system_prompt,
    card_qa_system_prompt,
    competitor_card_patch_system_prompt,
    detail_card_query_system_prompt,
    industry_card_patch_system_prompt,
)
from myagent.backend.app.config import Settings
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CardQAResult,
    CompetitorDetailCard,
    CompetitorDetailCardClean,
    DetailQueryPlan,
    IndustryCard,
    IndustryCardClean,
    IndustryQuery,
    QueryTemplate,
    RawSource,
    ReportReference,
    ResearchField,
    SearchPlan,
    SearchQuery,
    SelectedCompetitorSet,
    SourceProviderResult,
    SourceRef,
)
from myagent.backend.app.providers import dedupe_sources


StructuredCall = Callable[[type[BaseModel], list[tuple[str, str]]], Awaitable[BaseModel]]
SearchCollect = Callable[..., Awaitable[SourceProviderResult]]
CardProgressCallback = Callable[["CardCollectionResult"], Awaitable[None]]


COMPETITOR_CARD_FIELDS = [
    "name",
    "category",
    "website",
    "positioning",
    "target_users",
    "core_scenarios",
    "core_features",
    "key_parameters",
    "pricing",
    "business_model",
    "channels",
    "growth_signals",
    "funding_or_org_signals",
    "technology_or_product_features",
    "differentiation",
    "user_feedback",
    "risks",
    "recent_updates",
    "key_evidence",
]

INDUSTRY_CARD_FIELDS = [
    "basic_info.industry_name",
    "basic_info.industry_scope",
    "basic_info.main_users",
    "basic_info.core_scenarios",
    "basic_info.buyer_behavior_or_preferences",
    "market_situation.market_size",
    "market_situation.growth_trend",
    "market_situation.willingness_to_pay",
    "market_situation.cost_pressure",
    "market_situation.investment_environment",
    "competition_situation.competition_intensity",
    "competition_situation.entry_barriers",
    "competition_situation.substitutes",
    "competition_situation.user_switching_cost",
    "external_environment.regulation_or_laws",
    "external_environment.technology_trends",
    "external_environment.key_resource_dependencies",
]

COMPETITOR_HARD_REQUIRED_FIELDS = [
    "positioning",
    "target_users",
    "core_features",
    "pricing",
    "business_model",
    "growth_signals",
    "user_feedback",
    "risks",
]

INDUSTRY_HARD_REQUIRED_FIELDS = [
    "basic_info.industry_scope",
    "basic_info.main_users",
    "market_situation.market_size",
    "market_situation.growth_trend",
    "competition_situation.competition_intensity",
    "external_environment.regulation_or_laws",
    "external_environment.technology_trends",
]


class CardFieldPatch(BaseModel):
    value: str = Field(default="", max_length=4000)
    confidence: float = Field(default=0.0, ge=0, le=1)
    source_refs: list[SourceRef] = Field(default_factory=list, max_length=8)


class CardPatchOutput(BaseModel):
    fields: dict[str, CardFieldPatch] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list, max_length=32)


class CardGapQueryOutput(BaseModel):
    queries: list[SearchQuery] = Field(default_factory=list, max_length=3)


class CardCollectionResult(BaseModel):
    sources: list[RawSource] = Field(default_factory=list)
    provider_results: list[SourceProviderResult] = Field(default_factory=list)
    competitor_detail_cards: list[CompetitorDetailCard] = Field(default_factory=list)
    industry_card: IndustryCard | None = None
    competitor_detail_cards_clean: list[CompetitorDetailCardClean] = Field(default_factory=list)
    industry_card_clean: IndustryCardClean | None = None
    report_references: list[ReportReference] = Field(default_factory=list)
    risk_notices: list[str] = Field(default_factory=list)
    detail_seen_url_keys: list[str] = Field(default_factory=list)
    industry_seen_url_keys: list[str] = Field(default_factory=list)
    object_progress: list[dict[str, Any]] = Field(default_factory=list)


async def collect_detail_cards(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    search_collect: SearchCollect,
    benchmark: AnalysisBenchmark,
    selected_set: SelectedCompetitorSet,
    request,
    settings: Settings,
    task_id: str,
    existing_sources: list[RawSource],
    detail_seen_url_keys: set[str] | None = None,
    industry_seen_url_keys: set[str] | None = None,
    should_cancel=None,
    progress_callback: CardProgressCallback | None = None,
) -> CardCollectionResult:
    detail_seen = detail_seen_url_keys if detail_seen_url_keys is not None else set()
    industry_seen = industry_seen_url_keys if industry_seen_url_keys is not None else set()
    sources = dedupe_sources(existing_sources)
    provider_results: list[SourceProviderResult] = []
    risk_notices: list[str] = []
    max_retries = max(0, int(settings.card_search_iterations_per_object))
    object_progress: dict[str, dict[str, Any]] = {}
    progress_lock = asyncio.Lock()
    state_lock = asyncio.Lock()
    object_semaphore = asyncio.Semaphore(max(1, int(settings.card_object_concurrency)))
    competitor_cards: list[CompetitorDetailCard] = []
    industry_card: IndustryCard | None = None

    def set_object_progress(
        object_id: str,
        object_name: str,
        status: str,
        *,
        searched_sources: int = 0,
        analyzed_chunks: int = 0,
        merged_fields: int = 0,
        error: str | None = None,
    ) -> None:
        payload = {
            "stage": "phase_4_collect",
            "object_id": object_id,
            "object_name": object_name,
            "status": status,
            "searched_sources": searched_sources,
            "analyzed_chunks": analyzed_chunks,
            "merged_fields": merged_fields,
        }
        if error:
            payload["error"] = error
        object_progress[object_id] = payload

    async def emit_progress(
        *,
        current_competitor_card: CompetitorDetailCard | None = None,
        current_industry_card: IndustryCard | None = None,
        object_sources: list[RawSource] | None = None,
        object_results: list[SourceProviderResult] | None = None,
        object_risks: list[str] | None = None,
    ) -> None:
        if progress_callback is None:
            return
        async with progress_lock:
            snapshot_sources = dedupe_sources([*sources, *(object_sources or [])])[:320]
            snapshot_results = [*provider_results, *(object_results or [])]
            snapshot_risks = _dedupe_strings([*risk_notices, *(object_risks or [])])
            snapshot_competitor_cards = list(competitor_cards)
            if current_competitor_card is not None and current_competitor_card not in snapshot_competitor_cards:
                snapshot_competitor_cards.append(current_competitor_card)
            snapshot_industry_card = current_industry_card or industry_card
            snapshot_progress = list(object_progress.values())
            competitor_clean = [_clean_competitor_card(card) for card in snapshot_competitor_cards]
            industry_clean = _clean_industry_card(snapshot_industry_card) if snapshot_industry_card is not None else None
            references = build_card_report_references(snapshot_competitor_cards, snapshot_industry_card)
        await progress_callback(
            CardCollectionResult(
                sources=snapshot_sources,
                provider_results=snapshot_results,
                competitor_detail_cards=snapshot_competitor_cards,
                industry_card=snapshot_industry_card,
                competitor_detail_cards_clean=competitor_clean,
                industry_card_clean=industry_clean,
                report_references=references,
                risk_notices=snapshot_risks,
                detail_seen_url_keys=sorted(detail_seen),
                industry_seen_url_keys=sorted(industry_seen),
                object_progress=snapshot_progress,
            )
        )

    query_plan = await _generate_query_plan(
        model_available=model_available,
        structured_call=structured_call,
        benchmark=benchmark,
        competitors=selected_set.selected,
        request=request,
    )

    async def collect_one_competitor(index: int, competitor: CandidateCompetitor):
        object_id = f"competitor:{_slug(competitor.name)}"
        async with object_semaphore:
            if should_cancel:
                should_cancel()
            set_object_progress(object_id, competitor.name, "searching")
            await emit_progress()
            try:
                card, object_sources, object_results, risks = await _collect_competitor_card(
                    model_available=model_available,
                    structured_call=structured_call,
                    search_collect=search_collect,
                    benchmark=benchmark,
                    competitor=competitor,
                    request=request,
                    settings=settings,
                    task_id=task_id,
                    templates=query_plan.competitor_query_templates,
                    seen_url_keys=detail_seen,
                    max_retries=max_retries,
                    should_cancel=should_cancel,
                    progress_callback=lambda current_card, object_sources, object_results, object_risks, status="analyzing", analyzed_chunks=0, merged_fields=0: (
                        set_object_progress(
                            object_id,
                            competitor.name,
                            status,
                            searched_sources=len(object_sources),
                            analyzed_chunks=analyzed_chunks,
                            merged_fields=merged_fields,
                        ),
                        emit_progress(
                            current_competitor_card=current_card,
                            object_sources=object_sources,
                            object_results=object_results,
                            object_risks=object_risks,
                        ),
                    )[1],
                )
                set_object_progress(
                    object_id,
                    competitor.name,
                    "done",
                    searched_sources=len(object_sources),
                    analyzed_chunks=_count_card_source_refs(card),
                    merged_fields=_count_competitor_fields(card),
                )
                async with state_lock:
                    sources[:] = dedupe_sources([*sources, *object_sources])[:260]
                    provider_results.extend(object_results)
                    risk_notices.extend(risks)
                return index, card, object_sources, object_results, risks
            except Exception as exc:
                if _is_task_interrupt(exc):
                    raise
                set_object_progress(object_id, competitor.name, "failed", error=str(exc))
                await emit_progress(object_risks=[str(exc)])
                return index, CompetitorDetailCard(competitor=competitor.name, competitor_type=competitor.competitor_type), [], [], [str(exc)]

    async def collect_industry():
        object_name = request.product_category or request.market or benchmark.target_product
        object_id = f"industry:{_slug(object_name)}"
        async with object_semaphore:
            set_object_progress(object_id, object_name, "searching")
            await emit_progress()
            try:
                card, object_sources, object_results, risks = await _collect_industry_card(
                    model_available=model_available,
                    structured_call=structured_call,
                    search_collect=search_collect,
                    benchmark=benchmark,
                    competitors=selected_set.selected,
                    request=request,
                    settings=settings,
                    task_id=task_id,
                    queries=query_plan.industry_queries,
                    seen_url_keys=industry_seen,
                    max_retries=max_retries,
                    should_cancel=should_cancel,
                    progress_callback=lambda current_card, object_sources, object_results, object_risks, status="analyzing", analyzed_chunks=0, merged_fields=0: (
                        set_object_progress(
                            object_id,
                            object_name,
                            status,
                            searched_sources=len(object_sources),
                            analyzed_chunks=analyzed_chunks,
                            merged_fields=merged_fields,
                        ),
                        emit_progress(
                            current_industry_card=current_card,
                            object_sources=object_sources,
                            object_results=object_results,
                            object_risks=object_risks,
                        ),
                    )[1],
                )
                set_object_progress(
                    object_id,
                    object_name,
                    "done",
                    searched_sources=len(object_sources),
                    analyzed_chunks=_count_card_source_refs(card),
                    merged_fields=_count_industry_fields(card),
                )
                async with state_lock:
                    sources[:] = dedupe_sources([*sources, *object_sources])[:320]
                    provider_results.extend(object_results)
                    risk_notices.extend(risks)
                return card, object_sources, object_results, risks
            except Exception as exc:
                if _is_task_interrupt(exc):
                    raise
                card = IndustryCard()
                set_object_progress(object_id, object_name, "failed", error=str(exc))
                await emit_progress(current_industry_card=card, object_risks=[str(exc)])
                return card, [], [], [str(exc)]

    competitor_results, industry_result = await asyncio.gather(
        asyncio.gather(*(collect_one_competitor(index, competitor) for index, competitor in enumerate(selected_set.selected))),
        collect_industry(),
    )
    competitor_cards = [item[1] for item in sorted(competitor_results, key=lambda item: item[0])]
    industry_card, _industry_sources, _industry_results, _industry_risks = industry_result

    competitor_clean = [_clean_competitor_card(card) for card in competitor_cards]
    industry_clean = _clean_industry_card(industry_card)
    references = build_card_report_references(competitor_cards, industry_card)
    return CardCollectionResult(
        sources=sources,
        provider_results=provider_results,
        competitor_detail_cards=competitor_cards,
        industry_card=industry_card,
        competitor_detail_cards_clean=competitor_clean,
        industry_card_clean=industry_clean,
        report_references=references,
        risk_notices=_dedupe_strings(risk_notices),
        detail_seen_url_keys=sorted(detail_seen),
        industry_seen_url_keys=sorted(industry_seen),
        object_progress=list(object_progress.values()),
    )


async def _collect_competitor_card(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    search_collect: SearchCollect,
    benchmark: AnalysisBenchmark,
    competitor: CandidateCompetitor,
    request,
    settings: Settings,
    task_id: str,
    templates: list[QueryTemplate],
    seen_url_keys: set[str],
    max_retries: int,
    should_cancel,
    progress_callback: Callable[
        [CompetitorDetailCard, list[RawSource], list[SourceProviderResult], list[str]],
        Awaitable[None],
    ]
    | None = None,
) -> tuple[CompetitorDetailCard, list[RawSource], list[SourceProviderResult], list[str]]:
    card = CompetitorDetailCard(competitor=competitor.name, competitor_type=competitor.competitor_type)
    all_sources: list[RawSource] = []
    provider_results: list[SourceProviderResult] = []
    risk_notices: list[str] = []

    initial_queries = _competitor_queries(competitor, templates, COMPETITOR_CARD_FIELDS, 0)
    sources, results = await _search_and_patch_competitor(
        model_available=model_available,
        structured_call=structured_call,
        search_collect=search_collect,
        card=card,
        competitor=competitor,
        queries=initial_queries,
        request=request,
        settings=settings,
        task_id=task_id,
        benchmark=benchmark,
        seen_url_keys=seen_url_keys,
        should_cancel=should_cancel,
        progress_callback=lambda new_sources, new_results, **progress: progress_callback(
            card,
            dedupe_sources([*all_sources, *new_sources])[:100],
            [*provider_results, *new_results],
            risk_notices,
            **progress,
        )
        if progress_callback
        else None,
    )
    all_sources = dedupe_sources([*all_sources, *sources])[:100]
    provider_results.extend(results)

    qa = await _qa_card(model_available, structured_call, "competitor", competitor.name, card)
    retries = 0
    while _qa_needs_retry(qa) and retries < max_retries:
        retries += 1
        failed_fields = _qa_failed_fields(qa) or _missing_competitor_fields(card)
        if not failed_fields:
            break
        _clear_error_fields(card, _hard_error_fields(qa, COMPETITOR_HARD_REQUIRED_FIELDS), allowed=COMPETITOR_CARD_FIELDS)
        gap_queries = await _generate_gap_queries(
            model_available=model_available,
            structured_call=structured_call,
            benchmark=benchmark,
            object_type="competitor",
            object_name=competitor.name,
            missing_fields=failed_fields,
            fallback_queries=_competitor_gap_queries(competitor, failed_fields, retries),
        )
        if not gap_queries:
            break
        sources, results = await _search_and_patch_competitor(
            model_available=model_available,
            structured_call=structured_call,
            search_collect=search_collect,
            card=card,
            competitor=competitor,
            queries=gap_queries,
            request=request,
            settings=settings,
            task_id=task_id,
            benchmark=benchmark,
            seen_url_keys=seen_url_keys,
            should_cancel=should_cancel,
            focus_fields=failed_fields,
            progress_callback=lambda new_sources, new_results, **progress: progress_callback(
                card,
                dedupe_sources([*all_sources, *new_sources])[:100],
                [*provider_results, *new_results],
                risk_notices,
            )
            if progress_callback
            else None,
        )
        all_sources = dedupe_sources([*all_sources, *sources])[:100]
        provider_results.extend(results)
        qa = await _qa_card(model_available, structured_call, "competitor", competitor.name, card)

    if _qa_needs_retry(qa):
        risk_notices.append(
            f"竞品资料卡 {competitor.name} 在 {max_retries} 轮补搜后仍有缺口："
            + "、".join(_dedupe_strings([*_qa_failed_fields(qa), *_missing_competitor_fields(card)])[:12])
        )
    return card, all_sources, provider_results, risk_notices


async def _collect_industry_card(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    search_collect: SearchCollect,
    benchmark: AnalysisBenchmark,
    competitors: list[CandidateCompetitor],
    request,
    settings: Settings,
    task_id: str,
    queries: list[IndustryQuery],
    seen_url_keys: set[str],
    max_retries: int,
    should_cancel,
    progress_callback: Callable[
        [IndustryCard, list[RawSource], list[SourceProviderResult], list[str]],
        Awaitable[None],
    ]
    | None = None,
) -> tuple[IndustryCard, list[RawSource], list[SourceProviderResult], list[str]]:
    industry_name = request.product_category or request.market or benchmark.target_product
    card = IndustryCard()
    all_sources: list[RawSource] = []
    provider_results: list[SourceProviderResult] = []
    risk_notices: list[str] = []

    initial_queries = _industry_queries(industry_name, queries, INDUSTRY_CARD_FIELDS, 0)
    sources, results = await _search_and_patch_industry(
        model_available=model_available,
        structured_call=structured_call,
        search_collect=search_collect,
        card=card,
        industry_name=industry_name,
        queries=initial_queries,
        request=request,
        settings=settings,
        task_id=task_id,
        benchmark=benchmark,
        competitors=competitors,
        seen_url_keys=seen_url_keys,
        should_cancel=should_cancel,
        progress_callback=lambda new_sources, new_results, **progress: progress_callback(
            card,
            dedupe_sources([*all_sources, *new_sources])[:120],
            [*provider_results, *new_results],
            risk_notices,
            **progress,
        )
        if progress_callback
        else None,
    )
    all_sources = dedupe_sources([*all_sources, *sources])[:120]
    provider_results.extend(results)

    qa = await _qa_card(model_available, structured_call, "industry", industry_name, card)
    retries = 0
    while _qa_needs_retry(qa) and retries < max_retries:
        retries += 1
        failed_fields = _qa_failed_fields(qa) or _missing_industry_fields(card)
        if not failed_fields:
            break
        _clear_error_fields(card, _hard_error_fields(qa, INDUSTRY_HARD_REQUIRED_FIELDS), allowed=INDUSTRY_CARD_FIELDS)
        gap_queries = await _generate_gap_queries(
            model_available=model_available,
            structured_call=structured_call,
            benchmark=benchmark,
            object_type="industry",
            object_name=industry_name,
            missing_fields=failed_fields,
            fallback_queries=_industry_gap_queries(industry_name, failed_fields, retries),
        )
        if not gap_queries:
            break
        sources, results = await _search_and_patch_industry(
            model_available=model_available,
            structured_call=structured_call,
            search_collect=search_collect,
            card=card,
            industry_name=industry_name,
            queries=gap_queries,
            request=request,
            settings=settings,
            task_id=task_id,
            benchmark=benchmark,
            competitors=competitors,
            seen_url_keys=seen_url_keys,
            should_cancel=should_cancel,
            focus_fields=failed_fields,
            progress_callback=lambda new_sources, new_results, **progress: progress_callback(
                card,
                dedupe_sources([*all_sources, *new_sources])[:120],
                [*provider_results, *new_results],
                risk_notices,
            )
            if progress_callback
            else None,
        )
        all_sources = dedupe_sources([*all_sources, *sources])[:120]
        provider_results.extend(results)
        qa = await _qa_card(model_available, structured_call, "industry", industry_name, card)

    if _qa_needs_retry(qa):
        risk_notices.append(
            f"行业资料卡 {industry_name} 在 {max_retries} 轮补搜后仍有缺口："
            + "、".join(_dedupe_strings([*_qa_failed_fields(qa), *_missing_industry_fields(card)])[:12])
        )
    return card, all_sources, provider_results, risk_notices


async def _search_and_patch_competitor(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    search_collect: SearchCollect,
    card: CompetitorDetailCard,
    competitor: CandidateCompetitor,
    queries: list[SearchQuery],
    request,
    settings: Settings,
    task_id: str,
    benchmark: AnalysisBenchmark,
    seen_url_keys: set[str],
    should_cancel,
    focus_fields: list[str] | None = None,
    progress_callback: Callable[[list[RawSource], list[SourceProviderResult]], Awaitable[None]] | None = None,
) -> tuple[list[RawSource], list[SourceProviderResult]]:
    all_sources: list[RawSource] = []
    provider_results: list[SourceProviderResult] = []
    for query in queries:
        if should_cancel:
            should_cancel()
        result = await search_collect(
            SearchPlan(queries=[query], rationale="竞品资料卡单 query 采集。"),
            request,
            settings,
            task_id=task_id,
            benchmark=benchmark,
            competitors=[competitor],
            max_sources=12,
            seen_url_keys=seen_url_keys,
            should_cancel=should_cancel,
        )
        provider_results.append(result)
        query_sources = _sources_for_competitor_name(competitor.name, result.sources)
        query_sources, filter_stats = filter_relevant_sources(
            query_sources,
            object_name=competitor.name,
            context_terms=[benchmark.target_product, request.market or "", request.product_category or "", query.query],
            expected_fields=focus_fields or query.expected_fields or COMPETITOR_CARD_FIELDS,
            min_chars=80,
        )
        result.metadata["chunk_filter"] = filter_stats.to_dict()
        all_sources = dedupe_sources([*all_sources, *query_sources])
        await _patch_competitor_card(
            model_available=model_available,
            structured_call=structured_call,
            card=card,
            competitor=competitor,
            sources=query_sources,
            allow_fallback=not model_available,
            focus_fields=focus_fields or query.expected_fields or COMPETITOR_CARD_FIELDS,
        )
        if progress_callback is not None:
            await progress_callback(all_sources, provider_results)
    return all_sources, provider_results


async def _search_and_patch_industry(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    search_collect: SearchCollect,
    card: IndustryCard,
    industry_name: str,
    queries: list[SearchQuery],
    request,
    settings: Settings,
    task_id: str,
    benchmark: AnalysisBenchmark,
    competitors: list[CandidateCompetitor],
    seen_url_keys: set[str],
    should_cancel,
    focus_fields: list[str] | None = None,
    progress_callback: Callable[[list[RawSource], list[SourceProviderResult]], Awaitable[None]] | None = None,
) -> tuple[list[RawSource], list[SourceProviderResult]]:
    all_sources: list[RawSource] = []
    provider_results: list[SourceProviderResult] = []
    for query in queries:
        if should_cancel:
            should_cancel()
        result = await search_collect(
            SearchPlan(queries=[query], rationale="行业资料卡单 query 采集。"),
            request,
            settings,
            task_id=task_id,
            benchmark=benchmark,
            competitors=competitors,
            max_sources=12,
            seen_url_keys=seen_url_keys,
            should_cancel=should_cancel,
        )
        provider_results.append(result)
        query_sources, filter_stats = filter_relevant_sources(
            result.sources,
            object_name=industry_name,
            context_terms=[benchmark.target_product, request.market or "", request.product_category or "", query.query],
            expected_fields=focus_fields or query.expected_fields or INDUSTRY_CARD_FIELDS,
            min_chars=80,
        )
        result.metadata["chunk_filter"] = filter_stats.to_dict()
        all_sources = dedupe_sources([*all_sources, *query_sources])
        await _patch_industry_card(
            model_available=model_available,
            structured_call=structured_call,
            card=card,
            industry_name=industry_name,
            sources=query_sources,
            allow_fallback=not model_available,
            focus_fields=focus_fields or query.expected_fields or INDUSTRY_CARD_FIELDS,
        )
        if progress_callback is not None:
            await progress_callback(all_sources, provider_results)
    return all_sources, provider_results


async def _call_search_collect(
    search_collect: SearchCollect,
    search_plan: SearchPlan,
    request,
    settings: Settings,
    **kwargs,
) -> SourceProviderResult:
    signature = inspect.signature(search_collect)
    supports_on_sources = "on_sources" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if not supports_on_sources:
        kwargs.pop("on_sources", None)
    return await search_collect(search_plan, request, settings, **kwargs)


async def _build_competitor_patch(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    competitor: CandidateCompetitor,
    source: RawSource,
    fields_to_fill: list[str],
    current_card_values: dict[str, str],
    semaphore: asyncio.Semaphore,
    index: int,
) -> tuple[int, CardPatchOutput | None, bool]:
    if not model_available or structured_call is None or not (source.content or "").strip():
        return index, None, True
    async with semaphore:
        try:
            output = await structured_call(
                CardPatchOutput,
                [
                    ("system", competitor_card_patch_system_prompt),
                    (
                        "user",
                        json.dumps(
                            {
                                "competitor": competitor.name,
                                "fields_to_fill": fields_to_fill,
                                "current_card_values": current_card_values,
                                "chunk": _source_prompt_payload(source),
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
            )
            assert isinstance(output, CardPatchOutput)
            _normalize_patch_refs(output.fields, source)
            return index, output, False
        except Exception as exc:
            if _is_task_interrupt(exc):
                raise
            return index, None, True


async def _build_industry_patch(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    industry_name: str,
    source: RawSource,
    fields_to_fill: list[str],
    current_card_values: dict[str, str],
    semaphore: asyncio.Semaphore,
    index: int,
) -> tuple[int, CardPatchOutput | None, bool]:
    if not model_available or structured_call is None or not (source.content or "").strip():
        return index, None, True
    async with semaphore:
        try:
            output = await structured_call(
                CardPatchOutput,
                [
                    ("system", industry_card_patch_system_prompt),
                    (
                        "user",
                        json.dumps(
                            {
                                "industry": industry_name,
                                "fields_to_fill": fields_to_fill,
                                "current_card_values": current_card_values,
                                "chunk": _source_prompt_payload(source),
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
            )
            assert isinstance(output, CardPatchOutput)
            _normalize_patch_refs(output.fields, source)
            return index, output, False
        except Exception as exc:
            if _is_task_interrupt(exc):
                raise
            return index, None, True


async def _search_and_patch_competitor(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    search_collect: SearchCollect,
    card: CompetitorDetailCard,
    competitor: CandidateCompetitor,
    queries: list[SearchQuery],
    request,
    settings: Settings,
    task_id: str,
    benchmark: AnalysisBenchmark,
    seen_url_keys: set[str],
    should_cancel,
    focus_fields: list[str] | None = None,
    progress_callback: Callable[..., Awaitable[None]] | None = None,
) -> tuple[list[RawSource], list[SourceProviderResult]]:
    all_sources: list[RawSource] = []
    provider_results: list[SourceProviderResult] = []
    patch_jobs: list[asyncio.Task[tuple[int, CardPatchOutput | None, bool]]] = []
    patch_semaphore = asyncio.Semaphore(max(1, settings.effective_card_patch_concurrency))

    async def schedule_sources(raw_sources: list[RawSource]) -> None:
        nonlocal all_sources
        query_text = " ".join(query.query for query in queries)
        expected_fields = focus_fields or _dedupe_strings(field for query in queries for field in (query.expected_fields or [])) or COMPETITOR_CARD_FIELDS
        query_sources = _sources_for_competitor_name(competitor.name, raw_sources)
        query_sources, _filter_stats = filter_relevant_sources(
            query_sources,
            object_name=competitor.name,
            context_terms=[benchmark.target_product, request.market or "", request.product_category or "", query_text],
            expected_fields=expected_fields,
            min_chars=80,
        )
        start_index = len(all_sources)
        all_sources = dedupe_sources([*all_sources, *query_sources])
        for offset, source in enumerate(query_sources):
            patch_jobs.append(
                asyncio.create_task(
                    _build_competitor_patch(
                        model_available=model_available,
                        structured_call=structured_call,
                        competitor=competitor,
                        source=source,
                        fields_to_fill=expected_fields,
                        current_card_values=_competitor_value_snapshot(card),
                        semaphore=patch_semaphore,
                        index=start_index + offset,
                    )
                )
            )
        if progress_callback is not None:
            await progress_callback(all_sources, provider_results, status="analyzing" if patch_jobs else "searching")

    if should_cancel:
        should_cancel()
    result = await _call_search_collect(
        search_collect,
        SearchPlan(queries=queries, rationale="Competitor detail card parallel query collection."),
        request,
        settings,
        task_id=task_id,
        benchmark=benchmark,
        competitors=[competitor],
        max_sources=max(12, 12 * max(1, len(queries))),
        seen_url_keys=seen_url_keys,
        should_cancel=should_cancel,
        on_sources=schedule_sources,
    )
    provider_results.append(result)
    if not all_sources:
        await schedule_sources(result.sources)
    result.metadata["chunk_filter"] = {"kept_count": len(all_sources)}
    patch_results = await asyncio.gather(*patch_jobs) if patch_jobs else []
    fallback_needed = any(item[2] for item in patch_results)
    for _index, output, _fallback in sorted(patch_results, key=lambda item: item[0]):
        if output is not None:
            _merge_field_patches(card, output.fields, allowed=COMPETITOR_CARD_FIELDS)
    if fallback_needed or not model_available:
        _fallback_patch_competitor(card, competitor, all_sources)
    if progress_callback is not None:
        await progress_callback(
            all_sources,
            provider_results,
            status="merged",
            analyzed_chunks=len([item for item in patch_results if item[1] is not None]),
            merged_fields=_count_competitor_fields(card),
        )
    return all_sources, provider_results


async def _search_and_patch_industry(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    search_collect: SearchCollect,
    card: IndustryCard,
    industry_name: str,
    queries: list[SearchQuery],
    request,
    settings: Settings,
    task_id: str,
    benchmark: AnalysisBenchmark,
    competitors: list[CandidateCompetitor],
    seen_url_keys: set[str],
    should_cancel,
    focus_fields: list[str] | None = None,
    progress_callback: Callable[..., Awaitable[None]] | None = None,
) -> tuple[list[RawSource], list[SourceProviderResult]]:
    all_sources: list[RawSource] = []
    provider_results: list[SourceProviderResult] = []
    patch_jobs: list[asyncio.Task[tuple[int, CardPatchOutput | None, bool]]] = []
    patch_semaphore = asyncio.Semaphore(max(1, settings.effective_card_patch_concurrency))

    async def schedule_sources(raw_sources: list[RawSource]) -> None:
        nonlocal all_sources
        query_text = " ".join(query.query for query in queries)
        expected_fields = focus_fields or _dedupe_strings(field for query in queries for field in (query.expected_fields or [])) or INDUSTRY_CARD_FIELDS
        query_sources, _filter_stats = filter_relevant_sources(
            raw_sources,
            object_name=industry_name,
            context_terms=[benchmark.target_product, request.market or "", request.product_category or "", query_text],
            expected_fields=expected_fields,
            min_chars=80,
        )
        start_index = len(all_sources)
        all_sources = dedupe_sources([*all_sources, *query_sources])
        for offset, source in enumerate(query_sources):
            patch_jobs.append(
                asyncio.create_task(
                    _build_industry_patch(
                        model_available=model_available,
                        structured_call=structured_call,
                        industry_name=industry_name,
                        source=source,
                        fields_to_fill=expected_fields,
                        current_card_values=_industry_value_snapshot(card),
                        semaphore=patch_semaphore,
                        index=start_index + offset,
                    )
                )
            )
        if progress_callback is not None:
            await progress_callback(all_sources, provider_results, status="analyzing" if patch_jobs else "searching")

    if should_cancel:
        should_cancel()
    result = await _call_search_collect(
        search_collect,
        SearchPlan(queries=queries, rationale="Industry detail card parallel query collection."),
        request,
        settings,
        task_id=task_id,
        benchmark=benchmark,
        competitors=competitors,
        max_sources=max(12, 12 * max(1, len(queries))),
        seen_url_keys=seen_url_keys,
        should_cancel=should_cancel,
        on_sources=schedule_sources,
    )
    provider_results.append(result)
    if not all_sources:
        await schedule_sources(result.sources)
    result.metadata["chunk_filter"] = {"kept_count": len(all_sources)}
    patch_results = await asyncio.gather(*patch_jobs) if patch_jobs else []
    fallback_needed = any(item[2] for item in patch_results)
    for _index, output, _fallback in sorted(patch_results, key=lambda item: item[0]):
        if output is not None:
            _merge_field_patches(card, output.fields, allowed=INDUSTRY_CARD_FIELDS)
    if fallback_needed or not model_available:
        _fallback_patch_industry(card, industry_name, all_sources)
    if progress_callback is not None:
        await progress_callback(
            all_sources,
            provider_results,
            status="merged",
            analyzed_chunks=len([item for item in patch_results if item[1] is not None]),
            merged_fields=_count_industry_fields(card),
        )
    return all_sources, provider_results


async def _generate_query_plan(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    benchmark: AnalysisBenchmark,
    competitors: list[CandidateCompetitor],
    request,
) -> DetailQueryPlan:
    fallback = _fallback_query_plan(benchmark, request)
    if not model_available or structured_call is None:
        return fallback
    try:
        output = await structured_call(
            DetailQueryPlan,
            [
                ("system", detail_card_query_system_prompt),
                (
                    "user",
                    json.dumps(
                        {
                            "analysis_goal": benchmark.analysis_goal,
                            "market": request.market,
                            "product_category": request.product_category,
                            "competitors": [item.name for item in competitors],
                            "competitor_card_fields": COMPETITOR_CARD_FIELDS,
                            "industry_card_fields": INDUSTRY_CARD_FIELDS,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
        )
        assert isinstance(output, DetailQueryPlan)
        return _normalize_query_plan(output, fallback)
    except Exception as exc:
        if _is_task_interrupt(exc):
            raise
        return fallback


async def _patch_competitor_card(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    card: CompetitorDetailCard,
    competitor: CandidateCompetitor,
    sources: list[RawSource],
    allow_fallback: bool,
    focus_fields: list[str],
) -> None:
    usable_sources = [source for source in sources if (source.content or "").strip()]
    if not usable_sources:
        return
    if model_available and structured_call is not None:
        fields_to_fill = _dedupe_strings(focus_fields) or COMPETITOR_CARD_FIELDS
        for source in usable_sources:
            try:
                output = await structured_call(
                    CardPatchOutput,
                    [
                        ("system", competitor_card_patch_system_prompt),
                        (
                            "user",
                            json.dumps(
                                {
                                    "competitor": competitor.name,
                                    "fields_to_fill": fields_to_fill,
                                    "current_card_values": _competitor_value_snapshot(card),
                                    "chunk": _source_prompt_payload(source),
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    ],
                )
                assert isinstance(output, CardPatchOutput)
                _normalize_patch_refs(output.fields, source)
                _merge_field_patches(card, output.fields, allowed=COMPETITOR_CARD_FIELDS)
                fields_to_fill = _dedupe_strings(output.missing_fields) or _empty_competitor_fields(card)
            except Exception as exc:
                if _is_task_interrupt(exc):
                    raise
                allow_fallback = True
                break
        if not allow_fallback:
            return
    if allow_fallback:
        _fallback_patch_competitor(card, competitor, usable_sources)


async def _patch_industry_card(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    card: IndustryCard,
    industry_name: str,
    sources: list[RawSource],
    allow_fallback: bool,
    focus_fields: list[str],
) -> None:
    usable_sources = [source for source in sources if (source.content or "").strip()]
    if not usable_sources:
        return
    if model_available and structured_call is not None:
        fields_to_fill = _dedupe_strings(focus_fields) or INDUSTRY_CARD_FIELDS
        for source in usable_sources:
            try:
                output = await structured_call(
                    CardPatchOutput,
                    [
                        ("system", industry_card_patch_system_prompt),
                        (
                            "user",
                            json.dumps(
                                {
                                    "industry": industry_name,
                                    "fields_to_fill": fields_to_fill,
                                    "current_card_values": _industry_value_snapshot(card),
                                    "chunk": _source_prompt_payload(source),
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    ],
                )
                assert isinstance(output, CardPatchOutput)
                _normalize_patch_refs(output.fields, source)
                _merge_field_patches(card, output.fields, allowed=INDUSTRY_CARD_FIELDS)
                fields_to_fill = _dedupe_strings(output.missing_fields) or _empty_industry_fields(card)
            except Exception as exc:
                if _is_task_interrupt(exc):
                    raise
                allow_fallback = True
                break
        if not allow_fallback:
            return
    if allow_fallback:
        _fallback_patch_industry(card, industry_name, usable_sources)


async def _generate_gap_queries(
    *,
    model_available: bool,
    structured_call: StructuredCall | None,
    benchmark: AnalysisBenchmark,
    object_type: Literal["competitor", "industry"],
    object_name: str,
    missing_fields: list[str],
    fallback_queries: list[SearchQuery],
) -> list[SearchQuery]:
    if not missing_fields:
        return []
    if not model_available or structured_call is None:
        return fallback_queries[:3]
    try:
        output = await structured_call(
            CardGapQueryOutput,
            [
                ("system", card_gap_query_system_prompt),
                (
                    "user",
                    json.dumps(
                        {
                            "target_product": benchmark.target_product,
                            "object_type": object_type,
                            "object_name": object_name,
                            "missing_fields": missing_fields[:12],
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
        )
        assert isinstance(output, CardGapQueryOutput)
        return _normalize_gap_queries(output.queries, fallback_queries, object_type, object_name, missing_fields)
    except Exception as exc:
        if _is_task_interrupt(exc):
            raise
        return fallback_queries[:3]


async def _qa_card(
    model_available: bool,
    structured_call: StructuredCall | None,
    object_type: Literal["competitor", "industry"],
    object_name: str,
    card: CompetitorDetailCard | IndustryCard,
) -> CardQAResult:
    fallback_missing = _missing_competitor_fields(card) if object_type == "competitor" else _missing_industry_fields(card)  # type: ignore[arg-type]
    fallback = CardQAResult(
        passed=not fallback_missing,
        missing_fields=fallback_missing,
        needs_retry=bool(fallback_missing),
        reason="规则兜底 QA 仅检查必填字段是否已有 value。",
    )
    if not model_available or structured_call is None:
        return fallback
    try:
        output = await structured_call(
            CardQAResult,
            [
                ("system", card_qa_system_prompt),
                (
                    "user",
                    json.dumps(
                        {
                            "object_type": object_type,
                            "object_name": object_name,
                            "hard_required_fields": COMPETITOR_HARD_REQUIRED_FIELDS if object_type == "competitor" else INDUSTRY_HARD_REQUIRED_FIELDS,
                            "optional_fields": _optional_fields(object_type),
                            "card": _qa_card_payload(card),
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
        )
        assert isinstance(output, CardQAResult)
        return _normalize_card_qa_result(output, object_type)
    except Exception as exc:
        if _is_task_interrupt(exc):
            raise
        return fallback


def build_card_report_references(
    competitor_cards: list[CompetitorDetailCard],
    industry_card: IndustryCard | None,
) -> list[ReportReference]:
    references: list[ReportReference] = []
    seen: set[str] = set()

    def add(reference_id: str, title: str, summary: str, payload: dict[str, Any]) -> None:
        if reference_id in seen or not str(summary or "").strip():
            return
        seen.add(reference_id)
        references.append(
            ReportReference(
                id=reference_id,
                title=title,
                reference_type="clean_card",
                anchor=_reference_anchor(reference_id),
                summary=_clip(summary, 1000),
                payload=payload,
            )
        )

    for card in competitor_cards:
        competitor_slug = _slug(card.competitor)
        for field_name in COMPETITOR_CARD_FIELDS:
            field = getattr(card, field_name)
            if isinstance(field, ResearchField) and field.value.strip():
                reference_id = f"card.competitor.{competitor_slug}.{field_name}"
                add(reference_id, f"{card.competitor} {field_name}", field.value, _reference_payload(field, field_name, card.competitor))
    if industry_card:
        for field_path in INDUSTRY_CARD_FIELDS:
            field = _get_field_by_path(industry_card, field_path)
            if isinstance(field, ResearchField) and field.value.strip():
                reference_id = f"card.industry.{field_path}"
                add(reference_id, f"Industry {field_path}", field.value, _reference_payload(field, field_path, None))
    return references


def _fallback_query_plan(benchmark: AnalysisBenchmark, request) -> DetailQueryPlan:
    industry_name = request.product_category or request.market or benchmark.target_product
    return DetailQueryPlan(
        competitor_query_templates=[
            QueryTemplate(template="{competitor} 官网 定价 功能 目标用户", target_fields=["pricing", "core_features", "target_users"], priority=1),
            QueryTemplate(template="{competitor} 商业模式 渠道 增长 新闻", target_fields=["business_model", "channels", "growth_signals", "recent_updates"], priority=2),
            QueryTemplate(template="{competitor} 用户评价 风险 投诉 差异化", target_fields=["user_feedback", "risks", "differentiation"], priority=3),
        ],
        industry_queries=[
            IndustryQuery(query=f"{industry_name} 市场规模 增长趋势 行业报告", target_fields=["market_situation.market_size", "market_situation.growth_trend"], priority=1),
            IndustryQuery(query=f"{industry_name} 竞争格局 进入壁垒 替代品", target_fields=["competition_situation.competition_intensity", "competition_situation.entry_barriers", "competition_situation.substitutes"], priority=2),
            IndustryQuery(query=f"{industry_name} 政策 监管 技术趋势 用户需求", target_fields=["external_environment.regulation_or_laws", "external_environment.technology_trends", "basic_info.main_users"], priority=3),
        ],
        rationale="规则兜底资料卡查询计划。",
    )


def _normalize_query_plan(plan: DetailQueryPlan, fallback: DetailQueryPlan) -> DetailQueryPlan:
    competitor_templates = [
        item
        for item in plan.competitor_query_templates
        if item.template.strip() and "{competitor}" in item.template
    ]
    industry_queries = [
        item
        for item in plan.industry_queries
        if item.query.strip()
    ]
    competitor_templates = _dedupe_templates([*competitor_templates, *fallback.competitor_query_templates])[:3]
    industry_queries = _dedupe_industry_queries([*industry_queries, *fallback.industry_queries])[:3]
    return DetailQueryPlan(
        competitor_query_templates=competitor_templates,
        industry_queries=industry_queries,
        rationale=plan.rationale or fallback.rationale,
    )


def _normalize_gap_queries(
    queries: list[SearchQuery],
    fallback_queries: list[SearchQuery],
    object_type: Literal["competitor", "industry"],
    object_name: str,
    missing_fields: list[str],
) -> list[SearchQuery]:
    normalized: list[SearchQuery] = []
    for query in [*queries, *fallback_queries]:
        text = query.query.strip()
        if not text:
            continue
        if object_type == "competitor" and object_name.lower() not in text.lower():
            text = f"{object_name} {text}"
        normalized.append(
            query.model_copy(
                update={
                    "query": text,
                    "competitor": object_name if object_type == "competitor" else None,
                    "expected_fields": query.expected_fields or missing_fields[:6],
                }
            )
        )
    return _dedupe_queries(normalized)[:3]


def _clean_competitor_card(card: CompetitorDetailCard) -> CompetitorDetailCardClean:
    fields: dict[str, str] = {}
    reference_ids: dict[str, str] = {}
    slug = _slug(card.competitor)
    for field_name in COMPETITOR_CARD_FIELDS:
        field = getattr(card, field_name)
        if isinstance(field, ResearchField) and field.value.strip():
            fields[field_name] = field.value
            reference_ids[field_name] = f"card.competitor.{slug}.{field_name}"
    return CompetitorDetailCardClean(
        competitor=card.competitor,
        competitor_type=card.competitor_type,
        fields=fields,
        reference_ids=reference_ids,
    )


def _clean_industry_card(card: IndustryCard) -> IndustryCardClean:
    fields: dict[str, str] = {}
    reference_ids: dict[str, str] = {}
    for field_path in INDUSTRY_CARD_FIELDS:
        field = _get_field_by_path(card, field_path)
        if isinstance(field, ResearchField) and field.value.strip():
            fields[field_path] = field.value
            reference_ids[field_path] = f"card.industry.{field_path}"
    return IndustryCardClean(fields=fields, reference_ids=reference_ids)


def _merge_field_patches(target: BaseModel, patches: dict[str, CardFieldPatch], *, allowed: list[str]) -> None:
    allowed_keys = {_field_key(item): item for item in allowed}
    suffix_counts: dict[str, int] = {}
    for item in allowed:
        suffix = _field_key(item.split(".")[-1])
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    for item in allowed:
        suffix = _field_key(item.split(".")[-1])
        if suffix_counts.get(suffix) == 1:
            allowed_keys.setdefault(suffix, item)
    for raw_name, patch in patches.items():
        field_path = allowed_keys.get(_field_key(raw_name))
        if not field_path or not patch.value.strip():
            continue
        current = _get_field_by_path(target, field_path)
        if not isinstance(current, ResearchField):
            continue
        merged = _merge_research_field(current, patch)
        _set_field_by_path(target, field_path, merged)


def _merge_research_field(current: ResearchField, patch: CardFieldPatch) -> ResearchField:
    refs = _dedupe_refs([*current.source_refs, *patch.source_refs])[:5]
    if not current.value.strip() or patch.confidence > current.confidence + 0.08:
        return ResearchField(value=_clip(patch.value, 3000), confidence=patch.confidence, source_refs=refs)
    if patch.value.strip() and patch.value.strip() not in current.value:
        value = _clip(f"{current.value}\n{patch.value}", 3000)
    else:
        value = current.value
    return ResearchField(value=value, confidence=max(current.confidence, patch.confidence), source_refs=refs)


def _normalize_patch_refs(fields: dict[str, CardFieldPatch], source: RawSource) -> None:
    for field in fields.values():
        if not field.value.strip():
            continue
        if not field.source_refs:
            field.source_refs = [_source_ref(source, quote=field.value)]
            continue
        normalized: list[SourceRef] = []
        for ref in field.source_refs:
            normalized.append(
                ref.model_copy(
                    update={
                        "source_id": ref.source_id or source.id,
                        "chunk_id": ref.chunk_id or source.metadata.get("chunk_id"),
                        "url": ref.url or source.url,
                        "title": ref.title if ref.title != "Untitled source" else source.title,
                        "quote": ref.quote or _clip(field.value, 900),
                        "screenshot_path": ref.screenshot_path or source.metadata.get("screenshot_path"),
                        "evidence_screenshot_path": ref.evidence_screenshot_path or source.metadata.get("evidence_screenshot_path") or source.metadata.get("quote_shot_path"),
                    }
                )
            )
        field.source_refs = normalized


def _fallback_patch_competitor(card: CompetitorDetailCard, competitor: CandidateCompetitor, sources: list[RawSource]) -> None:
    keywords = {
        "name": [competitor.name],
        "website": ["official", "website", "官网"],
        "positioning": ["position", "about", "target", "定位", "简介"],
        "target_users": ["user", "customer", "audience", "用户", "客户"],
        "core_scenarios": ["scenario", "use case", "场景"],
        "core_features": ["feature", "capability", "release", "功能"],
        "key_parameters": ["parameter", "limit", "spec", "参数"],
        "pricing": ["pricing", "price", "subscription", "定价", "价格", "收费"],
        "business_model": ["business model", "revenue", "商业模式", "收入"],
        "channels": ["channel", "app store", "website", "渠道"],
        "growth_signals": ["growth", "download", "mau", "增长", "下载"],
        "funding_or_org_signals": ["funding", "financing", "hiring", "融资"],
        "technology_or_product_features": ["technology", "ai", "api", "技术"],
        "differentiation": ["advantage", "differentiation", "差异化", "优势"],
        "user_feedback": ["review", "rating", "feedback", "评价", "评论"],
        "risks": ["risk", "complaint", "regulation", "风险", "投诉"],
        "recent_updates": ["news", "update", "release", "新闻", "更新"],
        "key_evidence": [competitor.name],
    }
    for field_name, terms in keywords.items():
        if getattr(card, field_name).value:
            continue
        field = _field_from_sources(sources, terms)
        if field.value:
            setattr(card, field_name, field)


def _fallback_patch_industry(card: IndustryCard, industry_name: str, sources: list[RawSource]) -> None:
    mapping = {
        "basic_info.industry_name": [industry_name],
        "basic_info.industry_scope": ["industry", "market", "scope", "行业", "市场"],
        "basic_info.main_users": ["user", "customer", "consumer", "用户", "消费者"],
        "basic_info.core_scenarios": ["scenario", "use case", "场景"],
        "basic_info.buyer_behavior_or_preferences": ["buyer", "preference", "consumer", "偏好"],
        "market_situation.market_size": ["market size", "tam", "revenue", "市场规模"],
        "market_situation.growth_trend": ["growth", "cagr", "trend", "增长", "趋势"],
        "market_situation.willingness_to_pay": ["pricing", "willingness to pay", "付费"],
        "market_situation.cost_pressure": ["cost", "expense", "成本"],
        "market_situation.investment_environment": ["funding", "investment", "融资", "投资"],
        "competition_situation.competition_intensity": ["competition", "competitor", "竞争"],
        "competition_situation.entry_barriers": ["barrier", "moat", "壁垒"],
        "competition_situation.substitutes": ["substitute", "alternative", "替代"],
        "competition_situation.user_switching_cost": ["switching cost", "retention", "迁移成本"],
        "external_environment.regulation_or_laws": ["regulation", "law", "policy", "监管", "政策"],
        "external_environment.technology_trends": ["technology", "ai", "trend", "技术"],
        "external_environment.key_resource_dependencies": ["resource", "dependency", "supply", "资源", "供应"],
    }
    for field_path, terms in mapping.items():
        current = _get_field_by_path(card, field_path)
        if isinstance(current, ResearchField) and current.value:
            continue
        field = _field_from_sources(sources, terms)
        if field.value:
            _set_field_by_path(card, field_path, field)


def _field_from_sources(sources: list[RawSource], keywords: list[str]) -> ResearchField:
    for source in sources:
        text = f"{source.title} {source.content}"
        lower = text.lower()
        if any(term.lower() in lower for term in keywords):
            quote = _clip(_sentence_for_terms(source.content, keywords) or source.content, 900)
            return ResearchField(value=quote, confidence=0.38, source_refs=[_source_ref(source, quote=quote)])
    if sources:
        quote = _clip(sources[0].content, 700)
        return ResearchField(value=quote, confidence=0.25, source_refs=[_source_ref(sources[0], quote=quote)])
    return ResearchField()


def _missing_competitor_fields(card: CompetitorDetailCard) -> list[str]:
    return [field for field in COMPETITOR_HARD_REQUIRED_FIELDS if not getattr(card, field).value.strip()]


def _missing_industry_fields(card: IndustryCard) -> list[str]:
    return [field for field in INDUSTRY_HARD_REQUIRED_FIELDS if not _get_field_by_path(card, field).value.strip()]


def _empty_competitor_fields(card: CompetitorDetailCard) -> list[str]:
    return [field for field in COMPETITOR_CARD_FIELDS if not getattr(card, field).value.strip()]


def _empty_industry_fields(card: IndustryCard) -> list[str]:
    return [field for field in INDUSTRY_CARD_FIELDS if not _get_field_by_path(card, field).value.strip()]


def _count_competitor_fields(card: CompetitorDetailCard) -> int:
    return len([field for field in COMPETITOR_CARD_FIELDS if getattr(card, field).value.strip()])


def _count_industry_fields(card: IndustryCard) -> int:
    return len([field for field in INDUSTRY_CARD_FIELDS if _get_field_by_path(card, field).value.strip()])


def _count_card_source_refs(card: CompetitorDetailCard | IndustryCard) -> int:
    if isinstance(card, CompetitorDetailCard):
        fields = [getattr(card, field) for field in COMPETITOR_CARD_FIELDS]
    else:
        fields = [_get_field_by_path(card, field) for field in INDUSTRY_CARD_FIELDS]
    return sum(len(field.source_refs) for field in fields if isinstance(field, ResearchField))


def _competitor_queries(competitor: CandidateCompetitor, templates: list[QueryTemplate], missing: list[str], iteration: int) -> list[SearchQuery]:
    queries: list[SearchQuery] = []
    usable_templates = [item for item in templates if "{competitor}" in item.template]
    for template in usable_templates[:3]:
        queries.append(
            SearchQuery(
                query=template.template.replace("{competitor}", competitor.name),
                competitor=competitor.name,
                target_site_types=template.target_site_types or ["official_site", "news", "reviews", "industry_report"],
                expected_fields=template.target_fields or missing[:6],
                priority=template.priority,
                rationale=template.rationale or f"竞品资料卡第 {iteration} 轮查询。",
            )
        )
    return queries[:3]


def _industry_queries(industry_name: str, planned: list[IndustryQuery], missing: list[str], iteration: int) -> list[SearchQuery]:
    queries = [
        SearchQuery(
            query=item.query,
            expected_fields=item.target_fields or missing[:6],
            target_site_types=item.target_site_types or ["industry_report", "news", "official_statistics"],
            priority=item.priority,
            rationale=item.rationale or f"行业资料卡第 {iteration} 轮查询。",
        )
        for item in planned[:3]
    ]
    if not queries:
        queries = [
            SearchQuery(query=f"{industry_name} 市场规模 增长 竞争 监管 技术趋势", expected_fields=missing[:6], priority=1),
        ]
    return queries[:3]


def _competitor_gap_queries(competitor: CandidateCompetitor, missing: list[str], iteration: int) -> list[SearchQuery]:
    templates = {
        "growth_signals": f"{competitor.name} 财报 收入 用户 增长 下滑 2023 2024",
        "funding_or_org_signals": f"{competitor.name} 融资 投资 招聘 团队 组织 新闻",
        "key_parameters": f"{competitor.name} 参数 规格 限制 配置 帮助文档",
        "user_feedback": f"{competitor.name} 用户投诉 评价 售后 发货 正品",
        "risks": f"{competitor.name} 投诉 风险 监管 争议 售后 问题",
        "pricing": f"{competitor.name} 官网 定价 价格 收费 订阅",
        "business_model": f"{competitor.name} 商业模式 收入 营收 财报",
    }
    queries: list[SearchQuery] = []
    for field in missing:
        query = templates.get(field)
        if query:
            queries.append(
                SearchQuery(
                    query=query,
                    competitor=competitor.name,
                    expected_fields=[field],
                    priority=1,
                    rationale=f"资料卡补搜第 {iteration} 轮：定向补齐 {field}。",
                )
            )
    if queries:
        return queries[:3]
    focus = " ".join(field.replace("_", " ") for field in missing[:4])
    return [
        SearchQuery(query=f"{competitor.name} {focus} 官网 新闻 评价", competitor=competitor.name, expected_fields=missing[:6], priority=1, rationale=f"竞品资料卡补搜第 {iteration} 轮。")
    ]


def _industry_gap_queries(industry_name: str, missing: list[str], iteration: int) -> list[SearchQuery]:
    focus = " ".join(field.split(".")[-1].replace("_", " ") for field in missing[:4])
    return [
        SearchQuery(query=f"{industry_name} {focus} 行业报告 数据 政策", expected_fields=missing[:6], priority=1, rationale=f"行业资料卡补搜第 {iteration} 轮。")
    ]


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
    return matched or sources


def _competitor_value_snapshot(card: CompetitorDetailCard) -> dict[str, str]:
    return {field: getattr(card, field).value for field in COMPETITOR_CARD_FIELDS if getattr(card, field).value.strip()}


def _industry_value_snapshot(card: IndustryCard) -> dict[str, str]:
    values: dict[str, str] = {}
    for field_path in INDUSTRY_CARD_FIELDS:
        field = _get_field_by_path(card, field_path)
        if field.value.strip():
            values[field_path] = field.value
    return values


def _qa_card_payload(card: CompetitorDetailCard | IndustryCard) -> dict[str, Any]:
    if isinstance(card, CompetitorDetailCard):
        return {
            "competitor": card.competitor,
            "fields": {
                field_name: _qa_field_payload(getattr(card, field_name))
                for field_name in COMPETITOR_CARD_FIELDS
            },
        }
    return {
        "fields": {
            field_path: _qa_field_payload(_get_field_by_path(card, field_path))
            for field_path in INDUSTRY_CARD_FIELDS
        }
    }


def _qa_field_payload(field: ResearchField) -> dict[str, Any]:
    return {
        "value": field.value,
        "source_refs": [{"quote": ref.quote, "url": ref.url} for ref in field.source_refs if ref.quote.strip() or ref.url.strip()],
    }


def _qa_needs_retry(qa: CardQAResult) -> bool:
    return bool(qa.needs_retry or not qa.passed or qa.missing_fields or qa.error_fields or qa.reference_issues)


def _qa_failed_fields(qa: CardQAResult) -> list[str]:
    return _dedupe_strings([*qa.missing_fields, *qa.error_fields])


def _optional_fields(object_type: Literal["competitor", "industry"]) -> list[str]:
    if object_type == "competitor":
        return [field for field in COMPETITOR_CARD_FIELDS if field not in COMPETITOR_HARD_REQUIRED_FIELDS]
    return [field for field in INDUSTRY_CARD_FIELDS if field not in INDUSTRY_HARD_REQUIRED_FIELDS]


def _normalize_card_qa_result(qa: CardQAResult, object_type: Literal["competitor", "industry"]) -> CardQAResult:
    hard_required = set(COMPETITOR_HARD_REQUIRED_FIELDS if object_type == "competitor" else INDUSTRY_HARD_REQUIRED_FIELDS)
    hard_missing = [field for field in qa.missing_fields if field in hard_required]
    hard_errors = [field for field in qa.error_fields if field in hard_required]
    optional_issues = [
        field
        for field in [*qa.missing_fields, *qa.error_fields]
        if field and field not in hard_required
    ]
    reference_issues = list(qa.reference_issues)
    if optional_issues:
        reference_issues.append("Optional fields missing or weakly supported: " + ", ".join(_dedupe_strings(optional_issues)[:12]))
    needs_retry = bool(hard_missing or hard_errors or (qa.needs_retry and (hard_missing or hard_errors)))
    return qa.model_copy(
        update={
            "passed": not needs_retry,
            "needs_retry": needs_retry,
            "missing_fields": hard_missing,
            "error_fields": hard_errors,
            "reference_issues": reference_issues,
        }
    )


def _hard_error_fields(qa: CardQAResult, hard_required: list[str]) -> list[str]:
    hard = set(hard_required)
    return [field for field in qa.error_fields if field in hard]


def _clear_error_fields(target: BaseModel, fields: list[str], *, allowed: list[str]) -> None:
    allowed_keys = {_field_key(item): item for item in allowed}
    for raw_field in fields:
        field_path = allowed_keys.get(_field_key(raw_field))
        if not field_path:
            continue
        current = _get_field_by_path(target, field_path)
        if isinstance(current, ResearchField):
            _set_field_by_path(target, field_path, ResearchField())


def _source_prompt_payload(source: RawSource) -> dict[str, Any]:
    return {
        "source_id": source.id,
        "chunk_id": source.metadata.get("chunk_id"),
        "title": _strip_prompt_links(source.title),
        "url": source.url,
        "content": _strip_prompt_links(_clip(source.content, 2400)),
    }


def _source_ref(source: RawSource, *, quote: str) -> SourceRef:
    return SourceRef(
        source_id=source.id,
        chunk_id=source.metadata.get("chunk_id"),
        url=source.url,
        title=source.title,
        quote=_clip(quote or source.content, 900),
        screenshot_path=source.metadata.get("screenshot_path"),
        evidence_screenshot_path=source.metadata.get("evidence_screenshot_path") or source.metadata.get("quote_shot_path"),
        chunk_shot_paths=[],
    )


def _reference_payload(field: ResearchField, field_name: str, competitor: str | None) -> dict[str, Any]:
    return {
        "field": field_name,
        "competitor": competitor,
        "confidence": field.confidence,
        "source_refs": [ref.model_dump(mode="json") for ref in field.source_refs[:5]],
    }


def _get_field_by_path(value: Any, field_path: str) -> ResearchField:
    item = value
    for part in field_path.split("."):
        item = getattr(item, part)
    return item


def _set_field_by_path(value: Any, field_path: str, field: ResearchField) -> None:
    parts = field_path.split(".")
    item = value
    for part in parts[:-1]:
        item = getattr(item, part)
    setattr(item, parts[-1], field)


def _sentence_for_terms(content: str, terms: list[str]) -> str:
    sentences = re.split(r"(?<=[.!?。！？])\s+", content.strip())
    for sentence in sentences:
        lower = sentence.lower()
        if any(term.lower() in lower for term in terms):
            return sentence
    return ""


def _dedupe_refs(refs: list[SourceRef]) -> list[SourceRef]:
    seen: set[str] = set()
    result: list[SourceRef] = []
    for ref in refs:
        key = f"{ref.source_id or ''}|{ref.chunk_id or ''}|{ref.quote[:120]}"
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result


def _dedupe_strings(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _dedupe_queries(values: list[SearchQuery]) -> list[SearchQuery]:
    seen: set[str] = set()
    result: list[SearchQuery] = []
    for value in values:
        key = f"{value.query.strip().lower()}|{value.competitor or ''}"
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _dedupe_templates(values: list[QueryTemplate]) -> list[QueryTemplate]:
    seen: set[str] = set()
    result: list[QueryTemplate] = []
    for value in values:
        key = value.template.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _dedupe_industry_queries(values: list[IndustryQuery]) -> list[IndustryQuery]:
    seen: set[str] = set()
    result: list[IndustryQuery] = []
    for value in values:
        key = value.query.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _field_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", str(value or "").strip()).strip("-").lower()
    return cleaned or "item"


def _reference_anchor(reference_id: str) -> str:
    return "ref-" + re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", reference_id).strip("-").lower()


def _clip(value: str, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}..."


def _strip_prompt_links(value: str) -> str:
    text = re.sub(r"\b[a-z][a-z0-9+.-]*://\S+|www\.\S+", "[link_removed]", str(value or ""))
    return re.sub(r"\b(?:search_runs|task_runs|screenshots|extractions)/\S+", "[path_removed]", text)


def _is_task_interrupt(exc: Exception) -> bool:
    return exc.__class__.__name__ in {"LLMDecisionNeeded", "TaskCancelled"}
