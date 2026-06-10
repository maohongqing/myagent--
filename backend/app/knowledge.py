from __future__ import annotations

import re
from typing import Iterable

from myagent.backend.app.models import (
    CompetitorDetailCardClean,
    CompetitorReport,
    CreateTaskRequest,
    EvidenceItem,
    IndustryCardClean,
    KnowledgeCompetitorCard,
    KnowledgeImportCommitRequest,
    KnowledgeImportPreview,
    KnowledgeIndustryCard,
    KnowledgeReuseSummary,
    KnowledgeSource,
    KnowledgeUpdateDiff,
    KnowledgeUpdateSuggestion,
    utc_now,
)
from myagent.backend.app.storage import Storage


def build_import_preview(report: CompetitorReport) -> KnowledgeImportPreview:
    sources = [_source_from_evidence(report.task_id, item) for item in report.evidence]
    source_ids = [item.id for item in sources]
    competitor_cards = [
        _competitor_card_from_clean(report, card, source_ids)
        for card in report.competitor_detail_cards_clean
        if card.competitor.strip()
    ]
    industry_cards = []
    if report.industry_card_clean:
        industry_cards.append(_industry_card_from_clean(report, report.industry_card_clean, source_ids))
    return KnowledgeImportPreview(
        task_id=report.task_id,
        competitor_cards=competitor_cards,
        industry_cards=industry_cards,
        sources=sources,
    )


def commit_import(storage: Storage, report: CompetitorReport, request: KnowledgeImportCommitRequest) -> KnowledgeImportPreview:
    preview = build_import_preview(report)
    selected_competitors = set(request.competitor_card_ids)
    selected_industries = set(request.industry_card_ids)
    selected_source_ids: set[str] = set()
    saved_competitors: list[KnowledgeCompetitorCard] = []
    saved_industries: list[KnowledgeIndustryCard] = []

    for card in preview.competitor_cards:
        if card.id in selected_competitors:
            saved_competitors.append(card)
            selected_source_ids.update(card.knowledge_source_ids)
    for card in preview.industry_cards:
        if card.id in selected_industries:
            saved_industries.append(card)
            selected_source_ids.update(card.knowledge_source_ids)
    for source in preview.sources:
        if source.id in selected_source_ids:
            storage.save_knowledge_source(source)
    for card in saved_competitors:
        storage.save_knowledge_competitor_card(card, version_action="import_from_task")
    for card in saved_industries:
        storage.save_knowledge_industry_card(card, version_action="import_from_task")
    return KnowledgeImportPreview(
        task_id=report.task_id,
        competitor_cards=saved_competitors,
        industry_cards=saved_industries,
        sources=[source for source in preview.sources if source.id in selected_source_ids],
    )


def retrieve_knowledge_context(storage: Storage, request: CreateTaskRequest, limit: int = 8) -> KnowledgeReuseSummary:
    terms = _request_terms(request)
    competitor_cards: list[KnowledgeCompetitorCard] = []
    industry_cards: list[KnowledgeIndustryCard] = []

    for term in terms:
        competitor_cards.extend(storage.list_knowledge_competitor_cards(q=term, limit=limit))
        industry_cards.extend(storage.list_knowledge_industry_cards(q=term, limit=limit))

    competitor_cards = _dedupe_by_id(competitor_cards)[:limit]
    industry_cards = _dedupe_by_id(industry_cards)[:3]
    source_ids = sorted({source_id for card in [*competitor_cards, *industry_cards] for source_id in card.knowledge_source_ids})
    notes: list[str] = []
    if competitor_cards:
        notes.append("Reused competitor cards: " + ", ".join(card.name for card in competitor_cards))
    if industry_cards:
        notes.append("Reused industry cards: " + ", ".join(card.title for card in industry_cards))
    return KnowledgeReuseSummary(
        competitor_cards=competitor_cards,
        industry_cards=industry_cards,
        source_ids=source_ids,
        notes=notes,
    )


def generate_update_suggestions(storage: Storage, report: CompetitorReport) -> list[KnowledgeUpdateSuggestion]:
    preview = build_import_preview(report)
    for source in preview.sources:
        storage.save_knowledge_source(source)

    suggestions: list[KnowledgeUpdateSuggestion] = []
    for proposed in preview.competitor_cards:
        existing = _find_competitor_by_name(storage, proposed.name)
        if existing is None:
            suggestion = KnowledgeUpdateSuggestion(
                kind="create",
                card_type="competitor",
                title=f"新增竞品资料卡：{proposed.name}",
                summary=f"从任务 {report.task_id} 生成的新竞品资料卡。",
                proposed_competitor_card=proposed,
                source_ids=proposed.knowledge_source_ids,
                origin_task_id=report.task_id,
            )
        else:
            diffs = _field_diffs(existing.card.fields, proposed.card.fields, proposed.card.reference_ids)
            new_sources = sorted(set(proposed.knowledge_source_ids) - set(existing.knowledge_source_ids))
            if diffs:
                suggestion = KnowledgeUpdateSuggestion(
                    kind="update",
                    card_type="competitor",
                    target_card_id=existing.id,
                    title=f"更新竞品资料卡：{existing.name}",
                    summary=f"从任务 {report.task_id} 发现 {len(diffs)} 个字段变化。",
                    proposed_competitor_card=proposed,
                    diffs=diffs,
                    source_ids=proposed.knowledge_source_ids,
                    origin_task_id=report.task_id,
                )
            elif new_sources:
                suggestion = KnowledgeUpdateSuggestion(
                    kind="append_sources",
                    card_type="competitor",
                    target_card_id=existing.id,
                    title=f"追加竞品来源：{existing.name}",
                    summary=f"从任务 {report.task_id} 发现 {len(new_sources)} 条新来源。",
                    proposed_competitor_card=proposed,
                    source_ids=new_sources,
                    origin_task_id=report.task_id,
                )
            else:
                continue
        storage.save_knowledge_suggestion(suggestion)
        suggestions.append(suggestion)

    for proposed in preview.industry_cards:
        existing = _find_industry_by_title(storage, proposed.title)
        if existing is None:
            suggestion = KnowledgeUpdateSuggestion(
                kind="create",
                card_type="industry",
                title=f"新增行业资料卡：{proposed.title}",
                summary=f"从任务 {report.task_id} 生成的新行业资料卡。",
                proposed_industry_card=proposed,
                source_ids=proposed.knowledge_source_ids,
                origin_task_id=report.task_id,
            )
        else:
            diffs = _field_diffs(existing.card.fields, proposed.card.fields, proposed.card.reference_ids)
            new_sources = sorted(set(proposed.knowledge_source_ids) - set(existing.knowledge_source_ids))
            if diffs:
                suggestion = KnowledgeUpdateSuggestion(
                    kind="update",
                    card_type="industry",
                    target_card_id=existing.id,
                    title=f"更新行业资料卡：{existing.title}",
                    summary=f"从任务 {report.task_id} 发现 {len(diffs)} 个字段变化。",
                    proposed_industry_card=proposed,
                    diffs=diffs,
                    source_ids=proposed.knowledge_source_ids,
                    origin_task_id=report.task_id,
                )
            elif new_sources:
                suggestion = KnowledgeUpdateSuggestion(
                    kind="append_sources",
                    card_type="industry",
                    target_card_id=existing.id,
                    title=f"追加行业来源：{existing.title}",
                    summary=f"从任务 {report.task_id} 发现 {len(new_sources)} 条新来源。",
                    proposed_industry_card=proposed,
                    source_ids=new_sources,
                    origin_task_id=report.task_id,
                )
            else:
                continue
        storage.save_knowledge_suggestion(suggestion)
        suggestions.append(suggestion)

    return suggestions


def accept_suggestion(storage: Storage, suggestion_id: str) -> KnowledgeUpdateSuggestion | None:
    suggestion = storage.get_knowledge_suggestion(suggestion_id)
    if not suggestion or suggestion.status != "pending":
        return suggestion

    if suggestion.card_type == "competitor":
        proposed = suggestion.proposed_competitor_card
        if proposed is None:
            return suggestion
        if suggestion.kind == "create" or not suggestion.target_card_id:
            storage.save_knowledge_competitor_card(proposed, version_action="accept_suggestion_create")
        else:
            existing = storage.get_knowledge_competitor_card(suggestion.target_card_id)
            if existing:
                if suggestion.kind == "update":
                    existing.card.fields.update(proposed.card.fields)
                    existing.card.reference_ids.update(proposed.card.reference_ids)
                existing.knowledge_source_ids = sorted(set(existing.knowledge_source_ids) | set(suggestion.source_ids))
                existing.last_verified_at = utc_now()
                storage.save_knowledge_competitor_card(existing, version_action=f"accept_suggestion_{suggestion.kind}")
    else:
        proposed = suggestion.proposed_industry_card
        if proposed is None:
            return suggestion
        if suggestion.kind == "create" or not suggestion.target_card_id:
            storage.save_knowledge_industry_card(proposed, version_action="accept_suggestion_create")
        else:
            existing = storage.get_knowledge_industry_card(suggestion.target_card_id)
            if existing:
                if suggestion.kind == "update":
                    existing.card.fields.update(proposed.card.fields)
                    existing.card.reference_ids.update(proposed.card.reference_ids)
                existing.knowledge_source_ids = sorted(set(existing.knowledge_source_ids) | set(suggestion.source_ids))
                existing.last_verified_at = utc_now()
                storage.save_knowledge_industry_card(existing, version_action=f"accept_suggestion_{suggestion.kind}")

    suggestion.status = "accepted"
    suggestion.resolved_at = utc_now()
    storage.save_knowledge_suggestion(suggestion)
    return suggestion


def reject_suggestion(storage: Storage, suggestion_id: str) -> KnowledgeUpdateSuggestion | None:
    suggestion = storage.get_knowledge_suggestion(suggestion_id)
    if not suggestion or suggestion.status != "pending":
        return suggestion
    suggestion.status = "rejected"
    suggestion.resolved_at = utc_now()
    storage.save_knowledge_suggestion(suggestion)
    return suggestion


def _source_from_evidence(task_id: str, evidence: EvidenceItem) -> KnowledgeSource:
    return KnowledgeSource(
        id=f"ksrc_{_slug(task_id)}_{_slug(evidence.id)}",
        url=evidence.url,
        title=evidence.title,
        excerpt=evidence.excerpt,
        source_type=evidence.source_type,
        credibility=evidence.credibility,
        published_at=evidence.published_at,
        captured_at=evidence.captured_at,
        origin_task_id=task_id,
        metadata={
            "evidence_id": evidence.id,
            "related_fields": evidence.related_fields,
            "competitor": evidence.competitor,
            "field_name": evidence.field_name,
        },
    )


def _competitor_card_from_clean(
    report: CompetitorReport, card: CompetitorDetailCardClean, source_ids: list[str]
) -> KnowledgeCompetitorCard:
    clean_card = CompetitorDetailCardClean.model_validate(card.model_dump(mode="json"))
    industry_name = _report_industry_name(report)
    return KnowledgeCompetitorCard(
        id=f"kcomp_{_slug(card.competitor)}",
        name=card.competitor,
        canonical_name=card.competitor,
        industry=industry_name,
        market=report.target_product,
        tags=[card.competitor_type or "competitor"],
        confidence=_confidence_from_field_count(card.fields),
        freshness_status="fresh",
        origin_task_id=report.task_id,
        card=clean_card,
        knowledge_source_ids=source_ids,
        last_verified_at=report.generated_at,
    )


def _industry_card_from_clean(
    report: CompetitorReport, card: IndustryCardClean, source_ids: list[str]
) -> KnowledgeIndustryCard:
    clean_card = IndustryCardClean.model_validate(card.model_dump(mode="json"))
    title = _report_industry_name(report)
    return KnowledgeIndustryCard(
        id=f"kind_{_slug(title)}",
        title=title,
        canonical_name=title,
        industry=title,
        market=report.target_product,
        tags=["industry"],
        confidence=_confidence_from_field_count(card.fields),
        freshness_status="fresh",
        origin_task_id=report.task_id,
        card=clean_card,
        knowledge_source_ids=source_ids,
        last_verified_at=report.generated_at,
    )


def _report_industry_name(report: CompetitorReport) -> str:
    if report.analysis_intent and report.analysis_intent.industry.strip():
        return report.analysis_intent.industry.strip()
    if report.analysis_benchmark and report.analysis_benchmark.target_product.strip():
        return report.analysis_benchmark.target_product.strip()
    return report.target_product.strip()


def _request_terms(request: CreateTaskRequest) -> list[str]:
    values = [
        request.target_product,
        request.raw_description or "",
        request.market or "",
        request.product_category or "",
        request.geography or "",
        request.our_product or "",
        *request.competitors,
    ]
    terms: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned:
            terms.append(cleaned)
        for token in re.split(r"[\s,，/|;；:：]+", cleaned):
            if len(token) >= 2:
                terms.append(token)
    return _dedupe_strings(terms)[:12]


def _field_diffs(
    existing: dict[str, str], proposed: dict[str, str], reference_ids: dict[str, str]
) -> list[KnowledgeUpdateDiff]:
    diffs: list[KnowledgeUpdateDiff] = []
    for field, new_value in proposed.items():
        old_value = existing.get(field, "")
        if new_value.strip() and new_value.strip() != old_value.strip():
            diffs.append(
                KnowledgeUpdateDiff(
                    field=field,
                    old_value=old_value,
                    new_value=new_value,
                    reason="New task output differs from the current knowledge card field.",
                    evidence_refs=[reference_ids[field]] if field in reference_ids else [],
                    confidence=0.7,
                )
            )
    return diffs


def _find_competitor_by_name(storage: Storage, name: str) -> KnowledgeCompetitorCard | None:
    target = name.strip().lower()
    for card in storage.list_knowledge_competitor_cards(q=name, limit=20):
        names = [card.name, card.canonical_name, *card.aliases]
        if target in {item.strip().lower() for item in names if item.strip()}:
            return card
    return None


def _find_industry_by_title(storage: Storage, title: str) -> KnowledgeIndustryCard | None:
    target = title.strip().lower()
    for card in storage.list_knowledge_industry_cards(q=title, limit=20):
        names = [card.title, card.canonical_name, *card.aliases]
        if target in {item.strip().lower() for item in names if item.strip()}:
            return card
    return None


def _confidence_from_field_count(fields: dict[str, str]) -> float:
    return min(0.9, 0.45 + 0.05 * len([value for value in fields.values() if value.strip()]))


def _dedupe_by_id(items: Iterable):
    seen: set[str] = set()
    result = []
    for item in items:
        if item.id not in seen:
            seen.add(item.id)
            result.append(item)
    return result


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", value.strip().lower()).strip("_")
    return text[:80] or "item"
