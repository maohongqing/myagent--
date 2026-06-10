from __future__ import annotations

from myagent.backend.app.knowledge import (
    accept_suggestion,
    build_import_preview,
    commit_import,
    generate_update_suggestions,
    retrieve_knowledge_context,
)
from myagent.backend.app.models import (
    CreateTaskRequest,
    KnowledgeCompetitorCard,
    KnowledgeImportCommitRequest,
)
from myagent.backend.app.storage import JSONStorage, SQLiteStorage

from conftest import make_report


def test_sqlite_knowledge_crud_soft_delete_and_versions(tmp_path):
    storage = SQLiteStorage(tmp_path / "app.db")
    storage.init()
    report = make_report("task_knowledge")
    preview = build_import_preview(report)
    card = preview.competitor_cards[0]

    for source in preview.sources:
        storage.save_knowledge_source(source)
    storage.save_knowledge_competitor_card(card, version_action="create")
    loaded = storage.get_knowledge_competitor_card(card.id)

    assert loaded is not None
    assert loaded.name == "BetaSuite"
    assert storage.list_knowledge_competitor_cards(q="BetaSuite")[0].id == card.id
    assert storage.list_knowledge_versions("competitor", card.id)

    deleted = storage.delete_knowledge_competitor_card(card.id)
    assert deleted is not None
    assert storage.list_knowledge_competitor_cards(q="BetaSuite") == []


def test_json_knowledge_import_and_retrieve(tmp_path):
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    report = make_report("task_import")
    preview = build_import_preview(report)
    saved = commit_import(
        storage,
        report,
        KnowledgeImportCommitRequest(
            competitor_card_ids=[preview.competitor_cards[0].id],
            industry_card_ids=[preview.industry_cards[0].id],
        ),
    )

    assert saved.competitor_cards[0].name == "BetaSuite"
    context = retrieve_knowledge_context(storage, CreateTaskRequest(target_product="Acme Workspace"))
    assert [card.name for card in context.competitor_cards] == ["BetaSuite"]
    assert context.industry_cards


def test_update_suggestion_accepts_new_card(memory_storage):
    report = make_report("task_suggestion")
    suggestions = generate_update_suggestions(memory_storage, report)
    create_suggestion = next(item for item in suggestions if item.card_type == "competitor")

    accepted = accept_suggestion(memory_storage, create_suggestion.id)

    assert accepted is not None
    assert accepted.status == "accepted"
    assert memory_storage.list_knowledge_competitor_cards(q="BetaSuite")


def test_update_suggestion_detects_field_diff(memory_storage):
    report = make_report("task_existing")
    existing = build_import_preview(report).competitor_cards[0]
    existing.card.fields["pricing"] = "Old pricing"
    memory_storage.save_knowledge_competitor_card(existing)

    suggestions = generate_update_suggestions(memory_storage, report)
    update = next(item for item in suggestions if item.card_type == "competitor" and item.kind == "update")

    assert update.target_card_id == existing.id
    assert any(diff.field == "pricing" for diff in update.diffs)
