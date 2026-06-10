from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from myagent.backend.app.config import Settings
from myagent.backend.app.models import (
    AgentRunEvent,
    AnalysisTask,
    CompetitorReport,
    FreshnessStatus,
    KnowledgeCardType,
    KnowledgeCardVersion,
    KnowledgeCompetitorCard,
    KnowledgeIndustryCard,
    KnowledgeSource,
    KnowledgeUpdateSuggestion,
    SurveyAnalysisResult,
    SurveyDesign,
    utc_now,
)


class Storage(ABC):
    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_task(self, task: AnalysisTask) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_task(self, task_id: str) -> AnalysisTask | None:
        raise NotImplementedError

    @abstractmethod
    def list_tasks(self, limit: int = 50) -> list[AnalysisTask]:
        raise NotImplementedError

    @abstractmethod
    def save_report(self, report: CompetitorReport) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_report(self, task_id: str) -> CompetitorReport | None:
        raise NotImplementedError

    @abstractmethod
    def append_event(self, event: AgentRunEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_events(self, task_id: str) -> list[AgentRunEvent]:
        raise NotImplementedError

    def save_survey_design(self, design: SurveyDesign) -> None:
        raise NotImplementedError

    def get_survey_design(self, task_id: str) -> SurveyDesign | None:
        raise NotImplementedError

    def save_survey_analysis(self, analysis: SurveyAnalysisResult) -> None:
        raise NotImplementedError

    def get_survey_analysis(self, task_id: str) -> SurveyAnalysisResult | None:
        raise NotImplementedError

    def save_knowledge_competitor_card(self, card: KnowledgeCompetitorCard, *, version_action: str = "save") -> None:
        raise NotImplementedError

    def get_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        raise NotImplementedError

    def list_knowledge_competitor_cards(
        self,
        q: str = "",
        industry: str = "",
        tag: str = "",
        freshness_status: FreshnessStatus | None = None,
        limit: int = 50,
    ) -> list[KnowledgeCompetitorCard]:
        raise NotImplementedError

    def delete_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        raise NotImplementedError

    def save_knowledge_industry_card(self, card: KnowledgeIndustryCard, *, version_action: str = "save") -> None:
        raise NotImplementedError

    def get_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        raise NotImplementedError

    def list_knowledge_industry_cards(
        self,
        q: str = "",
        industry: str = "",
        tag: str = "",
        freshness_status: FreshnessStatus | None = None,
        limit: int = 50,
    ) -> list[KnowledgeIndustryCard]:
        raise NotImplementedError

    def delete_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        raise NotImplementedError

    def save_knowledge_source(self, source: KnowledgeSource) -> None:
        raise NotImplementedError

    def get_knowledge_source(self, source_id: str) -> KnowledgeSource | None:
        raise NotImplementedError

    def list_knowledge_sources(self, source_ids: list[str] | None = None, limit: int = 100) -> list[KnowledgeSource]:
        raise NotImplementedError

    def append_knowledge_version(self, version: KnowledgeCardVersion) -> None:
        raise NotImplementedError

    def list_knowledge_versions(self, card_type: KnowledgeCardType, card_id: str) -> list[KnowledgeCardVersion]:
        raise NotImplementedError

    def save_knowledge_suggestion(self, suggestion: KnowledgeUpdateSuggestion) -> None:
        raise NotImplementedError

    def get_knowledge_suggestion(self, suggestion_id: str) -> KnowledgeUpdateSuggestion | None:
        raise NotImplementedError

    def list_knowledge_suggestions(
        self, status: str = "pending", limit: int = 100
    ) -> list[KnowledgeUpdateSuggestion]:
        raise NotImplementedError


def _dump_model(model) -> str:
    return model.model_dump_json()


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _matches_knowledge_card(
    item,
    q: str = "",
    industry: str = "",
    tag: str = "",
    freshness_status: FreshnessStatus | None = None,
) -> bool:
    if getattr(item, "deleted", False):
        return False
    if freshness_status and item.freshness_status != freshness_status:
        return False
    haystack_parts = [
        getattr(item, "name", ""),
        getattr(item, "title", ""),
        item.canonical_name,
        item.industry,
        item.market,
        " ".join(item.aliases),
        " ".join(item.tags),
    ]
    haystack = " ".join(part for part in haystack_parts if part).lower()
    if q and q.strip().lower() not in haystack:
        return False
    if industry and industry.strip().lower() not in (item.industry or "").lower():
        return False
    if tag and tag.strip().lower() not in [value.lower() for value in item.tags]:
        return False
    return True


def _touch_card(card):
    card.updated_at = utc_now()
    card.source_count = len(set(card.knowledge_source_ids)) or card.source_count
    return card


class SQLiteStorage(Storage):
    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def init(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                create table if not exists tasks (
                    id text primary key,
                    payload text not null,
                    status text not null,
                    updated_at text not null
                )
                """
            )
            db.execute(
                """
                create table if not exists reports (
                    task_id text primary key,
                    payload text not null,
                    generated_at text not null
                )
                """
            )
            db.execute(
                """
                create table if not exists events (
                    id text primary key,
                    task_id text not null,
                    payload text not null,
                    created_at text not null
                )
                """
            )
            db.execute("create index if not exists idx_events_task on events(task_id, created_at)")
            db.execute(
                """
                create table if not exists survey_designs (
                    task_id text primary key,
                    payload text not null,
                    created_at text not null,
                    updated_at text not null
                )
                """
            )
            db.execute(
                """
                create table if not exists survey_analyses (
                    task_id text primary key,
                    payload text not null,
                    created_at text not null,
                    updated_at text not null
                )
                """
            )
            db.execute(
                """
                create table if not exists knowledge_competitor_cards (
                    id text primary key,
                    name text not null,
                    industry text not null default '',
                    market text not null default '',
                    freshness_status text not null default 'unknown',
                    deleted integer not null default 0,
                    payload text not null,
                    updated_at text not null
                )
                """
            )
            db.execute(
                """
                create table if not exists knowledge_industry_cards (
                    id text primary key,
                    title text not null,
                    industry text not null default '',
                    market text not null default '',
                    freshness_status text not null default 'unknown',
                    deleted integer not null default 0,
                    payload text not null,
                    updated_at text not null
                )
                """
            )
            db.execute(
                """
                create table if not exists knowledge_sources (
                    id text primary key,
                    url text not null default '',
                    payload text not null,
                    captured_at text not null
                )
                """
            )
            db.execute(
                """
                create table if not exists knowledge_card_sources (
                    card_type text not null,
                    card_id text not null,
                    source_id text not null,
                    primary key(card_type, card_id, source_id)
                )
                """
            )
            db.execute(
                """
                create table if not exists knowledge_card_versions (
                    id text primary key,
                    card_type text not null,
                    card_id text not null,
                    payload text not null,
                    created_at text not null
                )
                """
            )
            db.execute(
                """
                create table if not exists knowledge_update_suggestions (
                    id text primary key,
                    status text not null,
                    card_type text not null,
                    target_card_id text,
                    payload text not null,
                    created_at text not null
                )
                """
            )
            db.execute("create index if not exists idx_knowledge_competitor_lookup on knowledge_competitor_cards(deleted, updated_at)")
            db.execute("create index if not exists idx_knowledge_industry_lookup on knowledge_industry_cards(deleted, updated_at)")
            db.execute("create index if not exists idx_knowledge_suggestions_status on knowledge_update_suggestions(status, created_at)")

    def save_task(self, task: AnalysisTask) -> None:
        task.updated_at = utc_now()
        with self._connect() as db:
            db.execute(
                """
                insert into tasks(id, payload, status, updated_at)
                values (?, ?, ?, ?)
                on conflict(id) do update set
                    payload=excluded.payload,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (task.id, _dump_model(task), task.status, task.updated_at.isoformat()),
            )

    def get_task(self, task_id: str) -> AnalysisTask | None:
        with self._connect() as db:
            row = db.execute("select payload from tasks where id = ?", (task_id,)).fetchone()
        return AnalysisTask.model_validate_json(row["payload"]) if row else None

    def list_tasks(self, limit: int = 50) -> list[AnalysisTask]:
        with self._connect() as db:
            rows = db.execute(
                "select payload from tasks order by updated_at desc limit ?", (limit,)
            ).fetchall()
        return [AnalysisTask.model_validate_json(row["payload"]) for row in rows]

    def save_report(self, report: CompetitorReport) -> None:
        with self._connect() as db:
            db.execute(
                """
                insert into reports(task_id, payload, generated_at)
                values (?, ?, ?)
                on conflict(task_id) do update set
                    payload=excluded.payload,
                    generated_at=excluded.generated_at
                """,
                (report.task_id, _dump_model(report), report.generated_at.isoformat()),
            )

    def get_report(self, task_id: str) -> CompetitorReport | None:
        with self._connect() as db:
            row = db.execute("select payload from reports where task_id = ?", (task_id,)).fetchone()
        return CompetitorReport.model_validate_json(row["payload"]) if row else None

    def append_event(self, event: AgentRunEvent) -> None:
        with self._connect() as db:
            db.execute(
                "insert or ignore into events(id, task_id, payload, created_at) values (?, ?, ?, ?)",
                (event.id, event.task_id, _dump_model(event), event.created_at.isoformat()),
            )

    def list_events(self, task_id: str) -> list[AgentRunEvent]:
        with self._connect() as db:
            rows = db.execute(
                "select payload from events where task_id = ? order by created_at asc", (task_id,)
            ).fetchall()
        return [AgentRunEvent.model_validate_json(row["payload"]) for row in rows]

    def save_survey_design(self, design: SurveyDesign) -> None:
        now = utc_now().isoformat()
        with self._connect() as db:
            db.execute(
                """
                insert into survey_designs(task_id, payload, created_at, updated_at)
                values (?, ?, ?, ?)
                on conflict(task_id) do update set
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (design.task_id, _dump_model(design), design.created_at.isoformat(), now),
            )

    def get_survey_design(self, task_id: str) -> SurveyDesign | None:
        with self._connect() as db:
            row = db.execute("select payload from survey_designs where task_id = ?", (task_id,)).fetchone()
        return SurveyDesign.model_validate_json(row["payload"]) if row else None

    def save_survey_analysis(self, analysis: SurveyAnalysisResult) -> None:
        now = utc_now().isoformat()
        with self._connect() as db:
            db.execute(
                """
                insert into survey_analyses(task_id, payload, created_at, updated_at)
                values (?, ?, ?, ?)
                on conflict(task_id) do update set
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (analysis.task_id, _dump_model(analysis), analysis.created_at.isoformat(), now),
            )

    def get_survey_analysis(self, task_id: str) -> SurveyAnalysisResult | None:
        with self._connect() as db:
            row = db.execute("select payload from survey_analyses where task_id = ?", (task_id,)).fetchone()
        return SurveyAnalysisResult.model_validate_json(row["payload"]) if row else None

    def save_knowledge_competitor_card(self, card: KnowledgeCompetitorCard, *, version_action: str = "save") -> None:
        before = self.get_knowledge_competitor_card(card.id)
        card = _touch_card(card)
        with self._connect() as db:
            db.execute(
                """
                insert into knowledge_competitor_cards(id, name, industry, market, freshness_status, deleted, payload, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    name=excluded.name,
                    industry=excluded.industry,
                    market=excluded.market,
                    freshness_status=excluded.freshness_status,
                    deleted=excluded.deleted,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    card.id,
                    card.name,
                    card.industry,
                    card.market,
                    card.freshness_status,
                    int(card.deleted),
                    _dump_model(card),
                    card.updated_at.isoformat(),
                ),
            )
            for source_id in card.knowledge_source_ids:
                db.execute(
                    "insert or ignore into knowledge_card_sources(card_type, card_id, source_id) values (?, ?, ?)",
                    ("competitor", card.id, source_id),
                )
        self.append_knowledge_version(
            KnowledgeCardVersion(
                card_type="competitor",
                card_id=card.id,
                action=version_action,
                before=before.model_dump(mode="json") if before else None,
                after=card.model_dump(mode="json"),
                origin_task_id=card.origin_task_id,
            )
        )

    def get_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        with self._connect() as db:
            row = db.execute("select payload from knowledge_competitor_cards where id = ?", (card_id,)).fetchone()
        return KnowledgeCompetitorCard.model_validate_json(row["payload"]) if row else None

    def list_knowledge_competitor_cards(
        self,
        q: str = "",
        industry: str = "",
        tag: str = "",
        freshness_status: FreshnessStatus | None = None,
        limit: int = 50,
    ) -> list[KnowledgeCompetitorCard]:
        with self._connect() as db:
            rows = db.execute(
                "select payload from knowledge_competitor_cards order by updated_at desc limit ?",
                (max(limit * 5, limit),),
            ).fetchall()
        cards = [KnowledgeCompetitorCard.model_validate_json(row["payload"]) for row in rows]
        return [card for card in cards if _matches_knowledge_card(card, q, industry, tag, freshness_status)][:limit]

    def delete_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        card = self.get_knowledge_competitor_card(card_id)
        if not card:
            return None
        card.deleted = True
        self.save_knowledge_competitor_card(card, version_action="delete")
        return card

    def save_knowledge_industry_card(self, card: KnowledgeIndustryCard, *, version_action: str = "save") -> None:
        before = self.get_knowledge_industry_card(card.id)
        card = _touch_card(card)
        with self._connect() as db:
            db.execute(
                """
                insert into knowledge_industry_cards(id, title, industry, market, freshness_status, deleted, payload, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    title=excluded.title,
                    industry=excluded.industry,
                    market=excluded.market,
                    freshness_status=excluded.freshness_status,
                    deleted=excluded.deleted,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    card.id,
                    card.title,
                    card.industry,
                    card.market,
                    card.freshness_status,
                    int(card.deleted),
                    _dump_model(card),
                    card.updated_at.isoformat(),
                ),
            )
            for source_id in card.knowledge_source_ids:
                db.execute(
                    "insert or ignore into knowledge_card_sources(card_type, card_id, source_id) values (?, ?, ?)",
                    ("industry", card.id, source_id),
                )
        self.append_knowledge_version(
            KnowledgeCardVersion(
                card_type="industry",
                card_id=card.id,
                action=version_action,
                before=before.model_dump(mode="json") if before else None,
                after=card.model_dump(mode="json"),
                origin_task_id=card.origin_task_id,
            )
        )

    def get_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        with self._connect() as db:
            row = db.execute("select payload from knowledge_industry_cards where id = ?", (card_id,)).fetchone()
        return KnowledgeIndustryCard.model_validate_json(row["payload"]) if row else None

    def list_knowledge_industry_cards(
        self,
        q: str = "",
        industry: str = "",
        tag: str = "",
        freshness_status: FreshnessStatus | None = None,
        limit: int = 50,
    ) -> list[KnowledgeIndustryCard]:
        with self._connect() as db:
            rows = db.execute(
                "select payload from knowledge_industry_cards order by updated_at desc limit ?",
                (max(limit * 5, limit),),
            ).fetchall()
        cards = [KnowledgeIndustryCard.model_validate_json(row["payload"]) for row in rows]
        return [card for card in cards if _matches_knowledge_card(card, q, industry, tag, freshness_status)][:limit]

    def delete_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        card = self.get_knowledge_industry_card(card_id)
        if not card:
            return None
        card.deleted = True
        self.save_knowledge_industry_card(card, version_action="delete")
        return card

    def save_knowledge_source(self, source: KnowledgeSource) -> None:
        with self._connect() as db:
            db.execute(
                """
                insert into knowledge_sources(id, url, payload, captured_at)
                values (?, ?, ?, ?)
                on conflict(id) do update set
                    url=excluded.url,
                    payload=excluded.payload,
                    captured_at=excluded.captured_at
                """,
                (source.id, source.url, _dump_model(source), source.captured_at.isoformat()),
            )

    def get_knowledge_source(self, source_id: str) -> KnowledgeSource | None:
        with self._connect() as db:
            row = db.execute("select payload from knowledge_sources where id = ?", (source_id,)).fetchone()
        return KnowledgeSource.model_validate_json(row["payload"]) if row else None

    def list_knowledge_sources(self, source_ids: list[str] | None = None, limit: int = 100) -> list[KnowledgeSource]:
        with self._connect() as db:
            if source_ids:
                rows = [
                    row
                    for source_id in source_ids
                    if (row := db.execute("select payload from knowledge_sources where id = ?", (source_id,)).fetchone())
                ]
            else:
                rows = db.execute("select payload from knowledge_sources order by captured_at desc limit ?", (limit,)).fetchall()
        return [KnowledgeSource.model_validate_json(row["payload"]) for row in rows][:limit]

    def append_knowledge_version(self, version: KnowledgeCardVersion) -> None:
        with self._connect() as db:
            db.execute(
                "insert or ignore into knowledge_card_versions(id, card_type, card_id, payload, created_at) values (?, ?, ?, ?, ?)",
                (version.id, version.card_type, version.card_id, _dump_model(version), version.created_at.isoformat()),
            )

    def list_knowledge_versions(self, card_type: KnowledgeCardType, card_id: str) -> list[KnowledgeCardVersion]:
        with self._connect() as db:
            rows = db.execute(
                "select payload from knowledge_card_versions where card_type = ? and card_id = ? order by created_at desc",
                (card_type, card_id),
            ).fetchall()
        return [KnowledgeCardVersion.model_validate_json(row["payload"]) for row in rows]

    def save_knowledge_suggestion(self, suggestion: KnowledgeUpdateSuggestion) -> None:
        with self._connect() as db:
            db.execute(
                """
                insert into knowledge_update_suggestions(id, status, card_type, target_card_id, payload, created_at)
                values (?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    status=excluded.status,
                    card_type=excluded.card_type,
                    target_card_id=excluded.target_card_id,
                    payload=excluded.payload,
                    created_at=excluded.created_at
                """,
                (
                    suggestion.id,
                    suggestion.status,
                    suggestion.card_type,
                    suggestion.target_card_id,
                    _dump_model(suggestion),
                    suggestion.created_at.isoformat(),
                ),
            )

    def get_knowledge_suggestion(self, suggestion_id: str) -> KnowledgeUpdateSuggestion | None:
        with self._connect() as db:
            row = db.execute("select payload from knowledge_update_suggestions where id = ?", (suggestion_id,)).fetchone()
        return KnowledgeUpdateSuggestion.model_validate_json(row["payload"]) if row else None

    def list_knowledge_suggestions(
        self, status: str = "pending", limit: int = 100
    ) -> list[KnowledgeUpdateSuggestion]:
        with self._connect() as db:
            if status == "all":
                rows = db.execute(
                    "select payload from knowledge_update_suggestions order by created_at desc limit ?", (limit,)
                ).fetchall()
            else:
                rows = db.execute(
                    "select payload from knowledge_update_suggestions where status = ? order by created_at desc limit ?",
                    (status, limit),
                ).fetchall()
        return [KnowledgeUpdateSuggestion.model_validate_json(row["payload"]) for row in rows]


class JSONStorage(Storage):
    def __init__(self, directory: Path):
        self.directory = directory

    def init(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "tasks").mkdir(exist_ok=True)
        (self.directory / "reports").mkdir(exist_ok=True)
        (self.directory / "events").mkdir(exist_ok=True)
        (self.directory / "survey_designs").mkdir(exist_ok=True)
        (self.directory / "survey_analyses").mkdir(exist_ok=True)
        (self.directory / "knowledge_competitor_cards").mkdir(exist_ok=True)
        (self.directory / "knowledge_industry_cards").mkdir(exist_ok=True)
        (self.directory / "knowledge_sources").mkdir(exist_ok=True)
        (self.directory / "knowledge_versions").mkdir(exist_ok=True)
        (self.directory / "knowledge_suggestions").mkdir(exist_ok=True)

    def save_task(self, task: AnalysisTask) -> None:
        task.updated_at = utc_now()
        _write_json(self.directory / "tasks" / f"{task.id}.json", task.model_dump(mode="json"))

    def get_task(self, task_id: str) -> AnalysisTask | None:
        path = self.directory / "tasks" / f"{task_id}.json"
        if not path.exists():
            return None
        return AnalysisTask.model_validate(_load_json(path, {}))

    def list_tasks(self, limit: int = 50) -> list[AnalysisTask]:
        tasks: list[AnalysisTask] = []
        for path in (self.directory / "tasks").glob("*.json"):
            tasks.append(AnalysisTask.model_validate(_load_json(path, {})))
        return sorted(tasks, key=lambda item: item.updated_at, reverse=True)[:limit]

    def save_report(self, report: CompetitorReport) -> None:
        _write_json(self.directory / "reports" / f"{report.task_id}.json", report.model_dump(mode="json"))

    def get_report(self, task_id: str) -> CompetitorReport | None:
        path = self.directory / "reports" / f"{task_id}.json"
        if not path.exists():
            return None
        return CompetitorReport.model_validate(_load_json(path, {}))

    def append_event(self, event: AgentRunEvent) -> None:
        path = self.directory / "events" / f"{event.task_id}.json"
        events = _load_json(path, [])
        events.append(event.model_dump(mode="json"))
        _write_json(path, events)

    def list_events(self, task_id: str) -> list[AgentRunEvent]:
        path = self.directory / "events" / f"{task_id}.json"
        return [AgentRunEvent.model_validate(item) for item in _load_json(path, [])]

    def save_survey_design(self, design: SurveyDesign) -> None:
        _write_json(self.directory / "survey_designs" / f"{design.task_id}.json", design.model_dump(mode="json"))

    def get_survey_design(self, task_id: str) -> SurveyDesign | None:
        path = self.directory / "survey_designs" / f"{task_id}.json"
        return SurveyDesign.model_validate(_load_json(path, {})) if path.exists() else None

    def save_survey_analysis(self, analysis: SurveyAnalysisResult) -> None:
        _write_json(self.directory / "survey_analyses" / f"{analysis.task_id}.json", analysis.model_dump(mode="json"))

    def get_survey_analysis(self, task_id: str) -> SurveyAnalysisResult | None:
        path = self.directory / "survey_analyses" / f"{task_id}.json"
        return SurveyAnalysisResult.model_validate(_load_json(path, {})) if path.exists() else None

    def save_knowledge_competitor_card(self, card: KnowledgeCompetitorCard, *, version_action: str = "save") -> None:
        before = self.get_knowledge_competitor_card(card.id)
        card = _touch_card(card)
        _write_json(self.directory / "knowledge_competitor_cards" / f"{card.id}.json", card.model_dump(mode="json"))
        self.append_knowledge_version(
            KnowledgeCardVersion(
                card_type="competitor",
                card_id=card.id,
                action=version_action,
                before=before.model_dump(mode="json") if before else None,
                after=card.model_dump(mode="json"),
                origin_task_id=card.origin_task_id,
            )
        )

    def get_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        path = self.directory / "knowledge_competitor_cards" / f"{card_id}.json"
        return KnowledgeCompetitorCard.model_validate(_load_json(path, {})) if path.exists() else None

    def list_knowledge_competitor_cards(
        self,
        q: str = "",
        industry: str = "",
        tag: str = "",
        freshness_status: FreshnessStatus | None = None,
        limit: int = 50,
    ) -> list[KnowledgeCompetitorCard]:
        cards = [
            KnowledgeCompetitorCard.model_validate(_load_json(path, {}))
            for path in (self.directory / "knowledge_competitor_cards").glob("*.json")
        ]
        cards = sorted(cards, key=lambda item: item.updated_at, reverse=True)
        return [card for card in cards if _matches_knowledge_card(card, q, industry, tag, freshness_status)][:limit]

    def delete_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        card = self.get_knowledge_competitor_card(card_id)
        if not card:
            return None
        card.deleted = True
        self.save_knowledge_competitor_card(card, version_action="delete")
        return card

    def save_knowledge_industry_card(self, card: KnowledgeIndustryCard, *, version_action: str = "save") -> None:
        before = self.get_knowledge_industry_card(card.id)
        card = _touch_card(card)
        _write_json(self.directory / "knowledge_industry_cards" / f"{card.id}.json", card.model_dump(mode="json"))
        self.append_knowledge_version(
            KnowledgeCardVersion(
                card_type="industry",
                card_id=card.id,
                action=version_action,
                before=before.model_dump(mode="json") if before else None,
                after=card.model_dump(mode="json"),
                origin_task_id=card.origin_task_id,
            )
        )

    def get_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        path = self.directory / "knowledge_industry_cards" / f"{card_id}.json"
        return KnowledgeIndustryCard.model_validate(_load_json(path, {})) if path.exists() else None

    def list_knowledge_industry_cards(
        self,
        q: str = "",
        industry: str = "",
        tag: str = "",
        freshness_status: FreshnessStatus | None = None,
        limit: int = 50,
    ) -> list[KnowledgeIndustryCard]:
        cards = [
            KnowledgeIndustryCard.model_validate(_load_json(path, {}))
            for path in (self.directory / "knowledge_industry_cards").glob("*.json")
        ]
        cards = sorted(cards, key=lambda item: item.updated_at, reverse=True)
        return [card for card in cards if _matches_knowledge_card(card, q, industry, tag, freshness_status)][:limit]

    def delete_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        card = self.get_knowledge_industry_card(card_id)
        if not card:
            return None
        card.deleted = True
        self.save_knowledge_industry_card(card, version_action="delete")
        return card

    def save_knowledge_source(self, source: KnowledgeSource) -> None:
        _write_json(self.directory / "knowledge_sources" / f"{source.id}.json", source.model_dump(mode="json"))

    def get_knowledge_source(self, source_id: str) -> KnowledgeSource | None:
        path = self.directory / "knowledge_sources" / f"{source_id}.json"
        return KnowledgeSource.model_validate(_load_json(path, {})) if path.exists() else None

    def list_knowledge_sources(self, source_ids: list[str] | None = None, limit: int = 100) -> list[KnowledgeSource]:
        if source_ids:
            sources = [source for source_id in source_ids if (source := self.get_knowledge_source(source_id))]
        else:
            sources = [
                KnowledgeSource.model_validate(_load_json(path, {}))
                for path in (self.directory / "knowledge_sources").glob("*.json")
            ]
        return sorted(sources, key=lambda item: item.captured_at, reverse=True)[:limit]

    def append_knowledge_version(self, version: KnowledgeCardVersion) -> None:
        path = self.directory / "knowledge_versions" / f"{version.card_type}_{version.card_id}.json"
        versions = _load_json(path, [])
        versions.append(version.model_dump(mode="json"))
        _write_json(path, versions)

    def list_knowledge_versions(self, card_type: KnowledgeCardType, card_id: str) -> list[KnowledgeCardVersion]:
        path = self.directory / "knowledge_versions" / f"{card_type}_{card_id}.json"
        versions = [KnowledgeCardVersion.model_validate(item) for item in _load_json(path, [])]
        return sorted(versions, key=lambda item: item.created_at, reverse=True)

    def save_knowledge_suggestion(self, suggestion: KnowledgeUpdateSuggestion) -> None:
        _write_json(self.directory / "knowledge_suggestions" / f"{suggestion.id}.json", suggestion.model_dump(mode="json"))

    def get_knowledge_suggestion(self, suggestion_id: str) -> KnowledgeUpdateSuggestion | None:
        path = self.directory / "knowledge_suggestions" / f"{suggestion_id}.json"
        return KnowledgeUpdateSuggestion.model_validate(_load_json(path, {})) if path.exists() else None

    def list_knowledge_suggestions(
        self, status: str = "pending", limit: int = 100
    ) -> list[KnowledgeUpdateSuggestion]:
        suggestions = [
            KnowledgeUpdateSuggestion.model_validate(_load_json(path, {}))
            for path in (self.directory / "knowledge_suggestions").glob("*.json")
        ]
        if status != "all":
            suggestions = [item for item in suggestions if item.status == status]
        return sorted(suggestions, key=lambda item: item.created_at, reverse=True)[:limit]


class PostgresStorage(Storage):
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    def init(self) -> None:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    create table if not exists tasks (
                        id text primary key,
                        payload jsonb not null,
                        status text not null,
                        updated_at timestamptz not null
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists reports (
                        task_id text primary key,
                        payload jsonb not null,
                        generated_at timestamptz not null
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists events (
                        id text primary key,
                        task_id text not null,
                        payload jsonb not null,
                        created_at timestamptz not null
                    )
                    """
                )
                cur.execute("create index if not exists idx_events_task on events(task_id, created_at)")
                cur.execute(
                    """
                    create table if not exists survey_designs (
                        task_id text primary key,
                        payload jsonb not null,
                        created_at timestamptz not null,
                        updated_at timestamptz not null
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists survey_analyses (
                        task_id text primary key,
                        payload jsonb not null,
                        created_at timestamptz not null,
                        updated_at timestamptz not null
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists knowledge_competitor_cards (
                        id text primary key,
                        name text not null,
                        industry text not null default '',
                        market text not null default '',
                        freshness_status text not null default 'unknown',
                        deleted boolean not null default false,
                        payload jsonb not null,
                        updated_at timestamptz not null
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists knowledge_industry_cards (
                        id text primary key,
                        title text not null,
                        industry text not null default '',
                        market text not null default '',
                        freshness_status text not null default 'unknown',
                        deleted boolean not null default false,
                        payload jsonb not null,
                        updated_at timestamptz not null
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists knowledge_sources (
                        id text primary key,
                        url text not null default '',
                        payload jsonb not null,
                        captured_at timestamptz not null
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists knowledge_card_sources (
                        card_type text not null,
                        card_id text not null,
                        source_id text not null,
                        primary key(card_type, card_id, source_id)
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists knowledge_card_versions (
                        id text primary key,
                        card_type text not null,
                        card_id text not null,
                        payload jsonb not null,
                        created_at timestamptz not null
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists knowledge_update_suggestions (
                        id text primary key,
                        status text not null,
                        card_type text not null,
                        target_card_id text,
                        payload jsonb not null,
                        created_at timestamptz not null
                    )
                    """
                )

    def save_task(self, task: AnalysisTask) -> None:
        from psycopg.types.json import Jsonb

        task.updated_at = utc_now()
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into tasks(id, payload, status, updated_at)
                    values (%s, %s, %s, %s)
                    on conflict(id) do update set
                        payload=excluded.payload,
                        status=excluded.status,
                        updated_at=excluded.updated_at
                    """,
                    (task.id, Jsonb(task.model_dump(mode="json")), task.status, task.updated_at),
                )

    def get_task(self, task_id: str) -> AnalysisTask | None:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from tasks where id = %s", (task_id,))
                row = cur.fetchone()
        return AnalysisTask.model_validate(row[0]) if row else None

    def list_tasks(self, limit: int = 50) -> list[AnalysisTask]:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from tasks order by updated_at desc limit %s", (limit,))
                rows = cur.fetchall()
        return [AnalysisTask.model_validate(row[0]) for row in rows]

    def save_report(self, report: CompetitorReport) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into reports(task_id, payload, generated_at)
                    values (%s, %s, %s)
                    on conflict(task_id) do update set
                        payload=excluded.payload,
                        generated_at=excluded.generated_at
                    """,
                    (report.task_id, Jsonb(report.model_dump(mode="json")), report.generated_at),
                )

    def get_report(self, task_id: str) -> CompetitorReport | None:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from reports where task_id = %s", (task_id,))
                row = cur.fetchone()
        return CompetitorReport.model_validate(row[0]) if row else None

    def append_event(self, event: AgentRunEvent) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into events(id, task_id, payload, created_at)
                    values (%s, %s, %s, %s)
                    on conflict(id) do nothing
                    """,
                    (
                        event.id,
                        event.task_id,
                        Jsonb(event.model_dump(mode="json")),
                        event.created_at,
                    ),
                )

    def list_events(self, task_id: str) -> list[AgentRunEvent]:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    "select payload from events where task_id = %s order by created_at asc",
                    (task_id,),
                )
                rows = cur.fetchall()
        return [AgentRunEvent.model_validate(row[0]) for row in rows]

    def save_survey_design(self, design: SurveyDesign) -> None:
        from psycopg.types.json import Jsonb

        now = utc_now()
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into survey_designs(task_id, payload, created_at, updated_at)
                    values (%s, %s, %s, %s)
                    on conflict(task_id) do update set
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (design.task_id, Jsonb(design.model_dump(mode="json")), design.created_at, now),
                )

    def get_survey_design(self, task_id: str) -> SurveyDesign | None:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from survey_designs where task_id = %s", (task_id,))
                row = cur.fetchone()
        return SurveyDesign.model_validate(row[0]) if row else None

    def save_survey_analysis(self, analysis: SurveyAnalysisResult) -> None:
        from psycopg.types.json import Jsonb

        now = utc_now()
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into survey_analyses(task_id, payload, created_at, updated_at)
                    values (%s, %s, %s, %s)
                    on conflict(task_id) do update set
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (analysis.task_id, Jsonb(analysis.model_dump(mode="json")), analysis.created_at, now),
                )

    def get_survey_analysis(self, task_id: str) -> SurveyAnalysisResult | None:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from survey_analyses where task_id = %s", (task_id,))
                row = cur.fetchone()
        return SurveyAnalysisResult.model_validate(row[0]) if row else None

    def save_knowledge_competitor_card(self, card: KnowledgeCompetitorCard, *, version_action: str = "save") -> None:
        from psycopg.types.json import Jsonb

        before = self.get_knowledge_competitor_card(card.id)
        card = _touch_card(card)
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into knowledge_competitor_cards(id, name, industry, market, freshness_status, deleted, payload, updated_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict(id) do update set
                        name=excluded.name,
                        industry=excluded.industry,
                        market=excluded.market,
                        freshness_status=excluded.freshness_status,
                        deleted=excluded.deleted,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (
                        card.id,
                        card.name,
                        card.industry,
                        card.market,
                        card.freshness_status,
                        card.deleted,
                        Jsonb(card.model_dump(mode="json")),
                        card.updated_at,
                    ),
                )
                for source_id in card.knowledge_source_ids:
                    cur.execute(
                        """
                        insert into knowledge_card_sources(card_type, card_id, source_id)
                        values (%s, %s, %s)
                        on conflict do nothing
                        """,
                        ("competitor", card.id, source_id),
                    )
        self.append_knowledge_version(
            KnowledgeCardVersion(
                card_type="competitor",
                card_id=card.id,
                action=version_action,
                before=before.model_dump(mode="json") if before else None,
                after=card.model_dump(mode="json"),
                origin_task_id=card.origin_task_id,
            )
        )

    def get_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from knowledge_competitor_cards where id = %s", (card_id,))
                row = cur.fetchone()
        return KnowledgeCompetitorCard.model_validate(row[0]) if row else None

    def list_knowledge_competitor_cards(
        self,
        q: str = "",
        industry: str = "",
        tag: str = "",
        freshness_status: FreshnessStatus | None = None,
        limit: int = 50,
    ) -> list[KnowledgeCompetitorCard]:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from knowledge_competitor_cards order by updated_at desc limit %s", (max(limit * 5, limit),))
                rows = cur.fetchall()
        cards = [KnowledgeCompetitorCard.model_validate(row[0]) for row in rows]
        return [card for card in cards if _matches_knowledge_card(card, q, industry, tag, freshness_status)][:limit]

    def delete_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        card = self.get_knowledge_competitor_card(card_id)
        if not card:
            return None
        card.deleted = True
        self.save_knowledge_competitor_card(card, version_action="delete")
        return card

    def save_knowledge_industry_card(self, card: KnowledgeIndustryCard, *, version_action: str = "save") -> None:
        from psycopg.types.json import Jsonb

        before = self.get_knowledge_industry_card(card.id)
        card = _touch_card(card)
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into knowledge_industry_cards(id, title, industry, market, freshness_status, deleted, payload, updated_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict(id) do update set
                        title=excluded.title,
                        industry=excluded.industry,
                        market=excluded.market,
                        freshness_status=excluded.freshness_status,
                        deleted=excluded.deleted,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (
                        card.id,
                        card.title,
                        card.industry,
                        card.market,
                        card.freshness_status,
                        card.deleted,
                        Jsonb(card.model_dump(mode="json")),
                        card.updated_at,
                    ),
                )
                for source_id in card.knowledge_source_ids:
                    cur.execute(
                        """
                        insert into knowledge_card_sources(card_type, card_id, source_id)
                        values (%s, %s, %s)
                        on conflict do nothing
                        """,
                        ("industry", card.id, source_id),
                    )
        self.append_knowledge_version(
            KnowledgeCardVersion(
                card_type="industry",
                card_id=card.id,
                action=version_action,
                before=before.model_dump(mode="json") if before else None,
                after=card.model_dump(mode="json"),
                origin_task_id=card.origin_task_id,
            )
        )

    def get_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from knowledge_industry_cards where id = %s", (card_id,))
                row = cur.fetchone()
        return KnowledgeIndustryCard.model_validate(row[0]) if row else None

    def list_knowledge_industry_cards(
        self,
        q: str = "",
        industry: str = "",
        tag: str = "",
        freshness_status: FreshnessStatus | None = None,
        limit: int = 50,
    ) -> list[KnowledgeIndustryCard]:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from knowledge_industry_cards order by updated_at desc limit %s", (max(limit * 5, limit),))
                rows = cur.fetchall()
        cards = [KnowledgeIndustryCard.model_validate(row[0]) for row in rows]
        return [card for card in cards if _matches_knowledge_card(card, q, industry, tag, freshness_status)][:limit]

    def delete_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        card = self.get_knowledge_industry_card(card_id)
        if not card:
            return None
        card.deleted = True
        self.save_knowledge_industry_card(card, version_action="delete")
        return card

    def save_knowledge_source(self, source: KnowledgeSource) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into knowledge_sources(id, url, payload, captured_at)
                    values (%s, %s, %s, %s)
                    on conflict(id) do update set
                        url=excluded.url,
                        payload=excluded.payload,
                        captured_at=excluded.captured_at
                    """,
                    (source.id, source.url, Jsonb(source.model_dump(mode="json")), source.captured_at),
                )

    def get_knowledge_source(self, source_id: str) -> KnowledgeSource | None:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from knowledge_sources where id = %s", (source_id,))
                row = cur.fetchone()
        return KnowledgeSource.model_validate(row[0]) if row else None

    def list_knowledge_sources(self, source_ids: list[str] | None = None, limit: int = 100) -> list[KnowledgeSource]:
        if source_ids:
            return [source for source_id in source_ids if (source := self.get_knowledge_source(source_id))][:limit]
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from knowledge_sources order by captured_at desc limit %s", (limit,))
                rows = cur.fetchall()
        return [KnowledgeSource.model_validate(row[0]) for row in rows]

    def append_knowledge_version(self, version: KnowledgeCardVersion) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into knowledge_card_versions(id, card_type, card_id, payload, created_at)
                    values (%s, %s, %s, %s, %s)
                    on conflict(id) do nothing
                    """,
                    (version.id, version.card_type, version.card_id, Jsonb(version.model_dump(mode="json")), version.created_at),
                )

    def list_knowledge_versions(self, card_type: KnowledgeCardType, card_id: str) -> list[KnowledgeCardVersion]:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    "select payload from knowledge_card_versions where card_type = %s and card_id = %s order by created_at desc",
                    (card_type, card_id),
                )
                rows = cur.fetchall()
        return [KnowledgeCardVersion.model_validate(row[0]) for row in rows]

    def save_knowledge_suggestion(self, suggestion: KnowledgeUpdateSuggestion) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute(
                    """
                    insert into knowledge_update_suggestions(id, status, card_type, target_card_id, payload, created_at)
                    values (%s, %s, %s, %s, %s, %s)
                    on conflict(id) do update set
                        status=excluded.status,
                        card_type=excluded.card_type,
                        target_card_id=excluded.target_card_id,
                        payload=excluded.payload,
                        created_at=excluded.created_at
                    """,
                    (
                        suggestion.id,
                        suggestion.status,
                        suggestion.card_type,
                        suggestion.target_card_id,
                        Jsonb(suggestion.model_dump(mode="json")),
                        suggestion.created_at,
                    ),
                )

    def get_knowledge_suggestion(self, suggestion_id: str) -> KnowledgeUpdateSuggestion | None:
        with self._connect() as db:
            with db.cursor() as cur:
                cur.execute("select payload from knowledge_update_suggestions where id = %s", (suggestion_id,))
                row = cur.fetchone()
        return KnowledgeUpdateSuggestion.model_validate(row[0]) if row else None

    def list_knowledge_suggestions(
        self, status: str = "pending", limit: int = 100
    ) -> list[KnowledgeUpdateSuggestion]:
        with self._connect() as db:
            with db.cursor() as cur:
                if status == "all":
                    cur.execute("select payload from knowledge_update_suggestions order by created_at desc limit %s", (limit,))
                else:
                    cur.execute(
                        "select payload from knowledge_update_suggestions where status = %s order by created_at desc limit %s",
                        (status, limit),
                    )
                rows = cur.fetchall()
        return [KnowledgeUpdateSuggestion.model_validate(row[0]) for row in rows]


def create_storage(settings: Settings) -> Storage:
    stores: dict[str, Storage] = {
        "sqlite": SQLiteStorage(settings.sqlite_path),
        "json": JSONStorage(settings.json_storage_dir),
        "postgres": PostgresStorage(settings.postgres_dsn),
    }
    storage = stores[settings.storage_backend]
    storage.init()
    return storage


def flatten_evidence_ids(items: Iterable[CompetitorReport]) -> set[str]:
    ids: set[str] = set()
    for report in items:
        ids.update(evidence.id for evidence in report.evidence)
    return ids
