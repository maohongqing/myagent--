from __future__ import annotations

import sys
from datetime import timedelta
from types import ModuleType

from myagent.backend.app.config import Settings
from myagent.backend.app.models import AgentRunEvent, AnalysisTask, CreateTaskRequest, utc_now
from myagent.backend.app.storage import JSONStorage, PostgresStorage, SQLiteStorage, create_storage


def test_sqlite_database_storage_crud_and_event_deduplication(tmp_path):
    storage = SQLiteStorage(tmp_path / "app.db")
    storage.init()
    task = AnalysisTask(id="task_sqlite", request=CreateTaskRequest(target_product="Acme"))
    event = AgentRunEvent(
        id="evt_same",
        task_id=task.id,
        node="collector_agent",
        status="running",
        message="collecting",
        details={"kind": "collector", "source_count": 2},
        created_at=utc_now() - timedelta(seconds=1),
    )
    second_event = AgentRunEvent(
        id="evt_second",
        task_id=task.id,
        node="qa_agent",
        status="completed",
        message="done",
        created_at=utc_now(),
    )

    from conftest import make_report

    report = make_report(task.id, task.request)
    storage.save_task(task)
    task.status = "completed"
    storage.save_task(task)
    storage.save_report(report)
    storage.append_event(event)
    storage.append_event(event)
    storage.append_event(second_event)

    assert storage.get_task(task.id).status == "completed"
    assert storage.get_report(task.id).task_id == task.id
    assert [item.id for item in storage.list_tasks(limit=1)] == [task.id]
    assert [item.id for item in storage.list_events(task.id)] == ["evt_same", "evt_second"]
    assert storage.list_events(task.id)[0].details["source_count"] == 2


def test_json_database_storage_crud_and_event_order(tmp_path):
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(id="task_json", request=CreateTaskRequest(target_product="Acme"))
    first_event = AgentRunEvent(
        task_id=task.id,
        node="collector_agent",
        status="running",
        message="first",
        details={"kind": "collector", "source_count": 1},
    )
    second_event = AgentRunEvent(task_id=task.id, node="writer_agent", status="completed", message="second")

    from conftest import make_report

    report = make_report(task.id, task.request)
    storage.save_task(task)
    storage.save_report(report)
    storage.append_event(first_event)
    storage.append_event(second_event)

    assert (tmp_path / "json" / "tasks").exists()
    assert storage.get_task(task.id).id == task.id
    assert storage.get_report(task.id).target_product == "Acme"
    assert [event.message for event in storage.list_events(task.id)] == ["first", "second"]
    assert storage.list_events(task.id)[0].details["source_count"] == 1
    assert storage.list_tasks(limit=1)[0].id == task.id


def test_create_storage_initializes_selected_backend(tmp_path):
    sqlite_settings = Settings(storage_backend="sqlite", sqlite_path=tmp_path / "selected.db")
    sqlite_storage = create_storage(sqlite_settings)

    assert isinstance(sqlite_storage, SQLiteStorage)
    assert (tmp_path / "selected.db").exists()

    json_settings = Settings(storage_backend="json", json_storage_dir=tmp_path / "selected_json")
    json_storage = create_storage(json_settings)

    assert isinstance(json_storage, JSONStorage)
    assert (tmp_path / "selected_json" / "tasks").exists()


def test_postgres_database_storage_uses_sql_without_real_database(monkeypatch):
    fake_db = FakePostgresDatabase()

    fake_psycopg = ModuleType("psycopg")
    fake_psycopg.__path__ = []
    fake_psycopg.connect = lambda dsn: FakeConnection(fake_db, dsn)

    fake_types = ModuleType("psycopg.types")
    fake_json = ModuleType("psycopg.types.json")
    fake_json.Jsonb = FakeJsonb
    fake_types.json = fake_json
    fake_psycopg.types = fake_types

    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.types", fake_types)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", fake_json)

    storage = PostgresStorage("postgresql://unit-test")
    task = AnalysisTask(id="task_pg", request=CreateTaskRequest(target_product="Acme"))
    event = AgentRunEvent(
        id="evt_pg",
        task_id=task.id,
        node="qa_agent",
        status="completed",
        message="qa ok",
        details={"kind": "qa", "score": 1.0},
    )

    from conftest import make_report

    report = make_report(task.id, task.request)
    storage.init()
    storage.save_task(task)
    storage.save_report(report)
    storage.append_event(event)
    storage.append_event(event)

    assert fake_db.connected_dsns
    assert all(dsn == "postgresql://unit-test" for dsn in fake_db.connected_dsns)
    assert storage.get_task(task.id).id == task.id
    assert storage.list_tasks(limit=5)[0].id == task.id
    assert storage.get_report(task.id).task_id == task.id
    assert [item.id for item in storage.list_events(task.id)] == ["evt_pg"]
    assert storage.list_events(task.id)[0].details["score"] == 1.0
    assert any("create table if not exists tasks" in sql.lower() for sql in fake_db.executed_sql)


class FakeJsonb:
    def __init__(self, payload):
        self.payload = payload


class FakePostgresDatabase:
    def __init__(self) -> None:
        self.tasks = {}
        self.reports = {}
        self.events = {}
        self.executed_sql: list[str] = []
        self.connected_dsns: list[str] = []


class FakeConnection:
    def __init__(self, db: FakePostgresDatabase, dsn: str):
        self.db = db
        self.db.connected_dsns.append(dsn)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def cursor(self):
        return FakeCursor(self.db)


class FakeCursor:
    def __init__(self, db: FakePostgresDatabase):
        self.db = db
        self.one = None
        self.many = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, sql: str, params=None):
        normalized = " ".join(sql.lower().split())
        self.db.executed_sql.append(sql)
        params = params or ()

        if normalized.startswith("insert into tasks"):
            task_id, payload, status, updated_at = params
            self.db.tasks[task_id] = {
                "payload": unwrap_jsonb(payload),
                "status": status,
                "updated_at": updated_at,
            }
        elif normalized.startswith("select payload from tasks where id"):
            row = self.db.tasks.get(params[0])
            self.one = (row["payload"],) if row else None
        elif normalized.startswith("select payload from tasks order by"):
            limit = params[0]
            rows = sorted(self.db.tasks.values(), key=lambda item: item["updated_at"], reverse=True)[:limit]
            self.many = [(row["payload"],) for row in rows]
        elif normalized.startswith("insert into reports"):
            task_id, payload, generated_at = params
            self.db.reports[task_id] = {"payload": unwrap_jsonb(payload), "generated_at": generated_at}
        elif normalized.startswith("select payload from reports"):
            row = self.db.reports.get(params[0])
            self.one = (row["payload"],) if row else None
        elif normalized.startswith("insert into events"):
            event_id, task_id, payload, created_at = params
            self.db.events.setdefault(event_id, {"task_id": task_id, "payload": unwrap_jsonb(payload), "created_at": created_at})
        elif normalized.startswith("select payload from events"):
            task_id = params[0]
            rows = [row for row in self.db.events.values() if row["task_id"] == task_id]
            rows = sorted(rows, key=lambda item: item["created_at"])
            self.many = [(row["payload"],) for row in rows]

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.many


def unwrap_jsonb(value):
    return value.payload if isinstance(value, FakeJsonb) else value
