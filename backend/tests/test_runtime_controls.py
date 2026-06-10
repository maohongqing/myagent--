from __future__ import annotations

import json
from datetime import timedelta

from fastapi.testclient import TestClient

from myagent.backend.app import main
from myagent.backend.app.config import Settings
from myagent.backend.app.models import AnalysisTask, CreateTaskRequest, DecisionOption, DecisionRequest, utc_now
from myagent.backend.tests.conftest import MemoryStorage


def _client(tmp_path, storage: MemoryStorage | None = None) -> TestClient:
    main.state.settings = Settings(sqlite_path=tmp_path / "data" / "app.db", json_storage_dir=tmp_path / "json")
    main.state.storage = storage or MemoryStorage()
    main.state.storage.init()
    main.state.event_bus = main.EventBus(main.state.storage)
    return TestClient(main.app)


def test_cancel_endpoint_sets_cancelling(tmp_path):
    storage = MemoryStorage()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"), status="running")
    storage.save_task(task)
    client = _client(tmp_path, storage)

    response = client.post(f"/api/tasks/{task.id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"
    assert storage.get_task(task.id).status == "cancelling"


def test_decision_continue_without_llm_requeues_task(tmp_path, monkeypatch):
    storage = MemoryStorage()
    decision = DecisionRequest(
        node="search_agent",
        question="LLM failed",
        reason="connection failed",
        options=[DecisionOption(label="规则继续", value="continue_without_llm")],
    )
    task = AnalysisTask(
        request=CreateTaskRequest(target_product="Acme"),
        status="waiting_decision",
        pending_decision=decision,
    )
    storage.save_task(task)
    client = _client(tmp_path, storage)
    monkeypatch.setattr(main, "run_task", lambda task_id: None)

    response = client.post(f"/api/tasks/{task.id}/decisions", json={"action": "continue_without_llm"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["request"]["llm_failure_policy"] == "continue_without_llm"


def test_resume_failed_task_requeues_same_task(tmp_path, monkeypatch):
    storage = MemoryStorage()
    task = AnalysisTask(
        request=CreateTaskRequest(target_product="Acme"),
        status="failed",
        current_node=None,
        error="network interrupted",
        completed_at=utc_now(),
    )
    storage.save_task(task)
    client = _client(tmp_path, storage)
    scheduled: list[str] = []
    monkeypatch.setattr(main, "run_task", lambda task_id: scheduled.append(task_id))

    response = client.post(f"/api/tasks/{task.id}/resume")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == task.id
    assert payload["status"] == "queued"
    assert payload["error"] is None
    assert payload["completed_at"] is None
    assert scheduled == [task.id]


def test_resume_latest_skips_completed_and_uses_most_recent_unfinished(tmp_path, monkeypatch):
    storage = MemoryStorage()
    older = AnalysisTask(request=CreateTaskRequest(target_product="Older"), status="failed")
    older.updated_at = utc_now() - timedelta(hours=1)
    latest_completed = AnalysisTask(request=CreateTaskRequest(target_product="Done"), status="completed")
    latest_completed.updated_at = utc_now()
    newer = AnalysisTask(request=CreateTaskRequest(target_product="Newer"), status="cancelled")
    newer.updated_at = utc_now() - timedelta(minutes=5)
    for task in [older, latest_completed, newer]:
        storage.save_task(task)
    client = _client(tmp_path, storage)
    scheduled: list[str] = []
    monkeypatch.setattr(main, "run_task", lambda task_id: scheduled.append(task_id))

    response = client.post("/api/tasks/resume-latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == newer.id
    assert payload["status"] == "queued"
    assert scheduled == [newer.id]


def test_resume_completed_task_returns_conflict(tmp_path, monkeypatch):
    storage = MemoryStorage()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"), status="completed")
    storage.save_task(task)
    client = _client(tmp_path, storage)
    monkeypatch.setattr(main, "run_task", lambda task_id: None)

    response = client.post(f"/api/tasks/{task.id}/resume")

    assert response.status_code == 409


def test_search_runs_endpoint_handles_partial_json(tmp_path):
    storage = MemoryStorage()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"))
    storage.save_task(task)
    run_dir = tmp_path / "data" / "search_runs" / task.id / "candidate_discovery" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text("{", encoding="utf-8")
    client = _client(tmp_path, storage)

    response = client.get(f"/api/tasks/{task.id}/search-runs")

    assert response.status_code == 200
    runs = response.json()["runs"]
    assert runs[0]["status"] == "writing"


def test_search_runs_endpoint_lists_completed_run(tmp_path):
    storage = MemoryStorage()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"))
    storage.save_task(task)
    run_dir = tmp_path / "data" / "search_runs" / task.id / "targeted_collection" / "run_1"
    (run_dir / "extractions").mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "query": "Acme competitors",
                "methods": ["duckduckgo_text"],
                "deep_backend": "none",
                "created_at": "2026-06-03T00:00:00+08:00",
                "results": [{"title": "Result", "url": "https://example.com"}],
                "skipped_duplicate_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "extractions" / "chunks.json").write_text(json.dumps({"chunks": [{"chunk_id": "chunk_1"}]}), encoding="utf-8")
    client = _client(tmp_path, storage)

    response = client.get(f"/api/tasks/{task.id}/search-runs")

    assert response.status_code == 200
    run = response.json()["runs"][0]
    assert run["query"] == "Acme competitors"
    assert run["result_count"] == 1
    assert run["chunk_count"] == 1
    assert run["skipped_duplicate_count"] == 2


def test_task_outputs_endpoint_lists_saved_files(tmp_path):
    storage = MemoryStorage()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"))
    storage.save_task(task)
    output_dir = tmp_path / "data" / "task_runs" / task.id / "steps"
    output_dir.mkdir(parents=True)
    (output_dir / "001_workflow_running_event.json").write_text("{}", encoding="utf-8")
    client = _client(tmp_path, storage)

    response = client.get(f"/api/tasks/{task.id}/outputs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == task.id
    assert payload["files"][0]["path"] == "steps/001_workflow_running_event.json"
    assert payload["files"][0]["url"].endswith(f"/task_runs/{task.id}/steps/001_workflow_running_event.json")
