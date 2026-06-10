from pathlib import Path

from myagent.backend.app.agents.fallbacks import build_fallback_report
from myagent.backend.app.exporters import export_report, report_to_markdown
from myagent.backend.app.models import AgentRunEvent, AnalysisTask, CreateTaskRequest
from myagent.backend.app.storage import JSONStorage, SQLiteStorage


def exercise_storage(storage):
    storage.init()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Feishu", competitors=["DingTalk"]))
    report = build_fallback_report(task.id, task.request, sources=[])
    event = AgentRunEvent(
        task_id=task.id,
        node="workflow",
        status="queued",
        message="created",
        details={"kind": "workflow", "step": "created"},
    )

    storage.save_task(task)
    storage.save_report(report)
    storage.append_event(event)

    assert storage.get_task(task.id).id == task.id
    assert storage.get_report(task.id).task_id == task.id
    assert storage.list_events(task.id)[0].message == "created"
    assert storage.list_events(task.id)[0].details["step"] == "created"


def test_sqlite_storage(tmp_path: Path):
    exercise_storage(SQLiteStorage(tmp_path / "app.db"))


def test_json_storage(tmp_path: Path):
    exercise_storage(JSONStorage(tmp_path / "json"))


def test_export_formats_smoke():
    request = CreateTaskRequest(target_product="Feishu", competitors=["DingTalk"])
    report = build_fallback_report("task_test", request, sources=[])

    markdown = report_to_markdown(report)
    assert "竞品分析报告" in markdown
    assert "## 一、报告名称与报告日期" in markdown

    for fmt in ("md", "json", "pdf", "docx"):
        payload, media_type, suffix = export_report(report, fmt)
        assert payload
        assert suffix == fmt
        assert media_type
