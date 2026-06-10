from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
DB_PATH = DATA_DIR / "app.db"
SOURCE_TASK_ID = "task_089d036475e0"
TARGET_TASK_ID = f"{SOURCE_TASK_ID}_traf_{datetime.now(timezone.utc).strftime('%H%M%S')}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rewrite_task_payload(payload: dict) -> dict:
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    payload["id"] = TARGET_TASK_ID
    payload["status"] = "queued"
    payload["current_node"] = "workflow"
    payload["pending_clarification"] = None
    payload["pending_decision"] = None
    payload["error"] = None
    payload["created_at"] = now_iso()
    payload["updated_at"] = now_iso()
    payload["completed_at"] = None
    payload.setdefault("request", {})["enable_readability_extraction"] = True
    return payload


def rewrite_checkpoint(payload: dict) -> dict:
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    payload["task_id"] = TARGET_TASK_ID
    payload["updated_at"] = datetime.now(timezone.utc).timestamp()
    state = payload.setdefault("state", {})
    state["task_id"] = TARGET_TASK_ID
    state.setdefault("request", {})["enable_readability_extraction"] = True
    return payload


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise RuntimeError(f"Target already exists: {target}")
    if source.exists():
        shutil.copytree(source, target)


def copy_task_rows() -> None:
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        task_row = db.execute("select payload from tasks where id = ?", (SOURCE_TASK_ID,)).fetchone()
        if not task_row:
            raise RuntimeError(f"Source task not found in DB: {SOURCE_TASK_ID}")
        task_payload = rewrite_task_payload(json.loads(task_row["payload"]))
        db.execute(
            """
            insert into tasks(id, payload, status, updated_at)
            values (?, ?, ?, ?)
            """,
            (
                TARGET_TASK_ID,
                json.dumps(task_payload, ensure_ascii=False),
                task_payload["status"],
                task_payload["updated_at"],
            ),
        )

        event_rows = db.execute(
            "select payload from events where task_id = ? order by created_at asc",
            (SOURCE_TASK_ID,),
        ).fetchall()
        for row in event_rows:
            event_payload = json.loads(row["payload"])
            event_payload["id"] = f"evt_{uuid4().hex[:12]}"
            event_payload["task_id"] = TARGET_TASK_ID
            db.execute(
                "insert into events(id, task_id, payload, created_at) values (?, ?, ?, ?)",
                (
                    event_payload["id"],
                    TARGET_TASK_ID,
                    json.dumps(event_payload, ensure_ascii=False),
                    event_payload.get("created_at") or now_iso(),
                ),
            )

        report_row = db.execute("select payload from reports where task_id = ?", (SOURCE_TASK_ID,)).fetchone()
        if report_row:
            report_payload = json.loads(report_row["payload"])
            report_payload["task_id"] = TARGET_TASK_ID
            db.execute(
                "insert into reports(task_id, payload, generated_at) values (?, ?, ?)",
                (
                    TARGET_TASK_ID,
                    json.dumps(report_payload, ensure_ascii=False),
                    report_payload.get("generated_at") or now_iso(),
                ),
            )


def main() -> None:
    task_source = DATA_DIR / "task_runs" / SOURCE_TASK_ID
    task_target = DATA_DIR / "task_runs" / TARGET_TASK_ID
    search_source = DATA_DIR / "search_runs" / SOURCE_TASK_ID
    search_target = DATA_DIR / "search_runs" / TARGET_TASK_ID

    copy_tree(task_source, task_target)
    copy_tree(search_source, search_target)

    checkpoint_path = task_target / "checkpoint.json"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint_path.write_text(
        json.dumps(rewrite_checkpoint(json.loads(checkpoint_path.read_text(encoding="utf-8"))), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    copy_task_rows()
    print(TARGET_TASK_ID)


if __name__ == "__main__":
    main()
