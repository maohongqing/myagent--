from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

THIS_FILE = Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parents[1]
WORKSPACE_ROOT = THIS_FILE.parents[3]

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from myagent.backend.app.agents.state_tree import create_initial_state_tree
from myagent.backend.app.agents.workflow import AgentWorkflow
from myagent.backend.app.config import Settings
from myagent.backend.app.events import EventBus
from myagent.backend.app.models import CompetitorReport, CreateTaskRequest
from myagent.backend.app.storage import JSONStorage


def load_settings(output_dir: Path, duckduckgo_max_results: int | None) -> Settings:
    env_file = BACKEND_DIR / ".env"
    settings = Settings(_env_file=env_file) if env_file.exists() else Settings(_env_file=None)
    updates: dict[str, Any] = {
        "storage_backend": "json",
        "sqlite_path": output_dir / "app.db",
        "json_storage_dir": output_dir / "storage",
    }
    if duckduckgo_max_results is not None:
        updates["duckduckgo_max_results"] = duckduckgo_max_results
    return settings.model_copy(update=updates)


def resolve_path(value: str | Path, *, base: Path = BACKEND_DIR) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def make_output_dir(output_root: str | Path) -> Path:
    root = resolve_path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = root / f"{timestamp}_state_tree_debug"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def load_request(input_path: str | Path, *, enable_screenshots: bool) -> tuple[CreateTaskRequest, Path]:
    path = resolve_path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["enable_screenshots"] = enable_screenshots
    return CreateTaskRequest.model_validate(payload), path


def state_tree_summary(state_tree: dict[str, Any]) -> dict[str, Any]:
    phase_2 = state_tree.get("phase_2_targets", {})
    phase_3 = state_tree.get("phase_3_strategy", {})
    raw = state_tree.get("raw_data_sandbox", {})
    phase_4 = state_tree.get("phase_4_synthesis", {})
    return {
        "status": state_tree.get("global_metadata", {}).get("status"),
        "product_name": state_tree.get("global_metadata", {}).get("product_name"),
        "phase_1": state_tree.get("phase_1_intent"),
        "target_count": len(phase_2.get("selected_competitors") or []),
        "selected_tools": phase_3.get("selected_tools") or [],
        "selected_methods": phase_3.get("selected_methods") or [],
        "selected_analysis_units": phase_3.get("selected_analysis_units") or [],
        "crawler_directive_count": len(phase_3.get("data_collection_spec", {}).get("crawler_directives") or []),
        "raw_data_counts": {key: len(value or []) for key, value in raw.items() if isinstance(value, list)},
        "dynamic_analysis_count": len(phase_4.get("dynamic_analyses") or []),
        "has_conclusion": bool(phase_4.get("conclusion", {}).get("competitive_strategy")),
        "qa": state_tree.get("quality_gates"),
    }


def snapshot_name(index: int, node: str, state_tree: dict[str, Any]) -> str:
    status = str(state_tree.get("global_metadata", {}).get("status") or "unknown")
    safe_node = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in node)
    safe_status = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in status)
    return f"{index:03d}_{safe_node}_{safe_status}.json"


def print_snapshot(index: int, node: str, state_tree: dict[str, Any], snapshot_path: Path) -> None:
    summary = state_tree_summary(state_tree)
    print(f"\n[{index:03d}] after {node}")
    print(f"- status: {summary.get('status')}")
    print(f"- targets: {summary.get('target_count')}")
    print(f"- analysis units: {', '.join(summary.get('selected_analysis_units') or []) or 'not selected'}")
    print(f"- dynamic analyses: {summary.get('dynamic_analysis_count')}")
    print(f"- snapshot file: {snapshot_path}")


async def run_debug(args: argparse.Namespace) -> int:
    output_dir = make_output_dir(args.output_root)
    snapshots_dir = output_dir / "state_tree_snapshots"
    request, input_path = load_request(args.input, enable_screenshots=args.enable_screenshots)
    settings = load_settings(output_dir, args.duckduckgo_max_results)
    storage = JSONStorage(output_dir / "storage")
    storage.init()
    event_bus = EventBus(storage)
    workflow = AgentWorkflow(settings, storage, event_bus)
    task_id = args.task_id or output_dir.name
    max_revision_loops = args.max_revision_loops
    if max_revision_loops is None:
        max_revision_loops = settings.max_revision_loops

    initial_state = {
        "task_id": task_id,
        "request": request.model_dump(mode="json"),
        "revision_count": 0,
        "max_revision_loops": max_revision_loops,
        "qa_history": [],
        "state_tree": create_initial_state_tree(request),
        "workflow_phase": "phase_1_intent",
        "phase2_supplement_count": 0,
    }
    current_state = dict(initial_state)
    timeline: list[dict[str, Any]] = []

    write_json(output_dir / "input.json", request)
    write_json(output_dir / "initial_state.json", initial_state)
    initial_snapshot_path = snapshots_dir / "000_initial.json"
    write_json(initial_snapshot_path, initial_state["state_tree"])
    print("JSON State Tree debug run")
    print(f"Input file: {input_path}")
    print(f"Task id: {task_id}")
    print(f"Output dir: {output_dir}")
    print(f"LLM: {'enabled' if settings.openai_api_key else 'disabled, fallback only'}")
    print_snapshot(0, "initial", initial_state["state_tree"], initial_snapshot_path)

    snapshot_index = 1
    try:
        async for chunk in workflow.graph.astream(
            initial_state,
            {"recursion_limit": args.recursion_limit},
            stream_mode="updates",
        ):
            for node, updates in chunk.items():
                if not isinstance(updates, dict):
                    continue
                current_state.update(updates)
                state_tree = current_state.get("state_tree")
                if not isinstance(state_tree, dict):
                    continue
                summary = state_tree_summary(state_tree)
                timeline.append({"index": snapshot_index, "node": node, "summary": summary})
                snapshot_path = snapshots_dir / snapshot_name(snapshot_index, node, state_tree)
                write_json(snapshot_path, state_tree)
                print_snapshot(snapshot_index, node, state_tree, snapshot_path)
                snapshot_index += 1
    except Exception as exc:
        write_json(output_dir / "timeline.json", timeline)
        write_json(output_dir / "events.json", storage.list_events(task_id))
        write_json(output_dir / "error.json", {"error": str(exc), "current_state": current_state})
        print(f"\nWorkflow failed: {exc}")
        print(f"Partial snapshots saved to: {snapshots_dir}")
        return 1

    if "report" in current_state:
        report = CompetitorReport.model_validate(current_state["report"])
        storage.save_report(report)
        write_json(output_dir / "report.json", report)
    if isinstance(current_state.get("state_tree"), dict):
        write_json(output_dir / "final_state_tree.json", current_state["state_tree"])
    write_json(output_dir / "final_state.json", current_state)
    write_json(output_dir / "timeline.json", timeline)
    write_json(output_dir / "events.json", storage.list_events(task_id))

    print("\nDebug run completed")
    print(f"- snapshots: {snapshots_dir}")
    print(f"- timeline:  {output_dir / 'timeline.json'}")
    print(f"- final tree:{output_dir / 'final_state_tree.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the workflow and save JSON State Tree snapshots after each graph node update.",
    )
    parser.add_argument("--input", default="demos/sample_task.json", help="CreateTaskRequest JSON file.")
    parser.add_argument("--output-root", default=str(BACKEND_DIR / "demo_outputs"))
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--duckduckgo-max-results", type=int, default=1)
    parser.add_argument("--max-revision-loops", type=int, default=None)
    parser.add_argument("--recursion-limit", type=int, default=32)
    parser.add_argument("--enable-screenshots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run_debug(args)))


if __name__ == "__main__":
    main()
