from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import warnings
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

from myagent.backend.app.agents.collector import collect_sources, duckduckgo_search, fetch_url
from myagent.backend.app.config import Settings
from myagent.backend.app.events import EventBus
from myagent.backend.app.exporters import report_to_markdown
from myagent.backend.app.models import AgentRunEvent, CompetitorReport, CreateTaskRequest, RawSource
from myagent.backend.app.storage import JSONStorage


def configure_console(verbose: bool) -> None:
    if verbose:
        return
    logging.getLogger("competitor_agent").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed to `ddgs`.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def load_settings() -> Settings:
    env_file = BACKEND_DIR / ".env"
    return Settings(_env_file=env_file) if env_file.exists() else Settings(_env_file=None)


def compact_text(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def resolve_path(value: str | Path, *, base: Path = BACKEND_DIR) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def make_output_dir(command: str, output_root: str | Path) -> Path:
    root = resolve_path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = root / f"{timestamp}_{command}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def demo_settings(args: argparse.Namespace, output_dir: Path, *, storage_backend: str | None = None) -> Settings:
    settings = load_settings()
    updates: dict[str, Any] = {
        "sqlite_path": output_dir / "app.db",
        "json_storage_dir": output_dir / "storage",
    }
    if storage_backend:
        updates["storage_backend"] = storage_backend
    if hasattr(args, "max_results") and args.max_results is not None:
        updates["duckduckgo_max_results"] = args.max_results
    if hasattr(args, "duckduckgo_max_results") and args.duckduckgo_max_results is not None:
        updates["duckduckgo_max_results"] = args.duckduckgo_max_results
    if hasattr(args, "enable_tavily") and not args.enable_tavily:
        updates["tavily_api_key"] = None
    return settings.model_copy(update=updates)


def load_request(input_path: str | Path, *, enable_screenshots: bool) -> tuple[CreateTaskRequest, Path]:
    path = resolve_path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["enable_screenshots"] = enable_screenshots
    return CreateTaskRequest.model_validate(payload), path


def print_output_dir(output_dir: Path) -> None:
    print(f"\nSaved full output to: {output_dir}")


def print_source_summary(source: RawSource, index: int, content_chars: int) -> None:
    print(f"\n[{index}] {source.title}")
    print(f"    type: {source.source_type}")
    print(f"    url:  {source.url}")
    if source.metadata:
        print(f"    metadata: {json.dumps(source.metadata, ensure_ascii=False)}")
    print(f"    content: {compact_text(source.content, content_chars)}")


def print_sources(sources: list[RawSource], content_chars: int) -> None:
    print(f"\nRaw sources: {len(sources)}")
    for index, source in enumerate(sources, start=1):
        print_source_summary(source, index, content_chars)


def print_runtime_flags(settings: Settings, args: argparse.Namespace, *, include_llm: bool = False) -> None:
    print(f"DuckDuckGo max results: {settings.duckduckgo_max_results}")
    if hasattr(args, "enable_tavily"):
        tavily_state = "enabled" if args.enable_tavily and settings.tavily_api_key else "disabled"
        if args.enable_tavily and not settings.tavily_api_key:
            tavily_state = "requested, but TAVILY_API_KEY is not configured"
        print(f"Tavily: {tavily_state}")
    if hasattr(args, "enable_screenshots"):
        print(f"Screenshots: {'enabled' if args.enable_screenshots else 'disabled'}")
    if include_llm:
        print(f"LLM: {'enabled' if settings.openai_api_key else 'disabled, using fallback logic'}")


def print_events(events: list[AgentRunEvent]) -> None:
    print(f"\nAgent events: {len(events)}")
    for event in events:
        detail = event.output_summary or event.message or event.error or ""
        duration = f", {event.duration_ms}ms" if event.duration_ms is not None else ""
        print(f"- {event.node}: {event.status}{duration} {detail}".rstrip())


async def run_duckduckgo(args: argparse.Namespace) -> int:
    output_dir = make_output_dir("duckduckgo", args.output_root)
    settings = demo_settings(args, output_dir)
    input_payload = {"query": args.query, "max_results": settings.duckduckgo_max_results}

    print("DuckDuckGo demo")
    print(f"Query: {args.query}")
    print_runtime_flags(settings, args)
    write_json(output_dir / "input.json", input_payload)

    sources = await duckduckgo_search(args.query, settings)
    write_json(output_dir / "raw_sources.json", sources)
    print_sources(sources, args.content_chars)
    print_output_dir(output_dir)
    return 0


async def run_fetch_url(args: argparse.Namespace) -> int:
    output_dir = make_output_dir("fetch-url", args.output_root)
    input_payload = {"url": args.url}

    print("Fetch URL demo")
    print(f"URL: {args.url}")
    write_json(output_dir / "input.json", input_payload)

    try:
        source = await fetch_url(args.url)
    except Exception as exc:
        source = RawSource(
            url=args.url,
            title=f"Fetch failed: {args.url}",
            content=f"URL fetch failed: {exc}",
            source_type="url",
            metadata={"error": str(exc)},
        )
        write_json(output_dir / "error.json", {"error": str(exc)})

    sources = [source]
    write_json(output_dir / "raw_sources.json", sources)
    print_sources(sources, args.content_chars)
    print_output_dir(output_dir)
    return 0


async def run_collect(args: argparse.Namespace) -> int:
    output_dir = make_output_dir("collect", args.output_root)
    request, input_path = load_request(args.input, enable_screenshots=args.enable_screenshots)
    settings = demo_settings(args, output_dir)

    print("Collector demo")
    print(f"Input file: {input_path}")
    print(f"Target product: {request.target_product}")
    print(f"Competitors: {', '.join(request.competitors) or 'not specified'}")
    print_runtime_flags(settings, args)
    write_json(output_dir / "input.json", request)

    sources = await collect_sources(request, settings, task_id=args.task_id or output_dir.name)
    write_json(output_dir / "raw_sources.json", sources)
    print_sources(sources, args.content_chars)
    print_output_dir(output_dir)
    return 0


async def run_workflow(args: argparse.Namespace) -> int:
    from myagent.backend.app.agents.workflow import AgentWorkflow

    output_dir = make_output_dir("workflow", args.output_root)
    request, input_path = load_request(args.input, enable_screenshots=args.enable_screenshots)
    settings = demo_settings(args, output_dir, storage_backend="json")
    storage = JSONStorage(output_dir / "storage")
    storage.init()
    event_bus = EventBus(storage)
    workflow = AgentWorkflow(settings, storage, event_bus)
    task_id = args.task_id or output_dir.name
    max_revision_loops = args.max_revision_loops
    if max_revision_loops is None:
        max_revision_loops = settings.max_revision_loops

    print("Workflow demo")
    print(f"Input file: {input_path}")
    print(f"Task id: {task_id}")
    print(f"Target product: {request.target_product}")
    print(f"Competitors: {', '.join(request.competitors) or 'not specified'}")
    print_runtime_flags(settings, args, include_llm=True)
    print(f"Max revision loops: {max_revision_loops}")
    write_json(output_dir / "input.json", request)

    try:
        initial_state = {
            "task_id": task_id,
            "request": request.model_dump(mode="json"),
            "revision_count": 0,
            "max_revision_loops": max_revision_loops,
            "qa_history": [],
        }
        result = await workflow.graph.ainvoke(initial_state, {"recursion_limit": 24})
        report = CompetitorReport.model_validate(result["report"])
        storage.save_report(report)
        raw_sources = [RawSource.model_validate(item) for item in result.get("raw_sources", [])]
    except Exception as exc:
        events = storage.list_events(task_id)
        write_json(output_dir / "events.json", events)
        write_json(output_dir / "error.json", {"error": str(exc)})
        print_events(events)
        print(f"\nWorkflow failed: {exc}")
        print_output_dir(output_dir)
        return 1

    events = storage.list_events(task_id)
    markdown = report_to_markdown(report)
    write_json(output_dir / "raw_sources.json", raw_sources)
    write_json(output_dir / "report.json", report)
    write_text(output_dir / "report.md", markdown)
    write_json(output_dir / "events.json", events)

    print_events(events)
    latest_qa = report.qa_history[-1] if report.qa_history else None
    print("\nFinal report")
    print(f"- evidence count: {len(report.evidence)}")
    if latest_qa:
        print(f"- QA passed: {latest_qa.passed}")
        print(f"- QA score: {latest_qa.score}")
    if report.risk_notice:
        print(f"- risk notice: {compact_text(report.risk_notice, args.content_chars)}")
    print(f"- report json: {output_dir / 'report.json'}")
    print(f"- report md:   {output_dir / 'report.md'}")
    print_output_dir(output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real-call demos for the competitor analysis agents.",
    )
    parser.add_argument(
        "--output-root",
        default=str(BACKEND_DIR / "demo_outputs"),
        help="Directory where demo output folders are created.",
    )
    parser.add_argument(
        "--content-chars",
        type=int,
        default=300,
        help="How many characters of source/report text to print in the console.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show library warnings and full agent exception logs.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    duckduckgo = subparsers.add_parser("duckduckgo", help="Run a real DuckDuckGo search.")
    add_common_output_options(duckduckgo)
    duckduckgo.add_argument("--query", required=True, help="Search query.")
    duckduckgo.add_argument("--max-results", type=int, default=5, help="Maximum DuckDuckGo results.")
    duckduckgo.set_defaults(handler=run_duckduckgo)

    fetch = subparsers.add_parser("fetch-url", help="Fetch and clean a real web page.")
    add_common_output_options(fetch)
    fetch.add_argument("--url", required=True, help="URL to fetch.")
    fetch.set_defaults(handler=run_fetch_url)

    collect = subparsers.add_parser("collect", help="Run the collector with real search and URL fetches.")
    add_common_output_options(collect)
    collect.add_argument("--input", default="demos/sample_task.json", help="CreateTaskRequest JSON file.")
    collect.add_argument("--duckduckgo-max-results", type=int, default=None, help="DuckDuckGo results per query.")
    collect.add_argument("--enable-tavily", action="store_true", help="Use Tavily when TAVILY_API_KEY is configured.")
    collect.add_argument("--enable-screenshots", action="store_true", help="Capture screenshots with Playwright.")
    collect.add_argument("--task-id", default=None, help="Optional task id used for screenshots/output labels.")
    collect.set_defaults(handler=run_collect)

    workflow = subparsers.add_parser("workflow", help="Run the full multi-agent workflow.")
    add_common_output_options(workflow)
    workflow.add_argument("--input", default="demos/sample_task.json", help="CreateTaskRequest JSON file.")
    workflow.add_argument("--duckduckgo-max-results", type=int, default=None, help="DuckDuckGo results per query.")
    workflow.add_argument("--enable-tavily", action="store_true", help="Use Tavily when TAVILY_API_KEY is configured.")
    workflow.add_argument("--enable-screenshots", action="store_true", help="Capture screenshots with Playwright.")
    workflow.add_argument("--max-revision-loops", type=int, default=None, help="QA revision loop budget.")
    workflow.add_argument("--task-id", default=None, help="Optional workflow task id.")
    workflow.set_defaults(handler=run_workflow)

    return parser


def add_common_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        default=argparse.SUPPRESS,
        help="Directory where demo output folders are created.",
    )
    parser.add_argument(
        "--content-chars",
        type=int,
        default=argparse.SUPPRESS,
        help="How many characters of source/report text to print in the console.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show library warnings and full agent exception logs.",
    )


async def main_async(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_console(args.verbose)
    return await args.handler(args)


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
