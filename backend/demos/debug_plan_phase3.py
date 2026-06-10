from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from myagent.backend.app.agents.llm import friendly_llm_error, get_chat_model, structured_ainvoke
from myagent.backend.app.agents.prompt_bridge import phase_3_plan_intent, plan_system_prompt
from myagent.backend.app.agents.state_tree import merge_updated_nodes, render_prompt, tool_plan_to_phase_3
from myagent.backend.app.agents.strategy import build_collection_plan, build_tool_plan
from myagent.backend.app.agents.workflow import StateTreePatch
from myagent.backend.app.config import Settings
from myagent.backend.app.models import AnalysisBenchmark, CreateTaskRequest, SelectedCompetitorSet


DEFAULT_INPUT = BACKEND_DIR / "demos" / "qa_checkpoint1_output.json"
DEFAULT_FALLBACK_INPUT = BACKEND_DIR / "demos" / "search_to_analyst_output.json"
DEFAULT_OUTPUT = BACKEND_DIR / "demos" / "plan_phase3_output.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_context(input_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(input_path)
    if "next_state" in payload:
        state_tree = payload["next_state"].get("state_tree") or {}
        source_path = Path(str(payload.get("input") or DEFAULT_FALLBACK_INPUT))
        source_payload = _load_json(source_path) if source_path.exists() else {}
        merged_payload = {**source_payload, "qa_checkpoint_output": payload}
        return merged_payload, state_tree

    state_tree = payload.get("state_tree") or {}
    return payload, state_tree


async def run(
    input_path: Path,
    output_path: Path,
    show_prompt: bool,
    timeout_seconds: float | None,
    no_llm: bool,
) -> None:
    payload, state_tree = _load_context(input_path)
    request = CreateTaskRequest.model_validate(payload["request"])
    benchmark = AnalysisBenchmark.model_validate(payload["benchmark"])
    selected_set = SelectedCompetitorSet.model_validate(payload["selected_competitor_set"])

    tool_plan = build_tool_plan(benchmark, selected_set, request)
    collection_plan = build_collection_plan(benchmark, selected_set, tool_plan)

    prompt = render_prompt(phase_3_plan_intent, current_state_tree=state_tree)
    if show_prompt:
        print("=== plan_agent system ===")
        print(plan_system_prompt)
        print("=== plan_agent user ===")
        print(prompt)

    settings = Settings()
    if timeout_seconds is not None:
        settings.openai_timeout_seconds = timeout_seconds
    model = None if no_llm else get_chat_model(settings)

    llm_error: str | None = None
    llm_patch: dict[str, Any] | None = None
    state_tree_after_llm = state_tree
    if model is not None:
        try:
            patch = await structured_ainvoke(
                model,
                StateTreePatch,
                [("system", plan_system_prompt), ("user", prompt)],
            )
            llm_patch = patch.model_dump(mode="json")
            state_tree_after_llm = merge_updated_nodes(state_tree_after_llm, llm_patch)
        except Exception as exc:
            llm_error = friendly_llm_error(exc)
            print(f"Plan Agent phase-3 LLM call failed; using local tool plan: {type(exc).__name__}: {exc}")

    local_phase_3_strategy = tool_plan_to_phase_3(tool_plan, collection_plan)
    final_state_tree = merge_updated_nodes(
        state_tree_after_llm,
        {"updated_nodes": {"phase_3_strategy": local_phase_3_strategy}},
    )
    final_state_tree.setdefault("global_metadata", {})["status"] = "phase_3_strategy_ready"

    output_payload = {
        "input": str(input_path),
        "plan_prompt": prompt if show_prompt else None,
        "llm_error": llm_error,
        "llm_patch": llm_patch,
        "tool_plan": tool_plan.model_dump(mode="json"),
        "collection_plan": collection_plan.model_dump(mode="json"),
        "local_phase_3_strategy": local_phase_3_strategy,
        "final_phase_3_strategy": final_state_tree.get("phase_3_strategy"),
        "state_tree": final_state_tree,
        "next_workflow_phase": "phase_4_collect",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved Plan Agent phase-3 output to: {output_path}")
    print("Selected analysis units:")
    for item in final_state_tree.get("phase_3_strategy", {}).get("selected_analysis_units", []):
        print(f"- {item}")
    print(f"Collection instructions: {len(collection_plan.instructions)}")
    print(f"Next workflow_phase: {output_payload['next_workflow_phase']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Plan Agent phase 3 from QA checkpoint output and write strategy/collection plan JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to qa_checkpoint1_output.json or search_to_analyst_output.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write Plan Agent phase-3 debug JSON.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print and save the rendered Plan Agent prompt.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override OPENAI_TIMEOUT_SECONDS for this debug run.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM call and use the workflow's local tool/collection plan.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.show_prompt, args.timeout, args.no_llm))


if __name__ == "__main__":
    main()
