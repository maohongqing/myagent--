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
from myagent.backend.app.agents.prompt_bridge import phase_2_qa_intent, qa_system_prompt
from myagent.backend.app.agents.state_tree import render_prompt
from myagent.backend.app.agents.workflow import QAInspectionOutput
from myagent.backend.app.config import Settings


DEFAULT_INPUT = BACKEND_DIR / "demos" / "search_to_analyst_output.json"
DEFAULT_OUTPUT = BACKEND_DIR / "demos" / "qa_checkpoint1_output.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_state_tree(payload: dict[str, Any]) -> dict[str, Any]:
    state_tree = payload.get("state_tree")
    if isinstance(state_tree, dict):
        return state_tree

    phase_1_intent = payload.get("phase_1_intent") or {}
    phase_2_targets = payload.get("phase_2_targets") or {}
    return {
        "phase_1_intent": phase_1_intent,
        "phase_2_targets": phase_2_targets,
        "quality_gates": {},
        "global_metadata": {},
    }


async def run(
    input_path: Path,
    output_path: Path,
    show_prompt: bool,
    timeout_seconds: float | None,
    no_llm: bool,
) -> None:
    payload = _load_json(input_path)
    state_tree = _build_state_tree(payload)

    prompt = render_prompt(
        phase_2_qa_intent,
        phase_1_intent=state_tree.get("phase_1_intent", {}),
        phase_2_targets=state_tree.get("phase_2_targets", {}),
    )

    if show_prompt:
        print("=== qa_agent system ===")
        print(qa_system_prompt)
        print("=== qa_agent user ===")
        print(prompt)

    settings = Settings()
    if timeout_seconds is not None:
        settings.openai_timeout_seconds = timeout_seconds
    model = None if no_llm else get_chat_model(settings)

    llm_error: str | None = None
    inspection = QAInspectionOutput(inspection_status="pass")
    if model is not None:
        try:
            inspection = await structured_ainvoke(
                model,
                QAInspectionOutput,
                [("system", qa_system_prompt), ("user", prompt)],
            )
        except Exception as exc:
            llm_error = friendly_llm_error(exc)
            print(f"QA Agent LLM call failed; using default pass inspection: {type(exc).__name__}: {exc}")

    selected = state_tree.get("phase_2_targets", {}).get("selected_competitors", [])
    issues = list(inspection.issues_found)
    if len(selected) != 5:
        issues.append(f"Expected 5 selected competitors, got {len(selected)}.")

    passed = inspection.inspection_status == "pass" and not issues
    state_tree.setdefault("quality_gates", {})["checkpoint_1"] = {
        "inspection_status": "pass" if passed else "reject",
        "issues_found": issues,
        "feedback": inspection.feedback_to_analyzer,
    }

    next_state = {
        "state_tree": state_tree,
        "workflow_phase": "phase_3_strategy" if passed else "phase_2_analyze",
        "qa_checkpoint": None,
        "revision_count_delta": 0 if passed else 1,
    }
    output_payload = {
        "input": str(input_path),
        "qa_prompt": prompt if show_prompt else None,
        "llm_error": llm_error,
        "inspection": inspection.model_dump(mode="json"),
        "selected_count": len(selected),
        "passed": passed,
        "checkpoint_1": state_tree["quality_gates"]["checkpoint_1"],
        "next_state": next_state,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved QA checkpoint output to: {output_path}")
    print(f"Passed: {passed}")
    print(f"Next workflow_phase: {next_state['workflow_phase']}")
    print(json.dumps(output_payload["checkpoint_1"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QA Agent checkpoint 1 using phase_2_targets from a search-to-analyst debug JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to search_to_analyst_output.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write QA checkpoint debug JSON.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print and save the rendered QA prompt.",
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
        help="Skip the LLM call and run only the workflow's local checkpoint checks.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.show_prompt, args.timeout, args.no_llm))


if __name__ == "__main__":
    main()
