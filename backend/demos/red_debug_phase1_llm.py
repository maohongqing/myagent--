from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from myagent.backend.app.agents.llm import get_chat_model, structured_ainvoke
from myagent.backend.app.agents.prompt_bridge import phase_1_plan_intent, plan_system_prompt
from myagent.backend.app.agents.state_tree import create_initial_state_tree, render_prompt
from myagent.backend.app.agents.workflow import StateTreePatch
from myagent.backend.app.config import Settings
from myagent.backend.app.models import CreateTaskRequest


DEFAULT_INPUT = "我想做一个二次元手办交易平台，目前主要用作决策支持"


async def run(raw_description: str, show_prompt: bool) -> None:
    request = CreateTaskRequest(raw_description=raw_description)
    state_tree = create_initial_state_tree(request)
    prompt = render_prompt(
        phase_1_plan_intent,
        current_state_tree=state_tree,
        phase_input=request.model_dump(mode="json"),
    )

    if show_prompt:
        print("=== system ===")
        print(plan_system_prompt)
        print("=== user ===")
        print(prompt)
        print("=== llm_patch ===")

    settings = Settings()
    model = get_chat_model(settings)
    if model is None:
        raise RuntimeError(
            "No chat model is available. Check OPENAI_API_KEY, OPENAI_BASE_URL, "
            "OPENAI_MODEL, and whether backend/.env is loaded."
        )

    patch = await structured_ainvoke(
        model,
        StateTreePatch,
        [("system", plan_system_prompt), ("user", prompt)],
    )
    print(json.dumps(patch.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Plan Agent phase-1 LLM prompt and print the StateTreePatch JSON."
    )
    parser.add_argument(
        "raw_description",
        nargs="?",
        default=DEFAULT_INPUT,
        help="User input text to pass as CreateTaskRequest.raw_description.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the rendered system/user prompt before the LLM patch.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.raw_description, args.show_prompt))


if __name__ == "__main__":
    main()
