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

from myagent.backend.app.agents.candidate_selector import (
    generate_candidate_pool_with_llm_or_rules,
    select_candidates_with_llm_or_rules,
)
from myagent.backend.app.agents.collector import collect_candidate_sources
from myagent.backend.app.agents.llm import get_chat_model
from myagent.backend.app.agents.prompt_bridge import phase_2_search_intent, search_system_prompt
from myagent.backend.app.agents.state_tree import (
    create_initial_state_tree,
    merge_updated_nodes,
    render_prompt,
    selected_set_to_phase_2,
)
from myagent.backend.app.agents.strategy import build_candidate_queries, infer_benchmark, score_and_select_candidates
from myagent.backend.app.config import Settings
from myagent.backend.app.models import CreateTaskRequest


DEFAULT_INPUT = "我想做一个二次元手办交易平台，目前主要用作决策支持"
DEFAULT_OUTPUT = BACKEND_DIR / "demos" / "search_to_analyst_output.json"
DEFAULT_PHASE_1_INTENT: dict[str, str] = {
    "product_lifecycle": "concept",
    "analysis_purpose": "decision_support",
    "core_problem_to_solve": "评估是否以及如何打造一个二次元手办交易平台，为产品立项、市场进入和差异化定位提供决策依据",
    "specific_goal": "通过竞品分析明确二次元手办交易平台的市场机会、核心用户需求、主要竞品格局、功能差异化方向与初步商业策略",
}


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


async def _call_search_prompt(model: Any, prompt: str, *, show_prompt: bool) -> str | None:
    if show_prompt:
        print("=== search_agent system ===")
        print(search_system_prompt)
        print("=== search_agent user ===")
        print(prompt)
    if model is None:
        return None
    try:
        response = await model.ainvoke([("system", search_system_prompt), ("user", prompt)])
    except Exception as exc:
        print(f"Search Agent prompt call failed and will be treated as advisory-only: {type(exc).__name__}: {exc}")
        return None
    return _response_text(response).strip()


async def run(
    raw_description: str,
    output_path: Path,
    show_prompt: bool,
    timeout_seconds: float | None,
    max_sources: int,
    max_candidates: int,
    no_llm: bool,
    prompt_only: bool,
) -> None:
    request = CreateTaskRequest(raw_description=raw_description)
    state_tree = create_initial_state_tree(request)
    state_tree = merge_updated_nodes(
        state_tree,
        {"updated_nodes": {"phase_1_intent": DEFAULT_PHASE_1_INTENT}},
    )

    benchmark, clarification = infer_benchmark(request)
    if not benchmark:
        raise RuntimeError(f"Benchmark could not be inferred: {clarification}")
    request.target_product = request.target_product or benchmark.target_product

    settings = Settings()
    if timeout_seconds is not None:
        settings.openai_timeout_seconds = timeout_seconds
    model = None if no_llm else get_chat_model(settings)

    search_prompt = render_prompt(
        phase_2_search_intent,
        phase_1_intent=state_tree.get("phase_1_intent", {}),
    )
    prompt_output = await _call_search_prompt(model, search_prompt, show_prompt=show_prompt)
    if prompt_only:
        print(prompt_output or "No Search Agent prompt output was produced.")
        return

    print("=== build_candidate_queries ===")
    queries = build_candidate_queries(benchmark, request)
    for index, query in enumerate(queries, start=1):
        print(f"{index}. {query}")

    print("=== collect_candidate_sources ===")
    sources = await collect_candidate_sources(queries, settings, max_sources=max_sources)
    print(f"Collected sources: {len(sources)}")

    print("=== generate_candidate_pool_with_llm_or_rules ===")
    candidates, candidate_pool_notes, candidate_pool_mode = await generate_candidate_pool_with_llm_or_rules(
        model,
        benchmark,
        request,
        sources,
        max_candidates=max_candidates,
    )
    print(f"Generated candidates: {len(candidates)} ({candidate_pool_mode})")

    print("=== analyst_phase_2_selection ===")
    try:
        selected_set = await select_candidates_with_llm_or_rules(model, benchmark, candidates, sources, budget=5)
        analyst_mode = "llm_selection" if model else "rule_selection_no_model"
    except Exception as exc:
        print(f"Analyst LLM selection failed; falling back to rule scoring: {type(exc).__name__}: {exc}")
        selected_set = score_and_select_candidates(benchmark, candidates, budget=5)
        selected_set.fallback_notes.append(f"大模型语义筛选失败，已回退到规则评分：{exc}")
        analyst_mode = "rule_selection_after_llm_error"

    phase_2_targets = selected_set_to_phase_2(selected_set)
    state_tree = merge_updated_nodes(
        state_tree,
        {"updated_nodes": {"phase_2_targets": phase_2_targets}},
    )
    state_tree["global_metadata"]["status"] = "phase_2_targets_locked"

    payload = {
        "request": request.model_dump(mode="json"),
        "benchmark": benchmark.model_dump(mode="json"),
        "phase_1_intent": state_tree.get("phase_1_intent"),
        "search_agent_prompt_output": prompt_output,
        "queries": queries,
        "candidate_sources": [source.model_dump(mode="json") for source in sources],
        "candidate_pool_mode": candidate_pool_mode,
        "candidate_pool_notes": candidate_pool_notes,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "analyst_mode": analyst_mode,
        "selected_competitor_set": selected_set.model_dump(mode="json"),
        "phase_2_targets": phase_2_targets,
        "state_tree": state_tree,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved full search->analyst debug output to: {output_path}")
    print("Selected competitors:")
    for index, item in enumerate(selected_set.selected, start=1):
        print(f"{index}. {item.name} [{item.competitor_type}] score={item.score}")
        print(f"   {item.selection_reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the real phase-2 chain: Search Agent prompt, web sources, candidate pool, Analyst selection."
    )
    parser.add_argument(
        "raw_description",
        nargs="?",
        default=DEFAULT_INPUT,
        help="User input text used to build the initial request context.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the full JSON debug output.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the rendered Search Agent system/user prompt.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override OPENAI_TIMEOUT_SECONDS for this debug run.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=50,
        help="Maximum web sources to collect.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=30,
        help="Maximum candidates to generate from sources.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls and use rule-based fallbacks where possible.",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Only call/print the Search Agent prompt output; skip web collection and Analyst selection.",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.raw_description,
            args.output,
            args.show_prompt,
            args.timeout,
            args.max_sources,
            args.max_candidates,
            args.no_llm,
            args.prompt_only,
        )
    )


if __name__ == "__main__":
    main()
