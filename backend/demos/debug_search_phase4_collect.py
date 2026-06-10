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

from myagent.backend.app.agents.fallbacks import evidence_from_sources
from myagent.backend.app.agents.llm import get_chat_model
from myagent.backend.app.agents.prompt_bridge import phase_3_search_intent, search_system_prompt
from myagent.backend.app.agents.search_planner import generate_search_plan
from myagent.backend.app.agents.state_tree import merge_updated_nodes, render_prompt, sources_to_sandbox
from myagent.backend.app.config import Settings
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CollectionPlan,
    CreateTaskRequest,
    RawSource,
    SearchPlan,
    SelectedCompetitorSet,
    SourceProviderResult,
    ToolPlan,
)
from myagent.backend.app.agents.strategy import scoring_profile_for
from myagent.backend.app.providers.base import dedupe_sources
from myagent.backend.app.providers.cache import SourceCacheProvider
from myagent.backend.app.providers.search import SearchProvider
from myagent.backend.app.providers.web import WebProvider


DEFAULT_INPUT = BACKEND_DIR / "demos" / "plan_phase3_output.json"
DEFAULT_CONTEXT_INPUT = BACKEND_DIR / "demos" / "search_to_analyst_output.json"
DEFAULT_OUTPUT = BACKEND_DIR / "demos" / "search_phase4_output.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _limit_search_plan(search_plan: SearchPlan, max_queries: int) -> SearchPlan:
    if max_queries <= 0:
        return search_plan
    return search_plan.model_copy(update={"queries": search_plan.queries[:max_queries]})


def _context_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "request" in payload and "benchmark" in payload:
        return payload
    if DEFAULT_CONTEXT_INPUT.exists():
        context = _load_json(DEFAULT_CONTEXT_INPUT)
        return {**context, **payload}
    return payload


def _selected_set_from_state_tree(
    state_tree: dict[str, Any],
    benchmark: AnalysisBenchmark,
    fallback_payload: dict[str, Any],
) -> SelectedCompetitorSet:
    selected_items = state_tree.get("phase_2_targets", {}).get("selected_competitors", [])
    selected: list[CandidateCompetitor] = []
    for item in selected_items:
        name = str(item.get("competitor_name") or "").strip()
        if not name:
            continue
        selected.append(
            CandidateCompetitor(
                name=name,
                description=str(item.get("selection_reason") or ""),
                competitor_type=item.get("competitor_type") or "challenger",
                selection_reason=str(item.get("selection_reason") or ""),
                score=float(item.get("rank_score") or 0.0),
                tags=["phase_2_targets"],
            )
        )
    if selected:
        return SelectedCompetitorSet(
            total_budget=5,
            scoring_profile=scoring_profile_for(benchmark.analysis_purpose),
            candidates=selected,
            selected=selected,
            allocation={},
            fallback_notes=[],
        )
    return SelectedCompetitorSet.model_validate(fallback_payload["selected_competitor_set"])


async def _collect_sources(
    settings: Settings,
    request: CreateTaskRequest,
    benchmark: AnalysisBenchmark,
    selected_set: SelectedCompetitorSet,
    tool_plan: ToolPlan,
    search_plan: SearchPlan,
    *,
    task_id: str,
    max_sources: int,
    fetch_details: bool,
) -> tuple[list[RawSource], list[SourceProviderResult]]:
    cache_path = Path(settings.sqlite_path).parent / "source_cache.json"
    cache_provider = SourceCacheProvider(cache_path)
    search_provider = SearchProvider(settings)
    web_provider = WebProvider(settings)

    provider_results: list[SourceProviderResult] = []
    cache_result = await cache_provider.search_evidence(
        search_plan,
        benchmark,
        tool_plan,
        selected_set.selected,
        max_sources=max_sources,
    )
    provider_results.append(cache_result)
    sources: list[RawSource] = list(cache_result.sources)

    if len(sources) < min(12, max_sources):
        search_result = await search_provider.search_evidence(
            search_plan,
            benchmark,
            tool_plan,
            selected_set.selected,
            max_sources=max_sources,
        )
        provider_results.append(search_result)
        sources.extend(search_result.sources)

    if fetch_details:
        urls = [source.url for source in sources if source.url.startswith(("http://", "https://"))]
        urls.extend(request.urls)
        if urls:
            cached_detail = await cache_provider.fetch_detail(urls, task_id=task_id, max_sources=min(20, max_sources))
            provider_results.append(cached_detail)
            sources.extend(cached_detail.sources)
            missing_urls = [
                url
                for url in urls
                if url
                and not any(source.url == url and not source.metadata.get("error") for source in cached_detail.sources)
            ]
            if missing_urls:
                web_result = await web_provider.fetch_detail(
                    missing_urls[: min(16, max_sources)],
                    task_id=task_id,
                    max_sources=min(16, max_sources),
                )
                provider_results.append(web_result)
                sources.extend(web_result.sources)

    sources = dedupe_sources(sources)[:max_sources]
    if not sources:
        description = request.raw_description or request.notes or request.target_product
        sources = [
            RawSource(
                url="manual://task-input",
                title="Task input fallback",
                content=(
                    f"User asked to analyze {benchmark.target_product}. "
                    f"Raw description: {description or 'not provided'}. "
                    "No cache, search provider, or URL fetch returned usable evidence."
                ),
                source_type="manual",
                metadata={"provider": "fallback"},
            )
        ]

    cache_provider.remember_sources(sources)
    return sources, provider_results


async def run(
    input_path: Path,
    output_path: Path,
    show_prompt: bool,
    timeout_seconds: float | None,
    no_llm: bool,
    max_queries: int,
    max_sources: int,
    fetch_details: bool,
) -> None:
    payload = _load_json(input_path)
    payload = _context_payload(payload)
    state_tree = payload["state_tree"]
    request = CreateTaskRequest.model_validate(payload["request"])
    benchmark = AnalysisBenchmark.model_validate(payload["benchmark"])
    selected_set = _selected_set_from_state_tree(state_tree, benchmark, payload)
    tool_plan = ToolPlan.model_validate(payload["tool_plan"])
    collection_plan = CollectionPlan.model_validate(payload["collection_plan"])

    settings = Settings()
    if timeout_seconds is not None:
        settings.openai_timeout_seconds = timeout_seconds
    model = None if no_llm else get_chat_model(settings)

    search_plan = await generate_search_plan(
        model,
        benchmark,
        selected_set,
        tool_plan,
        collection_plan,
        missing_fields=[],
    )
    full_query_count = len(search_plan.queries)
    search_plan = _limit_search_plan(search_plan, max_queries)

    prompt = render_prompt(
        phase_3_search_intent,
        analysis_agent_request=state_tree.get("phase_3_strategy", {}).get("data_collection_spec", {}),
    )
    prompt_output: str | None = None
    prompt_error: str | None = None
    if show_prompt:
        print("=== search_agent system ===")
        print(search_system_prompt)
        print("=== search_agent user ===")
        print(prompt)
    if model is not None:
        try:
            response = await model.ainvoke([("system", search_system_prompt), ("user", prompt)])
            content = getattr(response, "content", response)
            prompt_output = "\n".join(str(item) for item in content) if isinstance(content, list) else str(content)
        except Exception as exc:
            prompt_error = f"{type(exc).__name__}: {exc}"
            print(f"Search Agent phase-4 prompt call failed and will be skipped: {prompt_error}")

    sources, provider_results = await _collect_sources(
        settings,
        request,
        benchmark,
        selected_set,
        tool_plan,
        search_plan,
        task_id="debug-phase4",
        max_sources=max_sources,
        fetch_details=fetch_details,
    )
    evidence = evidence_from_sources(sources)
    raw_data_sandbox = sources_to_sandbox(sources, evidence)
    state_tree = merge_updated_nodes(
        state_tree,
        {"updated_nodes": {"raw_data_sandbox": raw_data_sandbox}},
    )
    state_tree.setdefault("global_metadata", {})["status"] = "raw_data_collected"

    output_payload = {
        "input": str(input_path),
        "search_prompt": prompt if show_prompt else None,
        "search_prompt_output": prompt_output,
        "search_prompt_error": prompt_error,
        "full_search_plan_query_count": full_query_count,
        "used_search_plan_query_count": len(search_plan.queries),
        "search_plan": search_plan.model_dump(mode="json"),
        "source_provider_results": [item.model_dump(mode="json") for item in provider_results],
        "raw_sources": [source.model_dump(mode="json") for source in sources],
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "raw_data_sandbox": raw_data_sandbox,
        "state_tree": state_tree,
        "next_workflow_phase": "phase_4_synthesis",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved Search Agent phase-4 output to: {output_path}")
    print(f"SearchPlan queries: {len(search_plan.queries)} used / {full_query_count} generated")
    print(f"Raw sources: {len(sources)}")
    print(f"Evidence: {len(evidence)}")
    print(f"Next workflow_phase: {output_payload['next_workflow_phase']}")
    print("Source preview:")
    for index, source in enumerate(sources[:8], start=1):
        competitor = source.metadata.get("competitor")
        tool = source.metadata.get("tool")
        print(f"{index}. [{competitor or '-'} | {tool or '-'}] {source.title} -> {source.url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Search Agent phase 4 collection from Plan Agent phase-3 output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to plan_phase3_output.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write Search Agent phase-4 debug JSON.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print and save the rendered Search Agent phase-4 prompt.",
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
        help="Skip LLM calls and use fallback SearchPlan only.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=12,
        help="Maximum SearchPlan queries to execute. Use 0 for all queries.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=40,
        help="Maximum raw sources to collect.",
    )
    parser.add_argument(
        "--fetch-details",
        action="store_true",
        help="Fetch detail pages for collected URLs, like the full workflow does.",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.input,
            args.output,
            args.show_prompt,
            args.timeout,
            args.no_llm,
            args.max_queries,
            args.max_sources,
            args.fetch_details,
        )
    )


if __name__ == "__main__":
    main()
