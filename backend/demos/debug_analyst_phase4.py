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

from myagent.backend.app.agents.fallbacks import build_fallback_report
from myagent.backend.app.agents.llm import friendly_llm_error, get_chat_model, structured_ainvoke
from myagent.prompt.analyst_agent import analyst_system_prompt
from myagent.backend.app.agents.state_tree import merge_updated_nodes, report_to_phase_4
from myagent.backend.app.agents.strategy import apply_selection_to_profiles, scoring_profile_for
from myagent.backend.app.agents.tool_executor import execute_tools, tool_results_to_sandboxes
from myagent.backend.app.config import Settings
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CollectionPlan,
    CompetitorReport,
    CreateTaskRequest,
    EvidenceItem,
    EvidenceSignal,
    RawSource,
    SearchPlan,
    SelectedCompetitorSet,
    SourceProviderResult,
    ToolPlan,
)
from myagent.backend.app.agents.workflow import AnalystOutput


DEFAULT_INPUT = BACKEND_DIR / "demos" / "search_phase4_output.json"
DEFAULT_CONTEXT_INPUT = BACKEND_DIR / "demos" / "search_to_analyst_output.json"
DEFAULT_PLAN_INPUT = BACKEND_DIR / "demos" / "plan_phase3_output.json"
DEFAULT_OUTPUT = BACKEND_DIR / "demos" / "analyst_phase4_output.json"
DEFAULT_SYNTHESIS_OUTPUT = BACKEND_DIR / "demos" / "phase_4_synthesis_only.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _context_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if DEFAULT_PLAN_INPUT.exists():
        plan_payload = _load_json(DEFAULT_PLAN_INPUT)
        payload = {**plan_payload, **payload}
        if "state_tree" in plan_payload and "state_tree" in payload:
            merged_state_tree = {**plan_payload["state_tree"], **payload["state_tree"]}
            payload["state_tree"] = merged_state_tree
    if "request" in payload and "benchmark" in payload and "tool_plan" in payload:
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


def _json_for_llm(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


async def run(
    input_path: Path,
    output_path: Path,
    synthesis_output_path: Path,
    timeout_seconds: float | None,
    no_llm: bool,
) -> None:
    payload = _context_payload(_load_json(input_path))
    state_tree = payload["state_tree"]
    request = CreateTaskRequest.model_validate(payload["request"])
    benchmark = AnalysisBenchmark.model_validate(payload["benchmark"])
    selected_set = _selected_set_from_state_tree(state_tree, benchmark, payload)
    tool_plan = ToolPlan.model_validate(payload["tool_plan"])
    collection_plan = CollectionPlan.model_validate(payload["collection_plan"])
    search_plan = SearchPlan.model_validate(payload["search_plan"])
    sources = [RawSource.model_validate(item) for item in payload.get("raw_sources", [])]
    evidence = [EvidenceItem.model_validate(item) for item in payload.get("evidence", [])]
    signals = [EvidenceSignal.model_validate(item) for item in payload.get("evidence_signals", [])]
    provider_results = [
        SourceProviderResult.model_validate(item) for item in payload.get("source_provider_results", [])
    ]

    settings = Settings()
    if timeout_seconds is not None:
        settings.openai_timeout_seconds = timeout_seconds
    model = None if no_llm else get_chat_model(settings)

    tool_results = await execute_tools(
        model,
        benchmark,
        selected_set,
        tool_plan,
        evidence,
        search_plan=search_plan,
        signals=signals,
    )
    tool_sandboxes = tool_results_to_sandboxes(tool_results)

    structured = build_fallback_report(
        "debug-phase4",
        request,
        sources,
        benchmark=benchmark,
        selected_competitor_set=selected_set,
        tool_plan=tool_plan,
        collection_plan=collection_plan,
        search_plan=search_plan,
        source_provider_results=provider_results,
        tool_results=tool_results,
        tool_sandboxes=tool_sandboxes,
        risk_notice=payload.get("mock_data_notice"),
    )
    structured.evidence = evidence
    structured.analysis_benchmark = benchmark
    structured.selected_competitor_set = selected_set
    structured.tool_plan = tool_plan
    structured.collection_plan = collection_plan
    structured.search_plan = search_plan
    structured.source_provider_results = provider_results
    structured.tool_results = tool_results
    structured.tool_sandboxes = tool_sandboxes
    structured.competitors = apply_selection_to_profiles(structured.competitors, selected_set)

    analyst_error: str | None = None
    report: CompetitorReport = structured
    if model is not None:
        try:
            messages = [
                ("system", analyst_system_prompt),
                (
                    "user",
                    "Analyze this structured competitor knowledge and return a complete report JSON.\n"
                    f"Analysis benchmark: {benchmark.model_dump_json()}\n"
                    f"Target product: {benchmark.target_product}\n"
                    f"Our product: {request.our_product or benchmark.target_product}\n"
                    f"Market: {request.market or ''}\n"
                    f"Product category: {request.product_category or ''}\n"
                    f"Notes: {request.notes or ''}\n"
                    f"Structured knowledge: {_json_for_llm(structured)}",
                ),
            ]
            report = await structured_ainvoke(model, AnalystOutput, messages)
            report.task_id = "debug-phase4"
            report.target_product = benchmark.target_product
            report.evidence = evidence
        except Exception as exc:
            analyst_error = friendly_llm_error(exc)
            print(f"Analyst Agent phase-4 LLM call failed; using structured fallback report: {type(exc).__name__}: {exc}")

    report.analysis_benchmark = structured.analysis_benchmark
    report.selected_competitor_set = structured.selected_competitor_set
    report.tool_plan = structured.tool_plan
    report.collection_plan = structured.collection_plan
    report.search_plan = structured.search_plan
    report.source_provider_results = structured.source_provider_results
    report.tool_results = structured.tool_results
    report.tool_sandboxes = structured.tool_sandboxes
    report.competitors = apply_selection_to_profiles(report.competitors, selected_set)

    phase_4_synthesis = report_to_phase_4(report)
    state_tree = merge_updated_nodes(
        state_tree,
        {"updated_nodes": {"phase_4_synthesis": phase_4_synthesis}},
    )
    state_tree.setdefault("global_metadata", {})["status"] = "phase_4_synthesis_completed"

    output_payload = {
        "input": str(input_path),
        "analyst_error": analyst_error,
        "tool_results": [item.model_dump(mode="json") for item in tool_results],
        "tool_sandboxes": [item.model_dump(mode="json") for item in tool_sandboxes],
        "structured_report": structured.model_dump(mode="json"),
        "analysis_report": report.model_dump(mode="json"),
        "phase_4_synthesis": phase_4_synthesis,
        "state_tree": state_tree,
        "next_workflow_phase": "checkpoint_2",
        "qa_checkpoint": "checkpoint_2",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    synthesis_payload = {
        "input": str(input_path),
        "analyst_error": analyst_error,
        "phase_4_synthesis": phase_4_synthesis,
        "analysis_summary": {
            "target_product": report.target_product,
            "competitors": [item.name for item in report.competitors],
            "finding_count": len(report.findings),
            "dynamic_analysis_count": len(phase_4_synthesis.get("dynamic_analyses", [])),
            "recommendations": report.recommendations,
            "risk_notice": report.risk_notice,
        },
        "next_workflow_phase": "checkpoint_2",
        "qa_checkpoint": "checkpoint_2",
    }
    synthesis_output_path.parent.mkdir(parents=True, exist_ok=True)
    synthesis_output_path.write_text(
        json.dumps(synthesis_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved Analyst Agent phase-4 output to: {output_path}")
    print(f"Saved phase_4_synthesis JSON to: {synthesis_output_path}")
    print(f"Tool results: {len(tool_results)}")
    print(f"Competitor profiles: {len(report.competitors)}")
    print(f"Findings: {len(report.findings)}")
    print(f"Dynamic analyses: {len(phase_4_synthesis.get('dynamic_analyses', []))}")
    print(f"Next QA checkpoint: {output_payload['qa_checkpoint']}")
    print("Phase 4 synthesis preview:")
    print(json.dumps(phase_4_synthesis, ensure_ascii=False, indent=2)[:3000])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Analyst Agent phase 4 synthesis from Search Agent phase-4 output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to search_phase4_output.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write Analyst Agent phase-4 debug JSON.",
    )
    parser.add_argument(
        "--synthesis-output",
        type=Path,
        default=DEFAULT_SYNTHESIS_OUTPUT,
        help="Path to write the compact phase_4_synthesis result JSON.",
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
        help="Skip LLM calls and use fallback structured analysis.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.synthesis_output, args.timeout, args.no_llm))


if __name__ == "__main__":
    main()
