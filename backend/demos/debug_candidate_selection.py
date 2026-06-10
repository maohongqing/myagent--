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

from myagent.backend.app.agents.collector import collect_candidate_sources
from myagent.backend.app.agents.candidate_selector import generate_candidate_pool_with_llm_or_rules, select_candidates_with_llm_or_rules
from myagent.backend.app.agents.llm import friendly_llm_error, get_chat_model
from myagent.backend.app.agents.state_tree import benchmark_to_phase_1, selected_set_to_phase_2
from myagent.backend.app.agents.strategy import (
    build_candidate_queries,
    build_candidates_from_sources,
    infer_benchmark,
    score_and_select_candidates,
)
from myagent.backend.app.config import Settings
from myagent.backend.app.models import CandidateCompetitor, CreateTaskRequest, RawSource


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


def resolve_path(value: str | Path, *, base: Path = BACKEND_DIR, must_exist: bool = False) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        WORKSPACE_ROOT / path,
        BACKEND_DIR / path,
        THIS_FILE.parent / path,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    if must_exist:
        checked = "\n".join(f"- {candidate.resolve()}" for candidate in candidates)
        raise FileNotFoundError(f"Cannot find path: {value}\nChecked:\n{checked}")
    return (base / path).resolve()


def make_output_dir(output_root: str | Path) -> Path:
    root = resolve_path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = root / f"{timestamp}_candidate_selection_debug"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def load_request(input_path: str | Path) -> tuple[CreateTaskRequest, Path]:
    path = resolve_path(input_path, must_exist=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CreateTaskRequest.model_validate(payload), path


def candidate_summary(candidate: CandidateCompetitor) -> dict[str, Any]:
    is_llm_supplement = "llm_supplement" in candidate.tags
    is_llm_selected = "llm_selected" in candidate.tags or is_llm_supplement
    return {
        "name": candidate.name,
        "competitor_type": candidate.competitor_type,
        "score": candidate.score,
        "selected_by_llm": is_llm_selected,
        "supplemented_by_llm": is_llm_supplement,
        "description": candidate.description[:240],
        "source_titles": candidate.source_titles[:3],
        "source_urls": candidate.source_urls[:3],
        "tags": candidate.tags,
        "rank_factors": candidate.rank_factors.model_dump(mode="json"),
        "selection_reason": candidate.selection_reason,
    }


def source_summary(source: RawSource) -> dict[str, Any]:
    return {
        "title": source.title,
        "url": source.url,
        "source_type": source.source_type,
        "provider": source.metadata.get("provider"),
        "query": source.metadata.get("query"),
        "content_preview": source.content[:240],
    }


async def run_debug(args: argparse.Namespace) -> int:
    output_dir = make_output_dir(args.output_root)
    request, input_path = load_request(args.input)
    settings = load_settings(output_dir, args.duckduckgo_max_results)
    env_file = BACKEND_DIR / ".env"

    benchmark, clarification = infer_benchmark(request)
    if clarification or benchmark is None:
        write_json(output_dir / "clarification.json", clarification)
        print("Clarification is required before candidate selection.")
        print(f"Output dir: {output_dir}")
        return 2

    queries = build_candidate_queries(benchmark, request)
    sources = await collect_candidate_sources(queries, settings, max_sources=args.max_sources)
    rule_candidates = build_candidates_from_sources(benchmark, request, sources)
    model = None if args.no_llm else get_chat_model(settings)
    candidates, candidate_pool_notes, candidate_pool_mode = await generate_candidate_pool_with_llm_or_rules(
        model,
        benchmark,
        request,
        sources,
        max_candidates=args.max_candidates,
    )
    rule_selected_set = score_and_select_candidates(
        benchmark,
        [candidate.model_copy(deep=True) for candidate in rule_candidates],
        budget=args.budget,
    )
    selection_mode = "llm_semantic_selection" if model else "rule_based_selection"
    try:
        selected_set = await select_candidates_with_llm_or_rules(
            model,
            benchmark,
            candidates,
            sources,
            budget=args.budget,
        )
    except Exception as exc:
        selection_mode = "rule_based_selection_after_llm_error"
        selected_set = score_and_select_candidates(benchmark, candidates, budget=args.budget)
        selected_set.fallback_notes.append(f"大模型语义筛选失败，已回退到规则评分：{friendly_llm_error(exc)}")

    state_tree_preview = {
        "phase_1_intent": benchmark_to_phase_1(benchmark),
        "phase_2_targets": selected_set_to_phase_2(selected_set),
    }
    selected_names = {candidate.name for candidate in selected_set.selected}
    supplemented = [candidate for candidate in selected_set.selected if "llm_supplement" in candidate.tags]
    selected_from_rule_pool = [candidate for candidate in selected_set.selected if candidate.name in {item.name for item in candidates}]
    output = {
        "flow_description": [
            "1. search_agent 采集网页来源，生成 sources。",
            "2. 大模型参考采集网页生成 candidate_pool；规则抽取只作为模型失败时的兜底。",
            "3. 规则候选池仍会生成 rule_fallback_candidate_pool，方便对照和兜底。",
            "4. analyst_agent 再调用大模型做语义判断、生成入选理由；候选不足时继续由大模型补充真实竞品。",
            "5. 最终 selected_competitor_set 写入 JSON State Tree 的 phase_2_targets。",
        ],
        "input_file": str(input_path),
        "env_file": str(env_file),
        "benchmark": benchmark,
        "selection_mode": selection_mode,
        "queries": queries,
        "stage_1_collected_sources": {
            "source_count": len(sources),
            "sources": [source_summary(source) for source in sources],
        },
        "stage_2_llm_generated_candidate_pool": {
            "candidate_pool_mode": candidate_pool_mode,
            "candidate_pool_notes": candidate_pool_notes,
            "candidate_count": len(candidates),
            "candidates": [candidate_summary(candidate) for candidate in candidates],
        },
        "stage_3_rule_fallback_candidate_pool": {
            "candidate_count": len(rule_candidates),
            "candidates": [candidate_summary(candidate) for candidate in rule_candidates],
        },
        "stage_4_rule_baseline_selection": {
            "selected_count": len(rule_selected_set.selected),
            "selected": [candidate_summary(candidate) for candidate in rule_selected_set.selected],
            "fallback_notes": rule_selected_set.fallback_notes,
        },
        "stage_5_llm_semantic_selection": {
            "llm_enabled": model is not None,
            "selection_mode": selection_mode,
            "selected_count": len(selected_set.selected),
            "selected": [candidate_summary(candidate) for candidate in selected_set.selected],
            "selected_from_rule_pool": [candidate_summary(candidate) for candidate in selected_from_rule_pool],
            "supplemented_by_llm": [candidate_summary(candidate) for candidate in supplemented],
            "selected_names": list(selected_names),
            "fallback_notes": selected_set.fallback_notes,
        },
        "fallback_notes": selected_set.fallback_notes,
        "state_tree_preview": state_tree_preview,
    }

    write_json(output_dir / "candidate_selection_debug.json", output)
    write_json(output_dir / "01_collected_sources.json", sources)
    write_json(output_dir / "02_llm_generated_candidate_pool.json", candidates)
    write_json(output_dir / "03_rule_fallback_candidate_pool.json", rule_candidates)
    write_json(output_dir / "04_rule_baseline_selection.json", rule_selected_set)
    write_json(output_dir / "05_llm_semantic_selection.json", selected_set)
    write_json(output_dir / "06_state_tree_preview.json", state_tree_preview)
    write_json(output_dir / "sources.json", sources)
    write_json(output_dir / "candidates.json", candidates)
    write_json(output_dir / "selected_competitor_set.json", selected_set)
    write_json(output_dir / "state_tree_preview.json", state_tree_preview)

    print("Candidate selection debug")
    print(f"Input file: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Env file: {env_file}")
    print(f"Target product: {benchmark.target_product}")
    print(f"Analysis purpose: {benchmark.analysis_purpose}")
    print(f"Candidate pool mode: {candidate_pool_mode}")
    print(f"Selection mode: {selection_mode}")
    print(f"Sources: {len(sources)}")
    print(f"LLM-generated candidates: {len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        print(f"- C{index}: {candidate.name} | {candidate.competitor_type} | score={candidate.score}")
    print(f"Rule fallback candidates: {len(rule_candidates)}")
    for index, candidate in enumerate(rule_candidates, start=1):
        print(f"- F{index}: {candidate.name} | {candidate.competitor_type} | score={candidate.score}")
    print(f"Rule fallback baseline selected: {len(rule_selected_set.selected)}")
    for index, candidate in enumerate(rule_selected_set.selected, start=1):
        print(f"- R{index}: {candidate.name} | {candidate.competitor_type} | score={candidate.score}")
    print(f"Final selected after LLM semantic step: {len(selected_set.selected)}")
    for index, candidate in enumerate(selected_set.selected, start=1):
        source = "LLM补充" if "llm_supplement" in candidate.tags else "规则池/LLM选择"
        print(f"- S{index}: {candidate.name} | {candidate.competitor_type} | score={candidate.score} | {source}")
        if candidate.selection_reason:
            print(f"  reason: {candidate.selection_reason}")
    if selected_set.fallback_notes:
        print("Fallback notes:")
        for note in selected_set.fallback_notes:
            print(f"- {note}")
    print(f"Full result: {output_dir / 'candidate_selection_debug.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug candidate collection, sanitization, and final target selection.")
    parser.add_argument("--input", default="demos/sample_task.json", help="CreateTaskRequest JSON file.")
    parser.add_argument("--output-root", default=str(BACKEND_DIR / "demo_outputs"))
    parser.add_argument("--duckduckgo-max-results", type=int, default=3)
    parser.add_argument("--max-sources", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM candidate pool generation and semantic selection.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run_debug(args)))


if __name__ == "__main__":
    main()
