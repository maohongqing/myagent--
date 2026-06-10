from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from myagent.backend.app.agents.candidate_selector import select_candidates_with_llm_or_rules
from myagent.backend.app.agents.llm import get_chat_model
from myagent.backend.app.agents.state_tree import selected_set_to_phase_2
from myagent.backend.app.agents.strategy import infer_benchmark, score_and_select_candidates
from myagent.backend.app.config import Settings
from myagent.backend.app.models import CandidateCompetitor, CreateTaskRequest


DEFAULT_INPUT = "我想做一个二次元手办交易平台，目前主要用作决策支持"
DEFAULT_CANDIDATE_POOL = BACKEND_DIR / "demos" / "search_agent_output.txt"
DEFAULT_OUTPUT = BACKEND_DIR / "demos" / "analyst_phase2_output.json"

TYPE_MAP = {
    "头部直接竞品": "head_direct",
    "直接竞品": "head_direct",
    "行业龙头": "head_direct",
    "追赶型黑马竞品": "challenger",
    "追赶型黑马": "challenger",
    "黑马竞品": "challenger",
    "间接/跨界竞品": "indirect_cross",
    "间接竞品": "indirect_cross",
    "跨界竞品": "indirect_cross",
    "潜在替代品": "potential_substitute",
    "替代品": "potential_substitute",
}


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Candidate pool must be a JSON array.")
    return [item for item in parsed if isinstance(item, dict)]


def _candidate_type(raw_type: str | None) -> str:
    text = (raw_type or "").strip()
    if text in TYPE_MAP:
        return TYPE_MAP[text]
    for label, competitor_type in TYPE_MAP.items():
        if label in text:
            return competitor_type
    return "challenger"


def _load_candidates(path: Path) -> list[CandidateCompetitor]:
    items = _extract_json_array(path.read_text(encoding="utf-8"))
    candidates: list[CandidateCompetitor] = []
    seen: set[str] = set()
    for item in items:
        name = str(item.get("name") or "").strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        description = str(item.get("description") or "").strip()
        background = str(item.get("developer_background") or "").strip()
        if background:
            description = f"{description} 开发商背景：{background}".strip()
        candidates.append(
            CandidateCompetitor(
                name=name,
                description=description,
                competitor_type=_candidate_type(str(item.get("type") or "")),
                tags=["debug_search_agent_output"],
                selection_reason="来自 search_agent 调试输出的候选池。",
            )
        )
    return candidates


async def run(
    raw_description: str,
    candidate_pool_path: Path,
    output_path: Path,
    timeout_seconds: float | None,
    no_llm: bool,
) -> None:
    request = CreateTaskRequest(raw_description=raw_description)
    benchmark, clarification = infer_benchmark(request)
    if not benchmark:
        raise RuntimeError(f"Benchmark could not be inferred: {clarification}")

    candidates = _load_candidates(candidate_pool_path)
    if not candidates:
        raise RuntimeError(f"No candidates loaded from {candidate_pool_path}")

    model = None
    mode = "rule_selection"
    if not no_llm:
        settings = Settings()
        if timeout_seconds is not None:
            settings.openai_timeout_seconds = timeout_seconds
        model = get_chat_model(settings)
        mode = "llm_selection" if model else "rule_selection_no_model"

    if model is None:
        selected_set = score_and_select_candidates(benchmark, candidates, budget=5)
    else:
        selected_set = await select_candidates_with_llm_or_rules(model, benchmark, candidates, [], budget=5)

    phase_2_targets = selected_set_to_phase_2(selected_set)
    payload = {
        "mode": mode,
        "candidate_pool_path": str(candidate_pool_path),
        "candidate_count": len(candidates),
        "benchmark": benchmark.model_dump(mode="json"),
        "selected_competitor_set": selected_set.model_dump(mode="json"),
        "state_tree_patch": {
            "updated_nodes": {
                "phase_2_targets": phase_2_targets,
            }
        },
        "phase_2_targets": phase_2_targets,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Loaded candidates: {len(candidates)}")
    print(f"Mode: {mode}")
    print(f"Saved analyst phase-2 output to: {output_path}")
    print("Selected competitors:")
    for index, item in enumerate(selected_set.selected, start=1):
        print(f"{index}. {item.name} [{item.competitor_type}] score={item.score}")
        print(f"   {item.selection_reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Analyst Agent phase-2 selection from a Search Agent candidate pool."
    )
    parser.add_argument(
        "raw_description",
        nargs="?",
        default=DEFAULT_INPUT,
        help="User input text used to infer the analysis benchmark.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=DEFAULT_CANDIDATE_POOL,
        help="Path to search_agent_output.txt or another JSON candidate pool.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the analyst phase-2 JSON output.",
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
        help="Skip the LLM and use rule-based scoring only.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.raw_description, args.candidate_pool, args.output, args.timeout, args.no_llm))


if __name__ == "__main__":
    main()
