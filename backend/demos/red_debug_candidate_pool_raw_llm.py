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

from myagent.backend.app.agents.candidate_selector import _candidate_pool_messages
from myagent.backend.app.agents.collector import collect_candidate_sources
from myagent.backend.app.agents.llm import get_chat_model
from myagent.backend.app.agents.strategy import build_candidate_queries, infer_benchmark
from myagent.backend.app.config import Settings
from myagent.backend.app.models import CreateTaskRequest


DEFAULT_INPUT = "我想做一个二次元手办交易平台，目前主要用作决策支持"
DEFAULT_OUTPUT = BACKEND_DIR / "demos" / "candidate_pool_raw_llm_output.txt"
DEFAULT_CONTEXT_OUTPUT = BACKEND_DIR / "demos" / "candidate_pool_raw_llm_context.json"


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


async def run(
    raw_description: str,
    output_path: Path,
    context_output_path: Path,
    timeout_seconds: float | None,
    max_sources: int,
    max_candidates: int,
    show_messages: bool,
) -> None:
    request = CreateTaskRequest(raw_description=raw_description)
    benchmark, clarification = infer_benchmark(request)
    if not benchmark:
        raise RuntimeError(f"Benchmark could not be inferred: {clarification}")
    request.target_product = request.target_product or benchmark.target_product

    settings = Settings()
    if timeout_seconds is not None:
        settings.openai_timeout_seconds = timeout_seconds
    model = get_chat_model(settings)
    if model is None:
        raise RuntimeError(
            "No chat model is available. Check OPENAI_API_KEY, OPENAI_BASE_URL, "
            "OPENAI_MODEL, and whether backend/.env is loaded."
        )

    queries = build_candidate_queries(benchmark, request)
    print("Collecting candidate sources...")
    sources = await collect_candidate_sources(queries, settings, max_sources=max_sources)
    print(f"Collected sources: {len(sources)}")

    messages = _candidate_pool_messages(benchmark, request, sources, max_candidates)
    if show_messages:
        print("=== raw candidate pool messages ===")
        for role, content in messages:
            print(f"--- {role} ---")
            print(content)

    print("Calling raw LLM without structured schema...")
    response = await model.ainvoke(messages)
    text = _response_text(response).strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    context_output_path.write_text(
        json.dumps(
            {
                "benchmark": benchmark.model_dump(mode="json"),
                "request": request.model_dump(mode="json"),
                "queries": queries,
                "sources": [source.model_dump(mode="json") for source in sources],
                "messages": [{"role": role, "content": content} for role, content in messages],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved raw LLM output to: {output_path}")
    print(f"Saved prompt/source context to: {context_output_path}")
    print(f"Characters: {len(text)}")
    if len(text) <= 5000:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call the candidate-pool LLM prompt without structured output and save the raw response."
    )
    parser.add_argument(
        "raw_description",
        nargs="?",
        default=DEFAULT_INPUT,
        help="User input text used to infer the analysis benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the raw LLM response txt.",
    )
    parser.add_argument(
        "--context-output",
        type=Path,
        default=DEFAULT_CONTEXT_OUTPUT,
        help="Path to write the queries, sources, and prompt messages JSON.",
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
        help="Candidate budget passed into the LLM prompt.",
    )
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="Print the raw system/user messages sent to the LLM.",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.raw_description,
            args.output,
            args.context_output,
            args.timeout,
            args.max_sources,
            args.max_candidates,
            args.show_messages,
        )
    )


if __name__ == "__main__":
    main()
