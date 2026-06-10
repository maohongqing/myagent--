from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

try:
    from myagent.search.evidence import apply_quote_shots_to_extraction, capture_quote_shots_for_extraction
except ModuleNotFoundError:
    def capture_quote_shots_for_extraction(*_args, **_kwargs):
        return {}

    def apply_quote_shots_to_extraction(*_args, **_kwargs):
        return None

from myagent.backend.app.agents.chunk_filter import filter_relevant_sources
from myagent.backend.app.agents.llm import retry_structured_ainvoke
from myagent.backend.app.agents.prompt_bridge import (
    candidate_chunk_generation_prompt,
)
from myagent.backend.app.agents.strategy import build_candidates_from_sources, score_and_select_candidates, scoring_profile_for
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CandidateGapRequest,
    CompetitorType,
    CreateTaskRequest,
    RawSource,
    SelectedCompetitorSet,
    SourceRef,
)


class LLMSourceRef(BaseModel):
    source_id: str | None = Field(default=None, max_length=120)
    chunk_id: str | None = Field(default=None, max_length=120)
    url: str = Field(default="", max_length=1000)
    title: str = Field(default="Untitled source", max_length=300)
    quote: str = Field(default="", max_length=1200)
    _resolved_evidence_screenshot_paths: list[str] = PrivateAttr(default_factory=list)


class LLMSelectedCandidate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    competitor_type: CompetitorType = "challenger"
    description: str = Field(default="", max_length=800)
    selection_reason: str = Field(..., min_length=1, max_length=1000)
    evidence_source_urls: list[str] = Field(default_factory=list, max_length=8)
    source_refs: list[LLMSourceRef] = Field(default_factory=list, max_length=8)
    confidence: float = Field(default=0.55, ge=0, le=1)
    is_supplemented: bool = False


class LLMCandidateSelectionOutput(BaseModel):
    selected: list[LLMSelectedCandidate] = Field(default_factory=list, max_length=5)
    fallback_notes: list[str] = Field(default_factory=list, max_length=8)
    gap_request: CandidateGapRequest | None = None


class LLMCandidatePoolOutput(BaseModel):
    """Legacy compatibility container for the retired whole-pool LLM step."""

    candidates: list[LLMSelectedCandidate] = Field(default_factory=list, max_length=30)
    generation_notes: list[str] = Field(default_factory=list, max_length=8)


class LLMCandidateChunkOutput(BaseModel):
    candidates: list[LLMSelectedCandidate] = Field(default_factory=list, max_length=8)
    generation_notes: list[str] = Field(default_factory=list, max_length=4)


_chunk_log_locks: dict[str, Lock] = defaultdict(Lock)
MAX_PROMPT_CHUNK_CHARS = 3000


async def generate_candidate_pool_with_llm_or_rules(
    model,
    benchmark: AnalysisBenchmark,
    request: CreateTaskRequest,
    sources: list[RawSource],
    *,
    max_candidates: int = 30,
    existing_candidate_names: list[str] | None = None,
) -> tuple[list[CandidateCompetitor], list[str], str]:
    """Generate candidate pool with LLM; rules are fallback only."""
    if model is None:
        candidates = build_candidates_from_sources(benchmark, request, sources)
        return candidates[:max_candidates], ["未配置大模型，使用规则候选池兜底。"], "rule_fallback_no_llm"

    try:
        llm_output = await retry_structured_ainvoke(
            model,
            LLMCandidatePoolOutput,
            _candidate_pool_messages(benchmark, request, sources, max_candidates, existing_candidate_names),
        )
    except Exception as exc:
        candidates = build_candidates_from_sources(benchmark, request, sources)
        return candidates[:max_candidates], [f"大模型候选池生成失败，使用规则兜底：{exc}"], "rule_fallback_llm_error"
    candidates = _candidate_pool_from_llm_output(benchmark, llm_output, max_candidates)
    if candidates:
        return candidates, llm_output.generation_notes, "llm_candidate_pool"

    return [], _dedupe([*llm_output.generation_notes, "大模型未识别到有效候选，未启用规则兜底。"])[:8], "empty_llm"


async def generate_candidate_pool_from_chunks_with_llm_or_rules(
    model,
    benchmark: AnalysisBenchmark,
    request: CreateTaskRequest,
    sources: list[RawSource],
    *,
    max_candidates: int = 30,
    task_output_dir: Path | None = None,
    task_id: str | None = None,
    existing_candidate_names: list[str] | None = None,
) -> tuple[list[CandidateCompetitor], list[str], str]:
    """Analyze each source chunk with the candidate chunk prompt, then merge candidates."""
    if model is None:
        candidates = build_candidates_from_sources(benchmark, request, sources)
        return candidates[:max_candidates], ["未配置大模型，使用规则候选池兜底。"], "rule_fallback_no_llm"

    raw_sources = [source for source in sources if (source.content or "").strip()]
    filtered_sources, filter_stats = filter_relevant_sources(
        raw_sources,
        object_name=benchmark.target_product,
        context_terms=[
            benchmark.target_product,
            benchmark.analysis_goal,
            request.market or "",
            request.product_category or "",
            request.geography or "",
            *(request.competitors or []),
        ],
        expected_fields=[
            "positioning",
            "core_features",
            "growth_signals",
            "funding_or_org_signals",
            "user_feedback",
            "risks",
        ],
        min_chars=80,
    )
    indexed_sources = [(index, source) for index, source in enumerate(filtered_sources, start=1)]
    if not indexed_sources:
        return [], ["没有可分析的网页 chunk。"], "empty_sources"

    async def analyze(
        index: int,
        source: RawSource,
    ) -> tuple[int, RawSource, list[tuple[str, str]], LLMCandidateChunkOutput | None, str | None]:
        messages = _candidate_chunk_messages(benchmark, request, source, index, existing_candidate_names)
        started = time.perf_counter()
        try:
            output = await retry_structured_ainvoke(model, LLMCandidateChunkOutput, messages)
            return index, source, messages, output, None
        except Exception as exc:
            _save_chunk_llm_call(
                task_output_dir,
                task_id,
                "search_agent",
                "candidate_chunk_generation",
                source,
                index,
                messages,
                error={"error": str(exc)},
                started_at=started,
            )
            return index, source, messages, None, str(exc)

    results = await asyncio.gather(*(analyze(index, source) for index, source in indexed_sources))
    quote_shots = await asyncio.to_thread(_capture_quote_shots_for_chunk_results, results, task_output_dir, task_id)
    llm_candidates: list[LLMSelectedCandidate] = []
    notes: list[str] = [
        f"chunk filter kept {filter_stats.kept_count}/{filter_stats.original_count}; "
        f"llm_calls={filter_stats.llm_call_count}; skipped={dict(filter_stats.skipped_reasons)}"
    ]
    errors: list[str] = []
    failed_sources: list[RawSource] = []
    for index, source, messages, output, error in sorted(results, key=lambda item: item[0]):
        if error:
            errors.append(f"{source.title}: {error}")
            failed_sources.append(source)
            continue
        if output is None:
            continue
        notes.extend(output.generation_notes)
        normalized_candidates = _normalize_chunk_candidate_refs(output.candidates, source, quote_shots.get(index, {}))
        normalized_output = LLMCandidateChunkOutput(
            candidates=normalized_candidates,
            generation_notes=output.generation_notes,
        )
        _save_chunk_llm_call(
            task_output_dir,
            task_id,
            "search_agent",
            "candidate_chunk_generation",
            source,
            index,
            messages,
            output=output,
            postprocessed_output=_chunk_output_payload(normalized_output, include_quote_shots=True),
        )
        llm_candidates.extend(normalized_candidates)

    if llm_candidates:
        candidates = _candidate_pool_from_llm_candidates(benchmark, llm_candidates, max_candidates)
        if candidates:
            if errors and len(candidates) < max_candidates:
                fallback = build_candidates_from_sources(benchmark, request, failed_sources)
                if fallback:
                    candidates = _dedupe_candidates([*candidates, *fallback])[:max_candidates]
                    notes.append(f"{len(errors)} 个 chunk 调用失败，已仅对失败 chunk 使用规则兜底。")
            mode = "llm_candidate_chunk_pool"
            if errors:
                notes.append(f"{len(errors)} 个 chunk 分析失败。")
            return candidates, _dedupe(notes)[:8], mode

    if failed_sources:
        fallback = build_candidates_from_sources(benchmark, request, failed_sources)
        if fallback:
            return fallback[:max_candidates], ["chunk 大模型调用失败，已仅对失败 chunk 使用规则兜底。", *errors[:3]], "rule_fallback_llm_error"
        return [], ["chunk 大模型调用失败，但规则兜底未生成有效候选。", *errors[:3]], "empty_chunk_llm_error"
    return [], _dedupe([*notes, "chunk 大模型未识别到有效候选，未启用规则兜底。"])[:8], "empty_chunk_llm"


async def select_candidates_with_llm_or_rules(
    model,
    benchmark: AnalysisBenchmark,
    candidates: list[CandidateCompetitor],
    sources: list[RawSource],
    *,
    budget: int = 5,
    existing_candidate_names: list[str] | None = None,
) -> SelectedCompetitorSet:
    """Compatibility wrapper: final selection is rule-based and does not call an LLM."""
    selected_set = score_and_select_candidates(benchmark, candidates, budget=budget)
    if len(selected_set.selected) < budget:
        selected_set.gap_request = _build_gap_request(benchmark, selected_set.selected, candidates, budget)
    return selected_set


def build_rule_based_selection(
    benchmark: AnalysisBenchmark,
    candidates: list[CandidateCompetitor],
    *,
    budget: int = 5,
) -> SelectedCompetitorSet:
    return score_and_select_candidates(benchmark, candidates, budget=budget)


def _known_candidate_names(request: CreateTaskRequest, existing_candidate_names: list[str] | None = None) -> list[str]:
    return _dedupe([*request.competitors, *(existing_candidate_names or [])])[:80]


def _candidate_pool_messages(
    benchmark: AnalysisBenchmark,
    request: CreateTaskRequest,
    sources: list[RawSource],
    max_candidates: int,
    existing_candidate_names: list[str] | None = None,
) -> list[tuple[str, str]]:
    raise RuntimeError("候选池全量汇总 prompt 已下线，请使用 chunk 级候选生成流程。")


def _candidate_chunk_messages(
    benchmark: AnalysisBenchmark,
    request: CreateTaskRequest,
    source: RawSource,
    call_index: int,
    existing_candidate_names: list[str] | None = None,
) -> list[tuple[str, str]]:
    known_candidate_names = _known_candidate_names(request, existing_candidate_names)
    payload = {
        "analysis_intent": benchmark.model_dump(mode="json"),
        "request_context": {
            "target_product": request.target_product,
            "raw_description": request.raw_description,
            "market": request.market,
            "product_category": request.product_category,
            "geography": request.geography,
            "analysis_goal": request.analysis_goal,
            "known_competitors": request.competitors,
        },
        "existing_candidate_names": known_candidate_names,
        "source": _source_payload(source),
        "call_index": call_index,
    }
    system_prompt = candidate_chunk_generation_prompt
    if known_candidate_names:
        system_prompt += "\n\n已收集候选名称（不要重复输出）： " + "、".join(known_candidate_names)
    return [
        ("system", system_prompt),
        (
            "user",
            "请分析下面这个完整网页 chunk；如果没有明确竞品证据，返回空 candidates。\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}",
        ),
    ]


def _candidate_selection_messages(
    benchmark: AnalysisBenchmark,
    candidates: list[CandidateCompetitor],
    sources: list[RawSource],
    budget: int,
    existing_candidate_names: list[str] | None = None,
) -> list[tuple[str, str]]:
    raise RuntimeError("候选选择大模型 prompt 已下线，请使用规则评分选择流程。")


def _candidate_pool_from_llm_output(
    benchmark: AnalysisBenchmark,
    llm_output: LLMCandidatePoolOutput,
    max_candidates: int,
) -> list[CandidateCompetitor]:
    return _candidate_pool_from_llm_candidates(benchmark, llm_output.candidates, max_candidates)


async def generate_candidate_pool_with_llm_or_rules(
    model,
    benchmark: AnalysisBenchmark,
    request: CreateTaskRequest,
    sources: list[RawSource],
    *,
    max_candidates: int = 30,
    existing_candidate_names: list[str] | None = None,
) -> tuple[list[CandidateCompetitor], list[str], str]:
    """Compatibility wrapper: the legacy whole-pool LLM step is retired and now delegates to chunk analysis."""
    return await generate_candidate_pool_from_chunks_with_llm_or_rules(
        model,
        benchmark,
        request,
        sources,
        max_candidates=max_candidates,
        existing_candidate_names=existing_candidate_names,
    )


def _candidate_pool_from_llm_candidates(
    benchmark: AnalysisBenchmark,
    items: list[LLMSelectedCandidate],
    max_candidates: int,
) -> list[CandidateCompetitor]:
    candidates: list[CandidateCompetitor] = []
    used: set[str] = set()
    for item in items:
        key = _name_key(item.name)
        if not key or key in used or is_invalid_candidate_name(item.name, benchmark):
            continue
        source_refs = _source_refs_from_llm(item.source_refs)
        source_urls = _dedupe([*item.evidence_source_urls, *[ref.url for ref in source_refs]])
        source_titles = _dedupe([ref.title for ref in source_refs])
        tags = ["llm_candidate_pool"]
        if not source_refs:
            tags.append("missing_source_refs")
        confidence = item.confidence if source_refs else min(item.confidence, 0.35)
        candidates.append(
            CandidateCompetitor(
                name=item.name.strip(),
                competitor_type=item.competitor_type,
                description=item.description or "Candidate generated by the LLM from collected web sources.",
                source_urls=source_urls,
                source_titles=source_titles,
                source_refs=source_refs,
                tags=tags,
                score=round(confidence, 4),
                selection_reason=item.selection_reason,
            )
        )
        used.add(key)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _candidate_payload(candidate: CandidateCompetitor) -> dict:
    return {
        "name": candidate.name,
        "competitor_type": candidate.competitor_type,
        "description": candidate.description[:600],
        "source_titles": candidate.source_titles[:5],
        "source_urls": candidate.source_urls[:5],
        "source_refs": [_source_ref_payload(ref) for ref in candidate.source_refs[:5]],
        "score": candidate.score,
        "rank_factors": candidate.rank_factors.model_dump(mode="json"),
        "tags": candidate.tags,
    }


def _source_ref_payload(ref: SourceRef) -> dict[str, Any]:
    return {
        "source_id": ref.source_id,
        "chunk_id": ref.chunk_id,
        "url": ref.url,
        "title": ref.title,
        "quote": ref.quote,
    }


def _source_payload(source: RawSource) -> dict:
    return {
        "content": (source.content or "")[:MAX_PROMPT_CHUNK_CHARS],
    }


def _normalize_chunk_candidate_refs(
    items: list[LLMSelectedCandidate],
    source: RawSource,
    quote_shot_paths: dict[tuple[int, int], list[str]] | None = None,
) -> list[LLMSelectedCandidate]:
    normalized: list[LLMSelectedCandidate] = []
    for candidate_index, item in enumerate(items, start=1):
        refs = list(item.source_refs)
        if not refs:
            quote = _best_quote_for_candidate(item.name, source.content)
            if not quote:
                continue
            refs = [LLMSourceRef(quote=quote)]
        resolved_refs: list[LLMSourceRef] = []
        for ref_index, ref in enumerate(refs, start=1):
            resolved = ref.model_copy(
                update={
                    "source_id": ref.source_id or source.id,
                    "chunk_id": ref.chunk_id or source.metadata.get("chunk_id"),
                    "url": ref.url or source.url,
                    "title": ref.title or source.title,
                }
            )
            resolved._resolved_evidence_screenshot_paths = _dedupe((quote_shot_paths or {}).get((candidate_index, ref_index), []))
            resolved_refs.append(resolved)
        refs = resolved_refs
        normalized.append(item.model_copy(update={"source_refs": refs, "evidence_source_urls": _dedupe([*item.evidence_source_urls, source.url])}))
    return normalized


def _best_quote_for_candidate(name: str, content: str) -> str:
    cleaned = content or ""
    if not cleaned.strip():
        return ""
    index = cleaned.lower().find((name or "").lower())
    if index < 0:
        return cleaned[:220].strip()
    start = max(0, index - 80)
    end = min(len(cleaned), index + len(name) + 160)
    return cleaned[start:end].strip()


def _refs_for_quote_capture(candidate: LLMSelectedCandidate, source: RawSource) -> list[LLMSourceRef]:
    refs = list(candidate.source_refs)
    if refs:
        return refs
    quote = _best_quote_for_candidate(candidate.name, source.content)
    return [LLMSourceRef(quote=quote)] if quote else []


def _capture_quote_shots_for_chunk_results(
    results: list[tuple[int, RawSource, list[tuple[str, str]], LLMCandidateChunkOutput | None, str | None]],
    task_output_dir: Path | None,
    task_id: str | None,
) -> dict[int, dict[tuple[int, int], list[str]]]:
    if task_output_dir is None or not task_id:
        return {}

    data_dir = task_output_dir.parent
    grouped: dict[str, dict[str, Any]] = {}
    lookup: dict[tuple[str, str], tuple[int, int, int]] = {}
    max_quote_shots_per_run = 24

    for call_index, source, _messages, output, error in results:
        if error or output is None:
            continue
        if not source.metadata.get("run_dir"):
            continue
        run_dir = _quote_run_dir(source, task_output_dir, task_id)
        run_key = str(run_dir.resolve())
        group = grouped.setdefault(run_key, {"run_dir": run_dir, "facts": []})
        facts = group["facts"]
        for candidate_index, candidate in enumerate(output.candidates, start=1):
            for ref_index, ref in enumerate(_refs_for_quote_capture(candidate, source), start=1):
                if len(facts) >= max_quote_shots_per_run:
                    continue
                quote = (ref.quote or "").strip()
                if not quote:
                    continue
                fact_id = f"call_{call_index:04d}_candidate_{candidate_index:02d}_ref_{ref_index:02d}"
                facts.append(
                    {
                        "id": fact_id,
                        "candidate_name": candidate.name,
                        "candidate_type": candidate.competitor_type,
                        "url": source.url,
                        "title": source.title,
                        "evidence_quote": quote,
                    }
                )
                lookup[(run_key, fact_id)] = (call_index, candidate_index, ref_index)

    paths: dict[int, dict[tuple[int, int], list[str]]] = defaultdict(dict)
    for run_key, group in grouped.items():
        facts = group["facts"]
        if not facts:
            continue
        run_dir = group["run_dir"]
        extraction = {
            "query": "候选竞品引用定位",
            "summary": "",
            "facts": facts,
            "entities": [],
            "evidence": [],
            "chunk_count": 0,
            "errors": [],
        }
        try:
            quote_map = capture_quote_shots_for_extraction(extraction, run_dir, min_score=0.72)
            apply_quote_shots_to_extraction(extraction, quote_map)
        except Exception as exc:
            extraction["errors"].append({"stage": "quote_shots", "error": str(exc)})
        _write_candidate_quote_artifacts(run_dir, extraction)
        for fact in extraction.get("facts") or []:
            shot_path = fact.get("quote_shot_path")
            call_ref = lookup.get((run_key, str(fact.get("id") or "")))
            if not shot_path or not call_ref:
                continue
            call_index, candidate_index, ref_index = call_ref
            paths[call_index][(candidate_index, ref_index)] = [
                _relative_data_file(run_dir / str(shot_path), data_dir)
            ]
    return {index: dict(items) for index, items in paths.items()}


def _quote_run_dir(source: RawSource, task_output_dir: Path, task_id: str) -> Path:
    run_dir = source.metadata.get("run_dir")
    data_dir = task_output_dir.parent
    if run_dir:
        path = Path(str(run_dir))
        if not path.is_absolute():
            path = data_dir / path
    else:
        source_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.id).strip("._") or "source"
        path = task_output_dir / task_id / "quote_shots" / source_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relative_data_file(path: Path, data_dir: Path) -> str:
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_candidate_quote_artifacts(run_dir: Path, extraction: dict[str, Any]) -> None:
    extraction_dir = run_dir / "extractions"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    (extraction_dir / "candidate_quote_refs.json").write_text(
        json.dumps(_jsonable_local(extraction), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    lines = ["# 候选引用截图", ""]
    facts = extraction.get("facts") or []
    if not facts:
        lines.append("未产生可定位的候选引用。")
    for index, fact in enumerate(facts, start=1):
        lines.extend(
            [
                f"## {index}. {fact.get('candidate_name') or '未命名候选'}",
                f"- 来源页面：{fact.get('title') or '未命名页面'}",
                f"- 页面地址：{fact.get('url') or '未提供'}",
                f"- 引用原文：{fact.get('evidence_quote') or '未提供'}",
            ]
        )
        if fact.get("quote_shot_path"):
            lines.append(f"- 引用截图：{fact.get('quote_shot_path')}")
        if fact.get("quote_shot_error"):
            lines.append(f"- 截图状态：{fact.get('quote_shot_error')}")
        lines.append("")
    if extraction.get("errors"):
        lines.extend(["## 处理提示", ""])
        for item in extraction.get("errors") or []:
            lines.append(f"- {item.get('error') or item}")
    (extraction_dir / "candidate_quote_refs.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _chunk_output_payload(output: LLMCandidateChunkOutput, *, include_quote_shots: bool) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "name": candidate.name,
                "competitor_type": candidate.competitor_type,
                "description": candidate.description,
                "selection_reason": candidate.selection_reason,
                "evidence_source_urls": candidate.evidence_source_urls,
                "source_refs": [
                    _llm_source_ref_payload(ref, include_quote_shots=include_quote_shots)
                    for ref in candidate.source_refs
                ],
                "confidence": candidate.confidence,
                "is_supplemented": candidate.is_supplemented,
            }
            for candidate in output.candidates
        ],
        "generation_notes": output.generation_notes,
    }


def _llm_source_ref_payload(ref: LLMSourceRef, *, include_quote_shots: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_id": ref.source_id,
        "chunk_id": ref.chunk_id,
        "url": ref.url,
        "title": ref.title,
        "quote": ref.quote,
    }
    if include_quote_shots:
        paths = _dedupe(getattr(ref, "_resolved_evidence_screenshot_paths", []))
        payload["evidence_screenshot_path"] = paths[0] if paths else None
        payload["chunk_shot_paths"] = []
    return payload


def _save_chunk_llm_call(
    task_output_dir: Path | None,
    task_id: str | None,
    node: str,
    prompt_template: str,
    source: RawSource,
    call_index: int,
    messages: list[tuple[str, str]],
    *,
    output: Any | None = None,
    postprocessed_output: Any | None = None,
    error: Any | None = None,
    started_at: float | None = None,
) -> None:
    if task_output_dir is None or not task_id:
        return
    source_result_id = str(source.metadata.get("source_result_id") or source.metadata.get("chunk_id") or source.id)
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_result_id).strip("._") or "source"
    path = task_output_dir / task_id / "llm_calls" / node / prompt_template / f"{safe_source}.json"
    system = next((content for role, content in messages if role == "system"), "")
    user = "\n\n".join(content for role, content in messages if role == "user")
    call_payload = {
        "call_index": call_index,
        "chunk_id": source.metadata.get("chunk_id"),
        "source_id": source.id,
        "url": source.url,
        "title": source.title,
        "webpage_content": source.content,
        "user": user,
        "output": output,
        "postprocessed_output": postprocessed_output,
        "error": error,
        "duration_ms": int((time.perf_counter() - started_at) * 1000) if started_at else None,
        "created_at": time.time(),
    }
    with _chunk_log_locks[str(path)]:
        payload: dict[str, Any]
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        payload.setdefault("task_id", task_id)
        payload.setdefault("node", node)
        payload.setdefault("prompt_template", prompt_template)
        payload.setdefault("system", system)
        payload.setdefault(
            "source",
            {
                "source_result_id": source_result_id,
                "url": source.url,
                "title": source.title,
            },
        )
        calls = [item for item in payload.get("calls", []) if int(item.get("call_index") or -1) != call_index]
        calls.append(call_payload)
        payload["calls"] = sorted(calls, key=lambda item: int(item.get("call_index") or 0))
        path.parent.mkdir(parents=True, exist_ok=True)
        jsonable_payload = _jsonable_local(payload)
        path.write_text(json.dumps(jsonable_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        path.with_suffix(".md").write_text(_format_chunk_llm_markdown(jsonable_payload), encoding="utf-8")


def _format_chunk_llm_markdown(payload: dict[str, Any]) -> str:
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    lines = [
        "# 搜索阶段大模型输出",
        "",
        f"- 页面标题：{source.get('title') or '未命名页面'}",
        f"- 页面地址：{source.get('url') or '未提供'}",
        "",
    ]
    calls = payload.get("calls") if isinstance(payload.get("calls"), list) else []
    for call in calls:
        if not isinstance(call, dict):
            continue
        lines.extend([f"## 第 {call.get('call_index') or '?'} 次网页片段分析", ""])
        if call.get("chunk_id"):
            lines.append(f"- 片段编号：{call.get('chunk_id')}")
        if call.get("duration_ms") is not None:
            lines.append(f"- 耗时：{call.get('duration_ms')} 毫秒")
        if call.get("error"):
            lines.extend(["", "### 处理结果", f"调用失败：{_plain_text(call.get('error'), 600)}", ""])
            continue
        content = _plain_text(call.get("webpage_content"), 2400)
        if content:
            lines.extend(["", "### 网页正文", content, ""])
        output = call.get("postprocessed_output") or call.get("output") or {}
        lines.extend(["### 识别结果", ""])
        candidates = output.get("candidates") if isinstance(output, dict) else []
        if not candidates:
            lines.append("未从该网页片段识别出明确竞品。")
        for index, candidate in enumerate(candidates or [], start=1):
            if not isinstance(candidate, dict):
                continue
            lines.append(f"{index}. {candidate.get('name') or '未命名候选'}")
            if candidate.get("competitor_type"):
                lines.append(f"   - 分类：{_competitor_type_label(str(candidate.get('competitor_type')))}")
            if candidate.get("description"):
                lines.append(f"   - 简介：{_plain_text(candidate.get('description'), 420)}")
            if candidate.get("selection_reason"):
                lines.append(f"   - 理由：{_plain_text(candidate.get('selection_reason'), 420)}")
            refs = candidate.get("source_refs") if isinstance(candidate.get("source_refs"), list) else []
            for ref in refs[:3]:
                if not isinstance(ref, dict):
                    continue
                if ref.get("quote"):
                    lines.append(f"   - 引用：{_plain_text(ref.get('quote'), 360)}")
                shots = ref.get("chunk_shot_paths") if isinstance(ref.get("chunk_shot_paths"), list) else []
                if shots:
                    lines.append(f"   - 引用截图：{', '.join(str(item) for item in shots)}")
            lines.append("")
        notes = output.get("generation_notes") if isinstance(output, dict) else []
        if notes:
            lines.extend(["### 备注", ""])
            lines.extend(f"- {_plain_text(item, 260)}" for item in notes)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _plain_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) <= limit else f"{text[:limit]}..."


def _competitor_type_label(value: str) -> str:
    return {
        "head_direct": "头部直接竞品",
        "challenger": "追赶型竞品",
        "indirect_cross": "间接或跨界竞品",
        "potential_substitute": "潜在替代品",
    }.get(value, value)


def _jsonable_local(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable_local(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_local(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_local(item) for item in value]
    return value


def _selected_set_from_llm_output(
    benchmark: AnalysisBenchmark,
    candidates: list[CandidateCompetitor],
    llm_output: LLMCandidateSelectionOutput,
    budget: int,
) -> SelectedCompetitorSet:
    scoring_profile = scoring_profile_for(benchmark.analysis_purpose)
    rule_ranked = score_and_select_candidates(benchmark, candidates, budget=max(budget, len(candidates))).candidates
    existing_by_key = {_name_key(item.name): item for item in rule_ranked}
    selected: list[CandidateCompetitor] = []
    used: set[str] = set()
    notes = list(llm_output.fallback_notes)

    for item in llm_output.selected:
        key = _name_key(item.name)
        if not key or key in used or is_invalid_candidate_name(item.name, benchmark):
            continue
        item_refs = _source_refs_from_llm(item.source_refs)
        base = existing_by_key.get(key)
        if base:
            candidate = base.model_copy(deep=True)
            candidate.competitor_type = item.competitor_type or candidate.competitor_type
            candidate.selection_reason = item.selection_reason
            confidence = item.confidence if (item_refs or candidate.source_refs) else min(item.confidence, 0.35)
            candidate.score = max(candidate.score, round(confidence, 4))
            candidate.tags = _dedupe([*candidate.tags, "llm_selected"] + ([] if (item_refs or candidate.source_refs) else ["missing_source_refs"]))
            if item.description:
                candidate.description = item.description
            candidate.source_refs = _merge_source_refs(candidate.source_refs, item_refs)
            for url in _dedupe([*item.evidence_source_urls, *[ref.url for ref in item_refs]]):
                if url and url not in candidate.source_urls:
                    candidate.source_urls.append(url)
            for title in [ref.title for ref in item_refs]:
                if title and title not in candidate.source_titles:
                    candidate.source_titles.append(title)
        else:
            confidence = item.confidence if item_refs else min(item.confidence, 0.35)
            candidate = CandidateCompetitor(
                name=item.name,
                description=item.description or "LLM-supplemented real competitor that needs follow-up evidence collection.",
                competitor_type=item.competitor_type,
                source_urls=_dedupe([*item.evidence_source_urls, *[ref.url for ref in item_refs]]),
                source_titles=_dedupe([ref.title for ref in item_refs]),
                source_refs=item_refs,
                tags=["llm_supplement"] + ([] if item_refs else ["missing_source_refs"]),
                score=round(confidence, 4),
                selection_reason=item.selection_reason,
            )
            notes.append(f"Candidate supplemented by LLM because the pool was incomplete: {item.name}.")
        selected.append(candidate)
        used.add(key)
        if len(selected) >= budget:
            break

    if len(selected) < budget:
        for candidate in rule_ranked:
            key = _name_key(candidate.name)
            if key not in used and not is_invalid_candidate_name(candidate.name, benchmark):
                fallback_candidate = candidate.model_copy(deep=True)
                fallback_candidate.tags = _dedupe([*fallback_candidate.tags, "rule_backfill"])
                selected.append(fallback_candidate)
                used.add(key)
            if len(selected) >= budget:
                break

    if len(selected) < budget:
        notes.append(f"Only {len(selected)} competitors were selected for a budget of {budget}.")

    gap_request = llm_output.gap_request or _build_gap_request(benchmark, selected, candidates, budget)
    all_candidates = _dedupe_candidates([*rule_ranked, *selected])
    return SelectedCompetitorSet(
        total_budget=budget,
        scoring_profile=scoring_profile,
        candidates=all_candidates,
        selected=selected[:budget],
        allocation={
            item.competitor_type: sum(1 for selected_item in selected if selected_item.competitor_type == item.competitor_type)
            for item in selected
        },
        fallback_notes=_dedupe(notes),
        gap_request=gap_request if len(selected) < budget else llm_output.gap_request,
    )


def _source_refs_from_llm(items: list[LLMSourceRef]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    seen: set[str] = set()
    for item in items:
        url = item.url.strip()
        if not url:
            continue
        key = f"{item.source_id or ''}|{item.chunk_id or ''}|{url}|{item.quote[:80]}"
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            SourceRef(
                source_id=item.source_id,
                chunk_id=item.chunk_id,
                url=url,
                title=item.title or "Untitled source",
                quote=item.quote,
                screenshot_path=None,
                evidence_screenshot_path=(_dedupe(getattr(item, "_resolved_evidence_screenshot_paths", [])) or [None])[0],
                chunk_shot_paths=[],
            )
        )
    return refs


def _merge_source_refs(existing: list[SourceRef], additions: list[SourceRef]) -> list[SourceRef]:
    merged = list(existing)
    seen = {f"{item.source_id or ''}|{item.chunk_id or ''}|{item.url}|{item.quote[:80]}" for item in merged}
    for item in additions:
        key = f"{item.source_id or ''}|{item.chunk_id or ''}|{item.url}|{item.quote[:80]}"
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged[:12]


def _build_gap_request(
    benchmark: AnalysisBenchmark,
    selected: list[CandidateCompetitor],
    candidates: list[CandidateCompetitor],
    budget: int,
) -> CandidateGapRequest | None:
    if len(selected) >= budget:
        return None
    wanted_types: list[CompetitorType] = ["head_direct", "challenger", "indirect_cross", "potential_substitute"]
    present = {item.competitor_type for item in selected}
    missing_types = [item for item in wanted_types if item not in present]
    weak_source_count = sum(1 for item in candidates if not item.source_refs)
    missing_factors = ["competitor_type_coverage", "heat", "growth", "similarity", "risk"]
    if weak_source_count:
        missing_factors.append("strict_source_refs")
    focus = [
        f"{benchmark.target_product} alternatives competitors",
        f"{benchmark.target_product} market map emerging products",
        f"{benchmark.target_product} funding growth downloads reviews competitors",
    ]
    return CandidateGapRequest(
        missing_competitor_types=missing_types,
        missing_factors=missing_factors,
        suggested_query_focus=focus,
        rationale=f"Only {len(selected)} reliable competitors were selected for a budget of {budget}. More sourced candidates are needed.",
    )


def is_invalid_candidate_name(name: str, benchmark: AnalysisBenchmark | str) -> bool:
    cleaned = name.strip()
    if not cleaned or len(cleaned) < 2:
        return True
    compact = re.sub(r"\s+", "", cleaned)
    key = _name_key(cleaned)
    target_product = benchmark.target_product if isinstance(benchmark, AnalysisBenchmark) else str(benchmark or "")
    target_key = _name_key(target_product)
    if target_key and (key == target_key or target_key in key):
        return True
    if len(cleaned) > 60 or (re.search(r"[\u4e00-\u9fff]", cleaned) and len(compact) > 24):
        return True
    if re.search(r"[。！？!?；;，,：:]", cleaned):
        return True
    if re.search(r"\s+(or|and|with|for|to|from|of|in)\s+", cleaned, flags=re.IGNORECASE):
        return True
    phrase_markers = {
        "骑自行车",
        "骑乘",
        "上班",
        "选择",
        "购买",
        "使用",
        "回收",
        "出行",
        "步行",
        "种树",
        "获得积分",
        "绿色行为",
        "绿色产品",
        "更环保",
        "无纸化",
        "消费者行为",
        "采购偏好",
    }
    if any(marker in compact for marker in phrase_markers):
        return True
    generic_suffixes = (
        "产品",
        "服务",
        "平台",
        "行为",
        "方式",
        "方案",
        "内容",
        "用户",
        "客户",
        "功能",
        "应用程式",
        "应用程序",
    )
    known_short_product_suffixes = ("森林", "支付", "书", "钉钉", "飞书")
    if len(compact) > 8 and compact.endswith(generic_suffixes) and not compact.endswith(known_short_product_suffixes):
        return True
    invalid_exact = {
        "duckduckgo",
        "google",
        "baidu",
        "search",
        "map",
        "competitors",
        "alternatives",
        "market",
        "competitor",
        "product",
        "users",
        "reviews",
        "analysis",
        "report",
        "竞品",
        "替代品",
        "头部产品",
        "类似产品",
        "用户评价",
        "功能体验",
    }
    if key in invalid_exact or cleaned.lower() in invalid_exact:
        return True
    return not bool(re.search(r"[\u4e00-\u9fffA-Za-z]", cleaned))


_is_invalid_llm_candidate_name = is_invalid_candidate_name


def _name_key(value: str) -> str:
    return re.sub(r"[\W_]+", "", value or "").lower()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = (item or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _dedupe_candidates(candidates: list[CandidateCompetitor]) -> list[CandidateCompetitor]:
    seen: set[str] = set()
    result: list[CandidateCompetitor] = []
    for candidate in candidates:
        key = _name_key(candidate.name)
        if key and key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


async def generate_candidate_pool_with_llm_or_rules(
    model,
    benchmark: AnalysisBenchmark,
    request: CreateTaskRequest,
    sources: list[RawSource],
    *,
    max_candidates: int = 30,
    existing_candidate_names: list[str] | None = None,
) -> tuple[list[CandidateCompetitor], list[str], str]:
    """Legacy compatibility wrapper for the retired whole-pool candidate step."""
    return await generate_candidate_pool_from_chunks_with_llm_or_rules(
        model,
        benchmark,
        request,
        sources,
        max_candidates=max_candidates,
        existing_candidate_names=existing_candidate_names,
    )


def _candidate_pool_messages(
    benchmark: AnalysisBenchmark,
    request: CreateTaskRequest,
    sources: list[RawSource],
    max_candidates: int,
    existing_candidate_names: list[str] | None = None,
) -> list[tuple[str, str]]:
    raise RuntimeError("候选池全量汇总 prompt 已下线，请使用 chunk 级候选生成流程。")
