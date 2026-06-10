from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

from .models import LinkCandidate, LinkDecision, TextChunk
from .reporting import write_chunks_progress, write_extraction_progress
from .utils import clean_page_text, clean_text, extract_json_object, progress, safe_error


def get_openai_client() -> tuple[Any | None, str | None]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "OPENAI_API_KEY is not set."
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        return None, f"openai package is not installed: {safe_error(exc)}"

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return OpenAI(**kwargs), None
    except Exception as exc:
        return None, f"OpenAI client init failed: {safe_error(exc)}"


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def call_vision_summary(client: Any, model: str, image_path: Path) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是网页截图视觉信息分析助手。请用中文简洁描述截图中的非纯文字信息，"
                    "重点关注图片、图表、产品界面、价格表、流程图、客户 logo 和可用于竞品分析的视觉线索。"
                    "不要编造截图里看不到的信息。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请提取这张网页 chunk 截图中的视觉信息。若没有有价值的视觉信息，请返回空字符串。",
                    },
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            },
        ],
        temperature=0.1,
    )
    return clean_page_text(response.choices[0].message.content or "", max_chars=2000)


def link_relevance_system_prompt() -> str:
    return (
        "You are a link relevance classifier for competitive-analysis web research. "
        "Return only valid JSON. Select only links that are likely to contain useful details for the user's search query. "
        "Do not select login, legal, privacy, social sharing, download, or generic navigation links."
    )


def link_relevance_user_prompt(query: str, candidates: list[LinkCandidate], selected_limit: int) -> str:
    payload = [
        {
            "id": item.id,
            "url": item.url,
            "text": item.text,
            "title": item.title,
            "context": item.context,
            "discovery_type": item.discovery_type,
        }
        for item in candidates
    ]
    return (
        "Search query:\n"
        f"{query}\n\n"
        f"Select at most {selected_limit} relevant links. Return JSON with this shape:\n"
        '{"decisions":[{"id":"link_0001","relevance":0.0,"selected":false,"reason":"short reason"}]}\n\n'
        f"Candidates:\n{payload}"
    )


def rank_link_candidates_with_llm(
    query: str,
    candidates: list[LinkCandidate],
    *,
    selected_limit: int,
) -> tuple[list[LinkDecision], list[dict[str, Any]]]:
    client, client_error = get_openai_client()
    if client_error or client is None:
        return [
            LinkDecision(
                id=item.id,
                source_url=item.source_url,
                url=item.url,
                text=item.text,
                discovery_type=item.discovery_type,
                selected=False,
                filter_method="llm",
                error=client_error or "OpenAI client unavailable.",
            )
            for item in candidates
        ], [{"stage": "link_llm_init", "error": client_error or "OpenAI client unavailable."}]

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    by_id = {item.id: item for item in candidates}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": link_relevance_system_prompt()},
                {"role": "user", "content": link_relevance_user_prompt(query, candidates, selected_limit)},
            ],
            temperature=0.0,
        )
        payload = extract_json_object(response.choices[0].message.content or "")
    except Exception as exc:
        error = safe_error(exc)
        return [
            LinkDecision(
                id=item.id,
                source_url=item.source_url,
                url=item.url,
                text=item.text,
                discovery_type=item.discovery_type,
                selected=False,
                filter_method="llm",
                error=error,
            )
            for item in candidates
        ], [{"stage": "link_llm", "error": error}]

    decisions: list[LinkDecision] = []
    selected_count = 0
    for raw in payload.get("decisions") or []:
        if not isinstance(raw, dict):
            continue
        candidate = by_id.get(str(raw.get("id") or ""))
        if not candidate:
            continue
        try:
            relevance = max(0.0, min(1.0, float(raw.get("relevance") or 0.0)))
        except (TypeError, ValueError):
            relevance = 0.0
        selected = bool(raw.get("selected")) and relevance >= 0.55 and selected_count < selected_limit
        if selected:
            selected_count += 1
        decisions.append(
            LinkDecision(
                id=candidate.id,
                source_url=candidate.source_url,
                url=candidate.url,
                text=candidate.text,
                discovery_type=candidate.discovery_type,
                selected=selected,
                relevance=round(relevance, 4),
                reason=clean_text(raw.get("reason") or "", max_chars=300),
                filter_method="llm",
            )
        )
    decided_ids = {item.id for item in decisions}
    for candidate in candidates:
        if candidate.id in decided_ids:
            continue
        decisions.append(
            LinkDecision(
                id=candidate.id,
                source_url=candidate.source_url,
                url=candidate.url,
                text=candidate.text,
                discovery_type=candidate.discovery_type,
                selected=False,
                relevance=0.0,
                reason="LLM did not return a decision for this link.",
                filter_method="llm",
            )
        )
    return decisions, []

def run_image_understanding_for_chunks(
    chunks: list[TextChunk],
    run_dir: Path,
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not enabled:
        return errors
    client, client_error = get_openai_client()
    if client_error or client is None:
        error = client_error or "OpenAI client unavailable."
        for chunk in chunks:
            chunk.visual_error = error
        return [{"stage": "image_understanding_init", "error": error}]

    model = (
        os.getenv("OPENAI_VISION_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or "gpt-4o-mini"
    )
    for chunk in chunks:
        if not chunk.chunk_shot_path:
            chunk.visual_error = "Image understanding skipped: chunk has no screenshot."
            errors.append({"stage": "image_understanding", "chunk_id": chunk.chunk_id, "error": chunk.visual_error})
            continue
        try:
            summary = call_vision_summary(client, model, run_dir / chunk.chunk_shot_path)
            chunk.visual_summary = summary or None
            if summary:
                sources = list(chunk.text_sources or ["dom"])
                if "vision" not in sources:
                    sources.append("vision")
                chunk.text_sources = sources
                chunk.text = f"{chunk.text}\n\n[视觉补充]\n{summary}".strip()
                chunk.end_char = max(chunk.end_char, chunk.start_char + len(chunk.text))
        except Exception as exc:
            chunk.visual_error = safe_error(exc)
            errors.append(
                {
                    "stage": "image_understanding",
                    "chunk_id": chunk.chunk_id,
                    "path": chunk.chunk_shot_path,
                    "error": chunk.visual_error,
                }
            )
    return errors

def extraction_system_prompt() -> str:
    return (
        "你是一个竞品分析信息提取助手。请从网页正文片段中提取结构化事实。"
        "只能返回合法 JSON，不要使用 Markdown，不要添加解释性文字。"
        "除 JSON 字段名、枚举值和 evidence_quote 外，所有内容必须使用中文填写。"
        "只能使用片段中明确出现的信息，不要推测、不要编造。"
        "每条事实都必须包含 evidence_quote，且 evidence_quote 必须从原文片段逐字复制，"
        "用于后续核对文字证据；截图会直接引用该片段对应的网页视口图或 chunk 图。"
        "事实优先基于 DOM 正文；OCR 和视觉补充只能作为辅助来源。"
    )

def extraction_user_prompt(chunk: TextChunk) -> str:
    return f"""
请从下面这个网页正文片段中提取竞品分析相关信息。

请严格返回以下 JSON 结构。字段名必须保持英文不变；枚举值也保持英文不变；字段内容用中文填写：
{{
  "summary": "用中文概括本片段中的有用信息，1-2 句话",
  "facts": [
    {{
      "field": "product_positioning|features|pricing|customers|integrations|case_studies|news|strengths|weaknesses|other",
      "subject": "中文填写事实主体，例如产品/公司/功能/价格计划名称",
      "claim": "中文填写简洁事实陈述",
      "evidence_quote": "从原文片段逐字复制的短证据，不要翻译，不要改写",
      "evidence_source": "dom|ocr|vision",
      "confidence": 0.0
    }}
  ],
  "entities": [
    {{
      "name": "中文填写实体名称；如原文只有英文专名，可保留英文专名",
      "type": "product|company|person|customer|feature|pricing_plan|integration|other",
      "description": "中文填写该实体的简短说明",
      "evidence_quote": "从原文片段逐字复制的短证据，不要翻译，不要改写",
      "evidence_source": "dom|ocr|vision"
    }}
  ]
}}

规则：
- 重点提取对竞品分析有价值的信息，例如产品定位、核心功能、价格、客户、集成、案例、新闻、优势、弱点。
- summary、subject、claim、description 必须用中文填写；如果涉及品牌名、产品名、英文专有名词，可以保留原名。
- field 和 type 只能使用上面列出的英文枚举值。
- confidence 必须是 0 到 1 之间的数字。
- evidence_source 必须填写证据来自哪一段：普通正文填 dom，OCR补充填 ocr，视觉补充填 vision。
- 如果事实主要来自 OCR 或视觉补充，请适当降低 confidence，除非证据非常明确。
- evidence_quote 必须是原文片段中的连续短文本，建议 12-160 个字符；为了定位截图，不允许翻译或改写。
- 如果片段中没有有价值的信息，请返回空的 facts 和 entities。

来源信息：
- chunk_id: {chunk.chunk_id}
- source_result_id: {chunk.source_result_id}
- title: {chunk.title}
- url: {chunk.url}
- chunk_mode: {chunk.chunk_mode}
- chunk text range: {chunk.start_char}-{chunk.end_char}
- text_sources: {", ".join(chunk.text_sources or ["dom"])}
- screenshot_warning: {chunk.screenshot_mismatch_warning or ""}

网页正文片段：
{chunk.text}
""".strip()

def call_llm_json(client: Any, model: str, chunk: TextChunk) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": extraction_system_prompt()},
            {"role": "user", "content": extraction_user_prompt(chunk)},
        ],
        temperature=0.1,
    )
    content = response.choices[0].message.content or ""
    return extract_json_object(content)

def normalize_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))

def normalize_evidence_source(value: Any, default: str = "dom") -> str:
    source = clean_text(value or default, max_chars=40).lower()
    if source in {"dom", "ocr", "vision"}:
        return source
    return default

def quote_position(chunk: TextChunk, quote: str) -> tuple[int | None, int | None]:
    quote = quote.strip()
    if not quote:
        return None, None
    local = chunk.text.find(quote)
    if local < 0:
        compact_chunk = re.sub(r"\s+", " ", chunk.text)
        compact_quote = re.sub(r"\s+", " ", quote)
        local = compact_chunk.find(compact_quote)
        if local < 0:
            return None, None
        return chunk.start_char + local, chunk.start_char + local + len(compact_quote)
    return chunk.start_char + local, chunk.start_char + local + len(quote)

def normalize_chunk_extraction(raw: dict[str, Any], chunk: TextChunk) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []

    for index, item in enumerate(raw.get("facts") or [], start=1):
        if not isinstance(item, dict):
            continue
        quote = clean_text(item.get("evidence_quote") or "", max_chars=500)
        start, end = quote_position(chunk, quote)
        evidence_source = normalize_evidence_source(item.get("evidence_source"))
        facts.append(
            {
                "id": f"{chunk.chunk_id}_fact_{index:03d}",
                "field": clean_text(item.get("field") or "other", max_chars=80),
                "subject": clean_text(item.get("subject") or "", max_chars=200),
                "claim": clean_text(item.get("claim") or "", max_chars=1000),
                "evidence_quote": quote,
                "evidence_source": evidence_source,
                "confidence": normalize_confidence(item.get("confidence")),
                "source_result_id": chunk.source_result_id,
                "source_title": chunk.title,
                "url": chunk.url,
                "rank": chunk.rank,
                "method": chunk.method,
                "chunk_id": chunk.chunk_id,
                "chunk_mode": chunk.chunk_mode,
                "content_path": chunk.content_path,
                "chunk_start_char": chunk.start_char,
                "chunk_end_char": chunk.end_char,
                "scroll_y": chunk.scroll_y,
                "viewport_height": chunk.viewport_height,
                "viewport_width": chunk.viewport_width,
                "page_height": chunk.page_height,
                "text_sources": chunk.text_sources or ["dom"],
                "dom_range_start": chunk.dom_range_start,
                "dom_range_end": chunk.dom_range_end,
                "dom_gap_filled_count": chunk.dom_gap_filled_count,
                "ocr_extra_chars": chunk.ocr_extra_chars,
                "screenshot_mismatch_warning": chunk.screenshot_mismatch_warning,
                "evidence_start_char": start,
                "evidence_end_char": end,
                "evidence_shot_path": None,
                "chunk_shot_paths": chunk.chunk_shot_paths,
                "quote_shot_path": None,
                "quote_shot_error": None,
                "screenshot_error": None,
            }
        )

    for index, item in enumerate(raw.get("entities") or [], start=1):
        if not isinstance(item, dict):
            continue
        quote = clean_text(item.get("evidence_quote") or "", max_chars=500)
        start, end = quote_position(chunk, quote)
        evidence_source = normalize_evidence_source(item.get("evidence_source"))
        entities.append(
            {
                "id": f"{chunk.chunk_id}_entity_{index:03d}",
                "name": clean_text(item.get("name") or "", max_chars=200),
                "type": clean_text(item.get("type") or "other", max_chars=80),
                "description": clean_text(item.get("description") or "", max_chars=700),
                "evidence_quote": quote,
                "evidence_source": evidence_source,
                "source_result_id": chunk.source_result_id,
                "source_title": chunk.title,
                "url": chunk.url,
                "rank": chunk.rank,
                "method": chunk.method,
                "chunk_id": chunk.chunk_id,
                "chunk_mode": chunk.chunk_mode,
                "content_path": chunk.content_path,
                "scroll_y": chunk.scroll_y,
                "viewport_height": chunk.viewport_height,
                "viewport_width": chunk.viewport_width,
                "page_height": chunk.page_height,
                "text_sources": chunk.text_sources or ["dom"],
                "dom_range_start": chunk.dom_range_start,
                "dom_range_end": chunk.dom_range_end,
                "dom_gap_filled_count": chunk.dom_gap_filled_count,
                "evidence_start_char": start,
                "evidence_end_char": end,
                "quote_shot_path": None,
                "quote_shot_error": None,
            }
        )

    return {
        "chunk_id": chunk.chunk_id,
        "source_result_id": chunk.source_result_id,
        "summary": clean_text(raw.get("summary") or "", max_chars=1000),
        "facts": facts,
        "entities": entities,
    }

def extract_chunks_with_llm(
    chunks: list[TextChunk],
    run_dir: Path,
    *,
    query: str | None = None,
    initial_errors: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    client, client_error = get_openai_client()
    if client_error or client is None:
        errors.append({"stage": "llm_init", "error": client_error or "OpenAI client unavailable."})
        return [], errors

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    extractions: list[dict[str, Any]] = []
    total = len(chunks)
    progress(f"LLM extraction start: model={model}, chunks={total}")
    for index, chunk in enumerate(chunks, start=1):
        try:
            raw = call_llm_json(client, model, chunk)
            normalized = normalize_chunk_extraction(raw, chunk)
            extractions.append(normalized)
            progress(
                f"LLM {index}/{total} {chunk.chunk_id}: ok, "
                f"facts={len(normalized.get('facts') or [])}, entities={len(normalized.get('entities') or [])}"
            )
        except Exception as exc:
            error = safe_error(exc)
            errors.append({"stage": "chunk_extract", "chunk_id": chunk.chunk_id, "url": chunk.url, "error": error})
            progress(f"LLM {index}/{total} {chunk.chunk_id}: failed - {error}")
        write_chunks_progress(run_dir, chunks, extractions)
        if query is not None:
            write_extraction_progress(
                run_dir,
                merge_extractions(query, chunks, extractions, list(initial_errors or []) + errors),
            )
    progress(f"LLM extraction done: success={len(extractions)}, errors={len(errors)}")
    return extractions, errors

def merge_extractions(
    query: str,
    chunks: list[TextChunk],
    chunk_extractions: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    fact_keys: set[tuple[str, str, str]] = set()
    entity_keys: set[tuple[str, str]] = set()
    summaries: list[str] = []

    for item in chunk_extractions:
        summary = clean_text(item.get("summary") or "", max_chars=500)
        if summary:
            summaries.append(summary)
        for fact in item.get("facts") or []:
            key = (
                str(fact.get("field") or "").lower(),
                str(fact.get("claim") or "").lower(),
                str(fact.get("evidence_quote") or "").lower(),
            )
            if key in fact_keys or not fact.get("claim"):
                continue
            fact_keys.add(key)
            facts.append(fact)
        for entity in item.get("entities") or []:
            key = (str(entity.get("name") or "").lower(), str(entity.get("type") or "").lower())
            if key in entity_keys or not entity.get("name"):
                continue
            entity_keys.add(key)
            entities.append(entity)

    evidence = []
    for fact in facts:
        evidence.append(
            {
                "fact_id": fact.get("id"),
                "url": fact.get("url"),
                "rank": fact.get("rank"),
                "method": fact.get("method"),
                "chunk_id": fact.get("chunk_id"),
                "chunk_mode": fact.get("chunk_mode"),
                "content_path": fact.get("content_path"),
                "scroll_y": fact.get("scroll_y"),
                "viewport_height": fact.get("viewport_height"),
                "viewport_width": fact.get("viewport_width"),
                "page_height": fact.get("page_height"),
                "start_char": fact.get("evidence_start_char"),
                "end_char": fact.get("evidence_end_char"),
                "evidence_quote": fact.get("evidence_quote"),
                "evidence_source": fact.get("evidence_source"),
                "confidence": fact.get("confidence"),
                "evidence_shot_path": fact.get("evidence_shot_path"),
                "chunk_shot_paths": fact.get("chunk_shot_paths"),
                "quote_shot_path": fact.get("quote_shot_path"),
                "quote_shot_error": fact.get("quote_shot_error"),
                "screenshot_error": fact.get("screenshot_error"),
                "screenshot_mismatch_warning": fact.get("screenshot_mismatch_warning"),
            }
        )

    return {
        "query": query,
        "summary": " ".join(summaries[:5]),
        "facts": facts,
        "entities": entities,
        "evidence": evidence,
        "chunk_count": len(chunks),
        "errors": errors,
    }
