from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .browser import Screenshotter, text_range_page_bounds
from .models import ChunkShotCapture, SearchResult, TextChunk
from .utils import clean_page_text, clean_text, dom_node_text, dom_nodes_full_text, normalize_quote_match_text


LOW_VALUE_DOM_STRONG_PATTERNS = [
    "cookie",
    "cookies",
    "隐私政策",
    "隐私权政策",
    "个人信息保护",
    "收集您的个人信息",
    "使用您的个人信息",
    "共享您的个人信息",
    "转让您的个人信息",
    "公开披露您的个人信息",
    "删除您的个人信息",
    "更正您的个人信息",
    "未成年人",
    "web beacon",
    "beacons",
    "注销申请",
    "用户协议",
    "服务条款",
    "法律声明",
    "免责声明",
    "版权所有",
    "copyright",
    "备案号",
    "icp",
    "公网安备",
]

LOW_VALUE_DOM_SOFT_PATTERNS = [
    "个人信息",
    "联系我们",
    "联系方式",
    "联系电话",
    "联系邮箱",
    "客服电话",
    "公司地址",
    "客服邮箱",
    "关于我们",
    "官方公众号",
    "预约演示",
    "免费试用",
    "市场合作",
]

LOW_VALUE_DOM_CONTEXT_STRONG_PATTERNS = [
    "footer",
    "site-footer",
    "page-footer",
    "global-footer",
    "copyright",
    "beian",
    "icp",
    "privacy",
    "policy",
    "terms",
    "cookie",
    "cookies",
    "legal",
    "页脚",
    "底部",
    "隐私",
    "政策",
    "版权",
    "备案",
]

LOW_VALUE_DOM_CONTEXT_CONTACT_PATTERNS = [
    "contact",
    "contact-us",
    "contactus",
    "联系我们",
    "联系方式",
]

LOW_VALUE_DOM_KEEP_MARKERS = [
    "竞品",
    "竞争对手",
    "产品功能",
    "价格",
    "客户",
    "案例",
    "市场",
    "功能",
    "解决方案",
    "产品优势",
]

LOW_VALUE_DOM_SECTION_START_PATTERNS = [
    "本政策将帮助您了解",
    "在使用云听cem的产品或服务前",
    "请您务必仔细阅读并透彻理解本政策",
    "欢迎使用云听cem官方网站",
    "我们非常重视对用户",
    "个人信息保护政策",
]

LOW_VALUE_DOM_SECTION_END_PATTERNS = [
    "更新日期",
    "最近更新",
    "生效日期",
    "云听cem官网浏览体验反馈问卷",
    "提交获取资料",
    "版权所有",
    "copyright",
]

LOW_VALUE_DOM_FORM_START_PATTERNS = [
    "云听cem官网浏览体验反馈问卷",
    "您对于云听cem官网是否满意",
    "您的信息仅用来在第一时间与您联系进行系统演示",
    "提交获取资料",
    "为了更好地回应您的建议",
]

LOW_VALUE_DOM_FOOTER_START_PATTERNS = [
    "国内头部客户体验管理saas平台",
    "预约演示市场合作",
    "市场合作：",
    "市场合作:",
    "资源与支持成功案例干货文章视频/直播白皮书/报告下载联系我们",
]

def normalized_text_index(value: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    original_indexes: list[int] = []
    for index, char in enumerate(value or ""):
        if char.isspace():
            continue
        normalized_chars.append(char.lower())
        original_indexes.append(index)
    return "".join(normalized_chars), original_indexes

def normalized_pattern(value: str) -> str:
    return normalize_quote_match_text(value)

def find_normalized_pattern_start(normalized_text: str, pattern: str, *, start_at: int = 0) -> int:
    wanted = normalized_pattern(pattern)
    if not wanted:
        return -1
    return normalized_text.find(wanted, max(0, start_at))

def normalized_range_to_original(
    original_indexes: list[int],
    normalized_start: int,
    normalized_end_exclusive: int,
    fallback_end: int,
) -> tuple[int, int] | None:
    if normalized_start < 0 or normalized_start >= len(original_indexes):
        return None
    if normalized_end_exclusive <= normalized_start:
        return None
    last_index = min(normalized_end_exclusive - 1, len(original_indexes) - 1)
    return original_indexes[normalized_start], min(fallback_end, original_indexes[last_index] + 1)

def low_value_dom_section_ranges(full_text: str) -> list[tuple[int, int, str]]:
    normalized_text, original_indexes = normalized_text_index(full_text)
    if not normalized_text or not original_indexes:
        return []

    ranges: list[tuple[int, int, str]] = []

    def add_range_from_patterns(
        start_patterns: list[str],
        end_patterns: list[str],
        label: str,
        *,
        max_chars_if_no_end: int,
        start_at_ratio: float = 0.0,
    ) -> None:
        min_start = int(len(normalized_text) * max(0.0, min(1.0, start_at_ratio)))
        for start_pattern in start_patterns:
            start_norm = find_normalized_pattern_start(normalized_text, start_pattern, start_at=min_start)
            if start_norm < 0:
                continue
            end_norm = -1
            for end_pattern in end_patterns:
                candidate = find_normalized_pattern_start(normalized_text, end_pattern, start_at=start_norm + 1)
                if candidate >= 0 and (end_norm < 0 or candidate < end_norm):
                    end_norm = candidate + len(normalized_pattern(end_pattern))
            if end_norm < 0:
                end_norm = min(len(normalized_text), start_norm + max_chars_if_no_end)
            original_range = normalized_range_to_original(original_indexes, start_norm, end_norm, len(full_text))
            if original_range:
                ranges.append((original_range[0], original_range[1], label))

    add_range_from_patterns(
        LOW_VALUE_DOM_SECTION_START_PATTERNS,
        LOW_VALUE_DOM_SECTION_END_PATTERNS,
        "policy_section",
        max_chars_if_no_end=30_000,
    )
    add_range_from_patterns(
        LOW_VALUE_DOM_FORM_START_PATTERNS,
        ["提交获取资料", "技术支持", "点我阅读"],
        "contact_feedback_form",
        max_chars_if_no_end=8_000,
    )
    add_range_from_patterns(
        LOW_VALUE_DOM_FOOTER_START_PATTERNS,
        ["提交获取资料", "版权所有", "copyright", "备案号"],
        "footer_contact_nav",
        max_chars_if_no_end=12_000,
        start_at_ratio=0.45,
    )

    merged: list[tuple[int, int, str]] = []
    for start, end, label in sorted(ranges, key=lambda item: (item[0], item[1])):
        if not merged or start > merged[-1][1]:
            merged.append((start, end, label))
            continue
        previous_start, previous_end, previous_label = merged[-1]
        labels: list[str] = []
        for item in f"{previous_label}+{label}".split("+"):
            if item and item not in labels:
                labels.append(item)
        merged[-1] = (previous_start, max(previous_end, end), "+".join(labels))
    return merged

def text_range_overlaps_any(start: int, end: int, ranges: list[tuple[int, int, str]]) -> tuple[bool, str | None]:
    for range_start, range_end, label in ranges:
        if end > range_start and start < range_end:
            return True, label
    return False, None

def dom_node_context_text(node: dict[str, Any]) -> str:
    return clean_text(
        " ".join(
            str(node.get(key) or "")
            for key in ["tag", "id", "className", "role", "ariaLabel", "context"]
        ),
        max_chars=3000,
    ).lower()

def is_low_value_dom_node(node: dict[str, Any]) -> bool:
    text = dom_node_text(node)
    cleaned = clean_text(text, max_chars=4000).lower()
    if not cleaned:
        return True

    context = dom_node_context_text(node)
    has_keep_marker = any(marker.lower() in cleaned for marker in LOW_VALUE_DOM_KEEP_MARKERS)
    if has_keep_marker and not any(pattern.lower() in cleaned for pattern in LOW_VALUE_DOM_STRONG_PATTERNS):
        return False

    if any(pattern.lower() in cleaned for pattern in LOW_VALUE_DOM_STRONG_PATTERNS):
        return True

    soft_hits = sum(1 for pattern in LOW_VALUE_DOM_SOFT_PATTERNS if pattern.lower() in cleaned)
    strong_context = any(pattern.lower() in context for pattern in LOW_VALUE_DOM_CONTEXT_STRONG_PATTERNS)
    contact_context = any(pattern.lower() in context for pattern in LOW_VALUE_DOM_CONTEXT_CONTACT_PATTERNS)
    contact_text = any(
        marker in cleaned
        for marker in ["电话", "邮箱", "地址", "客服", "微信", "qq", "email", "e-mail", "tel:"]
    )

    if strong_context:
        return True
    if "@" in cleaned and len(cleaned) <= 200:
        return True
    if contact_context and (soft_hits > 0 or contact_text or len(cleaned) <= 500):
        return True
    if soft_hits >= 2:
        return True
    if soft_hits >= 1 and contact_text and len(cleaned) <= 600:
        return True
    return False

def node_has_layout_coordinates(node: dict[str, Any]) -> bool:
    top = node.get("top")
    bottom = node.get("bottom")
    return isinstance(top, (int, float)) and isinstance(bottom, (int, float)) and float(bottom) >= float(top)

def filter_low_value_dom_nodes(dom_nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    removed_nodes: list[dict[str, Any]] = []
    layout_nodes: list[dict[str, Any]] = []
    for node in dom_nodes:
        text = dom_node_text(node)
        if not text:
            continue
        if not node_has_layout_coordinates(node):
            removed_nodes.append(
                {
                    "order": node.get("order"),
                    "start": node.get("start"),
                    "end": node.get("end"),
                    "top": node.get("top"),
                    "bottom": node.get("bottom"),
                    "reason": "no_layout_coordinates",
                    "text_preview": clean_text(text, max_chars=180),
                    "context": clean_text(dom_node_context_text(node), max_chars=240),
                }
            )
            continue
        layout_nodes.append(node)

    full_text = dom_nodes_full_text(layout_nodes)
    section_ranges = low_value_dom_section_ranges(full_text)
    kept_nodes: list[dict[str, Any]] = []
    cursor = 0
    source_cursor = 0
    for node in layout_nodes:
        text = dom_node_text(node)
        if not text:
            continue
        layout_start = source_cursor
        layout_end = layout_start + len(text)
        source_cursor = layout_end + 1
        in_section, section_label = text_range_overlaps_any(layout_start, layout_end, section_ranges)
        remove_reason = section_label if in_section else None
        if remove_reason is None and is_low_value_dom_node(node):
            remove_reason = "low_value_node"
        if remove_reason:
            removed_nodes.append(
                {
                    "order": node.get("order"),
                    "start": node.get("start"),
                    "end": node.get("end"),
                    "layout_start": layout_start,
                    "layout_end": layout_end,
                    "top": node.get("top"),
                    "bottom": node.get("bottom"),
                    "reason": remove_reason,
                    "text_preview": clean_text(text, max_chars=180),
                    "context": clean_text(dom_node_context_text(node), max_chars=240),
                }
            )
            continue
        item = dict(node)
        item["original_start"] = node.get("start")
        item["original_end"] = node.get("end")
        item["start"] = cursor
        item["end"] = cursor + len(text)
        item["length"] = len(text)
        item["text"] = text
        kept_nodes.append(item)
        cursor = item["end"] + 1
    return kept_nodes, removed_nodes

def source_result_id(result: SearchResult) -> str:
    rank = result.rank if result.rank is not None else "na"
    digest = hashlib.sha1(f"{result.method}|{rank}|{result.url}".encode("utf-8")).hexdigest()[:8]
    return f"{result.method}_{rank}_{digest}"

def chunk_dom_text(
    results: list[SearchResult],
    run_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    viewport_height: int,
    capture_screenshots: bool,
    start_sequence: int = 1,
) -> tuple[list[TextChunk], list[dict[str, Any]]]:
    chunks: list[TextChunk] = []
    errors: list[dict[str, Any]] = []
    sequence = start_sequence
    seen_urls: set[str] = set()
    with Screenshotter(run_dir, full_page=False, viewport_height=viewport_height) as screenshotter:
        for result in results:
            if result.rank is None or not result.url or result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            source_id = source_result_id(result)
            page, dom_nodes, error = screenshotter.load_quote_dom_page(result.url)
            if error or page is None:
                errors.append(
                    {
                        "stage": "dom_text_chunking",
                        "source_result_id": source_id,
                        "url": result.url,
                        "rank": result.rank,
                        "method": result.method,
                        "error": error or "Page could not be loaded.",
                    }
                )
                continue
            try:
                original_dom_node_count = len(dom_nodes)
                original_dom_text_length = len(dom_nodes_full_text(dom_nodes))
                cleaned_dom_nodes, removed_dom_nodes = filter_low_value_dom_nodes(dom_nodes)
                dom_nodes = cleaned_dom_nodes
                if removed_dom_nodes:
                    errors.append(
                        {
                            "stage": "dom_text_cleanup",
                            "source_result_id": source_id,
                            "url": result.url,
                            "rank": result.rank,
                            "method": result.method,
                            "removed_nodes": len(removed_dom_nodes),
                            "removed_chars": sum(
                                max(0, int(item.get("end") or 0) - int(item.get("start") or 0))
                                for item in removed_dom_nodes
                            ),
                            "original_nodes": original_dom_node_count,
                            "original_chars": original_dom_text_length,
                            "removed_previews": removed_dom_nodes[:12],
                            "message": "Removed low-value cookie/privacy/footer/contact DOM text before chunking.",
                        }
                    )
                full_text = clean_page_text(dom_nodes_full_text(dom_nodes), max_chars=1_000_000)
                if not full_text:
                    errors.append(
                        {
                            "stage": "dom_text_chunking",
                            "source_result_id": source_id,
                            "url": result.url,
                            "rank": result.rank,
                            "method": result.method,
                            "error": "Visible DOM text is empty.",
                        }
                    )
                    continue
                page_height = int(
                    page.evaluate(
                        """
                        () => Math.max(
                            document.body ? document.body.scrollHeight : 0,
                            document.documentElement ? document.documentElement.scrollHeight : 0,
                            window.innerHeight
                        )
                        """
                    )
                    or 0
                )
                raw_viewport = page.viewport_size
                viewport = raw_viewport if isinstance(raw_viewport, dict) else {}
                viewport_width = int(viewport.get("width") or screenshotter.viewport_width)
                actual_viewport_height = int(viewport.get("height") or screenshotter.viewport_height)
                start = 0
                while start < len(full_text):
                    end = min(start + chunk_size, len(full_text))
                    chunk_body = full_text[start:end]
                    if not chunk_body.strip():
                        break
                    chunk_id = f"chunk_{sequence:04d}"
                    chunk_text_path = screenshotter._write_chunk_text(
                        result.url,
                        f"{chunk_id}_{result.method}_{result.rank}",
                        chunk_body,
                    )
                    page_top, page_bottom = text_range_page_bounds(dom_nodes, start, end)
                    shot_capture = ChunkShotCapture(paths=[])
                    if capture_screenshots:
                        shot_capture = screenshotter.capture_chunk_range_on_page(
                            page,
                            dom_nodes,
                            result.url,
                            chunk_body,
                            f"chunk_{sequence:04d}_{result.method}_{result.rank}",
                            range_start=start,
                            range_end=end,
                        )
                    chunk = TextChunk(
                        chunk_id=chunk_id,
                        source_result_id=source_id,
                        method=result.method,
                        rank=result.rank,
                        title=result.title,
                        url=result.url,
                        content_path=chunk_text_path,
                        start_char=start,
                        end_char=end,
                        text=chunk_body,
                        chunk_mode="text",
                        scroll_y=shot_capture.scroll_y if shot_capture.scroll_y is not None else page_top,
                        viewport_height=actual_viewport_height,
                        viewport_width=viewport_width,
                        page_height=page_height,
                        dom_text_length=len(full_text),
                        dom_range_start=start,
                        dom_range_end=end,
                        text_sources=["dom"],
                        full_dom_text_path=None,
                        screenshot_mismatch_warning=None,
                        chunk_shot_path=shot_capture.paths[0] if shot_capture.paths else None,
                        chunk_shot_error=shot_capture.error,
                        chunk_shot_paths=shot_capture.paths or None,
                    )
                    chunks.append(chunk)
                    if shot_capture.error:
                        errors.append(
                            {
                                "stage": "dom_text_chunk_shot",
                                "chunk_id": chunk.chunk_id,
                                "url": result.url,
                                "rank": result.rank,
                                "method": result.method,
                                "error": shot_capture.error,
                            }
                        )
                    sequence += 1
                    if end >= len(full_text):
                        break
                    start = max(end - chunk_overlap, start + 1)
            finally:
                try:
                    page.close()
                except Exception:
                    pass
    return chunks, errors

def chunk_text(
    results: list[SearchResult],
    run_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    start_sequence: int = 1,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    sequence = start_sequence
    for result in results:
        if result.rank is None or not result.url or not result.content_path:
            continue
        content_file = run_dir / result.content_path
        try:
            text = content_file.read_text(encoding="utf-8")
        except OSError:
            continue
        text = text.strip()
        if not text:
            continue

        start = 0
        source_id = source_result_id(result)
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_body = text[start:end]
            chunk_id = f"chunk_{sequence:04d}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    source_result_id=source_id,
                    method=result.method,
                    rank=result.rank,
                    title=result.title,
                    url=result.url,
                    content_path=result.content_path,
                    start_char=start,
                    end_char=end,
                    text=chunk_body,
                    text_sources=[result.content_source or "file"],
                )
            )
            sequence += 1
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks

def chunk_viewports(
    results: list[SearchResult],
    run_dir: Path,
    *,
    max_scrolls: int,
    overlap_ratio: float,
    text_max_chars: int,
    text_overlap_chars: int,
    viewport_height: int,
    capture_screenshots: bool,
    max_screenshots: int,
) -> tuple[list[TextChunk], list[dict[str, Any]]]:
    chunks: list[TextChunk] = []
    errors: list[dict[str, Any]] = []
    sequence = 1
    remaining_shots = max_screenshots if capture_screenshots else 0

    with Screenshotter(run_dir, full_page=False, viewport_height=viewport_height) as screenshotter:
        for result in results:
            if result.rank is None or not result.url:
                continue
            source_id = source_result_id(result)
            viewport_chunks, error = screenshotter.capture_viewport_chunks(
                result.url,
                f"chunk_{result.method}_{result.rank}",
                max_scrolls=max_scrolls,
                overlap_ratio=overlap_ratio,
                text_max_chars=text_max_chars,
                text_overlap_chars=text_overlap_chars,
                capture_screenshots=capture_screenshots,
                max_screenshots=remaining_shots,
            )
            if capture_screenshots:
                captured_count = len({item.chunk_shot_path for item in viewport_chunks if item.chunk_shot_path})
                remaining_shots = max(0, remaining_shots - captured_count)
            if error:
                errors.append(
                    {
                        "stage": "viewport_chunking",
                        "source_result_id": source_id,
                        "url": result.url,
                        "rank": result.rank,
                        "method": result.method,
                        "error": error,
                    }
                )
                continue
            if not viewport_chunks:
                errors.append(
                    {
                        "stage": "viewport_chunking",
                        "source_result_id": source_id,
                        "url": result.url,
                        "rank": result.rank,
                        "method": result.method,
                        "error": "No viewport chunks were generated.",
                    }
                )
                continue
            for viewport_index, viewport_chunk in enumerate(viewport_chunks, start=1):
                if not viewport_chunk.text:
                    errors.append(
                        {
                            "stage": "viewport_text",
                            "source_result_id": source_id,
                            "url": result.url,
                            "rank": result.rank,
                            "method": result.method,
                            "viewport_index": viewport_index,
                            "scroll_y": viewport_chunk.scroll_y,
                            "error": "Visible DOM text is empty for this viewport chunk.",
                        }
                    )
                    continue
                chunk_id = f"chunk_{sequence:04d}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        source_result_id=source_id,
                        method=result.method,
                        rank=result.rank,
                        title=result.title,
                        url=result.url,
                        content_path=result.content_path or "",
                        start_char=viewport_chunk.start_char,
                        end_char=viewport_chunk.end_char,
                        text=viewport_chunk.text,
                        chunk_mode="viewport",
                        scroll_y=viewport_chunk.scroll_y,
                        viewport_height=viewport_chunk.viewport_height,
                        viewport_width=viewport_chunk.viewport_width,
                        page_height=viewport_chunk.page_height,
                        dom_text_length=viewport_chunk.dom_text_length,
                        dom_range_start=viewport_chunk.dom_range_start,
                        dom_range_end=viewport_chunk.dom_range_end,
                        dom_gap_filled_count=viewport_chunk.dom_gap_filled_count,
                        text_sources=viewport_chunk.text_sources,
                        full_dom_text_path=viewport_chunk.full_dom_text_path,
                        ocr_text=viewport_chunk.ocr_text,
                        ocr_error=viewport_chunk.ocr_error,
                        ocr_extra_chars=viewport_chunk.ocr_extra_chars,
                        visual_summary=viewport_chunk.visual_summary,
                        visual_error=viewport_chunk.visual_error,
                        screenshot_mismatch_warning=viewport_chunk.screenshot_mismatch_warning,
                        chunk_shot_path=viewport_chunk.chunk_shot_path,
                        chunk_shot_error=viewport_chunk.chunk_shot_error,
                    )
                )
                sequence += 1
    return chunks, errors
