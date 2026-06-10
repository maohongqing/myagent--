from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    method: str
    query: str
    rank: int | None
    title: str
    url: str
    snippet: str
    content_path: str | None = None
    content_chars: int = 0
    content_preview: str | None = None
    content_error: str | None = None
    content_source: str | None = None
    screenshot_path: str | None = None
    error: str | None = None
    screenshot_error: str | None = None
    access_status: str | None = None
    http_status: int | None = None
    final_url: str | None = None
    block_reason: str | None = None
    robots_warning: str | None = None
    access_diagnostics: dict[str, Any] | None = None
    source_url: str | None = None
    discovery_type: str | None = None
    depth: int = 0
    link_relevance: float | None = None
    link_reason: str | None = None


@dataclass
class SearchPageScreenshot:
    method: str
    url: str
    screenshot_path: str | None = None
    error: str | None = None


@dataclass
class PageData:
    content_path: str | None = None
    content_chars: int = 0
    content_preview: str | None = None
    content_error: str | None = None
    content_source: str | None = None
    screenshot_path: str | None = None
    screenshot_error: str | None = None
    access_status: str | None = None
    http_status: int | None = None
    final_url: str | None = None
    block_reason: str | None = None
    robots_warning: str | None = None
    access_diagnostics: dict[str, Any] | None = None


@dataclass
class ViewportChunkData:
    text: str
    start_char: int
    end_char: int
    scroll_y: int
    viewport_height: int
    viewport_width: int
    page_height: int
    dom_text_length: int = 0
    dom_range_start: int | None = None
    dom_range_end: int | None = None
    dom_gap_filled_count: int = 0
    text_sources: list[str] | None = None
    full_dom_text_path: str | None = None
    ocr_text: str | None = None
    ocr_error: str | None = None
    ocr_extra_chars: int = 0
    visual_summary: str | None = None
    visual_error: str | None = None
    screenshot_mismatch_warning: str | None = None
    chunk_shot_path: str | None = None
    chunk_shot_error: str | None = None


@dataclass
class TextChunk:
    chunk_id: str
    source_result_id: str
    method: str
    rank: int | None
    title: str
    url: str
    content_path: str
    start_char: int
    end_char: int
    text: str
    chunk_mode: str = "text"
    scroll_y: int | None = None
    viewport_height: int | None = None
    viewport_width: int | None = None
    page_height: int | None = None
    dom_text_length: int = 0
    dom_range_start: int | None = None
    dom_range_end: int | None = None
    dom_gap_filled_count: int = 0
    text_sources: list[str] | None = None
    full_dom_text_path: str | None = None
    ocr_text: str | None = None
    ocr_error: str | None = None
    ocr_extra_chars: int = 0
    visual_summary: str | None = None
    visual_error: str | None = None
    screenshot_mismatch_warning: str | None = None
    chunk_shot_path: str | None = None
    chunk_shot_error: str | None = None
    chunk_shot_paths: list[str] | None = None


@dataclass
class QuoteLocationItem:
    source_id: str | None
    fact_id: str | None
    url: str
    quote: str


@dataclass
class QuoteMatch:
    method: str
    score: float
    dom_start: int
    dom_end: int
    page_top: int | None = None
    page_bottom: int | None = None


@dataclass
class ChunkShotCapture:
    paths: list[str]
    error: str | None = None
    scroll_y: int | None = None
    page_top: int | None = None
    page_bottom: int | None = None


@dataclass
class LinkCandidate:
    id: str
    source_url: str
    url: str
    text: str = ""
    title: str = ""
    rel: str = ""
    context: str = ""
    discovery_type: str = "relevant_link"
    depth: int = 1


@dataclass
class LinkDecision:
    id: str
    source_url: str
    url: str
    text: str = ""
    discovery_type: str = "relevant_link"
    selected: bool = False
    relevance: float | None = None
    reason: str | None = None
    filter_method: str = "heuristic"
    error: str | None = None
