from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import ipaddress
import json
import os
import re
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .models import QuoteMatch


DEFAULT_METHODS = ["duckduckgo_text", "duckduckgo_news", "tavily"]
API_SEARCH_METHODS = [
    "brave_web",
    "searchapi_google",
    "searchapi_baidu",
    "searchapi_duckduckgo",
    "serpapi_google",
    "serpapi_baidu",
]
SUPPORTED_METHODS = set(DEFAULT_METHODS + API_SEARCH_METHODS)
SCREENSHOT_MODES = {"search_page", "result_pages", "both", "none"}
TRACKING_QUERY_PARAMS = {"fbclid", "gclid", "yclid", "mc_cid", "mc_eid"}


def parse_methods(value: str) -> list[str]:
    raw_methods = [item.strip() for item in value.split(",") if item.strip()]
    if len(raw_methods) == 1 and raw_methods[0].lower() == "all":
        return list(DEFAULT_METHODS)
    unknown = [method for method in raw_methods if method not in SUPPORTED_METHODS]
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_METHODS))
        raise argparse.ArgumentTypeError(f"Unknown method(s): {', '.join(unknown)}. Supported: {supported}")
    if not raw_methods:
        raise argparse.ArgumentTypeError("At least one search method is required.")
    return raw_methods

def clean_text(value: Any, max_chars: int = 2000) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]

def clean_page_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_chars]

def normalize_for_overlap(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()

def count_extra_text_chars(base_text: str, candidate_text: str) -> int:
    base = normalize_for_overlap(base_text)
    extra = 0
    seen: set[str] = set()
    for line in clean_page_text(candidate_text, max_chars=100_000).splitlines():
        normalized = normalize_for_overlap(line)
        if len(normalized) < 4 or normalized in seen:
            continue
        seen.add(normalized)
        if normalized not in base:
            extra += len(line.strip())
    return extra

def normalize_quote_match_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()

def quote_match_score(quote: str, candidate: str) -> float:
    wanted = normalize_quote_match_text(quote)
    current = normalize_quote_match_text(candidate)
    if not wanted or not current:
        return 0.0
    if wanted in current or current in wanted:
        shorter = min(len(wanted), len(current))
        longer = max(len(wanted), len(current))
        return shorter / max(1, longer)
    return difflib.SequenceMatcher(None, wanted, current).ratio()

def progress(message: str) -> None:
    print(message, flush=True)

def canonical_url_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if not netloc:
        return raw.lower().rstrip("/")
    path = parsed.path.rstrip("/") or "/"
    params: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lower_key = key.lower()
        if lower_key.startswith("utm_") or lower_key.startswith("spm") or lower_key in TRACKING_QUERY_PARAMS:
            continue
        params.append((key, value))
    query = urlencode(sorted(params))
    return urlunparse((scheme, netloc, path, "", query, ""))

@contextlib.contextmanager
def suppress_library_output() -> Any:
    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        yield

def dom_node_text(node: dict[str, Any]) -> str:
    return str(node.get("text") or "").strip()

def dom_nodes_full_text(dom_nodes: list[dict[str, Any]]) -> str:
    return "\n".join(dom_node_text(node) for node in dom_nodes)

def locate_text_range_in_dom_nodes(
    dom_nodes: list[dict[str, Any]],
    start: int,
    end: int,
    *,
    method: str,
    score: float,
) -> QuoteMatch | None:
    matched_nodes = [
        node
        for node in dom_nodes
        if int(node.get("end") or 0) > start and int(node.get("start") or 0) < end
    ]
    if not matched_nodes:
        return None
    page_tops = [node.get("top") for node in matched_nodes if isinstance(node.get("top"), (int, float))]
    page_bottoms = [node.get("bottom") for node in matched_nodes if isinstance(node.get("bottom"), (int, float))]
    return QuoteMatch(
        method=method,
        score=round(float(score), 4),
        dom_start=start,
        dom_end=end,
        page_top=int(min(page_tops)) if page_tops else None,
        page_bottom=int(max(page_bottoms)) if page_bottoms else None,
    )

def find_normalized_substring_range(full_text: str, quote: str) -> tuple[int, int] | None:
    wanted = normalize_quote_match_text(quote)
    if not wanted:
        return None

    normalized_chars: list[str] = []
    original_indexes: list[int] = []
    for index, char in enumerate(full_text):
        if char.isspace():
            continue
        normalized_chars.append(char.lower())
        original_indexes.append(index)
    normalized_text = "".join(normalized_chars)
    normalized_start = normalized_text.find(wanted)
    if normalized_start < 0:
        return None
    normalized_end = normalized_start + len(wanted) - 1
    return original_indexes[normalized_start], original_indexes[normalized_end] + 1

def find_quote_match(dom_nodes: list[dict[str, Any]], quote: str, *, min_score: float) -> QuoteMatch | None:
    quote = clean_page_text(quote, max_chars=1000)
    if not quote:
        return None

    full_text = dom_nodes_full_text(dom_nodes)
    exact_start = full_text.find(quote)
    if exact_start >= 0:
        return locate_text_range_in_dom_nodes(
            dom_nodes,
            exact_start,
            exact_start + len(quote),
            method="exact",
            score=1.0,
        )

    normalized_range = find_normalized_substring_range(full_text, quote)
    if normalized_range is not None:
        match = locate_text_range_in_dom_nodes(
            dom_nodes,
            normalized_range[0],
            normalized_range[1],
            method="exact",
            score=1.0,
        )
        if match:
            return match

    for candidate in evidence_quote_candidates(quote):
        normalized_range = find_normalized_substring_range(full_text, candidate)
        if normalized_range is None:
            continue
        match = locate_text_range_in_dom_nodes(
            dom_nodes,
            normalized_range[0],
            normalized_range[1],
            method="candidate",
            score=quote_match_score(quote, candidate),
        )
        if match:
            return match

    best_match: QuoteMatch | None = None
    best_score = 0.0
    quote_len = max(1, len(quote))
    for start_index, node in enumerate(dom_nodes):
        text_parts: list[str] = []
        start_char = int(node.get("start") or 0)
        end_char = start_char
        for next_node in dom_nodes[start_index : min(len(dom_nodes), start_index + 8)]:
            part = dom_node_text(next_node)
            if not part:
                continue
            text_parts.append(part)
            end_char = int(next_node.get("end") or end_char)
            candidate_text = "\n".join(text_parts)
            score = quote_match_score(quote, candidate_text)
            if score > best_score:
                match = locate_text_range_in_dom_nodes(
                    dom_nodes,
                    start_char,
                    end_char,
                    method="fuzzy",
                    score=score,
                )
                if match:
                    best_match = match
                    best_score = score
            if len(candidate_text) >= quote_len * 2 and len(candidate_text) > 500:
                break
    if best_match and best_score >= min_score:
        return best_match
    return None

def split_node_indexes_by_text_length(
    dom_nodes: list[dict[str, Any]],
    node_indexes: list[int],
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[list[int]]:
    if not node_indexes:
        return []
    max_chars = max(1, max_chars)
    overlap_chars = max(0, overlap_chars)
    groups: list[list[int]] = []
    current: list[int] = []
    current_chars = 0

    def length_for(index: int) -> int:
        return len(dom_node_text(dom_nodes[index])) + 1

    def overlap_tail(indexes: list[int]) -> list[int]:
        if overlap_chars <= 0:
            return []
        tail: list[int] = []
        total = 0
        for index in reversed(indexes):
            tail.insert(0, index)
            total += length_for(index)
            if total >= overlap_chars:
                break
        return tail

    for node_index in node_indexes:
        item_chars = length_for(node_index)
        if current and current_chars + item_chars > max_chars:
            groups.append(current)
            current = overlap_tail(current)
            current_chars = sum(length_for(index) for index in current)
        current.append(node_index)
        current_chars += item_chars
    if current:
        groups.append(current)
    return groups

def evidence_quote_candidates(quote: str) -> list[str]:
    normalized = clean_text(quote, max_chars=500)
    cleaned = re.sub(r"[.。…]{2,}|\.{3,}", " ", normalized)
    parts = [part.strip() for part in re.split(r"[。！？；;.!?\n]+", cleaned) if part.strip()]
    candidates: list[str] = []

    def add(value: str) -> None:
        value = clean_text(value, max_chars=180)
        if len(value) >= 8 and value not in candidates:
            candidates.append(value)

    add(normalized)
    add(cleaned)
    for part in sorted(parts, key=len, reverse=True):
        add(part)
        if len(part) > 42:
            add(part[:42])
            add(part[-42:])
    if len(cleaned) > 42:
        add(cleaned[:42])
        add(cleaned[-42:])
    words = re.findall(r"[\w\u4e00-\u9fff]{2,}", cleaned)
    for index in range(0, max(0, len(words) - 3)):
        add("".join(words[index : index + 4]))
        if len(candidates) >= 12:
            break
    return candidates[:12]

def safe_error(exc: BaseException | str) -> str:
    text = str(exc)
    return clean_text(text, max_chars=1000)

def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def load_default_env_files() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in [root / ".env", root / "myagent" / "backend" / ".env", root / "competitorsmart-main" / ".env"]:
        load_env_file(path)

def is_public_http_url(url: str) -> bool:
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host or host == "localhost" or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )

def filename_for_url(prefix: str, url: str, extension: str = "png") -> str:
    parsed = urlparse(url)
    host = parsed.netloc or parsed.hostname or "page"
    host = re.sub(r"[^\w.\-]+", "_", host)[:80].strip("_") or "page"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    safe_prefix = re.sub(r"[^\w.\-]+", "_", prefix)[:60].strip("_") or "shot"
    safe_extension = extension.lstrip(".") or "txt"
    return f"{safe_prefix}_{host}_{digest}.{safe_extension}"

def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response JSON is not an object.")
    return value

