from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from .browser import Screenshotter
from .llm import rank_link_candidates_with_llm
from .models import LinkCandidate, LinkDecision, SearchResult
from .utils import clean_text, is_public_http_url, safe_error


LOW_VALUE_LINK_PATTERNS = [
    "login",
    "signin",
    "sign-in",
    "signup",
    "register",
    "privacy",
    "terms",
    "cookie",
    "contact",
    "about",
    "download",
    "mailto:",
    "javascript:",
    "share",
]
PAGINATION_TEXT_PATTERNS = [
    "next",
    "next page",
    "下一页",
    "下页",
    "查看更多",
    "更多",
    "load more",
    "older",
]


def normalize_host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def same_domain(left: str, right: str) -> bool:
    left_host = normalize_host(left)
    right_host = normalize_host(right)
    return bool(left_host and right_host and (left_host == right_host or right_host.endswith("." + left_host) or left_host.endswith("." + right_host)))


def normalize_link(base_url: str, href: str) -> str:
    return urljoin(base_url, href or "").split("#", 1)[0].strip()


def is_low_value_link(url: str, text: str = "") -> bool:
    lowered = f"{url} {text}".lower()
    return any(pattern in lowered for pattern in LOW_VALUE_LINK_PATTERNS)


def extract_link_candidates_from_page(
    page: Any,
    source_url: str,
    *,
    depth: int,
    max_candidates: int,
) -> list[LinkCandidate]:
    raw_links = page.evaluate(
        """
        () => {
            const clean = (value) => (value || '').replace(/\\s+/g, ' ').trim();
            const links = [];
            const nextLinks = Array.from(document.querySelectorAll('link[rel~="next"]')).map((item) => ({
                href: item.href || item.getAttribute('href') || '',
                text: 'rel=next',
                title: item.title || '',
                rel: item.rel || 'next',
                context: 'link rel next'
            }));
            for (const item of nextLinks) links.push(item);
            for (const anchor of Array.from(document.querySelectorAll('a[href]'))) {
                const box = anchor.getBoundingClientRect();
                const text = clean(anchor.innerText || anchor.textContent || anchor.getAttribute('aria-label') || '');
                const parent = anchor.closest('article, section, main, li, div, p');
                links.push({
                    href: anchor.href || anchor.getAttribute('href') || '',
                    text,
                    title: clean(anchor.title || anchor.getAttribute('aria-label') || ''),
                    rel: anchor.rel || '',
                    context: clean(parent ? parent.innerText : ''),
                    top: box.top + window.scrollY
                });
            }
            return links.slice(0, 500);
        }
        """
    )
    candidates: list[LinkCandidate] = []
    seen: set[str] = set()
    for item in raw_links if isinstance(raw_links, list) else []:
        if not isinstance(item, dict):
            continue
        url = normalize_link(source_url, str(item.get("href") or ""))
        if not is_public_http_url(url) or url in seen:
            continue
        seen.add(url)
        text = clean_text(item.get("text") or item.get("title") or "", max_chars=300)
        rel = clean_text(item.get("rel") or "", max_chars=80).lower()
        context = clean_text(item.get("context") or "", max_chars=700)
        discovery_type = "pagination" if is_pagination_candidate(url, text, rel) else "relevant_link"
        candidates.append(
            LinkCandidate(
                id=f"link_{len(candidates) + 1:04d}",
                source_url=source_url,
                url=url,
                text=text,
                title=clean_text(item.get("title") or "", max_chars=300),
                rel=rel,
                context=context,
                discovery_type=discovery_type,
                depth=depth,
            )
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


def is_pagination_candidate(url: str, text: str, rel: str = "") -> bool:
    lowered_text = (text or "").strip().lower()
    if "next" in (rel or "").lower():
        return True
    if any(pattern in lowered_text for pattern in PAGINATION_TEXT_PATTERNS):
        return True
    return bool(re.search(r"([?&](page|p)=\d+|/page/\d+|/p/\d+)", url, flags=re.IGNORECASE))


def filter_candidates(
    candidates: list[LinkCandidate],
    *,
    source_url: str,
    link_scope: str,
    allowlist: set[str],
    include_pagination: bool,
    include_relevant_links: bool,
) -> list[LinkCandidate]:
    filtered: list[LinkCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.url in seen:
            continue
        seen.add(candidate.url)
        if is_low_value_link(candidate.url, candidate.text):
            continue
        if link_scope == "same_domain" and not same_domain(source_url, candidate.url):
            continue
        if link_scope == "allowlist" and normalize_host(candidate.url) not in allowlist:
            continue
        if candidate.discovery_type == "pagination" and include_pagination:
            filtered.append(candidate)
        elif candidate.discovery_type != "pagination" and include_relevant_links:
            filtered.append(candidate)
    return filtered


def heuristic_decisions(
    query: str,
    candidates: list[LinkCandidate],
    *,
    selected_limit: int,
    filter_method: str,
) -> list[LinkDecision]:
    query_terms = [item.lower() for item in re.findall(r"[\w\u4e00-\u9fff]{2,}", query)]
    decisions: list[LinkDecision] = []
    selected_count = 0
    for candidate in candidates:
        haystack = f"{candidate.text} {candidate.title} {candidate.url} {candidate.context}".lower()
        score = 1.0 if candidate.discovery_type == "pagination" else 0.0
        if query_terms:
            matches = sum(1 for term in query_terms if term in haystack)
            score = max(score, min(1.0, matches / max(1, len(query_terms))))
        selected = selected_count < selected_limit and score > 0
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
                relevance=round(score, 4),
                reason="pagination link" if candidate.discovery_type == "pagination" else "heuristic query-term match",
                filter_method=filter_method,
            )
        )
    return decisions


def decide_links(
    query: str,
    candidates: list[LinkCandidate],
    *,
    filter_method: str,
    selected_limit: int,
) -> tuple[list[LinkDecision], list[dict[str, Any]]]:
    if not candidates or selected_limit <= 0:
        return [], []
    pagination = [item for item in candidates if item.discovery_type == "pagination"]
    relevant = [item for item in candidates if item.discovery_type != "pagination"]
    decisions = heuristic_decisions(query, pagination, selected_limit=selected_limit, filter_method="heuristic")
    remaining = max(0, selected_limit - sum(1 for item in decisions if item.selected))
    errors: list[dict[str, Any]] = []

    if relevant and remaining > 0:
        if filter_method == "none":
            decisions.extend(heuristic_decisions(query, relevant, selected_limit=remaining, filter_method="none"))
        elif filter_method == "heuristic":
            decisions.extend(heuristic_decisions(query, relevant, selected_limit=remaining, filter_method="heuristic"))
        else:
            llm_decisions, llm_errors = rank_link_candidates_with_llm(query, relevant, selected_limit=remaining)
            decisions.extend(llm_decisions)
            errors.extend(llm_errors)
    return decisions, errors


def result_from_decision(source: SearchResult, decision: LinkDecision, *, rank: int) -> SearchResult:
    return SearchResult(
        method=source.method,
        query=source.query,
        rank=rank,
        title=decision.text or decision.url,
        url=decision.url,
        snippet=decision.reason or "",
        source_url=decision.source_url,
        discovery_type=decision.discovery_type,
        depth=source.depth + 1,
        link_relevance=decision.relevance,
        link_reason=decision.reason,
    )


def expand_results_with_links(
    results: list[SearchResult],
    run_dir: Path,
    *,
    query: str,
    follow_pages: str,
    link_scope: str,
    link_allowlist: list[str],
    max_link_depth: int,
    max_linked_pages_per_result: int,
    max_candidate_links: int,
    link_filter: str,
    viewport_height: int,
) -> tuple[list[SearchResult], dict[str, Any]]:
    payload: dict[str, Any] = {"candidates": [], "decisions": [], "errors": []}
    if follow_pages == "none" or max_link_depth < 1 or max_linked_pages_per_result < 1:
        return [], payload

    include_pagination = follow_pages in {"pagination", "both"}
    include_relevant_links = follow_pages in {"relevant_links", "both"}
    allowlist = {item.strip().lower() for item in link_allowlist if item.strip()}
    expanded: list[SearchResult] = []
    seen_urls = {result.url for result in results if result.url}

    with Screenshotter(run_dir, full_page=False, viewport_height=viewport_height) as screenshotter:
        for result in results:
            if result.rank is None or not result.url or result.depth >= max_link_depth:
                continue
            page, _, error = screenshotter.load_quote_dom_page(result.url)
            if error or page is None:
                payload["errors"].append({"stage": "link_expansion_load", "url": result.url, "error": error or "Page could not be loaded."})
                continue
            try:
                raw_candidates = extract_link_candidates_from_page(
                    page,
                    result.url,
                    depth=result.depth + 1,
                    max_candidates=max_candidate_links,
                )
                candidates = filter_candidates(
                    raw_candidates,
                    source_url=result.url,
                    link_scope=link_scope,
                    allowlist=allowlist,
                    include_pagination=include_pagination,
                    include_relevant_links=include_relevant_links,
                )
                payload["candidates"].extend(asdict(item) for item in candidates)
                decisions, errors = decide_links(
                    query,
                    candidates,
                    filter_method=link_filter,
                    selected_limit=max_linked_pages_per_result,
                )
                payload["decisions"].extend(asdict(item) for item in decisions)
                payload["errors"].extend(errors)
                for decision in decisions:
                    if not decision.selected or decision.url in seen_urls:
                        continue
                    seen_urls.add(decision.url)
                    expanded.append(result_from_decision(result, decision, rank=len(results) + len(expanded) + 1))
            except Exception as exc:
                payload["errors"].append({"stage": "link_expansion", "url": result.url, "error": safe_error(exc)})
            finally:
                try:
                    page.close()
                except Exception:
                    pass

    write_link_decisions(run_dir, payload)
    return expanded, payload


def write_link_decisions(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = run_dir / "link_decisions.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
