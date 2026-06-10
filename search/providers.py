from __future__ import annotations

import json
import os
import warnings
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from .models import SearchResult
from .utils import clean_text, safe_error


SEARCHAPI_ENGINES = {
    "searchapi_google": "google",
    "searchapi_baidu": "baidu",
    "searchapi_duckduckgo": "duckduckgo",
}
SERPAPI_ENGINES = {
    "serpapi_google": "google",
    "serpapi_baidu": "baidu",
}


def error_result(method: str, query: str, title: str, error: str) -> list[SearchResult]:
    return [
        SearchResult(
            method=method,
            query=query,
            rank=None,
            title=title,
            url="",
            snippet="",
            error=error,
        )
    ]


def fetch_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("JSON response root is not an object.")
    return payload


def duckduckgo_search_url(query: str, method: str) -> str:
    encoded = quote_plus(query)
    if method == "duckduckgo_news":
        return f"https://duckduckgo.com/?q={encoded}&iar=news&ia=news"
    return f"https://duckduckgo.com/?q={encoded}"

def ddgs_classes() -> list[tuple[str, Any]]:
    classes: list[tuple[str, Any]] = []
    try:
        from ddgs import DDGS  # type: ignore

        classes.append(("ddgs", DDGS))
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS as DuckDuckGoSearchDDGS  # type: ignore

        classes.append(("duckduckgo-search", DuckDuckGoSearchDDGS))
    except ImportError:
        pass
    if not classes:
        raise ImportError("Neither `ddgs` nor `duckduckgo-search` is installed.")
    return classes

def close_if_possible(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()

def search_duckduckgo(query: str, max_results: int, method: str) -> list[SearchResult]:
    errors: list[str] = []
    raw_results: list[dict[str, Any]] = []
    try:
        classes = ddgs_classes()
    except Exception as exc:
        return [
            SearchResult(
                method=method,
                query=query,
                rank=None,
                title=f"{method} failed",
                url="",
                snippet="",
                error=safe_error(exc),
            )
        ]

    for package_name, ddgs_cls in classes:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                client = ddgs_cls()
                try:
                    if method == "duckduckgo_news":
                        raw_results = list(client.news(query, max_results=max_results))
                    else:
                        raw_results = list(client.text(query, max_results=max_results))
                finally:
                    close_if_possible(client)
            break
        except Exception as exc:
            errors.append(f"{package_name}: {safe_error(exc)}")
    else:
        return [
            SearchResult(
                method=method,
                query=query,
                rank=None,
                title=f"{method} failed",
                url="",
                snippet="",
                error="; ".join(errors),
            )
        ]

    results: list[SearchResult] = []
    for index, item in enumerate(raw_results[:max_results], start=1):
        url = clean_text(item.get("href") or item.get("url") or "")
        title = clean_text(item.get("title") or url or f"{method} result {index}", max_chars=500)
        snippet = clean_text(item.get("body") or item.get("snippet") or item.get("excerpt") or "")
        if method == "duckduckgo_news":
            source = clean_text(item.get("source") or "")
            date = clean_text(item.get("date") or "")
            details = " | ".join(part for part in [source, date] if part)
            if details:
                snippet = f"{snippet} ({details})" if snippet else details
        results.append(
            SearchResult(
                method=method,
                query=query,
                rank=index,
                title=title,
                url=url,
                snippet=snippet,
            )
        )
    return results

def search_tavily(query: str, max_results: int, *, search_depth: str = "basic") -> list[SearchResult]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return [
            SearchResult(
                method="tavily",
                query=query,
                rank=None,
                title="Tavily skipped",
                url="",
                snippet="",
                error="TAVILY_API_KEY is not set.",
            )
        ]

    try:
        from tavily import TavilyClient  # type: ignore
    except ImportError as exc:
        return [
            SearchResult(
                method="tavily",
                query=query,
                rank=None,
                title="Tavily unavailable",
                url="",
                snippet="",
                error=f"tavily-python is not installed: {safe_error(exc)}",
            )
        ]

    try:
        client = TavilyClient(api_key=api_key)
        search_kwargs: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        if search_depth == "auto":
            search_kwargs["auto_parameters"] = True
        else:
            search_kwargs["search_depth"] = search_depth
        response = client.search(
            **search_kwargs,
        )
        raw_results = response.get("results", []) if isinstance(response, dict) else []
    except Exception as exc:
        return [
            SearchResult(
                method="tavily",
                query=query,
                rank=None,
                title="Tavily search failed",
                url="",
                snippet="",
                error=safe_error(exc),
            )
        ]

    results: list[SearchResult] = []
    for index, item in enumerate(raw_results[:max_results], start=1):
        url = clean_text(item.get("url") or "")
        title = clean_text(item.get("title") or url or f"Tavily result {index}", max_chars=500)
        snippet = clean_text(item.get("content") or item.get("snippet") or "")
        results.append(
            SearchResult(
                method="tavily",
                query=query,
                rank=index,
                title=title,
                url=url,
                snippet=snippet,
            )
        )
    return results


def search_brave(query: str, max_results: int) -> list[SearchResult]:
    method = "brave_web"
    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        return error_result(method, query, "Brave skipped", "BRAVE_SEARCH_API_KEY is not set.")

    params = urlencode({"q": query, "count": max(1, min(max_results, 20)), "offset": 0})
    url = f"https://api.search.brave.com/res/v1/web/search?{params}"
    try:
        payload = fetch_json(
            url,
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
        )
    except HTTPError as exc:
        return error_result(method, query, "Brave search failed", f"HTTP {exc.code}: {safe_error(exc)}")
    except (URLError, OSError, json.JSONDecodeError, ValueError) as exc:
        return error_result(method, query, "Brave search failed", safe_error(exc))

    raw_results = []
    web = payload.get("web")
    if isinstance(web, dict) and isinstance(web.get("results"), list):
        raw_results = web["results"]
    results: list[SearchResult] = []
    for index, item in enumerate(raw_results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        url_value = clean_text(item.get("url") or "")
        title = clean_text(item.get("title") or url_value or f"Brave result {index}", max_chars=500)
        snippet = clean_text(item.get("description") or item.get("snippet") or "")
        results.append(SearchResult(method=method, query=query, rank=index, title=title, url=url_value, snippet=snippet))
    if not results:
        return error_result(method, query, "Brave returned no results", "No organic web results were returned.")
    return results


def search_searchapi(query: str, max_results: int, method: str) -> list[SearchResult]:
    api_key = os.getenv("SEARCHAPI_API_KEY", "").strip()
    if not api_key:
        return error_result(method, query, "SearchApi skipped", "SEARCHAPI_API_KEY is not set.")

    engine = SEARCHAPI_ENGINES[method]
    params = urlencode({"engine": engine, "q": query, "api_key": api_key, "num": max_results})
    url = f"https://www.searchapi.io/api/v1/search?{params}"
    try:
        payload = fetch_json(url)
    except HTTPError as exc:
        return error_result(method, query, "SearchApi search failed", f"HTTP {exc.code}: {safe_error(exc)}")
    except (URLError, OSError, json.JSONDecodeError, ValueError) as exc:
        return error_result(method, query, "SearchApi search failed", safe_error(exc))

    raw_results = payload.get("organic_results")
    if not isinstance(raw_results, list):
        raw_results = payload.get("results") if isinstance(payload.get("results"), list) else []
    results: list[SearchResult] = []
    for index, item in enumerate(raw_results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        url_value = clean_text(item.get("link") or item.get("url") or "")
        title = clean_text(item.get("title") or url_value or f"SearchApi result {index}", max_chars=500)
        snippet = clean_text(item.get("snippet") or item.get("description") or "")
        results.append(SearchResult(method=method, query=query, rank=index, title=title, url=url_value, snippet=snippet))
    if not results:
        return error_result(method, query, "SearchApi returned no results", "No organic results were returned.")
    return results


def search_serpapi(query: str, max_results: int, method: str) -> list[SearchResult]:
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return error_result(method, query, "SerpApi skipped", "SERPAPI_API_KEY is not set.")

    engine = SERPAPI_ENGINES[method]
    params = urlencode({"engine": engine, "q": query, "api_key": api_key, "num": max_results})
    url = f"https://serpapi.com/search.json?{params}"
    try:
        payload = fetch_json(url)
    except HTTPError as exc:
        return error_result(method, query, "SerpApi search failed", f"HTTP {exc.code}: {safe_error(exc)}")
    except (URLError, OSError, json.JSONDecodeError, ValueError) as exc:
        return error_result(method, query, "SerpApi search failed", safe_error(exc))

    raw_results = payload.get("organic_results")
    if not isinstance(raw_results, list):
        raw_results = []
    results: list[SearchResult] = []
    for index, item in enumerate(raw_results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        url_value = clean_text(item.get("link") or item.get("url") or "")
        title = clean_text(item.get("title") or url_value or f"SerpApi result {index}", max_chars=500)
        snippet = clean_text(item.get("snippet") or item.get("description") or "")
        results.append(SearchResult(method=method, query=query, rank=index, title=title, url=url_value, snippet=snippet))
    if not results:
        return error_result(method, query, "SerpApi returned no results", "No organic results were returned.")
    return results

def collect_results(
    query: str,
    methods: list[str],
    max_results: int,
    *,
    tavily_search_depth: str = "basic",
) -> list[SearchResult]:
    all_results: list[SearchResult] = []
    for method in methods:
        if method in {"duckduckgo_text", "duckduckgo_news"}:
            all_results.extend(search_duckduckgo(query, max_results, method))
        elif method == "tavily":
            all_results.extend(search_tavily(query, max_results, search_depth=tavily_search_depth))
        elif method == "brave_web":
            all_results.extend(search_brave(query, max_results))
        elif method in SEARCHAPI_ENGINES:
            all_results.extend(search_searchapi(query, max_results, method))
        elif method in SERPAPI_ENGINES:
            all_results.extend(search_serpapi(query, max_results, method))
    return all_results
