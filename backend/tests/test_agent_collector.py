from __future__ import annotations

import sys
from types import ModuleType

import pytest

from myagent.backend.app.agents import collector
from myagent.backend.app.config import Settings
from myagent.backend.app.models import CreateTaskRequest, RawSource


def test_collector_text_and_url_helpers_clean_inputs():
    html = """
    <html><head><title>  Example   Product  </title><style>.x{}</style></head>
    <body><script>alert(1)</script><h1>Hello</h1><p>World</p></body></html>
    """

    assert collector._clean_text(html) == "Example Product Hello World"
    assert collector._title_from_html(html, "fallback") == "Example Product"
    assert collector._title_from_html("<p>No title</p>", "fallback") == "fallback"
    assert collector._normalize_url("example.com/pricing") == "https://example.com/pricing"
    assert collector._normalize_url(" manual://note ") == "manual://note"
    assert collector._is_public_http_url("https://example.com") is True
    assert collector._is_public_http_url("http://localhost:8000") is False
    assert collector._is_public_http_url("http://127.0.0.1/page") is False
    assert collector._is_public_http_url("https://internal.local/page") is False


def test_collector_builds_manual_sources_for_competitor_profile(sample_request):
    sources = collector._competitor_manual_sources(sample_request.competitor_profiles[0])

    fields = {source.metadata["field"] for source in sources}
    assert {"website", "website_copy", "sales_notes", "win_loss_notes", "tags"} <= fields
    assert any(source.url == "https://betasuite.example" for source in sources)
    assert all(source.source_type == "manual" for source in sources)


def test_collector_builds_search_queries_from_task_context(sample_request):
    queries = collector._search_queries(sample_request)

    assert any("Acme Workspace" in query for query in queries)
    assert any("BetaSuite pricing plans" in query for query in queries)
    assert any("collaboration software" in query for query in queries)


def test_collector_selects_public_screenshot_candidates(sample_request):
    sources = [
        RawSource(
            url="https://betasuite.example/pricing",
            title="Pricing plans",
            content="plans",
            source_type="search",
        ),
        RawSource(
            url="http://localhost/pricing",
            title="Pricing local",
            content="plans",
            source_type="search",
        ),
    ]

    urls = collector._candidate_screenshot_urls(sample_request, sources)

    assert ("https://betasuite.example", "BetaSuite official website") in urls
    assert any(url == "https://betasuite.example/pricing" for url, _label in urls)
    assert all("localhost" not in url for url, _label in urls)


@pytest.mark.asyncio
async def test_fetch_url_downloads_and_extracts_page(monkeypatch):
    calls = {}

    class FakeResponse:
        text = "<html><title>Fetched Title</title><body><h1>Fetched body</h1></body></html>"
        url = "https://example.com/final"
        status_code = 200

        def raise_for_status(self):
            calls["raised"] = False

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            calls["url"] = url
            return FakeResponse()

    monkeypatch.setattr(collector.httpx, "AsyncClient", FakeAsyncClient)

    source = await collector.fetch_url("example.com/page")

    assert calls["url"] == "https://example.com/page"
    assert source.url == "https://example.com/final"
    assert source.title == "Fetched Title"
    assert source.content == "Fetched Title Fetched body"
    assert source.metadata["status_code"] == 200


@pytest.mark.asyncio
async def test_tavily_search_maps_results(monkeypatch, app_settings):
    calls = {}

    class FakeAsyncTavilyClient:
        def __init__(self, api_key: str):
            calls["api_key"] = api_key

        async def search(self, **kwargs):
            calls["kwargs"] = kwargs
            return {
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "Result A",
                        "raw_content": "<p>Result body</p>",
                        "score": 0.9,
                    },
                    {"url": "https://example.com/empty", "title": "Empty"},
                ]
            }

    fake_module = ModuleType("tavily")
    fake_module.AsyncTavilyClient = FakeAsyncTavilyClient
    monkeypatch.setitem(sys.modules, "tavily", fake_module)

    settings = app_settings.model_copy(update={"tavily_api_key": "tvly", "tavily_max_results": 3})
    sources = await collector.tavily_search("competitor pricing", settings)

    assert calls["api_key"] == "tvly"
    assert calls["kwargs"]["max_results"] == 3
    assert len(sources) == 1
    assert sources[0].content == "Result body"
    assert sources[0].metadata == {"score": 0.9, "query": "competitor pricing", "provider": "tavily"}


@pytest.mark.asyncio
async def test_duckduckgo_search_maps_results_and_failures(monkeypatch, app_settings):
    class FakeDDGS:
        def text(self, query: str, max_results: int):
            assert query == "beta pricing"
            assert max_results == 2
            return [{"href": "https://example.com/result", "title": "Result", "body": "<b>Body</b>"}]

    fake_module = ModuleType("ddgs")
    fake_module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)

    settings = app_settings.model_copy(update={"duckduckgo_max_results": 2})
    sources = await collector.duckduckgo_search("beta pricing", settings)

    assert sources[0].url == "https://example.com/result"
    assert sources[0].content == "Body"
    assert sources[0].metadata["provider"] == "duckduckgo"

    class FailingDDGS:
        def text(self, query: str, max_results: int):
            raise RuntimeError("network unavailable")

    fake_module.DDGS = FailingDDGS
    failed = await collector.duckduckgo_search("beta pricing", settings)

    assert failed[0].source_type == "manual"
    assert failed[0].metadata["error"] == "network unavailable"
    assert failed[0].metadata["provider"] == "duckduckgo"


@pytest.mark.asyncio
async def test_collect_sources_combines_search_url_manual_and_screenshots(monkeypatch, app_settings, sample_request):
    async def fake_tavily(query: str, settings: Settings):
        return [
            RawSource(
                url="https://betasuite.example/pricing",
                title="BetaSuite pricing",
                content="pricing search result",
                source_type="search",
                metadata={"provider": "tavily", "query": query},
            )
        ]

    async def fake_duckduckgo(query: str, settings: Settings):
        return [
            RawSource(
                url="https://betasuite.example/pricing",
                title="BetaSuite pricing",
                content="duplicate search result",
                source_type="search",
                metadata={"provider": "duckduckgo", "query": query},
            )
        ]

    async def fake_fetch(url: str):
        return RawSource(url=url, title="Fetched page", content="Fetched content", source_type="url")

    async def fake_capture(url: str, label: str, task_id: str, settings: Settings):
        return RawSource(
            url=url,
            title=f"Screenshot: {label}",
            content="Image path: screenshots/task_test/page.png",
            source_type="screenshot",
            metadata={"image_path": "screenshots/task_test/page.png", "image_alt": label},
        )

    monkeypatch.setattr(collector, "tavily_search", fake_tavily)
    monkeypatch.setattr(collector, "duckduckgo_search", fake_duckduckgo)
    monkeypatch.setattr(collector, "fetch_url", fake_fetch)
    monkeypatch.setattr(collector, "capture_screenshot", fake_capture)

    request = sample_request.model_copy(update={"enable_screenshots": True})
    sources = await collector.collect_sources(request, app_settings, task_id="task_test")

    assert any(source.metadata.get("field") == "website_copy" for source in sources)
    assert any(source.source_type == "search" and source.metadata["provider"] == "tavily" for source in sources)
    assert any(source.source_type == "search" and source.metadata["provider"] == "duckduckgo" for source in sources)
    assert any(source.source_type == "url" for source in sources)
    assert any(source.source_type == "screenshot" for source in sources)


@pytest.mark.asyncio
async def test_collect_sources_returns_task_input_fallback_when_nothing_collected(monkeypatch, app_settings):
    async def no_search(query: str, settings: Settings):
        return []

    monkeypatch.setattr(collector, "tavily_search", no_search)
    monkeypatch.setattr(collector, "duckduckgo_search", no_search)

    request = CreateTaskRequest(target_product="Acme Workspace", competitors=[])
    sources = await collector.collect_sources(request, app_settings, task_id="task_empty")

    assert len(sources) == 1
    assert sources[0].url == "manual://task-input"
    assert sources[0].source_type == "manual"
