import pytest
from types import ModuleType

from myagent.backend.app.agents import collector
from myagent.backend.app.config import Settings
from myagent.backend.app.models import CreateTaskRequest, RawSource


def test_clean_source_text_uses_readability_when_enabled(monkeypatch):
    monkeypatch.setattr(collector, "_extract_readable_text", lambda value, max_chars=12000: "Readable article body")

    assert collector.clean_source_text("<nav>menu</nav><article>body</article>", use_readability=True) == "Readable article body"
    assert "menu" in collector.clean_source_text("<nav>menu</nav><article>body</article>", use_readability=False)


@pytest.mark.asyncio
async def test_collect_sources_uses_manual_competitor_context(monkeypatch, tmp_path):
    async def fake_search(query: str, settings: Settings):
        return [
            RawSource(
                url=f"https://example.com/search/{query[:8]}",
                title="Mock search",
                content="Mock search content",
                source_type="search",
            )
        ]

    monkeypatch.setattr(collector, "duckduckgo_search", fake_search)
    async def no_tavily(query: str, settings: Settings):
        return []

    async def fake_fetch(url: str, **kwargs):
        return RawSource(url=url, title="Fetched page", content="Fetched content", source_type="url")

    monkeypatch.setattr(collector, "tavily_search", no_tavily)
    monkeypatch.setattr(collector, "fetch_url", fake_fetch)

    settings = Settings(openai_api_key=None, tavily_api_key=None, sqlite_path=tmp_path / "app.db")
    request = CreateTaskRequest(
        target_product="飞书",
        competitors=[],
        competitor_profiles=[
            {
                "name": "钉钉",
                "website": "https://www.dingtalk.com",
                "website_copy": "企业协同办公平台",
                "sales_notes": "销售常遇到的竞品",
                "win_loss_notes": "流程审批能力强",
            }
        ],
    )

    sources = await collector.collect_sources(request, settings, task_id="task_test")

    assert any(source.metadata.get("field") == "website_copy" for source in sources)
    assert any(source.metadata.get("field") == "sales_notes" for source in sources)
    assert any(source.source_type == "search" for source in sources)


@pytest.mark.asyncio
async def test_collect_sources_does_not_capture_screenshots_when_disabled(monkeypatch, tmp_path):
    async def fail_screenshot(*args, **kwargs):
        raise AssertionError("screenshot should not be called")

    async def no_search(query: str, settings: Settings):
        return []

    async def no_tavily(query: str, settings: Settings):
        return []

    async def fake_fetch(url: str, **kwargs):
        return RawSource(url=url, title="Fetched page", content="Fetched content", source_type="url")

    monkeypatch.setattr(collector, "duckduckgo_search", no_search)
    monkeypatch.setattr(collector, "tavily_search", no_tavily)
    monkeypatch.setattr(collector, "capture_screenshot", fail_screenshot)
    monkeypatch.setattr(collector, "fetch_url", fake_fetch)

    settings = Settings(openai_api_key=None, tavily_api_key=None, sqlite_path=tmp_path / "app.db")
    request = CreateTaskRequest(
        target_product="飞书",
        competitor_profiles=[{"name": "钉钉", "website": "https://www.dingtalk.com"}],
        enable_screenshots=False,
    )

    sources = await collector.collect_sources(request, settings, task_id="task_test")

    assert sources
    assert all(source.source_type != "screenshot" for source in sources)


@pytest.mark.asyncio
async def test_collect_sources_records_screenshot_failure(monkeypatch, tmp_path):
    async def fake_screenshot(*args, **kwargs):
        raise RuntimeError("browser missing")

    async def no_search(query: str, settings: Settings):
        return []

    async def no_tavily(query: str, settings: Settings):
        return []

    async def fake_fetch(url: str, **kwargs):
        return RawSource(url=url, title="Fetched page", content="Fetched content", source_type="url")

    monkeypatch.setattr(collector, "duckduckgo_search", no_search)
    monkeypatch.setattr(collector, "tavily_search", no_tavily)
    monkeypatch.setattr(collector, "capture_screenshot", fake_screenshot)
    monkeypatch.setattr(collector, "fetch_url", fake_fetch)

    settings = Settings(openai_api_key=None, tavily_api_key=None, sqlite_path=tmp_path / "app.db")
    request = CreateTaskRequest(
        target_product="飞书",
        competitor_profiles=[{"name": "钉钉", "website": "https://www.dingtalk.com"}],
        enable_screenshots=True,
    )

    sources = await collector.collect_sources(request, settings, task_id="task_test")

    assert any(source.metadata.get("field") == "screenshot" and source.metadata.get("error") for source in sources)


@pytest.mark.asyncio
async def test_duckduckgo_search_uses_settings_max_results(monkeypatch, tmp_path):
    calls = {}

    class FakeDDGS:
        def text(self, query: str, max_results: int):
            calls["query"] = query
            calls["max_results"] = max_results
            return [
                {
                    "href": "https://example.com/result",
                    "title": "Example result",
                    "body": "Example body",
                }
            ]

    fake_module = ModuleType("ddgs")
    fake_module.DDGS = FakeDDGS
    monkeypatch.setitem(__import__("sys").modules, "ddgs", fake_module)

    settings = Settings(
        openai_api_key=None,
        tavily_api_key=None,
        duckduckgo_max_results=6,
        sqlite_path=tmp_path / "app.db",
    )

    sources = await collector.duckduckgo_search("example query", settings)

    assert calls == {"query": "example query", "max_results": 6}
    assert sources[0].metadata["provider"] == "duckduckgo"
