import json
from pathlib import Path

import pytest

from myagent.search import cli, providers, tavily_collection
from myagent.search.models import PageData, SearchResult


def test_collect_results_dispatches_selected_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, str]] = []

    def fake_duckduckgo(query: str, max_results: int, method: str) -> list[SearchResult]:
        calls.append((query, max_results, method))
        return [SearchResult(method=method, query=query, rank=1, title=method, url="https://example.com", snippet="")]

    def fake_tavily(query: str, max_results: int, *, search_depth: str = "basic") -> list[SearchResult]:
        calls.append((query, max_results, "tavily"))
        return [SearchResult(method="tavily", query=query, rank=1, title="tavily", url="https://example.org", snippet="")]

    def fake_brave(query: str, max_results: int) -> list[SearchResult]:
        calls.append((query, max_results, "brave_web"))
        return [SearchResult(method="brave_web", query=query, rank=1, title="brave", url="https://example.net", snippet="")]

    monkeypatch.setattr(providers, "search_duckduckgo", fake_duckduckgo)
    monkeypatch.setattr(providers, "search_tavily", fake_tavily)
    monkeypatch.setattr(providers, "search_brave", fake_brave)

    results = providers.collect_results("query", ["duckduckgo_text", "tavily", "brave_web"], 2)

    assert [result.method for result in results] == ["duckduckgo_text", "tavily", "brave_web"]
    assert calls == [("query", 2, "duckduckgo_text"), ("query", 2, "tavily"), ("query", 2, "brave_web")]


def test_tavily_without_api_key_returns_skip_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = providers.search_tavily("query", 3)[0]

    assert result.method == "tavily"
    assert result.rank is None
    assert result.error == "TAVILY_API_KEY is not set."


def test_tavily_search_depth_maps_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "key"

        def search(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {"results": [{"title": "A", "url": "https://example.com/a", "content": "Snippet"}]}

    monkeypatch.setenv("TAVILY_API_KEY", "key")
    monkeypatch.setitem(__import__("sys").modules, "tavily", type("FakeModule", (), {"TavilyClient": FakeClient}))

    basic = providers.search_tavily("query", 1, search_depth="advanced")[0]
    auto = providers.search_tavily("query", 1, search_depth="auto")[0]

    assert basic.url == auto.url == "https://example.com/a"
    assert calls[0]["search_depth"] == "advanced"
    assert "auto_parameters" not in calls[0]
    assert calls[1]["auto_parameters"] is True
    assert "search_depth" not in calls[1]


def test_new_api_methods_without_keys_return_skip_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("SEARCHAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    assert providers.search_brave("query", 3)[0].error == "BRAVE_SEARCH_API_KEY is not set."
    assert providers.search_searchapi("query", 3, "searchapi_google")[0].error == "SEARCHAPI_API_KEY is not set."
    assert providers.search_serpapi("query", 3, "serpapi_google")[0].error == "SERPAPI_API_KEY is not set."


def test_brave_maps_web_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "key")
    monkeypatch.setattr(
        providers,
        "fetch_json",
        lambda url, headers=None, timeout=30: {
            "web": {"results": [{"title": "A", "url": "https://example.com/a", "description": "Snippet"}]}
        },
    )

    result = providers.search_brave("query", 1)[0]

    assert result.method == "brave_web"
    assert result.title == "A"
    assert result.url == "https://example.com/a"
    assert result.snippet == "Snippet"


def test_searchapi_and_serpapi_map_organic_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCHAPI_API_KEY", "key")
    monkeypatch.setenv("SERPAPI_API_KEY", "key")
    monkeypatch.setattr(
        providers,
        "fetch_json",
        lambda url, headers=None, timeout=30: {
            "organic_results": [{"title": "A", "link": "https://example.com/a", "snippet": "Snippet"}]
        },
    )

    searchapi = providers.search_searchapi("query", 1, "searchapi_google")[0]
    serpapi = providers.search_serpapi("query", 1, "serpapi_google")[0]

    assert searchapi.method == "searchapi_google"
    assert serpapi.method == "serpapi_google"
    assert searchapi.url == serpapi.url == "https://example.com/a"


def test_cli_requires_query_unless_locate_extraction() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--content", "none", "--screenshots", "none"])

    assert exc.value.code == 2


def test_cli_smoke_without_network_or_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.setattr(
        cli,
        "collect_results",
        lambda query, methods, max_results, tavily_search_depth="basic": [
            SearchResult(
                method=methods[0],
                query=query,
                rank=1,
                title="Title",
                url="https://example.com",
                snippet="Snippet",
            )
        ],
    )

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--methods",
            "duckduckgo_text",
            "--max-results",
            "1",
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Saved 1 result row(s)." in output
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    payload = json.loads((run_dirs[0] / "results.json").read_text(encoding="utf-8"))
    assert payload["query"] == "demo"
    assert payload["results"][0]["title"] == "Title"
    assert payload["deep_backend"] == "none"
    assert "follow_pages" not in payload
    assert payload["tavily_collection"]["enabled"] is False


def test_cli_default_search_does_not_deep_expand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.setattr(
        cli,
        "collect_results",
        lambda query, methods, max_results, tavily_search_depth="basic": [
            SearchResult(
                method=methods[0],
                query=query,
                rank=1,
                title="Title",
                url="https://example.com/a",
                snippet="Snippet",
            )
        ],
    )
    monkeypatch.setattr(cli, "expand_results_with_links", lambda *args, **kwargs: pytest.fail("local expansion should not run"))
    monkeypatch.setattr(cli, "collect_tavily_pages", lambda *args, **kwargs: pytest.fail("tavily deep search should not run"))

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--methods",
            "duckduckgo_text",
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    run_dir = next(tmp_path.iterdir())
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["deep_backend"] == "none"
    assert payload["tavily_collection"]["enabled"] is False


def test_runner_filters_duplicate_urls_before_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from myagent.search import runner

    monkeypatch.setattr(runner, "load_default_env_files", lambda: None)
    monkeypatch.setattr(
        runner,
        "collect_results",
        lambda query, methods, max_results, tavily_search_depth="basic": [
            SearchResult(method="duckduckgo_text", query=query, rank=1, title="A", url="https://example.com/a?utm_source=x", snippet="A"),
            SearchResult(method="tavily", query=query, rank=1, title="A duplicate", url="https://example.com/a", snippet="A duplicate"),
            SearchResult(method="duckduckgo_news", query=query, rank=1, title="B", url="https://example.com/b", snippet="B"),
        ],
    )

    result = runner.run_search(
        "demo",
        runner.SearchRunOptions(
            methods=["duckduckgo_text", "tavily", "duckduckgo_news"],
            output_dir=tmp_path,
            content="none",
            screenshots="none",
            collect_chunks=False,
        ),
    )

    assert [item.url for item in result.results] == ["https://example.com/a?utm_source=x", "https://example.com/b"]
    assert result.metadata["skipped_duplicate_count"] == 1
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["skipped_duplicate_count"] == 1


def test_runner_passes_browser_disable_proxy_to_screenshotter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from myagent.search import runner

    captured: dict[str, object] = {}

    class FakeScreenshotter:
        def __init__(self, run_dir: Path, full_page: bool, *, disable_proxy: bool = True) -> None:
            captured["run_dir"] = run_dir
            captured["full_page"] = full_page
            captured["disable_proxy"] = disable_proxy

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def process_result_page(self, *args, **kwargs) -> PageData:
            return PageData(
                content_path="contents/result.txt",
                content_chars=11,
                content_preview="hello world",
                content_source="visible_text",
                access_status="success",
            )

    monkeypatch.setattr(runner, "load_default_env_files", lambda: None)
    monkeypatch.setattr(runner, "Screenshotter", FakeScreenshotter)
    monkeypatch.setattr(
        runner,
        "collect_results",
        lambda query, methods, max_results, tavily_search_depth="basic": [
            SearchResult(
                method="duckduckgo_text",
                query=query,
                rank=1,
                title="Title",
                url="https://example.com/a",
                snippet="Snippet",
            )
        ],
    )

    result = runner.run_search(
        "demo",
        runner.SearchRunOptions(
            methods=["duckduckgo_text"],
            output_dir=tmp_path,
            content="result_pages",
            screenshots="none",
            collect_chunks=False,
            browser_disable_proxy=False,
        ),
    )

    assert captured["disable_proxy"] is False
    assert result.metadata["browser_disable_proxy"] is False
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["browser_disable_proxy"] is False


def test_cli_start_urls_writes_seed_without_provider_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.setattr(cli, "collect_results", lambda *args, **kwargs: pytest.fail("provider search should not run"))
    monkeypatch.setattr(
        cli,
        "expand_results_with_links",
        lambda results, run_dir, **kwargs: ([], {"errors": []}),
    )

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--start-urls",
            "https://example.com, ftp://ignored.example/file, https://example.com",
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    run_dir = next(tmp_path.iterdir())
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["methods"] == ["seed_url"]
    assert payload["start_urls"] == ["https://example.com"]
    assert payload["deep_backend"] == "local_relevant_links"
    assert payload["seed_url_count"] == 1
    assert "follow_pages" not in payload
    assert payload["results"][0]["method"] == "seed_url"
    assert payload["results"][0]["title"] == "Seed page: https://example.com"


def test_cli_start_urls_rejects_when_no_valid_public_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--query",
                "demo",
                "--start-urls",
                "ftp://example.com/file, http://127.0.0.1",
                "--content",
                "none",
                "--screenshots",
                "none",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert exc.value.code == 2


def test_cli_start_urls_local_backend_expands_seed_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expansion_calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.setattr(cli, "collect_results", lambda *args, **kwargs: pytest.fail("provider search should not run"))

    def fake_expand(results: list[SearchResult], run_dir: Path, **kwargs: object) -> tuple[list[SearchResult], dict[str, object]]:
        expansion_calls.append({"results": list(results), "kwargs": kwargs})
        return (
            [
                SearchResult(
                    method="seed_url",
                    query="demo",
                    rank=2,
                    title="Pricing",
                    url="https://example.com/pricing",
                    snippet="",
                    source_url="https://example.com",
                    discovery_type="relevant_link",
                    depth=1,
                )
            ],
            {"errors": []},
        )

    monkeypatch.setattr(cli, "expand_results_with_links", fake_expand)

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--start-urls",
            "https://example.com",
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(expansion_calls) == 1
    assert expansion_calls[0]["results"][0].method == "seed_url"
    assert expansion_calls[0]["kwargs"]["follow_pages"] == "relevant_links"
    run_dir = next(tmp_path.iterdir())
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert [result["url"] for result in payload["results"]] == ["https://example.com", "https://example.com/pricing"]


@pytest.mark.parametrize(
    ("deep_backend", "follow_pages"),
    [
        ("local_pagination", "pagination"),
        ("local_relevant_links", "relevant_links"),
        ("local_all_links", "both"),
    ],
)
def test_cli_search_local_deep_backend_maps_to_follow_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deep_backend: str,
    follow_pages: str,
) -> None:
    expansion_calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.setattr(
        cli,
        "collect_results",
        lambda query, methods, max_results, tavily_search_depth="basic": [
            SearchResult(
                method=methods[0],
                query=query,
                rank=1,
                title="Title",
                url="https://example.com/a",
                snippet="Snippet",
            )
        ],
    )

    def fake_expand(results: list[SearchResult], run_dir: Path, **kwargs: object) -> tuple[list[SearchResult], dict[str, object]]:
        expansion_calls.append({"results": list(results), "kwargs": kwargs})
        return ([], {"errors": []})

    monkeypatch.setattr(cli, "expand_results_with_links", fake_expand)
    monkeypatch.setattr(cli, "collect_tavily_pages", lambda *args, **kwargs: pytest.fail("tavily deep search should not run"))

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--methods",
            "duckduckgo_text",
            "--deep-backend",
            deep_backend,
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(expansion_calls) == 1
    assert expansion_calls[0]["kwargs"]["follow_pages"] == follow_pages


def test_cli_start_urls_tavily_backend_uses_seed_and_default_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tavily_calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.setattr(cli, "collect_results", lambda *args, **kwargs: pytest.fail("provider search should not run"))
    monkeypatch.setattr(cli, "expand_results_with_links", lambda *args, **kwargs: pytest.fail("local expansion should not run"))

    def fake_tavily(results: list[SearchResult], run_dir: Path, **kwargs: object) -> dict[str, object]:
        tavily_calls.append({"results": list(results), "kwargs": kwargs})
        return {"enabled": True, "tools": kwargs["tools"], "seed_urls": kwargs["seed_urls"], "errors": []}

    monkeypatch.setattr(cli, "collect_tavily_pages", fake_tavily)

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--start-urls",
            "https://example.com",
            "--deep-backend",
            "tavily",
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(tavily_calls) == 1
    assert tavily_calls[0]["kwargs"]["methods"] == ["tavily"]
    assert tavily_calls[0]["kwargs"]["tools"] == ["map", "crawl", "extract"]
    assert tavily_calls[0]["kwargs"]["seed_urls"] == ["https://example.com"]
    run_dir = next(tmp_path.iterdir())
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["methods"] == ["seed_url"]
    assert payload["deep_backend"] == "tavily"
    assert payload["tavily_collection"]["tools"] == ["map", "crawl", "extract"]


def test_cli_search_tavily_backend_uses_result_roots_and_default_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tavily_calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.setattr(
        cli,
        "collect_results",
        lambda query, methods, max_results, tavily_search_depth="basic": [
            SearchResult(
                method="duckduckgo_text",
                query=query,
                rank=1,
                title="A",
                url="https://example.com/a",
                snippet="Snippet",
            ),
            SearchResult(
                method="duckduckgo_text",
                query=query,
                rank=2,
                title="B",
                url="https://docs.example.org/path",
                snippet="Snippet",
            ),
        ],
    )
    monkeypatch.setattr(cli, "expand_results_with_links", lambda *args, **kwargs: pytest.fail("local expansion should not run"))

    def fake_tavily(results: list[SearchResult], run_dir: Path, **kwargs: object) -> dict[str, object]:
        tavily_calls.append({"results": list(results), "kwargs": kwargs})
        return {"enabled": True, "tools": kwargs["tools"], "seed_urls": kwargs["seed_urls"], "errors": []}

    monkeypatch.setattr(cli, "collect_tavily_pages", fake_tavily)

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--methods",
            "duckduckgo_text",
            "--deep-backend",
            "tavily",
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(tavily_calls) == 1
    assert tavily_calls[0]["kwargs"]["methods"] == ["duckduckgo_text", "tavily"]
    assert tavily_calls[0]["kwargs"]["tools"] == ["map", "crawl", "extract"]
    assert tavily_calls[0]["kwargs"]["seed_urls"] == []
    run_dir = next(tmp_path.iterdir())
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["methods"] == ["duckduckgo_text"]
    assert payload["deep_backend"] == "tavily"


def test_cli_local_relevant_links_tavily_expands_then_runs_tavily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.setattr(cli, "collect_results", lambda *args, **kwargs: pytest.fail("provider search should not run"))

    def fake_expand(results: list[SearchResult], run_dir: Path, **kwargs: object) -> tuple[list[SearchResult], dict[str, object]]:
        order.append(f"expand:{len(results)}")
        return (
            [
                SearchResult(
                    method="seed_url",
                    query="demo",
                    rank=2,
                    title="Docs",
                    url="https://example.com/docs",
                    snippet="",
                    source_url="https://example.com",
                    discovery_type="relevant_link",
                    depth=1,
                )
            ],
            {"errors": []},
        )

    def fake_tavily(results: list[SearchResult], run_dir: Path, **kwargs: object) -> dict[str, object]:
        order.append(f"tavily:{len(results)}")
        return {"enabled": True, "tools": kwargs["tools"], "seed_urls": kwargs["seed_urls"], "errors": []}

    monkeypatch.setattr(cli, "expand_results_with_links", fake_expand)
    monkeypatch.setattr(cli, "collect_tavily_pages", fake_tavily)

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--start-urls",
            "https://example.com",
            "--deep-backend",
            "local_relevant_links_tavily",
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert order == ["expand:1", "tavily:2"]
    run_dir = next(tmp_path.iterdir())
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["deep_backend"] == "local_relevant_links_tavily"
    assert payload["tavily_collection"]["enabled"] is True


def test_cli_brave_without_key_gracefully_writes_error_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--methods",
            "brave_web",
            "--max-results",
            "1",
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    run_dir = next(tmp_path.iterdir())
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["results"][0]["method"] == "brave_web"
    assert payload["results"][0]["error"] == "BRAVE_SEARCH_API_KEY is not set."


def test_cli_tavily_tools_writes_collection_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_default_env_files", lambda: None)
    monkeypatch.setattr(
        cli,
        "collect_results",
        lambda query, methods, max_results, tavily_search_depth="basic": [
            SearchResult(
                method="tavily",
                query=query,
                rank=1,
                title="Title",
                url="https://example.com",
                snippet="Snippet",
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "collect_tavily_pages",
        lambda results, run_dir, **kwargs: {
            "enabled": True,
            "tools": kwargs["tools"],
            "extract_success": 1,
            "errors": [],
        },
    )

    exit_code = cli.main(
        [
            "--query",
            "demo",
            "--methods",
            "tavily",
            "--tavily-tools",
            "search,extract,map,crawl",
            "--deep-backend",
            "tavily",
            "--content",
            "none",
            "--screenshots",
            "none",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    run_dir = next(tmp_path.iterdir())
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["tavily_collection"]["enabled"] is True
    assert payload["tavily_collection"]["tools"] == ["search", "extract", "map", "crawl"]


def test_tavily_collection_maps_crawls_extracts_and_writes_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def map(self, url: str, **kwargs: object) -> dict[str, object]:
            return {"results": [{"url": "https://example.com/pricing", "title": "Pricing"}]}

        def crawl(self, url: str, **kwargs: object) -> dict[str, object]:
            return {
                "results": [
                    {
                        "url": "https://example.com/docs",
                        "title": "Docs",
                        "raw_content": "Crawl body with competitor details.",
                    }
                ]
            }

        def extract(self, urls: list[str], **kwargs: object) -> dict[str, object]:
            return {
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "A",
                        "raw_content": "Extract body with pricing details.",
                    }
                ],
                "failed_results": [{"url": "https://example.com/pricing", "error": "empty"}],
            }

    results = [
        SearchResult(
            method="tavily",
            query="query",
            rank=1,
            title="A",
            url="https://example.com/a",
            snippet="Snippet",
        )
    ]
    monkeypatch.setattr(tavily_collection, "tavily_client", lambda: (FakeClient(), None))

    payload = tavily_collection.collect_tavily_pages(
        results,
        tmp_path,
        query="query",
        methods=["tavily"],
        tools=["search", "extract", "map", "crawl"],
        content_max_chars=12000,
        seed_urls=["https://seed.example/"],
        seed_results=1,
        map_limit=10,
        crawl_limit=5,
        max_depth=1,
        max_breadth=10,
        extract_limit=10,
        extract_depth="basic",
        tavily_format="markdown",
    )

    assert payload["map_added"] >= 1
    assert payload["crawl_pages"] >= 1
    assert payload["crawl_content_success"] >= 1
    assert payload["extract_success"] == 1
    assert payload["extract_failed"] == 1
    by_url = {result.url: result for result in results}
    assert by_url["https://example.com/a"].content_source == "tavily_extract"
    assert by_url["https://example.com/docs"].content_source == "tavily_crawl"
    assert (tmp_path / by_url["https://example.com/a"].content_path).exists()
    assert by_url["https://example.com/pricing"].discovery_type == "tavily_map"


def test_tavily_collection_uses_any_result_roots_as_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def map(self, url: str, **kwargs: object) -> dict[str, object]:
            return {"results": []}

        def crawl(self, url: str, **kwargs: object) -> dict[str, object]:
            return {"results": []}

        def extract(self, urls: list[str], **kwargs: object) -> dict[str, object]:
            return {"results": []}

    results = [
        SearchResult(
            method="duckduckgo_text",
            query="query",
            rank=1,
            title="A",
            url="https://example.com/a",
            snippet="Snippet",
        ),
        SearchResult(
            method="brave_web",
            query="query",
            rank=2,
            title="B",
            url="https://docs.example.org/path",
            snippet="Snippet",
        ),
    ]
    monkeypatch.setattr(tavily_collection, "tavily_client", lambda: (FakeClient(), None))

    payload = tavily_collection.collect_tavily_pages(
        results,
        tmp_path,
        query="query",
        methods=["duckduckgo_text", "tavily"],
        tools=["map"],
        content_max_chars=12000,
        seed_urls=[],
        seed_results=2,
        map_limit=10,
        crawl_limit=5,
        max_depth=1,
        max_breadth=10,
        extract_limit=10,
        extract_depth="basic",
        tavily_format="markdown",
    )

    assert payload["seed_urls"] == ["https://example.com/", "https://docs.example.org/"]
