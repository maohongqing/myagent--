from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from myagent.search.models import SearchResult, TextChunk
from myagent.search.chunking import source_result_id
from myagent.search.runner import SearchRunResult

from myagent.backend.app.agents import candidate_selector, search_runner_adapter
from myagent.backend.app.agents.candidate_selector import (
    LLMCandidateChunkOutput,
    LLMCandidatePoolOutput,
    LLMSelectedCandidate,
    LLMSourceRef,
    _candidate_pool_from_llm_output,
    generate_candidate_pool_from_chunks_with_llm_or_rules,
    is_invalid_candidate_name,
)
from myagent.backend.app.agents.search_runner_adapter import (
    collect_candidate_existence_sources_with_search_runner,
    collect_candidate_sources_with_search_runner,
)
from myagent.backend.app.config import Settings
from myagent.backend.app.models import AnalysisBenchmark, CreateTaskRequest, RawSource


def _benchmark() -> AnalysisBenchmark:
    return AnalysisBenchmark(
        target_product="Acme Workspace",
        raw_description="Collaboration product",
        lifecycle_stage="concept",
        analysis_purpose="decision_support",
        analysis_goal="Analyze competitors.",
    )


@pytest.mark.asyncio
async def test_search_runner_adapter_passes_methods_and_deep_backend(monkeypatch, tmp_path):
    calls = {}

    def fake_run_search(query, options):
        calls["query"] = query
        calls["methods"] = options.methods
        calls["deep_backend"] = options.deep_backend
        calls["max_results"] = options.max_results
        calls["browser_disable_proxy"] = options.browser_disable_proxy
        calls["chunk_size"] = options.chunk_size
        calls["chunk_overlap"] = options.chunk_overlap
        run_dir = tmp_path / "data" / "search_runs" / "run"
        run_dir.mkdir(parents=True)
        result = SearchResult(
            method="brave_web",
            query=query,
            rank=1,
            title="BetaSuite features",
            url="https://example.com/beta",
            snippet="BetaSuite competes in collaboration.",
            screenshot_path="screenshots/result.png",
        )
        chunk = TextChunk(
            chunk_id="chunk_0001",
            source_result_id=source_result_id(result),
            method="brave_web",
            rank=1,
            title="BetaSuite features",
            url="https://example.com/beta",
            content_path="contents/chunk.txt",
            start_char=0,
            end_char=48,
            text="BetaSuite competes in collaboration.",
            chunk_shot_path="chunk_shots/chunk.png",
            chunk_shot_paths=["chunk_shots/chunk.png"],
        )
        return SearchRunResult(
            query=query,
            run_dir=run_dir,
            methods=["brave_web"],
            results=[result],
            chunks=[chunk],
            search_page_screenshots=[],
            screenshots_mode="both",
            deep_backend=options.deep_backend,
            errors=[],
            json_path=run_dir / "results.json",
            report_path=run_dir / "report.md",
        )

    monkeypatch.setattr(search_runner_adapter, "run_search", fake_run_search)
    request = CreateTaskRequest(
        target_product="Acme Workspace",
        search_methods=["brave_web"],
        deep_search_backend="local_relevant_links",
        enable_screenshots=True,
    )
    sources, errors = await collect_candidate_sources_with_search_runner(
        ["BetaSuite competitors"],
        request,
        Settings(sqlite_path=tmp_path / "data" / "app.db", search_browser_disable_proxy=False),
        task_id="task_x",
    )

    assert errors == []
    assert calls["methods"] == ["brave_web"]
    assert calls["deep_backend"] == "local_relevant_links"
    assert calls["max_results"] == 5
    assert calls["browser_disable_proxy"] is False
    assert calls.get("chunk_size") == 3000
    assert calls.get("chunk_overlap") == 300
    assert sources[0].metadata["chunk_id"] == "chunk_0001"
    assert sources[0].metadata["image_path"] is None
    assert sources[0].metadata["chunk_shot_paths"] == []


@pytest.mark.asyncio
async def test_candidate_existence_search_can_request_six_results(monkeypatch, tmp_path):
    calls = {}

    def fake_run_search(query, options):
        calls["max_results"] = options.max_results
        run_dir = tmp_path / "data" / "search_runs" / "existence"
        run_dir.mkdir(parents=True)
        results = [
            SearchResult(
                method="duckduckgo_text",
                query=query,
                rank=index + 1,
                title=f"Result {index}",
                url=f"https://example.com/{index}",
                snippet="Result",
            )
            for index in range(6)
        ]
        return SearchRunResult(
            query=query,
            run_dir=run_dir,
            methods=["duckduckgo_text"],
            results=results,
            chunks=[],
            search_page_screenshots=[],
            screenshots_mode="none",
            deep_backend="none",
            errors=[],
            json_path=run_dir / "results.json",
            report_path=run_dir / "report.md",
        )

    monkeypatch.setattr(search_runner_adapter, "run_search", fake_run_search)

    sources, errors = await collect_candidate_existence_sources_with_search_runner(
        "BetaSuite",
        CreateTaskRequest(target_product="Acme Workspace"),
        Settings(sqlite_path=tmp_path / "data" / "app.db"),
        task_id="task_existence_six",
        max_sources=6,
    )

    assert errors == []
    assert calls["max_results"] == 6
    assert len(sources) == 6


@pytest.mark.asyncio
async def test_search_runner_adapter_splits_long_chunks_with_overlap(monkeypatch, tmp_path):
    long_text = "A" * 3000 + "B" * 300 + "C" * 2700 + "D" * 500

    def fake_run_search(query, options):
        run_dir = tmp_path / "data" / "search_runs" / "long_chunk"
        run_dir.mkdir(parents=True)
        result = SearchResult(
            method="duckduckgo_text",
            query=query,
            rank=1,
            title="Long source",
            url="https://example.com/long",
            snippet="Long source",
        )
        chunk = TextChunk(
            chunk_id="chunk_0001",
            source_result_id=source_result_id(result),
            method="duckduckgo_text",
            rank=1,
            title="Long source",
            url="https://example.com/long",
            content_path="contents/chunk.txt",
            start_char=0,
            end_char=len(long_text),
            text=long_text,
        )
        return SearchRunResult(
            query=query,
            run_dir=run_dir,
            methods=["duckduckgo_text"],
            results=[result],
            chunks=[chunk],
            search_page_screenshots=[],
            screenshots_mode="none",
            deep_backend="none",
            errors=[],
            json_path=run_dir / "results.json",
            report_path=run_dir / "report.md",
        )

    monkeypatch.setattr(search_runner_adapter, "run_search", fake_run_search)

    sources, errors = await collect_candidate_sources_with_search_runner(
        ["long source"],
        CreateTaskRequest(target_product="Acme Workspace"),
        Settings(sqlite_path=tmp_path / "data" / "app.db"),
        task_id="task_long_chunk",
    )

    assert errors == []
    assert [len(source.content) for source in sources] == [3000, 3000, 1100]
    assert sources[0].content[-300:] == sources[1].content[:300]
    assert sources[1].content[-300:] == sources[2].content[:300]
    assert [source.metadata["chunk_id"] for source in sources] == [
        "chunk_0001",
        "chunk_0001_part_02",
        "chunk_0001_part_03",
    ]


@pytest.mark.asyncio
async def test_search_runner_adapter_reuses_seen_url_keys(monkeypatch, tmp_path):
    seen_on_calls: list[set[str]] = []

    def fake_run_search(query, options):
        seen_on_calls.append(set(options.exclude_url_keys))
        run_dir = tmp_path / "data" / "search_runs" / query.replace(" ", "_")
        run_dir.mkdir(parents=True)
        result = SearchResult(
            method="duckduckgo_text",
            query=query,
            rank=1,
            title=query,
            url=f"https://example.com/{query[-1]}?utm_source=test",
            snippet="Result",
        )
        return SearchRunResult(
            query=query,
            run_dir=run_dir,
            methods=["duckduckgo_text"],
            results=[result],
            chunks=[],
            search_page_screenshots=[],
            screenshots_mode="none",
            deep_backend="none",
            errors=[],
            json_path=run_dir / "results.json",
            report_path=run_dir / "report.md",
            metadata={"seen_url_keys": [f"https://example.com/{query[-1]}"]},
        )

    monkeypatch.setattr(search_runner_adapter, "run_search", fake_run_search)
    request = CreateTaskRequest(target_product="Acme Workspace")
    seen: set[str] = set()

    await collect_candidate_sources_with_search_runner(
        ["query a", "query b"],
        request,
        Settings(sqlite_path=tmp_path / "data" / "app.db", search_query_concurrency=1),
        task_id="task_seen",
        seen_url_keys=seen,
    )

    assert seen_on_calls[0] == set()
    assert "https://example.com/a" in seen_on_calls[1]
    assert seen == {"https://example.com/a", "https://example.com/b"}


@pytest.mark.asyncio
async def test_candidate_search_executes_at_most_five_queries(monkeypatch, tmp_path):
    executed: list[str] = []

    def fake_run_search(query, options):
        executed.append(query)
        run_dir = tmp_path / "data" / "search_runs" / query.replace(" ", "_")
        run_dir.mkdir(parents=True)
        return SearchRunResult(
            query=query,
            run_dir=run_dir,
            methods=["duckduckgo_text"],
            results=[],
            chunks=[],
            search_page_screenshots=[],
            screenshots_mode="none",
            deep_backend="none",
            errors=[],
            json_path=run_dir / "results.json",
            report_path=run_dir / "report.md",
        )

    monkeypatch.setattr(search_runner_adapter, "run_search", fake_run_search)
    await collect_candidate_sources_with_search_runner(
        [f"query {index}" for index in range(12)],
        CreateTaskRequest(target_product="Acme Workspace"),
        Settings(sqlite_path=tmp_path / "data" / "app.db", max_queries_per_generation=5),
        task_id="task_limit",
    )

    assert executed == [f"query {index}" for index in range(5)]


def test_candidate_pool_preserves_source_refs_and_penalizes_missing_refs():
    output = LLMCandidatePoolOutput(
        candidates=[
            LLMSelectedCandidate(
                name="BetaSuite",
                competitor_type="head_direct",
                description="Collaboration suite.",
                selection_reason="Supported by source chunk.",
                confidence=0.9,
                source_refs=[
                    LLMSourceRef(
                        source_id="raw_1",
                        chunk_id="chunk_0001",
                        url="https://example.com/beta",
                        title="BetaSuite features",
                        quote="BetaSuite competes in collaboration.",
                        screenshot_path="search_runs/run/screenshots/result.png",
                        chunk_shot_paths=["search_runs/run/chunk_shots/chunk.png"],
                    )
                ],
            ),
            LLMSelectedCandidate(
                name="GammaFlow",
                competitor_type="challenger",
                description="Workflow product.",
                selection_reason="No strict source.",
                confidence=0.9,
            ),
        ]
    )

    candidates = _candidate_pool_from_llm_output(_benchmark(), output, 30)

    beta = next(item for item in candidates if item.name == "BetaSuite")
    gamma = next(item for item in candidates if item.name == "GammaFlow")
    assert beta.source_refs[0].chunk_id == "chunk_0001"
    assert beta.source_urls == ["https://example.com/beta"]
    assert "missing_source_refs" in gamma.tags
    assert gamma.score <= 0.35


def test_candidate_pool_filters_behavior_phrase_names():
    output = LLMCandidatePoolOutput(
        candidates=[
            LLMSelectedCandidate(
                name="骑自行车上班或选择更环保的产品",
                competitor_type="indirect_cross",
                description="A behavior phrase incorrectly extracted from an Ant Forest article.",
                selection_reason="The phrase appears in the source paragraph.",
                confidence=0.8,
                source_refs=[
                    LLMSourceRef(
                        source_id="raw_1",
                        chunk_id="chunk_0001",
                        url="https://www.bbc.com/zhongwen/articles/example",
                        title="蚂蚁森林新闻",
                        quote="玩家可以透过在现实世界中践行环保行为来获得积分，例如骑自行车上班或选择更环保的产品。",
                    )
                ],
            ),
            LLMSelectedCandidate(
                name="蚂蚁森林",
                competitor_type="head_direct",
                description="A branded green behavior gamification product.",
                selection_reason="The source names Ant Forest as a mobile app.",
                confidence=0.88,
                source_refs=[
                    LLMSourceRef(
                        source_id="raw_1",
                        chunk_id="chunk_0001",
                        url="https://www.bbc.com/zhongwen/articles/example",
                        title="蚂蚁森林新闻",
                        quote="蚂蚁森林是一款广受好评的中国行动应用程式。",
                    )
                ],
            ),
        ]
    )

    candidates = _candidate_pool_from_llm_output(_benchmark(), output, 30)

    assert is_invalid_candidate_name("骑自行车上班或选择更环保的产品", _benchmark())
    assert [item.name for item in candidates] == ["蚂蚁森林"]


@pytest.mark.asyncio
async def test_candidate_chunk_llm_limits_chunk_size_and_groups_logs(tmp_path):
    calls: list[str] = []

    class FakeStructured:
        async def ainvoke(self, messages):
            user = next(content for role, content in messages if role == "user")
            calls.append(user)
            return LLMCandidateChunkOutput(
                candidates=[
                    LLMSelectedCandidate(
                        name="BetaSuite",
                        competitor_type="head_direct",
                        description="Collaboration suite.",
                        selection_reason="Chunk evidence mentions BetaSuite.",
                        confidence=0.9,
                        source_refs=[
                            LLMSourceRef(
                                url="https://example.com/beta",
                                title="BetaSuite",
                                quote="BetaSuite competes with Acme Workspace.",
                            )
                        ],
                    )
                ]
            )

    class FakeModel:
        def with_structured_output(self, _schema):
            return FakeStructured()

    full_chunk = "BetaSuite competes with Acme Workspace. " + ("full chunk content " * 220)
    sources = [
        RawSource(
            id="raw_1",
            url="https://example.com/beta",
            title="BetaSuite",
            content=full_chunk,
            source_type="search",
            metadata={
                "source_result_id": "src_1",
                "chunk_id": "chunk_0001",
                "screenshot_path": "search_runs/run/screenshots/result.png",
                "chunk_shot_paths": ["search_runs/run/chunk_shots/chunk.png"],
            },
        ),
        RawSource(
            id="raw_2",
            url="https://example.com/beta",
            title="BetaSuite part 2",
            content="No extra competitor here.",
            source_type="search",
            metadata={"source_result_id": "src_1", "chunk_id": "chunk_0002"},
        ),
    ]

    candidates, notes, mode = await generate_candidate_pool_from_chunks_with_llm_or_rules(
        FakeModel(),
        _benchmark(),
        CreateTaskRequest(target_product="Acme Workspace"),
        sources,
        task_output_dir=tmp_path,
        task_id="task_chunk",
    )

    assert mode == "llm_candidate_chunk_pool"
    assert candidates[0].source_refs[0].chunk_id == "chunk_0001"
    assert full_chunk[:3000] in calls[0]
    assert len(json.loads(calls[0][calls[0].index("{") :])["source"]["content"]) <= 3000
    for forbidden in ["screenshot_path", "chunk_shot_paths", "run_dir", "provider", "query", "method", "rank", "content_chars"]:
        assert forbidden not in calls[0]
    log_path = tmp_path / "task_chunk" / "llm_calls" / "search_agent" / "candidate_chunk_generation" / "src_1.json"
    payload = __import__("json").loads(log_path.read_text(encoding="utf-8"))
    assert payload["system"]
    assert len(payload["calls"]) == 1
    assert payload["calls"][0]["chunk_id"] == "chunk_0001"
    assert any("chunk filter kept 1/2" in note for note in notes)
    md_text = log_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "搜索阶段大模型输出" in md_text
    assert "网页正文" in md_text
    assert "\\n" not in md_text
    assert "screenshot_path" not in md_text


@pytest.mark.asyncio
async def test_candidate_chunk_pool_trims_more_than_thirty_chunk_candidates(monkeypatch):
    async def fake_retry_structured_ainvoke(_model, _schema, messages, *_args, **_kwargs):
        user = next(content for role, content in messages if role == "user")
        payload = json.loads(user[user.index("{") :])
        call_index = int(payload["call_index"])
        return LLMCandidateChunkOutput(
            candidates=[
                LLMSelectedCandidate(
                    name=f"Rival{call_index:02d}{item:02d}",
                    competitor_type="challenger",
                    description="Collaboration product.",
                    selection_reason="The source names this product as a competitor.",
                    confidence=0.82,
                    source_refs=[
                        LLMSourceRef(
                            quote=f"Rival{call_index:02d}{item:02d} competes with Acme Workspace.",
                        )
                    ],
                )
                for item in range(8)
            ]
        )

    monkeypatch.setattr(candidate_selector, "retry_structured_ainvoke", fake_retry_structured_ainvoke)
    sources = [
        RawSource(
            id=f"raw_{index}",
            url=f"https://example.com/source-{index}",
            title=f"Source {index}",
            content=f"Rival{index:02d}00 competes with Acme Workspace.",
            source_type="search",
            metadata={"chunk_id": f"chunk_{index:04d}"},
        )
        for index in range(12)
    ]

    candidates, _notes, mode = await generate_candidate_pool_from_chunks_with_llm_or_rules(
        object(),
        _benchmark(),
        CreateTaskRequest(target_product="Acme Workspace"),
        sources,
        max_candidates=30,
    )

    assert mode == "llm_candidate_chunk_pool"
    assert len(candidates) == 30
    assert candidates[0].name == "Rival0100"


@pytest.mark.asyncio
async def test_candidate_chunk_prompt_includes_existing_candidate_names(monkeypatch):
    captured_messages: list[list[tuple[str, str]]] = []

    async def fake_retry_structured_ainvoke(_model, _schema, messages, *_args, **_kwargs):
        captured_messages.append(messages)
        return LLMCandidateChunkOutput(candidates=[])

    monkeypatch.setattr(candidate_selector, "retry_structured_ainvoke", fake_retry_structured_ainvoke)

    await generate_candidate_pool_from_chunks_with_llm_or_rules(
        object(),
        _benchmark(),
        CreateTaskRequest(target_product="Acme Workspace", competitors=["Bilibili"]),
        [
            RawSource(
                id="raw_prompt",
                url="https://example.com/beta",
                title="BetaSuite",
                content="BetaSuite competes with Acme Workspace.",
                source_type="search",
            )
        ],
        existing_candidate_names=["Bilibili", "Slack"],
    )

    assert captured_messages
    system_prompt = next(content for role, content in captured_messages[0] if role == "system")
    user = next(content for role, content in captured_messages[0] if role == "user")
    payload = json.loads(user[user.index("{") :])
    assert "existing_candidate_names" in payload
    assert payload["existing_candidate_names"] == ["Bilibili", "Slack"]
    assert "Bilibili" in system_prompt


@pytest.mark.asyncio
async def test_candidate_chunk_empty_llm_output_does_not_use_rule_fallback(monkeypatch):
    async def fake_retry_structured_ainvoke(_model, _schema, _messages, *_args, **_kwargs):
        return LLMCandidateChunkOutput(candidates=[], generation_notes=["No explicit competitor in this chunk."])

    monkeypatch.setattr(candidate_selector, "retry_structured_ainvoke", fake_retry_structured_ainvoke)
    candidates, notes, mode = await generate_candidate_pool_from_chunks_with_llm_or_rules(
        object(),
        _benchmark(),
        CreateTaskRequest(target_product="Acme Workspace"),
        [
            RawSource(
                id="raw_empty",
                url="https://example.com/beta",
                title="BetaSuite - collaboration software",
                content="BetaSuite competes with Acme Workspace, but the test LLM intentionally returns no candidates.",
                source_type="search",
            )
        ],
    )

    assert candidates == []
    assert mode == "empty_chunk_llm"
    assert any("未启用规则兜底" in note for note in notes)


@pytest.mark.asyncio
async def test_candidate_chunk_llm_error_uses_rule_fallback(monkeypatch):
    async def failing_retry_structured_ainvoke(_model, _schema, _messages, *_args, **_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(candidate_selector, "retry_structured_ainvoke", failing_retry_structured_ainvoke)
    candidates, notes, mode = await generate_candidate_pool_from_chunks_with_llm_or_rules(
        object(),
        _benchmark(),
        CreateTaskRequest(target_product="Acme Workspace"),
        [
            RawSource(
                id="raw_error",
                url="https://example.com/beta",
                title="BetaSuite - collaboration software",
                content="BetaSuite competes with Acme Workspace and offers messaging, docs, and workflows.",
                source_type="search",
            )
        ],
    )

    assert mode == "rule_fallback_llm_error"
    assert candidates
    assert any("大模型调用失败" in note for note in notes)


@pytest.mark.asyncio
async def test_candidate_chunk_quote_shot_paths_are_attached_after_llm(monkeypatch, tmp_path):
    class FakeStructured:
        async def ainvoke(self, _messages):
            return LLMCandidateChunkOutput(
                candidates=[
                    LLMSelectedCandidate(
                        name="BetaSuite",
                        competitor_type="head_direct",
                        description="Collaboration suite.",
                        selection_reason="Chunk evidence mentions BetaSuite.",
                        confidence=0.9,
                        source_refs=[
                            LLMSourceRef(
                                quote="BetaSuite competes with Acme Workspace.",
                            )
                        ],
                    )
                ]
            )

    class FakeModel:
        def with_structured_output(self, _schema):
            return FakeStructured()

    def fake_capture(extraction, run_dir, *, min_score):
        try:
            asyncio.get_running_loop()
            calls_running_loop = True
        except RuntimeError:
            calls_running_loop = False
        fake_capture.ran_inside_asyncio_loop = calls_running_loop
        assert run_dir == tmp_path / "data" / "search_runs" / "task_quote" / "candidate_discovery" / "run_1"
        assert min_score == 0.72
        return {fact["id"]: {"path": "quote_shots/ref.png", "error": None} for fact in extraction["facts"]}

    def fake_apply(extraction, quote_map):
        for fact in extraction["facts"]:
            shot = quote_map[fact["id"]]
            fact["quote_shot_path"] = shot["path"]
            fact["quote_shot_error"] = shot["error"]

    monkeypatch.setattr(candidate_selector, "capture_quote_shots_for_extraction", fake_capture)
    monkeypatch.setattr(candidate_selector, "apply_quote_shots_to_extraction", fake_apply)

    run_dir = tmp_path / "data" / "search_runs" / "task_quote" / "candidate_discovery" / "run_1"
    run_dir.mkdir(parents=True)
    source = RawSource(
        id="raw_1",
        url="https://example.com/beta",
        title="BetaSuite",
        content="BetaSuite competes with Acme Workspace.",
        source_type="search",
        metadata={
            "source_result_id": "src_1",
            "chunk_id": "chunk_0001",
            "run_dir": "search_runs/task_quote/candidate_discovery/run_1",
        },
    )

    candidates, _notes, mode = await generate_candidate_pool_from_chunks_with_llm_or_rules(
        FakeModel(),
        _benchmark(),
        CreateTaskRequest(target_product="Acme Workspace"),
        [source],
        task_output_dir=tmp_path / "data" / "task_runs",
        task_id="task_quote",
    )

    assert mode == "llm_candidate_chunk_pool"
    ref = candidates[0].source_refs[0]
    assert ref.screenshot_path is None
    assert ref.chunk_shot_paths == []
    assert ref.evidence_screenshot_path == "search_runs/task_quote/candidate_discovery/run_1/quote_shots/ref.png"
    assert (run_dir / "extractions" / "candidate_quote_refs.md").exists()
    assert fake_capture.ran_inside_asyncio_loop is False
