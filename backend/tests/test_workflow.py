import asyncio
import json

import pytest

from myagent.backend.app.agents import search_runner_adapter, workflow as workflow_module
from myagent.backend.app.agents.workflow import AgentWorkflow
from myagent.backend.app.config import Settings
from myagent.backend.app.events import EventBus
from myagent.backend.app.agents.state_tree import selected_set_to_phase_2
from myagent.backend.app.agents.strategy import score_and_select_candidates
from myagent.backend.app.models import AnalysisBenchmark, AnalysisTask, CandidateCompetitor, CreateTaskRequest, RankFactors, RawSource, SourceProviderResult, SourceRef
from myagent.backend.app.storage import JSONStorage


def test_phase_1_display_uses_structured_analysis_intent():
    markdown = workflow_module._display_phase_1(
        {
            "target_product": "二次元手办交易平台",
            "lifecycle_stage": "concept",
            "analysis_purpose": "decision_support",
            "analysis_goal": "决策支持",
            "confidence": 0.7,
        },
        {
            "target_product": "二次元手办交易平台",
            "goal": "评估当前二次元手办交易赛道的剩余市场空间",
            "problems": ["不清楚市场饱和度", "无法确定差异化切入点"],
            "confidence": 0.95,
        },
    )

    assert "分析目标：评估当前二次元手办交易赛道的剩余市场空间" in markdown
    assert "核心问题：不清楚市场饱和度；无法确定差异化切入点" in markdown
    assert "核心问题：决策支持" not in markdown
    assert "置信度：0.95" in markdown


@pytest.mark.asyncio
async def test_workflow_runs_without_external_keys(monkeypatch, tmp_path):
    async def fake_candidate_sources(queries, request, settings, *, task_id, max_sources=50, seen_url_keys=None, should_cancel=None, on_sources=None):
        if seen_url_keys is not None:
            seen_url_keys.add("https://example.com/candidate")
        return [
            RawSource(
                url=f"https://example.com/{abs(hash(queries[0]))}",
                title="DingTalk competitor result",
                content="DingTalk is a collaboration product with messaging, docs, workflows, pricing and enterprise channels.",
                source_type="search",
                metadata={"provider": "search_runner", "query": queries[0], "chunk_id": "chunk_0001"},
            )
        ], []

    async def fake_evidence(search_plan, request, settings, *, task_id, benchmark, competitors, max_sources=80, seen_url_keys=None, should_cancel=None):
        if seen_url_keys is not None:
            seen_url_keys.add("https://example.com/evidence")
        sources = [
            RawSource(
                url="https://example.com/dingtalk/pricing",
                title="DingTalk pricing",
                content="DingTalk offers enterprise collaboration, workflow automation, docs, and pricing evidence.",
                source_type="search",
                metadata={"provider": "search_runner", "query": search_plan.queries[0].query, "competitor": competitors[0].name},
            )
        ]
        return SourceProviderResult(provider="search", sources=sources, signals=[], cache_hit=False)

    monkeypatch.setattr(workflow_module, "collect_candidate_sources_with_search_runner", fake_candidate_sources)
    monkeypatch.setattr(search_runner_adapter, "collect_evidence_with_search_runner", fake_evidence)
    monkeypatch.setattr(workflow_module, "collect_evidence_with_search_runner", fake_evidence)
    settings = Settings(
        openai_api_key=None,
        tavily_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        max_revision_loops=1,
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    workflow = AgentWorkflow(settings, storage, EventBus(storage))

    report = await workflow.run(
        "task_test",
        CreateTaskRequest(target_product="飞书", competitors=["钉钉"], urls=[], llm_failure_policy="continue_without_llm"),
        max_revision_loops=1,
    )

    assert report.task_id == "task_test"
    assert report.evidence
    assert report.qa_history
    assert report.competitor_detail_cards_clean
    assert report.industry_card_clean
    assert report.report_references
    assert report.tool_results
    assert storage.get_report("task_test") is not None

    events = storage.list_events("task_test")
    perception_event = next(event for event in events if event.node == "plan_agent" and event.status == "completed")
    source_event = next(event for event in events if event.node == "source_collector_agent" and event.status == "completed")
    tool_event = next(event for event in events if event.node == "tool_executor_agent" and event.status == "completed")

    assert perception_event.details["kind"] == "state_tree_phase_1"
    assert perception_event.details["benchmark"]["analysis_purpose"]
    assert not any(event.node == "search_planner_agent" and event.status == "completed" for event in events)
    assert source_event.details["kind"] == "source_collection"
    assert source_event.details["sources"]["items"]
    assert source_event.details["competitor_detail_cards_clean"]["items"]
    assert source_event.details["industry_card_clean"]
    display_markdown = source_event.details.get("display_markdown", "")
    assert "本阶段采集完成" in display_markdown
    assert "pricing" in display_markdown
    assert tool_event.details["kind"] == "tool_execution"
    assert tool_event.details["results"]["items"]

    task_output_dir = settings.task_output_dir / "task_test"
    assert (task_output_dir / "steps").exists()
    assert list((task_output_dir / "steps").glob("*_event.json"))
    assert (task_output_dir / "final" / "report.json").exists()
    assert (task_output_dir / "final" / "report.md").exists()


@pytest.mark.asyncio
async def test_workflow_resumes_phase_4_from_tool_executor_checkpoint(monkeypatch, tmp_path):
    from conftest import make_report

    settings = Settings(
        openai_api_key=None,
        tavily_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        max_revision_loops=1,
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    request = CreateTaskRequest(target_product="Acme", competitors=["BetaSuite"], llm_failure_policy="continue_without_llm")
    fixture_report = make_report("task_resume_tool_executor", request)
    state = {
        "task_id": "task_resume_tool_executor",
        "request": request.model_dump(mode="json"),
        "revision_count": 0,
        "max_revision_loops": 1,
        "qa_history": [],
        "state_tree": workflow_module.create_initial_state_tree(request),
        "workflow_phase": "phase_4_synthesis",
        "phase_4_synthesis_step": "analyst_agent",
        "benchmark": fixture_report.analysis_benchmark.model_dump(mode="json"),
        "selected_competitor_set": fixture_report.selected_competitor_set.model_dump(mode="json"),
        "tool_plan": fixture_report.tool_plan.model_dump(mode="json"),
        "collection_plan": fixture_report.collection_plan.model_dump(mode="json"),
        "search_plan": fixture_report.search_plan.model_dump(mode="json"),
        "raw_sources": [
            RawSource(
                url="https://betasuite.example/pricing",
                title="BetaSuite pricing",
                content="BetaSuite offers a free plan and paid business subscriptions.",
                source_type="search",
                metadata={"competitor": "BetaSuite"},
            ).model_dump(mode="json")
        ],
        "evidence": [item.model_dump(mode="json") for item in fixture_report.evidence],
        "source_provider_results": [item.model_dump(mode="json") for item in fixture_report.source_provider_results],
        "tool_results": [item.model_dump(mode="json") for item in fixture_report.tool_results],
        "tool_sandboxes": [item.model_dump(mode="json") for item in fixture_report.tool_sandboxes],
    }
    workflow._save_checkpoint("task_resume_tool_executor", state, "analyst_agent")

    async def unexpected_tool_executor(_state):
        raise AssertionError("tool_executor_agent should not rerun after analyst checkpoint")

    monkeypatch.setattr(workflow, "tool_executor_agent", unexpected_tool_executor)

    report = await workflow.run("task_resume_tool_executor", request, max_revision_loops=1)

    assert report.task_id == "task_resume_tool_executor"
    assert storage.get_report("task_resume_tool_executor") is not None
    nodes = [event.node for event in storage.list_events("task_resume_tool_executor")]
    assert "analyst_agent" in nodes
    assert "structurer_agent" not in nodes
    assert "tool_executor_agent" not in nodes


@pytest.mark.asyncio
async def test_candidate_search_resume_reuses_checkpoint_sources_without_search_runner(monkeypatch, tmp_path):
    async def unexpected_candidate_sources(*_args, **_kwargs):
        raise AssertionError("candidate search runner should not rerun after search_complete checkpoint")

    captured: dict[str, object] = {}

    async def fake_candidate_pool(_model, _benchmark, _request, sources, **kwargs):
        captured["sources"] = list(sources)
        captured["existing_candidate_names"] = list(kwargs.get("existing_candidate_names") or [])
        return (
            [
                CandidateCompetitor(
                    name="ResumeRival",
                    description="Recovered from checkpoint sources.",
                    source_urls=["https://example.com/resume"],
                    tags=["llm_candidate_pool"],
                    score=0.8,
                    selection_reason="Recovered chunk analysis.",
                )
            ],
            ["resumed from checkpoint"],
            "llm_candidate_chunk_pool",
        )

    async def fake_candidate_qa(_task_id, _request, candidates):
        return candidates, [], []

    monkeypatch.setattr(workflow_module, "collect_candidate_sources_with_search_runner", unexpected_candidate_sources)
    monkeypatch.setattr(workflow_module, "generate_candidate_pool_from_chunks_with_llm_or_rules", fake_candidate_pool)

    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    monkeypatch.setattr(workflow, "_qa_verify_candidate_existence", fake_candidate_qa)

    request = CreateTaskRequest(target_product="Acme Workspace")
    benchmark = AnalysisBenchmark(
        target_product="Acme Workspace",
        raw_description="Collaboration product",
        lifecycle_stage="concept",
        analysis_purpose="decision_support",
        analysis_goal="Analyze competitors.",
    )
    saved_source = RawSource(
        id="raw_saved",
        url="https://example.com/resume",
        title="Resume source",
        content="ResumeRival competes with Acme Workspace.",
        source_type="search",
        metadata={"chunk_id": "chunk_saved"},
    )
    previous_candidate = CandidateCompetitor(
        name="AlreadyFound",
        description="Already extracted before the failure.",
        score=0.7,
        selection_reason="Previous chunk result.",
    )
    state = {
        "task_id": "task_resume_candidate_search",
        "request": request.model_dump(mode="json"),
        "benchmark": benchmark.model_dump(mode="json"),
        "workflow_phase": "phase_2_search",
        "candidate_search_stage": "search_complete",
        "candidate_active_queries": ["Acme Workspace competitors"],
        "candidate_query_history": ["Acme Workspace competitors"],
        "candidate_sources": [saved_source.model_dump(mode="json")],
        "candidates": [previous_candidate.model_dump(mode="json")],
        "searched_url_keys": ["https://example.com/resume"],
        "candidate_search_errors": [],
    }

    result = await workflow._search_phase_2_candidates(state)

    assert captured["sources"] == [saved_source]
    assert captured["existing_candidate_names"] == ["AlreadyFound"]
    assert [item["name"] for item in result["candidates"]] == ["AlreadyFound", "ResumeRival"]
    assert result["workflow_phase"] == "phase_2_analyze"


@pytest.mark.asyncio
async def test_candidate_existence_qa_removes_nonexistent_and_tags_verified(monkeypatch, tmp_path):
    async def fake_existence_sources(candidate_name, request, settings, *, task_id, max_sources=3, should_cancel=None):
        return [
            RawSource(
                url=f"https://example.com/{candidate_name.lower()}",
                title=f"{candidate_name} official",
                content=f"{candidate_name} is a real collaboration product.",
                source_type="search",
            )
        ], []

    monkeypatch.setattr(workflow_module, "collect_candidate_existence_sources_with_search_runner", fake_existence_sources)

    class FakeWorkflowModel:
        pass

    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"))
    task.id = "task_candidate_qa"
    storage.save_task(task)
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    workflow.model = FakeWorkflowModel()

    async def fake_llm_structured(task_id, node, schema, messages, *, llm_lane="main"):
        payload = messages[-1][1]
        return schema(
            exists="Ghost" not in payload,
            confidence=0.91,
            reason="Confirmed in search results." if "Ghost" not in payload else "No real product found.",
            matched_urls=["https://example.com/betasuite"],
        )

    monkeypatch.setattr(workflow, "_llm_structured", fake_llm_structured)
    candidates = [
        CandidateCompetitor(name="BetaSuite", description="Real product."),
        CandidateCompetitor(name="GhostThing", description="Bad candidate."),
    ]

    verified, notes, qa_sources = await workflow._qa_verify_candidate_existence(
        "task_candidate_qa",
        task.request,
        candidates,
    )

    assert [item.name for item in verified] == ["BetaSuite"]
    assert "qa_verified" in verified[0].tags
    assert verified[0].source_refs
    assert qa_sources
    assert any("GhostThing" in note for note in notes)


@pytest.mark.asyncio
async def test_candidate_existence_qa_merges_known_aliases_when_reason_only(monkeypatch, tmp_path):
    async def fake_existence_sources(candidate_name, request, settings, *, task_id, max_sources=3, should_cancel=None):
        return [
            RawSource(
                url=f"https://example.com/{candidate_name}",
                title=f"{candidate_name} official",
                content=f"{candidate_name} is a real video platform.",
                source_type="search",
            )
        ], []

    monkeypatch.setattr(workflow_module, "collect_candidate_existence_sources_with_search_runner", fake_existence_sources)

    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme Video"))
    task.id = "task_candidate_alias_qa"
    storage.save_task(task)
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    workflow.model = object()

    async def fake_llm_structured(task_id, node, schema, messages, *, llm_lane="main"):
        return schema(
            exists=True,
            confidence=0.9,
            reason="与已有 B站 为同一实体，但这里只写在自然语言原因里。",
            matched_urls=[],
            merge_target=None,
            canonical_name=None,
        )

    monkeypatch.setattr(workflow, "_llm_structured", fake_llm_structured)
    candidates = [
        CandidateCompetitor(name="B站", description="Short public alias."),
        CandidateCompetitor(name="哔哩哔哩（B站）", description="Chinese name with alias."),
        CandidateCompetitor(name="Bilibili", description="English public name."),
    ]

    verified, notes, qa_sources = await workflow._qa_verify_candidate_existence(
        "task_candidate_alias_qa",
        task.request,
        candidates,
    )

    assert [item.name for item in verified] == ["B站"]
    assert "alias_canonicalized" in verified[0].tags
    assert len(verified[0].source_refs) == 3
    assert workflow._last_candidate_qa_change_log.merged_candidates
    assert not notes
    assert len(qa_sources) == 3


@pytest.mark.asyncio
async def test_candidate_existence_qa_merges_bidirectional_app_alias(monkeypatch, tmp_path):
    async def fake_existence_sources(candidate_name, request, settings, *, task_id, max_sources=3, should_cancel=None):
        return [
            RawSource(
                url=f"https://example.com/{candidate_name}",
                title=f"{candidate_name} official",
                content=f"{candidate_name} is a real resale platform.",
                source_type="search",
            )
        ], []

    monkeypatch.setattr(workflow_module, "collect_candidate_existence_sources_with_search_runner", fake_existence_sources)

    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Figure resale"))
    task.id = "task_candidate_xianyu_alias_qa"
    storage.save_task(task)
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    workflow.model = object()

    async def fake_llm_structured(task_id, node, schema, messages, *, llm_lane="main"):
        payload = messages[-1][1]
        if '"name": "闲鱼App"' in payload:
            return schema(
                exists=True,
                confidence=0.95,
                reason="Same product as existing 闲鱼.",
                matched_urls=[],
                merge_target="闲鱼",
                canonical_name="闲鱼",
            )
        return schema(
            exists=True,
            confidence=0.98,
            reason="Same product as existing 闲鱼App.",
            matched_urls=[],
            merge_target="闲鱼App",
            canonical_name="闲鱼App",
        )

    monkeypatch.setattr(workflow, "_llm_structured", fake_llm_structured)
    candidates = [
        CandidateCompetitor(name="闲鱼", description="Alibaba resale platform.", score=0.98),
        CandidateCompetitor(name="闲鱼App", description="Alibaba resale app.", score=0.95),
    ]

    verified, notes, qa_sources = await workflow._qa_verify_candidate_existence(
        "task_candidate_xianyu_alias_qa",
        task.request,
        candidates,
    )

    assert [item.name for item in verified] == ["闲鱼"]
    assert len(verified[0].source_refs) == 2
    assert "qa_merged_alias" in verified[0].tags
    assert any(
        item["from_name"] == "闲鱼App" and item["to_name"] == "闲鱼"
        for item in workflow._last_candidate_qa_change_log.merged_candidates
    )
    assert not notes
    assert len(qa_sources) == 2


def test_merge_candidates_strips_generic_app_suffix():
    change_log = workflow_module.CandidateQAChangeLog()
    candidates = [
        CandidateCompetitor(name="闲鱼", description="Alibaba resale platform."),
        CandidateCompetitor(name="闲鱼App", description="Alibaba resale app."),
    ]

    merged = workflow_module._apply_candidate_alias_decisions(candidates, {}, change_log)

    assert [item.name for item in merged] == ["闲鱼"]
    assert "alias_canonicalized" in merged[0].tags
    assert any(
        item["from_name"] == "闲鱼App" and item["to_name"] == "闲鱼"
        for item in change_log.merged_candidates
    )


@pytest.mark.asyncio
async def test_candidate_existence_qa_removes_candidate_when_llm_fails(monkeypatch, tmp_path):
    async def fake_existence_sources(candidate_name, request, settings, *, task_id, max_sources=3, should_cancel=None):
        return [
            RawSource(
                url=f"https://example.com/{candidate_name.lower()}",
                title=f"{candidate_name} official",
                content=f"{candidate_name} is a real product.",
                source_type="search",
            )
        ], []

    monkeypatch.setattr(workflow_module, "collect_candidate_existence_sources_with_search_runner", fake_existence_sources)

    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"))
    task.id = "task_candidate_qa_fail"
    storage.save_task(task)
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    workflow.model = object()

    async def failing_llm_structured(*_args, **_kwargs):
        raise RuntimeError("qa unavailable")

    monkeypatch.setattr(workflow, "_llm_structured", failing_llm_structured)
    candidates = [CandidateCompetitor(name="BetaSuite", description="Real product.")]

    verified, notes, _qa_sources = await workflow._qa_verify_candidate_existence(
        "task_candidate_qa_fail",
        task.request,
        candidates,
    )

    assert verified == []
    assert any("BetaSuite: Removed because existence QA failed after retries" in note for note in notes)
    assert workflow._last_candidate_qa_change_log.removed_candidates
    assert not workflow._last_candidate_qa_change_log.unverified_candidates


@pytest.mark.asyncio
async def test_candidate_existence_qa_removes_candidate_when_search_has_no_usable_content(monkeypatch, tmp_path):
    async def fake_existence_sources(candidate_name, request, settings, *, task_id, max_sources=3, should_cancel=None):
        return [
            RawSource(
                url=f"https://example.com/{candidate_name.lower()}/{index}",
                title=f"{candidate_name} result {index}",
                content="   ",
                source_type="search",
            )
            for index in range(6)
        ], []

    monkeypatch.setattr(workflow_module, "collect_candidate_existence_sources_with_search_runner", fake_existence_sources)

    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"))
    task.id = "task_candidate_no_content"
    storage.save_task(task)
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    workflow.model = object()
    llm_called = False

    async def fake_llm_structured(*_args, **_kwargs):
        nonlocal llm_called
        llm_called = True
        raise AssertionError("LLM should not run without usable existence sources.")

    monkeypatch.setattr(workflow, "_llm_structured", fake_llm_structured)

    verified, notes, qa_sources = await workflow._qa_verify_candidate_existence(
        "task_candidate_no_content",
        task.request,
        [CandidateCompetitor(name="BetaSuite", description="Candidate.")],
    )

    assert verified == []
    assert qa_sources
    assert not llm_called
    assert any("could not fetch usable page content from 6 results" in note for note in notes)
    assert workflow._last_candidate_qa_change_log.removed_candidates[0]["source"] == "search"


@pytest.mark.asyncio
async def test_candidate_existence_qa_uses_later_usable_sources_and_limits_llm_input(monkeypatch, tmp_path):
    captured_web_results: list[dict] = []
    captured_max_sources: list[int] = []

    async def fake_existence_sources(candidate_name, request, settings, *, task_id, max_sources=3, should_cancel=None):
        captured_max_sources.append(max_sources)
        return [
            *[
                RawSource(
                    url=f"https://example.com/empty/{index}",
                    title=f"Empty {index}",
                    content="",
                    source_type="search",
                )
                for index in range(3)
            ],
            *[
                RawSource(
                    url=f"https://example.com/usable/{index}",
                    title=f"Usable {index}",
                    content=f"Usable evidence {index} for {candidate_name}.",
                    source_type="search",
                )
                for index in range(3)
            ],
        ], []

    monkeypatch.setattr(workflow_module, "collect_candidate_existence_sources_with_search_runner", fake_existence_sources)

    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"))
    task.id = "task_candidate_later_sources"
    storage.save_task(task)
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    workflow.model = object()

    async def fake_llm_structured(task_id, node, schema, messages, *, llm_lane="main"):
        payload = json.loads(messages[-1][1])
        captured_web_results.extend(payload["web_results"])
        return schema(exists=True, confidence=0.9, reason="Confirmed.", matched_urls=[])

    monkeypatch.setattr(workflow, "_llm_structured", fake_llm_structured)

    verified, notes, qa_sources = await workflow._qa_verify_candidate_existence(
        "task_candidate_later_sources",
        task.request,
        [CandidateCompetitor(name="BetaSuite", description="Candidate.")],
    )

    assert captured_max_sources == [6]
    assert [item.name for item in verified] == ["BetaSuite"]
    assert notes == []
    assert len(captured_web_results) == 3
    assert all("Usable evidence" in item["content_excerpt"] for item in captured_web_results)
    assert len(qa_sources) == 3


@pytest.mark.asyncio
async def test_candidate_existence_qa_continues_when_search_errors_have_usable_sources(monkeypatch, tmp_path):
    llm_called = False

    async def fake_existence_sources(candidate_name, request, settings, *, task_id, max_sources=3, should_cancel=None):
        return [
            RawSource(
                url="https://example.com/beta",
                title="BetaSuite official",
                content="BetaSuite is a real product.",
                source_type="search",
            )
        ], [{"error": "fetch failed"}, {"error": "blocked"}]

    monkeypatch.setattr(workflow_module, "collect_candidate_existence_sources_with_search_runner", fake_existence_sources)

    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"))
    task.id = "task_candidate_errors_with_sources"
    storage.save_task(task)
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    workflow.model = object()

    async def fake_llm_structured(task_id, node, schema, messages, *, llm_lane="main"):
        nonlocal llm_called
        llm_called = True
        return schema(exists=True, confidence=0.9, reason="Confirmed.", matched_urls=[])

    monkeypatch.setattr(workflow, "_llm_structured", fake_llm_structured)

    verified, notes, _qa_sources = await workflow._qa_verify_candidate_existence(
        "task_candidate_errors_with_sources",
        task.request,
        [CandidateCompetitor(name="BetaSuite", description="Candidate.")],
    )

    assert llm_called
    assert [item.name for item in verified] == ["BetaSuite"]
    assert any("existence search returned 2 error(s)" in note for note in notes)


@pytest.mark.asyncio
async def test_candidate_existence_qa_filters_behavior_phrase_without_llm(tmp_path):
    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(
        request=CreateTaskRequest(
            target_product="蚂蚁森林",
            llm_failure_policy="continue_without_llm",
        )
    )
    task.id = "task_candidate_qa_rule_filter"
    storage.save_task(task)
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    candidates = [
        CandidateCompetitor(name="骑自行车上班或选择更环保的产品", description="Behavior phrase."),
        CandidateCompetitor(name="GForest", description="A branded green gamification platform."),
    ]

    verified, notes, qa_sources = await workflow._qa_verify_candidate_existence(
        "task_candidate_qa_rule_filter",
        task.request,
        candidates,
    )

    assert [item.name for item in verified] == ["GForest"]
    assert "qa_unverified" not in verified[0].tags
    assert qa_sources == []
    assert any("骑自行车上班或选择更环保的产品" in note for note in notes)


def test_select_top_competitors_by_category_skips_zero_score_candidates():
    benchmark = AnalysisBenchmark(
        target_product="蚂蚁森林",
        raw_description="Green behavior gamification app",
        lifecycle_stage="launched",
        analysis_purpose="decision_support",
        analysis_goal="Select relevant competitors.",
    )
    candidates = [
        CandidateCompetitor(
            name="骑自行车上班或选择更环保的产品",
            description="Behavior phrase.",
            competitor_type="indirect_cross",
        ),
        CandidateCompetitor(
            name="LowSignal",
            description="No evidence or relevance signal.",
            competitor_type="potential_substitute",
        ),
        CandidateCompetitor(
            name="GForest",
            description="A branded green gamification platform.",
            competitor_type="challenger",
            rank_factors=RankFactors(heat=0.8, growth=0.7, similarity=0.75),
        ),
    ]

    selected_set = workflow_module._select_top_competitors_by_category(benchmark, candidates)

    assert [item.name for item in selected_set.selected] == ["GForest"]


def _selection_candidate(name: str, competitor_type: str, score: float = 0.8) -> CandidateCompetitor:
    ref = SourceRef(url=f"https://example.com/{name}", title=f"{name} source", quote=f"{name} evidence")
    return CandidateCompetitor(
        name=name,
        description=f"{name} competitor profile.",
        competitor_type=competitor_type,
        source_refs=[ref],
        source_urls=[ref.url],
        source_titles=[ref.title],
        rank_factors=RankFactors(heat=score, growth=score, similarity=score, risk=1.0),
    )


@pytest.mark.parametrize(
    ("purpose", "expected_types"),
    [
        ("decision_support", ["head_direct", "challenger", "indirect_cross"]),
        ("learning", ["head_direct", "indirect_cross", "indirect_cross"]),
        ("market_warning", ["head_direct", "indirect_cross", "potential_substitute"]),
    ],
)
def test_select_top_competitors_by_category_uses_three_competitor_rules(purpose, expected_types):
    benchmark = AnalysisBenchmark(
        target_product="Acme",
        raw_description="Acme collaboration product",
        lifecycle_stage="launched",
        analysis_purpose=purpose,
        analysis_goal="Select competitors.",
    )
    candidates = [
        _selection_candidate("DirectA", "head_direct", 0.9),
        _selection_candidate("DirectB", "challenger", 0.88),
        _selection_candidate("IndirectA", "indirect_cross", 0.86),
        _selection_candidate("IndirectB", "indirect_cross", 0.84),
        _selection_candidate("SubstituteA", "potential_substitute", 0.82),
    ]

    selected_set = workflow_module._select_top_competitors_by_category(benchmark, candidates)

    assert selected_set.total_budget == 3
    assert [item.competitor_type for item in selected_set.selected] == expected_types


def test_phase2_score_ignores_risk_factor():
    benchmark = AnalysisBenchmark(
        target_product="Acme",
        raw_description="Acme collaboration product",
        lifecycle_stage="launched",
        analysis_purpose="decision_support",
        analysis_goal="Select competitors.",
    )
    low_risk = _selection_candidate("LowRisk", "head_direct", 0.7)
    high_risk = _selection_candidate("HighRisk", "head_direct", 0.7)
    low_risk.rank_factors.risk = 0.0
    high_risk.rank_factors.risk = 1.0

    selected_set = score_and_select_candidates(benchmark, [low_risk, high_risk], budget=2)

    scores = {item.name: item.score for item in selected_set.candidates}
    assert scores["LowRisk"] == scores["HighRisk"]


def test_selected_set_to_phase_2_includes_sources_and_scoring_details():
    benchmark = AnalysisBenchmark(
        target_product="Acme",
        raw_description="Acme collaboration product",
        lifecycle_stage="launched",
        analysis_purpose="decision_support",
        analysis_goal="Select competitors.",
    )
    selected_set = workflow_module._select_top_competitors_by_category(
        benchmark,
        [
            _selection_candidate("DirectA", "head_direct", 0.9),
            _selection_candidate("DirectB", "challenger", 0.88),
            _selection_candidate("IndirectA", "indirect_cross", 0.86),
        ],
    )

    phase_2 = selected_set_to_phase_2(selected_set)
    first = phase_2["selected_competitors"][0]

    assert first["source_refs"]
    assert first["source_urls"]
    assert first["source_titles"]
    assert first["weight_assignment"]
    assert first["factor_scores"]
    assert "W_type" in first["weight_assignment"]
    assert "f_risk" in first["factor_scores"]


@pytest.mark.asyncio
async def test_candidate_existence_qa_runs_with_five_concurrency(monkeypatch, tmp_path):
    active = 0
    max_active = 0

    async def fake_existence_sources(candidate_name, request, settings, *, task_id, max_sources=3, should_cancel=None):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return [
            RawSource(
                url=f"https://example.com/{candidate_name.lower()}",
                title=f"{candidate_name} official",
                content=f"{candidate_name} is real. See https://example.com/internal for details.",
                source_type="search",
            )
        ], []

    monkeypatch.setattr(workflow_module, "collect_candidate_existence_sources_with_search_runner", fake_existence_sources)

    settings = Settings(
        openai_api_key=None,
        storage_backend="json",
        json_storage_dir=tmp_path / "json",
        sqlite_path=tmp_path / "app.db",
        candidate_qa_concurrency=5,
    )
    storage = JSONStorage(tmp_path / "json")
    storage.init()
    task = AnalysisTask(request=CreateTaskRequest(target_product="Acme"))
    task.id = "task_candidate_qa_concurrency"
    storage.save_task(task)
    workflow = AgentWorkflow(settings, storage, EventBus(storage))
    workflow.model = object()

    async def fake_llm_structured(task_id, node, schema, messages, *, llm_lane="main"):
        prompt = "\n".join(content for _role, content in messages)
        assert "https://example.com" not in prompt
        await asyncio.sleep(0.01)
        return schema(exists=True, confidence=0.9, reason="Confirmed.", matched_urls=[])

    monkeypatch.setattr(workflow, "_llm_structured", fake_llm_structured)
    candidates = [CandidateCompetitor(name=f"BetaSuite{i}", description="Real product.") for i in range(8)]

    verified, notes, qa_sources = await workflow._qa_verify_candidate_existence(
        "task_candidate_qa_concurrency",
        task.request,
        candidates,
    )

    assert max_active == 5
    assert [item.name for item in verified] == [item.name for item in candidates]
    assert not notes
    assert len(qa_sources) == len(candidates)
