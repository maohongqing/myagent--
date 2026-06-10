from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from myagent.backend.app.agents.strategy import merge_clarification_answer
from myagent.backend.app.agents.survey import (
    analyze_survey_responses,
    append_survey_analysis_to_markdown,
    build_fallback_survey_design,
    generate_survey_design,
)
from myagent.backend.app.agents.workflow import AgentWorkflow, ClarificationNeeded, LLMDecisionNeeded, TaskCancelled
from myagent.backend.app.config import BACKEND_DIR, Settings, get_settings
from myagent.backend.app.events import EventBus, format_sse
from myagent.backend.app.exporters import export_report
from myagent.backend.app.knowledge import (
    accept_suggestion,
    build_import_preview,
    commit_import,
    generate_update_suggestions,
    reject_suggestion,
)
from myagent.backend.app.logging_config import configure_logging
from myagent.backend.app.models import (
    AgentRunEvent,
    AnalysisTask,
    ClarificationAnswerRequest,
    CompetitorReport,
    CreateTaskRequest,
    DecisionAnswerRequest,
    FreshnessStatus,
    KnowledgeCompetitorCard,
    KnowledgeImportCommitRequest,
    KnowledgeIndustryCard,
    KnowledgeUpdateSuggestion,
    SurveyAnalysisResult,
    SurveyDesign,
    SurveyResponseAnalysisRequest,
    TaskDetail,
    EvidenceItem,
    ToolClaim,
    ToolExecutionResult,
    utc_now,
)
from myagent.backend.app.storage import Storage, create_storage

configure_logging()
logger = logging.getLogger("competitor_agent")


class AppState:
    settings: Settings
    storage: Storage
    event_bus: EventBus
    workflow: AgentWorkflow


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.settings = get_settings()
    state.storage = create_storage(state.settings)
    state.event_bus = EventBus(state.storage)
    state.workflow = AgentWorkflow(state.settings, state.storage, state.event_bus)
    yield


app = FastAPI(title="Competitor Analysis Agent", version="0.2.0", lifespan=lifespan)
app.mount("/data", StaticFiles(directory=BACKEND_DIR / "data"), name="data")

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_storage() -> Storage:
    return state.storage


def _save_event_snapshot(event: AgentRunEvent) -> None:
    output_dir = state.settings.task_output_dir / event.task_id / "steps"
    try:
        index = sum(1 for path in output_dir.glob("*_event.json")) + 1 if output_dir.exists() else 1
        node = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in event.node).strip("._") or "workflow"
        path = output_dir / f"{index:03d}_{node}_{event.status}_event.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(event.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save event snapshot for task %s node %s: %s", event.task_id, event.node, exc)


def _save_survey_design_snapshot(design: SurveyDesign) -> None:
    survey_dir = state.settings.task_output_dir / design.task_id / "survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    (survey_dir / "design.json").write_text(
        json.dumps(design.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _save_survey_analysis_snapshot(analysis: SurveyAnalysisResult) -> None:
    survey_dir = state.settings.task_output_dir / analysis.task_id / "survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    (survey_dir / "analysis.json").write_text(
        json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (survey_dir / "analysis.md").write_text(analysis.appendix_markdown, encoding="utf-8")


def _survey_design_markdown(design: SurveyDesign) -> str:
    lines = [
        f"## {design.title}",
        "",
        f"- 目标受访者：{design.target_respondents}",
        f"- 建议样本量：{design.suggested_sample_size}",
        f"- 投放渠道：{'、'.join(design.distribution_channels) or '待确认'}",
        "",
        "### 研究目标",
        *[f"- {item}" for item in design.research_goals],
        "",
        "### 问卷题目",
    ]
    for index, question in enumerate(design.questions, start=1):
        options = f" 选项：{' / '.join(question.options)}" if question.options else ""
        lines.append(f"{index}. [{question.type}] {question.title}（{question.dimension}）{options}")
    return "\n".join(lines).strip()


def _report_without_survey_appendix(markdown: str) -> str:
    marker = "## 用户问卷补充分析"
    return markdown.split(marker, 1)[0].rstrip() if marker in markdown else markdown.rstrip()


def _upsert_survey_evidence(report, analysis: SurveyAnalysisResult) -> None:
    evidence_id = f"survey_{analysis.id}"
    report.evidence = [item for item in report.evidence if item.id != evidence_id]
    report.evidence.append(
        EvidenceItem(
            id=evidence_id,
            url=f"manual://survey/{analysis.task_id}",
            title="用户问卷补充分析",
            source_type="manual",
            excerpt=analysis.summary or f"本次问卷分析纳入 {analysis.sample_size} 份有效样本。",
            credibility=0.55,
            related_fields=["survey_analysis", "first_party_research"],
        )
    )


def _upsert_survey_tool_result(report, analysis: SurveyAnalysisResult) -> None:
    report.tool_results = [
        item for item in report.tool_results if item.canonical_tool_name != "survey_analysis"
    ]
    report.tool_results.append(
        ToolExecutionResult(
            tool_name="问卷分析",
            canonical_tool_name="survey_analysis",
            scope_type="comparison",
            reference_id=f"survey_analysis_{analysis.task_id}",
            claims=[
                ToolClaim(
                    category="first_party_research",
                    title=f"问卷样本量 {analysis.sample_size}",
                    claim=analysis.summary or f"本次问卷分析纳入 {analysis.sample_size} 份有效样本。",
                    evidence_ids=[f"survey_{analysis.id}"],
                    confidence=0.55,
                    reasoning="用户上传问卷数据产生的一手研究补充，需结合样本局限解读。",
                )
            ],
            data=analysis.model_dump(mode="json"),
            evidence_ids=[f"survey_{analysis.id}"],
            confidence=0.55,
            reasoning="First-party survey analysis appended after the main report.",
            generated_by="fallback",
        )
    )


async def _publish_status(task: AnalysisTask, node: str, status: str, message: str) -> None:
    event = AgentRunEvent(task_id=task.id, node=node, status=status, message=message)
    _save_event_snapshot(event)
    await state.event_bus.publish(event)


async def run_task(task_id: str) -> None:
    task = state.storage.get_task(task_id)
    if not task:
        return

    try:
        if task.status in {"cancelling", "cancelled"}:
            raise TaskCancelled()
        task.status = "running"
        task.current_node = "workflow"
        task.pending_decision = None
        state.storage.save_task(task)
        await _publish_status(task, "workflow", "running", "任务开始执行。")

        max_revision_loops = task.request.max_revision_loops
        if max_revision_loops is None:
            max_revision_loops = state.settings.max_revision_loops
        report = await state.workflow.run(task.id, task.request, max_revision_loops=max_revision_loops)
        generate_update_suggestions(state.storage, report)

        task.status = "completed"
        task.current_node = None
        task.pending_clarification = None
        task.pending_decision = None
        task.completed_at = utc_now()
        state.storage.save_task(task)
        await _publish_status(task, "workflow", "completed", "任务完成。")
    except ClarificationNeeded as exc:
        task.status = "waiting_clarification"
        task.current_node = "review_agent"
        task.pending_clarification = exc.clarification
        task.pending_decision = None
        task.error = None
        state.storage.save_task(task)
        await _publish_status(task, "review_agent", "waiting", exc.clarification.question)
    except LLMDecisionNeeded as exc:
        task.status = "waiting_decision"
        task.current_node = exc.decision.node
        task.pending_decision = exc.decision
        task.pending_clarification = None
        task.error = exc.decision.reason
        state.storage.save_task(task)
        event = AgentRunEvent(
            task_id=task.id,
            node=exc.decision.node,
            status="waiting",
            message=exc.decision.question,
            error=exc.decision.reason,
            details={"kind": "llm_failure_decision", "decision": exc.decision.model_dump(mode="json")},
        )
        _save_event_snapshot(event)
        await state.event_bus.publish(event)
    except TaskCancelled:
        task.status = "cancelled"
        task.current_node = None
        task.error = "Task cancelled by user."
        task.completed_at = utc_now()
        state.storage.save_task(task)
        await _publish_status(task, "workflow", "cancelled", "任务已中止。")
    except Exception as exc:
        if task.status == "cancelling":
            task.status = "cancelled"
            task.current_node = None
            task.error = "Task cancelled by user."
            task.completed_at = utc_now()
            state.storage.save_task(task)
            await _publish_status(task, "workflow", "cancelled", "任务已中止。")
            return
        task.status = "failed"
        task.current_node = None
        task.error = str(exc)
        state.storage.save_task(task)
        logger.exception("Task %s failed", task.id)
        await _publish_status(task, "workflow", "failed", f"任务失败：{exc}")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/tasks", response_model=AnalysisTask)
async def create_task(request: CreateTaskRequest, background_tasks: BackgroundTasks) -> AnalysisTask:
    task = AnalysisTask(request=request)
    state.storage.save_task(task)
    await _publish_status(task, "workflow", "queued", "任务已创建，等待执行。")
    background_tasks.add_task(run_task, task.id)
    return task


async def _resume_task(task: AnalysisTask, background_tasks: BackgroundTasks) -> AnalysisTask:
    if task.status == "completed":
        raise HTTPException(status_code=409, detail="Task is already completed")
    if task.status == "waiting_clarification":
        raise HTTPException(status_code=409, detail="Task is waiting for clarification")
    if task.status == "waiting_decision":
        raise HTTPException(status_code=409, detail="Task is waiting for a decision")

    task.status = "queued"
    task.current_node = "workflow"
    task.pending_clarification = None
    task.pending_decision = None
    task.error = None
    task.completed_at = None
    state.storage.save_task(task)
    await _publish_status(task, "workflow", "queued", "任务已从检查点重新入队，准备继续执行。")
    background_tasks.add_task(run_task, task.id)
    return task


@app.post("/api/tasks/resume-latest", response_model=AnalysisTask)
async def resume_latest_task(background_tasks: BackgroundTasks, storage: Storage = Depends(get_storage)) -> AnalysisTask:
    for task in storage.list_tasks(limit=200):
        if task.status == "completed":
            continue
        return await _resume_task(task, background_tasks)
    raise HTTPException(status_code=404, detail="No resumable task found")


@app.post("/api/tasks/{task_id}/resume", response_model=AnalysisTask)
async def resume_task(task_id: str, background_tasks: BackgroundTasks, storage: Storage = Depends(get_storage)) -> AnalysisTask:
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return await _resume_task(task, background_tasks)


@app.post("/api/tasks/{task_id}/clarifications", response_model=AnalysisTask)
async def submit_clarification(
    task_id: str,
    request: ClarificationAnswerRequest,
    background_tasks: BackgroundTasks,
) -> AnalysisTask:
    task = state.storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "waiting_clarification" or not task.pending_clarification:
        raise HTTPException(status_code=409, detail="Task is not waiting for clarification")

    answer = request.selected_option or request.answer
    task.request = merge_clarification_answer(task.request, task.pending_clarification, answer)
    task.status = "queued"
    task.current_node = "workflow"
    task.pending_clarification = None
    task.pending_decision = None
    task.error = None
    state.storage.save_task(task)
    await _publish_status(task, "workflow", "queued", "已收到澄清补充，任务继续执行。")
    background_tasks.add_task(run_task, task.id)
    return task


@app.post("/api/tasks/{task_id}/cancel", response_model=AnalysisTask)
async def cancel_task(task_id: str, storage: Storage = Depends(get_storage)) -> AnalysisTask:
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status in {"completed", "failed", "cancelled"}:
        return task
    task.status = "cancelling"
    task.error = "Cancellation requested by user."
    storage.save_task(task)
    await _publish_status(task, "workflow", "revision", "正在安全中止，当前长调用结束后会停止。")
    return task


@app.post("/api/tasks/{task_id}/decisions", response_model=AnalysisTask)
async def submit_decision(
    task_id: str,
    request: DecisionAnswerRequest,
    background_tasks: BackgroundTasks,
) -> AnalysisTask:
    task = state.storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "waiting_decision" or not task.pending_decision:
        raise HTTPException(status_code=409, detail="Task is not waiting for a decision")

    if request.action == "cancel":
        task.status = "cancelled"
        task.current_node = None
        task.pending_decision = None
        task.error = "Task cancelled after LLM failure."
        task.completed_at = utc_now()
        state.storage.save_task(task)
        await _publish_status(task, "workflow", "cancelled", "任务已中止。")
        return task

    if request.action == "continue_without_llm":
        task.request.llm_failure_policy = "continue_without_llm"
    else:
        task.request.llm_failure_policy = "ask"

    task.status = "queued"
    task.current_node = "workflow"
    task.pending_decision = None
    task.error = None
    state.storage.save_task(task)
    await _publish_status(task, "workflow", "queued", "已收到处理选择，任务继续执行。")
    background_tasks.add_task(run_task, task.id)
    return task


@app.get("/api/tasks", response_model=list[AnalysisTask])
async def list_tasks(
    limit: int = Query(default=50, ge=1, le=200), storage: Storage = Depends(get_storage)
) -> list[AnalysisTask]:
    return storage.list_tasks(limit)


@app.get("/api/tasks/{task_id}", response_model=TaskDetail)
async def get_task(task_id: str, storage: Storage = Depends(get_storage)) -> TaskDetail:
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskDetail(
        task=task,
        report=storage.get_report(task_id),
        events=storage.list_events(task_id),
        survey_design=storage.get_survey_design(task_id),
        survey_analysis=storage.get_survey_analysis(task_id),
    )


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str, storage: Storage = Depends(get_storage)) -> StreamingResponse:
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    async def stream():
        for event in storage.list_events(task_id):
            yield format_sse("agent-event", event.model_dump(mode="json"))
        latest = storage.get_task(task_id)
        if latest and latest.status in {"completed", "failed", "waiting_clarification", "waiting_decision", "cancelled"}:
            yield format_sse("done", {"status": latest.status})
            return

        async for event in state.event_bus.subscribe(task_id):
            yield format_sse("agent-event", event.model_dump(mode="json"))
            latest = storage.get_task(task_id)
            if latest and latest.status in {"completed", "failed", "waiting_clarification", "waiting_decision", "cancelled"}:
                await asyncio.sleep(0.05)
                yield format_sse("done", {"status": latest.status})
                return

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/tasks/{task_id}/report")
async def get_report(task_id: str, storage: Storage = Depends(get_storage)):
    report = storage.get_report(task_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.get("/api/tasks/{task_id}/survey-design", response_model=SurveyDesign)
async def get_survey_design(task_id: str, storage: Storage = Depends(get_storage)) -> SurveyDesign:
    design = storage.get_survey_design(task_id)
    if not design:
        raise HTTPException(status_code=404, detail="Survey design not found")
    return design


@app.post("/api/tasks/{task_id}/survey-design/regenerate", response_model=SurveyDesign)
async def regenerate_survey_design(task_id: str, storage: Storage = Depends(get_storage)) -> SurveyDesign:
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    report = storage.get_report(task_id)
    benchmark = report.analysis_benchmark if report else None
    selected_set = report.selected_competitor_set if report else None
    if not benchmark and not selected_set:
        design = build_fallback_survey_design(task_id, task.request)
    else:
        design = await generate_survey_design(
            state.workflow.model,
            task_id,
            task.request,
            benchmark,
            selected_set,
            {"source": "api_regenerate"},
        )
    storage.save_survey_design(design)
    _save_survey_design_snapshot(design)
    await state.event_bus.publish(
        AgentRunEvent(
            task_id=task_id,
            node="survey_design_agent",
            status="completed",
            message="问卷设计已生成。",
            details={
                "kind": "survey_design",
                "survey_design": design.model_dump(mode="json"),
                "display_markdown": _survey_design_markdown(design),
                "ui_stage_key": "survey_design",
                "ui_title": "问卷设计",
            },
        )
    )
    return design


@app.post("/api/tasks/{task_id}/survey-responses/analyze", response_model=SurveyAnalysisResult)
async def analyze_task_survey_responses(
    task_id: str,
    request: SurveyResponseAnalysisRequest,
    storage: Storage = Depends(get_storage),
) -> SurveyAnalysisResult:
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        analysis = await analyze_survey_responses(
            state.workflow.model,
            task_id,
            task.request,
            storage.get_survey_design(task_id),
            request.raw_responses,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    storage.save_survey_analysis(analysis)
    _save_survey_analysis_snapshot(analysis)
    await state.event_bus.publish(
        AgentRunEvent(
            task_id=task_id,
            node="survey_analysis_agent",
            status="completed",
            message="问卷结果分析已生成。",
            details={
                "kind": "survey_analysis",
                "survey_analysis": analysis.model_dump(mode="json"),
                "display_markdown": analysis.appendix_markdown,
                "ui_stage_key": "survey_analysis",
                "ui_title": "问卷结果分析",
            },
        )
    )
    return analysis


@app.get("/api/tasks/{task_id}/survey-analysis", response_model=SurveyAnalysisResult)
async def get_survey_analysis(task_id: str, storage: Storage = Depends(get_storage)) -> SurveyAnalysisResult:
    analysis = storage.get_survey_analysis(task_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Survey analysis not found")
    return analysis


@app.post("/api/tasks/{task_id}/survey-analysis/append-to-report", response_model=CompetitorReport)
async def append_survey_analysis_to_report(task_id: str, storage: Storage = Depends(get_storage)) -> CompetitorReport:
    report = storage.get_report(task_id)
    if not report:
        raise HTTPException(status_code=409, detail="Report is not ready yet")
    analysis = storage.get_survey_analysis(task_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Survey analysis not found")
    report.survey_design = storage.get_survey_design(task_id)
    report.survey_analysis = analysis
    base_markdown = _report_without_survey_appendix(report.deep_dive_markdown)
    report.deep_dive_markdown = append_survey_analysis_to_markdown(base_markdown, analysis.appendix_markdown)
    _upsert_survey_evidence(report, analysis)
    _upsert_survey_tool_result(report, analysis)
    storage.save_report(report)
    state.workflow._save_final_report_snapshot(task_id, report)
    return report


@app.get("/api/tasks/{task_id}/sources")
async def get_sources(task_id: str, storage: Storage = Depends(get_storage)):
    report = storage.get_report(task_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report.evidence


@app.get("/api/tasks/{task_id}/search-runs")
async def list_search_runs(task_id: str, storage: Storage = Depends(get_storage)):
    if not storage.get_task(task_id) and not _search_runs_root(task_id).exists():
        raise HTTPException(status_code=404, detail="Task not found")
    runs = [_search_run_summary(path, task_id) for path in _search_run_dirs(task_id)]
    return {"task_id": task_id, "runs": sorted(runs, key=lambda item: item.get("created_at") or "", reverse=True)}


@app.get("/api/tasks/{task_id}/search-runs/{run_key}")
async def get_search_run(task_id: str, run_key: str, storage: Storage = Depends(get_storage)):
    if not storage.get_task(task_id) and not _search_runs_root(task_id).exists():
        raise HTTPException(status_code=404, detail="Task not found")
    run_dir = _decode_run_key(task_id, run_key)
    if not run_dir or not run_dir.exists():
        raise HTTPException(status_code=404, detail="Search run not found")
    summary = _search_run_summary(run_dir, task_id)
    results_payload, results_status = _safe_json(run_dir / "results.json")
    chunks_payload, chunks_status = _safe_json(run_dir / "extractions" / "chunks.json")
    summary["status"] = "complete" if results_status == "ready" else results_status
    return {
        **summary,
        "results": (results_payload or {}).get("results", []),
        "search_page_screenshots": (results_payload or {}).get("search_page_screenshots", []),
        "chunks": (chunks_payload or {}).get("chunks", []) if isinstance(chunks_payload, dict) else [],
        "results_status": results_status,
        "chunks_status": chunks_status,
    }


@app.get("/api/tasks/{task_id}/outputs")
async def list_task_outputs(task_id: str, storage: Storage = Depends(get_storage)):
    root = state.settings.task_output_dir / task_id
    if not storage.get_task(task_id) and not root.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    files: list[dict[str, object]] = []
    if root.exists():
        data_root = state.settings.data_dir.resolve()
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.stat().st_mtime):
            rel_data = path.resolve().relative_to(data_root).as_posix()
            files.append(
                {
                    "path": path.resolve().relative_to(root.resolve()).as_posix(),
                    "size": path.stat().st_size,
                    "modified_at": path.stat().st_mtime,
                    "url": f"/data/{rel_data}",
                }
            )
    return {"task_id": task_id, "root": str(root), "files": files}


@app.get("/api/tasks/{task_id}/export")
async def export_task_report(
    task_id: str,
    format: str = Query(pattern="^(md|json|pdf|docx)$"),
    storage: Storage = Depends(get_storage),
) -> Response:
    report = storage.get_report(task_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    payload, media_type, suffix = export_report(report, format)
    filename = f"{report.target_product}_competitor_report.{suffix}"
    headers = {
        "Content-Disposition": (
            f"attachment; filename=competitor_report.{suffix}; "
            f"filename*=UTF-8''{quote(filename)}"
        )
    }
    return Response(content=payload, media_type=media_type, headers=headers)


@app.get("/api/knowledge/competitors", response_model=list[KnowledgeCompetitorCard])
async def list_knowledge_competitors(
    q: str = "",
    industry: str = "",
    tag: str = "",
    freshness_status: FreshnessStatus | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    storage: Storage = Depends(get_storage),
) -> list[KnowledgeCompetitorCard]:
    return storage.list_knowledge_competitor_cards(q, industry, tag, freshness_status, limit)


@app.post("/api/knowledge/competitors", response_model=KnowledgeCompetitorCard)
async def create_knowledge_competitor(
    card: KnowledgeCompetitorCard, storage: Storage = Depends(get_storage)
) -> KnowledgeCompetitorCard:
    storage.save_knowledge_competitor_card(card, version_action="create")
    return card


@app.get("/api/knowledge/competitors/{card_id}")
async def get_knowledge_competitor(card_id: str, storage: Storage = Depends(get_storage)):
    card = storage.get_knowledge_competitor_card(card_id)
    if not card or card.deleted:
        raise HTTPException(status_code=404, detail="Knowledge competitor card not found")
    return {
        "card": card,
        "sources": storage.list_knowledge_sources(card.knowledge_source_ids),
        "versions": storage.list_knowledge_versions("competitor", card.id),
    }


@app.put("/api/knowledge/competitors/{card_id}", response_model=KnowledgeCompetitorCard)
async def update_knowledge_competitor(
    card_id: str, card: KnowledgeCompetitorCard, storage: Storage = Depends(get_storage)
) -> KnowledgeCompetitorCard:
    if card_id != card.id:
        raise HTTPException(status_code=400, detail="Card id mismatch")
    if not storage.get_knowledge_competitor_card(card_id):
        raise HTTPException(status_code=404, detail="Knowledge competitor card not found")
    storage.save_knowledge_competitor_card(card, version_action="update")
    return card


@app.delete("/api/knowledge/competitors/{card_id}", response_model=KnowledgeCompetitorCard)
async def delete_knowledge_competitor(card_id: str, storage: Storage = Depends(get_storage)) -> KnowledgeCompetitorCard:
    card = storage.delete_knowledge_competitor_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Knowledge competitor card not found")
    return card


@app.get("/api/knowledge/industries", response_model=list[KnowledgeIndustryCard])
async def list_knowledge_industries(
    q: str = "",
    industry: str = "",
    tag: str = "",
    freshness_status: FreshnessStatus | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    storage: Storage = Depends(get_storage),
) -> list[KnowledgeIndustryCard]:
    return storage.list_knowledge_industry_cards(q, industry, tag, freshness_status, limit)


@app.post("/api/knowledge/industries", response_model=KnowledgeIndustryCard)
async def create_knowledge_industry(
    card: KnowledgeIndustryCard, storage: Storage = Depends(get_storage)
) -> KnowledgeIndustryCard:
    storage.save_knowledge_industry_card(card, version_action="create")
    return card


@app.get("/api/knowledge/industries/{card_id}")
async def get_knowledge_industry(card_id: str, storage: Storage = Depends(get_storage)):
    card = storage.get_knowledge_industry_card(card_id)
    if not card or card.deleted:
        raise HTTPException(status_code=404, detail="Knowledge industry card not found")
    return {
        "card": card,
        "sources": storage.list_knowledge_sources(card.knowledge_source_ids),
        "versions": storage.list_knowledge_versions("industry", card.id),
    }


@app.put("/api/knowledge/industries/{card_id}", response_model=KnowledgeIndustryCard)
async def update_knowledge_industry(
    card_id: str, card: KnowledgeIndustryCard, storage: Storage = Depends(get_storage)
) -> KnowledgeIndustryCard:
    if card_id != card.id:
        raise HTTPException(status_code=400, detail="Card id mismatch")
    if not storage.get_knowledge_industry_card(card_id):
        raise HTTPException(status_code=404, detail="Knowledge industry card not found")
    storage.save_knowledge_industry_card(card, version_action="update")
    return card


@app.delete("/api/knowledge/industries/{card_id}", response_model=KnowledgeIndustryCard)
async def delete_knowledge_industry(card_id: str, storage: Storage = Depends(get_storage)) -> KnowledgeIndustryCard:
    card = storage.delete_knowledge_industry_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Knowledge industry card not found")
    return card


@app.post("/api/knowledge/import-from-task/{task_id}")
async def preview_knowledge_import(task_id: str, storage: Storage = Depends(get_storage)):
    report = storage.get_report(task_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return build_import_preview(report)


@app.post("/api/knowledge/import-from-task/{task_id}/commit")
async def commit_knowledge_import(
    task_id: str, request: KnowledgeImportCommitRequest, storage: Storage = Depends(get_storage)
):
    report = storage.get_report(task_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return commit_import(storage, report, request)


@app.get("/api/knowledge/suggestions", response_model=list[KnowledgeUpdateSuggestion])
async def list_knowledge_suggestions(
    status: str = "pending",
    limit: int = Query(default=100, ge=1, le=300),
    storage: Storage = Depends(get_storage),
) -> list[KnowledgeUpdateSuggestion]:
    return storage.list_knowledge_suggestions(status=status, limit=limit)


@app.post("/api/knowledge/suggestions/{suggestion_id}/accept", response_model=KnowledgeUpdateSuggestion)
async def accept_knowledge_suggestion(
    suggestion_id: str, storage: Storage = Depends(get_storage)
) -> KnowledgeUpdateSuggestion:
    suggestion = accept_suggestion(storage, suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Knowledge update suggestion not found")
    return suggestion


@app.post("/api/knowledge/suggestions/{suggestion_id}/reject", response_model=KnowledgeUpdateSuggestion)
async def reject_knowledge_suggestion(
    suggestion_id: str, storage: Storage = Depends(get_storage)
) -> KnowledgeUpdateSuggestion:
    suggestion = reject_suggestion(storage, suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Knowledge update suggestion not found")
    return suggestion


def _search_runs_root(task_id: str) -> Path:
    return Path(state.settings.sqlite_path).parent / "search_runs" / task_id


def _search_run_dirs(task_id: str) -> list[Path]:
    root = _search_runs_root(task_id)
    if not root.exists():
        return []
    return sorted([path for path in root.glob("*/*") if path.is_dir()], key=lambda item: item.stat().st_mtime, reverse=True)


def _encode_run_key(path: Path, task_id: str) -> str:
    relative = path.resolve().relative_to(_search_runs_root(task_id).resolve()).as_posix()
    return base64.urlsafe_b64encode(relative.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_run_key(task_id: str, run_key: str) -> Path | None:
    try:
        padded = run_key + "=" * (-len(run_key) % 4)
        relative = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception:
        return None
    root = _search_runs_root(task_id).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def _safe_json(path: Path) -> tuple[dict | None, str]:
    if not path.exists():
        return None, "writing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), "ready"
    except json.JSONDecodeError:
        return None, "writing"
    except OSError:
        return None, "unavailable"


def _search_run_summary(run_dir: Path, task_id: str) -> dict:
    payload, status = _safe_json(run_dir / "results.json")
    chunks_payload, chunks_status = _safe_json(run_dir / "extractions" / "chunks.json")
    payload = payload or {}
    chunks = chunks_payload.get("chunks", []) if isinstance(chunks_payload, dict) else []
    results = payload.get("results", []) if isinstance(payload.get("results"), list) else []
    rel_dir = run_dir.resolve().relative_to(Path(state.settings.sqlite_path).parent.resolve()).as_posix()
    purpose = run_dir.parent.name
    return {
        "run_key": _encode_run_key(run_dir, task_id),
        "task_id": task_id,
        "purpose": purpose,
        "query": payload.get("query") or "",
        "methods": payload.get("methods") or [],
        "deep_backend": payload.get("deep_backend") or "none",
        "status": "complete" if status == "ready" else status,
        "created_at": payload.get("created_at") or "",
        "run_dir": rel_dir,
        "report_path": f"{rel_dir}/report.md" if (run_dir / "report.md").exists() else None,
        "results_json_path": f"{rel_dir}/results.json" if (run_dir / "results.json").exists() else None,
        "result_count": len(results),
        "chunk_count": len(chunks),
        "screenshot_count": _count_screenshots(payload, chunks),
        "skipped_duplicate_count": int(payload.get("skipped_duplicate_count") or 0),
        "errors": _search_run_errors(payload),
        "chunks_status": chunks_status,
    }


def _count_screenshots(payload: dict, chunks: list) -> int:
    count = 0
    for item in payload.get("search_page_screenshots") or []:
        if item.get("screenshot_path"):
            count += 1
    for item in payload.get("results") or []:
        if item.get("screenshot_path"):
            count += 1
    for item in chunks:
        if item.get("chunk_shot_path"):
            count += 1
        count += len(item.get("chunk_shot_paths") or [])
    return count


def _search_run_errors(payload: dict) -> list[str]:
    errors: list[str] = []
    for item in payload.get("results") or []:
        error = item.get("error") or item.get("content_error") or item.get("screenshot_error")
        if error:
            errors.append(str(error))
    tavily = payload.get("tavily_collection") or {}
    for item in tavily.get("errors") or []:
        errors.append(str(item.get("error") or item))
    return errors[:12]
