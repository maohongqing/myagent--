from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from myagent.backend.app.agents.deep_report import build_deep_dive_markdown
from myagent.backend.app.agents.llm import friendly_llm_error, get_chat_model, structured_ainvoke
from myagent.backend.app.agents.prompt_bridge import (
    phase_4_qa_intent,
    qa_system_prompt,
)
from myagent.backend.app.agents.state_tree import merge_updated_nodes
from myagent.backend.app.agents.workflow import QAInspectionOutput
from myagent.backend.app.config import Settings
from myagent.backend.app.models import CompetitorReport


DEFAULT_INPUT = BACKEND_DIR / "demos" / "analyst_phase4_output.json"
DEFAULT_OUTPUT = BACKEND_DIR / "demos" / "final_report_output.json"
DEFAULT_MARKDOWN_OUTPUT = BACKEND_DIR / "demos" / "final_report.md"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _render_prompt(template: str, **kwargs: Any) -> str:
    rendered = template
    for key, value in kwargs.items():
        text = json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value
        rendered = rendered.replace("{{" + key + "}}", text)
    return rendered


def _qa2_local_check(state_tree: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    dynamic_analyses = state_tree.get("phase_4_synthesis", {}).get("dynamic_analyses", [])
    conclusion = state_tree.get("phase_4_synthesis", {}).get("conclusion", {})
    expected_units = state_tree.get("phase_3_strategy", {}).get("selected_analysis_units") or []
    actual_units = [
        item.get("analysis_unit_name") or item.get("framework_name")
        for item in dynamic_analyses
        if isinstance(item, dict)
    ]

    if not dynamic_analyses:
        issues.append("phase_4_synthesis.dynamic_analyses is empty.")
    missing_units = [unit for unit in expected_units if unit not in actual_units]
    if missing_units:
        issues.append(f"phase_4_synthesis.dynamic_analyses missing units: {', '.join(missing_units)}.")
    if not conclusion.get("competitive_strategy"):
        issues.append("phase_4_synthesis.conclusion.competitive_strategy is empty.")
    return not issues, issues


def _fallback_markdown(report: CompetitorReport, state_tree: dict[str, Any]) -> str:
    try:
        markdown = build_deep_dive_markdown(report)
        if markdown:
            return markdown
    except Exception:
        pass

    competitors = report.competitors
    phase_4 = state_tree.get("phase_4_synthesis", {})
    strategy = phase_4.get("conclusion", {}).get("competitive_strategy") or report.executive_summary
    suggestions = phase_4.get("conclusion", {}).get("actionable_suggestions") or [
        {"suggestion": item, "priority": "medium", "rationale": "Derived from analysis report."}
        for item in report.recommendations
    ]
    lines = [
        f"# {report.target_product} 竞品分析报告",
        "",
        "## 一、封面与综述",
        report.executive_summary or "本报告基于当前调试数据生成，用于验证工作流输出。",
        "",
        "## 二、竞品选择",
    ]
    for item in competitors:
        lines.append(f"- **{item.name}**：{item.positioning}")
    lines.extend(["", "## 三、关键发现"])
    for finding in report.findings[:5]:
        evidence = ", ".join(finding.evidence_ids) if finding.evidence_ids else "无证据 ID"
        lines.append(f"- **{finding.title}**：{finding.detail}（证据：{evidence}）")
    lines.extend(["", "## 四、多维深度拆解"])
    for analysis in phase_4.get("dynamic_analyses", []):
        name = analysis.get("analysis_unit_name") or analysis.get("framework_name")
        claims = analysis.get("framework_content", {}).get("claims", [])
        lines.append(f"### {name}")
        for claim in claims[:3]:
            evidence = ", ".join(claim.get("evidence_ids", [])) or "无证据 ID"
            lines.append(f"- {claim.get('claim', '')}（证据：{evidence}）")
    lines.extend(["", "## 五、结论与行动方案", strategy or "暂无明确策略结论。"])
    for item in suggestions[:5]:
        if isinstance(item, dict):
            lines.append(f"- [{item.get('priority', 'medium')}] {item.get('suggestion')}：{item.get('rationale')}")
        else:
            lines.append(f"- {item}")
    if report.risk_notice:
        lines.extend(["", "### 风险提示", report.risk_notice])
    return "\n".join(lines).strip() + "\n"


async def run(
    input_path: Path,
    output_path: Path,
    markdown_output_path: Path,
    timeout_seconds: float | None,
    no_llm: bool,
) -> None:
    payload = _load_json(input_path)
    state_tree = payload["state_tree"]
    report = CompetitorReport.model_validate(payload["analysis_report"])

    settings = Settings()
    if timeout_seconds is not None:
        settings.openai_timeout_seconds = timeout_seconds
    model = None if no_llm else get_chat_model(settings)

    qa_error: str | None = None
    inspection = QAInspectionOutput(inspection_status="pass")
    if model is not None:
        try:
            prompt = _render_prompt(phase_4_qa_intent, complete_state_tree_json=state_tree)
            inspection = await structured_ainvoke(
                model,
                QAInspectionOutput,
                [("system", qa_system_prompt), ("user", prompt)],
            )
        except Exception as exc:
            qa_error = friendly_llm_error(exc)
            print(f"QA checkpoint 2 LLM call failed; using local QA checks: {type(exc).__name__}: {exc}")

    local_passed, local_issues = _qa2_local_check(state_tree)
    issues = [*inspection.issues_found, *local_issues]
    passed = inspection.inspection_status == "pass" and local_passed and not issues
    state_tree.setdefault("quality_gates", {})["checkpoint_2"] = {
        "inspection_status": "pass" if passed else "reject",
        "issues_found": issues,
        "feedback": inspection.feedback_to_planner_or_analyzer or inspection.feedback_to_analyzer,
    }

    final_report = report
    writer_error: str | None = None

    markdown = final_report.deep_dive_markdown or _fallback_markdown(final_report, state_tree)
    final_report.deep_dive_markdown = markdown
    state_tree["final_report_markdown"] = markdown
    state_tree.setdefault("global_metadata", {})["status"] = "final_report_completed" if passed else "checkpoint_2_rejected"

    output_payload = {
        "input": str(input_path),
        "qa_checkpoint_2": {
            "passed": passed,
            "llm_error": qa_error,
            "inspection": inspection.model_dump(mode="json"),
            "local_issues": local_issues,
            "checkpoint": state_tree["quality_gates"]["checkpoint_2"],
        },
        "writer": {
            "llm_error": writer_error,
            "markdown_output": str(markdown_output_path),
        },
        "report": final_report.model_dump(mode="json"),
        "state_tree": state_tree,
        "final_report_markdown": markdown,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(markdown, encoding="utf-8")

    print(f"Saved final report JSON to: {output_path}")
    print(f"Saved final report Markdown to: {markdown_output_path}")
    print(f"QA checkpoint 2 passed: {passed}")
    print("Final report preview:")
    print(markdown[:3000])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QA checkpoint 2 and Writer Agent final report from Analyst phase-4 output."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to analyst_phase4_output.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write final report JSON.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="Path to write final Markdown report.",
    )
    parser.add_argument("--timeout", type=float, default=None, help="Override OPENAI_TIMEOUT_SECONDS.")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls and use fallback QA/writer.")
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.markdown_output, args.timeout, args.no_llm))


if __name__ == "__main__":
    main()
