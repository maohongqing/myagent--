from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from io import StringIO
from typing import Any

from myagent.backend.app.agents.llm import retry_structured_ainvoke
from myagent.backend.app.models import (
    AnalysisBenchmark,
    CandidateCompetitor,
    CreateTaskRequest,
    SelectedCompetitorSet,
    SurveyAnalysisResult,
    SurveyDesign,
    SurveyQuestion,
)
from myagent.prompt.survey_agent import (
    survey_analysis_system_prompt,
    survey_analysis_user_prompt,
    survey_design_system_prompt,
    survey_design_user_prompt,
)


def _competitor_names(selected_set: SelectedCompetitorSet | None, request: CreateTaskRequest) -> list[str]:
    names: list[str] = []
    if selected_set:
        names.extend(item.name for item in selected_set.selected)
    names.extend(request.competitors)
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        cleaned = str(name).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result[:8]


def build_fallback_survey_design(
    task_id: str,
    request: CreateTaskRequest,
    benchmark: AnalysisBenchmark | None = None,
    selected_set: SelectedCompetitorSet | None = None,
) -> SurveyDesign:
    competitors = _competitor_names(selected_set, request)
    product = request.target_product or request.our_product or (benchmark.target_product if benchmark else "") or "目标产品"
    category = request.product_category or request.market or (benchmark.target_product if benchmark else "") or "目标品类"
    mapped = competitors[:5]
    questions = [
        SurveyQuestion(
            type="single_choice",
            title=f"你是否购买或使用过{category}相关产品？",
            options=["经常购买/使用", "偶尔购买/使用", "了解但未购买/使用", "不了解"],
            dimension="用户基础与品类渗透",
        ),
        SurveyQuestion(
            type="multiple_choice",
            title="你选择这类产品时最看重哪些因素？",
            options=["价格", "正品/品质保障", "款式丰富度", "交易安全", "物流/履约", "社区口碑", "售后服务", "稀缺款获取"],
            dimension="购买决策因素",
        ),
        SurveyQuestion(
            type="multiple_choice",
            title="你听说过或使用过以下哪些平台/产品？",
            options=mapped or ["竞品A", "竞品B", product],
            dimension="竞品认知与触达",
            mapped_competitors=mapped,
        ),
        SurveyQuestion(
            type="ranking",
            title="请按偏好排序你更愿意使用的平台/产品。",
            options=mapped or ["竞品A", "竞品B", product],
            dimension="竞品偏好排序",
            mapped_competitors=mapped,
        ),
        SurveyQuestion(
            type="scale",
            title="你对当前常用平台/产品的整体满意度是多少？",
            options=["1", "2", "3", "4", "5"],
            dimension="满意度",
            mapped_competitors=mapped,
        ),
        SurveyQuestion(
            type="single_choice",
            title="如果平台提供更强的保障服务，你能接受的价格上浮幅度是？",
            options=["不能接受上浮", "5%以内", "5%-10%", "10%-20%", "20%以上"],
            dimension="价格敏感度与付费意愿",
        ),
        SurveyQuestion(
            type="multiple_choice",
            title="你在购买或交易过程中遇到过哪些问题？",
            options=["假货/瑕疵担忧", "价格不透明", "售后困难", "卖家/买家不可信", "物流慢", "抢不到热门款", "平台规则复杂", "没有明显问题"],
            dimension="痛点与风险",
        ),
        SurveyQuestion(
            type="text",
            title=f"如果让你改进{product}或同类平台，你最希望优先解决什么问题？",
            options=[],
            dimension="开放需求与机会点",
            mapped_competitors=mapped,
        ),
    ]
    return SurveyDesign(
        task_id=task_id,
        title=f"{category}用户需求与竞品偏好问卷",
        target_respondents=f"了解、购买或使用过{category}相关产品的潜在用户、活跃用户和流失用户。",
        research_goals=[
            "识别用户选择竞品时的核心决策因素。",
            "比较目标产品与主要竞品的认知、偏好和满意度差异。",
            "验证公开竞品分析中的需求、价格和信任假设。",
            "发现可转化为产品策略的用户痛点和机会点。",
        ],
        suggested_sample_size="建议至少 50 份有效样本；如要比较不同竞品用户，建议 100-200 份。",
        distribution_channels=["目标用户社群", "产品私域用户", "竞品相关社区", "电商/内容平台用户群", "访谈邀约名单"],
        questions=questions,
    )


async def generate_survey_design(
    model,
    task_id: str,
    request: CreateTaskRequest,
    benchmark: AnalysisBenchmark | None = None,
    selected_set: SelectedCompetitorSet | None = None,
    state_context: dict[str, Any] | None = None,
) -> SurveyDesign:
    fallback = build_fallback_survey_design(task_id, request, benchmark, selected_set)
    if not model or request.llm_failure_policy == "continue_without_llm":
        return fallback
    context = {
        "task_id": task_id,
        "request": request.model_dump(mode="json"),
        "benchmark": benchmark.model_dump(mode="json") if benchmark else None,
        "selected_competitors": [item.model_dump(mode="json") for item in selected_set.selected] if selected_set else [],
        "state_context": state_context or {},
        "fallback_question_count": len(fallback.questions),
    }
    try:
        result = await retry_structured_ainvoke(
            model,
            SurveyDesign,
            [
                ("system", survey_design_system_prompt),
                ("user", survey_design_user_prompt.format(context=json.dumps(context, ensure_ascii=False, default=str))),
            ],
        )
        result.task_id = task_id
        if len(result.questions) < 6:
            return fallback
        return result
    except Exception:
        return fallback


def parse_survey_rows(raw_responses: str) -> list[dict[str, str]]:
    text = raw_responses.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            rows: list[dict[str, str]] = []
            for item in parsed:
                if isinstance(item, dict):
                    rows.append({str(k): _cell_to_text(v) for k, v in item.items()})
            if rows:
                return rows
    except json.JSONDecodeError:
        pass

    for delimiter in [",", "\t", "|"]:
        reader = csv.DictReader(StringIO(text), delimiter=delimiter)
        rows = [{str(k): _cell_to_text(v) for k, v in row.items() if k is not None} for row in reader]
        if rows and any(row for row in rows):
            return rows
    return []


def _cell_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "、".join(_cell_to_text(item) for item in value)
    return str(value).strip()


def _split_multi_value(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，、;/；|]+", value) if item.strip()]


def _top_choice_findings(rows: list[dict[str, str]], limit: int = 5) -> list[str]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for key, value in row.items():
            if not value:
                continue
            values = _split_multi_value(value)
            if 1 <= len(values) <= 6 and len(value) <= 120:
                for item in values:
                    counters[key.strip()][item] += 1
    findings: list[str] = []
    sample_size = max(1, len(rows))
    for key, counter in counters.items():
        if not key or not counter:
            continue
        top, count = counter.most_common(1)[0]
        if count >= 2 or len(rows) <= 5:
            findings.append(f"{key}：选择“{top}”的样本最多，占 {count}/{sample_size}。")
        if len(findings) >= limit:
            break
    return findings


def _open_text_themes(rows: list[dict[str, str]], limit: int = 5) -> list[str]:
    texts: list[str] = []
    for row in rows:
        for value in row.values():
            if len(value) >= 12:
                texts.append(value)
    keywords = ["价格", "正品", "售后", "物流", "信任", "安全", "款式", "社区", "稀缺", "体验", "质量", "交易"]
    counter = Counter()
    for text in texts:
        for keyword in keywords:
            if keyword in text:
                counter[keyword] += 1
    themes = [f"开放题中多次出现“{keyword}”相关表达，出现 {count} 次。" for keyword, count in counter.most_common(limit)]
    if not themes and texts:
        themes.append(f"开放题共收集 {len(texts)} 条较长反馈，建议人工复核高频痛点和原始表述。")
    return themes[:limit]


def build_survey_appendix_markdown(result: SurveyAnalysisResult) -> str:
    lines = [
        "## 用户问卷补充分析",
        "",
        "### 样本概况",
        f"- 本次问卷分析纳入 {result.sample_size} 份有效样本。",
        f"- {result.summary}" if result.summary else "",
        "",
        "### 关键发现",
        *[f"- {item}" for item in result.key_findings],
        "",
        "### 竞品相关洞察",
        *[f"- {item}" for item in result.competitor_insights],
        "",
        "### 与原报告结论的一致与冲突",
        *[f"- {item}" for item in result.segment_insights],
        "",
        "### 产品建议补充",
        *[f"- {item}" for item in result.open_text_themes],
        "",
        "### 样本局限",
        *[f"- {item}" for item in result.limitations],
    ]
    return "\n".join(line for line in lines if line is not None).strip() + "\n"


async def analyze_survey_responses(
    model,
    task_id: str,
    request: CreateTaskRequest,
    survey_design: SurveyDesign | None,
    raw_responses: str,
) -> SurveyAnalysisResult:
    rows = parse_survey_rows(raw_responses)
    if not rows:
        raise ValueError("No valid survey rows found. Please provide CSV, JSON array, or table text with headers.")
    sample_size = len(rows)
    findings = _top_choice_findings(rows)
    themes = _open_text_themes(rows)
    competitor_names = []
    if survey_design:
        for question in survey_design.questions:
            competitor_names.extend(question.mapped_competitors)
    competitor_names.extend(request.competitors)
    competitor_names = list(dict.fromkeys(name for name in competitor_names if name))
    competitor_insights = []
    raw_text = json.dumps(rows, ensure_ascii=False)
    for name in competitor_names[:6]:
        mentions = raw_text.count(name)
        if mentions:
            competitor_insights.append(f"{name} 在问卷结果中被提及 {mentions} 次，可作为认知度或偏好线索继续复核。")
    if not competitor_insights:
        competitor_insights.append("当前数据中未形成明确的竞品提及差异，建议在后续问卷中增加竞品认知和偏好排序题。")

    result = SurveyAnalysisResult(
        task_id=task_id,
        survey_design_id=survey_design.id if survey_design else None,
        sample_size=sample_size,
        summary=f"本次上传数据包含 {sample_size} 行有效答卷，分析结果用于补充一手用户研究证据。",
        key_findings=findings or ["样本规模有限，暂未识别出稳定的高频选项差异。"],
        competitor_insights=competitor_insights,
        segment_insights=[
            "问卷结论可用于验证公开资料中的用户需求判断，但不能替代公开来源证据。",
            "如样本来源集中在特定社群，应优先代表核心用户，而非整体市场。",
        ],
        open_text_themes=themes or ["开放反馈不足，建议补充更多文本题回答以识别具体机会点。"],
        limitations=[
            "问卷样本由用户上传，系统无法验证样本来源和答题真实性。",
            "若样本量低于 50，结论应作为方向性线索，而非确定性市场判断。",
            "不同竞品用户样本量不均衡时，不宜直接比较满意度或偏好强弱。",
        ],
    )
    result.appendix_markdown = build_survey_appendix_markdown(result)

    if model and request.llm_failure_policy != "continue_without_llm":
        context = {
            "request": request.model_dump(mode="json"),
            "survey_design": survey_design.model_dump(mode="json") if survey_design else None,
            "rule_result": result.model_dump(mode="json"),
            "rows_preview": rows[:50],
        }
        try:
            llm_result = await retry_structured_ainvoke(
                model,
                SurveyAnalysisResult,
                [
                    ("system", survey_analysis_system_prompt),
                    ("user", survey_analysis_user_prompt.format(context=json.dumps(context, ensure_ascii=False, default=str))),
                ],
            )
            llm_result.task_id = task_id
            llm_result.survey_design_id = result.survey_design_id
            llm_result.sample_size = sample_size
            if not llm_result.appendix_markdown:
                llm_result.appendix_markdown = build_survey_appendix_markdown(llm_result)
            return llm_result
        except Exception:
            return result
    return result


def append_survey_analysis_to_markdown(markdown: str, appendix_markdown: str) -> str:
    marker = "## 用户问卷补充分析"
    base = (markdown or "").strip()
    appendix = appendix_markdown.strip()
    if marker in base:
        before = base.split(marker, 1)[0].rstrip()
        return f"{before}\n\n{appendix}\n"
    return f"{base}\n\n{appendix}\n" if base else f"{appendix}\n"
