import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Download,
  History,
  Image as ImageIcon,
  Loader2,
  Menu,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Square,
  Send,
  Sparkles,
  X
} from "lucide-react";
import { API_BASE, cancelTask, createTask, eventsUrl, exportUrl, getSearchRun, getSearchRuns, getTask, listTasks, resumeLatestTask, submitClarification, submitDecision } from "./api";
import type {
  AgentRunEvent,
  AnalysisPurpose,
  AnalysisTask,
  CandidateQAChangeLog,
  CompetitorReport,
  DeepSearchBackend,
  EvidenceItem,
  SearchMethod,
  SearchRunDetail,
  SearchRunSummary
} from "./types";

const nodeLabels: Record<string, string> = {
  workflow: "任务流",
  perception_agent: "需求理解",
  review_agent: "信息澄清",
  candidate_discovery_agent: "竞品发现",
  ranking_agent: "竞品筛选",
  planner_agent: "分析规划",
  search_planner_agent: "搜索规划",
  source_collector_agent: "来源采集",
  tool_executor_agent: "工具执行",
  targeted_collector_agent: "定向采集",
  analyst_agent: "分析推理",
  writer_agent: "报告生成",
  qa_agent: "质量检查"
};

const statusLabels: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  waiting: "等待中",
  waiting_clarification: "待补充",
  completed: "已完成",
  failed: "失败",
  revision: "返工"
};

const purposeOptions: Array<{ value: AnalysisPurpose; label: string; hint: string }> = [
  { value: "decision_support", label: "决策支持", hint: "判断是否进入、如何定位、怎么做取舍" },
  { value: "learning", label: "学习借鉴", hint: "拆解体验、功能路径和可复用做法" },
  { value: "market_warning", label: "市场预警", hint: "识别黑马、跨界威胁和替代品" }
];

type InstructionMode = "weak" | "strong";
type SearchMethodPreset = "default" | "all" | SearchMethod;

const defaultSearchMethods: SearchMethod[] = ["duckduckgo_text", "duckduckgo_news", "tavily"];
const allSearchMethods: SearchMethod[] = [
  "duckduckgo_text",
  "duckduckgo_news",
  "tavily",
  "brave_web",
  "searchapi_google",
  "searchapi_baidu",
  "searchapi_duckduckgo",
  "serpapi_google",
  "serpapi_baidu"
];

const searchMethodOptions: Array<{ value: SearchMethodPreset; label: string }> = [
  { value: "default", label: "默认组合" },
  { value: "all", label: "全部可用搜索源" },
  { value: "duckduckgo_text", label: "DuckDuckGo Text" },
  { value: "duckduckgo_news", label: "DuckDuckGo News" },
  { value: "tavily", label: "Tavily" },
  { value: "brave_web", label: "Brave Web" },
  { value: "searchapi_google", label: "SearchAPI Google" },
  { value: "searchapi_baidu", label: "SearchAPI Baidu" },
  { value: "searchapi_duckduckgo", label: "SearchAPI DuckDuckGo" },
  { value: "serpapi_google", label: "SerpAPI Google" },
  { value: "serpapi_baidu", label: "SerpAPI Baidu" }
];

const deepSearchOptions: Array<{ value: DeepSearchBackend; label: string }> = [
  { value: "none", label: "不深挖" },
  { value: "local_pagination", label: "本地分页" },
  { value: "local_relevant_links", label: "本地相关链接" },
  { value: "local_all_links", label: "本地全部链接" },
  { value: "tavily", label: "Tavily 深挖" },
  { value: "local_relevant_links_tavily", label: "本地相关链接 + Tavily" }
];

function methodsForPreset(preset: SearchMethodPreset): SearchMethod[] {
  if (preset === "default") return defaultSearchMethods;
  if (preset === "all") return allSearchMethods;
  return [preset];
}

interface ComposerPayload {
  targetProduct: string;
  analysisPurpose: AnalysisPurpose | "";
  ourProduct: string;
  market: string;
  productCategory: string;
  geography: string;
  companySize: string;
  competitors: string;
  urls: string;
  notes: string;
  instructionMode: InstructionMode;
  enableScreenshots: boolean;
  enableReadabilityExtraction: boolean;
  searchMethodPreset: SearchMethodPreset;
  deepSearchBackend: DeepSearchBackend;
  candidateTargetCount: number;
}

function splitLines(value: string) {
  return value
    .split(/\n|,|，/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function unique(items: string[]) {
  return Array.from(new Set(items.map((item) => item.trim()).filter(Boolean)));
}

function eventText(event: AgentRunEvent) {
  return localizeDisplayText(event.message || event.output_summary || event.input_summary || "节点状态已更新");
}

function eventStageKey(event: AgentRunEvent) {
  const key = event.details?.ui_stage_key;
  return typeof key === "string" && key.trim() ? key : event.node;
}

function eventTitle(event: AgentRunEvent) {
  const title = event.details?.ui_title;
  return typeof title === "string" && title.trim() ? title : nodeLabels[event.node] ?? event.node;
}

function compactEvents(events: AgentRunEvent[]) {
  const order: string[] = [];
  const byKey = new Map<string, AgentRunEvent>();
  for (const event of events) {
    const key = eventStageKey(event);
    if (!byKey.has(key)) order.push(key);
    byKey.set(key, event);
  }
  return order.map((key) => byKey.get(key)).filter(Boolean) as AgentRunEvent[];
}

function eventDisplayMarkdown(event: AgentRunEvent) {
  const value = event.details?.display_markdown;
  return typeof value === "string" ? value.trim() : "";
}

function isLlmDecisionWaitEvent(event: AgentRunEvent) {
  const message = event.message || "";
  const question = event.details?.decision && typeof event.details.decision === "object"
    ? String((event.details.decision as Record<string, unknown>).question || "")
    : "";
  return event.status === "waiting" && (message.includes("大模型连续调用失败") || question.includes("大模型连续调用失败"));
}

const displayValueLabels: Record<string, string> = {
  concept: "概念期",
  development: "开发期",
  launched: "已上线",
  decision_support: "决策支持",
  learning: "学习借鉴",
  market_warning: "市场预警",
  head_direct: "头部直接竞品",
  challenger: "挑战者",
  indirect_cross: "间接/跨界竞品",
  potential_substitute: "潜在替代品",
  source_segment: "网页片段",
  "source segment": "网页片段",
  source_chunk: "网页片段",
  extracted_fact: "抽取事实",
  complete: "已完成",
  ready: "已就绪",
  writing: "正在写入",
  unavailable: "暂不可用"
};

const cardFieldLabels: Record<string, string> = {
  website: "官网/渠道入口",
  positioning: "定位",
  target_users: "目标用户",
  core_features: "核心功能",
  key_parameters: "关键参数",
  pricing: "价格/收费",
  business_model: "商业模式",
  channels: "渠道",
  growth_signals: "增长信号",
  funding_or_org_signals: "融资/组织信号",
  technology_or_product_features: "技术/产品特征",
  differentiation: "差异化",
  user_feedback: "用户反馈",
  risks: "风险",
  "basic_info.industry_scope": "行业范围",
  "basic_info.main_users": "主要用户",
  "basic_info.core_scenarios": "核心场景",
  "basic_info.buyer_behavior_or_preferences": "购买行为/偏好",
  "market_situation.market_size": "市场规模",
  "market_situation.growth_trend": "增长趋势",
  "market_situation.willingness_to_pay": "付费意愿",
  "market_situation.investment_environment": "投融资/入局环境",
  "competition_situation.competition_intensity": "竞争强度",
  "external_environment.regulation_or_laws": "政策/法规环境",
  "external_environment.technology_trends": "技术趋势"
};

function displayLabel(value: unknown, fallback = "") {
  if (typeof value !== "string" && typeof value !== "number") return fallback;
  const text = String(value).trim();
  if (!text) return fallback;
  return displayValueLabels[text] ?? cardFieldLabels[text] ?? text;
}

function localizeDisplayText(value: string) {
  let text = value.replace(/\b(concept|development|launched|decision_support|learning|market_warning|head_direct|challenger|indirect_cross|potential_substitute|source_segment|source segment|source_chunk|extracted_fact)\b/g, (match) => displayValueLabels[match] ?? match);
  for (const [field, label] of Object.entries(cardFieldLabels)) {
    text = text.split(field).join(label);
  }
  return text;
}

function normalizeCandidateCountHeadings(content: string) {
  const lockedLine = content.match(/锁定竞品阵型\s*[:：]\s*([^\r\n]+)/);
  if (!lockedLine) return content;
  const count = lockedLine[1]
    .split(/[、,，/]/)
    .map((item) => item.trim())
    .filter(Boolean).length;
  if (!count) return content;
  return content.replace(/确定的\s*\d+\s*个竞品/g, `确定的 ${count} 个竞品`);
}

function StatusPill({ status }: { status: string }) {
  const overrides: Record<string, string> = {
    cancelling: "正在中止",
    waiting_decision: "等待决策",
    cancelled: "已中止"
  };
  return <span className={`status-pill status-${status}`}>{overrides[status] ?? statusLabels[status] ?? status}</span>;
}

function formatTime(value: string) {
  return new Date(value).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function WelcomeHeadline() {
  return (
    <section className="welcome" aria-label="开始分析">
      <h1>你好， 想分析哪个产品？</h1>
    </section>
  );
}

function Composer({
  onSubmit,
  working,
  cancelling,
  onCancel
}: {
  onSubmit: (payload: ComposerPayload) => Promise<void>;
  working: boolean;
  cancelling: boolean;
  onCancel: () => Promise<void>;
}) {
  const [targetProduct, setTargetProduct] = useState("");
  const [analysisPurpose, setAnalysisPurpose] = useState<AnalysisPurpose | "">("");
  const [expanded, setExpanded] = useState(false);
  const [ourProduct, setOurProduct] = useState("");
  const [market, setMarket] = useState("");
  const [productCategory, setProductCategory] = useState("");
  const [geography, setGeography] = useState("");
  const [companySize, setCompanySize] = useState("");
  const [competitors, setCompetitors] = useState("");
  const [urls, setUrls] = useState("");
  const [notes, setNotes] = useState("");
  const [instructionMode, setInstructionMode] = useState<InstructionMode>("weak");
  const [enableScreenshots, setEnableScreenshots] = useState(false);
  const [enableReadabilityExtraction, setEnableReadabilityExtraction] = useState(false);
  const [searchMethodPreset, setSearchMethodPreset] = useState<SearchMethodPreset>("default");
  const [deepSearchBackend, setDeepSearchBackend] = useState<DeepSearchBackend>("none");
  const [candidateTargetCount, setCandidateTargetCount] = useState(12);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (working) {
      await onCancel();
      return;
    }
    if (!targetProduct.trim()) {
      setError("请先输入必须分析的产品名。");
      return;
    }
    setError(null);
    await onSubmit({
      targetProduct,
      analysisPurpose,
      ourProduct,
      market,
      productCategory,
      geography,
      companySize,
      competitors,
      urls,
      notes,
      instructionMode,
      enableScreenshots,
      enableReadabilityExtraction,
      searchMethodPreset,
      deepSearchBackend,
      candidateTargetCount
    });
    setTargetProduct("");
    setAnalysisPurpose("");
    setOurProduct("");
    setMarket("");
    setProductCategory("");
    setGeography("");
    setCompanySize("");
    setCompetitors("");
    setUrls("");
    setNotes("");
    setInstructionMode("weak");
    setEnableScreenshots(false);
    setEnableReadabilityExtraction(false);
    setSearchMethodPreset("default");
    setDeepSearchBackend("none");
    setCandidateTargetCount(12);
    setExpanded(false);
  }

  return (
    <form className={`composer home-composer ${expanded ? "expanded" : ""}`} onSubmit={submit}>
      <div className="main-input-row">
        <button type="button" className="add-button" onClick={() => setExpanded((value) => !value)} title="展开补充材料">
          <ChevronDown size={22} className={expanded ? "rotated" : ""} />
        </button>
        <input
          id="target-product"
          name="target_product"
          value={targetProduct}
          onChange={(event) => setTargetProduct(event.target.value)}
          placeholder="必须输入产品名，例如：二次元手办交易平台"
          aria-label="产品名"
        />
        <button type="submit" className="send-button" disabled={cancelling || (!working && !targetProduct.trim())} title={working ? "中止运行" : "开始分析"}>
          {working ? (cancelling ? <Loader2 className="spin" size={18} /> : <Square size={18} />) : <Send size={18} />}
        </button>
      </div>

      <div className="purpose-row" aria-label="分析目的">
        <div className="purpose-options">
          {purposeOptions.map((option) => (
            <button
              type="button"
              key={option.value}
              className={analysisPurpose === option.value ? "selected" : ""}
              onClick={() => setAnalysisPurpose(option.value)}
              title={option.hint}
            >
              <span className="optional-marker" aria-hidden="true" />
              {option.label}
            </button>
          ))}
        </div>
        <span className="purpose-divider" aria-hidden="true" />
        <label className="toggle-button screenshot-toggle" title="启用网页截图证据" aria-label="启用网页截图证据">
          <input id="enable-screenshots" name="enable_screenshots" type="checkbox" checked={enableScreenshots} onChange={(event) => setEnableScreenshots(event.target.checked)} />
          <ImageIcon size={18} />
          <span>启用网页截图证据</span>
        </label>
      </div>

      <div className={`composer-details ${expanded ? "open" : ""}`} aria-hidden={!expanded}>
          <input id="our-product" name="our_product" value={ourProduct} onChange={(event) => setOurProduct(event.target.value)} placeholder="我方产品" />
          <input id="market" name="market" value={market} onChange={(event) => setMarket(event.target.value)} placeholder="市场 / 赛道" />
          <input id="product-category" name="product_category" value={productCategory} onChange={(event) => setProductCategory(event.target.value)} placeholder="产品类别" />
          <input id="geography" name="geography" value={geography} onChange={(event) => setGeography(event.target.value)} placeholder="地理范围" />
          <input id="company-size" name="company_size" value={companySize} onChange={(event) => setCompanySize(event.target.value)} placeholder="客户规模" />
          <label className="select-field" htmlFor="search-method-preset">
            <span>搜索方式</span>
            <select id="search-method-preset" name="search_method_preset" value={searchMethodPreset} onChange={(event) => setSearchMethodPreset(event.target.value as SearchMethodPreset)}>
              {searchMethodOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="select-field" htmlFor="deep-search-backend">
            <span>深度搜索</span>
            <select id="deep-search-backend" name="deep_search_backend" value={deepSearchBackend} onChange={(event) => setDeepSearchBackend(event.target.value as DeepSearchBackend)}>
              {deepSearchOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="toggle-button screenshot-toggle" title="启用正文抽取以减少广告和导航文本" aria-label="启用正文抽取">
            <input id="enable-readability-extraction" name="enable_readability_extraction" type="checkbox" checked={enableReadabilityExtraction} onChange={(event) => setEnableReadabilityExtraction(event.target.checked)} />
            <Sparkles size={18} />
            <span>正文抽取</span>
          </label>
          <label className="select-field" htmlFor="candidate-target-count">
            <span>候选竞品数量</span>
            <input
              id="candidate-target-count"
              name="candidate_target_count"
              type="number"
              min={1}
              max={30}
              value={candidateTargetCount}
              onChange={(event) => setCandidateTargetCount(Math.max(1, Math.min(30, Number(event.target.value) || 12)))}
            />
          </label>
          <textarea id="known-competitors" name="competitors" value={competitors} onChange={(event) => setCompetitors(event.target.value)} placeholder="已知竞品，每行一个" rows={2} />
          <textarea id="supplement-urls" name="urls" value={urls} onChange={(event) => setUrls(event.target.value)} placeholder="补充 URL，每行一个" rows={2} />
          <textarea id="analysis-notes" name="notes" value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="分析备注：重点关注功能、定价、增长、渠道等" rows={2} />

          <div className="instruction-row">
            <span>补充材料约束</span>
            <label className={instructionMode === "weak" ? "selected" : ""}>
              <input id="instruction-weak" name="instruction_mode" type="radio" checked={instructionMode === "weak"} onChange={() => setInstructionMode("weak")} />
              弱指令
            </label>
            <label className={instructionMode === "strong" ? "selected" : ""}>
              <input id="instruction-strong" name="instruction_mode" type="radio" checked={instructionMode === "strong"} onChange={() => setInstructionMode("strong")} />
              强指令
            </label>
          </div>
        </div>

      {error && <p className="form-error">{error}</p>}
    </form>
  );
}

function ClarificationCard({ task, onSubmitted }: { task: AnalysisTask; onSubmitted: (task: AnalysisTask) => void }) {
  const clarification = task.pending_clarification;
  const [custom, setCustom] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!clarification) return null;

  async function choose(answer: string, selected?: string) {
    setSubmitting(true);
    setError(null);
    try {
      onSubmitted(await submitClarification(task.id, answer, selected));
    } catch (err) {
      setError(err instanceof Error ? err.message : "提交澄清失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <article className="assistant-card attention">
      <div className="card-head">
        <AlertCircle size={18} />
        <strong>需要补充一个关键信息</strong>
      </div>
      <p>{clarification.question}</p>
      <div className="choice-grid">
        {clarification.options.map((option) => (
          <button type="button" key={option.value} onClick={() => void choose(option.label, option.value)} disabled={submitting}>
            <strong>{option.label}</strong>
            <span>{option.description}</span>
          </button>
        ))}
      </div>
      <form
        className="inline-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (custom.trim()) void choose(custom.trim());
        }}
      >
        <input id="clarification-answer" name="clarification_answer" value={custom} onChange={(event) => setCustom(event.target.value)} placeholder="也可以直接输入你的补充说明" />
        <button type="submit" disabled={submitting || !custom.trim()}>
          {submitting ? <Loader2 className="spin" size={16} /> : "提交"}
        </button>
      </form>
      {error && <p className="form-error">{error}</p>}
    </article>
  );
}

function DecisionCard({ task, onSubmitted }: { task: AnalysisTask; onSubmitted: (task: AnalysisTask) => void }) {
  const decision = task.pending_decision;
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!decision) return null;

  async function choose(action: "retry_llm" | "continue_without_llm" | "cancel") {
    setSubmitting(true);
    setError(null);
    try {
      onSubmitted(await submitDecision(task.id, action));
    } catch (err) {
      setError(err instanceof Error ? err.message : "提交处理选择失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <article className="assistant-card attention">
      <div className="card-head">
        <AlertCircle size={18} />
        <strong>大模型调用失败</strong>
      </div>
      <p>{decision.question}</p>
      {decision.reason && <p className="form-error">{decision.reason}</p>}
      <div className="choice-grid">
        {decision.options.map((option) => (
          <button type="button" key={option.value} onClick={() => void choose(option.value)} disabled={submitting}>
            <strong>{option.label}</strong>
            <span>{option.description}</span>
          </button>
        ))}
      </div>
      {error && <p className="form-error">{error}</p>}
    </article>
  );
}

function FormattedStageContent({ content }: { content: string }) {
  const lines = normalizeCandidateCountHeadings(content).split(/\r?\n/);
  return (
    <div className="event-detail-content">
      {lines.map((line, index) => {
        const trimmed = localizeDisplayText(line.trim());
        if (!trimmed) return <div key={index} className="detail-gap" />;
        if (/^#{1,6}\s*/.test(trimmed)) return <h4 key={index}>{trimmed.replace(/^#{1,6}\s*/, "")}</h4>;
        if (/^\d+\.\s/.test(trimmed)) return <p key={index} className="detail-number">{trimmed}</p>;
        if (line.startsWith("   - ")) return <p key={index} className="detail-subbullet">{trimmed.slice(2)}</p>;
        if (trimmed.startsWith("- ")) return <p key={index} className="detail-bullet">{trimmed.slice(2)}</p>;
        return <p key={index}>{trimmed}</p>;
      })}
    </div>
  );
}

function textField(item: Record<string, unknown>, keys: string[], fallback = "") {
  for (const key of keys) {
    const value = item[key];
    if (typeof value === "string" && value.trim()) return value;
    if (typeof value === "number") return String(value);
  }
  return fallback;
}

function recordArray(value: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && !Array.isArray(item));
}

function candidateName(item: Record<string, unknown>) {
  return textField(item, ["name", "competitor"], "未命名候选");
}

function dedupeCandidates(items: Array<Record<string, unknown>>) {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = candidateName(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function finalCandidateList(details?: AgentRunEvent["details"]) {
  const selected = recordArray(details?.selected);
  const selectedCompetitors = recordArray(details?.selected_competitors);
  const nestedSelected = details?.selected_competitor_set && typeof details.selected_competitor_set === "object"
    ? recordArray((details.selected_competitor_set as Record<string, unknown>).selected)
    : [];
  const candidates = recordArray(details?.candidates);
  return dedupeCandidates(selected.length ? selected : selectedCompetitors.length ? selectedCompetitors : nestedSelected.length ? nestedSelected : candidates);
}

function CandidateQAChangePanel({
  candidates,
  changeLog
}: {
  candidates?: Array<Record<string, unknown>>;
  changeLog?: CandidateQAChangeLog;
}) {
  const hasChanges = Boolean(
    changeLog &&
      (changeLog.removed_candidates.length ||
        changeLog.merged_candidates.length ||
        changeLog.type_corrections.length ||
        changeLog.unverified_candidates.length ||
        changeLog.qa_notes.length)
  );
  if (!candidates?.length && !hasChanges) return null;
  return (
    <div className="event-detail-content">
      {candidates?.length ? (
        <>
          <h4>确定的 {candidates.length} 个竞品</h4>
          {candidates.map((item, index) => (
            <p key={`${textField(item, ["name"], "candidate")}-${index}`} className="detail-number">
              {index + 1}. {candidateName(item)}
              {textField(item, ["competitor_type"]) ? ` · ${displayLabel(textField(item, ["competitor_type"]))}` : ""}
              {textField(item, ["selection_reason", "description"]) ? ` · ${textField(item, ["selection_reason", "description"])}` : ""}
            </p>
          ))}
        </>
      ) : null}
      {changeLog?.removed_candidates.length ? (
        <>
          <h4>QA 删除</h4>
          {changeLog.removed_candidates.map((item, index) => (
            <p key={`removed-${index}`} className="detail-bullet">
              {textField(item, ["name"], "未命名候选")}：{textField(item, ["reason"], "未通过存在性验证")}
            </p>
          ))}
        </>
      ) : null}
      {changeLog?.merged_candidates.length ? (
        <>
          <h4>QA 合并</h4>
          {changeLog.merged_candidates.map((item, index) => (
            <p key={`merged-${index}`} className="detail-bullet">
              {textField(item, ["from_name", "name"], "原候选")} → {textField(item, ["to_name", "merge_target"], "合并目标")}：{textField(item, ["reason"], "别名或同一产品")}
            </p>
          ))}
        </>
      ) : null}
      {changeLog?.type_corrections.length ? (
        <>
          <h4>类别修正</h4>
          {changeLog.type_corrections.map((item, index) => (
            <p key={`type-${index}`} className="detail-bullet">
              {textField(item, ["name"], "候选")}：{displayLabel(textField(item, ["from_type"], "原类别"))} → {displayLabel(textField(item, ["to_type"], "新类别"))}；{textField(item, ["reason"], "QA 修正")}
            </p>
          ))}
        </>
      ) : null}
      {changeLog?.unverified_candidates.length ? (
        <>
          <h4>未验证保留</h4>
          {changeLog.unverified_candidates.map((item, index) => (
            <p key={`unverified-${index}`} className="detail-bullet">
              {textField(item, ["name"], "候选")}：{textField(item, ["reason"], "验证失败但保留")}
            </p>
          ))}
        </>
      ) : null}
      {changeLog?.qa_notes.length ? (
        <>
          <h4>QA 备注</h4>
          {changeLog.qa_notes.map((note, index) => (
            <p key={`note-${index}`} className="detail-bullet">{note}</p>
          ))}
        </>
      ) : null}
    </div>
  );
}

function EventStream({ events, taskStatus }: { events: AgentRunEvent[]; taskStatus?: AnalysisTask["status"] }) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const visibleEvents = useMemo(
    () => compactEvents(events.filter((event) => taskStatus === "waiting_decision" || !isLlmDecisionWaitEvent(event))),
    [events, taskStatus]
  );
  if (!events.length) return null;
  return (
    <section className="stream">
      {visibleEvents.map((event) => {
        const display = eventDisplayMarkdown(event);
        const changeLog = event.details?.candidate_qa_change_log;
        const candidates = finalCandidateList(event.details);
        const stageKey = eventStageKey(event);
        const showInlineProgress = event.status === "running" && Boolean(display);
        const canExpand = event.status === "completed" && (Boolean(display) || Boolean(changeLog));
        const isExpanded = showInlineProgress || Boolean(expanded[stageKey]);
        return (
        <article className={`assistant-card event-card ${canExpand ? "has-detail" : ""}`} key={`${stageKey}-${event.id}`}>
          <div className="event-dot" />
          <div className="event-body">
            <div className="card-head">
              <strong>{eventTitle(event)}</strong>
              <StatusPill status={event.status} />
              {canExpand && (
                <button
                  type="button"
                  className="event-detail-toggle"
                  onClick={() => setExpanded((current) => ({ ...current, [stageKey]: !current[stageKey] }))}
                  title={isExpanded ? "收起阶段内容" : "展开阶段内容"}
                  aria-expanded={isExpanded}
                >
                  <ChevronDown size={16} className={isExpanded ? "rotated" : ""} />
                </button>
              )}
            </div>
            <p>{eventText(event)}</p>
            <div className="meta-row">
              <span>{formatTime(event.created_at)}</span>
              {event.duration_ms != null && <span>{event.duration_ms} ms</span>}
              {event.token_estimate != null && <span>约 {event.token_estimate} tokens</span>}
            </div>
            {event.error && <p className="form-error">{event.error}</p>}
            {(showInlineProgress || (canExpand && isExpanded)) && <FormattedStageContent content={display} />}
            {canExpand && isExpanded && <CandidateQAChangePanel candidates={candidates} changeLog={changeLog} />}
          </div>
        </article>
        );
      })}
    </section>
  );
}

function EvidenceButton({
  id,
  evidenceMap,
  onSelect
}: {
  id: string;
  evidenceMap: Map<string, EvidenceItem>;
  onSelect: (source: EvidenceItem) => void;
}) {
  const item = evidenceMap.get(id);
  if (!item) return <span className="evidence-chip missing">{id}</span>;
  return (
    <button type="button" className="evidence-chip" onClick={() => onSelect(item)} title={item.title}>
      {id}
    </button>
  );
}

function CleanCardFields({ fields }: { fields: Record<string, string> }) {
  const entries = Object.entries(fields).filter(([, value]) => String(value || "").trim());
  if (!entries.length) return <p className="muted">暂无已填写字段。</p>;
  return (
    <dl className="detail-field-list">
      {entries.slice(0, 16).map(([field, value]) => (
        <div key={field}>
          <dt>{displayLabel(field)}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function DetailCardSection({ report }: { report: CompetitorReport }) {
  const competitorCards = report.competitor_detail_cards_clean ?? [];
  const industryCard = report.industry_card_clean ?? null;
  if (!competitorCards.length && !industryCard) return null;
  return (
    <section className="assistant-card">
      <div className="card-head">
        <CheckCircle2 size={18} />
        <strong>资料卡</strong>
      </div>
      <div className="detail-card-grid">
        {competitorCards.map((card) => (
          <article className="compact-card detail-card" key={card.competitor}>
            <div className="compact-head">
              <strong>{card.competitor}</strong>
              <span>{displayLabel(card.competitor_type, "竞品")}</span>
            </div>
            <CleanCardFields fields={card.fields} />
          </article>
        ))}
        {industryCard && (
          <article className="compact-card detail-card">
            <div className="compact-head">
              <strong>行业资料卡</strong>
              <span>行业</span>
            </div>
            <CleanCardFields fields={industryCard.fields} />
          </article>
        )}
      </div>
    </section>
  );
}

function ObjectProgressPanel({ report }: { report: CompetitorReport }) {
  const items = report.object_progress ?? [];
  if (!items.length) return null;
  return (
    <section className="assistant-card object-progress">
      <div className="card-head">
        <BarChart3 size={18} />
        <strong>Object progress</strong>
      </div>
      <div className="object-progress-grid">
        {items.map((item) => (
          <article className="object-progress-item" key={item.object_id}>
            <div>
              <strong>{item.object_name}</strong>
              <span>{item.status}</span>
            </div>
            <dl>
              <div>
                <dt>sources</dt>
                <dd>{item.searched_sources}</dd>
              </div>
              <div>
                <dt>chunks</dt>
                <dd>{item.analyzed_chunks}</dd>
              </div>
              <div>
                <dt>fields</dt>
                <dd>{item.merged_fields}</dd>
              </div>
            </dl>
            {item.error && <p className="form-error">{item.error}</p>}
          </article>
        ))}
      </div>
    </section>
  );
}

function ReportView({ task, report, onSelectSource }: { task: AnalysisTask; report: CompetitorReport; onSelectSource: (source: EvidenceItem) => void }) {
  const evidenceMap = useMemo(() => new Map(report.evidence.map((item) => [item.id, item])), [report.evidence]);
  const latestQa = report.qa_history[report.qa_history.length - 1];
  return (
    <section className="report-view">
      <article className="assistant-card report-hero">
        <div className="report-title">
          <span>分析报告</span>
          <h2>{report.target_product} 竞品分析</h2>
          <p>{report.executive_summary}</p>
        </div>
        <div className="download-row">
          {(["md", "json", "pdf", "docx"] as const).map((format) => (
            <a key={format} href={exportUrl(task.id, format)} title={`导出 ${format.toUpperCase()}`}>
              <Download size={16} />
              {format.toUpperCase()}
            </a>
          ))}
        </div>
      </article>

      {report.risk_notice && (
        <article className="assistant-card attention">
          <div className="card-head">
            <AlertCircle size={18} />
            <strong>风险提示</strong>
          </div>
          <p>{report.risk_notice}</p>
        </article>
      )}

      <div className="metric-strip">
        <article>
          <span>竞品数量</span>
          <strong>{report.competitors.length}</strong>
        </article>
        <article>
          <span>关键发现</span>
          <strong>{report.findings.length}</strong>
        </article>
        <article>
          <span>证据来源</span>
          <strong>{report.evidence.length}</strong>
        </article>
        <article>
          <span>质量检查</span>
          <strong>{latestQa?.passed ? "通过" : "待确认"}</strong>
        </article>
      </div>

      <DetailCardSection report={report} />
      <ObjectProgressPanel report={report} />

      <section className="assistant-card">
        <div className="card-head">
          <BarChart3 size={18} />
          <strong>竞品画像</strong>
        </div>
        <div className="competitor-grid">
          {report.competitors.map((item) => (
            <article className="compact-card" key={item.name}>
              <div className="compact-head">
                <strong>{item.name}</strong>
                <span>{item.score != null ? item.score.toFixed(2) : "待评分"}</span>
              </div>
              <p>{item.selection_reason || item.positioning}</p>
              <dl>
                <dt>定位</dt>
                <dd>{item.positioning}</dd>
                <dt>商业模式</dt>
                <dd>{item.business_model}</dd>
              </dl>
              <div className="evidence-row">
                {item.evidence_ids.map((id) => (
                  <EvidenceButton key={id} id={id} evidenceMap={evidenceMap} onSelect={onSelectSource} />
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="assistant-card">
        <div className="card-head">
          <Sparkles size={18} />
          <strong>关键发现</strong>
        </div>
        <div className="finding-list">
          {report.findings.map((finding) => (
            <article className="compact-card" key={finding.id}>
              <div className="compact-head">
                <strong>{finding.title}</strong>
                <span>{finding.impact}</span>
              </div>
              <p>{finding.conclusion}</p>
              <div className="evidence-row">
                {finding.evidence_ids.map((id) => (
                  <EvidenceButton key={id} id={id} evidenceMap={evidenceMap} onSelect={onSelectSource} />
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="assistant-card">
        <div className="card-head">
          <CheckCircle2 size={18} />
          <strong>行动建议</strong>
        </div>
        <ul className="recommendations">
          {report.recommendations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </section>
  );
}

function SearchRunLibrary({ taskId }: { taskId: string }) {
  const [runs, setRuns] = useState<SearchRunSummary[]>([]);
  const [expandedRunKey, setExpandedRunKey] = useState<string | null>(null);
  const [detailsByRun, setDetailsByRun] = useState<Record<string, SearchRunDetail>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const payload = await getSearchRuns(taskId);
        if (!cancelled) {
          setRuns(payload.runs);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "读取来源库失败");
      }
    }
    void load();
    const timer = window.setInterval(() => void load(), 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [taskId]);

  async function toggleRun(run: SearchRunSummary) {
    if (expandedRunKey === run.run_key) {
      setExpandedRunKey(null);
      return;
    }
    setExpandedRunKey(run.run_key);
    if (detailsByRun[run.run_key]) return;
    try {
      const detail = await getSearchRun(taskId, run.run_key);
      setDetailsByRun((current) => ({ ...current, [run.run_key]: detail }));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "读取搜索详情失败");
    }
  }

  return (
    <div className="source-library">
      {error && <p className="form-error">{error}</p>}
      {!runs.length && <p className="muted">暂无搜索结果，运行开始后会自动刷新。</p>}
      <div className="run-list">
        {runs.map((run) => {
          const open = expandedRunKey === run.run_key;
          const detail = detailsByRun[run.run_key];
          return (
            <article key={run.run_key} className={`run-item ${open ? "open" : ""}`}>
              <button type="button" className="run-row" onClick={() => void toggleRun(run)} aria-expanded={open}>
                <span className="run-title-line">
                  <strong>{run.query || "正在写入搜索结果"}</strong>
                  <ChevronDown size={16} className={open ? "rotated" : ""} />
                </span>
                <span>{purposeLabel(run.purpose)} · {run.methods.join("、") || "未知来源"} · {deepBackendLabel(run.deep_backend)}</span>
                <small>{run.result_count} 条结果 · {run.chunk_count} 个网页片段 · 去重跳过 {run.skipped_duplicate_count} 条</small>
              </button>
              {open && <RunDetailInline run={detail ?? run} loading={!detail} />}
            </article>
          );
        })}
      </div>
    </div>
  );
}

function RunDetailInline({ run, loading }: { run: SearchRunSummary | SearchRunDetail; loading: boolean }) {
  const detail = "results" in run ? run : null;
  return (
    <div className="run-detail">
      <p>{searchStatusLabel(run.status)} · {run.result_count} 条结果 · {run.chunk_count} 个网页片段</p>
      <div className="run-links">
        {run.results_json_path && <a href={`${API_BASE}/data/${run.results_json_path}`} target="_blank" rel="noreferrer">搜索结果文件</a>}
        {run.report_path && <a href={`${API_BASE}/data/${run.report_path}`} target="_blank" rel="noreferrer">搜索报告</a>}
      </div>
      {loading && <p className="muted">正在读取搜索详情...</p>}
      {detail && (
        <div className="result-list">
          {detail.results.slice(0, 20).map((item, index) => {
            const title = String(item.title || item.url || `结果 ${index + 1}`);
            const url = String(item.url || "");
            const screenshot = item.screenshot_path ? `${detail.run_dir}/${String(item.screenshot_path)}` : "";
            const rowError = item.error ? String(item.error) : "";
            return (
              <section key={`${url}-${index}`} className="result-row">
                <strong>{title}</strong>
                {url && <a href={url} target="_blank" rel="noreferrer">{url}</a>}
                {screenshot && <a href={`${API_BASE}/data/${screenshot}`} target="_blank" rel="noreferrer">网页截图</a>}
                {rowError && <small className="form-error">{rowError}</small>}
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}

function purposeLabel(value: string) {
  const labels: Record<string, string> = {
    candidate_discovery: "泛搜索",
    targeted_collection: "定向搜索"
  };
  return labels[value] ?? value;
}

function deepBackendLabel(value: string) {
  const labels: Record<string, string> = {
    none: "不深挖",
    local_pagination: "本地分页",
    local_relevant_links: "本地相关链接",
    local_all_links: "本地全部链接",
    tavily: "Tavily 深挖",
    local_relevant_links_tavily: "本地相关链接 + Tavily"
  };
  return labels[value] ?? value;
}

function searchStatusLabel(value: string) {
  const labels: Record<string, string> = {
    complete: "已完成",
    ready: "已就绪",
    writing: "正在写入",
    unavailable: "暂不可用"
  };
  return labels[value] ?? value;
}

function SourcePanel({
  source,
  taskId,
  libraryOpen,
  onOpenLibrary,
  onClose
}: {
  source: EvidenceItem | null;
  taskId?: string;
  libraryOpen: boolean;
  onOpenLibrary: () => void;
  onClose: () => void;
}) {
  const open = Boolean(source) || libraryOpen;
  return (
    <aside className={`source-panel ${open ? "open" : ""}`} aria-label={open ? "来源面板" : "来源库"}>
      <button
        type="button"
        className="source-panel-toggle"
        onClick={open ? onClose : onOpenLibrary}
        title={open ? "收起来源库" : "展开来源库"}
        aria-label={open ? "收起来源库" : "展开来源库"}
      >
        {open ? <PanelRightClose size={24} /> : <PanelRightOpen size={24} />}
      </button>
      <div className="source-panel-inner">
        <div className="source-head">
          <strong>{libraryOpen ? "来源库" : "来源"}</strong>
          <button type="button" onClick={onClose} title="关闭来源面板">
            <X size={18} />
          </button>
        </div>
        <div className="source-content">
          {libraryOpen && taskId ? (
            <SearchRunLibrary taskId={taskId} />
          ) : source ? (
            <article>
              <span className="source-type">{source.source_type}</span>
              <h3>{source.title}</h3>
              <a href={source.url} target="_blank" rel="noreferrer">
                {source.url}
              </a>
              {source.image_path && <img src={`${API_BASE}/data/${source.image_path}`} alt={source.image_alt ?? source.title} />}
              <p>{source.excerpt}</p>
              <small>可信度 {Math.round(source.credibility * 100)}%</small>
            </article>
          ) : (
            <p className="muted">点击报告中的证据编号查看来源摘要。</p>
          )}
        </div>
      </div>
    </aside>
  );
}

function makeInstructionNote(payload: ComposerPayload) {
  const modeText =
    payload.instructionMode === "strong"
      ? "强指令：补充材料是硬约束，Agent 必须优先按照用户填写的我方产品、赛道、类别、范围、客户规模、竞品和 URL 进行分析；若公开资料冲突，需要显式说明冲突。"
      : "弱指令：补充材料仅作为参考线索，Agent 可以结合公开来源进行修正、扩展或降权。";
  const parts = [
    modeText,
    payload.ourProduct && `我方产品：${payload.ourProduct}`,
    payload.market && `市场/赛道：${payload.market}`,
    payload.productCategory && `产品类别：${payload.productCategory}`,
    payload.geography && `地理范围：${payload.geography}`,
    payload.companySize && `客户规模：${payload.companySize}`,
    payload.notes && `分析备注：${payload.notes}`
  ].filter(Boolean);
  return parts.join("\n");
}

export default function App() {
  const [task, setTask] = useState<AnalysisTask | null>(null);
  const [events, setEvents] = useState<AgentRunEvent[]>([]);
  const [report, setReport] = useState<CompetitorReport | null>(null);
  const [selectedSource, setSelectedSource] = useState<EvidenceItem | null>(null);
  const [sourceLibraryOpen, setSourceLibraryOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [loadingLatestResult, setLoadingLatestResult] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const conversationRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!task) return;
    if (task.status === "completed") return;
    setEvents([]);
    setReport(null);
    setSelectedSource(null);
    setSourceLibraryOpen(false);

    const source = new EventSource(eventsUrl(task.id));
    source.addEventListener("agent-event", (event) => {
      const parsed = JSON.parse((event as MessageEvent).data) as AgentRunEvent;
      setEvents((current) => (current.some((item) => item.id === parsed.id) ? current : [...current, parsed]));
    });
    source.addEventListener("done", async () => {
      source.close();
      const detail = await getTask(task.id);
      setTask(detail.task);
      setEvents(detail.events);
      setReport(detail.report ?? null);
    });
    source.onerror = () => {
      source.close();
      void getTask(task.id).then((detail) => {
        setTask(detail.task);
        setEvents(detail.events);
        setReport(detail.report ?? null);
      });
    };
    return () => source.close();
  }, [task?.id, task?.status]);

  useEffect(() => {
    conversationRef.current?.scrollTo({ top: conversationRef.current.scrollHeight, behavior: "smooth" });
  }, [events.length, report, task?.status]);

  async function startAnalysis(payload: ComposerPayload) {
    setSubmitting(true);
    setError(null);
    try {
      const normalizedPurpose: AnalysisPurpose = payload.analysisPurpose || "decision_support";
      const purposeLabel = purposeOptions.find((item) => item.value === normalizedPurpose)?.label ?? normalizedPurpose;
      const instructionNote = makeInstructionNote(payload);
      const rawDescription = [`目标产品：${payload.targetProduct}`, `分析目的：${purposeLabel}`, instructionNote].filter(Boolean).join("\n");
      const created = await createTask({
        raw_description: rawDescription,
        target_product: payload.targetProduct.trim(),
        analysis_purpose: normalizedPurpose,
        analysis_goal: purposeLabel,
        competitors: unique(splitLines(payload.competitors)),
        urls: unique(splitLines(payload.urls)),
        notes: instructionNote || undefined,
        our_product: payload.ourProduct.trim() || undefined,
        market: payload.market.trim() || undefined,
        product_category: payload.productCategory.trim() || undefined,
        geography: payload.geography.trim() || undefined,
        company_size: payload.companySize.trim() || undefined,
        enable_screenshots: payload.enableScreenshots,
        enable_readability_extraction: payload.enableReadabilityExtraction,
        search_methods: methodsForPreset(payload.searchMethodPreset),
        deep_search_backend: payload.deepSearchBackend,
        candidate_target_count: payload.candidateTargetCount,
        max_revision_loops: 2
      });
      setTask(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建任务失败");
    } finally {
      setSubmitting(false);
    }
  }

  async function stopAnalysis() {
    if (!task || task.status === "cancelling") return;
    setCancelling(true);
    setError(null);
    try {
      setTask(await cancelTask(task.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "中止任务失败");
    } finally {
      setCancelling(false);
    }
  }

  async function resumeLatestAnalysis() {
    setResuming(true);
    setError(null);
    try {
      setTask(await resumeLatestTask());
    } catch (err) {
      setError(err instanceof Error ? err.message : "续跑最新任务失败");
    } finally {
      setResuming(false);
    }
  }

  async function loadLatestCompletedResult() {
    setLoadingLatestResult(true);
    setError(null);
    try {
      const tasks = await listTasks(200);
      const latestCompleted = tasks.find((item) => item.status === "completed");
      if (!latestCompleted) {
        throw new Error("还没有已完成的任务结果可查看");
      }
      const detail = await getTask(latestCompleted.id);
      setTask(detail.task);
      setEvents(detail.events);
      setReport(detail.report ?? null);
      setSelectedSource(null);
      setSourceLibraryOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载最新结果失败");
    } finally {
      setLoadingLatestResult(false);
    }
  }

  const taskTitle = task?.request.target_product || task?.request.raw_description?.slice(0, 22) || "新的竞品分析";
  const isWorking = submitting || resuming || loadingLatestResult || task?.status === "queued" || task?.status === "running" || task?.status === "cancelling";

  return (
    <main className={`app-shell ${sidebarOpen ? "sidebar-open" : "sidebar-closed"} ${!task ? "home-shell" : ""}`}>
      <aside className="sidebar">
        <div className="sidebar-top">
          <button type="button" className="icon-button" onClick={() => setSidebarOpen((value) => !value)} title={sidebarOpen ? "收起侧边栏" : "展开侧边栏"}>
            {sidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}
          </button>
          <button type="button" className="new-chat" onClick={() => setTask(null)} title="新建分析">
            <MessageSquarePlus size={19} />
            <span>新建分析</span>
          </button>
        </div>
        <nav className="sidebar-nav" aria-label="工作区导航">
          <button type="button" className="active">
            <Sparkles size={18} />
            <span>Agent 工作台</span>
          </button>
          <button type="button">
            <History size={18} />
            <span>历史任务</span>
          </button>
        </nav>
        {task && (
          <div className="current-task">
            <span>当前任务</span>
            <strong>{taskTitle}</strong>
            <StatusPill status={task.status} />
          </div>
        )}
      </aside>

      <section className="main-pane">
        {!task ? (
          <div className="home-stage">
            <WelcomeHeadline />
            <button type="button" className="resume-latest-button" onClick={() => void resumeLatestAnalysis()} disabled={resuming}>
              {resuming ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              <span>续跑最新任务</span>
            </button>
            <button type="button" className="resume-latest-button" onClick={() => void loadLatestCompletedResult()} disabled={loadingLatestResult}>
              {loadingLatestResult ? <Loader2 className="spin" size={18} /> : <History size={18} />}
              <span>查看最新结果</span>
            </button>
            <Composer onSubmit={startAnalysis} working={isWorking} cancelling={cancelling} onCancel={stopAnalysis} />
            {error && (
              <article className="assistant-card attention home-error">
                <div className="card-head">
                  <AlertCircle size={18} />
                  <strong>请求失败</strong>
                </div>
                <p>{error}</p>
              </article>
            )}
          </div>
        ) : (
          <>
            <header className="topbar">
              <button type="button" className="mobile-menu" onClick={() => setSidebarOpen((value) => !value)} title="菜单">
                <Menu size={20} />
              </button>
              <button type="button" className="model-picker">
                竞品分析 Agent
                <ChevronDown size={16} />
              </button>
              <div className="avatar">A</div>
            </header>

            <div className="conversation" ref={conversationRef}>
              <article className="user-card">
                <p>{task.request.raw_description || task.request.target_product}</p>
                <div className="meta-row">
                  <Clock3 size={14} />
                  <span>{formatTime(task.created_at)}</span>
                </div>
              </article>
              {task.status === "waiting_clarification" && <ClarificationCard task={task} onSubmitted={setTask} />}
              {task.status === "waiting_decision" && <DecisionCard task={task} onSubmitted={setTask} />}
              {isWorking && !events.length && (
                <article className="assistant-card thinking">
                  <Loader2 className="spin" size={18} />
                  <span>Agent 正在理解需求并规划分析路径...</span>
                </article>
              )}
              <EventStream events={events} taskStatus={task.status} />
              {report && <ReportView task={task} report={report} onSelectSource={(source) => { setSourceLibraryOpen(false); setSelectedSource(source); }} />}
              {error && (
                <article className="assistant-card attention">
                  <div className="card-head">
                    <AlertCircle size={18} />
                    <strong>请求失败</strong>
                  </div>
                  <p>{error}</p>
                </article>
              )}
            </div>

            <div className="composer-wrap">
              <Composer onSubmit={startAnalysis} working={isWorking} cancelling={cancelling || task?.status === "cancelling"} onCancel={stopAnalysis} />
            </div>
          </>
        )}
      </section>

      {task && (
        <SourcePanel
          source={selectedSource}
          taskId={task.id}
          libraryOpen={sourceLibraryOpen}
          onOpenLibrary={() => {
            setSourceLibraryOpen(true);
            setSelectedSource(null);
          }}
          onClose={() => {
            setSelectedSource(null);
            setSourceLibraryOpen(false);
          }}
        />
      )}
    </main>
  );
}
