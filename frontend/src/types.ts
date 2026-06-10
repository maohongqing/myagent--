export type TaskStatus = "queued" | "running" | "cancelling" | "waiting_clarification" | "waiting_decision" | "completed" | "failed" | "cancelled";
export type AgentEventStatus = "queued" | "running" | "waiting" | "completed" | "failed" | "revision" | "cancelled";
export type AnalysisPurpose = "decision_support" | "learning" | "market_warning";
export type LifecycleStage = "concept" | "development" | "launched";
export type CompetitorType = "head_direct" | "challenger" | "indirect_cross" | "potential_substitute";
export type SearchMethod =
  | "duckduckgo_text"
  | "duckduckgo_news"
  | "tavily"
  | "brave_web"
  | "searchapi_google"
  | "searchapi_baidu"
  | "searchapi_duckduckgo"
  | "serpapi_google"
  | "serpapi_baidu";
export type DeepSearchBackend =
  | "none"
  | "local_pagination"
  | "local_relevant_links"
  | "local_all_links"
  | "tavily"
  | "local_relevant_links_tavily";

export interface CompetitorInput {
  name: string;
  category?: string;
  website?: string;
  website_copy?: string;
  sales_notes?: string;
  win_loss_notes?: string;
  tags?: string[];
}

export interface CreateTaskRequest {
  target_product?: string;
  raw_description?: string;
  lifecycle_stage?: LifecycleStage;
  analysis_purpose?: AnalysisPurpose;
  analysis_goal?: string;
  clarification_answers?: string[];
  competitors: string[];
  urls: string[];
  notes?: string;
  our_product?: string;
  market?: string;
  product_category?: string;
  geography?: string;
  company_size?: string;
  competitor_profiles?: CompetitorInput[];
  enable_screenshots?: boolean;
  enable_readability_extraction?: boolean;
  search_methods?: SearchMethod[];
  deep_search_backend?: DeepSearchBackend;
  candidate_target_count?: number;
  max_revision_loops?: number;
  llm_failure_policy?: "ask" | "continue_without_llm";
}

export interface ClarificationOption {
  label: string;
  value: string;
  description: string;
  updates: Record<string, string>;
}

export interface ClarificationRequest {
  id: string;
  missing_slots: string[];
  question: string;
  options: ClarificationOption[];
  context_summary: string;
  created_at: string;
}

export interface AnalysisTask {
  id: string;
  request: CreateTaskRequest;
  status: TaskStatus;
  current_node?: string | null;
  pending_clarification?: ClarificationRequest | null;
  pending_decision?: DecisionRequest | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
}

export type DecisionAction = "retry_llm" | "continue_without_llm" | "cancel";

export interface DecisionRequest {
  id: string;
  kind: "llm_failure";
  node: string;
  question: string;
  reason: string;
  options: Array<{ label: string; value: DecisionAction; description: string }>;
  context_summary: string;
  created_at: string;
}

export interface SearchRunSummary {
  run_key: string;
  task_id: string;
  purpose: string;
  query: string;
  methods: string[];
  deep_backend: string;
  status: string;
  created_at: string;
  run_dir: string;
  report_path?: string | null;
  results_json_path?: string | null;
  result_count: number;
  chunk_count: number;
  screenshot_count: number;
  skipped_duplicate_count: number;
  errors: string[];
  chunks_status?: string;
}

export interface TaskOutputFile {
  path: string;
  size: number;
  modified_at: number;
  url: string;
}

export interface TaskOutputList {
  task_id: string;
  root: string;
  files: TaskOutputFile[];
}

export interface SearchRunList {
  task_id: string;
  runs: SearchRunSummary[];
}

export interface SearchRunDetail extends SearchRunSummary {
  results: Array<Record<string, unknown>>;
  search_page_screenshots: Array<Record<string, unknown>>;
  chunks: Array<Record<string, unknown>>;
  results_status: string;
  chunks_status: string;
}

export interface AgentRunEvent {
  id: string;
  task_id: string;
  node: string;
  status: AgentEventStatus;
  input_summary?: string | null;
  output_summary?: string | null;
  message?: string | null;
  duration_ms?: number | null;
  token_estimate?: number | null;
  error?: string | null;
  details?: Record<string, unknown> & {
    ui_stage_key?: string;
    ui_title?: string;
    display_markdown?: string;
    candidate_qa_change_log?: CandidateQAChangeLog;
    candidates?: Array<Record<string, unknown>>;
  };
  created_at: string;
}

export interface CandidateQAChangeLog {
  removed_candidates: Array<Record<string, unknown>>;
  merged_candidates: Array<Record<string, unknown>>;
  type_corrections: Array<Record<string, unknown>>;
  unverified_candidates: Array<Record<string, unknown>>;
  qa_notes: string[];
}

export interface RankFactors {
  heat: number;
  growth: number;
  similarity: number;
  risk: number;
}

export interface EvidenceItem {
  id: string;
  url: string;
  title: string;
  source_type: "search" | "url" | "manual" | "generated" | "screenshot";
  published_at?: string | null;
  captured_at: string;
  excerpt: string;
  credibility: number;
  related_fields: string[];
  image_path?: string | null;
  image_alt?: string | null;
  evidence_kind?: "source_chunk" | "extracted_fact";
  competitor?: string | null;
  tool?: string | null;
  field_name?: string | null;
  source_refs?: Array<{
    source_id?: string | null;
    chunk_id?: string | null;
    url: string;
    title: string;
    quote: string;
    screenshot_path?: string | null;
    evidence_screenshot_path?: string | null;
    chunk_shot_paths: string[];
  }>;
  source_evidence_ids?: string[];
}

export interface CompetitorProfile {
  name: string;
  website?: string | null;
  positioning: string;
  target_users: string[];
  core_features: string[];
  pricing: string;
  business_model: string;
  channels: string[];
  growth_signals: string[];
  evidence_ids: string[];
  competitor_type?: CompetitorType | null;
  score?: number | null;
  selection_reason?: string;
  rank_factors?: RankFactors | null;
}

export interface AnalysisFinding {
  id: string;
  category: string;
  title: string;
  conclusion: string;
  evidence_ids: string[];
  confidence: number;
  impact: "low" | "medium" | "high";
}

export interface AnalysisBenchmark {
  target_product: string;
  raw_description: string;
  lifecycle_stage: LifecycleStage;
  analysis_purpose: AnalysisPurpose;
  analysis_goal: string;
  confidence: number;
  inferred_from: string[];
  locked: boolean;
}

export interface CandidateCompetitor {
  name: string;
  description: string;
  website?: string | null;
  competitor_type: CompetitorType;
  source_urls: string[];
  source_titles: string[];
  source_refs?: Array<{
    source_id?: string | null;
    chunk_id?: string | null;
    url: string;
    title: string;
    quote: string;
    screenshot_path?: string | null;
    evidence_screenshot_path?: string | null;
    chunk_shot_paths: string[];
  }>;
  tags: string[];
  rank_factors: RankFactors;
  score: number;
  selection_reason: string;
}

export interface SelectedCompetitorSet {
  total_budget: number;
  candidates: CandidateCompetitor[];
  selected: CandidateCompetitor[];
  allocation: Record<string, number>;
  fallback_notes: string[];
}

export interface ToolPlan {
  purpose: AnalysisPurpose;
  focus: string;
  frameworks: string[];
  methods: string[];
  selected_tools: string[];
  rationale: string;
}

export interface CollectionPlan {
  instructions: Array<{
    tool: string;
    competitor?: string | null;
    queries: string[];
    source_types: string[];
    extraction_targets: string[];
  }>;
  provider_interfaces: string[];
  notes: string[];
}

export interface SearchQuery {
  query: string;
  tool?: string | null;
  competitor?: string | null;
  target_site_types: string[];
  priority: number;
  expected_fields: string[];
  rationale: string;
}

export interface SearchPlan {
  queries: SearchQuery[];
  missing_fields: string[];
  provider_order: Array<"cache" | "search" | "web" | "app_store" | "industry_database" | "hiring_signal" | "financing_news">;
  rationale: string;
}

export interface ToolClaim {
  category: string;
  title: string;
  claim: string;
  evidence_ids: string[];
  confidence: number;
  reasoning: string;
}

export interface ToolExecutionResult {
  tool_name: string;
  canonical_tool_name: string;
  schema_name: string;
  claims: ToolClaim[];
  data: Record<string, unknown>;
  evidence_ids: string[];
  confidence: number;
  reasoning: string;
  generated_by: "llm" | "fallback";
}

export interface EvidenceSignal {
  id: string;
  provider: string;
  signal_type: string;
  competitor?: string | null;
  value: string;
  confidence: number;
  evidence_ids: string[];
  metadata: Record<string, unknown>;
}

export interface RawSource {
  id: string;
  url: string;
  title: string;
  content: string;
  source_type: "search" | "url" | "manual" | "generated" | "screenshot";
  captured_at: string;
  metadata: Record<string, unknown>;
}

export interface SourceProviderResult {
  provider: string;
  sources: RawSource[];
  signals: EvidenceSignal[];
  cache_hit: boolean;
  query?: string | null;
  errors: string[];
  metadata: Record<string, unknown>;
}

export interface FieldExtractionResult {
  competitor: string;
  tool: string;
  canonical_tool_name: string;
  field_name: string;
  facts: Array<{
    id: string;
    competitor: string;
    tool: string;
    canonical_tool_name: string;
    field_name: string;
    content: string;
    confidence: number;
    reasoning: string;
    source_refs: EvidenceItem["source_refs"];
    source_evidence_ids: string[];
  }>;
  missing: boolean;
  search_iterations: number;
  queries: SearchQuery[];
}

export interface ReportReference {
  id: string;
  title: string;
  reference_type: "clean_card" | "tool_result";
  anchor: string;
  summary: string;
  payload: Record<string, unknown>;
}

export interface ReportChapterDraft {
  title: string;
  markdown: string;
  reference_ids: string[];
}

export interface QAResult {
  passed: boolean;
  score: number;
  issues: Array<{ severity: string; field: string; message: string }>;
  revision_request?: {
    target_agent: string;
    reason: string;
    missing_fields: string[];
    suggested_queries: string[];
  } | null;
}

export type SurveyQuestionType = "single_choice" | "multiple_choice" | "scale" | "text" | "ranking";

export interface SurveyQuestion {
  id: string;
  type: SurveyQuestionType;
  title: string;
  options: string[];
  dimension: string;
  mapped_competitors: string[];
  required: boolean;
}

export interface SurveyDesign {
  id: string;
  task_id: string;
  title: string;
  target_respondents: string;
  research_goals: string[];
  suggested_sample_size: string;
  distribution_channels: string[];
  questions: SurveyQuestion[];
  created_at: string;
}

export interface SurveyResponseAnalysisRequest {
  raw_responses: string;
  format_hint?: "auto" | "csv" | "json" | "table";
}

export interface SurveyAnalysisResult {
  id: string;
  task_id: string;
  survey_design_id?: string | null;
  sample_size: number;
  summary: string;
  key_findings: string[];
  competitor_insights: string[];
  segment_insights: string[];
  open_text_themes: string[];
  limitations: string[];
  appendix_markdown: string;
  created_at: string;
}

export interface CompetitorReport {
  id: string;
  task_id: string;
  target_product: string;
  competitors: CompetitorProfile[];
  executive_summary: string;
  findings: AnalysisFinding[];
  comparison_matrix: Array<{
    competitor: string;
    dimension: string;
    value: string;
    evidence_ids: string[];
  }>;
  swot: {
    strengths: string[];
    weaknesses: string[];
    opportunities: string[];
    threats: string[];
  };
  recommendations: string[];
  evidence: EvidenceItem[];
  deep_dive_markdown: string;
  qa_history: QAResult[];
  generated_at: string;
  risk_notice?: string | null;
  analysis_benchmark?: AnalysisBenchmark | null;
  selected_competitor_set?: SelectedCompetitorSet | null;
  tool_plan?: ToolPlan | null;
  collection_plan?: CollectionPlan | null;
  search_plan?: SearchPlan | null;
  source_provider_results?: SourceProviderResult[];
  tool_results?: ToolExecutionResult[];
  field_extraction_results?: FieldExtractionResult[];
  competitor_brief_profiles?: Array<Record<string, unknown>>;
  competitor_detail_cards?: Array<Record<string, unknown>>;
  industry_card?: Record<string, unknown> | null;
  competitor_detail_cards_clean?: Array<{
    competitor: string;
    competitor_type?: CompetitorType | null;
    fields: Record<string, string>;
    reference_ids: Record<string, string>;
  }>;
  industry_card_clean?: {
    fields: Record<string, string>;
    reference_ids: Record<string, string>;
  } | null;
  object_progress?: Array<{
    stage: string;
    object_id: string;
    object_name: string;
    status: string;
    searched_sources: number;
    analyzed_chunks: number;
    merged_fields: number;
    error?: string;
  }>;
  report_references?: ReportReference[];
  report_chapters?: ReportChapterDraft[];
  knowledge_reuse_summary?: KnowledgeReuseSummary | null;
  knowledge_card_ids?: string[];
  knowledge_source_ids?: string[];
  survey_design?: SurveyDesign | null;
  survey_analysis?: SurveyAnalysisResult | null;
}

export interface TaskDetail {
  task: AnalysisTask;
  report?: CompetitorReport | null;
  events: AgentRunEvent[];
  survey_design?: SurveyDesign | null;
  survey_analysis?: SurveyAnalysisResult | null;
}

export type FreshnessStatus = "fresh" | "probably_stale" | "stale" | "unknown";
export type KnowledgeCardType = "competitor" | "industry";
export type KnowledgeSuggestionKind = "create" | "update" | "append_sources";
export type KnowledgeSuggestionStatus = "pending" | "accepted" | "rejected";

export interface KnowledgeSource {
  id: string;
  url: string;
  title: string;
  excerpt: string;
  source_type: EvidenceItem["source_type"];
  credibility: number;
  published_at?: string | null;
  captured_at: string;
  origin_task_id?: string | null;
  metadata: Record<string, unknown>;
}

export interface KnowledgeCompetitorCard {
  id: string;
  name: string;
  canonical_name: string;
  aliases: string[];
  industry: string;
  market: string;
  tags: string[];
  confidence: number;
  freshness_status: FreshnessStatus;
  source_count: number;
  origin_task_id?: string | null;
  card: {
    competitor: string;
    competitor_type?: CompetitorType | null;
    fields: Record<string, string>;
    reference_ids: Record<string, string>;
  };
  knowledge_source_ids: string[];
  deleted: boolean;
  created_at: string;
  updated_at: string;
  last_verified_at?: string | null;
}

export interface KnowledgeIndustryCard {
  id: string;
  title: string;
  canonical_name: string;
  aliases: string[];
  industry: string;
  market: string;
  tags: string[];
  confidence: number;
  freshness_status: FreshnessStatus;
  source_count: number;
  origin_task_id?: string | null;
  card: {
    fields: Record<string, string>;
    reference_ids: Record<string, string>;
  };
  knowledge_source_ids: string[];
  deleted: boolean;
  created_at: string;
  updated_at: string;
  last_verified_at?: string | null;
}

export interface KnowledgeReuseSummary {
  competitor_cards: KnowledgeCompetitorCard[];
  industry_cards: KnowledgeIndustryCard[];
  source_ids: string[];
  notes: string[];
}

export interface KnowledgeCardVersion {
  id: string;
  card_type: KnowledgeCardType;
  card_id: string;
  action: string;
  before?: Record<string, unknown> | null;
  after?: Record<string, unknown> | null;
  origin_task_id?: string | null;
  created_at: string;
}

export interface KnowledgeCardDetail<T> {
  card: T;
  sources: KnowledgeSource[];
  versions: KnowledgeCardVersion[];
}

export interface KnowledgeUpdateDiff {
  field: string;
  old_value: string;
  new_value: string;
  reason: string;
  evidence_refs: string[];
  confidence: number;
}

export interface KnowledgeUpdateSuggestion {
  id: string;
  kind: KnowledgeSuggestionKind;
  status: KnowledgeSuggestionStatus;
  card_type: KnowledgeCardType;
  target_card_id?: string | null;
  title: string;
  summary: string;
  proposed_competitor_card?: KnowledgeCompetitorCard | null;
  proposed_industry_card?: KnowledgeIndustryCard | null;
  diffs: KnowledgeUpdateDiff[];
  source_ids: string[];
  origin_task_id?: string | null;
  created_at: string;
  resolved_at?: string | null;
}

export interface KnowledgeImportPreview {
  task_id: string;
  competitor_cards: KnowledgeCompetitorCard[];
  industry_cards: KnowledgeIndustryCard[];
  sources: KnowledgeSource[];
}
