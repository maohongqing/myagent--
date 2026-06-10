from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


TaskStatus = Literal[
    "queued",
    "running",
    "cancelling",
    "waiting_clarification",
    "waiting_decision",
    "completed",
    "failed",
    "cancelled",
]
AgentEventStatus = Literal["queued", "running", "waiting", "completed", "failed", "revision", "cancelled"]
SourceType = Literal["search", "url", "manual", "generated", "screenshot"]
RevisionTarget = Literal[
    "candidate_discovery_agent",
    "targeted_collector_agent",
    "search_planner_agent",
    "source_collector_agent",
    "tool_executor_agent",
    "analyst_agent",
    "writer_agent",
]
LifecycleStage = Literal["concept", "development", "launched"]
AnalysisPurpose = Literal["decision_support", "learning", "market_warning"]
CompetitorType = Literal["head_direct", "challenger", "indirect_cross", "potential_substitute"]
SourceProviderName = Literal[
    "cache",
    "search",
    "web",
    "app_store",
    "industry_database",
    "hiring_signal",
    "financing_news",
]
SearchMethod = Literal[
    "duckduckgo_text",
    "duckduckgo_news",
    "tavily",
    "brave_web",
    "searchapi_google",
    "searchapi_baidu",
    "searchapi_duckduckgo",
    "serpapi_google",
    "serpapi_baidu",
]
DeepSearchBackend = Literal[
    "none",
    "local_pagination",
    "local_relevant_links",
    "local_all_links",
    "tavily",
    "local_relevant_links_tavily",
]
EvidenceSignalType = Literal[
    "positioning",
    "pricing",
    "feature",
    "review",
    "growth",
    "funding",
    "hiring",
    "technology",
    "risk",
    "market",
    "other",
]
SurveyQuestionType = Literal["single_choice", "multiple_choice", "scale", "text", "ranking"]


def _dedupe_strings(value: list[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for item in value:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            items.append(cleaned)
    return items


class CompetitorInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    category: str = Field(default="direct", max_length=40)
    website: str = Field(default="", max_length=500)
    website_copy: str = Field(default="", max_length=8000)
    sales_notes: str = Field(default="", max_length=8000)
    win_loss_notes: str = Field(default="", max_length=8000)
    tags: list[str] = Field(default_factory=list, max_length=24)

    @field_validator("name", "category", "website", "website_copy", "sales_notes", "win_loss_notes", mode="after")
    @classmethod
    def trim_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("tags", mode="after")
    @classmethod
    def trim_tags(cls, value: list[str]) -> list[str]:
        return _dedupe_strings(value)


def _normalize_purpose(value: str | None) -> str | None:
    if value is None:
        return None
    mapping = {
        "决策支持": "decision_support",
        "战略决策": "decision_support",
        "立项决策": "decision_support",
        "learning": "learning",
        "学习借鉴": "learning",
        "体验借鉴": "learning",
        "市场预警": "market_warning",
        "风险预警": "market_warning",
        "warning": "market_warning",
    }
    cleaned = value.strip()
    return mapping.get(cleaned, cleaned or None)


def _normalize_lifecycle(value: str | None) -> str | None:
    if value is None:
        return None
    mapping = {
        "概念期": "concept",
        "产品战略和产品规划": "concept",
        "想法期": "concept",
        "研发期": "development",
        "设计开发": "development",
        "开发期": "development",
        "已上线运营期": "launched",
        "产品运营": "launched",
        "运营期": "launched",
    }
    cleaned = value.strip()
    return mapping.get(cleaned, cleaned or None)


class CreateTaskRequest(BaseModel):
    target_product: str = Field(default="", max_length=120)
    raw_description: str | None = Field(default=None, max_length=8000)
    lifecycle_stage: LifecycleStage | None = None
    analysis_purpose: AnalysisPurpose | None = None
    analysis_goal: str | None = Field(default=None, max_length=1000)
    clarification_answers: list[str] = Field(default_factory=list, max_length=12)

    competitors: list[str] = Field(default_factory=list, max_length=24)
    urls: list[str] = Field(default_factory=list, max_length=24)
    notes: str | None = Field(default=None, max_length=4000)
    our_product: str | None = Field(default=None, max_length=120)
    market: str | None = Field(default=None, max_length=240)
    product_category: str | None = Field(default=None, max_length=160)
    geography: str | None = Field(default=None, max_length=120)
    company_size: str | None = Field(default=None, max_length=120)
    competitor_profiles: list[CompetitorInput] = Field(default_factory=list, max_length=24)
    enable_screenshots: bool = False
    enable_readability_extraction: bool = False
    search_methods: list[SearchMethod] = Field(
        default_factory=lambda: ["duckduckgo_text", "duckduckgo_news", "tavily"],
        max_length=9,
    )
    deep_search_backend: DeepSearchBackend = "none"
    candidate_target_count: int = Field(default=12, ge=1, le=30)
    max_revision_loops: int | None = Field(default=None, ge=0, le=5)
    llm_failure_policy: Literal["ask", "continue_without_llm"] = "ask"

    @field_validator("competitors", "urls", "clarification_answers", mode="after")
    @classmethod
    def trim_list(cls, value: list[str]) -> list[str]:
        return _dedupe_strings(value)

    @field_validator("search_methods", mode="after")
    @classmethod
    def normalize_search_methods(cls, value: list[SearchMethod]) -> list[SearchMethod]:
        return _dedupe_strings(value) or ["duckduckgo_text", "duckduckgo_news", "tavily"]  # type: ignore[return-value]

    @field_validator("target_product", mode="after")
    @classmethod
    def trim_required_text(cls, value: str) -> str:
        return value.strip()

    @field_validator(
        "raw_description",
        "notes",
        "our_product",
        "market",
        "product_category",
        "geography",
        "company_size",
        "analysis_goal",
        mode="after",
    )
    @classmethod
    def trim_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("analysis_purpose", mode="before")
    @classmethod
    def normalize_analysis_purpose(cls, value: str | None) -> str | None:
        return _normalize_purpose(value)

    @field_validator("lifecycle_stage", mode="before")
    @classmethod
    def normalize_stage(cls, value: str | None) -> str | None:
        return _normalize_lifecycle(value)

    @model_validator(mode="after")
    def sync_competitor_names(self) -> "CreateTaskRequest":
        if not self.target_product and not self.raw_description:
            raise ValueError("target_product or raw_description is required")
        names = list(self.competitors)
        seen = set(names)
        for profile in self.competitor_profiles:
            if profile.name not in seen:
                names.append(profile.name)
                seen.add(profile.name)
        self.competitors = names
        return self


class ClarificationOption(BaseModel):
    label: str
    value: str
    description: str = ""
    updates: dict[str, str] = Field(default_factory=dict)


class ClarificationRequest(BaseModel):
    id: str = Field(default_factory=lambda: f"clar_{uuid4().hex[:12]}")
    missing_slots: list[str] = Field(default_factory=list)
    question: str
    options: list[ClarificationOption] = Field(default_factory=list, min_length=1, max_length=4)
    context_summary: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class ClarificationAnswerRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=1000)
    selected_option: str | None = Field(default=None, max_length=200)

    @field_validator("answer", "selected_option", mode="after")
    @classmethod
    def trim_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


DecisionAction = Literal["retry_llm", "continue_without_llm", "cancel"]


class DecisionOption(BaseModel):
    label: str
    value: DecisionAction
    description: str = ""


class DecisionRequest(BaseModel):
    id: str = Field(default_factory=lambda: f"decision_{uuid4().hex[:12]}")
    kind: Literal["llm_failure"] = "llm_failure"
    node: str = "workflow"
    question: str
    reason: str = ""
    options: list[DecisionOption] = Field(default_factory=list, min_length=1, max_length=4)
    context_summary: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class DecisionAnswerRequest(BaseModel):
    action: DecisionAction


class AnalysisBenchmark(BaseModel):
    target_product: str
    raw_description: str = ""
    lifecycle_stage: LifecycleStage
    analysis_purpose: AnalysisPurpose
    analysis_goal: str
    confidence: float = Field(default=0.65, ge=0, le=1)
    inferred_from: list[str] = Field(default_factory=list)
    locked: bool = True


class RankFactors(BaseModel):
    heat: float = Field(default=0.0, ge=0, le=1)
    growth: float = Field(default=0.0, ge=0, le=1)
    similarity: float = Field(default=0.0, ge=0, le=1)
    risk: float = Field(default=0.0, ge=0, le=1)


class RiskFactors(BaseModel):
    capital: float = Field(default=0.0, ge=0, le=1)
    tech: float = Field(default=0.0, ge=0, le=1)
    model: float = Field(default=0.0, ge=0, le=1)
    team: float = Field(default=0.0, ge=0, le=1)


class SourceRef(BaseModel):
    source_id: str | None = None
    chunk_id: str | None = None
    url: str = ""
    title: str = "Untitled source"
    quote: str = Field(default="", max_length=1200)
    screenshot_path: str | None = None
    evidence_screenshot_path: str | None = None
    chunk_shot_paths: list[str] = Field(default_factory=list)


class CandidateGapRequest(BaseModel):
    missing_competitor_types: list[CompetitorType] = Field(default_factory=list)
    missing_factors: list[str] = Field(default_factory=list)
    suggested_query_focus: list[str] = Field(default_factory=list)
    rationale: str = ""


class CandidateCompetitor(BaseModel):
    name: str
    description: str = ""
    website: str | None = None
    competitor_type: CompetitorType = "challenger"
    source_urls: list[str] = Field(default_factory=list)
    source_titles: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    rank_factors: RankFactors = Field(default_factory=RankFactors)
    risk_factors: RiskFactors = Field(default_factory=RiskFactors)
    score: float = Field(default=0.0, ge=0)
    selection_reason: str = ""
    weight_assignment: dict[str, float] = Field(default_factory=dict)
    factor_scores: dict[str, float] = Field(default_factory=dict)
    scoring_reason: dict[str, Any] = Field(default_factory=dict)


class ScoringProfile(BaseModel):
    purpose: AnalysisPurpose
    alpha_heat: float
    beta_growth: float
    gamma_similarity: float
    delta_risk: float
    type_weights: dict[str, float]


class SelectedCompetitorSet(BaseModel):
    total_budget: int = 5
    scoring_profile: ScoringProfile
    candidates: list[CandidateCompetitor] = Field(default_factory=list)
    selected: list[CandidateCompetitor] = Field(default_factory=list)
    allocation: dict[str, int] = Field(default_factory=dict)
    fallback_notes: list[str] = Field(default_factory=list)
    gap_request: CandidateGapRequest | None = None


class ToolPlan(BaseModel):
    purpose: AnalysisPurpose
    focus: str
    frameworks: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    rationale: str = ""


class CollectionInstruction(BaseModel):
    tool: str
    competitor: str | None = None
    queries: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    extraction_targets: list[str] = Field(default_factory=list)


class CollectionPlan(BaseModel):
    instructions: list[CollectionInstruction] = Field(default_factory=list)
    provider_interfaces: list[str] = Field(
        default_factory=lambda: [
            "SearchProvider",
            "AppStoreProvider",
            "IndustryDatabaseProvider",
            "HiringSignalProvider",
            "FinancingNewsProvider",
        ]
    )
    notes: list[str] = Field(default_factory=list)


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    tool: str | None = Field(default=None, max_length=120)
    competitor: str | None = Field(default=None, max_length=160)
    target_site_types: list[str] = Field(default_factory=list)
    priority: int = Field(default=3, ge=1, le=5)
    expected_fields: list[str] = Field(default_factory=list)
    rationale: str = ""


class SearchPlan(BaseModel):
    queries: list[SearchQuery] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    provider_order: list[SourceProviderName] = Field(
        default_factory=lambda: ["cache", "app_store", "industry_database", "hiring_signal", "financing_news", "search", "web"]
    )
    rationale: str = ""


class ToolSpec(BaseModel):
    name: str
    canonical_name: str
    applicable_purposes: list[AnalysisPurpose] = Field(default_factory=list)
    description: str = ""
    input_schema_name: str = "tool_execution_input"
    output_schema_name: str = "tool_execution_result"
    required_evidence_types: list[EvidenceSignalType] = Field(default_factory=list)
    prompt: str = ""


class ToolClaim(BaseModel):
    category: str = Field(default="observation", max_length=80)
    title: str = Field(..., min_length=1, max_length=200)
    claim: str = Field(..., min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.45, ge=0, le=1)
    reasoning: str = Field(default="", max_length=2000)


class ToolExecutionResult(BaseModel):
    tool_name: str
    canonical_tool_name: str = ""
    schema_name: str = "tool_execution_result"
    scope_type: Literal["competitor", "industry", "comparison"] = "comparison"
    competitor_name: str | None = None
    input_reference_ids: list[str] = Field(default_factory=list)
    reference_id: str | None = None
    claims: list[ToolClaim] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.45, ge=0, le=1)
    reasoning: str = ""
    generated_by: Literal["llm", "fallback"] = "llm"


class ToolSandbox(BaseModel):
    tool: str
    schema_name: str
    scope_type: Literal["competitor", "industry", "comparison"] = "comparison"
    competitor_name: str | None = None
    input_reference_ids: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    claims: list[ToolClaim] = Field(default_factory=list)
    result: ToolExecutionResult | None = None


EvidenceKind = Literal["source_chunk", "extracted_fact"]



class EvidenceItem(BaseModel):
    id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    url: str
    title: str = "Untitled source"
    source_type: SourceType = "url"
    published_at: str | None = None
    captured_at: datetime = Field(default_factory=utc_now)
    excerpt: str = Field(..., min_length=1)
    credibility: float = Field(default=0.65, ge=0, le=1)
    related_fields: list[str] = Field(default_factory=list)
    image_path: str | None = None
    image_alt: str | None = None
    evidence_kind: EvidenceKind = "source_chunk"
    competitor: str | None = None
    tool: str | None = None
    field_name: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)


class ExtractedEvidenceItem(BaseModel):
    id: str = Field(default_factory=lambda: f"xf_{uuid4().hex[:12]}")
    competitor: str
    tool: str
    canonical_tool_name: str = ""
    field_name: str
    content: str = Field(..., min_length=1, max_length=2000)
    confidence: float = Field(default=0.45, ge=0, le=1)
    reasoning: str = Field(default="", max_length=1000)
    source_refs: list[SourceRef] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)


class FieldExtractionResult(BaseModel):
    competitor: str
    tool: str
    canonical_tool_name: str = ""
    field_name: str
    facts: list[ExtractedEvidenceItem] = Field(default_factory=list)
    missing: bool = True
    search_iterations: int = 0
    queries: list[SearchQuery] = Field(default_factory=list)


class RawSource(BaseModel):
    id: str = Field(default_factory=lambda: f"raw_{uuid4().hex[:12]}")
    url: str
    title: str = "Untitled source"
    content: str
    source_type: SourceType = "url"
    captured_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisIntentJSON(BaseModel):
    industry: str = ""
    analysis_purpose: AnalysisPurpose = "decision_support"
    goal: str = ""
    target: str = ""
    problems: list[str] = Field(default_factory=list)
    challenges: list[str] = Field(default_factory=list)
    has_own_product: bool = False
    own_product: str = ""
    target_product: str = ""
    confidence: float = Field(default=0.65, ge=0, le=1)
    source_summary: str = ""


class CandidateQAResult(BaseModel):
    exists: bool = False
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str = Field(default="", max_length=1000)
    matched_urls: list[str] = Field(default_factory=list, max_length=3)
    corrected_competitor_type: CompetitorType | None = None
    canonical_name: str | None = Field(default=None, max_length=160)
    merge_target: str | None = Field(default=None, max_length=160)


class ResearchField(BaseModel):
    value: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    source_refs: list[SourceRef] = Field(default_factory=list)


class CompetitorBriefProfile(BaseModel):
    competitor: str
    market_heat: ResearchField = Field(default_factory=ResearchField)
    development_status: ResearchField = Field(default_factory=ResearchField)
    source_refs: list[SourceRef] = Field(default_factory=list)
    qa_passed: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    qa_result: dict[str, Any] = Field(default_factory=dict)


class CompetitorDetailCard(BaseModel):
    competitor: str
    competitor_type: CompetitorType | None = None
    name: ResearchField = Field(default_factory=ResearchField)
    category: ResearchField = Field(default_factory=ResearchField)
    website: ResearchField = Field(default_factory=ResearchField)
    positioning: ResearchField = Field(default_factory=ResearchField)
    target_users: ResearchField = Field(default_factory=ResearchField)
    core_scenarios: ResearchField = Field(default_factory=ResearchField)
    core_features: ResearchField = Field(default_factory=ResearchField)
    key_parameters: ResearchField = Field(default_factory=ResearchField)
    pricing: ResearchField = Field(default_factory=ResearchField)
    business_model: ResearchField = Field(default_factory=ResearchField)
    channels: ResearchField = Field(default_factory=ResearchField)
    growth_signals: ResearchField = Field(default_factory=ResearchField)
    funding_or_org_signals: ResearchField = Field(default_factory=ResearchField)
    technology_or_product_features: ResearchField = Field(default_factory=ResearchField)
    differentiation: ResearchField = Field(default_factory=ResearchField)
    user_feedback: ResearchField = Field(default_factory=ResearchField)
    risks: ResearchField = Field(default_factory=ResearchField)
    recent_updates: ResearchField = Field(default_factory=ResearchField)
    key_evidence: ResearchField = Field(default_factory=ResearchField)


class IndustryBasicInfo(BaseModel):
    industry_name: ResearchField = Field(default_factory=ResearchField)
    industry_scope: ResearchField = Field(default_factory=ResearchField)
    main_users: ResearchField = Field(default_factory=ResearchField)
    core_scenarios: ResearchField = Field(default_factory=ResearchField)
    buyer_behavior_or_preferences: ResearchField = Field(default_factory=ResearchField)


class IndustryMarketSituation(BaseModel):
    market_size: ResearchField = Field(default_factory=ResearchField)
    growth_trend: ResearchField = Field(default_factory=ResearchField)
    willingness_to_pay: ResearchField = Field(default_factory=ResearchField)
    cost_pressure: ResearchField = Field(default_factory=ResearchField)
    investment_environment: ResearchField = Field(default_factory=ResearchField)


class IndustryCompetitionSituation(BaseModel):
    competition_intensity: ResearchField = Field(default_factory=ResearchField)
    entry_barriers: ResearchField = Field(default_factory=ResearchField)
    substitutes: ResearchField = Field(default_factory=ResearchField)
    user_switching_cost: ResearchField = Field(default_factory=ResearchField)


class IndustryExternalEnvironment(BaseModel):
    regulation_or_laws: ResearchField = Field(default_factory=ResearchField)
    technology_trends: ResearchField = Field(default_factory=ResearchField)
    key_resource_dependencies: ResearchField = Field(default_factory=ResearchField)


class IndustryCard(BaseModel):
    basic_info: IndustryBasicInfo = Field(default_factory=IndustryBasicInfo)
    market_situation: IndustryMarketSituation = Field(default_factory=IndustryMarketSituation)
    competition_situation: IndustryCompetitionSituation = Field(default_factory=IndustryCompetitionSituation)
    external_environment: IndustryExternalEnvironment = Field(default_factory=IndustryExternalEnvironment)


class QueryTemplate(BaseModel):
    template: str = Field(..., min_length=1, max_length=500)
    target_fields: list[str] = Field(default_factory=list)
    target_site_types: list[str] = Field(default_factory=list)
    priority: int = Field(default=3, ge=1, le=5)
    rationale: str = ""


class IndustryQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    target_fields: list[str] = Field(default_factory=list)
    target_site_types: list[str] = Field(default_factory=list)
    priority: int = Field(default=3, ge=1, le=5)
    rationale: str = ""


class DetailQueryPlan(BaseModel):
    competitor_query_templates: list[QueryTemplate] = Field(default_factory=list, max_length=12)
    industry_queries: list[IndustryQuery] = Field(default_factory=list, max_length=12)
    rationale: str = ""


class CardQAResult(BaseModel):
    passed: bool = False
    error_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    reference_issues: list[str] = Field(default_factory=list)
    needs_retry: bool = False
    reason: str = ""


class CompetitorDetailCardClean(BaseModel):
    competitor: str
    competitor_type: CompetitorType | None = None
    fields: dict[str, str] = Field(default_factory=dict)
    reference_ids: dict[str, str] = Field(default_factory=dict)


class IndustryCardClean(BaseModel):
    fields: dict[str, str] = Field(default_factory=dict)
    reference_ids: dict[str, str] = Field(default_factory=dict)


class CandidateQAChangeLog(BaseModel):
    removed_candidates: list[dict[str, Any]] = Field(default_factory=list)
    merged_candidates: list[dict[str, Any]] = Field(default_factory=list)
    type_corrections: list[dict[str, Any]] = Field(default_factory=list)
    unverified_candidates: list[dict[str, Any]] = Field(default_factory=list)
    qa_notes: list[str] = Field(default_factory=list)


class CompetitorRelevanceScore(BaseModel):
    competitor: str
    total_score: float = Field(default=0.0, ge=0, le=1)
    category_score: float = Field(default=0.0, ge=0, le=1)
    detailed_reason: str = ""
    suitable_for_deep_analysis: bool = True
    weight_assignment: dict[str, float] = Field(default_factory=dict)
    factor_scores: dict[str, float] = Field(default_factory=dict)
    scoring_reason: dict[str, Any] = Field(default_factory=dict)


class ReportReference(BaseModel):
    id: str
    title: str
    reference_type: Literal["clean_card", "tool_result"]
    anchor: str
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ReportChapterDraft(BaseModel):
    title: str
    markdown: str
    reference_ids: list[str] = Field(default_factory=list)


class EvidenceSignal(BaseModel):
    id: str = Field(default_factory=lambda: f"sig_{uuid4().hex[:12]}")
    provider: SourceProviderName = "search"
    signal_type: EvidenceSignalType = "other"
    competitor: str | None = None
    value: str
    confidence: float = Field(default=0.45, ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceProviderResult(BaseModel):
    provider: SourceProviderName
    sources: list[RawSource] = Field(default_factory=list)
    signals: list[EvidenceSignal] = Field(default_factory=list)
    cache_hit: bool = False
    query: str | None = None
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionInput(BaseModel):
    tool_name: str
    benchmark: AnalysisBenchmark
    competitors: list[CandidateCompetitor] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class CompetitorProfile(BaseModel):
    name: str
    website: str | None = None
    positioning: str = "Unknown"
    target_users: list[str] = Field(default_factory=list)
    core_features: list[str] = Field(default_factory=list)
    pricing: str = "Unknown"
    business_model: str = "Unknown"
    channels: list[str] = Field(default_factory=list)
    growth_signals: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    competitor_type: CompetitorType | None = None
    score: float | None = Field(default=None, ge=0)
    selection_reason: str = ""
    rank_factors: RankFactors | None = None


class AnalysisFinding(BaseModel):
    id: str = Field(default_factory=lambda: f"finding_{uuid4().hex[:12]}")
    category: Literal["positioning", "feature", "pricing", "growth", "risk", "opportunity", "strategy"]
    title: str
    conclusion: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    impact: Literal["low", "medium", "high"] = "medium"


class SWOT(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)


class ComparisonCell(BaseModel):
    competitor: str
    dimension: str
    value: str
    evidence_ids: list[str] = Field(default_factory=list)


class RevisionRequest(BaseModel):
    target_agent: RevisionTarget
    reason: str
    missing_fields: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)


class QAIssue(BaseModel):
    severity: Literal["low", "medium", "high"]
    field: str
    message: str


class QAResult(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=1)
    issues: list[QAIssue] = Field(default_factory=list)
    revision_request: RevisionRequest | None = None


FreshnessStatus = Literal["fresh", "probably_stale", "stale", "unknown"]
KnowledgeCardType = Literal["competitor", "industry"]
KnowledgeSuggestionKind = Literal["create", "update", "append_sources"]
KnowledgeSuggestionStatus = Literal["pending", "accepted", "rejected"]


class KnowledgeSource(BaseModel):
    id: str = Field(default_factory=lambda: f"ksrc_{uuid4().hex[:12]}")
    url: str = ""
    title: str = "Untitled source"
    excerpt: str = ""
    source_type: SourceType = "url"
    credibility: float = Field(default=0.65, ge=0, le=1)
    published_at: str | None = None
    captured_at: datetime = Field(default_factory=utc_now)
    origin_task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeCompetitorCard(BaseModel):
    id: str = Field(default_factory=lambda: f"kcomp_{uuid4().hex[:12]}")
    name: str
    canonical_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    industry: str = ""
    market: str = ""
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.65, ge=0, le=1)
    freshness_status: FreshnessStatus = "unknown"
    source_count: int = Field(default=0, ge=0)
    origin_task_id: str | None = None
    card: CompetitorDetailCardClean = Field(default_factory=lambda: CompetitorDetailCardClean(competitor=""))
    raw_card: CompetitorDetailCard | None = None
    knowledge_source_ids: list[str] = Field(default_factory=list)
    deleted: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_verified_at: datetime | None = None

    @model_validator(mode="after")
    def normalize_names(self) -> "KnowledgeCompetitorCard":
        self.name = self.name.strip()
        self.canonical_name = (self.canonical_name or self.name).strip()
        self.aliases = _dedupe_strings(self.aliases)
        self.tags = _dedupe_strings(self.tags)
        if not self.card.competitor:
            self.card.competitor = self.name
        self.source_count = len(set(self.knowledge_source_ids)) or self.source_count
        return self


class KnowledgeIndustryCard(BaseModel):
    id: str = Field(default_factory=lambda: f"kind_{uuid4().hex[:12]}")
    title: str
    canonical_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    industry: str = ""
    market: str = ""
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.65, ge=0, le=1)
    freshness_status: FreshnessStatus = "unknown"
    source_count: int = Field(default=0, ge=0)
    origin_task_id: str | None = None
    card: IndustryCardClean = Field(default_factory=IndustryCardClean)
    raw_card: IndustryCard | None = None
    knowledge_source_ids: list[str] = Field(default_factory=list)
    deleted: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_verified_at: datetime | None = None

    @model_validator(mode="after")
    def normalize_title(self) -> "KnowledgeIndustryCard":
        self.title = self.title.strip()
        self.canonical_name = (self.canonical_name or self.title).strip()
        self.aliases = _dedupe_strings(self.aliases)
        self.tags = _dedupe_strings(self.tags)
        self.source_count = len(set(self.knowledge_source_ids)) or self.source_count
        return self


class KnowledgeCardVersion(BaseModel):
    id: str = Field(default_factory=lambda: f"kver_{uuid4().hex[:12]}")
    card_type: KnowledgeCardType
    card_id: str
    action: str = "update"
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    origin_task_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeUpdateDiff(BaseModel):
    field: str
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.65, ge=0, le=1)


class KnowledgeUpdateSuggestion(BaseModel):
    id: str = Field(default_factory=lambda: f"ksug_{uuid4().hex[:12]}")
    kind: KnowledgeSuggestionKind
    status: KnowledgeSuggestionStatus = "pending"
    card_type: KnowledgeCardType
    target_card_id: str | None = None
    title: str
    summary: str = ""
    proposed_competitor_card: KnowledgeCompetitorCard | None = None
    proposed_industry_card: KnowledgeIndustryCard | None = None
    diffs: list[KnowledgeUpdateDiff] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    origin_task_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class KnowledgeReuseSummary(BaseModel):
    competitor_cards: list[KnowledgeCompetitorCard] = Field(default_factory=list)
    industry_cards: list[KnowledgeIndustryCard] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class KnowledgeImportPreview(BaseModel):
    task_id: str
    competitor_cards: list[KnowledgeCompetitorCard] = Field(default_factory=list)
    industry_cards: list[KnowledgeIndustryCard] = Field(default_factory=list)
    sources: list[KnowledgeSource] = Field(default_factory=list)


class KnowledgeImportCommitRequest(BaseModel):
    competitor_card_ids: list[str] = Field(default_factory=list)
    industry_card_ids: list[str] = Field(default_factory=list)


class SurveyQuestion(BaseModel):
    id: str = Field(default_factory=lambda: f"sq_{uuid4().hex[:8]}")
    type: SurveyQuestionType = "single_choice"
    title: str = Field(..., min_length=1, max_length=500)
    options: list[str] = Field(default_factory=list, max_length=20)
    dimension: str = Field(default="", max_length=120)
    mapped_competitors: list[str] = Field(default_factory=list, max_length=12)
    required: bool = True

    @field_validator("title", "dimension", mode="after")
    @classmethod
    def trim_question_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("options", "mapped_competitors", mode="after")
    @classmethod
    def trim_question_list(cls, value: list[str]) -> list[str]:
        return _dedupe_strings(value)


class SurveyDesign(BaseModel):
    id: str = Field(default_factory=lambda: f"survey_{uuid4().hex[:12]}")
    task_id: str
    title: str = Field(..., min_length=1, max_length=240)
    target_respondents: str = Field(default="", max_length=1000)
    research_goals: list[str] = Field(default_factory=list, max_length=12)
    suggested_sample_size: str = Field(default="50-100", max_length=120)
    distribution_channels: list[str] = Field(default_factory=list, max_length=12)
    questions: list[SurveyQuestion] = Field(default_factory=list, max_length=40)
    created_at: datetime = Field(default_factory=utc_now)


class SurveyResponseAnalysisRequest(BaseModel):
    raw_responses: str = Field(..., min_length=1, max_length=200000)
    format_hint: Literal["auto", "csv", "json", "table"] = "auto"


class SurveyAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: f"survey_analysis_{uuid4().hex[:12]}")
    task_id: str
    survey_design_id: str | None = None
    sample_size: int = Field(default=0, ge=0)
    summary: str = Field(default="", max_length=4000)
    key_findings: list[str] = Field(default_factory=list, max_length=20)
    competitor_insights: list[str] = Field(default_factory=list, max_length=20)
    segment_insights: list[str] = Field(default_factory=list, max_length=20)
    open_text_themes: list[str] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=12)
    appendix_markdown: str = Field(default="", max_length=50000)
    created_at: datetime = Field(default_factory=utc_now)


class CompetitorReport(BaseModel):
    id: str = Field(default_factory=lambda: f"report_{uuid4().hex[:12]}")
    task_id: str
    target_product: str
    competitors: list[CompetitorProfile]
    executive_summary: str
    findings: list[AnalysisFinding]
    comparison_matrix: list[ComparisonCell] = Field(default_factory=list)
    swot: SWOT = Field(default_factory=SWOT)
    recommendations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    deep_dive_markdown: str = ""
    qa_history: list[QAResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)
    risk_notice: str | None = None
    analysis_benchmark: AnalysisBenchmark | None = None
    analysis_intent: AnalysisIntentJSON | None = None
    selected_competitor_set: SelectedCompetitorSet | None = None
    tool_plan: ToolPlan | None = None
    collection_plan: CollectionPlan | None = None
    search_plan: SearchPlan | None = None
    source_provider_results: list[SourceProviderResult] = Field(default_factory=list)
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    tool_sandboxes: list[ToolSandbox] = Field(default_factory=list)
    field_extraction_results: list[FieldExtractionResult] = Field(default_factory=list)
    competitor_brief_profiles: list[CompetitorBriefProfile] = Field(default_factory=list)
    competitor_detail_cards: list[CompetitorDetailCard] = Field(default_factory=list)
    industry_card: IndustryCard | None = None
    competitor_detail_cards_clean: list[CompetitorDetailCardClean] = Field(default_factory=list)
    industry_card_clean: IndustryCardClean | None = None
    object_progress: list[dict[str, Any]] = Field(default_factory=list)
    report_references: list[ReportReference] = Field(default_factory=list)
    report_chapters: list[ReportChapterDraft] = Field(default_factory=list)
    knowledge_reuse_summary: KnowledgeReuseSummary | None = None
    knowledge_card_ids: list[str] = Field(default_factory=list)
    knowledge_source_ids: list[str] = Field(default_factory=list)
    survey_design: SurveyDesign | None = None
    survey_analysis: SurveyAnalysisResult | None = None


class AnalysisTask(BaseModel):
    id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    request: CreateTaskRequest
    status: TaskStatus = "queued"
    current_node: str | None = None
    pending_clarification: ClarificationRequest | None = None
    pending_decision: DecisionRequest | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class AgentRunEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    task_id: str
    node: str
    status: AgentEventStatus
    input_summary: str | None = None
    output_summary: str | None = None
    message: str | None = None
    duration_ms: int | None = None
    token_estimate: int | None = None
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TaskDetail(BaseModel):
    task: AnalysisTask
    report: CompetitorReport | None = None
    events: list[AgentRunEvent] = Field(default_factory=list)
    survey_design: SurveyDesign | None = None
    survey_analysis: SurveyAnalysisResult | None = None
