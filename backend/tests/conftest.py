from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myagent.backend.app.agents.deep_report import build_deep_dive_markdown
from myagent.backend.app.agents.strategy import build_collection_plan, build_tool_plan, scoring_profile_for
from myagent.backend.app.config import Settings
from myagent.backend.app.models import (
    AgentRunEvent,
    AnalysisBenchmark,
    AnalysisFinding,
    AnalysisTask,
    CandidateCompetitor,
    CompetitorDetailCardClean,
    CompetitorProfile,
    CompetitorReport,
    CreateTaskRequest,
    EvidenceItem,
    IndustryCardClean,
    KnowledgeCardType,
    KnowledgeCardVersion,
    KnowledgeCompetitorCard,
    KnowledgeIndustryCard,
    KnowledgeSource,
    KnowledgeUpdateSuggestion,
    RankFactors,
    RawSource,
    ReportReference,
    SearchPlan,
    SearchQuery,
    SelectedCompetitorSet,
    SourceProviderResult,
    SWOT,
    SurveyAnalysisResult,
    SurveyDesign,
    ToolClaim,
    ToolExecutionResult,
    ToolSandbox,
)
from myagent.backend.app.storage import Storage


class MemoryStorage(Storage):
    def __init__(self) -> None:
        self.tasks: dict[str, AnalysisTask] = {}
        self.reports: dict[str, CompetitorReport] = {}
        self.events: dict[str, list[AgentRunEvent]] = {}
        self.knowledge_competitors: dict[str, KnowledgeCompetitorCard] = {}
        self.knowledge_industries: dict[str, KnowledgeIndustryCard] = {}
        self.knowledge_sources: dict[str, KnowledgeSource] = {}
        self.knowledge_versions: dict[tuple[str, str], list[KnowledgeCardVersion]] = {}
        self.knowledge_suggestions: dict[str, KnowledgeUpdateSuggestion] = {}
        self.survey_designs: dict[str, SurveyDesign] = {}
        self.survey_analyses: dict[str, SurveyAnalysisResult] = {}

    def init(self) -> None:
        return None

    def save_task(self, task: AnalysisTask) -> None:
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> AnalysisTask | None:
        return self.tasks.get(task_id)

    def list_tasks(self, limit: int = 50) -> list[AnalysisTask]:
        return sorted(self.tasks.values(), key=lambda item: item.updated_at, reverse=True)[:limit]

    def save_report(self, report: CompetitorReport) -> None:
        self.reports[report.task_id] = report

    def get_report(self, task_id: str) -> CompetitorReport | None:
        return self.reports.get(task_id)

    def append_event(self, event: AgentRunEvent) -> None:
        self.events.setdefault(event.task_id, []).append(event)

    def list_events(self, task_id: str) -> list[AgentRunEvent]:
        return list(self.events.get(task_id, []))

    def save_survey_design(self, design: SurveyDesign) -> None:
        self.survey_designs[design.task_id] = design

    def get_survey_design(self, task_id: str) -> SurveyDesign | None:
        return self.survey_designs.get(task_id)

    def save_survey_analysis(self, analysis: SurveyAnalysisResult) -> None:
        self.survey_analyses[analysis.task_id] = analysis

    def get_survey_analysis(self, task_id: str) -> SurveyAnalysisResult | None:
        return self.survey_analyses.get(task_id)

    def save_knowledge_competitor_card(self, card: KnowledgeCompetitorCard, *, version_action: str = "save") -> None:
        self.knowledge_competitors[card.id] = card
        self.append_knowledge_version(KnowledgeCardVersion(card_type="competitor", card_id=card.id, action=version_action, after=card.model_dump(mode="json")))

    def get_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        return self.knowledge_competitors.get(card_id)

    def list_knowledge_competitor_cards(self, q: str = "", industry: str = "", tag: str = "", freshness_status=None, limit: int = 50) -> list[KnowledgeCompetitorCard]:
        q_lower = q.lower().strip()
        cards = [card for card in self.knowledge_competitors.values() if not card.deleted]
        if q_lower:
            cards = [card for card in cards if q_lower in " ".join([card.name, card.canonical_name, card.industry, card.market, *card.aliases, *card.tags]).lower()]
        return cards[:limit]

    def delete_knowledge_competitor_card(self, card_id: str) -> KnowledgeCompetitorCard | None:
        card = self.knowledge_competitors.get(card_id)
        if card:
            card.deleted = True
        return card

    def save_knowledge_industry_card(self, card: KnowledgeIndustryCard, *, version_action: str = "save") -> None:
        self.knowledge_industries[card.id] = card
        self.append_knowledge_version(KnowledgeCardVersion(card_type="industry", card_id=card.id, action=version_action, after=card.model_dump(mode="json")))

    def get_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        return self.knowledge_industries.get(card_id)

    def list_knowledge_industry_cards(self, q: str = "", industry: str = "", tag: str = "", freshness_status=None, limit: int = 50) -> list[KnowledgeIndustryCard]:
        q_lower = q.lower().strip()
        cards = [card for card in self.knowledge_industries.values() if not card.deleted]
        if q_lower:
            cards = [card for card in cards if q_lower in " ".join([card.title, card.canonical_name, card.industry, card.market, *card.aliases, *card.tags]).lower()]
        return cards[:limit]

    def delete_knowledge_industry_card(self, card_id: str) -> KnowledgeIndustryCard | None:
        card = self.knowledge_industries.get(card_id)
        if card:
            card.deleted = True
        return card

    def save_knowledge_source(self, source: KnowledgeSource) -> None:
        self.knowledge_sources[source.id] = source

    def get_knowledge_source(self, source_id: str) -> KnowledgeSource | None:
        return self.knowledge_sources.get(source_id)

    def list_knowledge_sources(self, source_ids: list[str] | None = None, limit: int = 100) -> list[KnowledgeSource]:
        if source_ids:
            return [source for source_id in source_ids if (source := self.knowledge_sources.get(source_id))][:limit]
        return list(self.knowledge_sources.values())[:limit]

    def append_knowledge_version(self, version: KnowledgeCardVersion) -> None:
        self.knowledge_versions.setdefault((version.card_type, version.card_id), []).append(version)

    def list_knowledge_versions(self, card_type: KnowledgeCardType, card_id: str) -> list[KnowledgeCardVersion]:
        return list(self.knowledge_versions.get((card_type, card_id), []))

    def save_knowledge_suggestion(self, suggestion: KnowledgeUpdateSuggestion) -> None:
        self.knowledge_suggestions[suggestion.id] = suggestion

    def get_knowledge_suggestion(self, suggestion_id: str) -> KnowledgeUpdateSuggestion | None:
        return self.knowledge_suggestions.get(suggestion_id)

    def list_knowledge_suggestions(self, status: str = "pending", limit: int = 100) -> list[KnowledgeUpdateSuggestion]:
        suggestions = list(self.knowledge_suggestions.values())
        if status != "all":
            suggestions = [item for item in suggestions if item.status == status]
        return suggestions[:limit]


@pytest.fixture
def app_settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key=None,
        tavily_api_key=None,
        storage_backend="json",
        sqlite_path=tmp_path / "app.db",
        json_storage_dir=tmp_path / "json",
        max_revision_loops=1,
    )


@pytest.fixture
def memory_storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def sample_request() -> CreateTaskRequest:
    return CreateTaskRequest(
        target_product="Acme Workspace",
        our_product="Acme Workspace",
        market="collaboration software",
        product_category="team productivity",
        geography="global",
        company_size="mid-market",
        competitors=["BetaSuite"],
        urls=["https://example.com/pricing"],
        competitor_profiles=[
            {
                "name": "BetaSuite",
                "website": "https://betasuite.example",
                "website_copy": "Team collaboration suite with docs, chat, and workflows.",
                "sales_notes": "Often appears in enterprise collaboration deals.",
                "win_loss_notes": "Strong templates and workflow automation.",
                "tags": ["docs", "workflow"],
            }
        ],
        notes="Focus on pricing, core features, and growth channels.",
    )


@pytest.fixture
def sample_raw_sources() -> list[RawSource]:
    return [
        RawSource(
            id="raw_home",
            url="https://betasuite.example",
            title="BetaSuite home",
            content="BetaSuite positions itself as a collaboration suite for growing teams.",
            source_type="url",
        ),
        RawSource(
            id="raw_pricing",
            url="https://betasuite.example/pricing",
            title="BetaSuite pricing",
            content="BetaSuite offers a free plan and paid business subscriptions.",
            source_type="search",
            metadata={"provider": "duckduckgo", "query": "BetaSuite pricing"},
        ),
    ]


def make_report(task_id: str = "task_test", request: CreateTaskRequest | None = None) -> CompetitorReport:
    request = request or CreateTaskRequest(target_product="Acme Workspace", competitors=["BetaSuite"])
    benchmark = AnalysisBenchmark(
        target_product=request.target_product,
        raw_description=request.raw_description or request.notes or request.target_product,
        lifecycle_stage=request.lifecycle_stage or "concept",
        analysis_purpose=request.analysis_purpose or "decision_support",
        analysis_goal=request.analysis_goal or "Support product positioning and competitor strategy.",
        confidence=0.86,
        inferred_from=["test_fixture"],
        locked=True,
    )
    candidate = CandidateCompetitor(
        name="BetaSuite",
        description="Collaboration suite for growing teams.",
        website="https://betasuite.example",
        competitor_type="head_direct",
        source_urls=["https://betasuite.example/pricing"],
        source_titles=["BetaSuite pricing"],
        tags=["user_specified"],
        rank_factors=RankFactors(heat=0.8, growth=0.55, similarity=0.72, risk=0.35),
        score=1.12,
        selection_reason="Head direct competitor selected by dynamic scoring.",
    )
    selected_set = SelectedCompetitorSet(
        total_budget=3,
        scoring_profile=scoring_profile_for("decision_support"),
        candidates=[candidate],
        selected=[candidate],
        allocation={"direct": 2, "indirect_cross": 1},
        fallback_notes=["Test fixture uses one selected competitor."],
    )
    tool_plan = build_tool_plan(benchmark, selected_set, request)
    collection_plan = build_collection_plan(benchmark, selected_set, tool_plan)
    search_plan = SearchPlan(
        queries=[
            SearchQuery(
                query="BetaSuite pricing features reviews",
                tool=tool_plan.selected_tools[0],
                competitor="BetaSuite",
                target_site_types=["official_site", "review"],
                priority=1,
                expected_fields=["pricing", "feature"],
                rationale="Test fixture search plan.",
            )
        ],
        rationale="Test search plan generated from selected tools.",
    )
    evidence = EvidenceItem(
        id="ev_beta",
        url="https://betasuite.example/pricing",
        title="BetaSuite pricing",
        source_type="url",
        excerpt="BetaSuite offers a free plan and paid business subscriptions.",
        credibility=0.82,
        related_fields=["pricing", "features"],
    )
    pricing_reference_id = "card.competitor.betasuite.pricing"
    industry_reference_id = "card.industry.market_situation.market_size"
    competitor_detail_card_clean = CompetitorDetailCardClean(
        competitor="BetaSuite",
        competitor_type="head_direct",
        fields={
            "positioning": "Collaboration suite for growing teams",
            "pricing": "Free plan plus paid business subscriptions",
        },
        reference_ids={
            "positioning": "card.competitor.betasuite.positioning",
            "pricing": pricing_reference_id,
        },
    )
    industry_card_clean = IndustryCardClean(
        fields={"market_situation.market_size": "Collaboration software demand is validated by pricing-page competition."},
        reference_ids={"market_situation.market_size": industry_reference_id},
    )
    report_references = [
        ReportReference(
            id="card.competitor.betasuite.positioning",
            title="BetaSuite positioning",
            reference_type="clean_card",
            anchor="ref-card-competitor-betasuite-positioning",
            summary="Collaboration suite for growing teams",
            payload={"field": "positioning", "competitor": "BetaSuite"},
        ),
        ReportReference(
            id=pricing_reference_id,
            title="BetaSuite pricing",
            reference_type="clean_card",
            anchor="ref-card-competitor-betasuite-pricing",
            summary="Free plan plus paid business subscriptions",
            payload={"field": "pricing", "competitor": "BetaSuite"},
        ),
        ReportReference(
            id=industry_reference_id,
            title="Industry market size",
            reference_type="clean_card",
            anchor="ref-card-industry-market-situation-market-size",
            summary="Collaboration software demand is validated by pricing-page competition.",
            payload={"field": "market_situation.market_size"},
        ),
    ]
    profile = CompetitorProfile(
        name="BetaSuite",
        website="https://betasuite.example",
        positioning="Collaboration suite for growing teams",
        target_users=["Growing teams", "Mid-market companies"],
        core_features=["Docs", "Chat", "Workflow automation"],
        pricing="Free plan plus paid business subscriptions",
        business_model="Subscription",
        channels=["Website", "Product-led growth"],
        growth_signals=["Public pricing page", "Template gallery"],
        evidence_ids=[evidence.id],
        competitor_type="head_direct",
        score=1.12,
        selection_reason="Head direct competitor selected by dynamic scoring.",
        rank_factors=RankFactors(heat=0.8, growth=0.55, similarity=0.72, risk=0.35),
    )
    report = CompetitorReport(
        task_id=task_id,
        target_product=request.target_product,
        competitors=[profile],
        executive_summary=(
            "This report compares Acme Workspace with BetaSuite across positioning, pricing, "
            "product capabilities, target users, channels, and growth signals using cited evidence."
        ),
        findings=[
            AnalysisFinding(
                id="finding_pricing",
                category="pricing",
                title="BetaSuite uses a freemium subscription motion",
                conclusion="The pricing evidence shows a free plan and paid business subscriptions.",
                evidence_ids=[evidence.id],
                confidence=0.82,
                impact="high",
            ),
            AnalysisFinding(
                id="finding_positioning",
                category="positioning",
                title="BetaSuite competes directly on collaboration positioning",
                conclusion="The cited source describes BetaSuite as a collaboration suite for growing teams.",
                evidence_ids=[evidence.id],
                confidence=0.78,
                impact="medium",
            ),
            AnalysisFinding(
                id="finding_action",
                category="strategy",
                title="Pricing should be tracked as a recurring signal",
                conclusion="Because the cited pricing source mentions free and paid plans, monthly monitoring can detect go-to-market changes.",
                evidence_ids=[evidence.id],
                confidence=0.76,
                impact="medium",
            ),
        ],
        swot=SWOT(
            strengths=["Clear collaboration positioning"],
            weaknesses=["Pricing depth still needs validation"],
            opportunities=["Differentiate workflow automation"],
            threats=["Freemium plan lowers adoption friction"],
        ),
        recommendations=["Track pricing page changes monthly using the cited pricing evidence ev_beta."],
        evidence=[evidence],
        analysis_benchmark=benchmark,
        selected_competitor_set=selected_set,
        tool_plan=tool_plan,
        collection_plan=collection_plan,
        search_plan=search_plan,
        source_provider_results=[
            SourceProviderResult(
                provider="cache",
                sources=[],
                signals=[],
                cache_hit=True,
            )
        ],
        tool_results=[
            ToolExecutionResult(
                tool_name=tool_plan.selected_tools[0],
                canonical_tool_name="swot",
                schema_name="test_schema",
                input_reference_ids=[pricing_reference_id],
                claims=[
                    ToolClaim(
                        category="pricing",
                        title="Freemium subscription motion",
                        claim="BetaSuite pricing evidence shows a free plan and paid subscriptions.",
                        reference_ids=[pricing_reference_id],
                        confidence=0.82,
                        reasoning="Directly cited from the pricing evidence.",
                    )
                ],
                data={"claims": ["Freemium subscription motion"]},
                evidence_ids=[],
                confidence=0.82,
                reasoning="Fixture tool execution result.",
                generated_by="llm",
            )
        ],
        tool_sandboxes=[
            ToolSandbox(
                tool=tool_plan.selected_tools[0],
                schema_name="test_schema",
                input_reference_ids=[pricing_reference_id],
                data={},
                evidence_ids=[],
            )
        ],
        competitor_detail_cards_clean=[competitor_detail_card_clean],
        industry_card_clean=industry_card_clean,
        report_references=report_references,
    )
    report.deep_dive_markdown = build_deep_dive_markdown(report, request)
    return report
