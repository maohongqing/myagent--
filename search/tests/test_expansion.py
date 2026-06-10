from myagent.search import expansion
from myagent.search.models import LinkCandidate, LinkDecision, SearchResult


def test_pagination_candidate_detection() -> None:
    assert expansion.is_pagination_candidate("https://example.com/page/2", "", "")
    assert expansion.is_pagination_candidate("https://example.com/news", "下一页", "")
    assert expansion.is_pagination_candidate("https://example.com/news", "", "next")
    assert not expansion.is_pagination_candidate("https://example.com/pricing", "Pricing", "")


def test_filter_candidates_respects_same_domain_and_low_value_links() -> None:
    candidates = [
        LinkCandidate(id="a", source_url="https://example.com", url="https://example.com/product", text="Product"),
        LinkCandidate(id="b", source_url="https://example.com", url="https://other.com/product", text="Product"),
        LinkCandidate(id="c", source_url="https://example.com", url="https://example.com/privacy", text="Privacy"),
    ]

    filtered = expansion.filter_candidates(
        candidates,
        source_url="https://example.com",
        link_scope="same_domain",
        allowlist=set(),
        include_pagination=False,
        include_relevant_links=True,
    )

    assert [item.id for item in filtered] == ["a"]


def test_filter_candidates_respects_allowlist() -> None:
    candidates = [
        LinkCandidate(id="a", source_url="https://example.com", url="https://allowed.com/product", text="Product"),
        LinkCandidate(id="b", source_url="https://example.com", url="https://other.com/product", text="Product"),
    ]

    filtered = expansion.filter_candidates(
        candidates,
        source_url="https://example.com",
        link_scope="allowlist",
        allowlist={"allowed.com"},
        include_pagination=False,
        include_relevant_links=True,
    )

    assert [item.id for item in filtered] == ["a"]


def test_decide_links_without_llm_key_does_not_select(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    candidate = LinkCandidate(
        id="a",
        source_url="https://example.com",
        url="https://example.com/product",
        text="Product",
    )

    decisions, errors = expansion.decide_links("query", [candidate], filter_method="llm", selected_limit=1)

    assert decisions[0].selected is False
    assert errors[0]["stage"] == "link_llm_init"


def test_result_from_decision_keeps_lineage() -> None:
    source = SearchResult(
        method="duckduckgo_text",
        query="query",
        rank=1,
        title="Source",
        url="https://example.com",
        snippet="",
    )
    decision = LinkDecision(
        id="a",
        source_url="https://example.com",
        url="https://example.com/product",
        text="Product",
        discovery_type="relevant_link",
        selected=True,
        relevance=0.8,
        reason="Useful product details",
    )

    result = expansion.result_from_decision(source, decision, rank=2)

    assert result.source_url == source.url
    assert result.depth == 1
    assert result.link_relevance == 0.8
