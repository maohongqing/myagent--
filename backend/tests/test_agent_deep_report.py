from __future__ import annotations

from myagent.backend.app.agents.deep_report import (
    DEEP_DIVE_SECTIONS,
    build_deep_dive_markdown,
    has_required_deep_dive_sections,
)


def test_deep_report_builder_includes_required_sections_context_and_evidence(sample_request):
    from conftest import make_report

    report = make_report("task_deep_report", sample_request)
    markdown = build_deep_dive_markdown(report, sample_request)

    assert markdown.startswith("# ")
    for section in DEEP_DIVE_SECTIONS:
        assert f"## {section}" in markdown
    assert sample_request.target_product in markdown
    assert "BetaSuite" in markdown
    assert "ev_beta" in markdown
    assert has_required_deep_dive_sections(markdown) is True


def test_deep_report_section_checker_rejects_incomplete_markdown():
    assert has_required_deep_dive_sections("# Title\n\n## Only one section\n") is False
