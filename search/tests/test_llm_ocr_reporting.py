import json
from pathlib import Path

from myagent.search import chunking, llm, ocr, reporting
from myagent.search.models import SearchResult, TextChunk


def make_chunk(**overrides: object) -> TextChunk:
    data = {
        "chunk_id": "chunk_0001",
        "source_result_id": "duckduckgo_text_1_abcd1234",
        "method": "duckduckgo_text",
        "rank": 1,
        "title": "Demo",
        "url": "https://example.com",
        "content_path": "contents/chunk.txt",
        "start_char": 10,
        "end_char": 40,
        "text": "Alpha product has pricing details.",
        "text_sources": ["dom"],
        "chunk_shot_path": "chunk_shots/shot.png",
    }
    data.update(overrides)
    return TextChunk(**data)


def test_run_ocr_for_chunks_appends_only_extra_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunk = make_chunk(text="Alpha product", end_char=23)
    (tmp_path / "chunk_shots").mkdir()
    (tmp_path / chunk.chunk_shot_path).write_bytes(b"fake")
    monkeypatch.setattr(ocr, "extract_ocr_text", lambda image_path, engine, lang: ("Beta pricing table", None))

    errors = ocr.run_ocr_for_chunks(
        [chunk],
        tmp_path,
        enabled=True,
        engine="tesseract",
        lang="eng",
        min_extra_chars=1,
    )

    assert errors == []
    assert "ocr" in (chunk.text_sources or [])
    assert "Beta pricing table" in chunk.text
    assert (tmp_path / "extractions" / "chunks.json").exists()


def test_normalize_and_merge_extractions_dedupe_facts() -> None:
    chunk = make_chunk()
    raw = {
        "summary": "Useful summary",
        "facts": [
            {
                "field": "pricing",
                "subject": "Product",
                "claim": "Has pricing details",
                "evidence_quote": "pricing details",
                "evidence_source": "dom",
                "confidence": "0.8",
            }
        ],
        "entities": [
            {
                "name": "Product",
                "type": "product",
                "description": "Demo product",
                "evidence_quote": "Alpha product",
                "evidence_source": "dom",
            }
        ],
    }

    normalized = llm.normalize_chunk_extraction(raw, chunk)
    merged = llm.merge_extractions("demo", [chunk], [normalized, normalized], [])

    assert normalized["facts"][0]["confidence"] == 0.8
    assert normalized["facts"][0]["evidence_start_char"] is not None
    assert len(merged["facts"]) == 1
    assert len(merged["entities"]) == 1
    assert merged["evidence"][0]["fact_id"] == normalized["facts"][0]["id"]


def test_reporting_writes_results_and_report(tmp_path: Path) -> None:
    result = SearchResult(
        method="duckduckgo_text",
        query="demo",
        rank=1,
        title="Title",
        url="https://example.com",
        snippet="Snippet",
    )

    json_path = reporting.write_json(
        run_dir=tmp_path,
        query="demo",
        methods=["duckduckgo_text"],
        max_results=1,
        screenshots_mode="none",
        full_page=False,
        content_mode="none",
        content_max_chars=12000,
        search_page_screenshots=[],
        results=[result],
    )
    report_path = reporting.write_report(
        run_dir=tmp_path,
        query="demo",
        methods=["duckduckgo_text"],
        search_page_screenshots=[],
        results=[result],
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["url"] == "https://example.com"
    assert "# Search Report: demo" in report_path.read_text(encoding="utf-8")


def test_chunk_text_consumes_tavily_content_file(tmp_path: Path) -> None:
    content_dir = tmp_path / "contents"
    content_dir.mkdir()
    content_path = content_dir / "tavily.txt"
    content_path.write_text("Alpha pricing details. Beta customer story.", encoding="utf-8")
    result = SearchResult(
        method="tavily",
        query="demo",
        rank=1,
        title="Title",
        url="https://example.com",
        snippet="Snippet",
        content_path="contents/tavily.txt",
        content_source="tavily_extract",
    )

    chunks = chunking.chunk_text([result], tmp_path, chunk_size=20, chunk_overlap=5)

    assert chunks
    assert chunks[0].text_sources == ["tavily_extract"]
    assert "Alpha pricing" in chunks[0].text
