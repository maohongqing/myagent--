import argparse

import pytest

from myagent.search import utils


def test_parse_methods_supports_all_and_rejects_unknown() -> None:
    assert utils.parse_methods("all") == utils.DEFAULT_METHODS
    assert utils.parse_methods("duckduckgo_text,tavily") == ["duckduckgo_text", "tavily"]

    with pytest.raises(argparse.ArgumentTypeError):
        utils.parse_methods("duckduckgo_text,unknown")


def test_clean_text_and_page_text_normalize_whitespace() -> None:
    assert utils.clean_text("  a\n\t b  ") == "a b"
    assert utils.clean_page_text(" a \r\n\r\n\r\n b\t c ", max_chars=20) == "a \n\n b c"


def test_is_public_http_url_blocks_private_and_local_urls() -> None:
    assert utils.is_public_http_url("https://example.com/path")
    assert not utils.is_public_http_url("ftp://example.com/file")
    assert not utils.is_public_http_url("http://localhost:8000")
    assert not utils.is_public_http_url("http://127.0.0.1")
    assert not utils.is_public_http_url("http://192.168.1.1")


def test_extract_json_object_accepts_fenced_and_embedded_json() -> None:
    assert utils.extract_json_object('```json\n{"ok": true}\n```') == {"ok": True}
    assert utils.extract_json_object('prefix {"value": 3} suffix') == {"value": 3}


def test_quote_match_handles_normalized_dom_text() -> None:
    dom_nodes = [
        {"text": "Alpha beta", "start": 0, "end": 10, "top": 100, "bottom": 120},
        {"text": "Gamma", "start": 11, "end": 16, "top": 130, "bottom": 150},
    ]

    match = utils.find_quote_match(dom_nodes, "Alpha  beta", min_score=0.72)

    assert match is not None
    assert match.method == "exact"
    assert match.score == 1.0
