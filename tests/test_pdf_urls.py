"""Offline tests for direct PDF URL recognition across search providers."""

from app.tools.pdf_urls import find_pdf_url, normalize_pdf_url


def test_recognizes_direct_pdf_and_query_pdf_endpoints():
    assert normalize_pdf_url("https://example.org/papers/study.pdf?download=1") == (
        "https://example.org/papers/study.pdf?download=1"
    )
    assert normalize_pdf_url("https://openreview.net/pdf?id=paper-id") == (
        "https://openreview.net/pdf?id=paper-id"
    )


def test_converts_arxiv_abstract_url_to_pdf_url():
    assert normalize_pdf_url("https://arxiv.org/abs/2501.01234v2") == (
        "https://arxiv.org/pdf/2501.01234v2"
    )


def test_finds_pdf_in_nested_provider_links():
    links = [
        {"@ref": "scopus", "@href": "https://example.org/article"},
        {"@ref": "full-text", "@href": "https://example.org/article.pdf"},
    ]

    assert find_pdf_url(links) == "https://example.org/article.pdf"


def test_ignores_normal_web_pages_and_non_http_urls():
    assert normalize_pdf_url("https://example.org/article") is None
    assert normalize_pdf_url("https://arxiv.org/abs/tavily:web-result") is None
    assert normalize_pdf_url("file:///tmp/article.pdf") is None
