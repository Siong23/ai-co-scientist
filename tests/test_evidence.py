"""Tests for provider-neutral academic and web evidence records."""

from app.evidence import (
    EvidenceDocument,
    academic_evidence_from_result,
    canonicalize_url,
    web_evidence_from_result,
)


def test_canonicalize_url_removes_fragment_tracking_and_default_port():
    assert (
        canonicalize_url("HTTPS://Example.COM:443/article/?utm_source=test&b=2&a=1#section")
        == "https://example.com/article?a=1&b=2"
    )


def test_academic_evidence_preserves_paper_specific_metadata():
    evidence = academic_evidence_from_result(
        {
            "arxiv_id": "2501.01234v2",
            "title": "A research paper",
            "abstract": "Academic evidence.",
            "authors": ["Alice Researcher"],
            "published": "2025-01-01",
            "doi": "10.1000/example",
            "journal_ref": "Example Journal",
            "arxiv_url": "https://arxiv.org/abs/2501.01234v2",
            "pdf_url": "https://arxiv.org/pdf/2501.01234v2",
            "source": "arxiv",
        },
        "arxiv",
    )

    assert evidence.source_id == "arXiv:2501.01234v2"
    assert evidence.source_type == "academic"
    assert isinstance(evidence, EvidenceDocument)
    assert evidence.source_family == "academic"
    assert evidence.document_type == "academic_paper"
    assert evidence.full_text_available is False
    assert evidence.canonical_url == "https://arxiv.org/abs/2501.01234v2"
    assert evidence.domain == "arxiv.org"
    assert evidence.doi == "10.1000/example"
    assert evidence.venue == "Example Journal"
    assert evidence.authors == ("Alice Researcher",)


def test_web_evidence_has_web_metadata_without_academic_fields():
    evidence = web_evidence_from_result(
        {
            "source_id": "web:guidance",
            "title": "Current guidance",
            "url": "https://agency.example/guidance?utm_campaign=test",
            "snippet": "Short result snippet.",
            "content": "Full retrieved page content.",
            "content_extracted": True,
            "updated_at": "2026-08-01",
            "page_type": "government_guidance",
            "score": 0.8,
        }
    )

    assert evidence.source_type == "web"
    assert evidence.source_family == "web"
    assert evidence.document_type == "official_docs"
    assert evidence.full_text_available is True
    assert evidence.canonical_url == "https://agency.example/guidance"
    assert evidence.domain == "agency.example"
    assert evidence.text == "Full retrieved page content."
    assert evidence.doi is None
    assert evidence.venue is None
