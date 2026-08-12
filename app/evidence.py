"""Provider-neutral evidence records for academic and web retrieval."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

EvidenceType = Literal["academic", "web"]
EvidenceDocumentType = Literal[
    "academic_paper",
    "official_docs",
    "technical_blog",
    "news",
    "webpage",
]

_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}


def canonicalize_url(url: str) -> str:
    """Return a stable URL for web deduplication without tracking parameters."""

    normalized = url.strip()
    if not normalized:
        return ""
    parsed = urlsplit(normalized)
    host = (parsed.hostname or "").casefold()
    if (
        parsed.port
        and not (parsed.scheme.casefold() == "http" and parsed.port == 80)
        and not (parsed.scheme.casefold() == "https" and parsed.port == 443)
    ):
        host = f"{host}:{parsed.port}"
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_") and key.casefold() not in _TRACKING_QUERY_KEYS
        )
    )
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.casefold(), host, path, query, ""))


@dataclass(frozen=True)
class EvidenceDocument:
    """One provider-neutral document that can support a research claim."""

    source_id: str
    source_family: EvidenceType
    document_type: EvidenceDocumentType
    provider: str
    title: str
    url: str
    summary: str
    content: str = ""
    published_at: str | None = None
    updated_at: str | None = None
    authors: tuple[str, ...] = ()
    pdf_url: str | None = None
    doi: str | None = None
    venue: str | None = None
    canonical_url: str | None = None
    domain: str | None = None
    page_type: str | None = None
    author: str | None = None
    language: str | None = None
    source_authority: str | None = None
    retrieval_query: str = ""
    sub_question: str | None = None
    purpose: str = ""
    evidence_requirement_id: str | None = None
    search_score: float | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None
    full_text_available: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_type(self) -> EvidenceType:
        """Compatibility name for the broad academic/web source family."""

        return self.source_family

    @property
    def text(self) -> str:
        """Best available bounded-text candidate for ranking and grading."""

        return self.content or self.summary

    @property
    def canonical_key(self) -> str:
        """Cross-query identity used by reciprocal-rank fusion."""

        if self.source_family == "web":
            return self.canonical_url or canonicalize_url(self.url) or self.source_id
        if self.doi:
            return f"doi:{self.doi.strip().casefold()}"
        source_id = self.source_id.strip()
        if source_id.casefold().startswith("arxiv:"):
            source_id = re.sub(r"v\d+$", "", source_id, flags=re.IGNORECASE)
        return source_id.casefold()

    def with_rrf_score(self, score: float) -> "EvidenceDocument":
        return replace(self, rrf_score=score)


# Compatibility alias for integrations using the previous provider-neutral name.
EvidenceSource = EvidenceDocument


@dataclass(frozen=True)
class EvidenceChunk:
    """One independently rankable passage extracted from an evidence document."""

    chunk_id: str
    source_id: str
    content: str
    retrieval_query: str
    sub_question: str | None
    document_type: EvidenceDocumentType
    chunk_index: int
    chunk_count: int
    rerank_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _source_id_from_academic_result(result: Mapping[str, Any], provider: str) -> str:
    raw_id = str(result.get("source_id") or result.get("arxiv_id") or "").strip()
    if not raw_id:
        url = str(result.get("arxiv_url") or result.get("entry_id") or "").strip()
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"{provider}:{digest}"
    known_prefixes = ("arxiv:", "s2:", "springer:", "elsevier:", "doi:")
    if raw_id.casefold().startswith(known_prefixes):
        if raw_id.casefold().startswith("arxiv:"):
            return f"arXiv:{raw_id.split(':', 1)[1]}"
        return raw_id
    is_arxiv_id = bool(re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", raw_id, re.IGNORECASE))
    return f"arXiv:{raw_id}" if provider == "arxiv" or result.get("source") == "arxiv" or is_arxiv_id else raw_id


def academic_evidence_from_result(
    result: Mapping[str, Any],
    provider: str,
) -> EvidenceDocument:
    """Adapt the existing academic-provider dictionaries at one boundary."""

    url = str(result.get("url") or result.get("arxiv_url") or result.get("entry_id") or "").strip()
    published = result.get("published_at") or result.get("published")
    updated = result.get("updated_at") or result.get("updated")
    authors = tuple(str(author).strip() for author in (result.get("authors") or ()) if str(author).strip())
    content = str(result.get("content") or "").strip()
    canonical_url = canonicalize_url(url)
    return EvidenceDocument(
        source_id=_source_id_from_academic_result(result, provider),
        source_family="academic",
        document_type="academic_paper",
        provider=str(result.get("provider") or result.get("source") or provider).strip(),
        title=str(result.get("title") or "Untitled academic source").strip(),
        url=url,
        summary=str(result.get("summary") or result.get("abstract") or "").strip(),
        content=content,
        published_at=str(published).strip() if published else None,
        updated_at=str(updated).strip() if updated else None,
        authors=authors,
        pdf_url=str(result.get("pdf_url") or "").strip() or None,
        doi=str(result.get("doi") or "").strip() or None,
        venue=str(result.get("venue") or result.get("journal_ref") or result.get("primary_category") or "").strip()
        or None,
        canonical_url=canonical_url or None,
        domain=(urlsplit(canonical_url or url).hostname or "").casefold() or None,
        retrieval_query=str(result.get("retrieval_query") or result.get("search_query") or "").strip(),
        sub_question=str(result.get("sub_question") or "").strip() or None,
        purpose=str(result.get("purpose") or "").strip(),
        evidence_requirement_id=str(result.get("evidence_requirement_id") or "").strip() or None,
        search_score=(
            float(result["search_score"])
            if isinstance(result.get("search_score"), (int, float))
            else None
        ),
        rerank_score=(
            float(result["rerank_score"])
            if isinstance(result.get("rerank_score"), (int, float))
            else None
        ),
        full_text_available=bool(
            result.get("full_text_available")
            or result.get("full_text_indexed")
            or content
        ),
        metadata=dict(result),
    )


def _web_document_type(result: Mapping[str, Any]) -> EvidenceDocumentType:
    explicit_type = str(result.get("document_type") or "").strip().casefold()
    if explicit_type in {
        "academic_paper",
        "official_docs",
        "technical_blog",
        "news",
        "webpage",
    }:
        return cast(EvidenceDocumentType, explicit_type)

    route = str(result.get("planned_source_type") or "").strip().casefold()
    if route == "official":
        return "official_docs"
    if route == "news":
        return "news"

    page_type = str(result.get("page_type") or "").strip().casefold()
    url = str(result.get("url") or result.get("canonical_url") or "").casefold()
    if any(marker in page_type for marker in ("official", "government", "documentation", "guidance")):
        return "official_docs"
    if "news" in page_type:
        return "news"
    if "blog" in page_type or "/blog" in url:
        return "technical_blog"
    return "webpage"


def web_evidence_from_result(
    result: Mapping[str, Any],
    provider: str = "tavily",
) -> EvidenceDocument:
    """Normalize a web result without pretending that it is a paper."""

    url = str(
        result.get("url") or result.get("canonical_url") or result.get("entry_id") or result.get("arxiv_url") or ""
    ).strip()
    canonical_url = canonicalize_url(str(result.get("canonical_url") or url))
    legacy_id = str(result.get("arxiv_id") or "").strip()
    source_id = str(
        result.get("source_id") or (legacy_id if legacy_id.casefold().startswith(("web:", "tavily:")) else "")
    ).strip()
    if not source_id:
        source_id = f"web:{hashlib.sha256(canonical_url.encode()).hexdigest()[:16]}"
    parsed = urlsplit(canonical_url or url)
    published = result.get("published_at") or result.get("published_date") or result.get("published")
    updated = result.get("updated_at") or result.get("updated") or published
    score = result.get("search_score", result.get("score"))
    content = str(result.get("content") or result.get("raw_content") or result.get("abstract") or "").strip()
    return EvidenceDocument(
        source_id=source_id,
        source_family="web",
        document_type=_web_document_type(result),
        provider=str(result.get("provider") or result.get("source") or provider).strip(),
        title=str(result.get("title") or "Untitled web source").strip(),
        url=url,
        summary=str(
            result.get("snippet") or result.get("summary") or result.get("abstract") or result.get("content") or ""
        ).strip(),
        content=content,
        published_at=str(published).strip() if published else None,
        updated_at=str(updated).strip() if updated else None,
        pdf_url=str(result.get("pdf_url") or "").strip() or None,
        canonical_url=canonical_url or None,
        domain=str(result.get("domain") or parsed.hostname or "").casefold() or None,
        page_type=str(result.get("page_type") or "web_page").strip(),
        author=str(result.get("author") or "").strip() or None,
        language=str(result.get("language") or "").strip() or None,
        source_authority=str(result.get("source_authority") or "").strip() or None,
        retrieval_query=str(result.get("retrieval_query") or result.get("search_query") or "").strip(),
        sub_question=str(result.get("sub_question") or "").strip() or None,
        purpose=str(result.get("purpose") or "").strip(),
        evidence_requirement_id=str(result.get("evidence_requirement_id") or "").strip() or None,
        search_score=float(score) if isinstance(score, (int, float)) else None,
        rerank_score=(
            float(result["rerank_score"])
            if isinstance(result.get("rerank_score"), (int, float))
            else None
        ),
        full_text_available=bool(
            result.get("full_text_available")
            or result.get("content_extracted")
            or result.get("raw_content")
        ),
        metadata=dict(result),
    )


def evidence_from_result(
    result: EvidenceDocument | Mapping[str, Any],
    provider: str,
    source_type: EvidenceType,
) -> EvidenceDocument:
    if isinstance(result, EvidenceDocument):
        return result
    if source_type == "web":
        return web_evidence_from_result(result, provider)
    return academic_evidence_from_result(result, provider)


def coerce_evidence(result: EvidenceDocument | Mapping[str, Any]) -> EvidenceDocument:
    """Normalize either the new model or a legacy provider dictionary."""

    if isinstance(result, EvidenceDocument):
        return result
    provider = str(result.get("provider") or result.get("source") or "arxiv").strip()
    source_id = str(result.get("source_id") or result.get("arxiv_id") or "").casefold()
    source_type: EvidenceType = (
        "web"
        if result.get("source_type") == "web"
        or result.get("source_family") == "web"
        or result.get("document_type") in {"official_docs", "technical_blog", "news", "webpage"}
        or provider.casefold() == "tavily"
        or source_id.startswith(("web:", "tavily:"))
        else "academic"
    )
    return evidence_from_result(result, provider, source_type)
