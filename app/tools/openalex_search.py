"""OpenAlex scholarly search integration.

OpenAlex indexes journals, conference proceedings and arXiv, so it keeps
scholarly recall when arXiv's own API is throttling this host. Queries use
OpenAlex's semantic search, which ranks by embedding similarity instead of the
keyword search's citation-weighted score: a planner's natural-language query
otherwise returns broad, highly cited surveys ahead of the specific work.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

import requests

from ..utils import logger, redact_secrets

_BASE_URL = "https://api.openalex.org/works"
_SELECT = ",".join(
    (
        "id",
        "doi",
        "title",
        "publication_date",
        "publication_year",
        "authorships",
        "primary_location",
        "best_oa_location",
        "locations",
        "abstract_inverted_index",
    )
)
_USER_AGENT = "open-ai-co-scientist/1.0 (+https://github.com/Siong23/ai-co-scientist)"
_DEFAULT_TIMEOUT = 30
# Semantic search is limited to one request per second and 50 results.
_MIN_REQUEST_INTERVAL_SECONDS = 1.1
_MAX_SEMANTIC_RESULTS = 50
_MAX_QUERY_CHARS = 2000
# An arXiv preprint's DataCite DOI; the arXiv identifier names the same paper.
_ARXIV_DOI_PREFIX = "10.48550/arxiv."
_ARXIV_URL_ID = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)

_request_slot_lock = threading.Lock()
_next_request_at = 0.0


def _wait_for_request_slot() -> None:
    """Keep every OpenAlex call in the process within the per-second limit."""

    global _next_request_at
    with _request_slot_lock:
        delay = _next_request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _next_request_at = time.monotonic() + _MIN_REQUEST_INTERVAL_SECONDS


def reconstruct_abstract(inverted_index: Any) -> str:
    """Rebuild plain text from OpenAlex's word -> positions abstract index."""

    if not isinstance(inverted_index, dict):
        return ""
    positioned = [
        (position, str(word))
        for word, positions in inverted_index.items()
        for position in (positions or ())
        if isinstance(position, int)
    ]
    return " ".join(word for _, word in sorted(positioned))


def _arxiv_id(work: dict[str, Any], doi: str | None) -> str | None:
    """Return the arXiv identifier of a work that has an arXiv version."""

    if doi and doi.casefold().startswith(_ARXIV_DOI_PREFIX):
        return doi[len(_ARXIV_DOI_PREFIX) :]
    locations = [work.get("primary_location"), work.get("best_oa_location"), *(work.get("locations") or ())]
    for location in locations:
        if not isinstance(location, dict):
            continue
        for url in (location.get("landing_page_url"), location.get("pdf_url")):
            match = _ARXIV_URL_ID.search(str(url or ""))
            if match:
                return match.group(1)
    return None


class OpenAlexSearchTool:
    """Search OpenAlex and normalize results for the RAG pipeline.

    ``OPENALEX_API_KEY`` is optional: without it OpenAlex allows a small daily
    budget, and a free key raises that budget tenfold.
    """

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results
        self.api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
        self.last_error_status: int | None = None
        self.last_error_kind = ""
        self.last_error_detail = ""

    def search_papers(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Return works with abstracts matching ``query``, or an empty list on failure."""

        query = " ".join(str(query).split())
        if not query:
            return []
        self.last_error_status = None
        self.last_error_kind = ""
        self.last_error_detail = ""
        limit = max(1, min(max_results if max_results is not None else self.max_results, _MAX_SEMANTIC_RESULTS))
        headers = {"User-Agent": _USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        _wait_for_request_slot()
        try:
            response = requests.get(
                _BASE_URL,
                params={"search.semantic": query[:_MAX_QUERY_CHARS], "per-page": limit, "select": _SELECT},
                headers=headers,
                timeout=_DEFAULT_TIMEOUT,
            )
        except requests.Timeout as exc:
            self.last_error_kind = "timeout"
            self.last_error_detail = redact_secrets(str(exc))
            logger.warning("OpenAlex search timed out for query %r.", query)
            return []
        except requests.RequestException as exc:
            self.last_error_kind = "provider_error"
            self.last_error_detail = redact_secrets(str(exc))
            logger.warning("OpenAlex search failed for query %r: %s", query, self.last_error_detail)
            return []

        if response.status_code != 200:
            self.last_error_status = response.status_code
            if response.status_code == 429 and self._daily_budget_spent(response):
                # A spent daily budget keeps answering 429 until it resets, so
                # it needs the long cooldown rather than a per-second retry.
                self.last_error_kind = "quota_or_plan_rejection"
            elif response.status_code == 429:
                self.last_error_kind = "rate_limited"
            else:
                self.last_error_kind = "provider_error"
            self.last_error_detail = redact_secrets(f"HTTP {response.status_code}: {response.text[:300]}")
            logger.warning("OpenAlex search failed for query %r: %s", query, self.last_error_detail)
            return []

        try:
            works = response.json().get("results") or []
        except ValueError as exc:
            self.last_error_kind = "provider_error"
            self.last_error_detail = redact_secrets(str(exc))
            logger.warning("OpenAlex returned invalid JSON for query %r.", query)
            return []

        papers = [paper for paper in (self._format_work(work) for work in works if isinstance(work, dict)) if paper]
        logger.debug("OpenAlex returned %d usable work(s) for query %r.", len(papers), query)
        return papers

    @staticmethod
    def _daily_budget_spent(response: requests.Response) -> bool:
        remaining = response.headers.get("X-RateLimit-Remaining-USD")
        try:
            return remaining is not None and float(remaining) <= 0
        except ValueError:
            return False

    @staticmethod
    def _format_work(work: dict[str, Any]) -> dict[str, Any] | None:
        """Convert an OpenAlex work into the arXiv-compatible shared schema."""

        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        title = " ".join(str(work.get("title") or "").split())
        if not abstract or not title:
            # Screening and grading read the abstract; a title alone cannot be judged.
            return None

        doi = str(work.get("doi") or "").strip()
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE) or None
        arxiv_id = _arxiv_id(work, doi)
        if doi and doi.casefold().startswith(_ARXIV_DOI_PREFIX):
            # Keep the arXiv identifier as the identity so the same preprint
            # found through arXiv itself merges with this result.
            doi = None

        primary = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
        best_oa = work.get("best_oa_location") if isinstance(work.get("best_oa_location"), dict) else {}
        venue = str((primary.get("source") or {}).get("display_name") or "").strip() or None
        openalex_id = str(work.get("id") or "").rsplit("/", 1)[-1]

        if arxiv_id:
            source_id = arxiv_id
            url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        else:
            source_id = f"doi:{doi}" if doi else f"openalex:{openalex_id}"
            url = str(primary.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else work.get("id") or ""))
            pdf_url = best_oa.get("pdf_url") or primary.get("pdf_url")

        year = work.get("publication_year")
        published = work.get("publication_date") or (f"{year}-01-01" if year else None)
        authors = [
            str((authorship.get("author") or {}).get("display_name") or "").strip()
            for authorship in (work.get("authorships") or ())
            if isinstance(authorship, dict)
        ]
        return {
            "arxiv_id": source_id,
            "entry_id": url,
            "title": title,
            "abstract": abstract,
            "authors": [author for author in authors if author],
            "primary_category": venue or "general",
            "categories": [],
            "published": published,
            "updated": published,
            "doi": doi,
            "pdf_url": str(pdf_url).strip() if pdf_url else None,
            "arxiv_url": url,
            "comment": None,
            "journal_ref": venue,
            "venue": venue,
            "source": "openalex",
        }
