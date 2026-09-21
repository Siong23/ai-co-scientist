"""Elsevier Scopus API integration for literature search and retrieval."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import requests

from .pdf_urls import find_pdf_url

_DEFAULT_TIMEOUT = 15
_DEFAULT_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
# Scopus reads these as search operators, so a query that merely contains one
# in prose is rejected with HTTP 400 "Error translating query".
_SCOPUS_OPERATORS = frozenset({"and", "or", "not", "pre", "w", "near", "onear"})
# Every remaining term is ANDed together, so prose filler collapses the result
# set to nothing. Content words are kept and the rest is dropped.
_SCOPUS_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "for",
        "with",
        "by",
        "from",
        "in",
        "on",
        "at",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "that",
        "this",
        "these",
        "those",
        "its",
        "their",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "will",
        "shall",
        "develop",
        "capable",
        "according",
        "using",
        "based",
        "toward",
        "towards",
        "under",
        "into",
    }
)
# Six content words still match broadly; eight already return zero on Scopus.
_MAX_QUERY_WORDS = 6


class ElsevierSearchTool:
    """Search Scopus and normalize its records to the shared literature schema."""

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results
        self.search_url = os.environ.get("ELSEVIER_SCOPUS_API_URL", _DEFAULT_SEARCH_URL).strip()
        self.api_key = os.environ.get("ELSEVIER_API_KEY", "").strip()
        self.institution_token = os.environ.get("ELSEVIER_INST_TOKEN", "").strip()
        self.last_error_status: int | None = None

    @property
    def is_configured(self) -> bool:
        """Whether authenticated Elsevier requests can be made."""

        return bool(self.api_key)

    def search_papers(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Return Scopus papers matching ``query``, or an empty list on failure."""

        query = query.strip()
        self.last_error_status = None
        if not query or not self.is_configured:
            return []

        scopus_query = self._scopus_query(query)
        if not scopus_query:
            return []

        limit = min(max_results if max_results is not None else self.max_results, 200)
        headers = {"Accept": "application/json", "X-ELS-APIKey": self.api_key}
        if self.institution_token:
            headers["X-ELS-Insttoken"] = self.institution_token

        try:
            response = requests.get(
                self.search_url,
                # The default STANDARD view omits dc:description, which would
                # drop every record at the abstract filter below.
                params={"query": scopus_query, "count": limit, "view": "COMPLETE"},
                headers=headers,
                timeout=_DEFAULT_TIMEOUT,
            )
            self.last_error_status = response.status_code
            response.raise_for_status()
            # Clear on success so _provider_status() does not treat HTTP 200
            # as a provider error (error_status must be None for "ok").
            self.last_error_status = None
            entries = response.json().get("search-results", {}).get("entry", [])
            papers = [self._format_paper(entry) for entry in entries if entry.get("dc:title")]
            usable_papers = [paper for paper in papers if paper.get("abstract")]
            from ..utils import logger

            logger.debug(
                "Elsevier Scopus returned %d usable paper(s) for query %r (sent as %r).",
                len(usable_papers),
                query,
                scopus_query,
            )
            return usable_papers
        except Exception as exc:
            from ..utils import logger, redact_secrets

            logger.error("Elsevier Scopus search failed for query %r: %s", query, redact_secrets(str(exc)))
            return []

    @staticmethod
    def _scopus_query(query: str, max_words: int = _MAX_QUERY_WORDS) -> str:
        """Reduce a natural-language query to terms Scopus can actually match."""

        words = [word.strip("-") for word in re.sub(r"[^\w\s-]", " ", query).split() if word.strip("-")]
        keywords = [
            word
            for word in words
            if word.casefold() not in _SCOPUS_OPERATORS and word.casefold() not in _SCOPUS_STOP_WORDS
        ]
        return " ".join((keywords or words)[:max_words])

    @classmethod
    def _format_paper(cls, entry: dict[str, Any]) -> dict[str, Any]:
        """Convert a Scopus search entry into the shared literature schema."""

        eid = str(entry.get("eid") or "").strip()
        doi = str(entry.get("prism:doi") or "").strip()
        raw_id = eid or doi
        source_id = (
            f"elsevier:{raw_id}"
            if raw_id
            else f"elsevier:{hashlib.sha256(str(entry.get('dc:title', '')).encode()).hexdigest()[:16]}"
        )
        authors = [author.strip() for author in str(entry.get("dc:creator") or "").split(",") if author.strip()]
        published = str(entry.get("prism:coverDate") or "").strip() or None
        publication_name = str(entry.get("prism:publicationName") or "").strip() or None
        abstract = str(entry.get("dc:description") or "").strip()
        entry_url = cls._entry_url(entry, doi, eid)
        pdf_url = find_pdf_url(entry.get("link"), entry.get("prism:url"), entry_url)
        return {
            "arxiv_id": source_id,
            "entry_id": entry_url,
            "title": str(entry.get("dc:title") or "Untitled").strip(),
            "abstract": abstract,
            "authors": authors,
            "primary_category": publication_name or "general",
            "categories": [publication_name] if publication_name else [],
            "published": published,
            "updated": published,
            "doi": doi or None,
            "pdf_url": pdf_url,
            "arxiv_url": entry_url,
            "comment": None,
            "journal_ref": publication_name,
            "source": "elsevier",
        }

    @staticmethod
    def _entry_url(entry: dict[str, Any], doi: str, eid: str) -> str:
        for link in entry.get("link") or []:
            if isinstance(link, dict) and link.get("@href"):
                return str(link["@href"])
        if doi:
            return f"https://doi.org/{doi}"
        if eid:
            return f"https://api.elsevier.com/content/abstract/eid/{eid}"
        return _DEFAULT_SEARCH_URL
