"""Elsevier Scopus API integration for literature search and retrieval."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import requests

from ..utils import logger, redact_secrets

_DEFAULT_TIMEOUT = 15
_DEFAULT_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"


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

        limit = min(max_results if max_results is not None else self.max_results, 200)
        headers = {"Accept": "application/json", "X-ELS-APIKey": self.api_key}
        if self.institution_token:
            headers["X-ELS-Insttoken"] = self.institution_token

        try:
            response = requests.get(
                self.search_url,
                params={"query": query, "count": limit},
                headers=headers,
                timeout=_DEFAULT_TIMEOUT,
            )
            self.last_error_status = response.status_code
            response.raise_for_status()
            entries = response.json().get("search-results", {}).get("entry", [])
            papers = [self._format_paper(entry) for entry in entries if entry.get("dc:title")]
            usable_papers = [paper for paper in papers if paper.get("abstract")]
            logger.info("Elsevier Scopus returned %d usable paper(s) for query %r.", len(usable_papers), query)
            return usable_papers
        except Exception as exc:
            logger.error("Elsevier Scopus search failed for query %r: %s", query, redact_secrets(str(exc)))
            return []

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
            "pdf_url": None,
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
