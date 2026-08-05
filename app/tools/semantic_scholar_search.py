"""Semantic Scholar Academic Graph search integration."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from ..utils import logger, redact_secrets

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,year,authors,externalIds,fieldsOfStudy,openAccessPdf,publicationDate"
_DEFAULT_TIMEOUT = 15
_MAX_ATTEMPTS = 3


class SemanticScholarSearchTool:
    """Search Semantic Scholar and normalize results for the RAG pipeline."""

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results
        self.last_error_status: int | None = None
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self._headers = {"x-api-key": api_key} if api_key else {}

    def search_papers(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return papers matching ``query``, or an empty list on failure."""

        query = query.strip()
        if not query:
            return []
        self.last_error_status = None

        limit = max_results if max_results is not None else self.max_results
        backoff_seconds = 2.0

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = requests.get(
                    _BASE_URL,
                    params={"query": query, "fields": _FIELDS, "limit": limit},
                    headers=self._headers,
                    timeout=_DEFAULT_TIMEOUT,
                )
                if response.status_code == 429:
                    self.last_error_status = response.status_code
                    if attempt < _MAX_ATTEMPTS:
                        logger.warning(
                            "Semantic Scholar rate-limited query %r; retrying after %.1f seconds (attempt %d/%d).",
                            query,
                            backoff_seconds,
                            attempt,
                            _MAX_ATTEMPTS,
                        )
                        time.sleep(backoff_seconds)
                        backoff_seconds *= 2
                        continue
                    logger.error(
                        "Semantic Scholar rate-limited query %r after %d attempts.",
                        query,
                        _MAX_ATTEMPTS,
                    )
                    return []

                response.raise_for_status()
                raw_papers = response.json().get("data", [])
                papers = [self._format_paper(paper) for paper in raw_papers if paper.get("abstract")]
                logger.info(
                    "Semantic Scholar returned %d usable paper(s) for query %r.",
                    len(papers),
                    query,
                )
                return papers
            except Exception as exc:
                self.last_error_status = getattr(getattr(exc, "response", None), "status_code", None)
                logger.error(
                    "Semantic Scholar search failed for query %r: %s",
                    query,
                    redact_secrets(str(exc)),
                )
                return []

        return []

    @staticmethod
    def _format_paper(paper: dict[str, Any]) -> dict[str, Any]:
        """Convert an API paper into the arXiv-compatible shared schema."""

        external_ids = paper.get("externalIds") or {}
        arxiv_id = str(external_ids.get("ArXiv") or "").strip()
        semantic_scholar_id = str(paper.get("paperId") or "").strip()
        source_id = arxiv_id or f"s2:{semantic_scholar_id}"

        publication_date = paper.get("publicationDate")
        year = paper.get("year")
        published = publication_date or (f"{year}-01-01" if year else None)
        fields = paper.get("fieldsOfStudy") or []
        open_access_pdf = paper.get("openAccessPdf") or {}
        source_url = (
            f"https://arxiv.org/abs/{arxiv_id}"
            if arxiv_id
            else f"https://www.semanticscholar.org/paper/{semantic_scholar_id}"
        )

        return {
            "arxiv_id": source_id,
            "entry_id": source_url,
            "title": str(paper.get("title") or "Untitled").strip(),
            "abstract": str(paper.get("abstract") or "").strip(),
            "authors": [author.get("name", "") for author in (paper.get("authors") or [])],
            "primary_category": fields[0] if fields else "general",
            "categories": fields,
            "published": published,
            "updated": published,
            "doi": external_ids.get("DOI"),
            "pdf_url": open_access_pdf.get("url"),
            "arxiv_url": source_url,
            "comment": None,
            "journal_ref": None,
            "source": "semantic_scholar",
        }
