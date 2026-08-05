"""Tavily web-search integration for supplementary research evidence."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import requests

from ..utils import logger, redact_secrets

_SEARCH_URL = "https://api.tavily.com/search"
_DEFAULT_TIMEOUT = 15


class TavilySearchTool:
    """Search Tavily and normalize web results into the shared paper schema."""

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results
        self.api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        self.last_error_status: int | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search_papers(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Return web results with usable text, or an empty list on failure."""

        query = query.strip()
        if not query or not self.is_configured:
            return []

        self.last_error_status = None
        limit = max_results if max_results is not None else self.max_results
        try:
            response = requests.post(
                _SEARCH_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": limit,
                    "include_answer": False,
                    "include_raw_content": "text",
                },
                timeout=_DEFAULT_TIMEOUT,
            )
            self.last_error_status = response.status_code if response.status_code in (429, 503) else None
            response.raise_for_status()
            results = response.json().get("results", [])
            papers = [self._format_result(result) for result in results if result.get("content")]
            logger.info("Tavily returned %d usable result(s) for query %r.", len(papers), query)
            return papers
        except Exception as exc:
            self.last_error_status = self.last_error_status or getattr(
                getattr(exc, "response", None), "status_code", None
            )
            logger.error("Tavily search failed for query %r: %s", query, redact_secrets(str(exc)))
            return []

    @staticmethod
    def _format_result(result: dict[str, Any]) -> dict[str, Any]:
        url = str(result.get("url") or "").strip()
        source_id = f"tavily:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        content = str(result.get("raw_content") or result.get("content") or "").strip()
        return {
            "arxiv_id": source_id,
            "entry_id": url,
            "title": str(result.get("title") or "Untitled web result").strip(),
            "abstract": content,
            "authors": [],
            "published": result.get("published_date"),
            "updated": result.get("published_date"),
            "primary_category": "web",
            "categories": ["web"],
            "arxiv_url": url,
            "pdf_url": None,
            "source": "tavily",
        }
