"""Tavily web search integration for supplementary RAG evidence."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import requests

from ..utils import logger, redact_secrets

_SEARCH_URL = "https://api.tavily.com/search"
_DEFAULT_TIMEOUT = 15


class TavilySearchTool:
    """Search the web with Tavily and normalize results for the RAG pipeline."""

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results
        self.api_key = os.environ.get("TAVILY_API_KEY", "").strip()

    @property
    def is_configured(self) -> bool:
        """Whether this tool can make authenticated Tavily requests."""

        return bool(self.api_key)

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Return web evidence matching ``query``, or an empty list on failure."""

        query = query.strip()
        if not query or not self.is_configured:
            return []

        limit = max_results if max_results is not None else self.max_results
        try:
            response = requests.post(
                _SEARCH_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "query": query,
                    "max_results": limit,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=_DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            results = [
                self._format_result(result)
                for result in response.json().get("results", [])
                if result.get("content") and result.get("url")
            ]
            logger.info("Tavily returned %d usable web result(s) for query %r.", len(results), query)
            return results
        except Exception as exc:
            logger.error("Tavily search failed for query %r: %s", query, redact_secrets(str(exc)))
            return []

    @staticmethod
    def _format_result(result: dict[str, Any]) -> dict[str, Any]:
        """Convert one Tavily result into the shared retrieval schema."""

        url = str(result["url"]).strip()
        source_id = f"tavily:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        published = result.get("published_date") or result.get("published")
        return {
            "arxiv_id": source_id,
            "entry_id": url,
            "title": str(result.get("title") or "Untitled").strip(),
            "abstract": str(result.get("content") or "").strip(),
            "authors": [],
            "primary_category": "web",
            "categories": ["web"],
            "published": published,
            "updated": published,
            "doi": None,
            "pdf_url": None,
            "arxiv_url": url,
            "comment": None,
            "journal_ref": None,
            "source": "tavily",
        }
