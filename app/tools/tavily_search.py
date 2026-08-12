"""Tavily web-search integration for supplementary research evidence."""

from __future__ import annotations

import hashlib
import os
from typing import Any, Sequence
from urllib.parse import urlsplit

import requests

from ..evidence import canonicalize_url
from ..utils import logger, redact_secrets
from .pdf_urls import find_pdf_url

_SEARCH_URL = "https://api.tavily.com/search"
_EXTRACT_URL = "https://api.tavily.com/extract"
_DEFAULT_SEARCH_TIMEOUT = 30
_DEFAULT_EXTRACT_TIMEOUT = 30


class TavilySearchTool:
    """Search Tavily and return provider-neutral web evidence records."""

    def __init__(
        self,
        max_results: int = 10,
        *,
        search_depth: str = "advanced",
        search_chunks_per_source: int = 3,
        extract_depth: str = "basic",
        extract_chunks_per_source: int = 3,
    ) -> None:
        self.max_results = max_results
        self.search_depth = search_depth
        self.search_chunks_per_source = max(1, min(3, search_chunks_per_source))
        self.extract_depth = extract_depth
        self.extract_chunks_per_source = max(1, min(5, extract_chunks_per_source))
        self.api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        self.last_error_status: int | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(
        self,
        query: str,
        max_results: int | None = None,
        *,
        include_domains: Sequence[str] = (),
        time_range: str | None = None,
        topic: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return web results with usable text, or an empty list on failure."""

        query = query.strip()
        if not query or not self.is_configured:
            return []

        self.last_error_status = None
        limit = max_results if max_results is not None else self.max_results
        domains = list(
            dict.fromkeys(
                str(domain).strip().casefold()
                for domain in include_domains
                if str(domain).strip()
            )
        )
        freshness = str(time_range or "").strip().casefold()
        if freshness not in {"day", "week", "month", "year"}:
            freshness = ""
        search_topic = str(topic or "").strip().casefold()
        if search_topic not in {"news"}:
            search_topic = ""
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": self.search_depth,
            "chunks_per_source": self.search_chunks_per_source,
            "max_results": limit,
            "include_answer": False,
            "include_raw_content": False,
        }
        if domains:
            payload["include_domains"] = domains
        if freshness:
            payload["time_range"] = freshness
        if search_topic:
            payload["topic"] = search_topic
        try:
            response = requests.post(
                _SEARCH_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=_DEFAULT_SEARCH_TIMEOUT,
            )
            self.last_error_status = response.status_code if response.status_code in (429, 503) else None
            response.raise_for_status()
            results = response.json().get("results", [])
            evidence = [
                self._format_result(result)
                for result in results
                if result.get("url") and (result.get("content") or result.get("raw_content"))
            ]
            logger.info("Tavily returned %d usable result(s) for query %r.", len(evidence), query)
            return evidence
        except Exception as exc:
            self.last_error_status = self.last_error_status or getattr(
                getattr(exc, "response", None), "status_code", None
            )
            logger.error("Tavily search failed for query %r: %s", query, redact_secrets(str(exc)))
            return []

    def extract(
        self,
        urls: Sequence[str],
        query: str,
        chunks_per_source: int | None = None,
    ) -> dict[str, str]:
        """Extract bounded query-relevant chunks from already selected URLs."""

        selected_urls = list(
            dict.fromkeys(str(url).strip() for url in urls if str(url).strip())
        )
        query = query.strip()
        if not selected_urls or not query or not self.is_configured:
            return {}

        chunk_limit = (
            self.extract_chunks_per_source
            if chunks_per_source is None
            else max(1, min(5, chunks_per_source))
        )
        self.last_error_status = None
        try:
            response = requests.post(
                _EXTRACT_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "urls": selected_urls,
                    "query": query,
                    "chunks_per_source": chunk_limit,
                    "extract_depth": self.extract_depth,
                    "format": "text",
                },
                timeout=_DEFAULT_EXTRACT_TIMEOUT,
            )
            self.last_error_status = (
                response.status_code if response.status_code in (429, 503) else None
            )
            response.raise_for_status()
            extracted = {
                canonicalize_url(str(result.get("url") or "")): str(
                    result.get("raw_content") or ""
                ).strip()
                for result in response.json().get("results", [])
                if result.get("url") and str(result.get("raw_content") or "").strip()
            }
            logger.info(
                "Tavily extracted bounded content from %d/%d selected URL(s).",
                len(extracted),
                len(selected_urls),
            )
            return extracted
        except Exception as exc:
            self.last_error_status = self.last_error_status or getattr(
                getattr(exc, "response", None), "status_code", None
            )
            logger.error(
                "Tavily extract failed for %d URL(s): %s",
                len(selected_urls),
                redact_secrets(str(exc)),
            )
            return {}

    def search_papers(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Compatibility alias for callers using the historical method name."""

        return self.search(query, max_results=max_results)

    @staticmethod
    def _format_result(result: dict[str, Any]) -> dict[str, Any]:
        url = str(result.get("url") or "").strip()
        canonical_url = canonicalize_url(url)
        source_id = f"web:{hashlib.sha256(canonical_url.encode()).hexdigest()[:16]}"
        snippet = str(result.get("content") or "").strip()
        content = str(result.get("raw_content") or "").strip()
        pdf_url = find_pdf_url(url)
        return {
            "source_id": source_id,
            "source_type": "web",
            "provider": "tavily",
            "title": str(result.get("title") or "Untitled web result").strip(),
            "url": url,
            "canonical_url": canonical_url,
            "domain": (urlsplit(canonical_url).hostname or "").casefold(),
            "page_type": "pdf" if pdf_url else "web_page",
            "snippet": snippet,
            "content": content,
            "content_extracted": bool(content),
            "published_at": result.get("published_date"),
            "updated_at": result.get("published_date"),
            "author": result.get("author"),
            "language": result.get("language"),
            "source_authority": result.get("source_authority"),
            "search_score": result.get("score"),
            "pdf_url": pdf_url,
        }
