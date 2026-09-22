"""Tavily web-search integration for supplementary research evidence."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
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

# Tavily bills per request, so one cycle's spend is tracked process-wide: the
# retriever is rebuilt several times per cycle (generation, reflection), and a
# per-instance counter would never see the whole run.
_USAGE_LOCK = threading.Lock()
_CYCLE_USAGE: dict[str, int] = {
    "search_calls": 0,
    "search_cache_hits": 0,
    "extract_calls": 0,
    "extract_urls": 0,
    "extract_cache_hits": 0,
    "budget_skips": 0,
}


def reset_cycle_usage() -> None:
    """Start a new accounting window for the cycle that is about to run."""

    with _USAGE_LOCK:
        for key in _CYCLE_USAGE:
            _CYCLE_USAGE[key] = 0


def cycle_usage() -> dict[str, int]:
    """Return the billed calls, cache hits, and skips of the current cycle."""

    with _USAGE_LOCK:
        return dict(_CYCLE_USAGE)


def _record_usage(key: str, amount: int = 1) -> int:
    with _USAGE_LOCK:
        _CYCLE_USAGE[key] = _CYCLE_USAGE.get(key, 0) + amount
        return _CYCLE_USAGE[key]


def _usage_count(key: str) -> int:
    with _USAGE_LOCK:
        return _CYCLE_USAGE.get(key, 0)


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
        cache_directory: str | Path | None = None,
        cache_ttl_seconds: float = 0.0,
        max_searches_per_cycle: int = 0,
        max_extracts_per_cycle: int = 0,
    ) -> None:
        self.max_results = max_results
        self.search_depth = search_depth
        self.search_chunks_per_source = max(1, min(3, search_chunks_per_source))
        self.extract_depth = extract_depth
        self.extract_chunks_per_source = max(1, min(5, extract_chunks_per_source))
        self.api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        # Caching and the per-cycle budget stay off unless a caller configures
        # them, so tests and one-off callers keep the plain request behaviour.
        self.cache_directory = Path(cache_directory) if cache_directory else None
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self.max_searches_per_cycle = max(0, int(max_searches_per_cycle))
        self.max_extracts_per_cycle = max(0, int(max_extracts_per_cycle))
        self.last_error_status: int | None = None
        self.last_error_kind: str | None = None
        self.last_error_detail: str = ""

    # ----------------------------------------------------------------
    # Response cache
    # ----------------------------------------------------------------

    def _cache_file(self, kind: str, payload: dict[str, Any]) -> Path | None:
        """Address one request by its payload so repeats never reach Tavily."""

        if self.cache_directory is None or self.cache_ttl_seconds <= 0:
            return None
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()
        return self.cache_directory / kind / f"{digest}.json"

    def _read_cache(self, cache_file: Path | None) -> list[dict[str, Any]] | None:
        if cache_file is None or not cache_file.exists():
            return None
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("Tavily cache read failed: %s", redact_secrets(str(exc)))
            return None
        if time.time() - float(cached.get("stored_at", 0.0)) > self.cache_ttl_seconds:
            return None
        results = cached.get("results")
        return results if isinstance(results, list) else None

    def _write_cache(self, cache_file: Path | None, results: list[dict[str, Any]]) -> None:
        if cache_file is None:
            return
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"stored_at": time.time(), "results": results}),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Tavily cache write failed: %s", redact_secrets(str(exc)))

    def _budget_exhausted(self, usage_key: str, limit: int, description: str) -> bool:
        """Report whether this cycle already spent its allowance of one call."""

        if limit <= 0:
            return False
        if _usage_count(usage_key) < limit:
            return False
        _record_usage("budget_skips")
        logger.warning(
            "Tavily %s skipped: this cycle already used its budget of %d call(s).",
            description,
            limit,
        )
        return True

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
        self.last_error_kind = None
        self.last_error_detail = ""
        limit = max_results if max_results is not None else self.max_results
        domains = list(
            dict.fromkeys(str(domain).strip().casefold() for domain in include_domains if str(domain).strip())
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

        cache_file = self._cache_file("search", payload)
        cached_results = self._read_cache(cache_file)
        if cached_results is not None:
            _record_usage("search_cache_hits")
            logger.debug("Tavily search served from cache for query %r.", query)
            return [
                self._format_result(result)
                for result in cached_results
                if result.get("url") and (result.get("content") or result.get("raw_content"))
            ]

        if self._budget_exhausted("search_calls", self.max_searches_per_cycle, "search"):
            self.last_error_kind = "cycle_budget_exhausted"
            self.last_error_detail = "The cycle's Tavily search budget is spent."
            return []

        try:
            _record_usage("search_calls")
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
            self._write_cache(cache_file, results)
            logger.debug("Tavily returned %d usable result(s) for query %r.", len(evidence), query)
            return evidence
        except Exception as exc:
            self.last_error_status = self.last_error_status or getattr(
                getattr(exc, "response", None), "status_code", None
            )
            self.last_error_kind = (
                "timeout"
                if isinstance(exc, requests.Timeout)
                else "quota_or_plan_rejection"
                if self.last_error_status in (401, 402, 403, 432, 433)
                else "rate_limited"
                if self.last_error_status in (429, 503)
                else "provider_error"
            )
            self.last_error_detail = redact_secrets(str(exc))
            logger.error("Tavily search failed for query %r: %s", query, redact_secrets(str(exc)))
            return []

    def extract(
        self,
        urls: Sequence[str],
        query: str,
        chunks_per_source: int | None = None,
    ) -> dict[str, str]:
        """Extract bounded query-relevant chunks from already selected URLs."""

        selected_urls = list(dict.fromkeys(str(url).strip() for url in urls if str(url).strip()))
        query = query.strip()
        if not selected_urls or not query or not self.is_configured:
            return {}

        chunk_limit = self.extract_chunks_per_source if chunks_per_source is None else max(1, min(5, chunks_per_source))
        self.last_error_status = None
        self.last_error_kind = None
        self.last_error_detail = ""
        payload = {
            "urls": selected_urls,
            "query": query,
            "chunks_per_source": chunk_limit,
            "extract_depth": self.extract_depth,
            "format": "text",
        }

        cache_file = self._cache_file("extract", payload)
        cached_results = self._read_cache(cache_file)
        if cached_results is not None:
            _record_usage("extract_cache_hits")
            logger.debug("Tavily extract served from cache for %d URL(s).", len(selected_urls))
            return self._extracted_by_url(cached_results)

        if self._budget_exhausted("extract_calls", self.max_extracts_per_cycle, "extract"):
            self.last_error_kind = "cycle_budget_exhausted"
            self.last_error_detail = "The cycle's Tavily extract budget is spent."
            return {}

        try:
            _record_usage("extract_calls")
            _record_usage("extract_urls", len(selected_urls))
            response = requests.post(
                _EXTRACT_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=_DEFAULT_EXTRACT_TIMEOUT,
            )
            self.last_error_status = response.status_code if response.status_code in (429, 503) else None
            response.raise_for_status()
            results = response.json().get("results", [])
            extracted = self._extracted_by_url(results)
            self._write_cache(cache_file, results)
            logger.debug(
                "Tavily extracted bounded content from %d/%d selected URL(s).",
                len(extracted),
                len(selected_urls),
            )
            return extracted
        except Exception as exc:
            self.last_error_status = self.last_error_status or getattr(
                getattr(exc, "response", None), "status_code", None
            )
            self.last_error_kind = (
                "timeout"
                if isinstance(exc, requests.Timeout)
                else "quota_or_plan_rejection"
                if self.last_error_status in (401, 402, 403, 432, 433)
                else "rate_limited"
                if self.last_error_status in (429, 503)
                else "provider_error"
            )
            self.last_error_detail = redact_secrets(str(exc))
            logger.error(
                "Tavily extract failed for %d URL(s): %s",
                len(selected_urls),
                redact_secrets(str(exc)),
            )
            return {}

    @staticmethod
    def _extracted_by_url(results: Sequence[dict[str, Any]]) -> dict[str, str]:
        """Key extracted page bodies by canonical URL, dropping empty results."""

        return {
            canonicalize_url(str(result.get("url") or "")): str(result.get("raw_content") or "").strip()
            for result in results
            if result.get("url") and str(result.get("raw_content") or "").strip()
        }

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
