"""Springer Nature API integration for literature search and retrieval."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import requests

from ..utils import logger, redact_secrets
from .pdf_urls import find_pdf_url

_DEFAULT_TIMEOUT = 15


class SpringerSearchTool:
    """Search Springer Nature Open Access & Meta APIs and normalize results."""

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results
        self.openaccess_url = os.environ.get(
            "SPRINGER_OPEN_ACCESS_API_URL",
            "https://api.springernature.com/openaccess/json",
        ).strip()
        self.meta_url = os.environ.get(
            "SPRINGER_META_API_URL",
            "https://api.springernature.com/meta/v2/json",
        ).strip()
        self.jats_url = os.environ.get(
            "SPRINGER_OPEN_ACCESS_JATS_URL",
            "https://api.springernature.com/openaccess/jats",
        ).strip()
        self.openaccess_key = (
            os.environ.get("SPRINGER_OPEN_ACCESS_API_KEY", "")
            or os.environ.get("SPRINGER_API_KEY", "")
        ).strip()
        self.meta_key = (
            os.environ.get("SPRINGER_META_API_KEY", "")
            or os.environ.get("SPRINGER_API_KEY", "")
        ).strip()
        self.api_key = self.openaccess_key or self.meta_key

    @property
    def is_configured(self) -> bool:
        """Whether authenticated Springer Nature requests can be made."""

        return bool(self.api_key)

    def search_papers(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return Springer papers matching ``query``, or an empty list on failure."""

        query = query.strip()
        if not query or not self.is_configured:
            return []

        limit = max_results if max_results is not None else self.max_results

        # Try Open Access endpoint first, fall back to Meta endpoint
        endpoints = [
            (self.openaccess_url, self.openaccess_key or self.meta_key),
            (self.meta_url, self.meta_key or self.openaccess_key),
        ]
        for url, key in endpoints:
            if not url or not key:
                continue
            try:
                response = requests.get(
                    url,
                    params={"q": query, "api_key": key, "p": limit},
                    timeout=_DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                records = data.get("records", [])
                papers = [self._format_paper(record) for record in records if record.get("title")]
                usable_papers = [paper for paper in papers if paper.get("abstract")]
                if usable_papers:
                    logger.info(
                        "Springer Nature returned %d usable paper(s) for query %r.",
                        len(usable_papers),
                        query,
                    )
                    return usable_papers
            except Exception as exc:
                logger.error(
                    "Springer Nature search failed for endpoint %r query %r: %s",
                    url,
                    query,
                    redact_secrets(str(exc)),
                )

        return []

    @classmethod
    def _format_paper(cls, record: dict[str, Any]) -> dict[str, Any]:
        """Convert a Springer API record into the shared literature schema."""

        doi = str(record.get("doi") or "").strip()
        identifier = str(record.get("identifier") or "").strip()
        raw_id = doi or identifier
        if raw_id:
            source_id = f"springer:{raw_id}"
        else:
            title_hash = hashlib.sha256(str(record.get("title", "")).encode()).hexdigest()[:16]
            source_id = f"springer:{title_hash}"

        abstract = cls._extract_abstract(record)
        creators = record.get("creators") or []
        authors = [
            c.get("creator", "").strip()
            for c in creators
            if isinstance(c, dict) and c.get("creator")
        ]

        publication_date = record.get("publicationDate")
        published = str(publication_date).strip() if publication_date else None
        publication_name = str(record.get("publicationName") or "").strip() or None

        urls = record.get("url") or []
        web_url = None
        pdf_url = find_pdf_url(urls)
        if isinstance(urls, list):
            for u in urls:
                if isinstance(u, dict):
                    val = u.get("value")
                    fmt = str(u.get("format", "")).casefold()
                    if fmt == "pdf" and val:
                        pdf_url = val
                    elif fmt == "html" or not web_url:
                        web_url = val

        entry_url = web_url or (f"https://doi.org/{doi}" if doi else f"https://api.springernature.com/meta/v2/json?q=doi:{doi}")

        return {
            "arxiv_id": source_id,
            "entry_id": entry_url,
            "title": str(record.get("title") or "Untitled").strip(),
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
            "source": "springer",
        }

    @staticmethod
    def _extract_abstract(record: dict[str, Any]) -> str:
        """Extract clean plaintext abstract from varying Springer record shapes."""

        abstract_obj = record.get("abstract")
        if not abstract_obj:
            return ""

        if isinstance(abstract_obj, str):
            return abstract_obj.strip()

        if isinstance(abstract_obj, dict):
            p_val = abstract_obj.get("p")
            if isinstance(p_val, str):
                return p_val.strip()
            if isinstance(p_val, list):
                return " ".join(str(item) for item in p_val if item).strip()
            return str(abstract_obj).strip()

        if isinstance(abstract_obj, list):
            parts = []
            for item in abstract_obj:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    p = item.get("p")
                    if p:
                        parts.append(str(p))
            return " ".join(parts).strip()

        return str(abstract_obj).strip()
