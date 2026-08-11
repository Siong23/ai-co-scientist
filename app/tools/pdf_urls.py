"""Recognize direct scholarly PDF URLs returned by search providers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse, urlunparse

_ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
_ARXIV_ID_PATTERN = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)


def find_pdf_url(*values: Any) -> str | None:
    """Return the first explicit or safely derivable PDF URL in provider data."""

    for value in values:
        for candidate in _iter_strings(value):
            pdf_url = normalize_pdf_url(candidate)
            if pdf_url:
                return pdf_url
    return None


def normalize_pdf_url(value: str) -> str | None:
    """Normalize direct PDF URLs and convert arXiv abstract URLs to PDFs."""

    candidate = value.strip()
    if not candidate:
        return None

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None

    host = parsed.hostname.casefold()
    path = parsed.path
    path_lower = path.casefold()
    if host in _ARXIV_HOSTS and path_lower.startswith("/abs/"):
        paper_id = path[len("/abs/") :].strip("/")
        if _ARXIV_ID_PATTERN.fullmatch(unquote(paper_id)):
            return urlunparse((parsed.scheme, parsed.netloc, f"/pdf/{paper_id}", "", parsed.query, ""))

    if _looks_like_pdf_endpoint(path_lower, parsed.query):
        return candidate
    return None


def _looks_like_pdf_endpoint(path: str, query: str) -> bool:
    if path.endswith(".pdf") or path.endswith("/pdf") or "/pdf/" in path or "/content/pdf/" in path:
        return True

    for key, value in parse_qsl(query, keep_blank_values=True):
        normalized_key = key.casefold()
        normalized_value = value.casefold()
        if normalized_value in {"pdf", "application/pdf"} and normalized_key in {
            "download",
            "format",
            "type",
            "content-type",
            "response-content-type",
        }:
            return True
    return False


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_strings(nested)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for nested in value:
            yield from _iter_strings(nested)
