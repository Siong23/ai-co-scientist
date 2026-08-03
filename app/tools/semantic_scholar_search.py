"""Semantic Scholar Academic Graph API search tool.

Used as a fallback when arXiv retrieval yields insufficient evidence for a
research goal.  The API is free without a key (100 req / 5 min public limit);
setting ``SEMANTIC_SCHOLAR_API_KEY`` in the environment enables the
higher-quota authenticated tier (~1 req / sec).

The returned paper dicts share the same schema as ``ArxivSearchTool`` so they
flow into the existing RRF + semantic-reranking pipeline unchanged.
"""

import logging
import os
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = (
    "title,abstract,year,authors,externalIds,"
    "fieldsOfStudy,openAccessPdf,publicationDate"
)
_DEFAULT_TIMEOUT = 15  # seconds


class SemanticScholarSearchTool:
    """Search the Semantic Scholar Academic Graph API.

    Results are formatted to match the arXiv paper dict schema so they can be
    fused by the existing ``reciprocal_rank_fusion`` pipeline without changes.
    """

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self._headers = {"x-api-key": api_key} if api_key else {}
        if api_key:
            logger.info("Semantic Scholar: using authenticated API key.")
        else:
            logger.debug(
                "Semantic Scholar: no API key set; using public rate limits "
                "(100 req / 5 min). Set SEMANTIC_SCHOLAR_API_KEY for higher quota."
            )

    def search_papers(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> List[Dict]:
        """Search Semantic Scholar for papers matching *query*.

        Args:
            query: Natural-language or keyword search string.
            max_results: Cap on results returned (defaults to ``self.max_results``).

        Returns:
            List of paper dicts in the shared arXiv-compatible schema.
            Returns ``[]`` on any error or rate-limit response so callers can
            always treat the result as a plain list.
        """
        limit = max_results if max_results is not None else self.max_results
        logger.info(
            "Semantic Scholar search – query: %r  max_results: %d", query, limit
        )
        
        backoff_sec = 2.0
        for attempt in range(3):
            try:
                start = time.time()
                resp = requests.get(
                    _BASE_URL,
                    params={"query": query, "fields": _FIELDS, "limit": limit},
                    headers=self._headers,
                    timeout=_DEFAULT_TIMEOUT,
                )
                elapsed_ms = (time.time() - start) * 1000

                if resp.status_code == 429:
                    logger.warning(
                        "Semantic Scholar rate-limited (429) for query %r. "
                        "Attempt %d/3. Retrying in %.1fs...",
                        query,
                        attempt + 1,
                        backoff_sec,
                    )
                    time.sleep(backoff_sec)
                    backoff_sec *= 2.0
                    continue

                resp.raise_for_status()
                data = resp.json()
                raw_papers = data.get("data", [])

                # Filter out papers without an abstract — the RAG pipeline
                # requires abstract text for embedding and evidence grading.
                papers = [self._format_paper(p) for p in raw_papers if p.get("abstract")]

                logger.info(
                    "Semantic Scholar search completed – %d results "
                    "(of %d total, %d had abstracts) in %.0f ms",
                    len(papers),
                    len(raw_papers),
                    len(papers),
                    elapsed_ms,
                )
                return papers

            except Exception as exc:
                logger.error(
                    "Semantic Scholar search failed for query %r: %s",
                    query,
                    exc,
                    exc_info=True,
                )
                return []

        logger.error(
            "Semantic Scholar search failed after 3 attempts due to rate-limiting for query %r",
            query,
        )
        return []

    @staticmethod
    def _format_paper(paper: Dict) -> Dict:
        """Convert a Semantic Scholar API result to the shared paper schema.

        The ``arxiv_id`` field is used as the deduplication key throughout the
        pipeline.  When a paper has a real arXiv ID (via ``externalIds``), we
        use it so that papers returned by both sources are deduplicated
        correctly.  Papers only in Semantic Scholar get a ``s2:`` prefixed ID
        that never collides with a real arXiv ID.
        """
        external = paper.get("externalIds") or {}
        arxiv_id = (external.get("ArXiv") or "").strip()
        doi = external.get("DOI")
        s2_id = paper.get("paperId", "")

        # Stable dedup key: prefer real arXiv ID, fall back to s2: prefix.
        dedup_id = arxiv_id if arxiv_id else f"s2:{s2_id}"

        year = paper.get("year")
        pub_date = paper.get("publicationDate")
        published = pub_date or (f"{year}-01-01" if year else None)

        authors = [a.get("name", "") for a in (paper.get("authors") or [])]
        fields = paper.get("fieldsOfStudy") or []

        pdf_info = paper.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url")

        arxiv_url = (
            f"https://arxiv.org/abs/{arxiv_id}"
            if arxiv_id
            else f"https://www.semanticscholar.org/paper/{s2_id}"
        )

        return {
            "arxiv_id": dedup_id,
            "entry_id": arxiv_url,
            "title": (paper.get("title") or "Untitled").strip(),
            "abstract": (paper.get("abstract") or "").strip(),
            "authors": authors,
            "primary_category": fields[0] if fields else "general",
            "categories": fields,
            "published": published,
            "updated": published,
            "doi": doi,
            "pdf_url": pdf_url,
            "arxiv_url": arxiv_url,
            "comment": None,
            "journal_ref": None,
            "source": "semantic_scholar",
        }
