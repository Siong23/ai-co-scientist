import logging
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import feedparser

from app.utils import redact_secrets

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 15
_ARXIV_API_ENDPOINT = "https://export.arxiv.org/api/query"
# arXiv's API terms allow one request every three seconds on a single
# connection, counted for the whole client rather than per query or thread.
_MIN_REQUEST_INTERVAL_SECONDS = 3.0
_request_slot_lock = threading.Lock()
_next_request_at = 0.0
_ARXIV_USER_AGENT = "open-ai-co-scientist/1.0 (+https://github.com/Siong23/ai-co-scientist)"
_SORT_PARAMETERS = ("relevance", "lastUpdatedDate", "submittedDate")
_ARXIV_FIELD_CLAUSE = re.compile(
    r"(?:^|\s|\()(?:all|ti|abs|au|co|jr|cat|rn|id|submittedDate):",
    re.IGNORECASE,
)
_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "develop",
    "during",
    "dynamically",
    "for",
    "framework",
    "from",
    "how",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "what",
    "with",
}


# arXiv ANDs every concept, and its corpus is small enough that a sixth term
# empties the result set: the same goal returns five papers at four concepts and
# nothing at six.
def build_arxiv_query(query: str, *, max_concepts: int = 4) -> str:
    """Convert a natural-language need into a bounded field-aware arXiv query.

    Existing arXiv field syntax is preserved. Natural-language searches retain
    compound concepts and connect a small number of meaningful terms with AND,
    avoiding the API's extremely broad default free-text behavior.
    """

    normalized = re.sub(r"\s+", " ", query).strip()
    if not normalized or _ARXIV_FIELD_CLAUSE.search(normalized):
        return normalized

    phrases: list[str] = []
    consumed_parts: set[str] = set()
    for compound in re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", normalized):
        phrase = compound.replace("-", " ").casefold()
        clause = f'all:"{phrase}"'
        if clause not in phrases:
            phrases.append(clause)
        consumed_parts.update(phrase.split())

    # Requiring every compound at once matches nothing: one paper rarely carries
    # three exact phrases, and a goal naming three of them returned zero results.
    # One OR group keeps their recall, and it counts as a single concept so the
    # plain terms below still bound the query.
    concepts: list[str] = []
    if phrases:
        concepts.append(phrases[0] if len(phrases) == 1 else "(" + " OR ".join(phrases) + ")")

    tokens = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9+._]*\b", normalized)
    prioritized = [token for token in tokens if any(char.isdigit() for char in token) or token.isupper()]
    prioritized.extend(tokens)
    for token in prioritized:
        folded = token.casefold().strip("._")
        if (
            not folded
            or folded in consumed_parts
            or folded in _QUERY_STOP_WORDS
            or (len(folded) < 3 and not any(char.isdigit() for char in folded))
        ):
            continue
        clause = f"all:{token}"
        if clause.casefold() not in {item.casefold() for item in concepts}:
            concepts.append(clause)
        if len(concepts) >= max(1, max_concepts):
            break

    return " AND ".join(concepts) if concepts else f'all:"{normalized.replace(chr(34), "")}"'


def _wait_for_request_slot() -> None:
    """Block until arXiv's per-client request interval has elapsed.

    Every arXiv search and lookup in the process shares one slot, so parallel
    agents or retrieval rounds cannot burst past the documented rate.
    """

    global _next_request_at
    with _request_slot_lock:
        delay = _next_request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _next_request_at = time.monotonic() + _MIN_REQUEST_INTERVAL_SECONDS


# arXiv's Varnish / Fastly frontend answers HTTP 406 if Python's TLS handshake
# includes the ALPN extension for http/1.1 (which http.client._create_https_context
# and urllib3 enable by default). Supplying an explicit ssl.create_default_context()
# omits ALPN, allowing arXiv's frontend to forward uncached queries normally.
def _fetch_feed(params: Dict[str, Any], timeout: int = _REQUEST_TIMEOUT_SECONDS) -> Any:
    """Fetch one arXiv Atom page and return the parsed feed."""

    _wait_for_request_slot()
    url = f"{_ARXIV_API_ENDPOINT}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": _ARXIV_USER_AGENT})
    ssl_context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
        payload = response.read()
    return feedparser.parse(payload)


def _feed_error(feed: Any) -> str:
    """Return arXiv's message when the feed carries an error instead of results."""

    entries = getattr(feed, "entries", []) or []
    if len(entries) == 1 and "api/errors" in str(entries[0].get("id", "")):
        return str(entries[0].get("summary", "")).strip()
    return ""


class ArxivSearchTool:
    """Tool for searching and retrieving papers from arXiv"""

    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.last_error_status: int | None = None
        self.last_error_kind = ""
        self.last_error_detail = ""

    def search_papers(
        self,
        query: str,
        max_results: Optional[int] = None,
        categories: Optional[List[str]] = None,
        sort_by: str = "relevance",
    ) -> List[Dict]:
        """
        Search arXiv for papers matching query

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            categories: List of arXiv categories to filter by (e.g., ['cs.AI', 'cs.LG'])
            sort_by: Sort criteria ('relevance', 'lastUpdatedDate', 'submittedDate')

        Returns:
            List of paper dictionaries with metadata
        """
        if max_results is None:
            max_results = self.max_results
        self.last_error_status = None
        self.last_error_kind = ""
        self.last_error_detail = ""

        # Build search query with category filter if provided
        search_query = build_arxiv_query(query)
        if categories:
            category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({search_query}) AND ({category_filter})"

        # Set sort criteria
        sort_parameter = sort_by if sort_by in _SORT_PARAMETERS else "relevance"

        # Log search parameters
        logger.debug(
            f"ArXiv search initiated - Query: '{query}', Max Results: {max_results}, "
            f"Categories: {categories}, Sort: {sort_by}"
        )
        if search_query != query:
            logger.debug(f"Expanded search query: '{search_query}'")

        try:
            import time

            start_time = time.time()

            feed = _fetch_feed(
                {
                    "search_query": search_query,
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": sort_parameter,
                    "sortOrder": "descending",
                }
            )
            error_text = _feed_error(feed)
            if error_text:
                raise ValueError(f"arXiv API returned an error: {error_text}")

            papers = [self._format_paper(entry) for entry in feed.entries]

            search_time = (time.time() - start_time) * 1000  # Convert to ms

            # Enhanced logging with performance metrics
            logger.debug(
                f"ArXiv search completed - Found {len(papers)} papers for query: '{query}' in {search_time:.2f}ms"
            )

            # Log paper details at debug level
            if papers and logger.isEnabledFor(logging.DEBUG):
                logger.debug("ArXiv papers found:")
                for i, paper in enumerate(papers[:3], 1):  # Log first 3 papers
                    logger.debug(f"  {i}. {paper['title']} ({paper['arxiv_id']}) - Published: {paper['published']}")
                if len(papers) > 3:
                    logger.debug(f"  ... and {len(papers) - 3} more papers")

            # Log categories distribution
            if papers:
                categories_count = {}
                for paper in papers:
                    for cat in paper.get("categories", []):
                        categories_count[cat] = categories_count.get(cat, 0) + 1
                top_categories = sorted(categories_count.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.debug(f"ArXiv search result categories: {dict(top_categories)}")

            return papers

        except Exception as e:
            # urllib raises HTTPError (carrying .code) for a rejected request and
            # surfaces read timeouts either directly or wrapped in URLError.reason.
            self.last_error_status = getattr(e, "code", None)
            reason = getattr(e, "reason", None)
            timed_out = isinstance(e, (socket.timeout, TimeoutError)) or isinstance(
                reason, (socket.timeout, TimeoutError)
            )
            self.last_error_kind = "timeout" if timed_out else "provider_error"
            self.last_error_detail = redact_secrets(str(e))
            logger.warning("ArXiv search failed for query %r: %s", redact_secrets(query), self.last_error_detail)
            return []

    def search_by_author(self, author_name: str, max_results: Optional[int] = None) -> List[Dict]:
        """Search for papers by a specific author"""
        query = f"au:{author_name}"
        return self.search_papers(query, max_results)

    def search_recent_papers(self, query: str, days_back: int = 7, max_results: Optional[int] = None) -> List[Dict]:
        """Search for recent papers within specified time frame"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Format dates for arXiv search
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        # Add date filter to query
        date_query = f"({query}) AND submittedDate:[{start_str} TO {end_str}]"
        return self.search_papers(date_query, max_results, sort_by="submittedDate")

    def search_by_category(
        self, category: str, max_results: Optional[int] = None, days_back: Optional[int] = None
    ) -> List[Dict]:
        """Search papers in a specific arXiv category"""
        query = f"cat:{category}"

        if days_back:
            return self.search_recent_papers(query, days_back, max_results)
        else:
            return self.search_papers(query, max_results)

    def get_paper_details(self, arxiv_id: str) -> Optional[Dict]:
        """Get detailed information for a specific paper by arXiv ID"""
        logger.debug(f"Fetching arXiv paper details for ID: {arxiv_id}")
        try:
            import time

            start_time = time.time()

            feed = _fetch_feed({"id_list": arxiv_id, "start": 0, "max_results": 1})
            error_text = _feed_error(feed)
            if error_text:
                raise ValueError(f"arXiv API returned an error: {error_text}")
            papers = list(feed.entries)

            fetch_time = (time.time() - start_time) * 1000

            if papers:
                paper = self._format_paper(papers[0])
                logger.debug(f"Successfully retrieved paper '{paper['title']}' ({arxiv_id}) in {fetch_time:.2f}ms")
                return paper
            else:
                logger.warning(f"No paper found with arXiv ID: {arxiv_id}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving paper {arxiv_id}: {e}", exc_info=True)
            return None

    def _format_paper(self, paper: Any) -> Dict:
        """Format one arXiv Atom entry into a standardized dictionary"""
        # The entry id is the abs URL; its tail is the versioned short ID.
        entry_id = str(paper.get("id", ""))
        arxiv_id = entry_id.rsplit("/abs/", 1)[-1] if "/abs/" in entry_id else entry_id

        # Clean and format abstract
        abstract = self._clean_text(paper.get("summary", ""))

        # Format authors
        authors = [str(author.get("name", "")) for author in paper.get("authors", []) if author.get("name")]

        # Extract DOI if available
        doi = paper.get("arxiv_doi") or None

        # Format categories
        categories = [tag.get("term", "") for tag in paper.get("tags", []) if tag.get("term")]

        primary = paper.get("arxiv_primary_category") or {}
        pdf_url = next(
            (link.get("href") for link in paper.get("links", []) if link.get("type") == "application/pdf"),
            None,
        )

        return {
            "arxiv_id": arxiv_id,
            "entry_id": entry_id,
            "title": self._clean_text(paper.get("title", "")),
            "abstract": abstract,
            "authors": authors,
            "primary_category": primary.get("term"),
            "categories": categories,
            "published": paper.get("published"),
            "updated": paper.get("updated"),
            "doi": doi,
            "pdf_url": pdf_url,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "comment": paper.get("arxiv_comment"),
            "journal_ref": paper.get("arxiv_journal_ref"),
            "source": "arxiv",
        }

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and newlines"""
        if not text:
            return ""
        # Replace multiple whitespace with single space
        cleaned = re.sub(r"\s+", " ", text)
        return cleaned.strip()

    def analyze_research_trends(self, query: str, days_back: int = 30) -> Dict:
        """Analyze research trends for a given topic"""
        logger.debug(f"Starting arXiv trends analysis for '{query}' over last {days_back} days")

        papers = self.search_recent_papers(query, days_back, max_results=50)

        if not papers:
            logger.warning(f"No papers found for trends analysis of '{query}' in last {days_back} days")
            return {"total_papers": 0, "categories": {}, "top_authors": {}, "papers": []}

        # Analyze categories
        category_counts = {}
        author_counts = {}

        for paper in papers:
            # Count categories
            for category in paper.get("categories", []):
                category_counts[category] = category_counts.get(category, 0) + 1

            # Count authors
            for author in paper.get("authors", []):
                author_counts[author] = author_counts.get(author, 0) + 1

        # Sort by frequency
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Log trends analysis results
        logger.debug(
            f"ArXiv trends analysis completed for '{query}': {len(papers)} papers, "
            f"top categories: {dict(top_categories[:3])}"
        )
        if top_authors:
            logger.debug(f"Most active authors: {dict(top_authors[:3])}")

        return {
            "total_papers": len(papers),
            "date_range": f"Last {days_back} days",
            "top_categories": top_categories,
            "top_authors": top_authors,
            "papers": papers,
        }


# Common arXiv categories for different fields
ARXIV_CATEGORIES = {
    "computer_science": [
        "cs.AI",  # Artificial Intelligence
        "cs.LG",  # Machine Learning
        "cs.CL",  # Computation and Language
        "cs.CV",  # Computer Vision
        "cs.RO",  # Robotics
        "cs.NE",  # Neural and Evolutionary Computing
    ],
    "physics": [
        "physics.data-an",  # Data Analysis
        "physics.comp-ph",  # Computational Physics
        "cond-mat.stat-mech",  # Statistical Mechanics
    ],
    "mathematics": [
        "math.ST",  # Statistics Theory
        "math.OC",  # Optimization and Control
        "math.PR",  # Probability
    ],
    "quantitative_biology": [
        "q-bio.QM",  # Quantitative Methods
        "q-bio.GN",  # Genomics
        "q-bio.BM",  # Biomolecules
    ],
}


def get_categories_for_field(field: str) -> List[str]:
    """Get relevant arXiv categories for a research field"""
    return ARXIV_CATEGORIES.get(field.lower(), [])
