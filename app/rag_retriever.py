from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .config import config
from .tools.arxiv_search import ArxivSearchTool
from .tools.elsevier_search import ElsevierSearchTool
from .tools.semantic_scholar_search import SemanticScholarSearchTool
from .tools.springer_search import SpringerSearchTool
from .tools.tavily_search import TavilySearchTool
from .utils import get_sentence_transformer_model, logger, redact_secrets


@dataclass(frozen=True)
class EvidenceAspect:
    """One indispensable evidence dimension extracted from a research goal."""

    aspect_id: str
    description: str


@dataclass(frozen=True)
class SearchQueryPlan:
    """Structured query-rewriting output used for literature retrieval."""

    queries: tuple[str, ...]
    required_terms: tuple[str, ...]
    explicit_requirements: tuple[EvidenceAspect, ...] = ()
    exploration_directions: tuple[str, ...] = ()


class SharedSentenceTransformerEmbeddings(Embeddings):
    """LangChain adapter around the project's shared embedding model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = get_sentence_transformer_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = get_sentence_transformer_model()
        vector = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


def _canonical_arxiv_id(arxiv_id: str) -> str:
    """Remove an arXiv version suffix so multiple versions deduplicate."""

    return re.sub(r"v\d+$", "", arxiv_id.strip(), flags=re.IGNORECASE)


def reciprocal_rank_fusion(
    ranked_results: Sequence[Sequence[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse result rankings and deduplicate papers by canonical ID."""

    scores: dict[str, float] = {}
    papers: dict[str, dict[str, Any]] = {}

    for results in ranked_results:
        for rank, paper in enumerate(results):
            raw_id = str(paper.get("arxiv_id", "")).strip()
            canonical_id = _canonical_arxiv_id(raw_id)
            if not canonical_id:
                continue

            papers.setdefault(canonical_id, paper)
            scores[canonical_id] = scores.get(canonical_id, 0.0) + (1.0 / (k + rank + 1))

    ranked_ids = sorted(
        scores,
        key=lambda arxiv_id: scores[arxiv_id],
        reverse=True,
    )
    return [
        {
            **papers[arxiv_id],
            "_rrf_score": scores[arxiv_id],
        }
        for arxiv_id in ranked_ids
    ]


class ArxivRAGRetriever:
    """Run multi-query arXiv retrieval and semantic reranking."""

    def __init__(
        self,
        query_count: int | None = None,
        results_per_query: int | None = None,
        top_k: int | None = None,
        minimum_relevant_sources: int | None = None,
        corrective_retrieval_rounds: int | None = None,
        generation_debate_rounds: int | None = None,
        rrf_k: int | None = None,
        max_abstract_chars: int | None = None,
    ) -> None:
        rag_config = config.get("rag", {})

        self.query_count = query_count or int(rag_config.get("query_count", 5))
        self.results_per_query = results_per_query or int(rag_config.get("results_per_query", 10))
        self.top_k = top_k or int(rag_config.get("top_k", 10))
        self.minimum_relevant_sources = minimum_relevant_sources or int(rag_config.get("minimum_relevant_sources", 3))
        self.corrective_retrieval_rounds = (
            corrective_retrieval_rounds
            if corrective_retrieval_rounds is not None
            else int(rag_config.get("corrective_retrieval_rounds", 2))
        )
        self.generation_debate_rounds = (
            generation_debate_rounds
            if generation_debate_rounds is not None
            else int(rag_config.get("generation_debate_rounds", 3))
        )
        self.top_k = max(self.top_k, self.minimum_relevant_sources)
        self.rrf_k = rrf_k or int(rag_config.get("rrf_k", 60))
        self.max_abstract_chars = max_abstract_chars or int(rag_config.get("max_abstract_chars", 4000))

        library_config = config.get("paper_library", {})
        require_indexed_sources = bool(library_config.get("enabled", True)) and bool(
            library_config.get("require_indexed_sources_for_generation", False)
        )
        self.minimum_downloadable_sources = (
            min(
                self.top_k,
                max(
                    0,
                    int(library_config.get("minimum_downloadable_sources", self.minimum_relevant_sources)),
                ),
            )
            if require_indexed_sources
            else 0
        )
        configured_pdf_hosts = library_config.get("allowed_pdf_hosts", ())
        self.downloadable_pdf_hosts = {
            str(host).strip().casefold() for host in configured_pdf_hosts if str(host).strip()
        }
        self.last_search_stats: list[dict[str, Any]] = []
        self._search_round = 0

        self.arxiv = ArxivSearchTool(max_results=self.results_per_query)
        semantic_scholar_config = config.get("semantic_scholar", {})
        semantic_scholar_results = int(semantic_scholar_config.get("results_per_query", self.results_per_query))
        self.semantic_scholar = (
            SemanticScholarSearchTool(max_results=semantic_scholar_results)
            if semantic_scholar_config.get("enabled", True)
            else None
        )
        springer_config = config.get("springer", {})
        springer_results = int(springer_config.get("results_per_query", self.results_per_query))
        self.springer = (
            SpringerSearchTool(max_results=springer_results) if springer_config.get("enabled", True) else None
        )
        elsevier_config = config.get("elsevier", {})
        elsevier_results = int(elsevier_config.get("results_per_query", self.results_per_query))
        self.elsevier = (
            ElsevierSearchTool(max_results=elsevier_results) if elsevier_config.get("enabled", True) else None
        )
        tavily_config = config.get("tavily", {})
        tavily_results = int(tavily_config.get("results_per_query", self.results_per_query))
        self.tavily = TavilySearchTool(max_results=tavily_results) if tavily_config.get("enabled", True) else None
        self.embeddings = SharedSentenceTransformerEmbeddings()

    def reset_search_stats(self) -> None:
        """Clear provider-call telemetry at the start of one generation run."""

        self.last_search_stats = []
        self._search_round = 0

    def _supplementary_sources(self):
        return (
            ("Semantic Scholar", self.semantic_scholar),
            ("Springer Nature", self.springer if self.springer is not None and self.springer.is_configured else None),
            ("Elsevier Scopus", self.elsevier if self.elsevier is not None and self.elsevier.is_configured else None),
            ("Tavily", self.tavily if self.tavily is not None and self.tavily.is_configured else None),
        )

    def _supplementary_results(self, queries: Sequence[str]) -> list[list[dict[str, Any]]]:
        """Search non-arXiv academic sources for the supplied queries."""

        return self._search_sources(queries, include_arxiv=False)

    @staticmethod
    def _source_results(source_name: str, source, queries: Sequence[str]) -> list[list[dict[str, Any]]]:
        """Search one source serially so its rate-limit stop remains effective."""

        ranked_results: list[list[dict[str, Any]]] = []
        for query in queries:
            ranked_results.append(source.search_papers(query=query))
            if getattr(source, "last_error_status", None) in (429, 503):
                logger.warning(
                    "%s returned HTTP %s; skipping its remaining queries in this retrieval round.",
                    source_name,
                    source.last_error_status,
                )
                break
        return ranked_results

    def _search_sources(
        self,
        queries: Sequence[str],
        *,
        include_arxiv: bool,
    ) -> list[list[dict[str, Any]]]:
        """Search configured providers concurrently while preserving result order."""

        normalized_queries = tuple(query.strip() for query in queries if query.strip())
        if not normalized_queries:
            return []

        tasks = []
        if include_arxiv:
            tasks.append(("arXiv", lambda: self._arxiv_results(normalized_queries)))
        tasks.extend(
            (
                source_name,
                lambda source_name=source_name, source=source: self._source_results(
                    source_name,
                    source,
                    normalized_queries,
                ),
            )
            for source_name, source in self._supplementary_sources()
            if source is not None
        )
        if not tasks:
            return []

        self._search_round += 1
        search_round = self._search_round
        started_at = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [(source_name, executor.submit(search)) for source_name, search in tasks]
            ranked_results: list[list[dict[str, Any]]] = []
            for source_name, future in futures:
                try:
                    source_results = future.result()
                    ranked_results.extend(source_results)
                    logger.info(
                        "%s search completed queries=%d/%d results=%d elapsed_ms=%d",
                        source_name,
                        len(source_results),
                        len(normalized_queries),
                        sum(len(results) for results in source_results),
                        int((time.monotonic() - started_at) * 1000),
                    )
                    self.last_search_stats.append(
                        {
                            "round": search_round,
                            "source": source_name,
                            "queries_completed": len(source_results),
                            "queries_requested": len(normalized_queries),
                            "results": sum(len(results) for results in source_results),
                            "elapsed_ms": int((time.monotonic() - started_at) * 1000),
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    logger.error("%s search failed: %s", source_name, redact_secrets(str(exc)))
                    self.last_search_stats.append(
                        {
                            "round": search_round,
                            "source": source_name,
                            "queries_completed": 0,
                            "queries_requested": len(normalized_queries),
                            "results": 0,
                            "elapsed_ms": int((time.monotonic() - started_at) * 1000),
                            "status": "error",
                        }
                    )
        return ranked_results

    def retrieve_original_goal(self, original_query: str) -> list[Document]:
        """Search the unmodified research goal across all configured sources concurrently."""

        original_query = original_query.strip()
        if not original_query:
            return []

        ranked_results = self._search_sources((original_query,), include_arxiv=True)

        return self._rank_documents(
            original_query,
            SearchQueryPlan(queries=(), required_terms=()),
            ranked_results,
        )

    def _arxiv_results(self, queries: Sequence[str]) -> list[list[dict[str, Any]]]:
        """Search arXiv, stopping the batch when the service rate-limits us."""

        ranked_results: list[list[dict[str, Any]]] = []
        for query in queries:
            ranked_results.append(
                self.arxiv.search_papers(
                    query=query,
                    max_results=self.results_per_query,
                    sort_by="relevance",
                )
            )
            if getattr(self.arxiv, "last_error_status", None) in (429, 503):
                logger.warning(
                    "arXiv returned HTTP %s; skipping its remaining queries in this retrieval round.",
                    self.arxiv.last_error_status,
                )
                break
        return ranked_results

    def _rank_documents(
        self,
        original_query: str,
        query_plan: SearchQueryPlan,
        ranked_results: Sequence[Sequence[dict[str, Any]]],
    ) -> list[Document]:
        """Fuse source results, enforce requested entities, and rerank."""

        fused_papers = reciprocal_rank_fusion(ranked_results, k=self.rrf_k)
        relevant_papers = [
            paper for paper in fused_papers if self._contains_required_term(paper, query_plan.required_terms)
        ]
        if not relevant_papers:
            logger.warning("RAG retrieval found no papers containing required terms %s.", query_plan.required_terms)
            return []

        documents = [
            self._paper_to_document(paper)
            for paper in relevant_papers
            if paper.get("abstract") and paper.get("arxiv_id")
        ]
        if not documents:
            return []

        vector_store = InMemoryVectorStore(embedding=self.embeddings)
        vector_store.add_documents(
            documents=documents,
            ids=[str(document.metadata["source_id"]) for document in documents],
        )
        ranked_documents = vector_store.similarity_search(original_query, k=len(documents))
        selected = list(ranked_documents[: min(self.top_k, len(ranked_documents))])
        selected = self._promote_downloadable_documents(selected, ranked_documents)
        logger.info(
            "RAG selected %d sources (%d downloadable) from %d entity-matched candidates: %s",
            len(selected),
            sum(self._has_allowed_pdf(document) for document in selected),
            len(documents),
            [document.metadata.get("source_id") for document in selected],
        )
        return selected

    def _has_allowed_pdf(self, document: Document) -> bool:
        pdf_url = str(document.metadata.get("pdf_url", "")).strip()
        parsed = urlparse(pdf_url)
        host = (parsed.hostname or "").casefold()
        return parsed.scheme in {"http", "https"} and host in self.downloadable_pdf_hosts

    def _promote_downloadable_documents(
        self,
        selected: list[Document],
        ranked_documents: Sequence[Document],
    ) -> list[Document]:
        """Keep enough safe PDF candidates in the semantic top-k when available."""

        required = self.minimum_downloadable_sources
        if required <= 0:
            return selected

        selected_ids = {str(document.metadata.get("source_id", "")) for document in selected}
        downloadable_count = sum(self._has_allowed_pdf(document) for document in selected)
        for candidate in ranked_documents:
            if downloadable_count >= required:
                break
            candidate_id = str(candidate.metadata.get("source_id", ""))
            if candidate_id in selected_ids or not self._has_allowed_pdf(candidate):
                continue
            replacement_index = next(
                (index for index in range(len(selected) - 1, -1, -1) if not self._has_allowed_pdf(selected[index])),
                None,
            )
            if replacement_index is None:
                if len(selected) >= self.top_k:
                    break
                selected.append(candidate)
            else:
                selected_ids.discard(str(selected[replacement_index].metadata.get("source_id", "")))
                selected[replacement_index] = candidate
            selected_ids.add(candidate_id)
            downloadable_count += 1

        if downloadable_count < required:
            logger.warning(
                "RAG found only %d/%d downloadable PDF source(s) in the retrieved candidate set.",
                downloadable_count,
                required,
            )
        return selected

    def retrieve(
        self,
        original_query: str,
        query_plan: SearchQueryPlan,
    ) -> list[Document]:
        """Retrieve expanded-query evidence from every configured source."""

        original_query = original_query.strip()
        if not original_query:
            return []

        ranked_results = self._search_sources(query_plan.queries, include_arxiv=True)
        return self._rank_documents(original_query, query_plan, ranked_results)

    @staticmethod
    def _contains_required_term(
        paper: dict[str, Any],
        required_terms: Sequence[str],
    ) -> bool:
        if not required_terms:
            return True
        searchable_text = " ".join(
            (
                str(paper.get("title", "")),
                str(paper.get("abstract", "")),
            )
        ).casefold()
        return any(term.casefold() in searchable_text for term in required_terms)

    def retrieve_fallback(
        self,
        original_query: str,
        query_plan: SearchQueryPlan,
    ) -> list[Document]:
        """Retrieve fallback academic papers after arXiv evidence is exhausted."""

        original_query = original_query.strip()
        if not original_query:
            return []

        ranked_results = self._supplementary_results(query_plan.queries)

        if not ranked_results:
            return []

        fused_papers = reciprocal_rank_fusion(ranked_results, k=self.rrf_k)
        documents = [
            self._paper_to_document(paper) for paper in fused_papers if paper.get("abstract") and paper.get("arxiv_id")
        ]
        if not documents:
            logger.info(
                "Fallback search returned no documents for query %r.",
                original_query,
            )
            return []

        vector_store = InMemoryVectorStore(embedding=self.embeddings)
        vector_store.add_documents(
            documents=documents,
            ids=[str(document.metadata["source_id"]) for document in documents],
        )
        selected = vector_store.similarity_search(
            original_query,
            k=min(self.top_k, len(documents)),
        )
        logger.info(
            "Fallback search selected %d source(s) from %d candidate(s).",
            len(selected),
            len(documents),
        )
        return selected

    def _paper_to_document(
        self,
        paper: dict[str, Any],
    ) -> Document:
        arxiv_id = str(paper["arxiv_id"])
        source_id = (
            arxiv_id
            if any(arxiv_id.startswith(p) for p in ("s2:", "springer:", "elsevier:", "tavily:", "doi:"))
            else f"arXiv:{arxiv_id}"
        )

        abstract = str(paper.get("abstract", ""))[: self.max_abstract_chars]

        page_content = (
            f"Source ID: {source_id}\n"
            f"Title: {paper.get('title', 'Untitled')}\n"
            f"Published: {paper.get('published', 'Unknown')}\n"
            f"Primary category: "
            f"{paper.get('primary_category', 'Unknown')}\n"
            f"Abstract: {abstract}"
        )

        return Document(
            page_content=page_content,
            metadata={
                "source_id": source_id,
                "arxiv_id": arxiv_id,
                "title": paper.get("title", "Untitled"),
                "abstract": abstract,
                "authors": paper.get("authors", []),
                "published": paper.get("published"),
                "primary_category": paper.get("primary_category"),
                "arxiv_url": paper.get("arxiv_url"),
                "pdf_url": paper.get("pdf_url"),
                "source": paper.get("source", "arxiv"),
                "rrf_score": paper.get("_rrf_score"),
            },
        )


def format_documents_for_prompt(
    documents: Sequence[Document],
) -> str:
    sections: list[str] = []

    for document in documents:
        source_id = document.metadata.get(
            "source_id",
            "unknown",
        )
        sections.append(f'<source id="{source_id}">\n{document.page_content}\n</source>')

    return "\n\n".join(sections)


def format_documents_for_grading(
    documents: Sequence[Document],
    *,
    max_abstract_chars: int = 1600,
    max_total_chars: int = 24000,
) -> str:
    """Build a bounded source digest for relevance and coverage grading.

    Every source receives a fair share of the total budget so later corrective
    retrieval rounds cannot crowd new evidence out of the grading prompt.
    """

    remaining = max(1000, int(max_total_chars))
    abstract_cap = max(0, int(max_abstract_chars))
    sections: list[str] = []
    document_list = list(documents)

    for index, document in enumerate(document_list):
        separator = "\n\n" if sections else ""
        remaining -= len(separator)
        sources_left = len(document_list) - index
        fair_share = max(1, remaining // max(1, sources_left))

        source_id = str(document.metadata.get("source_id", "unknown"))[:160]
        title = str(document.metadata.get("title", "Untitled"))[:300]
        published = str(document.metadata.get("published", "Unknown"))[:40]
        category = str(document.metadata.get("primary_category", "Unknown"))[:80]
        abstract = str(document.metadata.get("abstract", ""))
        header = (
            f'<source id="{source_id}">\n'
            f"Source ID: {source_id}\n"
            f"Title: {title}\n"
            f"Published: {published}\n"
            f"Primary category: {category}\n"
            "Abstract: "
        )
        footer = "\n</source>"
        available_abstract_chars = max(0, fair_share - len(header) - len(footer))
        bounded_abstract = abstract[: min(abstract_cap, available_abstract_chars)]
        section = f"{header}{bounded_abstract}{footer}"

        if len(section) > remaining:
            break
        sections.append(section)
        remaining -= len(section)

    return "\n\n".join(sections)


def serialize_documents(
    documents: Sequence[Document],
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": document.metadata.get("source_id"),
            "arxiv_id": document.metadata.get("arxiv_id"),
            "title": document.metadata.get("title"),
            "abstract": document.metadata.get("abstract"),
            "authors": document.metadata.get("authors", []),
            "published": document.metadata.get("published"),
            "primary_category": document.metadata.get("primary_category"),
            "arxiv_url": document.metadata.get("arxiv_url"),
            "pdf_url": document.metadata.get("pdf_url"),
            "source": document.metadata.get("source"),
            "rrf_score": document.metadata.get("rrf_score"),
            "full_text_indexed": document.metadata.get("full_text_indexed", False),
            "full_text_chunks_used": document.metadata.get("full_text_chunks_used", 0),
        }
        for document in documents
    ]
