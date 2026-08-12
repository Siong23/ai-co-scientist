from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence, cast
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .config import config
from .evidence import (
    EvidenceChunk,
    EvidenceDocumentType,
    EvidenceSource,
    EvidenceType,
    canonicalize_url,
    coerce_evidence,
    evidence_from_result,
)
from .tools.arxiv_search import ArxivSearchTool
from .tools.elsevier_search import ElsevierSearchTool
from .tools.pdf_urls import find_pdf_url
from .tools.semantic_scholar_search import SemanticScholarSearchTool
from .tools.springer_search import SpringerSearchTool
from .tools.tavily_search import TavilySearchTool
from .utils import get_sentence_transformer_model, logger, redact_secrets

_TAVILY_CHUNK_MARKER = re.compile(r"<chunk\s+\d+>\s*", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceAspect:
    """One indispensable evidence dimension extracted from a research goal."""

    aspect_id: str
    description: str


SearchRoute = Literal["academic", "web", "official", "news", "all"]


@dataclass(frozen=True)
class SearchQuery:
    """One routed search operation produced by the Query Rewriter."""

    query: str
    sub_question: str = ""
    purpose: str = ""
    source_type: SearchRoute = "all"
    preferred_domains: tuple[str, ...] = ()
    freshness: str | None = None
    evidence_requirement_id: str | None = None

    def __post_init__(self) -> None:
        source_type = str(self.source_type).strip().casefold()
        if source_type not in {"academic", "web", "official", "news", "all"}:
            raise ValueError(f"Unsupported search source_type: {self.source_type!r}")
        freshness = self.freshness.strip().casefold() if self.freshness else None
        if freshness not in {None, "day", "week", "month", "year"}:
            raise ValueError(f"Unsupported search freshness: {self.freshness!r}")
        object.__setattr__(self, "query", self.query.strip())
        object.__setattr__(self, "sub_question", self.sub_question.strip())
        object.__setattr__(self, "purpose", self.purpose.strip())
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(
            self,
            "preferred_domains",
            tuple(dict.fromkeys(domain.strip().casefold() for domain in self.preferred_domains if domain.strip())),
        )
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(
            self,
            "evidence_requirement_id",
            self.evidence_requirement_id.strip() if self.evidence_requirement_id else None,
        )


@dataclass(frozen=True)
class SearchQueryPlan:
    """Structured query-rewriting output used for literature retrieval."""

    queries: tuple[SearchQuery | str, ...]
    required_terms: tuple[str, ...]
    explicit_requirements: tuple[EvidenceAspect, ...] = ()
    exploration_directions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "queries",
            tuple(
                query
                if isinstance(query, SearchQuery)
                else SearchQuery(query=str(query), source_type="all")
                for query in self.queries
                if isinstance(query, SearchQuery) or str(query).strip()
            ),
        )

    @property
    def query_texts(self) -> tuple[str, ...]:
        """Return plain query strings for compatibility and display."""

        return tuple(query.query for query in self.queries)


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
    ranked_results: Sequence[Sequence[EvidenceSource | dict[str, Any]]],
    k: int = 60,
) -> list[EvidenceSource]:
    """Fuse result rankings and deduplicate provider-neutral evidence."""

    scores: dict[str, float] = {}
    evidence_by_key: dict[str, EvidenceSource] = {}

    for results in ranked_results:
        for rank, raw_evidence in enumerate(results):
            evidence = coerce_evidence(raw_evidence)
            canonical_key = evidence.canonical_key
            if not canonical_key:
                continue

            existing = evidence_by_key.get(canonical_key)
            if existing is None:
                evidence_by_key[canonical_key] = evidence
            else:
                existing_contexts = tuple(existing.metadata.get("query_contexts", ()))
                incoming_contexts = tuple(evidence.metadata.get("query_contexts", ()))
                merged_contexts = []
                seen_contexts: set[tuple[Any, ...]] = set()
                for context in (*existing_contexts, *incoming_contexts):
                    if not isinstance(context, dict):
                        continue
                    context_key = (
                        context.get("query"),
                        context.get("sub_question"),
                        context.get("purpose"),
                        context.get("source_type"),
                        context.get("evidence_requirement_id"),
                    )
                    if context_key not in seen_contexts:
                        seen_contexts.add(context_key)
                        merged_contexts.append(context)
                preferred = (
                    evidence
                    if (evidence.search_score or 0) > (existing.search_score or 0)
                    else existing
                )
                merged_metadata = dict(preferred.metadata)
                merged_metadata["query_contexts"] = tuple(merged_contexts)
                evidence_by_key[canonical_key] = replace(
                    preferred,
                    metadata=merged_metadata,
                )
            scores[canonical_key] = scores.get(canonical_key, 0.0) + (1.0 / (k + rank + 1))

    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )
    return [evidence_by_key[key].with_rrf_score(scores[key]) for key in ranked_keys]


class ResearchRetriever:
    """Orchestrate academic and web retrieval into one evidence layer."""

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
        self.max_web_extract_results = max(
            1,
            int(tavily_config.get("max_extract_results", 5)),
        )
        self.max_web_evidence_chunks = max(
            1,
            int(tavily_config.get("max_evidence_chunks", 8)),
        )
        self.max_web_chunk_chars = max(
            200,
            int(tavily_config.get("max_chunk_chars", 1600)),
        )
        self.tavily = (
            TavilySearchTool(
                max_results=tavily_results,
                search_depth=str(tavily_config.get("search_depth", "advanced")),
                search_chunks_per_source=int(
                    tavily_config.get("search_chunks_per_source", 3)
                ),
                extract_depth=str(tavily_config.get("extract_depth", "basic")),
                extract_chunks_per_source=int(
                    tavily_config.get("extract_chunks_per_source", 3)
                ),
            )
            if tavily_config.get("enabled", True)
            else None
        )
        self.embeddings = SharedSentenceTransformerEmbeddings()

    def reset_search_stats(self) -> None:
        """Clear provider-call telemetry at the start of one generation run."""

        self.last_search_stats = []
        self._search_round = 0

    def _academic_sources(self):
        return (
            ("Semantic Scholar", "semantic_scholar", self.semantic_scholar),
            (
                "Springer Nature",
                "springer",
                self.springer if self.springer is not None and self.springer.is_configured else None,
            ),
            (
                "Elsevier Scopus",
                "elsevier",
                self.elsevier if self.elsevier is not None and self.elsevier.is_configured else None,
            ),
        )

    def _web_sources(self):
        return (
            (
                "Tavily",
                "tavily",
                self.tavily if self.tavily is not None and self.tavily.is_configured else None,
            ),
        )

    def _supplementary_results(
        self,
        queries: Sequence[SearchQuery | str],
    ) -> list[list[EvidenceSource]]:
        """Search non-arXiv academic sources for the supplied queries."""

        return self._search_sources(queries, include_arxiv=False, include_web=False)

    @staticmethod
    def _source_results(
        source_name: str,
        provider: str,
        source_type: EvidenceType,
        source,
        queries: Sequence[SearchQuery],
    ) -> list[list[EvidenceSource]]:
        """Search and normalize one provider while preserving query ranking."""

        ranked_results: list[list[EvidenceSource]] = []
        for search_query in queries:
            if source_type == "web":
                search_options: dict[str, Any] = {}
                if search_query.preferred_domains:
                    search_options["include_domains"] = search_query.preferred_domains
                if search_query.freshness:
                    search_options["time_range"] = search_query.freshness
                if search_query.source_type == "news":
                    search_options["topic"] = "news"
                raw_results = source.search(query=search_query.query, **search_options)
            else:
                raw_results = source.search_papers(query=search_query.query)
            query_context = {
                "query": search_query.query,
                "sub_question": search_query.sub_question,
                "purpose": search_query.purpose,
                "source_type": search_query.source_type,
                "preferred_domains": search_query.preferred_domains,
                "freshness": search_query.freshness,
                "evidence_requirement_id": search_query.evidence_requirement_id,
            }
            normalized_results = []
            for result in raw_results:
                evidence = evidence_from_result(result, provider, source_type)
                document_type = evidence.document_type
                if source_type == "web" and search_query.source_type == "official":
                    document_type = "official_docs"
                elif source_type == "web" and search_query.source_type == "news":
                    document_type = "news"
                evidence = replace(
                    evidence,
                    document_type=document_type,
                    retrieval_query=search_query.query,
                    sub_question=search_query.sub_question or None,
                    purpose=search_query.purpose,
                    evidence_requirement_id=search_query.evidence_requirement_id,
                )
                metadata = dict(evidence.metadata)
                metadata.update(
                    {
                        "search_query": search_query.query,
                        "sub_question": search_query.sub_question,
                        "purpose": search_query.purpose,
                        "planned_source_type": search_query.source_type,
                        "preferred_domains": search_query.preferred_domains,
                        "freshness": search_query.freshness,
                        "evidence_requirement_id": search_query.evidence_requirement_id,
                        "query_contexts": (query_context,),
                    }
                )
                normalized_results.append(replace(evidence, metadata=metadata))
            ranked_results.append(normalized_results)
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
        queries: Sequence[SearchQuery | str],
        *,
        include_arxiv: bool,
        include_web: bool = True,
    ) -> list[list[EvidenceSource]]:
        """Search configured providers concurrently while preserving result order."""

        normalized_queries = tuple(
            query if isinstance(query, SearchQuery) else SearchQuery(query=str(query), source_type="all")
            for query in queries
            if isinstance(query, SearchQuery) or str(query).strip()
        )
        if not normalized_queries:
            return []

        academic_queries = tuple(
            query for query in normalized_queries if query.source_type in ("academic", "all")
        )
        web_queries = tuple(
            query for query in normalized_queries if query.source_type in ("web", "official", "news", "all")
        )

        tasks = []
        if include_arxiv and academic_queries:
            tasks.append(("arXiv", len(academic_queries), lambda: self._arxiv_results(academic_queries)))
        tasks.extend(
            (
                source_name,
                len(academic_queries),
                lambda source_name=source_name, provider=provider, source=source: self._source_results(
                    source_name,
                    provider,
                    "academic",
                    source,
                    academic_queries,
                ),
            )
            for source_name, provider, source in self._academic_sources()
            if source is not None and academic_queries
        )
        if include_web and web_queries:
            tasks.extend(
                (
                    source_name,
                    len(web_queries),
                    lambda source_name=source_name, provider=provider, source=source: self._source_results(
                        source_name,
                        provider,
                        "web",
                        source,
                        web_queries,
                    ),
                )
                for source_name, provider, source in self._web_sources()
                if source is not None
            )
        if not tasks:
            return []

        self._search_round += 1
        search_round = self._search_round
        started_at = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [
                (source_name, query_total, executor.submit(search))
                for source_name, query_total, search in tasks
            ]
            ranked_results: list[list[EvidenceSource]] = []
            for source_name, query_total, future in futures:
                try:
                    source_results = future.result()
                    ranked_results.extend(source_results)
                    logger.info(
                        "%s search completed queries=%d/%d results=%d elapsed_ms=%d",
                        source_name,
                        len(source_results),
                        query_total,
                        sum(len(results) for results in source_results),
                        int((time.monotonic() - started_at) * 1000),
                    )
                    self.last_search_stats.append(
                        {
                            "round": search_round,
                            "source": source_name,
                            "queries_completed": len(source_results),
                            "queries_requested": query_total,
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
                            "queries_requested": query_total,
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

    def _arxiv_results(self, queries: Sequence[SearchQuery]) -> list[list[EvidenceSource]]:
        """Search arXiv, stopping the batch when the service rate-limits us."""

        ranked_results: list[list[EvidenceSource]] = []
        for search_query in queries:
            raw_results = self.arxiv.search_papers(
                query=search_query.query,
                max_results=self.results_per_query,
                sort_by="relevance",
            )
            query_context = {
                "query": search_query.query,
                "sub_question": search_query.sub_question,
                "purpose": search_query.purpose,
                "source_type": search_query.source_type,
                "preferred_domains": search_query.preferred_domains,
                "freshness": search_query.freshness,
                "evidence_requirement_id": search_query.evidence_requirement_id,
            }
            normalized_results = []
            for result in raw_results:
                evidence = evidence_from_result(result, "arxiv", "academic")
                evidence = replace(
                    evidence,
                    retrieval_query=search_query.query,
                    sub_question=search_query.sub_question or None,
                    purpose=search_query.purpose,
                    evidence_requirement_id=search_query.evidence_requirement_id,
                )
                metadata = dict(evidence.metadata)
                metadata.update(
                    {
                        "search_query": search_query.query,
                        "sub_question": search_query.sub_question,
                        "purpose": search_query.purpose,
                        "planned_source_type": search_query.source_type,
                        "preferred_domains": search_query.preferred_domains,
                        "freshness": search_query.freshness,
                        "evidence_requirement_id": search_query.evidence_requirement_id,
                        "query_contexts": (query_context,),
                    }
                )
                normalized_results.append(replace(evidence, metadata=metadata))
            ranked_results.append(normalized_results)
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
        ranked_results: Sequence[Sequence[EvidenceSource]],
    ) -> list[Document]:
        """Fuse source results, enforce requested entities, and rerank."""

        fused_evidence = reciprocal_rank_fusion(ranked_results, k=self.rrf_k)
        relevant_evidence = [
            evidence for evidence in fused_evidence if self._contains_required_term(evidence, query_plan.required_terms)
        ]
        if not relevant_evidence:
            logger.warning("RAG retrieval found no evidence containing required terms %s.", query_plan.required_terms)
            return []

        documents = [
            self._evidence_to_document(evidence)
            for evidence in relevant_evidence
            if evidence.text and evidence.source_id
        ]
        if not documents:
            return []

        vector_store = InMemoryVectorStore(embedding=self.embeddings)
        vector_store.add_documents(
            documents=documents,
            ids=[str(document.metadata["source_id"]) for document in documents],
        )
        ranked_documents = self._similarity_rank_documents(
            vector_store,
            original_query,
            len(documents),
            score_field="document_rerank_score",
        )
        selected = list(ranked_documents[: min(self.top_k, len(ranked_documents))])
        selected = self._promote_downloadable_documents(selected, ranked_documents)
        selected = self._extract_selected_web_documents(selected, original_query)
        logger.info(
            "RAG selected %d sources (%d downloadable) from %d entity-matched candidates: %s",
            len(selected),
            sum(self._has_allowed_pdf(document) for document in selected),
            len(documents),
            [document.metadata.get("source_id") for document in selected],
        )
        return selected

    @staticmethod
    def _similarity_rank_documents(
        vector_store: InMemoryVectorStore,
        query: str,
        count: int,
        *,
        score_field: str,
    ) -> list[Document]:
        """Rank documents and retain a comparable similarity score."""

        scored_results = vector_store.similarity_search_with_score(query, k=count)
        if isinstance(scored_results, list) and all(
            isinstance(item, tuple) and len(item) == 2
            for item in scored_results
        ):
            ranked = []
            for document, score in scored_results:
                metadata = dict(document.metadata)
                metadata[score_field] = float(score)
                metadata["rerank_score"] = float(score)
                ranked.append(Document(page_content=document.page_content, metadata=metadata))
            return ranked

        # Test doubles and older vector-store implementations may expose only
        # similarity_search. Preserve deterministic rank-derived scores there.
        documents = vector_store.similarity_search(query, k=count)
        ranked = []
        for rank, document in enumerate(documents):
            score = 1.0 / (rank + 1)
            metadata = dict(document.metadata)
            metadata[score_field] = score
            metadata["rerank_score"] = score
            ranked.append(Document(page_content=document.page_content, metadata=metadata))
        return ranked

    def _extract_selected_web_documents(
        self,
        selected: Sequence[Document],
        query: str,
    ) -> list[Document]:
        """Turn top-ranked Tavily discoveries into bounded web evidence."""

        academic_documents = [
            document
            for document in selected
            if document.metadata.get("source_type") != "web"
        ]
        ready_web = [
            document
            for document in selected
            if document.metadata.get("source_type") == "web"
            and document.metadata.get("content_extracted") is True
        ]
        pending_web = [
            document
            for document in selected
            if document.metadata.get("source_type") == "web"
            and document.metadata.get("content_extracted") is not True
        ][: self.max_web_extract_results]
        if not pending_web:
            return academic_documents + self._rank_extracted_web_chunks(
                ready_web,
                query,
            )
        if self.tavily is None or not self.tavily.is_configured:
            logger.warning(
                "Discarding %d search-only web result(s) because extraction is unavailable.",
                len(pending_web),
            )
            return academic_documents + self._rank_extracted_web_chunks(ready_web, query)

        extraction_groups: dict[str, list[str]] = {}
        for document in pending_web:
            extraction_query = str(
                document.metadata.get("sub_question")
                or query
            ).strip()
            extraction_groups.setdefault(extraction_query, []).append(
                str(document.metadata.get("url") or "")
            )

        extracted_by_url: dict[str, str] = {}
        for extraction_query, urls in extraction_groups.items():
            extracted = self.tavily.extract(urls=urls, query=extraction_query)
            if isinstance(extracted, dict):
                extracted_by_url.update(extracted)

        enriched_web: list[Document] = []
        for document in pending_web:
            canonical_url = canonicalize_url(
                str(
                    document.metadata.get("canonical_url")
                    or document.metadata.get("url")
                    or ""
                )
            )
            extracted_content = str(extracted_by_url.get(canonical_url) or "").strip()
            if not extracted_content:
                continue
            enriched_web.append(
                self._with_extracted_web_content(document, extracted_content)
            )

        web_chunks = self._rank_extracted_web_chunks(
            [*ready_web, *enriched_web],
            query,
        )
        evidence = [*academic_documents, *web_chunks]

        logger.info(
            "Web evidence extraction retained %d/%d selected web discovery result(s).",
            sum(document.metadata.get("source_type") == "web" for document in evidence),
            sum(document.metadata.get("source_type") == "web" for document in selected),
        )
        return evidence

    def open_web_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> list[Document]:
        """Open known web URLs and return query-ranked passages.

        This is intentionally separate from search: callers must first select
        known evidence URLs, while ``query`` expresses the information need for
        extraction and chunk reranking.
        """

        query = query.strip()
        if not query or self.tavily is None or not self.tavily.is_configured:
            return []

        documents_by_url: dict[str, Document] = {}
        for document in documents:
            if document.metadata.get("source_type") != "web":
                continue
            canonical_url = canonicalize_url(
                str(
                    document.metadata.get("canonical_url")
                    or document.metadata.get("url")
                    or ""
                )
            )
            if canonical_url and canonical_url not in documents_by_url:
                documents_by_url[canonical_url] = document
            if len(documents_by_url) >= self.max_web_extract_results:
                break

        if not documents_by_url:
            return []

        started_at = time.monotonic()
        extracted_by_url = self.tavily.extract(
            urls=list(documents_by_url),
            query=query,
        )
        if not isinstance(extracted_by_url, dict):
            return []

        opened_documents: list[Document] = []
        for canonical_url, document in documents_by_url.items():
            extracted_content = str(extracted_by_url.get(canonical_url) or "").strip()
            if not extracted_content:
                continue
            metadata = dict(document.metadata)
            parent_source_id = str(
                metadata.get("parent_source_id")
                or metadata.get("source_id")
                or ""
            )
            metadata.update(
                {
                    "source_id": parent_source_id,
                    "parent_source_id": parent_source_id,
                    "canonical_url": canonical_url,
                    "retrieval_query": query,
                    "search_query": query,
                    "sub_question": query,
                }
            )
            for chunk_field in (
                "chunk_id",
                "chunk_index",
                "chunk_count",
                "chunk_rerank_score",
            ):
                metadata.pop(chunk_field, None)
            base_document = Document(
                page_content=document.page_content,
                metadata=metadata,
            )
            opened_documents.append(
                self._with_extracted_web_content(
                    base_document,
                    extracted_content,
                )
            )

        self.last_search_stats.append(
            {
                "round": self._search_round,
                "source": "Tavily Extract",
                "action": "open_url",
                "queries_completed": 1,
                "queries_requested": 1,
                "results": len(opened_documents),
                "elapsed_ms": int((time.monotonic() - started_at) * 1000),
                "status": "ok" if opened_documents else "empty",
            }
        )
        return self._rank_extracted_web_chunks(opened_documents, query)

    def _split_extracted_content(self, content: str) -> list[str]:
        """Split Tavily output into bounded, independently rankable chunks."""

        normalized = content.strip()
        if not normalized:
            return []
        if _TAVILY_CHUNK_MARKER.search(normalized):
            candidates = _TAVILY_CHUNK_MARKER.split(normalized)
        else:
            candidates = re.split(r"\n\s*\n", normalized)

        chunks: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            pieces = [
                candidate[index : index + self.max_web_chunk_chars].strip()
                for index in range(0, len(candidate), self.max_web_chunk_chars)
            ]
            for piece in pieces:
                normalized_piece = " ".join(piece.casefold().split())
                if piece and normalized_piece not in seen:
                    seen.add(normalized_piece)
                    chunks.append(piece)
        return chunks

    def _document_to_evidence_chunks(
        self,
        document: Document,
    ) -> list[Document]:
        """Create one Document per extracted passage while preserving provenance."""

        metadata = document.metadata
        content = str(metadata.get("content") or metadata.get("summary") or "")
        chunks = self._split_extracted_content(content)
        if not chunks:
            return []

        parent_source_id = str(metadata.get("parent_source_id") or metadata.get("source_id") or "")
        document_type = cast(
            EvidenceDocumentType,
            str(metadata.get("document_type") or "webpage"),
        )
        retrieval_query = str(metadata.get("retrieval_query") or metadata.get("search_query") or "")
        sub_question = str(metadata.get("sub_question") or "").strip() or None
        evidence_chunks = [
            EvidenceChunk(
                chunk_id=f"{parent_source_id}#chunk-{index}",
                source_id=parent_source_id,
                content=chunk,
                retrieval_query=retrieval_query,
                sub_question=sub_question,
                document_type=document_type,
                chunk_index=index,
                chunk_count=len(chunks),
                metadata=metadata,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]

        documents: list[Document] = []
        for chunk in evidence_chunks:
            chunk_metadata = dict(metadata)
            chunk_source_id = (
                parent_source_id
                if chunk.chunk_index == 1
                else chunk.chunk_id
            )
            chunk_metadata.update(
                {
                    "source_id": chunk_source_id,
                    "parent_source_id": parent_source_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_count": chunk.chunk_count,
                    "summary": chunk.content,
                    "content": chunk.content,
                    "abstract": chunk.content,
                    "content_extracted": True,
                    "full_text_available": True,
                    "retrieval_query": chunk.retrieval_query,
                    "sub_question": chunk.sub_question,
                    # Only the stable parent document should be considered for
                    # optional PDF indexing; sibling web chunks are evidence.
                    "pdf_url": metadata.get("pdf_url") if chunk.chunk_index == 1 else None,
                }
            )
            page_content = (
                f"Source ID: {chunk_source_id}\n"
                "Source type: web\n"
                f"Document type: {chunk.document_type}\n"
                f"Title: {metadata.get('title')}\n"
                f"URL: {metadata.get('url')}\n"
                f"Domain: {metadata.get('domain') or 'Unknown'}\n"
                f"Published: {metadata.get('published_at') or 'Unknown'}\n"
                f"Evidence chunk: {chunk.chunk_index}/{chunk.chunk_count}\n"
                f"Web content: {chunk.content}"
            )
            documents.append(Document(page_content=page_content, metadata=chunk_metadata))
        return documents

    def _rank_extracted_web_chunks(
        self,
        documents: Sequence[Document],
        fallback_query: str,
    ) -> list[Document]:
        """Rerank extracted passages against their assigned sub-question."""

        chunks = [
            chunk
            for document in documents
            for chunk in self._document_to_evidence_chunks(document)
        ]
        if not chunks:
            return []
        if len(chunks) == 1:
            metadata = dict(chunks[0].metadata)
            metadata["chunk_rerank_score"] = 1.0
            metadata["rerank_score"] = 1.0
            return [Document(page_content=chunks[0].page_content, metadata=metadata)]

        grouped_chunks: dict[str, list[Document]] = {}
        for chunk in chunks:
            ranking_query = str(
                chunk.metadata.get("sub_question")
                or chunk.metadata.get("retrieval_query")
                or fallback_query
            ).strip()
            grouped_chunks.setdefault(ranking_query, []).append(chunk)

        ranked_chunks: list[Document] = []
        for ranking_query, query_chunks in grouped_chunks.items():
            vector_store = InMemoryVectorStore(embedding=self.embeddings)
            vector_store.add_documents(
                documents=query_chunks,
                ids=[str(chunk.metadata["chunk_id"]) for chunk in query_chunks],
            )
            ranked_chunks.extend(
                self._similarity_rank_documents(
                    vector_store,
                    ranking_query,
                    len(query_chunks),
                    score_field="chunk_rerank_score",
                )
            )

        ranked_chunks.sort(
            key=lambda document: (
                float(document.metadata.get("chunk_rerank_score") or 0),
                float(document.metadata.get("document_rerank_score") or 0),
                float(document.metadata.get("search_score") or 0),
            ),
            reverse=True,
        )
        return ranked_chunks[: self.max_web_evidence_chunks]

    def _with_extracted_web_content(
        self,
        document: Document,
        extracted_content: str,
    ) -> Document:
        """Return a web document whose evidence text comes from Tavily Extract."""

        content = extracted_content[: self.max_abstract_chars]
        metadata = dict(document.metadata)
        metadata.update(
            {
                "snippet": metadata.get("summary") or metadata.get("snippet") or "",
                "summary": content,
                "content": content,
                "abstract": content,
                "content_extracted": True,
                "full_text_available": True,
            }
        )
        page_content = (
            f"Source ID: {metadata.get('source_id')}\n"
            "Source type: web\n"
            f"Title: {metadata.get('title')}\n"
            f"URL: {metadata.get('url')}\n"
            f"Domain: {metadata.get('domain') or 'Unknown'}\n"
            f"Published: {metadata.get('published_at') or 'Unknown'}\n"
            f"Web content: {content}"
        )
        return Document(page_content=page_content, metadata=metadata)

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
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index].metadata.get("source_type") != "web"
                    and not self._has_allowed_pdf(selected[index])
                ),
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
        evidence: EvidenceSource,
        required_terms: Sequence[str],
    ) -> bool:
        if not required_terms:
            return True
        searchable_text = " ".join(
            (
                evidence.title,
                evidence.text,
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

        fused_evidence = reciprocal_rank_fusion(ranked_results, k=self.rrf_k)
        documents = [
            self._evidence_to_document(evidence) for evidence in fused_evidence if evidence.text and evidence.source_id
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

    def _evidence_to_document(
        self,
        evidence: EvidenceSource,
    ) -> Document:
        source_id = evidence.source_id
        summary = evidence.text[: self.max_abstract_chars]
        pdf_url = find_pdf_url(
            evidence.pdf_url,
            evidence.url,
        )

        if evidence.source_family == "web":
            content_extracted = bool(evidence.metadata.get("content_extracted"))
            content_label = "Web content" if content_extracted else "Search snippet"
            page_content = (
                f"Source ID: {source_id}\n"
                f"Source type: web\n"
                f"Document type: {evidence.document_type}\n"
                f"Title: {evidence.title}\n"
                f"URL: {evidence.url}\n"
                f"Domain: {evidence.domain or 'Unknown'}\n"
                f"Published: {evidence.published_at or 'Unknown'}\n"
                f"{content_label}: {summary}"
            )
        else:
            page_content = (
                f"Source ID: {source_id}\n"
                f"Source type: academic\n"
                f"Document type: {evidence.document_type}\n"
                f"Title: {evidence.title}\n"
                f"Published: {evidence.published_at or 'Unknown'}\n"
                f"Venue: {evidence.venue or 'Unknown'}\n"
                f"Abstract: {summary}"
            )

        return Document(
            page_content=page_content,
            metadata={
                "source_id": source_id,
                "source_type": evidence.source_family,
                "source_family": evidence.source_family,
                "document_type": evidence.document_type,
                "provider": evidence.provider,
                "title": evidence.title,
                "summary": summary,
                "content": (
                    summary
                    if evidence.source_family == "web"
                    and evidence.metadata.get("content_extracted")
                    else ""
                ),
                "snippet": evidence.summary if evidence.source_family == "web" else "",
                "content_extracted": bool(
                    evidence.metadata.get("content_extracted")
                )
                if evidence.source_family == "web"
                else None,
                "authors": list(evidence.authors),
                "published_at": evidence.published_at,
                "updated_at": evidence.updated_at,
                "url": evidence.url,
                "canonical_url": evidence.canonical_url,
                "domain": evidence.domain,
                "page_type": evidence.page_type,
                "author": evidence.author,
                "language": evidence.language,
                "source_authority": evidence.source_authority,
                "search_score": evidence.search_score,
                "rerank_score": evidence.rerank_score,
                "full_text_available": evidence.full_text_available,
                "retrieval_query": evidence.retrieval_query,
                "search_query": evidence.retrieval_query,
                "sub_question": evidence.sub_question,
                "purpose": evidence.purpose,
                "planned_source_type": evidence.metadata.get("planned_source_type"),
                "preferred_domains": list(evidence.metadata.get("preferred_domains", ())),
                "freshness": evidence.metadata.get("freshness"),
                "evidence_requirement_id": evidence.evidence_requirement_id,
                "query_contexts": list(evidence.metadata.get("query_contexts", ())),
                "doi": evidence.doi,
                "venue": evidence.venue,
                "pdf_url": pdf_url,
                "rrf_score": evidence.rrf_score,
                # Compatibility fields for the current UI and saved runs.
                "arxiv_id": evidence.metadata.get("arxiv_id") if evidence.source_family == "academic" else None,
                "abstract": summary,
                "published": evidence.published_at,
                "primary_category": evidence.venue if evidence.source_family == "academic" else "web",
                "arxiv_url": evidence.url,
                "source": evidence.provider,
            },
        )

    def _paper_to_document(self, paper: dict[str, Any]) -> Document:
        """Compatibility adapter for callers using the historical helper."""

        source_type: EvidenceType = "web" if paper.get("source_type") == "web" else "academic"
        provider = str(paper.get("provider") or paper.get("source") or "arxiv")
        return self._evidence_to_document(evidence_from_result(paper, provider, source_type))


ArxivRAGRetriever = ResearchRetriever


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

    def field(value: Any, *, limit: int, default: str = "Unknown") -> str:
        """Keep provenance fields single-line so evidence cannot spoof labels."""

        if value is None:
            return default
        normalized = re.sub(r"\s+", " ", str(value)).strip()
        return normalized[:limit] if normalized else default

    for index, document in enumerate(document_list):
        separator = "\n\n" if sections else ""
        remaining -= len(separator)
        sources_left = len(document_list) - index
        fair_share = max(1, remaining // max(1, sources_left))

        source_id = field(document.metadata.get("source_id"), limit=160, default="unknown")
        title = field(document.metadata.get("title"), limit=300, default="Untitled")
        published = field(
            document.metadata.get("published_at") or document.metadata.get("published"),
            limit=40,
        )
        updated = field(document.metadata.get("updated_at"), limit=40)
        source_type = field(
            document.metadata.get("source_family") or document.metadata.get("source_type"),
            limit=20,
            default="academic",
        )
        document_type = field(document.metadata.get("document_type") or source_type, limit=40)
        provider = field(document.metadata.get("provider") or document.metadata.get("source"), limit=80)
        url = field(
            document.metadata.get("canonical_url")
            or document.metadata.get("url")
            or document.metadata.get("arxiv_url"),
            limit=400,
        )
        domain = field(document.metadata.get("domain") or urlparse(url).hostname, limit=120)
        authority = field(
            document.metadata.get("source_authority"),
            limit=120,
            default="Unassessed; infer from URL, domain, provider, and source type",
        )
        retrieval_query = field(
            document.metadata.get("retrieval_query") or document.metadata.get("search_query"),
            limit=300,
        )
        retrieved_for = field(
            document.metadata.get("sub_question")
            or document.metadata.get("purpose")
            or document.metadata.get("evidence_requirement_id"),
            limit=300,
        )
        requested_freshness = field(document.metadata.get("freshness"), limit=40, default="Not specified")
        summary = str(document.metadata.get("summary") or document.metadata.get("abstract") or "")
        venue_line = ""
        if source_type == "academic":
            venue = field(document.metadata.get("venue") or document.metadata.get("primary_category"), limit=120)
            venue_line = f"Venue: {venue}\n"
        header = (
            f'<source id="{source_id}" type="{source_type}">\n'
            f"Source ID: {source_id}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Domain: {domain}\n"
            f"Provider: {provider}\n"
            f"Source type: {source_type}\n"
            f"Document type: {document_type}\n"
            f"Authority: {authority}\n"
            f"Published: {published}\n"
            f"Updated: {updated}\n"
            f"Retrieved for: {retrieved_for}\n"
            f"Search query: {retrieval_query}\n"
            f"Requested freshness: {requested_freshness}\n"
            f"{venue_line}"
            "Content: "
        )
        footer = "\n</source>"
        available_abstract_chars = max(0, fair_share - len(header) - len(footer))
        bounded_summary = summary[: min(abstract_cap, available_abstract_chars)]
        section = f"{header}{bounded_summary}{footer}"

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
            "source_type": document.metadata.get("source_type", "academic"),
            "source_family": document.metadata.get("source_family")
            or document.metadata.get("source_type", "academic"),
            "document_type": document.metadata.get("document_type"),
            "provider": document.metadata.get("provider") or document.metadata.get("source"),
            "url": document.metadata.get("url") or document.metadata.get("arxiv_url"),
            "canonical_url": document.metadata.get("canonical_url"),
            "domain": document.metadata.get("domain"),
            "page_type": document.metadata.get("page_type"),
            "author": document.metadata.get("author"),
            "summary": document.metadata.get("summary") or document.metadata.get("abstract"),
            "content": document.metadata.get("content"),
            "content_extracted": document.metadata.get("content_extracted"),
            "published_at": document.metadata.get("published_at") or document.metadata.get("published"),
            "updated_at": document.metadata.get("updated_at"),
            "language": document.metadata.get("language"),
            "source_authority": document.metadata.get("source_authority"),
            "search_score": document.metadata.get("search_score"),
            "rerank_score": document.metadata.get("rerank_score"),
            "document_rerank_score": document.metadata.get("document_rerank_score"),
            "chunk_rerank_score": document.metadata.get("chunk_rerank_score"),
            "full_text_available": document.metadata.get("full_text_available", False),
            "retrieval_query": document.metadata.get("retrieval_query")
            or document.metadata.get("search_query"),
            "search_query": document.metadata.get("search_query"),
            "sub_question": document.metadata.get("sub_question"),
            "purpose": document.metadata.get("purpose"),
            "planned_source_type": document.metadata.get("planned_source_type"),
            "preferred_domains": document.metadata.get("preferred_domains", []),
            "freshness": document.metadata.get("freshness"),
            "evidence_requirement_id": document.metadata.get("evidence_requirement_id"),
            "query_contexts": document.metadata.get("query_contexts", []),
            "parent_source_id": document.metadata.get("parent_source_id"),
            "chunk_id": document.metadata.get("chunk_id"),
            "chunk_index": document.metadata.get("chunk_index"),
            "chunk_count": document.metadata.get("chunk_count"),
            "doi": document.metadata.get("doi"),
            "venue": document.metadata.get("venue"),
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
