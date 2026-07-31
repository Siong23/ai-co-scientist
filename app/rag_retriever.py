from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .config import config
from .tools.arxiv_search import ArxivSearchTool
from .utils import get_sentence_transformer_model, logger


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
    """Fuse arXiv result rankings and deduplicate papers by canonical ID."""

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
        self.results_per_query = results_per_query or int(rag_config.get("results_per_query", 6))
        self.top_k = top_k or int(rag_config.get("top_k", 4))
        self.minimum_relevant_sources = minimum_relevant_sources or int(
            rag_config.get("minimum_relevant_sources", 3)
        )
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
        self.max_abstract_chars = max_abstract_chars or int(rag_config.get("max_abstract_chars", 1800))

        self.arxiv = ArxivSearchTool(max_results=self.results_per_query)
        self.embeddings = SharedSentenceTransformerEmbeddings()

    def retrieve(
        self,
        original_query: str,
        query_plan: SearchQueryPlan,
    ) -> list[Document]:
        """Retrieve, filter, fuse, and rerank arXiv abstracts."""

        original_query = original_query.strip()
        if not original_query:
            return []

        ranked_results = [
            self.arxiv.search_papers(
                query=query,
                max_results=self.results_per_query,
                sort_by="relevance",
            )
            for query in query_plan.queries
        ]
        fused_papers = reciprocal_rank_fusion(
            ranked_results,
            k=self.rrf_k,
        )
        relevant_papers = [
            paper
            for paper in fused_papers
            if self._contains_required_term(
                paper,
                query_plan.required_terms,
            )
        ]

        if not relevant_papers:
            logger.warning(
                "RAG retrieval found no papers containing required terms %s.",
                query_plan.required_terms,
            )
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
        selected = vector_store.similarity_search(
            original_query,
            k=min(self.top_k, len(documents)),
        )
        logger.info(
            "RAG selected %d sources from %d entity-matched candidates: %s",
            len(selected),
            len(documents),
            [document.metadata.get("source_id") for document in selected],
        )
        return selected

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

    def _paper_to_document(
        self,
        paper: dict[str, Any],
    ) -> Document:
        arxiv_id = str(paper["arxiv_id"])
        source_id = f"arXiv:{arxiv_id}"
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
            "rrf_score": document.metadata.get("rrf_score"),
        }
        for document in documents
    ]
