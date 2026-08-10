"""Download shortlisted papers and persist traceable full-text evidence in Chroma."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import requests
from langchain_core.documents import Document

from .config import config
from .rag_retriever import SharedSentenceTransformerEmbeddings
from .utils import logger


@dataclass(frozen=True)
class PaperChunk:
    """One full-text result returned from the persistent paper library."""

    source_id: str
    title: str
    page: int
    text: str
    distance: float | None = None
    chunk_id: str = ""
    section: str = "Unknown"
    subsection: str = ""
    evidence_type: str = "full_text"
    parser: str = "pypdf"
    schema_version: str = ""


class ChromaPaperLibrary:
    """Maintain a bounded local collection of downloaded, chunked papers."""

    def __init__(
        self,
        embeddings: SharedSentenceTransformerEmbeddings | None = None,
        *,
        enabled: bool | None = None,
        persist_directory: str | Path | None = None,
        pdf_directory: str | Path | None = None,
        client: Any | None = None,
    ) -> None:
        library_config = config.get("paper_library", {})
        self.enabled = bool(library_config.get("enabled", True)) if enabled is None else enabled
        self.persist_directory = Path(persist_directory or library_config.get("persist_directory", "chroma_db"))
        self.pdf_directory = Path(pdf_directory or library_config.get("pdf_directory", ".cache/papers"))
        self.collection_prefix = str(library_config.get("collection_name", "research_papers"))
        self.index_schema_version = str(library_config.get("index_schema_version", "2"))
        self.parser_version = str(library_config.get("parser_version", "pypdf-1"))
        self.chunking_version = str(library_config.get("chunking_version", "page-boundary-2"))
        self.embedding_model = str(config.get("sentence_transformer_model", "default"))
        self.candidate_download_limit = max(
            1,
            int(library_config.get("candidate_download_limit", library_config.get("max_papers_per_run", 3))),
        )
        # Backwards-compatible alias for callers that configured the old name.
        self.max_papers_per_run = self.candidate_download_limit
        self.max_pages_per_paper = max(1, int(library_config.get("max_pages_per_paper", 20)))
        self.max_chunks_per_paper = max(1, int(library_config.get("max_chunks_per_paper", 24)))
        self.chunk_size = max(500, int(library_config.get("chunk_size_chars", 2400)))
        self.chunk_overlap = max(0, int(library_config.get("chunk_overlap_chars", 300)))
        self.chunk_overlap = min(self.chunk_overlap, self.chunk_size - 1)
        self.top_k_chunks = max(1, int(library_config.get("top_k_chunks", 6)))
        self.max_prompt_chars = max(1000, int(library_config.get("max_prompt_chars", 12000)))
        self.max_pdf_bytes = max(1, int(library_config.get("max_pdf_bytes", 25_000_000)))
        self.download_timeout_seconds = max(1, int(library_config.get("download_timeout_seconds", 30)))
        configured_hosts = library_config.get(
            "allowed_pdf_hosts",
            ["arxiv.org", "www.arxiv.org", "export.arxiv.org", "pdfs.semanticscholar.org"],
        )
        self.allowed_pdf_hosts = {str(host).strip().casefold() for host in configured_hosts if str(host).strip()}
        self.embeddings = embeddings or SharedSentenceTransformerEmbeddings()
        self._client = client
        self._vector_store = None

    @property
    def collection_name(self) -> str:
        """Use a distinct Chroma collection for each embedding model."""

        schema_key = "\0".join(
            (
                self.embedding_model,
                self.index_schema_version,
                self.parser_version,
                self.chunking_version,
            )
        )
        model_hash = hashlib.sha256(schema_key.encode("utf-8")).hexdigest()[:12]
        safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.collection_prefix).strip("_-")
        return f"{safe_prefix or 'research_papers'}_{model_hash}"

    def _get_vector_store(self):
        if self._vector_store is not None:
            return self._vector_store

        from langchain_chroma import Chroma

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._vector_store = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_metadata={
                "hnsw:space": "cosine",
                "index_schema_version": self.index_schema_version,
                "parser_version": self.parser_version,
                "chunking_version": self.chunking_version,
            },
        )
        return self._vector_store

    @staticmethod
    def _normalize_queries(queries: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(queries, str):
            values = (queries,)
        else:
            values = tuple(str(query) for query in queries)
        return tuple(dict.fromkeys(query.strip() for query in values if query.strip()))

    def enrich_documents(
        self,
        documents: Sequence[Document],
        queries: str | Sequence[str],
    ) -> list[Document]:
        """Acquire shortlisted PDFs, retrieve passages, and attach provenance.

        Callers must pass only papers that survived the abstract-level candidate
        filter.  This method never expands the shortlist on its own.
        """

        original_documents = list(documents)
        normalized_queries = self._normalize_queries(queries)
        if not self.enabled or not original_documents or not normalized_queries:
            return original_documents

        candidates = [document for document in original_documents if document.metadata.get("pdf_url")][
            : self.candidate_download_limit
        ]
        indexed_source_ids: set[str] = set()
        failed_source_ids: set[str] = set()
        for document in candidates:
            source_id = str(document.metadata.get("source_id", ""))
            try:
                if self.ensure_indexed(document):
                    indexed_source_ids.add(source_id)
                else:
                    failed_source_ids.add(source_id)
            except Exception as exc:
                failed_source_ids.add(source_id)
                logger.warning(
                    "Full-text indexing skipped for %s: %s",
                    source_id or "unknown",
                    exc,
                )

        chunks: list[PaperChunk] = []
        if indexed_source_ids:
            try:
                chunks = self.search_many(
                    normalized_queries,
                    sorted(indexed_source_ids),
                    self.top_k_chunks,
                )
            except Exception as exc:
                logger.warning("Chroma full-text retrieval failed; using abstracts only: %s", exc)

        chunks_by_source: dict[str, list[PaperChunk]] = {}
        used_chars = 0
        for chunk in chunks:
            if used_chars >= self.max_prompt_chars:
                break
            remaining = self.max_prompt_chars - used_chars
            text = chunk.text[:remaining].strip()
            if not text:
                continue
            chunks_by_source.setdefault(chunk.source_id, []).append(
                PaperChunk(
                    chunk.source_id,
                    chunk.title,
                    chunk.page,
                    text,
                    chunk.distance,
                    chunk.chunk_id,
                    chunk.section,
                    chunk.subsection,
                    chunk.evidence_type,
                    chunk.parser,
                    chunk.schema_version,
                )
            )
            used_chars += len(text)

        enriched: list[Document] = []
        for document in original_documents:
            source_id = str(document.metadata.get("source_id", ""))
            source_chunks = chunks_by_source.get(source_id, [])
            metadata = dict(document.metadata)
            metadata["full_text_indexed"] = source_id in indexed_source_ids
            metadata["full_text_chunks_used"] = len(source_chunks)
            if source_id in indexed_source_ids:
                evidence_status = "full_text"
            elif source_id in failed_source_ids:
                evidence_status = "full_text_failed"
            else:
                evidence_status = "abstract_only"
            metadata["evidence_status"] = evidence_status
            metadata["evidence_mode"] = "full_text" if source_chunks else "abstract_only"
            abstract_ref = {
                "source_id": source_id,
                "chunk_id": f"abstract:{source_id}",
                "section": "Abstract",
                "page": None,
                "evidence_type": "abstract_only",
            }
            full_text_refs = [
                {
                    "source_id": chunk.source_id,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "page": chunk.page,
                    "evidence_type": chunk.evidence_type,
                    "parser": chunk.parser,
                    "schema_version": chunk.schema_version,
                    "retrieval_score": chunk.distance,
                    "text": chunk.text,
                }
                for chunk in source_chunks
            ]
            metadata["evidence_refs"] = [abstract_ref, *full_text_refs]
            if source_chunks:
                excerpts = "\n\n".join(
                    (
                        f'<evidence chunk_id="{chunk.chunk_id}" source_id="{chunk.source_id}" '
                        f'section="{chunk.section}" page="{chunk.page}" '
                        f'evidence_type="{chunk.evidence_type}">\n{chunk.text}\n</evidence>'
                    )
                    for chunk in source_chunks
                )
                page_content = f"{document.page_content}\n\n{excerpts}"
            else:
                page_content = document.page_content
            enriched.append(Document(page_content=page_content, metadata=metadata))

        logger.info(
            "Chroma paper library indexed %d source(s) and supplied %d full-text chunk(s).",
            len(indexed_source_ids),
            sum(len(items) for items in chunks_by_source.values()),
        )
        return enriched

    def search_many(
        self,
        queries: Sequence[str],
        source_ids: Sequence[str],
        top_k: int | None = None,
    ) -> list[PaperChunk]:
        """Fuse dense passage rankings from focused evidence queries."""

        fused_scores: dict[str, float] = {}
        chunks_by_id: dict[str, PaperChunk] = {}
        for query in self._normalize_queries(queries):
            for rank, chunk in enumerate(self.search(query, source_ids, top_k), start=1):
                chunk_id = chunk.chunk_id or self._chunk_id(
                    chunk.source_id,
                    chunk.page,
                    rank,
                    chunk.text,
                )
                chunks_by_id.setdefault(
                    chunk_id,
                    chunk
                    if chunk.chunk_id
                    else PaperChunk(
                        chunk.source_id,
                        chunk.title,
                        chunk.page,
                        chunk.text,
                        chunk.distance,
                        chunk_id,
                        chunk.section,
                        chunk.subsection,
                        chunk.evidence_type,
                        chunk.parser,
                        chunk.schema_version or self.index_schema_version,
                    ),
                )
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [chunks_by_id[chunk_id] for chunk_id in ranked_ids[: (top_k or self.top_k_chunks)]]

    def ensure_indexed(self, document: Document) -> bool:
        """Download, extract, embed, and upsert one paper unless already cached."""

        source_id = str(document.metadata.get("source_id", "")).strip()
        pdf_url = str(document.metadata.get("pdf_url", "")).strip()
        if not source_id or not pdf_url:
            return False

        vector_store = self._get_vector_store()
        existing = vector_store.get(where={"source_id": source_id}, limit=1, include=["metadatas"])
        if existing.get("ids"):
            return True

        pdf_path = self._pdf_path(source_id)
        if not pdf_path.exists():
            self._download_pdf(pdf_url, pdf_path)
        pages = self._extract_pages(pdf_path)
        chunks = self._chunk_pages(source_id, document, pages)
        if not chunks:
            logger.warning("No extractable full text found in %s.", source_id)
            return False

        ids = []
        for index, chunk in enumerate(chunks):
            chunk_id = self._chunk_id(source_id, int(chunk.metadata["page"]), index, chunk.page_content)
            chunk.metadata["chunk_id"] = chunk_id
            ids.append(chunk_id)
        vector_store.add_documents(documents=chunks, ids=ids)
        logger.info("Indexed %d full-text chunks for %s in Chroma.", len(chunks), source_id)
        return True

    def search(self, query: str, source_ids: Sequence[str], top_k: int | None = None) -> list[PaperChunk]:
        """Search full-text chunks, restricted to the selected evidence papers."""

        normalized_ids = [source_id for source_id in dict.fromkeys(source_ids) if source_id]
        if not normalized_ids:
            return []

        vector_store = self._get_vector_store()
        where: dict[str, Any]
        if len(normalized_ids) == 1:
            where = {"source_id": normalized_ids[0]}
        else:
            where = {"source_id": {"$in": normalized_ids}}
        results = vector_store.similarity_search_with_score(
            query,
            k=top_k or self.top_k_chunks,
            filter=where,
        )
        chunks: list[PaperChunk] = []
        for document, distance in results:
            metadata = document.metadata
            chunks.append(
                PaperChunk(
                    source_id=str(metadata.get("source_id", "")),
                    title=str(metadata.get("title", "Untitled")),
                    page=int(metadata.get("page", 0)),
                    text=document.page_content,
                    distance=float(distance) if distance is not None else None,
                    chunk_id=str(metadata.get("chunk_id", "")),
                    section=str(metadata.get("section", "Unknown")),
                    subsection=str(metadata.get("subsection", "")),
                    evidence_type=str(metadata.get("evidence_type", "full_text")),
                    parser=str(metadata.get("parser", "pypdf")),
                    schema_version=str(metadata.get("schema_version", self.index_schema_version)),
                )
            )
        return chunks

    def _download_pdf(self, url: str, destination: Path) -> None:
        self._validate_pdf_url(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".part")
        headers = {"User-Agent": "Open-AI-Co-Scientist/1.0 (research paper retrieval)"}
        with requests.get(
            url,
            headers=headers,
            stream=True,
            allow_redirects=True,
            timeout=(5, self.download_timeout_seconds),
        ) as response:
            response.raise_for_status()
            self._validate_pdf_url(response.url)
            declared_size = int(response.headers.get("content-length", 0) or 0)
            if declared_size > self.max_pdf_bytes:
                raise ValueError(f"PDF exceeds the {self.max_pdf_bytes}-byte download limit.")

            downloaded = 0
            first_bytes = b""
            try:
                with temporary.open("wb") as output:
                    for block in response.iter_content(chunk_size=64 * 1024):
                        if not block:
                            continue
                        if not first_bytes:
                            first_bytes = block[:5]
                        downloaded += len(block)
                        if downloaded > self.max_pdf_bytes:
                            raise ValueError(f"PDF exceeds the {self.max_pdf_bytes}-byte download limit.")
                        output.write(block)
                if first_bytes != b"%PDF-":
                    raise ValueError("Downloaded content is not a PDF.")
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def _validate_pdf_url(self, url: str) -> None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme not in {"http", "https"} or host not in self.allowed_pdf_hosts:
            raise ValueError(f"PDF host is not allowed: {host or 'missing host'}")

    def _extract_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages: list[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages[: self.max_pages_per_paper], start=1):
            raw_text = (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.split("\n")]
            text = "\n".join(lines)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append((page_number, text))
        return pages

    def _chunk_pages(
        self,
        source_id: str,
        document: Document,
        pages: Sequence[tuple[int, str]],
    ) -> list[Document]:
        title = str(document.metadata.get("title", "Untitled"))
        pdf_url = str(document.metadata.get("pdf_url", ""))
        page_documents = [
            Document(
                page_content=text,
                metadata={
                    "source_id": source_id,
                    "title": title,
                    "page": page,
                    "pdf_url": pdf_url,
                    "embedding_model": self.embedding_model,
                    "section": "Unknown",
                    "subsection": "",
                    "evidence_type": "full_text",
                    "parser": "pypdf",
                    "parser_version": self.parser_version,
                    "chunking_version": self.chunking_version,
                    "schema_version": self.index_schema_version,
                },
            )
            for page, text in pages
        ]
        chunks: list[Document] = []
        for page_document in page_documents:
            for text in self._split_text(page_document.page_content):
                chunks.append(Document(page_content=text, metadata=dict(page_document.metadata)))
                if len(chunks) >= self.max_chunks_per_paper:
                    return chunks
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split on paragraph, line, sentence, then word boundaries when possible."""

        chunks: list[str] = []
        start = 0
        minimum_boundary = max(1, self.chunk_size // 2)
        while start < len(text):
            hard_end = min(start + self.chunk_size, len(text))
            end = hard_end
            if hard_end < len(text):
                boundary_floor = start + minimum_boundary
                candidates = [
                    text.rfind(separator, boundary_floor, hard_end) for separator in ("\n\n", "\n", ". ", " ")
                ]
                best_boundary = max(candidates)
                if best_boundary >= boundary_floor:
                    end = best_boundary + (2 if text.startswith(("\n\n", ". "), best_boundary) else 1)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return chunks

    def _pdf_path(self, source_id: str) -> Path:
        digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
        return self.pdf_directory / f"{digest}.pdf"

    @staticmethod
    def _chunk_id(source_id: str, page: int, index: int, text: str) -> str:
        digest = hashlib.sha256(f"{source_id}\0{page}\0{index}\0{text}".encode("utf-8")).hexdigest()
        return digest
