"""Download shortlisted papers and persist traceable full-text evidence in Chroma."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
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
    requirement_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedPaper:
    """Parsed pages plus an honest record of parser-side truncation."""

    pages: tuple[tuple[int, str], ...]
    total_pages: int
    truncated: bool


@dataclass(frozen=True)
class ChunkedPaper:
    """Bounded chunks plus an honest record of chunker-side truncation."""

    chunks: tuple[Document, ...]
    truncated: bool


@dataclass(frozen=True)
class IndexIntegrityReport:
    """Source-level comparison between the manifest and Chroma records."""

    source_id: str
    status: str
    expected_count: int
    actual_count: int
    missing_ids: tuple[str, ...] = ()
    stale_ids: tuple[str, ...] = ()
    content_hash_mismatches: tuple[str, ...] = ()
    metadata_mismatches: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def records_valid(self) -> bool:
        return not (
            self.missing_ids
            or self.stale_ids
            or self.content_hash_mismatches
            or self.metadata_mismatches
            or self.expected_count != self.actual_count
        )

    @property
    def ok(self) -> bool:
        return self.status == "COMMITTED" and not self.truncated and self.records_valid


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
        self.require_indexed_sources_for_generation = bool(
            library_config.get("require_indexed_sources_for_generation", False)
        )
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
        self.per_requirement_acquisition_limit = max(
            1,
            int(library_config.get("per_requirement_acquisition_limit", 2)),
        )
        self.max_pages_per_paper = max(1, int(library_config.get("max_pages_per_paper", 20)))
        self.max_chunks_per_paper = max(1, int(library_config.get("max_chunks_per_paper", 24)))
        self.chunk_size = max(500, int(library_config.get("chunk_size_chars", 2400)))
        self.chunk_overlap = max(0, int(library_config.get("chunk_overlap_chars", 300)))
        self.chunk_overlap = min(self.chunk_overlap, self.chunk_size - 1)
        self.top_k_chunks = max(1, int(library_config.get("top_k_chunks", 6)))
        self.max_prompt_chars = max(1000, int(library_config.get("max_prompt_chars", 12000)))
        self.max_pdf_bytes = max(1, int(library_config.get("max_pdf_bytes", 25_000_000)))
        self.download_timeout_seconds = max(1, int(library_config.get("download_timeout_seconds", 30)))
        self.retrieval_workers = max(1, int(library_config.get("retrieval_workers", 3)))
        configured_hosts = library_config.get(
            "allowed_pdf_hosts",
            ["arxiv.org", "www.arxiv.org", "export.arxiv.org", "pdfs.semanticscholar.org"],
        )
        self.allowed_pdf_hosts = {str(host).strip().casefold() for host in configured_hosts if str(host).strip()}
        self.embeddings = embeddings or SharedSentenceTransformerEmbeddings()
        self._client = client
        self._vector_store = None
        self._manifest_lock = RLock()
        self.last_evidence_diagnostics: list[dict[str, Any]] = []
        self._diagnostics_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        self._attempted_source_ids: set[str] = set()

    def begin_run(self) -> None:
        """Reset bounded acquisition state and diagnostics for one generation run."""

        self.last_evidence_diagnostics = []
        self._diagnostics_by_key = {}
        self._attempted_source_ids = set()

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

    @property
    def manifest_path(self) -> Path:
        """Keep the source-integrity ledger beside its Chroma collection."""

        return self.persist_directory / f"{self.collection_name}.manifest.json"

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
    def _empty_manifest() -> dict[str, Any]:
        return {"manifest_version": 1, "sources": {}}

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return self._empty_manifest()
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("sources"), dict):
                raise ValueError("manifest must contain a sources object")
            return payload
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid paper index manifest %s: %s", self.manifest_path, exc)
            return self._empty_manifest()

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def _manifest_source(self, source_id: str) -> dict[str, Any] | None:
        with self._manifest_lock:
            record = self._read_manifest()["sources"].get(source_id)
        return dict(record) if isinstance(record, dict) else None

    def _set_manifest_source(self, source_id: str, record: dict[str, Any]) -> None:
        with self._manifest_lock:
            manifest = self._read_manifest()
            manifest["sources"][source_id] = record
            self._write_manifest(manifest)

    def _set_manifest_status(self, source_id: str, status: str, failure_reason: str = "") -> None:
        with self._manifest_lock:
            manifest = self._read_manifest()
            record = manifest["sources"].get(source_id)
            if not isinstance(record, dict):
                return
            record["status"] = status
            record["failure_reason"] = failure_reason
            manifest["sources"][source_id] = record
            self._write_manifest(manifest)

    def get_index_status(self, source_id: str) -> str:
        """Return the durable source state without treating PARTIAL as complete."""

        record = self._manifest_source(source_id.strip())
        return str(record.get("status", "MISSING")) if record else "MISSING"

    def _current_index_signature(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "schema_version": self.index_schema_version,
            "parser_version": self.parser_version,
            "chunking_version": self.chunking_version,
            "max_pages_per_paper": self.max_pages_per_paper,
            "max_chunks_per_paper": self.max_chunks_per_paper,
        }

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _critical_chunk_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: metadata.get(key)
            for key in (
                "source_id",
                "chunk_id",
                "chunk_index",
                "chunk_count",
                "content_sha256",
                "page",
                "schema_version",
                "parser_version",
                "chunking_version",
                "index_completeness",
            )
        }

    def _stored_source_records(self, source_id: str) -> dict[str, tuple[str, dict[str, Any]]]:
        result = self._get_vector_store().get(
            where={"source_id": source_id},
            include=["documents", "metadatas"],
        )
        ids = list(result.get("ids") or [])
        documents = list(result.get("documents") or [])
        metadatas = list(result.get("metadatas") or [])
        records: dict[str, tuple[str, dict[str, Any]]] = {}
        for index, chunk_id in enumerate(ids):
            text = documents[index] if index < len(documents) and documents[index] is not None else ""
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            records[str(chunk_id)] = (str(text), dict(metadata))
        return records

    def verify_indexed_source(self, source_id: str) -> IndexIntegrityReport:
        """Compare all stored records with the durable expected source manifest."""

        normalized_source_id = source_id.strip()
        if not normalized_source_id:
            return IndexIntegrityReport("", "MISSING", 0, 0)

        record = self._manifest_source(normalized_source_id)
        actual_records = self._stored_source_records(normalized_source_id)
        if record is None:
            return IndexIntegrityReport(
                normalized_source_id,
                "MISSING",
                0,
                len(actual_records),
                stale_ids=tuple(sorted(actual_records)),
            )

        expected_records = record.get("chunks")
        if not isinstance(expected_records, dict):
            expected_records = {}
        expected_ids = set(expected_records)
        actual_ids = set(actual_records)
        missing_ids = tuple(sorted(expected_ids - actual_ids))
        stale_ids = tuple(sorted(actual_ids - expected_ids))
        content_mismatches: list[str] = []
        metadata_mismatches: list[str] = []

        for chunk_id in sorted(expected_ids & actual_ids):
            expected = expected_records.get(chunk_id)
            if not isinstance(expected, dict):
                metadata_mismatches.append(chunk_id)
                continue
            actual_text, actual_metadata = actual_records[chunk_id]
            expected_hash = str(expected.get("content_sha256", ""))
            if not expected_hash or self._content_hash(actual_text) != expected_hash:
                content_mismatches.append(chunk_id)
            expected_metadata = expected.get("metadata")
            if (
                not isinstance(expected_metadata, dict)
                or self._critical_chunk_metadata(actual_metadata) != expected_metadata
            ):
                metadata_mismatches.append(chunk_id)

        signature = record.get("index_signature")
        if signature != self._current_index_signature():
            metadata_mismatches.append("__index_signature__")
        if record.get("expected_chunk_count") != len(expected_ids):
            metadata_mismatches.append("__expected_chunk_count__")
        expected_ids_hash = self._content_hash("\n".join(sorted(expected_ids)))
        if record.get("expected_chunk_ids_sha256") != expected_ids_hash:
            metadata_mismatches.append("__expected_chunk_ids_sha256__")
        pdf_path = self._pdf_path(normalized_source_id)
        if pdf_path.exists() and record.get("document_sha256") != self._file_hash(pdf_path):
            content_mismatches.append("__document_sha256__")

        return IndexIntegrityReport(
            source_id=normalized_source_id,
            status=str(record.get("status", "MISSING")),
            expected_count=len(expected_ids),
            actual_count=len(actual_ids),
            missing_ids=missing_ids,
            stale_ids=stale_ids,
            content_hash_mismatches=tuple(content_mismatches),
            metadata_mismatches=tuple(metadata_mismatches),
            truncated=bool(record.get("truncated", False)),
        )

    @staticmethod
    def _normalize_queries(queries: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(queries, str):
            values = (queries,)
        else:
            values = tuple(str(query) for query in queries)
        return tuple(dict.fromkeys(query.strip() for query in values if query.strip()))

    @staticmethod
    def _normalize_query_specs(queries: str | Sequence[Any]) -> tuple[tuple[str, str], ...]:
        """Keep optional requirement identity attached to full-text queries."""

        values = (queries,) if isinstance(queries, str) else tuple(queries)
        specs: list[tuple[str, str]] = []
        for query in values:
            text = str(getattr(query, "query", query)).strip()
            requirement_id = str(getattr(query, "evidence_requirement_id", "") or "").strip()
            if text and (text, requirement_id) not in specs:
                specs.append((text, requirement_id))
        return tuple(specs)

    @staticmethod
    def _document_requirement_ids(document: Document) -> tuple[str, ...]:
        requirement_ids: list[str] = []
        for value in document.metadata.get("reserved_requirement_ids", ()):
            normalized = str(value or "").strip()
            if normalized and normalized not in requirement_ids:
                requirement_ids.append(normalized)
        direct = str(document.metadata.get("evidence_requirement_id") or "").strip()
        if direct and direct not in requirement_ids:
            requirement_ids.append(direct)
        for context in document.metadata.get("query_contexts", ()):
            if not isinstance(context, dict):
                continue
            normalized = str(context.get("evidence_requirement_id") or "").strip()
            if normalized and normalized not in requirement_ids:
                requirement_ids.append(normalized)
        return tuple(requirement_ids)

    def _record_evidence_diagnostic(
        self,
        document: Document,
        requirement_id: str,
        *,
        query: str = "",
        candidate_rank: int | None = None,
        pdf_eligible: bool | None = None,
        acquisition_attempted: bool | None = None,
        acquisition_result: str | None = None,
        selected_chunk_ids: Sequence[str] | None = None,
    ) -> None:
        source_id = str(document.metadata.get("source_id") or "")
        key = (source_id, requirement_id)
        diagnostic = self._diagnostics_by_key.get(
            key,
            {
                "requirement_id": requirement_id or None,
                "query": query or document.metadata.get("retrieval_query") or document.metadata.get("search_query"),
                "provider": document.metadata.get("provider") or document.metadata.get("source"),
                "raw_result_count": document.metadata.get("raw_result_count"),
                "candidate_source_id": source_id,
                "candidate_rank": candidate_rank,
                "reserved_for_requirement": bool(requirement_id),
                "pdf_eligible": bool(document.metadata.get("pdf_url")),
                "acquisition_attempted": False,
                "acquisition_result": "not_attempted",
                "index_status": self.get_index_status(source_id) if source_id else "MISSING",
                "full_text_chunk_count": 0,
                "selected_chunk_ids": [],
                "strict_gate_retained": False,
                "strict_gate_rejection_reason": "not_evaluated",
                "coverage_contribution": False,
            },
        )
        if query:
            diagnostic["query"] = query
        if candidate_rank is not None:
            diagnostic["candidate_rank"] = candidate_rank
        if pdf_eligible is not None:
            diagnostic["pdf_eligible"] = pdf_eligible
        if acquisition_attempted is not None:
            diagnostic["acquisition_attempted"] = acquisition_attempted
        if acquisition_result is not None:
            diagnostic["acquisition_result"] = acquisition_result
        if selected_chunk_ids is not None:
            diagnostic["selected_chunk_ids"] = list(dict.fromkeys(selected_chunk_ids))
        if source_id:
            status = self.get_index_status(source_id)
            diagnostic["index_status"] = status
            if status != "MISSING":
                diagnostic["full_text_chunk_count"] = self.verify_indexed_source(source_id).actual_count
        self._diagnostics_by_key[key] = diagnostic
        self.last_evidence_diagnostics = list(self._diagnostics_by_key.values())

    def record_strict_gate(self, source_id: str, *, retained: bool, reason: str) -> None:
        """Attach strict-gate decisions to every diagnostic lane for a source."""

        for (candidate_source_id, _requirement_id), diagnostic in self._diagnostics_by_key.items():
            if candidate_source_id == source_id:
                diagnostic["strict_gate_retained"] = retained
                diagnostic["strict_gate_rejection_reason"] = reason
        self.last_evidence_diagnostics = list(self._diagnostics_by_key.values())

    def record_coverage(self, aspect_source_ids: dict[str, Sequence[str]]) -> None:
        """Persist which strict evidence records contributed to requirement coverage."""

        for (source_id, requirement_id), diagnostic in self._diagnostics_by_key.items():
            if requirement_id:
                contributes = source_id in set(aspect_source_ids.get(requirement_id, ()))
            else:
                contributes = any(source_id in set(source_ids) for source_ids in aspect_source_ids.values())
            diagnostic["coverage_contribution"] = contributes
        self.last_evidence_diagnostics = list(self._diagnostics_by_key.values())

    def enrich_documents(
        self,
        documents: Sequence[Document],
        queries: str | Sequence[Any],
    ) -> list[Document]:
        """Acquire shortlisted PDFs, retrieve passages, and attach provenance.

        Callers must pass only papers that survived the abstract-level candidate
        filter.  This method never expands the shortlist on its own.
        """

        original_documents = list(documents)
        query_specs = self._normalize_query_specs(queries)
        if not self.enabled or not original_documents or not query_specs:
            return original_documents

        query_by_requirement = {requirement_id: text for text, requirement_id in query_specs if requirement_id}
        requirement_order = list(query_by_requirement)
        lanes: dict[str, list[Document]] = {requirement_id: [] for requirement_id in requirement_order}
        lanes["__unscoped__"] = []
        for document in original_documents:
            requirement_ids = self._document_requirement_ids(document)
            if requirement_ids:
                for requirement_id in requirement_ids:
                    lanes.setdefault(requirement_id, []).append(document)
                    if requirement_id not in requirement_order:
                        requirement_order.append(requirement_id)
            else:
                lanes["__unscoped__"].append(document)

        for requirement_id, lane_documents in lanes.items():
            for rank, document in enumerate(lane_documents, start=1):
                self._record_evidence_diagnostic(
                    document,
                    "" if requirement_id == "__unscoped__" else requirement_id,
                    query=query_by_requirement.get(requirement_id, ""),
                    candidate_rank=rank,
                    pdf_eligible=bool(document.metadata.get("pdf_url")),
                )

        indexed_source_ids: set[str] = set()
        partial_source_ids: set[str] = set()
        failed_source_ids: set[str] = set()
        acquisition_results: dict[str, str] = {}

        # Every already COMMITTED source is eligible without consuming an
        # acquisition attempt, regardless of its position in the accumulated list.
        for document in original_documents:
            source_id = str(document.metadata.get("source_id", ""))
            status = self.get_index_status(source_id) if source_id else "MISSING"
            if status == "PARTIAL":
                partial_source_ids.add(source_id)
                acquisition_results[source_id] = "partial"
            elif status == "FAILED":
                failed_source_ids.add(source_id)
                acquisition_results[source_id] = "failed"
            if source_id and self.has_indexed_source(source_id):
                indexed_source_ids.add(source_id)
                acquisition_results[source_id] = (
                    "committed" if source_id in self._attempted_source_ids else "cached_committed"
                )

        lane_positions = {requirement_id: 0 for requirement_id in lanes}
        lane_attempts = {requirement_id: 0 for requirement_id in lanes}
        global_attempts = 0

        def attempt_one(requirement_id: str) -> bool:
            """Make at most one new acquisition attempt in one requirement lane."""

            nonlocal global_attempts
            lane = lanes.get(requirement_id, [])
            lane_budget = (
                self.candidate_download_limit
                if requirement_id == "__unscoped__"
                else self.per_requirement_acquisition_limit
            )
            while (
                lane_positions[requirement_id] < len(lane)
                and lane_attempts[requirement_id] < lane_budget
                and global_attempts < self.candidate_download_limit
            ):
                document = lane[lane_positions[requirement_id]]
                lane_positions[requirement_id] += 1
                source_id = str(document.metadata.get("source_id", ""))
                public_requirement_id = "" if requirement_id == "__unscoped__" else requirement_id
                if not source_id or not document.metadata.get("pdf_url"):
                    acquisition_results[source_id] = "not_pdf_eligible"
                    continue
                if source_id in indexed_source_ids:
                    acquisition_results.setdefault(source_id, "cached_committed")
                    return True
                status = self.get_index_status(source_id)
                if status == "PARTIAL":
                    partial_source_ids.add(source_id)
                    acquisition_results[source_id] = "partial"
                    continue
                if source_id in self._attempted_source_ids:
                    acquisition_results.setdefault(source_id, "already_attempted")
                    continue

                lane_attempts[requirement_id] += 1
                global_attempts += 1
                self._attempted_source_ids.add(source_id)
                try:
                    if self.ensure_indexed(document):
                        indexed_source_ids.add(source_id)
                        acquisition_results[source_id] = "committed"
                        self._record_evidence_diagnostic(
                            document,
                            public_requirement_id,
                            query=query_by_requirement.get(requirement_id, ""),
                            acquisition_attempted=True,
                            acquisition_result="committed",
                        )
                        return True
                    if self.get_index_status(source_id) == "PARTIAL":
                        partial_source_ids.add(source_id)
                        result = "partial"
                    else:
                        failed_source_ids.add(source_id)
                        result = "failed"
                    acquisition_results[source_id] = result
                    self._record_evidence_diagnostic(
                        document,
                        public_requirement_id,
                        query=query_by_requirement.get(requirement_id, ""),
                        acquisition_attempted=True,
                        acquisition_result=result,
                    )
                    return False
                except Exception as exc:
                    failed_source_ids.add(source_id)
                    acquisition_results[source_id] = "failed"
                    self._record_evidence_diagnostic(
                        document,
                        public_requirement_id,
                        query=query_by_requirement.get(requirement_id, ""),
                        acquisition_attempted=True,
                        acquisition_result="failed",
                    )
                    logger.warning(
                        "Full-text indexing skipped for %s: %s",
                        source_id or "unknown",
                        exc,
                    )
                    return False
            return False

        # Requirement lanes receive the first bounded opportunities. Broad,
        # unscoped sources may use only the remaining global budget.
        lane_has_committed = {requirement_id: attempt_one(requirement_id) for requirement_id in requirement_order}
        # Only after every requirement has received its first opportunity may
        # failures consume a second per-requirement attempt.
        for requirement_id in requirement_order:
            if not lane_has_committed[requirement_id]:
                lane_has_committed[requirement_id] = attempt_one(requirement_id)
        while global_attempts < self.candidate_download_limit:
            previous_position = lane_positions["__unscoped__"]
            attempt_one("__unscoped__")
            if lane_positions["__unscoped__"] == previous_position:
                break

        source_ids_by_requirement = {
            requirement_id: [
                str(document.metadata.get("source_id", ""))
                for document in lanes.get(requirement_id, ())
                if str(document.metadata.get("source_id", "")) in indexed_source_ids
            ]
            for requirement_id in requirement_order
        }

        chunks: list[PaperChunk] = []
        if indexed_source_ids:
            try:
                chunks = self.search_many(
                    queries,
                    sorted(indexed_source_ids),
                    self.top_k_chunks,
                    source_ids_by_requirement=source_ids_by_requirement,
                )
            except Exception as exc:
                logger.warning("Chroma full-text retrieval failed; using abstracts only: %s", exc)

        # A COMMITTED first choice that yields no passage must not end a lane.
        # Spend the remaining per-lane/global budget on ranked failovers, then
        # rerun bounded passage retrieval once.
        covered_requirement_ids = {requirement_id for chunk in chunks for requirement_id in chunk.requirement_ids}
        failover_attempted = False
        for requirement_id in requirement_order:
            if requirement_id not in covered_requirement_ids and attempt_one(requirement_id):
                failover_attempted = True
        if failover_attempted:
            source_ids_by_requirement = {
                requirement_id: [
                    str(document.metadata.get("source_id", ""))
                    for document in lanes.get(requirement_id, ())
                    if str(document.metadata.get("source_id", "")) in indexed_source_ids
                ]
                for requirement_id in requirement_order
            }
            chunks = self.search_many(
                queries,
                sorted(indexed_source_ids),
                self.top_k_chunks,
                source_ids_by_requirement=source_ids_by_requirement,
            )

        chunks_by_source: dict[str, list[PaperChunk]] = {}
        used_chars = 0
        reserved_remaining = sum(bool(chunk.requirement_ids) for chunk in chunks)
        for chunk in chunks:
            if used_chars >= self.max_prompt_chars:
                break
            remaining = self.max_prompt_chars - used_chars
            allocation = remaining
            if chunk.requirement_ids and reserved_remaining:
                allocation = max(1, remaining // reserved_remaining)
                reserved_remaining -= 1
            text = chunk.text[:allocation].strip()
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
                    chunk.requirement_ids,
                )
            )
            used_chars += len(text)

        selected_chunk_ids_by_source: dict[str, list[str]] = {}
        for source_chunks in chunks_by_source.values():
            for chunk in source_chunks:
                selected_chunk_ids_by_source.setdefault(chunk.source_id, []).append(chunk.chunk_id)
        for requirement_id, lane_documents in lanes.items():
            for document in lane_documents:
                source_id = str(document.metadata.get("source_id", ""))
                self._record_evidence_diagnostic(
                    document,
                    "" if requirement_id == "__unscoped__" else requirement_id,
                    query=query_by_requirement.get(requirement_id, ""),
                    acquisition_result=acquisition_results.get(source_id),
                    selected_chunk_ids=selected_chunk_ids_by_source.get(source_id, ()),
                )

        enriched: list[Document] = []
        for document in original_documents:
            source_id = str(document.metadata.get("source_id", ""))
            source_chunks = chunks_by_source.get(source_id, [])
            metadata = dict(document.metadata)
            metadata["full_text_indexed"] = source_id in indexed_source_ids
            metadata["full_text_available"] = bool(metadata.get("content_extracted") or metadata["full_text_indexed"])
            metadata["full_text_chunks_used"] = len(source_chunks)
            metadata["index_status"] = self.get_index_status(source_id)
            metadata["index_truncated"] = source_id in partial_source_ids
            metadata["acquisition_attempted"] = source_id in self._attempted_source_ids
            metadata["acquisition_result"] = acquisition_results.get(source_id, "not_attempted")
            metadata["selected_chunk_ids"] = [chunk.chunk_id for chunk in source_chunks]
            if source_id in indexed_source_ids:
                evidence_status = "full_text"
            elif source_id in partial_source_ids:
                evidence_status = "full_text_partial"
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
                    "requirement_ids": list(chunk.requirement_ids),
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

    def has_indexed_source(self, source_id: str) -> bool:
        """Return True only for an exact, complete, committed source index."""

        normalized_source_id = source_id.strip()
        if not normalized_source_id:
            return False
        return self.verify_indexed_source(normalized_source_id).ok

    def search_many(
        self,
        queries: Sequence[Any],
        source_ids: Sequence[str],
        top_k: int | None = None,
        *,
        source_ids_by_requirement: dict[str, Sequence[str]] | None = None,
    ) -> list[PaperChunk]:
        """Reserve strong per-requirement passages, then fill globally."""

        fused_scores: dict[str, float] = {}
        chunks_by_id: dict[str, PaperChunk] = {}
        query_specs = self._normalize_query_specs(queries)
        if not query_specs or not source_ids:
            return []
        lane_sources = source_ids_by_requirement or {}

        def run_search(spec: tuple[str, str]) -> list[PaperChunk]:
            query, requirement_id = spec
            if requirement_id:
                restricted_ids = list(lane_sources.get(requirement_id, ()))
                return self.search(query, restricted_ids, top_k) if restricted_ids else []
            return self.search(query, source_ids, top_k)

        workers = min(self.retrieval_workers, len(query_specs))
        if workers == 1:
            rankings = [run_search(spec) for spec in query_specs]
        else:
            # Initialize the shared store before concurrent read-only queries.
            self._get_vector_store()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rankings = list(executor.map(run_search, query_specs))
        requirement_scores: dict[str, dict[str, float]] = {}
        # Fuse in query order so thread completion order cannot affect ties.
        for (_query, requirement_id), ranking in zip(query_specs, rankings):
            for rank, chunk in enumerate(ranking, start=1):
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
                if requirement_id:
                    scores = requirement_scores.setdefault(requirement_id, {})
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        limit = top_k or self.top_k_chunks
        selected_ids: list[str] = []
        requirements_by_chunk: dict[str, list[str]] = {}
        for _query, requirement_id in query_specs:
            if not requirement_id or requirement_id in {
                item for values in requirements_by_chunk.values() for item in values
            }:
                continue
            lane_ranked_ids = sorted(
                requirement_scores.get(requirement_id, {}),
                key=requirement_scores.get(requirement_id, {}).get,
                reverse=True,
            )
            if not lane_ranked_ids:
                continue
            winner_id = lane_ranked_ids[0]
            requirements_by_chunk.setdefault(winner_id, []).append(requirement_id)
            if winner_id not in selected_ids and len(selected_ids) < limit:
                selected_ids.append(winner_id)
        for chunk_id in ranked_ids:
            if len(selected_ids) >= limit:
                break
            if chunk_id not in selected_ids:
                selected_ids.append(chunk_id)

        selected: list[PaperChunk] = []
        for chunk_id in selected_ids:
            chunk = chunks_by_id[chunk_id]
            selected.append(
                PaperChunk(
                    chunk.source_id,
                    chunk.title,
                    chunk.page,
                    chunk.text,
                    chunk.distance,
                    chunk.chunk_id,
                    chunk.section,
                    chunk.subsection,
                    chunk.evidence_type,
                    chunk.parser,
                    chunk.schema_version,
                    tuple(requirements_by_chunk.get(chunk_id, ())),
                )
            )
        return selected

    def ensure_indexed(self, document: Document) -> bool:
        """Build, verify, and logically commit one complete paper index."""

        source_id = str(document.metadata.get("source_id", "")).strip()
        pdf_url = str(document.metadata.get("pdf_url", "")).strip()
        if not source_id or not pdf_url:
            return False

        if self.has_indexed_source(source_id):
            return True

        existing_report = self.verify_indexed_source(source_id)
        if existing_report.status == "PARTIAL" and existing_report.records_valid:
            logger.warning(
                "Source %s remains PARTIAL because configured ingestion limits truncated it.",
                source_id,
            )
            return False

        vector_store = self._get_vector_store()

        pdf_path = self._pdf_path(source_id)
        if not pdf_path.exists():
            self._download_pdf(pdf_url, pdf_path)
        extracted = self._extract_pages(pdf_path)
        if isinstance(extracted, ExtractedPaper):
            pages = extracted.pages
            pages_truncated = extracted.truncated
            total_pages = extracted.total_pages
        else:
            # Preserve compatibility with parser test doubles and custom parsers.
            pages = tuple(extracted)
            pages_truncated = False
            total_pages = len(pages)

        chunked = self._chunk_pages(source_id, document, pages)
        if isinstance(chunked, ChunkedPaper):
            chunks = list(chunked.chunks)
            chunks_truncated = chunked.truncated
        else:
            # Preserve compatibility with custom chunkers returning a plain list.
            chunks = list(chunked)
            chunks_truncated = False
        if not chunks:
            logger.warning("No extractable full text found in %s.", source_id)
            return False

        truncated = pages_truncated or chunks_truncated
        index_completeness = "partial" if truncated else "complete"
        ids: list[str] = []
        for index, chunk in enumerate(chunks):
            chunk_id = self._chunk_id(source_id, int(chunk.metadata["page"]), index, chunk.page_content)
            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "content_sha256": self._content_hash(chunk.page_content),
                    "index_completeness": index_completeness,
                }
            )
            ids.append(chunk_id)

        expected_chunks = {
            chunk_id: {
                "content_sha256": str(chunk.metadata["content_sha256"]),
                "metadata": self._critical_chunk_metadata(chunk.metadata),
            }
            for chunk_id, chunk in zip(ids, chunks)
        }
        truncation_reasons = []
        if pages_truncated:
            truncation_reasons.append(f"page limit {self.max_pages_per_paper} of {total_pages} pages")
        if chunks_truncated:
            truncation_reasons.append(f"chunk limit {self.max_chunks_per_paper}")
        manifest_record = {
            "status": "INDEXING",
            "failure_reason": "",
            "document_sha256": self._file_hash(pdf_path),
            "expected_chunk_count": len(ids),
            "expected_chunk_ids_sha256": self._content_hash("\n".join(sorted(ids))),
            "truncated": truncated,
            "truncation_reasons": truncation_reasons,
            "index_signature": self._current_index_signature(),
            "chunks": expected_chunks,
        }
        self._set_manifest_source(source_id, manifest_record)

        try:
            vector_store.add_documents(documents=chunks, ids=ids)
            report = self.verify_indexed_source(source_id)
            if report.missing_ids or report.content_hash_mismatches or report.metadata_mismatches:
                raise ValueError(
                    "Chroma read-after-write verification failed "
                    f"(missing={len(report.missing_ids)}, "
                    f"content_mismatches={len(report.content_hash_mismatches)}, "
                    f"metadata_mismatches={len(report.metadata_mismatches)})."
                )
            if report.stale_ids:
                vector_store.delete(ids=list(report.stale_ids))
                report = self.verify_indexed_source(source_id)
            if not report.records_valid:
                raise ValueError(
                    "Chroma source verification failed after stale-record repair "
                    f"(expected={report.expected_count}, actual={report.actual_count})."
                )

            if truncated:
                reason = "; ".join(truncation_reasons)
                self._set_manifest_status(source_id, "PARTIAL", reason)
                logger.warning(
                    "Indexed %d verified chunks for %s, but the source is PARTIAL: %s.",
                    len(chunks),
                    source_id,
                    reason,
                )
                return False

            self._set_manifest_status(source_id, "COMMITTED")
            committed_report = self.verify_indexed_source(source_id)
            if not committed_report.ok:
                self._set_manifest_status(source_id, "FAILED", "post-commit verification failed")
                return False
        except Exception as exc:
            self._set_manifest_status(source_id, "FAILED", str(exc))
            raise

        logger.info("Indexed and verified %d full-text chunks for %s in Chroma.", len(chunks), source_id)
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

    def _extract_pages(self, pdf_path: Path) -> ExtractedPaper:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages: list[tuple[int, str]] = []
        total_pages = len(reader.pages)
        for page_number, page in enumerate(reader.pages[: self.max_pages_per_paper], start=1):
            raw_text = (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.split("\n")]
            text = "\n".join(lines)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append((page_number, text))
        return ExtractedPaper(
            pages=tuple(pages),
            total_pages=total_pages,
            truncated=total_pages > self.max_pages_per_paper,
        )

    def _chunk_pages(
        self,
        source_id: str,
        document: Document,
        pages: Sequence[tuple[int, str]],
    ) -> ChunkedPaper:
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
        all_chunks: list[Document] = []
        for page_document in page_documents:
            for text in self._split_text(page_document.page_content):
                all_chunks.append(Document(page_content=text, metadata=dict(page_document.metadata)))
        return ChunkedPaper(
            chunks=tuple(all_chunks[: self.max_chunks_per_paper]),
            truncated=len(all_chunks) > self.max_chunks_per_paper,
        )

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
