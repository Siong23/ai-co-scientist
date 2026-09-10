"""Durable, resumable research-session state.

This module deliberately persists only research interpretation and orchestration
state.  Source documents, source chunks, and embeddings belong to the global
evidence layer and are represented here only by compact provenance identifiers.
Run/report snapshots remain the responsibility of :mod:`app.run_store`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from .models import ContextMemory, Hypothesis, ReflectionReport, ResearchGoal
from .research_modes import RESEARCH_TYPES
from .run_store import get_results_dir, sanitize

RESEARCH_STATE_SCHEMA = "open-ai-co-scientist.research-state"
RESEARCH_STATE_SCHEMA_VERSION = 1
RESEARCH_STATE_DIRECTORY = "research_state"

_SAFE_RESEARCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_INTEGRITY_ALGORITHM = "sha256"

_RESUME_LIMITATIONS = (
    "In-flight tasks are suspended on load and are not restarted automatically.",
    "Source documents, chunks, and embeddings are not stored here; evidence must be resolved from the global evidence store.",
)

# Fields in a source/chunk-shaped mapping which contain evidence bodies rather
# than durable provenance.  ``text`` is intentionally scoped to source-shaped
# mappings: hypothesis and review prose are valid research state.
_SOURCE_BODY_FIELDS = {
    "abstract",
    "body",
    "chunk_text",
    "content",
    "content_extracted",
    "document_text",
    "evidence_span",
    "evidence_spans",
    "full_text",
    "fulltext",
    "page_content",
    "passage",
    "raw_text",
    "snippet",
    "summary",
    "text",
}
_BULK_EVIDENCE_FIELDS = {
    "candidate_context",
    "chunks",
    "documents",
    "full_text_context",
    "last_retrieved_sources",
    "retrieved_context",
    "retrieved_documents",
    "source_context",
    "source_documents",
}
_SOURCE_CONTAINER_FIELDS = {
    "contradictory_evidence",
    "evidence_refs",
    "evidence_sources",
    "references",
    "sources",
    "supporting_evidence",
}
_PROVENANCE_FIELDS = {
    "abstract_acquisition_reason",
    "acquisition_result",
    "arxiv_id",
    "arxiv_url",
    "author",
    "authors",
    "canonical_url",
    "chunk_count",
    "chunk_id",
    "chunk_ids",
    "chunk_index",
    "context_chunk_ids",
    "doi",
    "document_type",
    "domain",
    "evidence_mode",
    "evidence_ref",
    "evidence_refs",
    "evidence_requirement_id",
    "evidence_status",
    "evidence_strength",
    "evidence_type",
    "expanded_chunk_ids",
    "freshness",
    "full_text_available",
    "full_text_cache_hit",
    "full_text_chunks_used",
    "full_text_indexed",
    "id",
    "index_status",
    "index_truncated",
    "language",
    "page",
    "page_type",
    "paper_id",
    "paper_version_id",
    "parent_source_id",
    "pdf_url",
    "preferred_domains",
    "provider",
    "published",
    "published_at",
    "purpose",
    "question_id",
    "relation",
    "reserved_requirement_ids",
    "retrieval_query",
    "section",
    "selected_chunk_ids",
    "source",
    "source_authority",
    "source_family",
    "source_id",
    "source_ids",
    "source_type",
    "source_version_id",
    "sub_question",
    "title",
    "updated_at",
    "url",
    "venue",
}
_CHUNK_LIST_FIELDS = (
    "chunk_ids",
    "selected_chunk_ids",
    "expanded_chunk_ids",
    "context_chunk_ids",
)


class ResearchStateError(Exception):
    """Base class for durable research-state failures."""


class ResearchStateNotFoundError(ResearchStateError, FileNotFoundError):
    """Raised when no state exists for a research ID."""


class ResearchStateIdentifierError(ResearchStateError, ValueError):
    """Raised when a research ID is missing, unsafe, or inconsistent."""


class ResearchStateSchemaError(ResearchStateError, ValueError):
    """Raised when a state document is malformed or uses an unknown schema."""


class ResearchStateCompatibilityError(ResearchStateSchemaError):
    """Raised when the reader is outside the document's compatibility range."""


class ResearchStateIntegrityError(ResearchStateError, ValueError):
    """Raised when a state document's content digest does not match."""


class ResearchStateSerializationError(ResearchStateError, ValueError):
    """Raised when current in-memory state cannot be serialized safely."""


@dataclass(frozen=True)
class ResumedResearchSession:
    """A reconstructed session plus the metadata needed for an honest resume."""

    research_goal: ResearchGoal
    context: ContextMemory
    research_id: str
    schema_version: int
    created_at: str
    updated_at: str
    compatibility: dict[str, Any]
    limitations: tuple[str, ...]
    source_path: Path

    @property
    def suspended_pending_tasks(self) -> tuple[Any, ...]:
        """Tasks recorded as pending at save time, now suspended for review."""

        resume_state = getattr(self.context, "resume_state", {})
        pending = resume_state.get("suspended_pending_tasks", []) if isinstance(resume_state, dict) else []
        return tuple(pending) if isinstance(pending, list) else ()


@runtime_checkable
class ResearchStateStore(Protocol):
    """Storage boundary for one mutable research session."""

    def save(self, research_goal: ResearchGoal, context: ContextMemory) -> Path:
        """Atomically persist the current research session and return its path."""

    def load(self, research_id: str) -> ResumedResearchSession:
        """Validate and reconstruct a saved research session."""

    def exists(self, research_id: str) -> bool:
        """Return whether state exists for ``research_id``."""


class LocalJSONResearchStateStore:
    """Atomic schema-versioned JSON implementation of :class:`ResearchStateStore`.

    ``root=None`` is resolved at operation time.  This both follows the same
    ``CO_SCIENTIST_RUNS_DIR`` behavior as run snapshots and keeps tests which
    patch that environment variable isolated after module import.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        root: str | Path | None = None,
    ):
        if root_dir is not None and root is not None:
            raise ValueError("Pass either root_dir or root, not both.")
        configured = root_dir if root_dir is not None else root
        self._configured_root = Path(configured) if configured is not None else None

    @property
    def root(self) -> Path:
        if self._configured_root is not None:
            return self._configured_root
        return get_results_dir() / RESEARCH_STATE_DIRECTORY

    def path_for(self, research_id: str) -> Path:
        return self.root / f"{_validate_research_id(research_id)}.json"

    def exists(self, research_id: str) -> bool:
        return self.path_for(research_id).is_file()

    def save(self, research_goal: ResearchGoal, context: ContextMemory) -> Path:
        research_id = _resolve_research_id(research_goal, context)
        path = self.path_for(research_id)
        now = _utc_now()
        created_at = now

        if path.exists():
            current = _read_and_validate(path)
            if current["research_id"] != research_id:
                raise ResearchStateIdentifierError(
                    f"State path for {research_id!r} contains research ID {current['research_id']!r}."
                )
            created_at = current["created_at"]

        document = _build_document(
            research_goal,
            context,
            research_id=research_id,
            created_at=created_at,
            updated_at=now,
        )
        _atomic_write_json(path, document)
        return path

    def load(self, research_id: str) -> ResumedResearchSession:
        safe_research_id = _validate_research_id(research_id)
        path = self.path_for(safe_research_id)
        document = _read_and_validate(path)
        if document["research_id"] != safe_research_id:
            raise ResearchStateIdentifierError(
                f"Requested research ID {safe_research_id!r} does not match saved ID {document['research_id']!r}."
            )

        research_goal = _restore_research_goal(document["research_goal"], safe_research_id)
        context, limitations = _restore_context(
            document["research_state"],
            research_id=safe_research_id,
            research_type=str(document.get("research_type") or "hypothesis_testing"),
        )
        return ResumedResearchSession(
            research_goal=research_goal,
            context=context,
            research_id=safe_research_id,
            schema_version=RESEARCH_STATE_SCHEMA_VERSION,
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            compatibility=dict(document["compatibility"]),
            limitations=limitations,
            source_path=path,
        )


def _resolve_research_id(research_goal: ResearchGoal, context: ContextMemory) -> str:
    goal_id = str(getattr(research_goal, "research_id", "") or "").strip()
    context_id = str(getattr(context, "research_id", "") or "").strip()
    if goal_id and context_id and goal_id != context_id:
        raise ResearchStateIdentifierError(f"Research goal ID {goal_id!r} does not match context ID {context_id!r}.")
    research_id = _validate_research_id(goal_id or context_id)
    research_goal.research_id = research_id
    context.research_id = research_id
    return research_id


def _validate_research_id(research_id: str) -> str:
    value = str(research_id or "").strip()
    if not _SAFE_RESEARCH_ID.fullmatch(value):
        raise ResearchStateIdentifierError(
            "Research ID must be 1-200 characters and contain only letters, numbers, '.', '_', or '-'."
        )
    return value


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _build_document(
    research_goal: ResearchGoal,
    context: ContextMemory,
    *,
    research_id: str,
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    research_type = str(
        getattr(research_goal, "resolved_research_type", None)
        or getattr(context, "research_type", None)
        or getattr(research_goal, "research_type", None)
        or "hypothesis_testing"
    )
    hypotheses = {
        hypothesis_id: _serialize_hypothesis(hypothesis)
        for hypothesis_id, hypothesis in sorted(
            getattr(context, "hypotheses", {}).items(), key=lambda item: str(item[0])
        )
    }

    provenance_inputs = [
        getattr(context, "evidence_references", []),
        getattr(context, "evidence_relationships", []),
        getattr(context, "last_retrieved_sources", []),
        getattr(context, "last_literature_synthesis", {}),
        getattr(context, "last_generation_diagnostics", {}),
        hypotheses,
    ]
    evidence_references = _collect_evidence_references(provenance_inputs)
    evidence_relationships = _collect_evidence_relationships(
        context,
        hypotheses,
        evidence_references,
        research_id=research_id,
    )

    state = {
        "research_type": research_type,
        "iteration_number": int(getattr(context, "iteration_number", 0)),
        "research_plan": _safe_state_value(getattr(context, "research_plan", {})),
        "sub_questions": _safe_state_value(getattr(context, "sub_questions", [])),
        "evidence_requirements": _safe_state_value(getattr(context, "evidence_requirements", [])),
        "hypotheses": hypotheses,
        "evidence_references": evidence_references,
        "evidence_relationships": evidence_relationships,
        "tournament_results": _safe_state_value(getattr(context, "tournament_results", [])),
        "meta_review_feedback": _safe_state_value(getattr(context, "meta_review_feedback", [])),
        "last_literature_synthesis": _safe_state_value(getattr(context, "last_literature_synthesis", {})),
        "last_generation_diagnostics": _safe_state_value(getattr(context, "last_generation_diagnostics", {})),
        "last_evolution_attempts": _safe_state_value(getattr(context, "last_evolution_attempts", [])),
        "last_hypothesis_audits": _safe_state_value(getattr(context, "last_hypothesis_audits", [])),
        "proximity_analysis": _safe_state_value(getattr(context, "proximity_analysis", {})),
        "supervisor_state": _safe_state_value(getattr(context, "supervisor_state", {})),
        "resume_state": _safe_state_value(getattr(context, "resume_state", {})),
    }
    document: dict[str, Any] = {
        "schema_name": RESEARCH_STATE_SCHEMA,
        "schema_version": RESEARCH_STATE_SCHEMA_VERSION,
        "compatibility": {
            "minimum_reader_schema_version": RESEARCH_STATE_SCHEMA_VERSION,
            "maximum_reader_schema_version": RESEARCH_STATE_SCHEMA_VERSION,
        },
        "research_id": research_id,
        "session_id": research_id,
        "research_type": research_type,
        "created_at": created_at,
        "updated_at": updated_at,
        "research_goal": _serialize_research_goal(research_goal, research_id),
        "research_state": state,
    }
    _validate_research_mode_consistency(document)
    document["integrity"] = {
        "algorithm": _INTEGRITY_ALGORITHM,
        "digest": _content_digest(document),
    }
    return document


def _serialize_research_goal(research_goal: ResearchGoal, research_id: str) -> dict[str, Any]:
    return _safe_state_value(
        {
            "description": getattr(research_goal, "description", ""),
            "preferences": getattr(research_goal, "preferences", ""),
            "idea_attributes": getattr(research_goal, "idea_attributes", ""),
            "constraints": getattr(research_goal, "constraints", {}),
            "llm_model": getattr(research_goal, "llm_model", None),
            "query_rewrite_model": getattr(research_goal, "query_rewrite_model", None),
            "num_hypotheses": getattr(research_goal, "num_hypotheses", None),
            "generation_temperature": getattr(research_goal, "generation_temperature", None),
            "reflection_temperature": getattr(research_goal, "reflection_temperature", None),
            "elo_k_factor": getattr(research_goal, "elo_k_factor", None),
            "top_k_hypotheses": getattr(research_goal, "top_k_hypotheses", None),
            "research_type": getattr(research_goal, "research_type", "auto"),
            "resolved_research_type": getattr(research_goal, "resolved_research_type", None),
            "research_id": research_id,
        }
    )


def _serialize_hypothesis(hypothesis: Hypothesis) -> dict[str, Any]:
    report = getattr(hypothesis, "reflection_report", None)
    if isinstance(report, BaseModel):
        report_payload = report.model_dump()
    elif isinstance(report, Mapping):
        report_payload = dict(report)
    elif report is None:
        report_payload = None
    else:
        raise ResearchStateSerializationError(
            f"Hypothesis {getattr(hypothesis, 'hypothesis_id', '')!r} has an invalid reflection report."
        )

    return {
        "id": str(getattr(hypothesis, "hypothesis_id", "")),
        "title": str(getattr(hypothesis, "title", "")),
        "text": str(getattr(hypothesis, "text", "")),
        "novelty_review": _safe_state_value(getattr(hypothesis, "novelty_review", None)),
        "feasibility_review": _safe_state_value(getattr(hypothesis, "feasibility_review", None)),
        "elo_score": float(getattr(hypothesis, "elo_score", 1200.0)),
        "review_comments": _safe_state_value(getattr(hypothesis, "review_comments", [])),
        "references": _compact_reference_list(getattr(hypothesis, "references", [])),
        "review_reference_ids": _safe_state_value(getattr(hypothesis, "review_reference_ids", [])),
        "is_active": bool(getattr(hypothesis, "is_active", True)),
        "deactivation_reason": _safe_state_value(getattr(hypothesis, "deactivation_reason", None)),
        "parent_ids": _safe_state_value(getattr(hypothesis, "parent_ids", [])),
        "evolution_strategy": _safe_state_value(getattr(hypothesis, "evolution_strategy", None)),
        "reflection_report": _safe_state_value(report_payload),
        "evidence_source_ids": _safe_state_value(getattr(hypothesis, "evidence_source_ids", [])),
        "evidence_refs": _compact_reference_list(getattr(hypothesis, "evidence_refs", [])),
        "evidence_sources": _compact_reference_list(getattr(hypothesis, "evidence_sources", [])),
        "audit_score": _safe_state_value(getattr(hypothesis, "audit_score", None)),
        "audit_verdict": _safe_state_value(getattr(hypothesis, "audit_verdict", None)),
        "audit_report": _safe_state_value(getattr(hypothesis, "audit_report", {})),
    }


def _compact_reference_list(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple, set)):
        return []
    compact: list[Any] = []
    for item in value:
        if isinstance(item, Mapping):
            reference = _compact_provenance(item)
            if reference:
                compact.append(reference)
        elif isinstance(item, (str, int)):
            compact.append(str(item))
    return compact


def _compact_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        normalized = key.casefold()
        if normalized not in _PROVENANCE_FIELDS or _is_embedding_field(normalized):
            continue
        if normalized in {"evidence_ref", "evidence_refs"}:
            nested = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
            nested_refs = [_compact_provenance(item) for item in nested if isinstance(item, Mapping)]
            if nested_refs:
                compact[key] = nested_refs
            continue
        compact[key] = _safe_state_value(raw_value, source_scope=True)
    return compact


def _safe_state_value(value: Any, *, source_scope: bool = False) -> Any:
    try:
        json_safe = _to_json_value(value)
        stripped = _strip_bulk_evidence(json_safe, source_scope=source_scope)
        return sanitize(stripped)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ResearchStateSerializationError(f"Research state contains an unsupported value: {exc}") from exc


def _to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("non-finite numbers are not supported")
        return value
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _to_json_value(value.value)
    if isinstance(value, BaseModel):
        return _to_json_value(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_value(item) for item in value]
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _strip_bulk_evidence(value: Any, *, source_scope: bool = False, parent_key: str = "") -> Any:
    if isinstance(value, list):
        return [_strip_bulk_evidence(item, source_scope=source_scope, parent_key=parent_key) for item in value]
    if not isinstance(value, dict):
        return value

    source_shaped = (
        source_scope
        or parent_key in _SOURCE_CONTAINER_FIELDS
        or any(key in value for key in ("source_id", "chunk_id", "evidence_type"))
    )
    stripped: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        normalized = key.casefold()
        if _is_embedding_field(normalized) or normalized in _BULK_EVIDENCE_FIELDS:
            continue
        if normalized == "evidence_spans" or (source_shaped and normalized in _SOURCE_BODY_FIELDS):
            continue
        child_source_scope = source_shaped or normalized in _SOURCE_CONTAINER_FIELDS
        stripped[key] = _strip_bulk_evidence(
            item,
            source_scope=child_source_scope,
            parent_key=normalized,
        )
    return stripped


def _is_embedding_field(field: str) -> bool:
    normalized = field.casefold().replace("-", "_")
    return "embedding" in normalized or normalized in {"vector", "vectors", "dense_vector", "sparse_vector"}


def _collect_evidence_references(values: list[Any]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            source_id = _optional_string(value.get("source_id"))
            chunk_id = _optional_string(value.get("chunk_id"))
            if source_id or chunk_id:
                compact = _compact_provenance(value)
                if compact:
                    collected.append(compact)
            if source_id:
                for field in _CHUNK_LIST_FIELDS:
                    chunk_ids = value.get(field)
                    if isinstance(chunk_ids, (list, tuple, set)):
                        for candidate in chunk_ids:
                            candidate_id = _optional_string(candidate)
                            if candidate_id:
                                collected.append({"source_id": source_id, "chunk_id": candidate_id})
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    for value in values:
        visit(value)

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for reference in collected:
        source_id = _optional_string(reference.get("source_id")) or ""
        chunk_id = _optional_string(reference.get("chunk_id")) or ""
        identity = (source_id, chunk_id)
        if not any(identity):
            continue
        current = merged.setdefault(identity, {})
        for key, value in reference.items():
            if value not in (None, "", [], {}):
                current.setdefault(key, value)

    return [merged[key] for key in sorted(merged, key=lambda item: (item[0], item[1]))]


def _collect_evidence_relationships(
    context: ContextMemory,
    hypotheses: Mapping[str, Mapping[str, Any]],
    evidence_references: list[dict[str, Any]],
    *,
    research_id: str,
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    source_by_chunk = _unambiguous_source_by_chunk(evidence_references)

    explicit = getattr(context, "evidence_relationships", [])
    if isinstance(explicit, list):
        for relationship in explicit:
            if isinstance(relationship, Mapping):
                relationships.extend(
                    _normalize_relationship(
                        relationship,
                        research_id=research_id,
                        source_by_chunk=source_by_chunk,
                    )
                )

    for hypothesis_id, hypothesis in hypotheses.items():
        audit = hypothesis.get("audit_report")
        if isinstance(audit, Mapping):
            claim_assessments = audit.get("claim_assessments", [])
            if isinstance(claim_assessments, list):
                for claim in claim_assessments:
                    if not isinstance(claim, Mapping):
                        continue
                    relationships.extend(
                        _normalize_relationship(
                            {
                                **claim,
                                "hypothesis_id": hypothesis_id,
                                "relation": _relation_from_status(claim.get("support_status")),
                            },
                            research_id=research_id,
                            source_by_chunk=source_by_chunk,
                        )
                    )

        reflection = hypothesis.get("reflection_report")
        if isinstance(reflection, Mapping):
            claims = reflection.get("claims", [])
            if isinstance(claims, list):
                for claim_index, claim in enumerate(claims, start=1):
                    if not isinstance(claim, Mapping):
                        continue
                    claim_id = str(claim.get("claim_id") or f"reflection_claim_{claim_index}")
                    common = {
                        "hypothesis_id": hypothesis_id,
                        "claim_id": claim_id,
                        "claim": claim.get("claim", ""),
                        "evidence_strength": claim.get("confidence"),
                    }
                    for key, relation in (
                        ("supporting_evidence", "supports"),
                        ("contradictory_evidence", "contradicts"),
                    ):
                        evidence = claim.get(key, [])
                        if isinstance(evidence, list):
                            for reference in evidence:
                                if isinstance(reference, Mapping):
                                    relationships.extend(
                                        _normalize_relationship(
                                            {**common, **reference, "relation": relation},
                                            research_id=research_id,
                                            source_by_chunk=source_by_chunk,
                                        )
                                    )

        evidence_refs = hypothesis.get("evidence_refs", [])
        source_ids = [str(value) for value in hypothesis.get("evidence_source_ids", []) if value]
        if isinstance(evidence_refs, list):
            for reference in evidence_refs:
                if isinstance(reference, Mapping):
                    raw = dict(reference)
                else:
                    raw = {"chunk_id": str(reference)}
                raw.update({"hypothesis_id": hypothesis_id, "relation": "relevant"})
                if "source_id" not in raw and len(source_ids) == 1:
                    raw["source_id"] = source_ids[0]
                relationships.extend(
                    _normalize_relationship(
                        raw,
                        research_id=research_id,
                        source_by_chunk=source_by_chunk,
                    )
                )

    synthesis = getattr(context, "last_literature_synthesis", {})
    if isinstance(synthesis, Mapping):
        for key, relation in (("established_findings", "supports"), ("contradictions", "contradicts")):
            findings = synthesis.get(key, [])
            if not isinstance(findings, list):
                continue
            for index, finding in enumerate(findings, start=1):
                if not isinstance(finding, Mapping):
                    continue
                refs = finding.get("evidence_refs", [])
                if not isinstance(refs, list):
                    continue
                for reference in refs:
                    if not isinstance(reference, Mapping):
                        continue
                    relationships.extend(
                        _normalize_relationship(
                            {
                                **reference,
                                "claim_id": str(finding.get("claim_id") or f"literature_{key}_{index}"),
                                "claim": finding.get("claim", ""),
                                "relation": relation,
                            },
                            research_id=research_id,
                            source_by_chunk=source_by_chunk,
                        )
                    )

    deduplicated: dict[str, dict[str, Any]] = {}
    for relationship in relationships:
        key = json.dumps(relationship, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        deduplicated[key] = relationship
    return [deduplicated[key] for key in sorted(deduplicated)]


def _normalize_relationship(
    value: Mapping[str, Any],
    *,
    research_id: str,
    source_by_chunk: Mapping[str, str],
) -> list[dict[str, Any]]:
    raw_chunk_ids = value.get("chunk_ids")
    chunk_ids = (
        [_optional_string(item) for item in raw_chunk_ids]
        if isinstance(raw_chunk_ids, (list, tuple, set))
        else [_optional_string(value.get("chunk_id"))]
    )
    chunk_ids = [item for item in chunk_ids if item]
    if not chunk_ids:
        chunk_ids = [None]

    normalized: list[dict[str, Any]] = []
    for chunk_id in chunk_ids:
        source_id = _optional_string(value.get("source_id"))
        if source_id is None and chunk_id is not None:
            source_id = source_by_chunk.get(chunk_id)
        relationship: dict[str, Any] = {
            "research_id": research_id,
            "question_id": _optional_string(value.get("question_id")),
            "hypothesis_id": _optional_string(value.get("hypothesis_id")),
            "claim_id": _optional_string(value.get("claim_id")),
            "claim": str(value.get("claim") or ""),
            "chunk_id": chunk_id,
            "source_id": source_id,
            "relation": _relation_from_status(value.get("relation") or value.get("status")),
            "evidence_strength": _safe_state_value(value.get("evidence_strength", value.get("confidence"))),
        }
        normalized.append({key: item for key, item in relationship.items() if item not in (None, "")})
    return normalized


def _relation_from_status(value: Any) -> str:
    status = str(value or "relevant").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "supported": "supports",
        "supporting": "supports",
        "entailed": "supports",
        "contradicted": "contradicts",
        "contradictory": "contradicts",
        "partially_supported": "mixed",
        "not_found": "insufficient",
        "unverified": "insufficient",
        "unsupported": "insufficient",
    }
    return aliases.get(status, status or "relevant")


def _unambiguous_source_by_chunk(evidence_references: list[dict[str, Any]]) -> dict[str, str]:
    candidates: dict[str, set[str]] = {}
    for reference in evidence_references:
        chunk_id = _optional_string(reference.get("chunk_id"))
        source_id = _optional_string(reference.get("source_id"))
        if chunk_id and source_id:
            candidates.setdefault(chunk_id, set()).add(source_id)
    return {chunk_id: next(iter(source_ids)) for chunk_id, source_ids in candidates.items() if len(source_ids) == 1}


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _restore_research_goal(payload: Mapping[str, Any], research_id: str) -> ResearchGoal:
    try:
        goal = ResearchGoal(
            description=str(payload.get("description") or ""),
            preferences=str(payload.get("preferences") or ""),
            idea_attributes=str(payload.get("idea_attributes") or ""),
            constraints=dict(payload.get("constraints") or {}),
            llm_model=_optional_string(payload.get("llm_model")),
            query_rewrite_model=_optional_string(payload.get("query_rewrite_model")),
            num_hypotheses=_optional_int(payload.get("num_hypotheses")),
            generation_temperature=_optional_float(payload.get("generation_temperature")),
            reflection_temperature=_optional_float(payload.get("reflection_temperature")),
            elo_k_factor=_optional_int(payload.get("elo_k_factor")),
            top_k_hypotheses=_optional_int(payload.get("top_k_hypotheses")),
            research_type=str(payload.get("research_type") or "auto"),
            research_id=research_id,
        )
    except (TypeError, ValueError) as exc:
        raise ResearchStateSchemaError(f"Saved research goal is invalid: {exc}") from exc
    goal.resolved_research_type = _optional_string(payload.get("resolved_research_type"))
    return goal


def _restore_context(
    payload: Mapping[str, Any],
    *,
    research_id: str,
    research_type: str,
) -> tuple[ContextMemory, tuple[str, ...]]:
    try:
        context = ContextMemory(research_id=research_id, research_type=research_type)
    except TypeError:
        # Compatibility with older ContextMemory constructors during rolling
        # upgrades; the attributes are set explicitly below.
        context = ContextMemory()
        context.research_id = research_id
        context.research_type = research_type

    context.iteration_number = int(payload.get("iteration_number", 0))
    context.research_plan = dict(payload.get("research_plan") or {})
    context.sub_questions = list(payload.get("sub_questions") or [])
    context.evidence_requirements = list(payload.get("evidence_requirements") or [])
    context.evidence_references = list(payload.get("evidence_references") or [])
    context.evidence_relationships = list(payload.get("evidence_relationships") or [])
    context.tournament_results = list(payload.get("tournament_results") or [])
    context.meta_review_feedback = list(payload.get("meta_review_feedback") or [])
    context.last_literature_synthesis = dict(payload.get("last_literature_synthesis") or {})
    context.last_generation_diagnostics = dict(payload.get("last_generation_diagnostics") or {})
    context.last_evolution_attempts = list(payload.get("last_evolution_attempts") or [])
    context.last_hypothesis_audits = list(payload.get("last_hypothesis_audits") or [])
    context.proximity_analysis = dict(payload.get("proximity_analysis") or {})

    hypotheses = payload.get("hypotheses", {})
    if not isinstance(hypotheses, Mapping):
        raise ResearchStateSchemaError("research_state.hypotheses must be an object keyed by hypothesis ID.")
    context.hypotheses = {}
    for key, hypothesis_payload in hypotheses.items():
        if not isinstance(hypothesis_payload, Mapping):
            raise ResearchStateSchemaError(f"Hypothesis {key!r} must be an object.")
        hypothesis = _restore_hypothesis(hypothesis_payload)
        if str(key) != hypothesis.hypothesis_id:
            raise ResearchStateSchemaError(
                f"Hypothesis map key {key!r} does not match hypothesis ID {hypothesis.hypothesis_id!r}."
            )
        context.add_hypothesis(hypothesis)

    supervisor_state = dict(payload.get("supervisor_state") or {})
    pending = supervisor_state.get("pending_tasks", [])
    suspended = list(pending) if isinstance(pending, list) else []
    supervisor_state["pending_tasks"] = []
    if suspended or str(supervisor_state.get("status", "")).casefold() in {
        "running",
        "pending",
        "in_progress",
    }:
        supervisor_state["status"] = "suspended"
    context.supervisor_state = supervisor_state

    saved_resume = payload.get("resume_state", {})
    resume_state = dict(saved_resume) if isinstance(saved_resume, Mapping) else {}
    saved_suspended = resume_state.get("suspended_pending_tasks", [])
    if isinstance(saved_suspended, list):
        suspended = _deduplicate_json_values([*saved_suspended, *suspended])
    limitations = tuple(
        dict.fromkeys(
            [
                *(str(item) for item in resume_state.get("limitations", []) if isinstance(item, str) and item.strip()),
                *_RESUME_LIMITATIONS,
            ]
        )
    )
    context.resume_state = {
        **resume_state,
        "status": "resumed",
        "resumed_at": _utc_now(),
        "requires_evidence_refresh": True,
        "suspended_pending_tasks": suspended,
        "limitations": list(limitations),
    }
    # Evidence bodies are intentionally absent and must never be reconstructed
    # from stale snapshots.
    context.last_retrieved_sources = []
    return context, limitations


def _restore_hypothesis(payload: Mapping[str, Any]) -> Hypothesis:
    hypothesis_id = _optional_string(payload.get("id"))
    if hypothesis_id is None:
        raise ResearchStateSchemaError("Every saved hypothesis must have a non-empty ID.")
    report_payload = payload.get("reflection_report")
    report = None
    if report_payload is not None:
        if not isinstance(report_payload, Mapping):
            raise ResearchStateSchemaError(f"Hypothesis {hypothesis_id!r} reflection_report must be an object.")
        try:
            report = ReflectionReport.model_validate(dict(report_payload))
        except ValidationError as exc:
            raise ResearchStateSchemaError(
                f"Hypothesis {hypothesis_id!r} has an invalid reflection report: {exc}"
            ) from exc

    try:
        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            title=str(payload.get("title") or ""),
            text=str(payload.get("text") or ""),
            elo_score=float(payload.get("elo_score", 1200.0)),
            deactivation_reason=_optional_string(payload.get("deactivation_reason")),
            reflection_report=report,
        )
    except (TypeError, ValueError) as exc:
        raise ResearchStateSchemaError(f"Hypothesis {hypothesis_id!r} is invalid: {exc}") from exc

    hypothesis.novelty_review = payload.get("novelty_review")
    hypothesis.feasibility_review = payload.get("feasibility_review")
    hypothesis.review_comments = list(payload.get("review_comments") or [])
    hypothesis.references = list(payload.get("references") or [])
    hypothesis.review_reference_ids = list(payload.get("review_reference_ids") or [])
    hypothesis.is_active = bool(payload.get("is_active", True))
    hypothesis.parent_ids = list(payload.get("parent_ids") or [])
    hypothesis.evolution_strategy = payload.get("evolution_strategy")
    hypothesis.evidence_source_ids = list(payload.get("evidence_source_ids") or [])
    hypothesis.evidence_refs = list(payload.get("evidence_refs") or [])
    hypothesis.evidence_sources = list(payload.get("evidence_sources") or [])
    hypothesis.audit_score = _optional_float(payload.get("audit_score"))
    hypothesis.audit_verdict = payload.get("audit_verdict")
    hypothesis.audit_report = dict(payload.get("audit_report") or {})
    return hypothesis


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _read_and_validate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResearchStateNotFoundError(f"No saved research state exists at {path}.")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResearchStateSchemaError(f"Research state at {path} is not valid JSON.") from exc
    except OSError as exc:
        raise ResearchStateError(f"Could not read research state at {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ResearchStateSchemaError("Research-state root must be a JSON object.")
    _validate_document_schema(document)
    _validate_document_integrity(document)
    return document


def _validate_document_schema(document: Mapping[str, Any]) -> None:
    if document.get("schema_name") != RESEARCH_STATE_SCHEMA:
        raise ResearchStateSchemaError(f"Unsupported research-state schema {document.get('schema_name')!r}.")
    version = document.get("schema_version")
    if type(version) is not int or version != RESEARCH_STATE_SCHEMA_VERSION:
        raise ResearchStateCompatibilityError(
            f"Research-state schema version {version!r} is incompatible with reader version "
            f"{RESEARCH_STATE_SCHEMA_VERSION}."
        )

    compatibility = document.get("compatibility")
    if not isinstance(compatibility, Mapping):
        raise ResearchStateSchemaError("Research state is missing compatibility metadata.")
    minimum = compatibility.get("minimum_reader_schema_version")
    maximum = compatibility.get("maximum_reader_schema_version")
    if type(minimum) is not int or type(maximum) is not int or minimum > maximum:
        raise ResearchStateSchemaError("Research-state compatibility range is invalid.")
    if not minimum <= RESEARCH_STATE_SCHEMA_VERSION <= maximum:
        raise ResearchStateCompatibilityError(
            f"Reader schema version {RESEARCH_STATE_SCHEMA_VERSION} is outside the saved compatibility range "
            f"{minimum}-{maximum}."
        )

    research_id = document.get("research_id")
    _validate_research_id(str(research_id or ""))
    if document.get("session_id") != research_id:
        raise ResearchStateSchemaError("session_id must match research_id.")
    for timestamp_key in ("created_at", "updated_at"):
        timestamp = document.get(timestamp_key)
        if not isinstance(timestamp, str):
            raise ResearchStateSchemaError(f"{timestamp_key} must be an ISO-8601 timestamp.")
        try:
            parsed = dt.datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise ResearchStateSchemaError(f"{timestamp_key} must be an ISO-8601 timestamp.") from exc
        if parsed.tzinfo is None:
            raise ResearchStateSchemaError(f"{timestamp_key} must include a timezone.")
    if not isinstance(document.get("research_goal"), Mapping):
        raise ResearchStateSchemaError("research_goal must be an object.")
    if not isinstance(document.get("research_state"), Mapping):
        raise ResearchStateSchemaError("research_state must be an object.")
    _validate_research_mode_consistency(document)


def _validate_research_mode_consistency(document: Mapping[str, Any]) -> None:
    research_type = document.get("research_type")
    if research_type not in RESEARCH_TYPES:
        raise ResearchStateSchemaError(f"Unsupported saved research type {research_type!r}.")

    goal = document.get("research_goal")
    state = document.get("research_state")
    if not isinstance(goal, Mapping) or not isinstance(state, Mapping):
        return

    requested = goal.get("research_type")
    if requested not in {*RESEARCH_TYPES, "auto"}:
        raise ResearchStateSchemaError(f"Unsupported research-goal type {requested!r}.")
    resolved = goal.get("resolved_research_type")
    if resolved is not None and resolved not in RESEARCH_TYPES:
        raise ResearchStateSchemaError(f"Unsupported resolved research type {resolved!r}.")
    if resolved is not None and resolved != research_type:
        raise ResearchStateSchemaError(
            f"Resolved research type {resolved!r} does not match session type {research_type!r}."
        )
    if requested != "auto" and requested != research_type:
        raise ResearchStateSchemaError(
            f"Research-goal type {requested!r} does not match session type {research_type!r}."
        )

    state_type = state.get("research_type")
    if state_type != research_type:
        raise ResearchStateSchemaError(
            f"Context research type {state_type!r} does not match session type {research_type!r}."
        )
    plan = state.get("research_plan")
    if isinstance(plan, Mapping) and plan.get("research_type") is not None:
        plan_type = plan.get("research_type")
        if plan_type not in RESEARCH_TYPES:
            raise ResearchStateSchemaError(f"Unsupported research-plan type {plan_type!r}.")
        if plan_type != research_type:
            raise ResearchStateSchemaError(
                f"Research-plan type {plan_type!r} does not match session type {research_type!r}."
            )


def _validate_document_integrity(document: Mapping[str, Any]) -> None:
    integrity = document.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ResearchStateIntegrityError("Research state is missing its integrity record.")
    if integrity.get("algorithm") != _INTEGRITY_ALGORITHM:
        raise ResearchStateIntegrityError(
            f"Unsupported research-state integrity algorithm {integrity.get('algorithm')!r}."
        )
    expected = integrity.get("digest")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ResearchStateIntegrityError("Research-state integrity digest is malformed.")
    unsigned = {key: value for key, value in document.items() if key != "integrity"}
    actual = _content_digest(unsigned)
    if not hmac.compare_digest(expected, actual):
        raise ResearchStateIntegrityError("Research-state integrity verification failed.")


def _content_digest(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deduplicate_json_values(values: list[Any]) -> list[Any]:
    deduplicated: list[Any] = []
    seen: set[str] = set()
    for value in values:
        identity = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if identity not in seen:
            deduplicated.append(value)
            seen.add(identity)
    return deduplicated


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if isinstance(exc, (TypeError, ValueError)):
            raise ResearchStateSerializationError(f"Could not serialize research state: {exc}") from exc
        raise ResearchStateError(f"Could not atomically save research state at {path}: {exc}") from exc


__all__ = [
    "LocalJSONResearchStateStore",
    "RESEARCH_STATE_SCHEMA",
    "RESEARCH_STATE_SCHEMA_VERSION",
    "ResearchStateCompatibilityError",
    "ResearchStateError",
    "ResearchStateIdentifierError",
    "ResearchStateIntegrityError",
    "ResearchStateNotFoundError",
    "ResearchStateSchemaError",
    "ResearchStateSerializationError",
    "ResearchStateStore",
    "ResumedResearchSession",
]
