"""Turn persisted evidence records into a DeepEval ``retrieval_context``.

DeepEval's RAG metrics judge generated text against the *text* of the retrieved
sources. A citation stub ("this DOI, this title") cannot support or contradict a
claim, so this module separates substantive source prose from identifying
metadata and refuses to fabricate the latter into the former.

Field names below were read off the persisted artifacts and the parent
application's serializer (``app/rag_retriever.py`` and ``app/paper_library.py``)
rather than assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rubrics.errors import LLMEvaluationError

# Fields that identify a source but never describe its findings. A record
# carrying only these is a citation stub, not readable evidence.
METADATA_ONLY_FIELDS = frozenset(
    {
        "arxiv_id",
        "arxiv_url",
        "author",
        "authors",
        "canonical_url",
        "doi",
        "domain",
        "pdf_url",
        "provider",
        "published",
        "published_at",
        "source",
        "source_id",
        "title",
        "updated_at",
        "url",
        "venue",
    }
)

# Persisted source-level prose fields, in preference order. ``content`` holds
# extracted page text, ``abstract``/``summary`` the publication abstract.
SOURCE_TEXT_FIELDS = ("content", "abstract", "summary")

# Retrieved chunks are persisted per source; each carries its own passage text.
PASSAGE_LIST_FIELD = "evidence_refs"
PASSAGE_TEXT_FIELD = "text"

# A passage shorter than this is a stub or a parser artifact, not evidence a
# judge can weigh a scientific claim against.
MIN_PASSAGE_CHARS = 200

NO_SOURCES_REASON = "the selected hypothesis has no persisted evidence sources"
METADATA_ONLY_REASON = (
    "the persisted evidence sources carry only citation metadata "
    f"(no source passage of at least {MIN_PASSAGE_CHARS} characters)"
)


@dataclass(frozen=True)
class RetrievalContext:
    """Substantive evidence passages extracted from a run artifact."""

    passages: tuple[str, ...]
    source_count: int
    sourced_count: int

    @property
    def is_substantive(self) -> bool:
        """Whether the artifact supplied evidence text a judge can read."""
        return bool(self.passages)

    @property
    def reason(self) -> str:
        """Explain why this context cannot back an evidence-grounded metric."""
        if self.is_substantive:
            return ""
        if not self.source_count:
            return NO_SOURCES_REASON
        return METADATA_ONLY_REASON

    def as_list(self) -> list[str] | None:
        """Return the DeepEval ``retrieval_context`` value, or ``None``."""
        return list(self.passages) if self.passages else None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable description for the report."""
        return {
            "substantive": self.is_substantive,
            "evidence_source_count": self.source_count,
            "sources_with_text": self.sourced_count,
            "passage_count": len(self.passages),
            "reason": self.reason or None,
        }


def _usable_passage(value: Any) -> str | None:
    """Return ``value`` verbatim when it is long enough to count as evidence."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if len(text) >= MIN_PASSAGE_CHARS else None


def _source_passages(source: Mapping[str, Any]) -> list[str]:
    """Collect one string per persisted passage, chunks before source prose."""
    passages: list[str] = []

    refs = source.get(PASSAGE_LIST_FIELD)
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        for ref in refs:
            if not isinstance(ref, Mapping):
                continue
            passage = _usable_passage(ref.get(PASSAGE_TEXT_FIELD))
            if passage is not None:
                passages.append(passage)

    for field in SOURCE_TEXT_FIELDS:
        passage = _usable_passage(source.get(field))
        if passage is not None:
            passages.append(passage)

    return passages


def extract_retrieval_context(sources: Any) -> RetrievalContext:
    """Build a ``RetrievalContext`` from persisted ``evidence_sources``.

    Each persisted passage stays its own string; nothing is concatenated,
    summarized, or invented. Duplicates are dropped because the serializer
    writes the abstract into both ``abstract`` and ``summary``.
    """
    if sources is None:
        sources = []
    if not isinstance(sources, list) or not all(isinstance(item, Mapping) for item in sources):
        raise LLMEvaluationError("evidence_sources must be a list of objects")

    passages: list[str] = []
    seen: set[str] = set()
    sourced_count = 0

    for source in sources:
        source_passages = _source_passages(source)
        if source_passages:
            sourced_count += 1
        for passage in source_passages:
            key = " ".join(passage.split())
            if key in seen:
                continue
            seen.add(key)
            passages.append(passage)

    return RetrievalContext(
        passages=tuple(passages),
        source_count=len(sources),
        sourced_count=sourced_count,
    )


__all__ = [
    "METADATA_ONLY_FIELDS",
    "MIN_PASSAGE_CHARS",
    "PASSAGE_LIST_FIELD",
    "PASSAGE_TEXT_FIELD",
    "SOURCE_TEXT_FIELDS",
    "RetrievalContext",
    "extract_retrieval_context",
]
