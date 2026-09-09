"""Dependency-light lexical retrieval for scientific evidence passages."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

_TOKEN_PATTERN = re.compile(r"(?u)[^\W\s]+(?:[.+:/#-][^\W\s]+)*")
_TOKEN_SEPARATOR = re.compile(r"[._+:/#-]+")


def tokenize_scientific_text(text: str) -> tuple[str, ...]:
    """Tokenize prose while retaining exact scientific identifiers.

    Hyphenated and punctuated identifiers are indexed both as one token and as
    components.  This keeps exact forms such as ``MMLU-Pro``, ``Qwen2.5-72B``,
    ``TP53``, and error codes searchable without a language-model dependency.
    """

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.casefold()):
        token = match.group(0)
        tokens.append(token)
        parts = tuple(part for part in _TOKEN_SEPARATOR.split(token) if part)
        if len(parts) > 1:
            tokens.extend(parts)
    return tuple(tokens)


@dataclass(frozen=True)
class LexicalPassage:
    """One stable passage identity and the text used by lexical retrieval."""

    chunk_id: str
    text: str


@dataclass(frozen=True)
class LexicalSearchResult:
    """One BM25 hit with a higher-is-better lexical score."""

    chunk_id: str
    lexical_score: float


class BM25PassageIndex:
    """Small in-memory Okapi BM25 index over an already persisted chunk set."""

    def __init__(
        self,
        passages: Sequence[LexicalPassage],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = max(0.01, float(k1))
        self.b = min(1.0, max(0.0, float(b)))
        unique_passages = {passage.chunk_id: passage for passage in passages if passage.chunk_id}
        self._passages = tuple(unique_passages[key] for key in sorted(unique_passages))
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_lengths: dict[str, int] = {}
        document_frequencies: Counter[str] = Counter()
        for passage in self._passages:
            term_frequencies = Counter(tokenize_scientific_text(passage.text))
            self._term_frequencies[passage.chunk_id] = term_frequencies
            self._document_lengths[passage.chunk_id] = sum(term_frequencies.values())
            document_frequencies.update(term_frequencies.keys())
        self._document_frequencies: Mapping[str, int] = document_frequencies
        total_length = sum(self._document_lengths.values())
        self._average_document_length = total_length / len(self._passages) if self._passages else 0.0

    def search(self, query: str, *, top_k: int) -> list[LexicalSearchResult]:
        """Return deterministic positive-scoring BM25 hits."""

        query_terms = Counter(tokenize_scientific_text(query))
        if not query_terms or not self._passages or top_k <= 0:
            return []

        document_count = len(self._passages)
        average_length = max(1.0, self._average_document_length)
        scored: list[LexicalSearchResult] = []
        for passage in self._passages:
            frequencies = self._term_frequencies[passage.chunk_id]
            document_length = self._document_lengths[passage.chunk_id]
            score = 0.0
            for term, query_frequency in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                document_frequency = self._document_frequencies.get(term, 0)
                inverse_document_frequency = math.log(
                    1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * document_length / average_length)
                query_weight = 1.0 + math.log(query_frequency)
                score += inverse_document_frequency * (term_frequency * (self.k1 + 1.0) / denominator) * query_weight
            if score > 0.0:
                scored.append(LexicalSearchResult(passage.chunk_id, score))

        scored.sort(key=lambda result: (-result.lexical_score, result.chunk_id))
        return scored[:top_k]


@dataclass(frozen=True)
class FusedPassageRank:
    """Passage-level fusion result, separate from provider/query RRF."""

    chunk_id: str
    hybrid_score: float
    dense_rank: int | None
    lexical_rank: int | None


def passage_rank_fusion(
    dense_chunk_ids: Sequence[str],
    lexical_chunk_ids: Sequence[str],
    *,
    k: int = 60,
) -> list[FusedPassageRank]:
    """Fuse dense and lexical rankings with deterministic reciprocal ranks."""

    rank_constant = max(1, int(k))
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}
    for ranks, chunk_ids in ((dense_ranks, dense_chunk_ids), (lexical_ranks, lexical_chunk_ids)):
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            if not chunk_id or chunk_id in ranks:
                continue
            ranks[chunk_id] = rank
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + rank)

    fused = [
        FusedPassageRank(
            chunk_id=chunk_id,
            hybrid_score=score,
            dense_rank=dense_ranks.get(chunk_id),
            lexical_rank=lexical_ranks.get(chunk_id),
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(
        key=lambda result: (
            -result.hybrid_score,
            min(result.dense_rank or math.inf, result.lexical_rank or math.inf),
            result.chunk_id,
        )
    )
    return fused
