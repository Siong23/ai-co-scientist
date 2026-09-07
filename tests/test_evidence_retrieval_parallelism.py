"""Evidence-query concurrency must preserve retrieval and fusion semantics."""

from threading import Barrier
from unittest.mock import Mock, patch

import pytest

from app.agents_modules.proximity_helpers import SimilarityConfig, SimilarityScorer
from app.paper_library import ChromaPaperLibrary, PaperChunk


def test_parallel_search_matches_serial_fusion_and_deduplicates_queries():
    library = ChromaPaperLibrary(enabled=True)
    library._vector_store = Mock()
    a = PaperChunk("s1", "A", 1, "first", chunk_id="a")
    b = PaperChunk("s1", "B", 2, "second", chunk_id="b")
    queries = {"q1": [a, b], "q2": [b, a], "q3": [a]}
    library.retrieval_workers = 1
    with patch.object(library, "search", side_effect=lambda q, *args: queries[q]):
        serial = library.search_many(["q1", "q2", "q3"], ["s1"], 2)
    rendezvous = Barrier(3, timeout=3)

    def search(query, source_ids, top_k):
        assert source_ids == ["s1"] and top_k == 2
        rendezvous.wait()
        return queries[query]

    library.retrieval_workers = 3
    with patch.object(library, "search", side_effect=search) as mocked:
        parallel = library.search_many(["q1", "q2", "q1", "q3"], ["s1"], 2)
    assert mocked.call_count == 3
    assert parallel == serial


def test_parallel_search_failure_propagates_instead_of_silently_dropping_evidence():
    library = ChromaPaperLibrary(enabled=True)
    library._vector_store = Mock()
    with patch.object(library, "search", side_effect=RuntimeError("retrieval failed")):
        with pytest.raises(RuntimeError, match="retrieval failed"):
            library.search_many(["q1", "q2"], ["s1"])


def test_empty_search_does_not_initialize_store():
    library = ChromaPaperLibrary(enabled=True)
    with patch.object(library, "_get_vector_store") as store:
        assert library.search_many([], ["s1"]) == []
        assert library.search_many(["q1"], []) == []
    store.assert_not_called()


def test_default_proximity_reuses_shared_embedding_provider():
    shared = Mock()
    with patch("app.utils.get_sentence_transformer_model", return_value=shared) as get_shared:
        scorer = SimilarityScorer()
        assert scorer._load_embedding_model() is shared
        assert scorer._load_embedding_model() is shared
    get_shared.assert_called_once_with()


def test_explicit_proximity_model_is_preserved():
    with patch("sentence_transformers.SentenceTransformer") as constructor:
        scorer = SimilarityScorer(SimilarityConfig(embedding_model_name="explicit-model"))
        assert scorer._load_embedding_model() is constructor.return_value
    constructor.assert_called_once_with("explicit-model")
