from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.models import ResearchGoal
from app.paper_library import ChromaPaperLibrary, IndexIntegrityReport, PaperChunk
from app.rag_retriever import EvidenceAspect, SearchQuery


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        return [1.0, 0.0] if "latency" in lowered else [0.0, 1.0]


def _document(source_id: str = "arXiv:1234.5678") -> Document:
    return Document(
        page_content="Title: Dense 5G Scheduling\nAbstract: A scheduling abstract.",
        metadata={
            "source_id": source_id,
            "title": "Dense 5G Scheduling",
            "pdf_url": "https://arxiv.org/pdf/1234.5678",
        },
    )


def _library(tmp_path: Path) -> ChromaPaperLibrary:
    library = ChromaPaperLibrary(
        embeddings=FakeEmbeddings(),
        enabled=True,
        persist_directory=tmp_path / "chroma",
        pdf_directory=tmp_path / "papers",
    )
    library.chunk_size = 500
    library.chunk_overlap = 50
    return library


def test_default_storage_directories_are_persistent_and_separate():
    library = ChromaPaperLibrary(embeddings=FakeEmbeddings(), enabled=False)

    assert library.pdf_directory == Path("app/paper")
    assert library.persist_directory == Path("chroma_db")
    assert library.require_indexed_sources_for_generation is True


def test_indexes_pdf_chunks_in_persistent_chroma_and_reuses_cache(tmp_path, monkeypatch):
    library = _library(tmp_path)
    download_calls = []

    def fake_download(url, destination):
        download_calls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF-test")

    monkeypatch.setattr(library, "_download_pdf", fake_download)
    monkeypatch.setattr(
        library,
        "_extract_pages",
        lambda _path: [
            (1, "Latency control evidence. " * 30),
            (2, "Throughput scheduling evidence. " * 30),
        ],
    )

    assert library.ensure_indexed(_document()) is True
    assert download_calls == ["https://arxiv.org/pdf/1234.5678"]
    results = library.search("latency reduction", ["arXiv:1234.5678"], top_k=2)
    assert results
    assert results[0].source_id == "arXiv:1234.5678"
    assert "Latency" in results[0].text

    reopened = _library(tmp_path)
    monkeypatch.setattr(reopened, "_download_pdf", lambda *_: pytest.fail("cached PDF was downloaded again"))
    monkeypatch.setattr(reopened, "_extract_pages", lambda *_: pytest.fail("cached chunks were extracted again"))
    assert reopened.ensure_indexed(_document()) is True


def test_one_orphaned_chunk_never_counts_as_a_complete_source(tmp_path):
    library = _library(tmp_path)
    source_id = "arXiv:partial"
    library._get_vector_store().add_documents(
        documents=[Document(page_content="Only one interrupted chunk", metadata={"source_id": source_id})],
        ids=["orphaned-chunk"],
    )

    report = library.verify_indexed_source(source_id)

    assert report.status == "MISSING"
    assert report.stale_ids == ("orphaned-chunk",)
    assert library.has_indexed_source(source_id) is False


def test_missing_corrupt_and_stale_chunks_are_detected_and_repaired(tmp_path, monkeypatch):
    library = _library(tmp_path)
    source_id = "arXiv:repair"
    document = _document(source_id)
    monkeypatch.setattr(
        library,
        "_download_pdf",
        lambda _url, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"%PDF-repair"),
        ),
    )
    monkeypatch.setattr(
        library,
        "_extract_pages",
        lambda _path: [(1, "Latency evidence. " * 80), (2, "Traffic spike evidence. " * 80)],
    )
    assert library.ensure_indexed(document) is True

    vector_store = library._get_vector_store()
    expected_ids = set(library._manifest_source(source_id)["chunks"])
    damaged_id, missing_id = sorted(expected_ids)[:2]
    stored = vector_store.get(ids=[damaged_id], include=["metadatas"])
    damaged_metadata = dict(stored["metadatas"][0])
    vector_store.add_documents(
        documents=[Document(page_content="corrupted content", metadata=damaged_metadata)],
        ids=[damaged_id],
    )
    vector_store.delete(ids=[missing_id])
    vector_store.add_documents(
        documents=[Document(page_content="stale content", metadata={"source_id": source_id})],
        ids=["stale-chunk"],
    )

    damaged = library.verify_indexed_source(source_id)
    assert missing_id in damaged.missing_ids
    assert damaged_id in damaged.content_hash_mismatches
    assert damaged.stale_ids == ("stale-chunk",)
    assert library.has_indexed_source(source_id) is False

    assert library.ensure_indexed(document) is True
    repaired = library.verify_indexed_source(source_id)
    assert repaired.ok is True
    assert set(library._stored_source_records(source_id)) == expected_ids


def test_interrupted_index_write_is_failed_then_repairable(tmp_path, monkeypatch):
    library = _library(tmp_path)
    source_id = "arXiv:interrupted"
    document = _document(source_id)
    monkeypatch.setattr(
        library,
        "_download_pdf",
        lambda _url, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"%PDF-interrupted"),
        ),
    )
    monkeypatch.setattr(library, "_extract_pages", lambda _path: [(1, "Latency evidence. " * 80)])
    vector_store = library._get_vector_store()
    add_documents = vector_store.add_documents

    def interrupted_write(*, documents, ids):
        add_documents(documents=documents[:1], ids=ids[:1])
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(vector_store, "add_documents", interrupted_write)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        library.ensure_indexed(document)

    assert library.get_index_status(source_id) == "FAILED"
    assert library.has_indexed_source(source_id) is False

    monkeypatch.setattr(vector_store, "add_documents", add_documents)
    assert library.ensure_indexed(document) is True
    assert library.verify_indexed_source(source_id).ok is True


def test_silent_partial_write_fails_read_after_write_verification(tmp_path, monkeypatch):
    library = _library(tmp_path)
    source_id = "arXiv:silent-partial"
    document = _document(source_id)
    monkeypatch.setattr(
        library,
        "_download_pdf",
        lambda _url, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"%PDF-silent-partial"),
        ),
    )
    monkeypatch.setattr(library, "_extract_pages", lambda _path: [(1, "Latency evidence. " * 80)])
    vector_store = library._get_vector_store()
    add_documents = vector_store.add_documents

    def partial_write(*, documents, ids):
        add_documents(documents=documents[:1], ids=ids[:1])

    monkeypatch.setattr(vector_store, "add_documents", partial_write)
    with pytest.raises(ValueError, match="read-after-write verification failed"):
        library.ensure_indexed(document)

    report = library.verify_indexed_source(source_id)
    assert report.status == "FAILED"
    assert report.missing_ids
    assert library.has_indexed_source(source_id) is False


def test_ingestion_truncation_is_partial_and_not_generation_eligible(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.max_chunks_per_paper = 1
    source_id = "arXiv:truncated"
    document = _document(source_id)
    monkeypatch.setattr(
        library,
        "_download_pdf",
        lambda _url, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"%PDF-truncated"),
        ),
    )
    monkeypatch.setattr(library, "_extract_pages", lambda _path: [(1, "Latency evidence. " * 80)])

    assert library.ensure_indexed(document) is False
    assert library.get_index_status(source_id) == "PARTIAL"
    assert library.verify_indexed_source(source_id).records_valid is True
    assert library.has_indexed_source(source_id) is False

    enriched = library.enrich_documents([document], "latency")
    assert enriched[0].metadata["full_text_indexed"] is False
    assert enriched[0].metadata["index_status"] == "PARTIAL"
    assert enriched[0].metadata["index_truncated"] is True
    assert enriched[0].metadata["evidence_status"] == "full_text_partial"


def test_search_is_restricted_to_selected_source_ids(tmp_path, monkeypatch):
    library = _library(tmp_path)
    monkeypatch.setattr(
        library,
        "_download_pdf",
        lambda _url, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"%PDF"),
        ),
    )
    monkeypatch.setattr(library, "_extract_pages", lambda _path: [(1, "Latency evidence " * 40)])
    assert library.ensure_indexed(_document("arXiv:1111.1111"))
    assert library.ensure_indexed(_document("arXiv:2222.2222"))

    results = library.search("latency", ["arXiv:2222.2222"], top_k=5)
    assert results
    assert {result.source_id for result in results} == {"arXiv:2222.2222"}


def test_web_search_pdf_is_automatically_downloaded_and_indexed(tmp_path, monkeypatch):
    library = _library(tmp_path)
    document = Document(
        page_content="Title: Web-discovered paper\nAbstract: Relevant open evidence.",
        metadata={
            "source_id": "tavily:open-paper",
            "title": "Web-discovered paper",
            "pdf_url": "https://dspace.networks.imdea.org/bitstream/handle/paper.pdf",
        },
    )
    download_calls = []

    def fake_download(url, destination):
        download_calls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF")

    monkeypatch.setattr(library, "_download_pdf", fake_download)
    monkeypatch.setattr(library, "_extract_pages", lambda _path: [(1, "Web full-text evidence " * 40)])

    enriched = library.enrich_documents([document], "open evidence")

    assert download_calls == ["https://dspace.networks.imdea.org/bitstream/handle/paper.pdf"]
    assert enriched[0].metadata["full_text_indexed"] is True
    assert enriched[0].metadata["full_text_available"] is True
    assert library.has_indexed_source("tavily:open-paper") is True


def test_enrichment_adds_bounded_full_text_and_index_metadata(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.max_prompt_chars = 35
    document = _document()
    monkeypatch.setattr(library, "ensure_indexed", lambda _document: True)
    monkeypatch.setattr(
        library,
        "search",
        lambda *_args, **_kwargs: [PaperChunk("arXiv:1234.5678", "Dense 5G Scheduling", 4, "A" * 100, 0.1)],
    )

    enriched = library.enrich_documents([document], "reduce latency")

    assert enriched[0].metadata["full_text_indexed"] is True
    assert enriched[0].metadata["full_text_chunks_used"] == 1
    assert '<evidence chunk_id="' in enriched[0].page_content
    assert 'page="4" evidence_type="full_text"' in enriched[0].page_content
    assert enriched[0].metadata["evidence_status"] == "full_text"
    assert enriched[0].metadata["evidence_refs"][1]["chunk_id"]
    assert "A" * 36 not in enriched[0].page_content


def test_cached_sources_do_not_consume_the_new_download_budget(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.max_papers_per_run = 1
    cached = _document("arXiv:1111.1111")
    new = _document("arXiv:2222.2222")
    monkeypatch.setattr(library, "has_indexed_source", lambda source_id: source_id == "arXiv:1111.1111")
    indexed_calls = []

    def index_new(document):
        indexed_calls.append(document.metadata["source_id"])
        return True

    monkeypatch.setattr(library, "ensure_indexed", index_new)
    monkeypatch.setattr(library, "search", lambda *_args, **_kwargs: [])

    enriched = library.enrich_documents([cached, new], "research query")

    assert indexed_calls == ["arXiv:2222.2222"]
    assert all(document.metadata["full_text_indexed"] is True for document in enriched)


def test_download_rejects_unapproved_pdf_hosts(tmp_path):
    library = _library(tmp_path)

    with pytest.raises(ValueError, match="not allowed"):
        library._validate_pdf_url("http://127.0.0.1/private.pdf")
    with pytest.raises(ValueError, match="not allowed"):
        library._validate_pdf_url("https://example.com/paper.pdf")


def test_download_allows_configured_springer_pdf_host(tmp_path):
    library = _library(tmp_path)

    library._validate_pdf_url("https://link.springer.com/content/pdf/10.1007/test.pdf")


@pytest.mark.parametrize("download_succeeds", [True, False])
def test_springer_metadata_without_pdf_reaches_acquisition_and_strict_gate(tmp_path, monkeypatch, download_succeeds):
    from app.agents_modules.generation import GenerationAgent
    from app.tools.springer_search import SpringerSearchTool

    paper = SpringerSearchTool._format_paper(
        {
            "doi": "10.1007/s10922-026-10060-7",
            "title": "5G bandwidth control",
            "abstract": "Latency control in network slices.",
            "url": [{"format": "html", "value": "https://doi.org/10.1007/s10922-026-10060-7"}],
        }
    )
    document = Document(page_content=paper["abstract"], metadata={**paper, "source_id": paper["arxiv_id"]})
    library = _library(tmp_path)
    attempts = []

    def download(url, destination):
        library._validate_pdf_url(url)
        attempts.append(url)
        if not download_succeeds:
            raise RuntimeError("Publisher download unavailable")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF-test")

    monkeypatch.setattr(library, "_download_pdf", download)
    monkeypatch.setattr(library, "_extract_pages", lambda _: [(1, "Latency control evidence. " * 30)])
    agent = GenerationAgent(paper_library=library)

    retained = agent._prepare_candidate_documents([document], ResearchGoal("5G latency control"))

    assert attempts == ["https://link.springer.com/content/pdf/10.1007/s10922-026-10060-7.pdf"]
    assert bool(retained) is download_succeeds
    if retained:
        assert retained[0].metadata["full_text_indexed"] is True
        assert retained[0].metadata["full_text_chunks_used"] > 0


def test_download_allows_configured_web_search_pdf_host(tmp_path):
    library = _library(tmp_path)

    library._validate_pdf_url("https://dspace.networks.imdea.org/bitstream/handle/paper.pdf")


def test_generation_full_text_failure_falls_back_to_abstracts(monkeypatch):
    from app.agents_modules.generation import GenerationAgent

    class BrokenLibrary:
        def enrich_documents(self, *_args):
            raise RuntimeError("vector database unavailable")

    agent = GenerationAgent(paper_library=BrokenLibrary())
    documents = [_document()]

    assert agent._enrich_with_full_text(documents, ResearchGoal("Reduce latency")) == documents


def test_generation_full_text_enrichment_uses_explicit_requirements():
    from app.agents_modules.generation import GenerationAgent

    class RecordingLibrary:
        def __init__(self):
            self.queries = ()

        def enrich_documents(self, documents, queries):
            self.queries = queries
            return list(documents)

    library = RecordingLibrary()
    agent = GenerationAgent(paper_library=library)
    documents = [_document()]

    assert (
        agent._enrich_with_full_text(
            documents,
            ResearchGoal("Reduce latency"),
            (EvidenceAspect("spikes", "traffic spike behavior"),),
        )
        == documents
    )
    assert library.queries[0].query == "Reduce latency"
    requirement_queries = [query for query in library.queries if query.evidence_requirement_id == "spikes"]
    assert requirement_queries
    assert all("traffic spike behavior" in query.query for query in requirement_queries)


def test_late_requirement_candidate_is_not_starved_by_broad_candidates(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.candidate_download_limit = 2
    library.per_requirement_acquisition_limit = 1
    broad_one = _document("arXiv:broad-1")
    broad_two = _document("arXiv:broad-2")
    corrective = _document("arXiv:corrective")
    corrective.metadata["evidence_requirement_id"] = "spikes"
    attempted = []

    monkeypatch.setattr(library, "has_indexed_source", lambda _source_id: False)
    monkeypatch.setattr(library, "get_index_status", lambda _source_id: "MISSING")

    def ensure(document):
        attempted.append(document.metadata["source_id"])
        return True

    monkeypatch.setattr(library, "ensure_indexed", ensure)
    monkeypatch.setattr(library, "search_many", lambda *_args, **_kwargs: [])

    library.enrich_documents(
        [broad_one, broad_two, corrective],
        (SearchQuery("traffic spike evidence", evidence_requirement_id="spikes"),),
    )

    assert attempted[0] == "arXiv:corrective"
    assert "arXiv:broad-2" not in attempted


def test_requirement_acquisition_fails_over_to_second_candidate(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.candidate_download_limit = 2
    library.per_requirement_acquisition_limit = 2
    first = _document("arXiv:first-choice")
    second = _document("arXiv:second-choice")
    for document in (first, second):
        document.metadata["evidence_requirement_id"] = "latency"
    statuses = {}
    attempted = []

    monkeypatch.setattr(library, "has_indexed_source", lambda source_id: statuses.get(source_id) == "COMMITTED")
    monkeypatch.setattr(library, "get_index_status", lambda source_id: statuses.get(source_id, "MISSING"))
    monkeypatch.setattr(
        library,
        "verify_indexed_source",
        lambda source_id: IndexIntegrityReport(
            source_id,
            statuses.get(source_id, "MISSING"),
            1 if statuses.get(source_id) == "COMMITTED" else 0,
            1 if statuses.get(source_id) == "COMMITTED" else 0,
        ),
    )

    def ensure(document):
        source_id = document.metadata["source_id"]
        attempted.append(source_id)
        statuses[source_id] = "FAILED" if source_id.endswith("first-choice") else "COMMITTED"
        return statuses[source_id] == "COMMITTED"

    monkeypatch.setattr(library, "ensure_indexed", ensure)
    monkeypatch.setattr(
        library,
        "search_many",
        lambda *_args, **_kwargs: [
            PaperChunk(
                "arXiv:second-choice",
                "Second choice",
                1,
                "Measured latency evidence",
                0.1,
                "second-chunk",
                requirement_ids=("latency",),
            )
        ],
    )

    enriched = library.enrich_documents(
        [first, second],
        (SearchQuery("latency evidence", evidence_requirement_id="latency"),),
    )

    assert attempted == ["arXiv:first-choice", "arXiv:second-choice"]
    assert enriched[1].metadata["full_text_indexed"] is True
    assert enriched[1].metadata["full_text_chunks_used"] == 1


def test_every_requirement_gets_first_opportunity_before_failover(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.candidate_download_limit = 3
    library.per_requirement_acquisition_limit = 2
    documents = [
        _document("arXiv:a-1"),
        _document("arXiv:a-2"),
        _document("arXiv:b-1"),
    ]
    documents[0].metadata["evidence_requirement_id"] = "a"
    documents[1].metadata["evidence_requirement_id"] = "a"
    documents[2].metadata["evidence_requirement_id"] = "b"
    attempted = []

    monkeypatch.setattr(library, "has_indexed_source", lambda _source_id: False)
    monkeypatch.setattr(library, "get_index_status", lambda _source_id: "MISSING")

    def fail(document):
        attempted.append(document.metadata["source_id"])
        return False

    monkeypatch.setattr(library, "ensure_indexed", fail)
    monkeypatch.setattr(library, "search_many", lambda *_args, **_kwargs: [])

    library.enrich_documents(
        documents,
        (
            SearchQuery("requirement a", evidence_requirement_id="a"),
            SearchQuery("requirement b", evidence_requirement_id="b"),
        ),
    )

    assert attempted == ["arXiv:a-1", "arXiv:b-1", "arXiv:a-2"]


def test_requirement_passages_are_reserved_before_global_chunk_fill(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.retrieval_workers = 1
    chunks = {
        "global evidence": [
            PaperChunk("source-a", "A", 1, "generic winner", 0.01, "a-1"),
            PaperChunk("source-a", "A", 2, "generic runner up", 0.02, "a-2"),
        ],
        "spike evidence": [PaperChunk("source-a", "A", 3, "spike passage", 0.03, "a-3")],
        "latency evidence": [PaperChunk("source-b", "B", 4, "latency passage", 0.04, "b-1")],
    }
    calls = []

    def search(query, source_ids, _top_k):
        calls.append((query, tuple(source_ids)))
        return chunks[query]

    monkeypatch.setattr(library, "search", search)
    selected = library.search_many(
        (
            SearchQuery("global evidence"),
            SearchQuery("spike evidence", evidence_requirement_id="spikes"),
            SearchQuery("latency evidence", evidence_requirement_id="latency"),
        ),
        ["source-a", "source-b"],
        top_k=2,
        source_ids_by_requirement={"spikes": ["source-a"], "latency": ["source-b"]},
    )

    assert [chunk.chunk_id for chunk in selected] == ["a-3", "b-1"]
    assert [chunk.requirement_ids for chunk in selected] == [("spikes",), ("latency",)]
    assert calls[1:] == [
        ("spike evidence", ("source-a",)),
        ("latency evidence", ("source-b",)),
    ]


def test_collection_name_changes_when_index_schema_changes(tmp_path):
    first = _library(tmp_path)
    second = _library(tmp_path)
    second.index_schema_version = "next-schema"

    assert first.collection_name != second.collection_name


def test_enrichment_records_abstract_only_and_full_text_failed_status(tmp_path, monkeypatch):
    library = _library(tmp_path)
    no_pdf = _document("arXiv:1111.1111")
    no_pdf.metadata.pop("pdf_url")
    failed_pdf = _document("arXiv:2222.2222")

    monkeypatch.setattr(library, "ensure_indexed", lambda _document: (_ for _ in ()).throw(RuntimeError("bad PDF")))

    enriched = library.enrich_documents([no_pdf, failed_pdf], "latency evidence")

    assert enriched[0].metadata["evidence_status"] == "abstract_only"
    assert enriched[1].metadata["evidence_status"] == "full_text_failed"
    assert all(document.metadata["evidence_mode"] == "abstract_only" for document in enriched)
