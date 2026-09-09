from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.paper_library import ChromaPaperLibrary
from app.scientific_documents import DocumentElement, ExtractedPaper
from app.source_registry import JsonSourceRegistry, resolve_source_version


class RecordingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.document_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_texts.extend(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        folded = text.casefold()
        return [float(len(text) % 19), float(folded.count("e") + 1), 1.0]


def _library(tmp_path: Path, embeddings: RecordingEmbeddings) -> ChromaPaperLibrary:
    library = ChromaPaperLibrary(
        embeddings=embeddings,
        enabled=True,
        persist_directory=tmp_path / "chroma",
        pdf_directory=tmp_path / "papers",
    )
    library.chunk_size = 500
    library.chunk_overlap = 0
    return library


def _document(
    source_id: str,
    *,
    arxiv_id: str,
    updated_at: str = "",
) -> Document:
    return Document(
        page_content="Abstract evidence.",
        metadata={
            "source_id": source_id,
            "source_type": "academic",
            "title": "Versioned Evidence",
            "authors": ["Ada Researcher"],
            "published_at": "2026-01-01T00:00:00+00:00",
            "updated_at": updated_at,
            "arxiv_id": arxiv_id,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        },
    )


def _extracted(*sections: tuple[str, str]) -> ExtractedPaper:
    elements = tuple(
        DocumentElement(
            element_type="paragraph",
            text=text,
            page=index,
            section=section,
            subsection="",
            section_path=(section,),
        )
        for index, (section, text) in enumerate(sections, start=1)
    )
    return ExtractedPaper(
        pages=tuple((element.page, element.text) for element in elements),
        total_pages=len(elements),
        truncated=False,
        elements=elements,
        parser="pypdf",
    )


def _raw_text_ids(library: ChromaPaperLibrary, source_id: str) -> dict[str, str]:
    return {
        str(metadata["raw_text"]): chunk_id
        for chunk_id, (_retrieval_text, metadata) in library._stored_source_records(source_id).items()
    }


def test_arxiv_versions_share_canonical_identity_and_reuse_unchanged_embeddings(tmp_path, monkeypatch):
    embeddings = RecordingEmbeddings()
    library = _library(tmp_path, embeddings)
    documents = [
        _document(
            "arXiv:2608.12345v1",
            arxiv_id="2608.12345v1",
            updated_at="2026-08-01T00:00:00+00:00",
        ),
        _document(
            "arXiv:2608.12345v2",
            arxiv_id="2608.12345v2",
            updated_at="2026-08-10T00:00:00+00:00",
        ),
    ]
    extracted_versions = iter(
        (
            _extracted(("Methods", "Stable method."), ("Results", "Original result."), ("Limits", "Stable limit.")),
            _extracted(
                ("Methods", "Stable method."),
                ("Results", "Revised result."),
                ("Limits", "Stable limit."),
                ("Discussion", "New discussion."),
            ),
        )
    )
    download_calls: list[str] = []

    def download(url: str, destination: Path) -> None:
        download_calls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"%PDF-{len(download_calls)}".encode())

    monkeypatch.setattr(library, "_download_pdf", download)
    monkeypatch.setattr(library, "_extract_pages", lambda _path: next(extracted_versions))

    assert library.ensure_indexed(documents[0]) is True
    assert len(embeddings.document_texts) == 3
    assert library.ensure_indexed(documents[1]) is True

    report = library.last_incremental_index_report
    assert report is not None
    assert report.canonical_source_id == "arXiv:2608.12345"
    assert report.version_key == "v2"
    assert report.remote_refresh is True
    assert report.embedded_chunks == 2
    assert report.reused_embeddings == 2
    assert report.deleted_chunks == 1
    assert len(embeddings.document_texts) == 5
    assert len(download_calls) == 2
    assert library.get_index_status("arXiv:2608.12345v1") == "SUPERSEDED"
    assert library._stored_source_records("arXiv:2608.12345v1") == {}
    assert library.verify_indexed_source("arXiv:2608.12345v2").ok is True

    source = library.source_registry.snapshot()["sources"]["arXiv:2608.12345"]
    assert set(source["versions"]) == {"v1", "v2"}
    assert source["latest_version_key"] == "v2"
    assert source["versions"]["v1"]["indexes"][library.collection_name]["status"] == "SUPERSEDED"

    assert library.ensure_indexed(documents[0]) is False
    assert len(download_calls) == 2
    assert library.verify_indexed_source("arXiv:2608.12345v2").ok is True


def test_updated_revision_incrementally_reuses_moves_and_deletes_chunks(tmp_path, monkeypatch):
    embeddings = RecordingEmbeddings()
    library = _library(tmp_path, embeddings)
    source_id = "arXiv:2608.54321"
    first = _document(source_id, arxiv_id="2608.54321", updated_at="2026-08-01T00:00:00+00:00")
    second = _document(source_id, arxiv_id="2608.54321", updated_at="2026-08-02T00:00:00+00:00")
    extracted_versions = iter(
        (
            _extracted(
                ("Methods", "Stable method."),
                ("Results", "Original result."),
                ("Ablation", "Removed ablation."),
                ("Limits", "Stable limit."),
            ),
            _extracted(
                ("Introduction", "New introduction."),
                ("Methods", "Stable method."),
                ("Results", "Revised result."),
                ("Limits", "Stable limit."),
            ),
        )
    )
    downloads = 0

    def download(_url: str, destination: Path) -> None:
        nonlocal downloads
        downloads += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"%PDF-revision-{downloads}".encode())

    monkeypatch.setattr(library, "_download_pdf", download)
    monkeypatch.setattr(library, "_extract_pages", lambda _path: next(extracted_versions))

    assert library.ensure_indexed(first) is True
    first_ids = _raw_text_ids(library, source_id)
    assert len(embeddings.document_texts) == 4
    assert library.has_current_indexed_source(second) is False

    assert library.ensure_indexed(second) is True
    second_ids = _raw_text_ids(library, source_id)
    report = library.last_incremental_index_report
    assert report is not None
    assert report.remote_refresh is True
    assert report.pdf_cache_hit is False
    assert report.artifact_cache_hit is False
    assert report.embedded_chunks == 2
    assert report.reused_embeddings == 2
    assert report.deleted_chunks == 2
    assert len(embeddings.document_texts) == 6
    assert downloads == 2
    assert first_ids["Stable method."] == second_ids["Stable method."]
    assert first_ids["Stable limit."] == second_ids["Stable limit."]
    assert set(second_ids) == {
        "New introduction.",
        "Stable method.",
        "Revised result.",
        "Stable limit.",
    }
    assert library.verify_indexed_source(source_id).ok is True
    assert library.has_current_indexed_source(first) is False
    assert library.has_current_indexed_source(second) is True


def test_parser_change_reparses_but_reuses_pdf_and_embeddings(tmp_path, monkeypatch):
    document = _document("arXiv:2608.77777", arxiv_id="2608.77777")
    original_embeddings = RecordingEmbeddings()
    original = _library(tmp_path, original_embeddings)
    monkeypatch.setattr(
        original,
        "_download_pdf",
        lambda _url, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"%PDF-parser"),
        ),
    )
    monkeypatch.setattr(
        original,
        "_extract_pages",
        lambda _path: _extracted(("Methods", "Stable method."), ("Results", "Stable result.")),
    )
    assert original.ensure_indexed(document) is True
    original_collection = original.collection_name
    assert len(original_embeddings.document_texts) == 2

    changed_embeddings = RecordingEmbeddings()
    changed = _library(tmp_path, changed_embeddings)
    changed.parser_version = "pypdf-structured-3"
    extract_calls = 0

    def extract(_path: Path) -> ExtractedPaper:
        nonlocal extract_calls
        extract_calls += 1
        return _extracted(("Methods", "Stable method."), ("Results", "Stable result."))

    monkeypatch.setattr(changed, "_download_pdf", lambda *_: (_ for _ in ()).throw(AssertionError("downloaded")))
    monkeypatch.setattr(changed, "_extract_pages", extract)

    assert changed.collection_name == original_collection
    assert changed.ensure_indexed(document) is True
    report = changed.last_incremental_index_report
    assert report is not None
    assert report.remote_refresh is False
    assert report.pdf_cache_hit is True
    assert report.artifact_cache_hit is False
    assert report.embedded_chunks == 0
    assert report.reused_embeddings == 2
    assert report.deleted_chunks == 0
    assert extract_calls == 1
    assert changed_embeddings.document_texts == []
    assert changed.verify_indexed_source(document.metadata["source_id"]).ok is True


def test_embedding_model_change_reuses_pdf_and_chunk_artifact_only(tmp_path, monkeypatch):
    document = _document("arXiv:2608.88888", arxiv_id="2608.88888")
    first_embeddings = RecordingEmbeddings()
    first = _library(tmp_path, first_embeddings)
    first.embedding_model = "embedding-model-a"
    monkeypatch.setattr(
        first,
        "_download_pdf",
        lambda _url, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"%PDF-model"),
        ),
    )
    monkeypatch.setattr(
        first,
        "_extract_pages",
        lambda _path: _extracted(("Methods", "Stable method."), ("Results", "Stable result.")),
    )
    assert first.ensure_indexed(document) is True
    first_collection = first.collection_name

    second_embeddings = RecordingEmbeddings()
    second = _library(tmp_path, second_embeddings)
    second.embedding_model = "embedding-model-b"
    monkeypatch.setattr(second, "_download_pdf", lambda *_: (_ for _ in ()).throw(AssertionError("downloaded")))
    monkeypatch.setattr(second, "_extract_pages", lambda *_: (_ for _ in ()).throw(AssertionError("reparsed")))

    assert second.collection_name != first_collection
    assert second.ensure_indexed(document) is True
    report = second.last_incremental_index_report
    assert report is not None
    assert report.remote_refresh is False
    assert report.pdf_cache_hit is True
    assert report.artifact_cache_hit is True
    assert report.embedded_chunks == 2
    assert report.reused_embeddings == 0
    assert report.deleted_chunks == 0
    assert len(second_embeddings.document_texts) == 2


def test_versioned_legacy_pdf_cache_is_adopted_without_redownload(tmp_path, monkeypatch):
    document = _document(
        "arXiv:2608.22222v2",
        arxiv_id="2608.22222v2",
        updated_at="2026-08-02T00:00:00+00:00",
    )
    embeddings = RecordingEmbeddings()
    library = _library(tmp_path, embeddings)
    legacy_path = library._pdf_path(document.metadata["source_id"])
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(b"%PDF-legacy-version")
    monkeypatch.setattr(library, "_download_pdf", lambda *_: (_ for _ in ()).throw(AssertionError("downloaded")))
    monkeypatch.setattr(library, "_extract_pages", lambda _path: _extracted(("Results", "Version two result.")))

    assert library.ensure_indexed(document) is True
    report = library.last_incremental_index_report
    assert report is not None
    assert report.pdf_cache_hit is True
    version = library.source_registry.get_version(library._source_identity(document))
    assert version is not None
    assert version["pdf_filename"] == legacy_path.name


def test_registry_round_trip_keeps_canonical_and_version_identity_separate(tmp_path):
    first = resolve_source_version(
        "arXiv:2608.99999v1",
        {"arxiv_id": "2608.99999v1", "updated": "2026-08-01"},
    )
    second = resolve_source_version(
        "arXiv:2608.99999v2",
        {"arxiv_id": "2608.99999v2", "updated": "2026-08-02"},
    )
    registry = JsonSourceRegistry(tmp_path / "registry.json")
    registry.update_version(first, document_sha256="first")
    registry.update_version(second, document_sha256="second")

    assert first.canonical_source_id == second.canonical_source_id == "arXiv:2608.99999"
    assert first.version_key == "v1"
    assert second.version_key == "v2"
    source = registry.snapshot()["sources"]["arXiv:2608.99999"]
    assert source["versions"]["v1"]["source_id"] == "arXiv:2608.99999v1"
    assert source["versions"]["v2"]["source_id"] == "arXiv:2608.99999v2"
    assert registry.get_version(first)["document_sha256"] == "first"


def test_abstract_gate_does_not_treat_an_outdated_version_as_a_cache_hit():
    from app.agents_modules.generation import GenerationAgent

    document = _document(
        "arXiv:2608.11111",
        arxiv_id="2608.11111",
        updated_at="2026-08-02T00:00:00+00:00",
    )

    class VersionAwareLibrary:
        enabled = True

        @staticmethod
        def has_current_indexed_source(candidate: Document) -> bool:
            assert candidate is document
            return False

        @staticmethod
        def has_indexed_source(_source_id: str) -> bool:
            return True

    agent = GenerationAgent(paper_library=VersionAwareLibrary())

    assert agent._has_verified_cached_full_text(document.metadata["source_id"], document) is False
