from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.models import ResearchGoal
from app.paper_library import ChromaPaperLibrary, PaperChunk


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
    assert "[Full-text evidence, page 4]" in enriched[0].page_content
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


def test_generation_full_text_failure_falls_back_to_abstracts(monkeypatch):
    from app.agents_modules.generation import GenerationAgent

    class BrokenLibrary:
        def enrich_documents(self, *_args):
            raise RuntimeError("vector database unavailable")

    agent = GenerationAgent(paper_library=BrokenLibrary())
    documents = [_document()]

    assert agent._enrich_with_full_text(documents, ResearchGoal("Reduce latency")) == documents
