from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.paper_library import ChromaPaperLibrary
from app.scientific_documents import (
    DocumentElement,
    ExtractedPaper,
    PypdfScientificParser,
    chunk_document_elements,
    recover_document_elements,
)


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
        lowered = text.casefold()
        return [1.0, 0.0] if "latency" in lowered else [0.0, 1.0]


def _library(tmp_path: Path, *, document_parser=None) -> ChromaPaperLibrary:
    library = ChromaPaperLibrary(
        embeddings=RecordingEmbeddings(),
        enabled=True,
        persist_directory=tmp_path / "chroma",
        pdf_directory=tmp_path / "papers",
        document_parser=document_parser,
    )
    library.chunk_size = 180
    library.chunk_overlap = 0
    # Most tests here check per-section metadata; combining has its own tests.
    library.chunk_combine_under = 0
    return library


def _paper() -> Document:
    return Document(
        page_content="Abstract candidate",
        metadata={
            "source_id": "arXiv:2608.12345v2",
            "title": "Adaptive Schedulers",
            "authors": ["Ada Lovelace", "Grace Hopper"],
            "venue": "Systems Journal",
            "published_at": "2026-08-10",
            "updated_at": "2026-08-22",
            "doi": "10.1000/scheduler",
            "arxiv_id": "2608.12345v2",
            "pdf_url": "https://arxiv.org/pdf/2608.12345v2",
            "source_type": "academic",
            "document_type": "preprint",
            # Research-state interpretations must never enter global embeddings.
            "provisional_hypothesis": "SECRET H1 says the scheduler is optimal",
        },
    )


def test_section_metadata_survives_and_chunks_do_not_cross_sections(tmp_path):
    library = _library(tmp_path)
    pages = (
        (
            1,
            "1 Introduction\nPrior schedulers have unstable tail latency.\n\n"
            "2 Methods\nWe evaluate a bounded controller under burst traffic.",
        ),
        (2, "2.1 Dataset\nThe trace contains one million requests."),
    )

    chunked = library._chunk_pages("arXiv:structured", _paper(), pages)

    assert [chunk.metadata["section"] for chunk in chunked.chunks] == [
        "Introduction",
        "Methods",
        "Methods",
    ]
    assert chunked.chunks[2].metadata["subsection"] == "Dataset"
    assert chunked.chunks[2].metadata["section_path"] == "Methods > Dataset"
    assert all(
        not ("Prior schedulers" in chunk.metadata["raw_text"] and "bounded controller" in chunk.metadata["raw_text"])
        for chunk in chunked.chunks
    )


def test_chunker_prefers_paragraph_and_sentence_boundaries():
    elements = (
        DocumentElement(
            "paragraph",
            "First methods sentence. Second methods sentence. Third methods sentence.",
            3,
            "Methods",
            "",
            ("Methods",),
        ),
        DocumentElement(
            "paragraph",
            "A separate results paragraph remains in its own section.",
            4,
            "Results",
            "",
            ("Results",),
        ),
    )

    chunks = chunk_document_elements(elements, max_chars=48)

    assert all(len(chunk.raw_text) <= 48 for chunk in chunks)
    assert all(not ({"Methods", "Results"} <= set(chunk.section_path)) for chunk in chunks)
    assert any(chunk.raw_text.endswith("sentence.") for chunk in chunks if chunk.section == "Methods")
    assert {chunk.section for chunk in chunks} == {"Methods", "Results"}


def test_running_headers_and_footers_are_dropped_and_never_open_sections():
    bodies = (
        "1 Introduction\nPrior schedulers have unstable tail latency.",
        "2 Methods\nWe evaluate a bounded controller.",
        "More method detail.",
        "3 Results\nLatency fell by 12 ms.",
    )
    pages = tuple(
        (page, f"{810 + page} Annals of Telecommunications (2026) 81:809–834\n{body}\nPage {page} of 4")
        for page, body in enumerate(bodies, start=1)
    )

    elements = recover_document_elements(pages)

    assert {element.section for element in elements} == {"Introduction", "Methods", "Results"}
    assert all("Annals" not in element.text and "Page " not in element.text for element in elements)


def test_figure_labels_formulas_and_sentence_fragments_do_not_open_sections():
    page = "\n".join(
        (
            "I. I NTRODUCTION",
            "Prior schedulers have unstable tail latency.",
            "II. R ELATED W ORK",
            "Earlier controllers ignore burst traffic.",
            "3 System Model",
            "The system has one scheduler.",
            "5G",
            "∑ K",
            "1G 2G 3G 4G 5G 6G",
            "NFVI",
            "10 m/s, indicating that the controller converges",
            "1 Department of Information Technology, Example University",
            "3.1 Channel Model",
            "The channel is Rayleigh fading.",
            "4 E VALUATION S TUDY C ASES",
            "Two cases are evaluated.",
            "5 F INAL REMARKS",
            "The controller is stable.",
            "6 A BRIEF OVERVIEW OF LEARNING",
            "Learning methods are summarized.",
            "VII. A CKNOWLEDGMENT",
            "We thank the reviewers.",
        )
    )

    elements = recover_document_elements(((1, page),))

    assert [element.section_path for element in elements] == [
        ("Introduction",),
        ("Related Work",),
        ("System Model",),
        ("System Model", "Channel Model"),
        ("Evaluation Study Cases",),
        ("Final Remarks",),
        ("A Brief Overview Of Learning",),
        ("Acknowledgements",),
    ]
    assert "NFVI" in elements[2].text
    assert "1 Department of Information Technology" in elements[2].text


def test_bibliography_lines_stay_in_references_and_are_not_indexed(tmp_path):
    library = _library(tmp_path)
    page = "\n".join(
        (
            "1 Introduction",
            "Prior schedulers have unstable tail latency.",
            "References",
            "[1] A. Author, Scheduler study, 2020.",
            "IEEE TRANSACTIONS ON NETWORKING",
            "2 Burst Traffic Control",
            "Appendix",
            "Proof details for the bound.",
        )
    )

    elements = recover_document_elements(((1, page),))
    chunked = library._chunk_pages("arXiv:references", _paper(), ((1, page),))

    assert [element.section for element in elements] == ["Introduction", "References", "Appendix"]
    assert "2 Burst Traffic Control" in elements[1].text
    assert [chunk.metadata["section"] for chunk in chunked.chunks] == ["Introduction", "Appendix"]
    assert all("IEEE TRANSACTIONS" not in chunk.metadata["raw_text"] for chunk in chunked.chunks)


def test_short_sections_combine_until_the_threshold_and_chunks_stay_bounded():
    elements = (
        DocumentElement("paragraph", "M" * 100, 1, "Methods", "", ("Methods",)),
        DocumentElement("paragraph", "S" * 100, 1, "Methods", "Setup", ("Methods", "Setup")),
        DocumentElement("paragraph", "R" * 700, 2, "Results", "", ("Results",)),
        DocumentElement("paragraph", "D" * 600, 3, "Discussion", "", ("Discussion",)),
    )

    combined = chunk_document_elements(elements, max_chars=1000, combine_under_chars=500)
    separate = chunk_document_elements(elements, max_chars=1000)

    assert [chunk.section_path for chunk in combined] == [("Results",), ("Discussion",)]
    assert combined[0].raw_text.startswith("M" * 100)
    assert (combined[0].page_start, combined[0].page_end) == (1, 2)
    assert all(len(chunk.raw_text) <= 1000 for chunk in combined)
    assert len(separate) == 4


def test_short_equations_join_text_while_long_tables_stay_standalone():
    table = "Table 1: Latency by load\n" + "\n".join(f"load{row} 1.0 2.0 3.0" for row in range(40))
    elements = (
        DocumentElement("paragraph", "We measure latency under burst traffic.", 4, "Results", "", ("Results",)),
        DocumentElement("equation", "L = q / μ", 4, "Results", "", ("Results",)),
        DocumentElement("paragraph", "Latency falls as service rate grows.", 4, "Results", "", ("Results",)),
        DocumentElement("table", table, 5, "Results", "", ("Results",)),
    )

    chunks = chunk_document_elements(elements, max_chars=400, overlap_chars=50, standalone_min_chars=100)

    assert chunks[0].element_type == "paragraph"
    assert "L = q / μ" in chunks[0].raw_text
    assert "Latency falls" in chunks[0].raw_text
    table_chunks = chunks[1:]
    assert len(table_chunks) > 1
    assert all(chunk.element_type == "table" and len(chunk.raw_text) <= 400 for chunk in table_chunks)
    assert sum(len(chunk.raw_text) for chunk in table_chunks) <= len(table)


def test_pypdf_parser_recovers_structure_without_an_advanced_dependency(monkeypatch, tmp_path):
    import pypdf

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        pages = [
            FakePage("ABSTRACT\nA faithful abstract.\n\nMETHODS\nA faithful method."),
            FakePage("RESULTS\nA faithful result."),
        ]

    monkeypatch.setattr(pypdf, "PdfReader", lambda _path: FakeReader())

    parsed = PypdfScientificParser().parse(tmp_path / "paper.pdf", max_pages=10)

    assert parsed.parser == "pypdf"
    assert parsed.truncated is False
    assert [element.section for element in parsed.elements] == ["Abstract", "Methods", "Results"]
    assert parsed.elements[-1].text == "A faithful result."


def test_failed_structured_parser_uses_pypdf_fallback(tmp_path, monkeypatch):
    class BrokenParser:
        name = "optional-structured-parser"

        def parse(self, _pdf_path: Path, *, max_pages: int) -> ExtractedPaper:
            raise RuntimeError(f"structure unavailable at limit {max_pages}")

    fallback = ExtractedPaper(
        pages=((1, "METHODS\nFallback method."),),
        total_pages=1,
        truncated=False,
        elements=recover_document_elements(((1, "METHODS\nFallback method."),)),
        parser="pypdf",
    )
    library = _library(tmp_path, document_parser=BrokenParser())
    monkeypatch.setattr(library._pypdf_parser, "parse", lambda *_args, **_kwargs: fallback)

    parsed = library._extract_pages(tmp_path / "paper.pdf")

    assert parsed is fallback
    assert parsed.elements[0].section == "Methods"


def test_raw_retrieval_and_display_text_are_distinct(tmp_path):
    library = _library(tmp_path)
    raw_text = "Measured p = 0.031; median latency fell by 12 ms."
    elements = (
        DocumentElement(
            "paragraph",
            raw_text,
            7,
            "Results",
            "Main Results",
            ("Results", "Main Results"),
        ),
    )

    chunk = library._chunk_pages(
        "arXiv:2608.12345v2",
        _paper(),
        ((7, raw_text),),
        elements=elements,
    ).chunks[0]

    assert chunk.metadata["raw_text"] == raw_text
    assert chunk.page_content.endswith(raw_text)
    assert "Paper: Adaptive Schedulers" in chunk.page_content
    assert "Section: Results > Main Results" in chunk.page_content
    assert "Authors: Ada Lovelace, Grace Hopper" in chunk.page_content
    assert "SECRET H1" not in chunk.page_content
    assert chunk.metadata["retrieval_template_version"] == "intrinsic-context-1"


def test_persistent_index_embeds_retrieval_text_and_returns_raw_evidence(tmp_path, monkeypatch):
    library = _library(tmp_path)
    paper = _paper()
    raw_text = "The measured latency improvement was 12 ms."
    limitation_text = "Energy use was not evaluated."
    extracted = ExtractedPaper(
        pages=((5, raw_text), (6, limitation_text)),
        total_pages=2,
        truncated=False,
        elements=(
            DocumentElement(
                "paragraph",
                raw_text,
                5,
                "Results",
                "Main Results",
                ("Results", "Main Results"),
            ),
            DocumentElement(
                "paragraph",
                limitation_text,
                6,
                "Limitations",
                "",
                ("Limitations",),
            ),
        ),
        parser="pypdf",
    )

    def download(_url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF-phase-b")

    monkeypatch.setattr(library, "_download_pdf", download)
    monkeypatch.setattr(library, "_extract_pages", lambda _path: extracted)

    assert library.ensure_indexed(paper) is True
    stored = library._stored_source_records("arXiv:2608.12345v2")
    assert len(stored) == 2
    records_by_section = {metadata["section"]: (text, metadata) for text, metadata in stored.values()}
    retrieval_text, metadata = records_by_section["Results"]
    _limitation_retrieval_text, limitation_metadata = records_by_section["Limitations"]
    assert retrieval_text in library.embeddings.document_texts
    assert retrieval_text != raw_text
    assert metadata["raw_text"] == raw_text
    assert metadata["content_sha256"] == library._content_hash(raw_text)
    assert metadata["retrieval_text_sha256"] == library._content_hash(retrieval_text)
    assert metadata["document_id"] == "arXiv:2608.12345v2"
    assert metadata["paper_version"] == "v2"
    assert metadata["section_path"] == "Results > Main Results"
    assert metadata["page_start"] == metadata["page_end"] == 5
    assert metadata["element_type"] == "paragraph"
    assert metadata["schema_version"] == "4"
    assert metadata["parser_version"] == "pypdf-structured-3"
    assert metadata["chunking_version"] == "section-paragraph-sentence-4"
    assert metadata["retrieval_template_version"] == "intrinsic-context-1"
    assert metadata["chunk_count"] == 2
    assert metadata["parent_id"]
    assert metadata["next_chunk_id"] == limitation_metadata["chunk_id"]
    assert limitation_metadata["previous_chunk_id"] == metadata["chunk_id"]
    assert "SECRET H1" not in retrieval_text

    result = library.search("latency", ["arXiv:2608.12345v2"], top_k=1)[0]
    assert result.text == result.raw_text == raw_text
    assert result.retrieval_text == retrieval_text
    assert result.display_text.endswith(raw_text)
    assert "Source ID: arXiv:2608.12345v2" in result.display_text
    assert result.section_path == ("Results", "Main Results")
    assert library.verify_indexed_source("arXiv:2608.12345v2").ok is True


def test_search_reads_legacy_chunks_without_new_metadata(tmp_path):
    library = _library(tmp_path)
    library._get_vector_store().add_documents(
        documents=[
            Document(
                page_content="Legacy raw evidence",
                metadata={
                    "source_id": "arXiv:legacy",
                    "title": "Legacy Paper",
                    "page": 9,
                },
            )
        ],
        ids=["legacy-chunk"],
    )

    result = library.search("legacy", ["arXiv:legacy"], top_k=1)[0]

    assert result.text == result.raw_text == "Legacy raw evidence"
    assert result.retrieval_text == "Legacy raw evidence"
    assert result.section == "Unknown"
    assert result.page_start == result.page_end == 9
