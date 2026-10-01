from app.experiments.paper_reader import PaperReader
from app.paper_library import PaperChunk


def _chunk(index: int, section: str, text: str) -> PaperChunk:
    return PaperChunk(
        source_id="arXiv:2003.11003v1",
        title="Learn to Schedule",
        page=index + 1,
        text=text,
        chunk_id=f"chunk-{index}",
        section=section,
        section_path=(section,),
        retrieval_text=f"Title: Learn to Schedule\nSection: {section}\n{text}",
        display_text=f"[arXiv:2003.11003v1] {section}\n{text}",
        content_sha256="a" * 64,
        chunk_index=index,
    )


def _paper() -> list[PaperChunk]:
    return [
        _chunk(0, "Abstract", "ABSTRACT-TEXT " + "a" * 800),
        _chunk(1, "Introduction", "INTRO-ONE " + "i" * 900),
        _chunk(2, "Introduction", "INTRO-TWO " + "i" * 900),
        _chunk(3, "Related Work", "RELATED " + "r" * 900),
        _chunk(4, "The Proposed Scheduler", "SETUP-TEXT " + "s" * 900),
        _chunk(5, "Results", "RESULTS-TEXT Table 2 shows 94.1% throughput, compared with 88.0% for PF. " + "x" * 700),
        _chunk(6, "Conclusion", "CONCLUSION " + "c" * 900),
    ]


def test_indexed_text_uses_passages_and_prefers_results_over_introduction():
    reader = PaperReader()
    reader.max_indexed_text_length = 2800

    text = reader._prepare_indexed_text(_paper())

    assert "PaperChunk(" not in text
    assert "retrieval_text" not in text
    assert text.startswith("ABSTRACT-TEXT")
    assert "RESULTS-TEXT Table 2 shows 94.1%" in text
    assert "INTRO-ONE" not in text
    assert "RELATED" not in text
    assert text.index("ABSTRACT-TEXT") < text.index("SETUP-TEXT") < text.index("RESULTS-TEXT")
    assert len(text.replace("\n\n", "")) <= reader.max_indexed_text_length


def test_result_cues_rescue_chunks_under_a_misdetected_heading():
    reader = PaperReader()
    reader.max_indexed_text_length = 1900
    chunks = [
        _chunk(0, "Unknown", "TITLE-PAGE " + "t" * 800),
        _chunk(1, "Existing Methods", "BACKGROUND " + "b" * 900),
        _chunk(2, "Existing Methods", "FIGURE 7 shows the optimizer outperforms by 12% and 9%. " + "g" * 800),
    ]

    text = reader._prepare_indexed_text(chunks)

    assert "FIGURE 7 shows" in text
    assert "BACKGROUND" not in text


def test_only_the_first_abstract_chunk_leads():
    reader = PaperReader()
    reader.max_indexed_text_length = 2800
    chunks = [
        _chunk(0, "Unknown", "JOURNAL-HEADER " + "h" * 800),
        _chunk(1, "Abstract", "REAL-ABSTRACT " + "a" * 800),
        _chunk(2, "Abstract", "MISLABELLED-INTRO " + "m" * 800),
        _chunk(3, "Evaluation", "EVALUATION Table 4 lists 3.2% loss. " + "e" * 800),
    ]

    text = reader._prepare_indexed_text(chunks)

    assert "REAL-ABSTRACT" in text
    assert "EVALUATION Table 4" in text
    assert "MISLABELLED-INTRO" not in text


def test_a_chunk_that_does_not_fit_is_skipped_for_a_smaller_one():
    reader = PaperReader()
    reader.max_indexed_text_length = 1500
    chunks = [
        _chunk(0, "Abstract", "ABSTRACT " + "a" * 900),
        _chunk(1, "Results", "BIG-RESULT Table 1 " + "x" * 900),
        _chunk(2, "Discussion", "SMALL-RESULT Fig. 3 " + "y" * 300),
    ]

    text = reader._prepare_indexed_text(chunks)

    assert "BIG-RESULT" not in text
    assert "SMALL-RESULT" in text


def test_read_experiment_reference_sends_results_from_indexed_chunks():
    class FakeLibrary:
        def __init__(self) -> None:
            self.requested: list[str] = []

        def get_source_chunks(self, source_id: str) -> list[PaperChunk]:
            self.requested.append(source_id)
            return _paper()

    library = FakeLibrary()
    reader = PaperReader(paper_library=library)
    reader.max_indexed_text_length = 2800

    reference = reader.read_experiment_reference(
        "https://arxiv.org/abs/2003.11003",
        source={"source_id": "arXiv:2003.11003"},
    )

    assert library.requested == ["arXiv:2003.11003"]
    assert reference["indexed"] is True
    assert "RESULTS-TEXT Table 2 shows 94.1%" in reference["results_text"]
    assert "PaperChunk(" not in reference["results_text"]


def test_metric_keywords_match_whole_words_only():
    # "generations" and "operation" contain "ratio"; "percentile" contains "percent".
    assert PaperReader._normalise_metric_value("convergence_generations_standard", 4) == 4
    assert PaperReader._normalise_metric_value("non_stationary_simulation_generations", 60) == 60
    assert PaperReader._infer_value_type("non_stationary_operation_periods") == "numeric"
    assert PaperReader._infer_value_type("p95_latency_percentile") != "percentage"

    assert PaperReader._infer_value_type("packet_delivery_ratio") == "percentage"
    assert PaperReader._infer_value_type("F1-score") == "percentage"
    assert PaperReader._infer_value_type("threefold_reduction") == "factor"
    assert PaperReader._normalise_metric_value("accuracy", 95) == 0.95
    assert PaperReader._normalise_metric_value("throughput_improvement_vs_pf_percent", 2.4) == 0.024


def test_primary_metrics_are_preserved_and_configurations_are_not_metrics():
    details = PaperReader._validate_and_normalise_details(
        {
            "metrics": ["latency", "deployment_scale"],
            "primary_metrics": ["Latency"],
            "metric_definitions": {
                "latency": {
                    "unit": "ms",
                    "value_type": "measurement",
                },
                "deployment_scale": {
                    "unit": "UEs",
                    "value_type": "configuration",
                },
            },
            "reference_metrics": {
                "latency": {
                    "value": 72.4,
                    "unit": "ms",
                    "value_type": "measured_value",
                    "relation": "exact",
                },
                "deployment_scale": {
                    "value": 500,
                    "unit": "UEs",
                    "value_type": "configuration",
                    "relation": "none",
                },
            },
        },
        raw_text="Latency was measured at 72.4 ms with 500 UEs.",
    )

    assert details["primary_metrics"] == ["latency"]
    assert details["metrics"] == ["latency"]
    assert "deployment_scale" not in details["reference_metrics"]
    assert details["reference_conditions"]["deployment_scale"]["value"] == 500
