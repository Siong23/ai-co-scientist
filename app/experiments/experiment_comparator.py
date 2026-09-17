"""
Experiment Comparator.

Compares the result of an automatically generated ML experiment
against the scientific result reported by the evidence supporting
the selected Rank #1 hypothesis.

Workflow:

    Final Rank #1 Hypothesis
            |
            +--> evidence_sources
            |
            v
    CodeGenerationAgent
            |
            v
    ExperimentRunner
            |
            v
    experiment_result
            |
            v
    ExperimentComparator
            |
            +--> Extract paper result from evidence
            +--> Extract experiment metrics
            +--> Check comparability
            +--> Calculate metric differences
            +--> Ask LLM to explain differences
            |
            v
    Comparison Result

Run with:
    pytest tests/test_experiment_comparator.py -v -s
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Dict, List, Optional
from app.experiments.paper_reader import PaperReader
from app.paper_library import ChromaPaperLibrary


class ExperimentComparator:
    """
    Compare an automated experiment result with results reported
    by the scientific evidence attached to a hypothesis.

    The comparator is responsible for COMPARISON and INTERPRETATION.

    It does not:
        - generate experiment code
        - execute experiments
        - perform ranking
        - modify hypotheses

    Python performs numerical calculations.
    The LLM is used only for scientific extraction and explanation.
    """

    COMPARABLE_METRICS = (
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    )

    METRIC_ALIASES = {
        "accuracy": [
            "accuracy",
            "acc",
        ],
        "precision_weighted": [
            "precision_weighted",
            "weighted_precision",
            "weighted precision",
            "precision",
        ],
        "recall_weighted": [
            "recall_weighted",
            "weighted_recall",
            "weighted recall",
            "recall",
            "sensitivity",
        ],
        "f1_weighted": [
            "f1_weighted",
            "weighted_f1",
            "weighted f1",
            "f1",
            "f1-score",
            "f1 score",
        ],
    }

    def __init__(
        self,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize the experiment comparator.

        Parameters
        ----------
        llm_model:
            Optional model name passed to the existing LLM facade.
            If None, the facade's default model is used.
        """

        self.llm_model = llm_model
        self.paper_reader = PaperReader()
        self.paper_library = ChromaPaperLibrary()

    # ============================================================
    # Utility
    # ============================================================

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """
        Convert a value into a finite float.

        Returns None when conversion fails or the value is
        NaN/infinite.
        """

        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number):
            return None

        return number

    @staticmethod
    def _normalise_metric_name(name: Any) -> str:
        """
        Normalize a metric name for comparison.
        """

        if name is None:
            return ""

        return re.sub(
            r"[^a-z0-9]+",
            "_",
            str(name).strip().lower(),
        ).strip("_")

    @staticmethod
    def _clamp_metric(value: float) -> float:
        """
        Convert percentage-style metric values to [0, 1].

        For example:

            96.5 -> 0.965
            0.965 -> 0.965

        This is useful because papers may report percentages while
        experiment metrics are usually stored as decimal values.
        """

        if value > 1.0 and value <= 100.0:
            return value / 100.0

        return value

    @staticmethod
    def _extract_json(
        response: str,
    ) -> Dict[str, Any]:
        """
        Extract the first valid JSON object from an LLM response.

        Handles:
        - Pure JSON responses
        - Markdown JSON fences
        - Extra explanation before or after the JSON object
        """

        if not response or not isinstance(response, str):
            raise ValueError(
                "The LLM returned an empty response."
            )

        cleaned_response = response.strip()

        # Remove Markdown code fences if present.
        cleaned_response = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned_response,
            flags=re.IGNORECASE,
        )

        cleaned_response = re.sub(
            r"\s*```$",
            "",
            cleaned_response,
        ).strip()

        # Try parsing the complete response first.
        try:
            parsed = json.loads(
                cleaned_response
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # Find the first JSON object inside extra LLM text.
        decoder = json.JSONDecoder()

        for index, character in enumerate(
            cleaned_response
        ):
            if character != "{":
                continue

            candidate = cleaned_response[index:]

            try:
                parsed, _ = decoder.raw_decode(
                    candidate
                )

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                continue

        raise ValueError(
            "The LLM response did not contain a valid JSON object."
        )

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """
        Convert non-finite floating-point values into JSON-safe values.
        """
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return value

        if isinstance(value, dict):
            return {
                key: ExperimentComparator._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                ExperimentComparator._json_safe(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                ExperimentComparator._json_safe(item)
                for item in value
            ]

        return value

    # ============================================================
    # Evidence Handling
    # ============================================================

    @staticmethod
    def _get_hypothesis_evidence(
        hypothesis: Any,
    ) -> List[Dict[str, Any]]:
        """
        Get evidence sources attached specifically to the selected
        hypothesis.

        This is preferred over using all generation sources because
        the evidence attached to Rank #1 is the evidence that
        supports that particular hypothesis.
        """

        if hypothesis is None:
            return []

        evidence = getattr(
            hypothesis,
            "evidence_sources",
            None,
        )

        if isinstance(evidence, list):
            return [
                item
                for item in evidence
                if isinstance(item, dict)
            ]

        # Also support dictionary hypotheses.
        if isinstance(hypothesis, dict):
            evidence = hypothesis.get(
                "evidence_sources",
                [],
            )

            if isinstance(evidence, list):
                return [
                    item
                    for item in evidence
                    if isinstance(item, dict)
                ]

        return []

    @staticmethod
    def _get_hypothesis_field(
        hypothesis: Any,
        field: str,
        default: Any = None,
    ) -> Any:
        """
        Read a field from either a Hypothesis object or dictionary.
        """

        if hypothesis is None:
            return default

        if isinstance(hypothesis, dict):
            return hypothesis.get(
                field,
                default,
            )

        return getattr(
            hypothesis,
            field,
            default,
        )

    @staticmethod
    def _format_evidence(
        evidence_sources: List[Dict[str, Any]],
    ) -> str:
        """
        Format scientific evidence for paper-result extraction.

        Results and evaluation evidence are prioritized over abstracts
        because they are more likely to contain exact numerical metrics.
        """

        if not evidence_sources:
            return "No scientific evidence was provided."

        priority_keywords = (
            "results",
            "result",
            "evaluation",
            "performance",
            "experiment",
            "experimental",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f1-score",
            "table",
            "comparison",
        )

        abstract_keywords = (
            "abstract",
            "summary",
        )

        prioritized_sources = []
        abstract_sources = []
        other_sources = []

        for index, source in enumerate(evidence_sources, start=1):
            if not isinstance(source, dict):
                continue

            title = str(source.get("title") or "")
            section = str(
                source.get("section")
                or source.get("section_title")
                or ""
            )

            content = (
                source.get("content")
                or source.get("text")
                or source.get("snippet")
                or source.get("abstract")
                or source.get("summary")
                or ""
            )

            content = str(content)

            searchable_text = (
                f"{title} {section} {content}"
            ).lower()

            formatted_source = (
                f"Source {index}\n"
                f"Title: {title}\n"
                f"Section: {section}\n"
                f"URL: {source.get('url') or 'N/A'}\n"
                f"Content:\n{content[:20000]}"
            )

            if any(
                keyword in searchable_text
                for keyword in priority_keywords
            ):
                prioritized_sources.append(formatted_source)

            elif any(
                keyword in searchable_text
                for keyword in abstract_keywords
            ):
                abstract_sources.append(formatted_source)

            else:
                other_sources.append(formatted_source)

        ordered_sources = (
            prioritized_sources
            + abstract_sources
            + other_sources
        )

        if not ordered_sources:
            return "No usable scientific evidence was provided."

        return "\n\n---\n\n".join(ordered_sources)

    def _load_paper_chunks(
        self,
        evidence_sources: List[Dict[str, Any]],
    ) -> List[Any]:
        """Load indexed chunks for the evidence sources."""

        chunks = []

        for source in evidence_sources:
            source_id = str(
                source.get("source_id")
                or source.get("id")
                or ""
            ).strip()

            if not source_id:
                continue

            try:
                source_chunks = self.paper_library.get_source_chunks(
                    source_id
                )
                chunks.extend(source_chunks)
            except Exception as error:
                print(
                    f"[ExperimentComparator] "
                    f"Failed to load source {source_id}: {error}"
                )

        return chunks
    
    # ============================================================
    # LLM
    # ============================================================

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Reuse the existing AI Co-Scientist LLM facade.

        No new LLM client is created here.
        """

        from .. import agents as facade

        kwargs: Dict[str, Any] = {}

        if self.llm_model:
            kwargs["model"] = self.llm_model

        try:
            return facade.call_llm(
                user_prompt,
                system_prompt=system_prompt,
                **kwargs,
            )
        except TypeError:
            # Fallback for facades that do not accept model=.
            return facade.call_llm(
                user_prompt,
                system_prompt=system_prompt,
            )

        
    # ============================================================
    # Paper Evidence Loading
    # ============================================================

    def _load_paper_evidence(
        self,
        evidence_sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Load full paper text from evidence sources."""

        loaded_sources = []

        for source in evidence_sources:
            source_copy = dict(source)

            url = self._get_paper_url(source)

            if url:
                try:
                    paper_text = self.paper_reader.read_paper(url)

                    if paper_text:
                        source_copy["content"] = paper_text
                        source_copy["paper_retrieved"] = True
                    else:
                        source_copy["paper_retrieved"] = False
                        source_copy["paper_retrieval_error"] = (
                            "Paper was downloaded but no text was extracted."
                        )

                except Exception as error:
                    source_copy["paper_retrieved"] = False
                    source_copy["paper_retrieval_error"] = str(error)

            loaded_sources.append(source_copy)

        return loaded_sources


    # ============================================================
    # Paper Result Extraction
    # ============================================================

    @staticmethod
    def _get_paper_url(
        source: Dict[str, Any],
    ) -> Optional[str]:
        """
        Get a PDF-compatible URL from an evidence source.

        arXiv evidence may provide an HTML URL such as:
            https://arxiv.org/html/2603.11006v1

        PaperReader expects a PDF URL, so convert arXiv HTML
        URLs to their corresponding PDF URLs.
        """

        if not isinstance(source, dict):
            return None

        url = (
            source.get("pdf_url")
            or source.get("arxiv_pdf_url")
            or source.get("url")
            or source.get("arxiv_url")
        )

        if not url:
            return None

        url = str(url).strip()

        if "arxiv.org/html/" in url:
            url = url.replace(
                "arxiv.org/html/",
                "arxiv.org/pdf/",
            )

        return url


    def _extract_results_from_chunks(
        self,
        chunks: List[Any],
        hypothesis_title: str,
        hypothesis_text: str,
    ) -> Dict[str, Any]:
        """
        Ask the LLM to inspect paper chunks sequentially and extract
        explicitly reported quantitative results.

        All relevant chunks are inspected so that metrics reported
        across different sections or pages can be combined.

        The LLM must not invent or calculate metrics that are not
        explicitly reported in the current chunk.
        """

        if not chunks:
            return {
                "success": False,
                "status": "no_chunks",
                "metrics": {},
                "errors": [
                    "No paper chunks were available for extraction."
                ],
            }

        system_prompt = """
You are a scientific evidence extraction assistant.

Inspect the supplied paper chunk and determine whether it contains
explicit quantitative results relevant to the selected hypothesis.

IMPORTANT RULES:

1. Return ONLY one valid JSON object.

2. Do not return Markdown or explanatory text outside the JSON.

3. Do not invent numerical values.

4. Do not calculate metrics that are not explicitly reported.

5. If the chunk contains methodology, background, or setup only,
return found_relevant_result as false.

6. If the chunk contains numerical results, extract only values
explicitly stated in the chunk.

7. Classification metrics such as accuracy, precision, recall, and
F1-score should only be extracted when explicitly reported.

8. Other quantitative results may include latency, risk, reward,
overhead, recovery time, throughput, or constraint feasibility.

9. The result must be relevant to the selected hypothesis.

10. If multiple metrics are explicitly reported in the same chunk,
extract all of them.

Return exactly this JSON structure:

{
    "found_relevant_result": false,
    "result_type": "none",
    "metrics": {},
    "evidence_quote": null,
    "reason": "",
    "chunk_id": null,
    "page_start": null,
    "page_end": null
}

Possible result_type values:

- "classification_metrics"
- "network_performance"
- "security_metrics"
- "optimization_metrics"
- "comparison"
- "other_quantitative_result"
- "none"
"""

        # Store results from every relevant chunk.
        extracted_results = []

        for chunk_index, chunk in enumerate(chunks):
            chunk_text = str(
                getattr(chunk, "text", "") or ""
            ).strip()

            if not chunk_text:
                continue

            chunk_id = getattr(
                chunk,
                "chunk_id",
                None,
            )

            page_start = getattr(
                chunk,
                "page_start",
                None,
            )

            page_end = getattr(
                chunk,
                "page_end",
                None,
            )

            user_prompt = f"""
Selected Rank #1 hypothesis:

Title:

{hypothesis_title}

Hypothesis:

{hypothesis_text}

Current paper chunk number:

{chunk_index + 1} of {len(chunks)}

Chunk ID:

{chunk_id}

Page range:

{page_start} - {page_end}

Paper chunk:

{chunk_text}

Determine whether this chunk contains explicit quantitative
results relevant to the selected hypothesis.

If it does not, return found_relevant_result as false.

If it does, extract ALL explicitly reported quantitative
results relevant to the selected hypothesis.

Do not calculate or infer missing values.
"""

            try:
                response = self._call_llm(
                    system_prompt,
                    user_prompt,
                )

                parsed = self._extract_json(response)

                if not isinstance(parsed, dict):
                    continue

                found_result = bool(
                    parsed.get(
                        "found_relevant_result",
                        False,
                    )
                )

                if not found_result:
                    continue

                # Preserve actual chunk identity when the LLM
                # omits or changes it.
                parsed["chunk_id"] = (
                    chunk_id
                )

                parsed["page_start"] = (
                    page_start
                )

                parsed["page_end"] = (
                    page_end
                )

                metrics = parsed.get(
                    "metrics",
                    {},
                )

                if not isinstance(metrics, dict):
                    metrics = {}

                parsed["metrics"] = metrics

                extracted_results.append(parsed)

            except Exception as error:
                print(
                    f"[ExperimentComparator] "
                    f"Chunk {chunk_index + 1} extraction failed: "
                    f"{error}"
                )
                continue

        # ========================================================
        # No results found
        # ========================================================

        if not extracted_results:
            return {
                "success": False,
                "status": "no_quantitative_results_found",
                "metrics": {},
                "errors": [
                    "No relevant quantitative results were found "
                    "in the available paper chunks."
                ],
            }

        # ========================================================
        # Combine metrics from all relevant chunks
        # ========================================================

        combined_metrics: Dict[str, Any] = {}

        for result in extracted_results:
            metrics = result.get(
                "metrics",
                {},
            )

            for metric_name, value in metrics.items():
                # Keep the first explicitly reported value for
                # each metric. This avoids silently overwriting
                # conflicting values from different chunks.
                if metric_name not in combined_metrics:
                    combined_metrics[metric_name] = value

        # ========================================================
        # Build combined result
        # ========================================================

        first_result = extracted_results[0]

        combined_result = {
            "success": True,
            "status": "results_found",
            "found_relevant_result": True,
            "result_type": first_result.get(
                "result_type",
                "other_quantitative_result",
            ),
            "metrics": combined_metrics,
            "evidence_quote": first_result.get(
                "evidence_quote"
            ),
            "reason": first_result.get(
                "reason",
                "",
            ),
            "chunk_id": first_result.get(
                "chunk_id"
            ),
            "page_start": first_result.get(
                "page_start"
            ),
            "page_end": first_result.get(
                "page_end"
            ),
            "extracted_chunks": [
                {
                    "chunk_id": result.get(
                        "chunk_id"
                    ),
                    "page_start": result.get(
                        "page_start"
                    ),
                    "page_end": result.get(
                        "page_end"
                    ),
                    "result_type": result.get(
                        "result_type"
                    ),
                    "metrics": result.get(
                        "metrics",
                        {},
                    ),
                    "evidence_quote": result.get(
                        "evidence_quote"
                    ),
                }
                for result in extracted_results
            ],
        }

        return combined_result


    def extract_paper_results(
        self,
        hypothesis: Any,
    ) -> Dict[str, Any]:
        """
        Extract numerical results reported by the paper/evidence
        supporting the selected hypothesis.

        The comparator retrieves indexed PaperChunk objects from
        ChromaPaperLibrary and asks the LLM to extract explicitly
        reported quantitative results.

        The LLM must NOT invent or calculate missing values.
        """

        started = time.perf_counter()

        # ========================================================
        # Get evidence supporting the selected hypothesis
        # ========================================================

        evidence_sources = self._get_hypothesis_evidence(
            hypothesis
        )

        print("\n===== HYPOTHESIS EVIDENCE SOURCES =====")
        print(json.dumps(evidence_sources, indent=2, default=str))
        print("=======================================\n")

        if not evidence_sources:
            return {
                "success": False,
                "status": "no_evidence",
                "comparable": False,
                "metrics": {},
                "errors": [
                    "The selected hypothesis has no attached evidence sources."
                ],
                "extraction_seconds": (
                    time.perf_counter() - started
                ),
            }

        # ========================================================
        # Get hypothesis information
        # ========================================================

        hypothesis_title = self._get_hypothesis_field(
            hypothesis,
            "title",
            "",
        )

        hypothesis_text = self._get_hypothesis_field(
            hypothesis,
            "text",
            "",
        )

        # ========================================================
        # Load indexed paper chunks
        # ========================================================

        chunks = self._load_paper_chunks(
            evidence_sources
        )

        if not chunks:
            return {
                "success": False,
                "status": "no_indexed_chunks",
                "comparable": False,
                "metrics": {},
                "errors": [
                    "No indexed paper chunks were found for "
                    "the evidence sources supporting the hypothesis."
                ],
                "extraction_seconds": (
                    time.perf_counter() - started
                ),
            }

        print("\n===== PAPER CHUNKS USED =====")
        print(f"Number of chunks: {len(chunks)}")

        for chunk in chunks:
            print(
                f"source_id={getattr(chunk, 'source_id', None)}, "
                f"chunk_id={getattr(chunk, 'chunk_id', None)}, "
                f"page={getattr(chunk, 'page', None)}, "
                f"page_start={getattr(chunk, 'page_start', None)}, "
                f"page_end={getattr(chunk, 'page_end', None)}, "
                f"section={getattr(chunk, 'section', None)}"
            )

        print("=============================\n")

        # ========================================================
        # Extract result from paper chunks
        # ========================================================

        try:
            parsed = self._extract_results_from_chunks(
                chunks,
                hypothesis_title,
                hypothesis_text,
            )

            if not isinstance(parsed, dict):
                return {
                    "success": False,
                    "status": "invalid_extraction_result",
                    "comparable": False,
                    "metrics": {},
                    "errors": [
                        "Paper-result extraction did not return "
                        "a valid result."
                    ],
                    "extraction_seconds": (
                        time.perf_counter() - started
                    ),
                }

            # If no quantitative result was found, preserve
            # the status returned by _extract_results_from_chunks().
            if not parsed.get("success", False):
                parsed["extraction_seconds"] = (
                    time.perf_counter() - started
                )
                return parsed

            # ====================================================
            # Normalize comparable metrics
            # ====================================================

            metrics = parsed.get(
                "metrics",
                {},
            )

            if not isinstance(metrics, dict):
                metrics = {}

            normalized_metrics: Dict[str, float] = {}

            for metric_name in self.COMPARABLE_METRICS:
                value = None

                aliases = self.METRIC_ALIASES.get(
                    metric_name,
                    [metric_name],
                )

                for alias in aliases:
                    normalized_alias = (
                        self._normalise_metric_name(alias)
                    )

                    for key, raw_value in metrics.items():
                        normalized_key = (
                            self._normalise_metric_name(key)
                        )

                        if normalized_key == normalized_alias:
                            value = self._safe_float(
                                raw_value
                            )
                            break

                    if value is not None:
                        break

                if value is not None:
                    normalized_metrics[
                        metric_name
                    ] = self._clamp_metric(value)

            parsed["metrics"] = normalized_metrics

            parsed["success"] = True

            parsed["extraction_seconds"] = (
                time.perf_counter() - started
            )

            return parsed

        except Exception as error:
            error_message = str(error)

            if "timed out" in error_message.lower():
                status = "llm_timeout"
                message = (
                    "The paper-result extraction LLM request timed out."
                )
            else:
                status = "llm_error"
                message = error_message

            return {
                "success": False,
                "status": status,
                "comparable": False,
                "metrics": {},
                "errors": [message],
                "extraction_seconds": (
                    time.perf_counter() - started
                ),
            }


    # ============================================================
    # Experiment Result Extraction
    # ============================================================

    def extract_experiment_results(
        self,
        experiment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract the standardized metrics collected by
        ExperimentRunner.
        """

        if not isinstance(
            experiment_result,
            dict,
        ):
            return {
                "success": False,
                "status": "invalid_experiment_result",
                "metrics": {},
                "errors": [
                    "Experiment result must be a dictionary."
                ],
            }

        if not experiment_result.get(
            "success",
            False,
        ):
            return {
                "success": False,
                "status": (
                    experiment_result.get(
                        "status",
                        "experiment_failed",
                    )
                ),
                "metrics": {},
                "errors": (
                    experiment_result.get(
                        "errors",
                        [
                            "Experiment did not complete successfully."
                        ],
                    )
                ),
            }

        outputs = experiment_result.get(
            "outputs"
        )

        if not isinstance(
            outputs,
            dict,
        ):
            return {
                "success": False,
                "status": "missing_outputs",
                "metrics": {},
                "errors": [
                    "ExperimentRunner did not return experiment outputs."
                ],
            }

        metrics = outputs.get(
            "metrics"
        )

        if not isinstance(
            metrics,
            dict,
        ):
            return {
                "success": False,
                "status": "missing_metrics",
                "metrics": {},
                "errors": [
                    "ExperimentRunner did not collect metrics.json."
                ],
            }

        normalized_metrics: Dict[str, float] = {}

        for metric_name in self.COMPARABLE_METRICS:
            value = None

            aliases = self.METRIC_ALIASES.get(
                metric_name,
                [metric_name],
            )

            for alias in aliases:
                normalized_alias = (
                    self._normalise_metric_name(
                        alias
                    )
                )

                for key, raw_value in metrics.items():
                    normalized_key = (
                        self._normalise_metric_name(
                            key
                        )
                    )

                    if (
                        normalized_key
                        == normalized_alias
                    ):
                        value = self._safe_float(
                            raw_value
                        )
                        break

                if value is not None:
                    break

            if value is not None:
                normalized_metrics[
                    metric_name
                ] = self._clamp_metric(
                    value
                )
            
        if not normalized_metrics:
            return {
                "success": False,
                "status": "no_valid_metrics",
                "metrics": {},
                "raw_metrics": metrics,
                "errors": [
                    "ExperimentRunner returned no valid finite "
                    "comparison metrics."
                ],
            }

        return {
            "success": True,
            "status": "completed",
            "metrics": normalized_metrics,
            "raw_metrics": metrics,
            "run_directory": experiment_result.get(
                "run_directory"
            ),
            "metrics_path": outputs.get(
                "metrics_path"
            ),
            "experiment_summary": outputs.get(
                "experiment_summary"
            ),
            "training_history": outputs.get(
                "training_history"
            ),
        }

    # ============================================================
    # Compatibility Check
    # ============================================================

    def check_comparability(
        self,
        paper_result: Dict[str, Any],
        experiment_result: Dict[str, Any],
        hypothesis: Any,
    ) -> Dict[str, Any]:
        """
        Determine whether the paper result and automated experiment
        can reasonably be compared.
        """

        paper_metrics = paper_result.get(
            "metrics",
            {},
        )

        experiment_metrics = experiment_result.get(
            "metrics",
            {},
        )

        common_metrics = sorted(
            set(paper_metrics)
            & set(experiment_metrics)
            & set(self.COMPARABLE_METRICS)
        )

        if not common_metrics:
            return {
                "comparable": False,
                "comparison_level": "none",
                "reason": (
                    "No common numerical evaluation metrics "
                    "were found between the paper and experiment."
                ),
                "common_metrics": [],
                "warnings": [],
            }

        paper_model = str(
            paper_result.get(
                "model_name",
                "",
            )
            or ""
        ).strip()

        paper_dataset = str(
            paper_result.get(
                "dataset",
                "",
            )
            or ""
        ).strip()

        experiment_summary = (
            experiment_result.get(
                "experiment_summary"
            )
            or {}
        )

        experiment_model = ""

        if isinstance(
            experiment_summary,
            dict,
        ):
            experiment_model = str(
                experiment_summary.get(
                    "model",
                    "",
                )
                or experiment_summary.get(
                    "model_name",
                    "",
                )
                or ""
            ).strip()

        hypothesis_text = str(
            self._get_hypothesis_field(
                hypothesis,
                "text",
                "",
            )
            or ""
        )

        # Dataset compatibility is important, but do not reject
        # comparison simply because the paper extractor could not
        # identify the dataset name.
        dataset_warning = None

        if paper_dataset:
            normalized_dataset = paper_dataset.lower()

            if (
                "5g-nidd" not in normalized_dataset
                and "5g nidd" not in normalized_dataset
            ):
                dataset_warning = (
                    "The paper appears to report results using a dataset "
                    "different from the offline 5G-NIDD dataset used by "
                    "the automated experiment."
                )

        comparison_level = (
            "direct"
            if paper_dataset
            and (
                "5g-nidd" in paper_dataset.lower()
                or "5g nidd" in paper_dataset.lower()
            )
            else "partial"
        )

        return {
            "comparable": True,
            "comparison_level": comparison_level,
            "common_metrics": common_metrics,
            "paper_model": paper_model,
            "experiment_model": experiment_model,
            "paper_dataset": paper_dataset,
            "dataset_warning": dataset_warning,
            "hypothesis": hypothesis_text,
        }

    # ============================================================
    # Numerical Comparison
    # ============================================================

    def compare_metrics(
        self,
        paper_result: Dict[str, Any],
        experiment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate metric differences using Python.

        Difference:

            experiment - paper

        Positive values mean the automated experiment achieved
        a higher score.
        """

        paper_metrics = paper_result.get(
            "metrics",
            {},
        )

        experiment_metrics = experiment_result.get(
            "metrics",
            {},
        )

        comparisons: Dict[str, Dict[str, Any]] = {}

        for metric_name in self.COMPARABLE_METRICS:
            paper_value = self._safe_float(
                paper_metrics.get(
                    metric_name
                )
            )

            experiment_value = self._safe_float(
                experiment_metrics.get(
                    metric_name
                )
            )

            if (
                paper_value is None
                or experiment_value is None
            ):
                continue

            difference = (
                experiment_value
                - paper_value
            )

            comparisons[metric_name] = {
                "paper": paper_value,
                "experiment": experiment_value,
                "difference": difference,
                "difference_percentage_points": (
                    difference * 100.0
                ),
                "absolute_difference": abs(
                    difference
                ),
                "higher_than_paper": (
                    difference > 0
                ),
                "same_as_paper": (
                    abs(difference) < 1e-9
                ),
            }

        if not comparisons:
            return {
                "success": False,
                "status": "no_common_metrics",
                "metrics": {},
                "errors": [
                    "No common numerical metrics could be compared."
                ],
            }

        improved = [
            name
            for name, values in comparisons.items()
            if values["difference"] > 0
        ]

        worse = [
            name
            for name, values in comparisons.items()
            if values["difference"] < 0
        ]

        unchanged = [
            name
            for name, values in comparisons.items()
            if values["same_as_paper"]
        ]

        average_difference = (
            sum(
                item["difference"]
                for item in comparisons.values()
            )
            / len(comparisons)
        )

        return {
            "success": True,
            "status": "compared",
            "metrics": comparisons,
            "improved_metrics": improved,
            "worse_metrics": worse,
            "unchanged_metrics": unchanged,
            "average_difference": average_difference,
            "average_difference_percentage_points": (
                average_difference * 100.0
            ),
        }

    # ============================================================
    # LLM Explanation
    # ============================================================

    def explain_difference(
        self,
        hypothesis: Any,
        paper_result: Dict[str, Any],
        experiment_result: Dict[str, Any],
        metric_comparison: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ask the LLM to explain why the automated experiment may
        differ from the paper result.

        The LLM is explicitly prevented from changing numerical
        results calculated by Python.
        """

        hypothesis_title = (
            self._get_hypothesis_field(
                hypothesis,
                "title",
                "",
            )
        )

        hypothesis_text = (
            self._get_hypothesis_field(
                hypothesis,
                "text",
                "",
            )
        )

        system_prompt = """
You are a scientific experiment analysis assistant.

Explain the difference between a published machine-learning
result and an automatically reproduced experiment.

IMPORTANT:

1. Do not invent numerical results.
2. Do not change any metric values supplied to you.
3. Do not claim that the automated experiment reproduced the
   paper exactly unless the evidence supports that conclusion.
4. Consider differences in:
   - dataset version
   - preprocessing
   - train/validation/test split
   - random seed
   - model architecture
   - hyperparameters
   - training duration
   - class distribution
   - feature selection
   - evaluation protocol
   - hardware/software environment
5. If the evidence is insufficient to identify the exact cause,
   explicitly say so.
6. Distinguish confirmed facts from plausible explanations.
7. Return ONLY valid JSON.

Use this schema:

{
    "overall_assessment": "...",
    "reproduction_level": "higher|similar|lower|inconclusive",
    "confirmed_observations": [],
    "possible_explanations": [],
    "limitations": [],
    "recommendation": "..."
}
"""

        user_prompt = f"""
Rank #1 Hypothesis:

Title:
{hypothesis_title}

Hypothesis:
{hypothesis_text}

Published/Paper Result:

{json.dumps(
    self._json_safe(paper_result),
    indent=2,
    ensure_ascii=False,
    default=str,
    allow_nan=False,
)}

Automated Experiment Result:

{json.dumps(
    self._json_safe(experiment_result),
    indent=2,
    ensure_ascii=False,
    default=str,
    allow_nan=False,
)}

Python-calculated Metric Comparison:

{json.dumps(
    self._json_safe(metric_comparison),
    indent=2,
    ensure_ascii=False,
    default=str,
    allow_nan=False,
)}

Explain the difference scientifically.

Do not recalculate or modify the numerical values.
"""

        try:
            response = self._call_llm(
                system_prompt,
                user_prompt,
            )

            print("\n===== COMPARISON EXTRACTION RESPONSE =====")
            print(repr(response))
            print("=====================================\n")

            parsed = self._extract_json(
                response
            )

            if isinstance(
                parsed,
                dict,
            ):
                return {
                    "success": True,
                    "status": "completed",
                    **parsed,
                }

            return {
                "success": False,
                "status": "invalid_llm_response",
                "overall_assessment": (
                    response.strip()
                    if isinstance(
                        response,
                        str,
                    )
                    else ""
                ),
                "confirmed_observations": [],
                "possible_explanations": [],
                "limitations": [],
                "recommendation": "",
            }

        except Exception as error:
            return {
                "success": False,
                "status": "llm_error",
                "overall_assessment": "",
                "confirmed_observations": [],
                "possible_explanations": [],
                "limitations": [],
                "recommendation": "",
                "errors": [
                    str(error)
                ],
            }

    # ============================================================
    # Complete Comparison
    # ============================================================

    def compare(
        self,
        hypothesis: Any,
        experiment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform the complete paper-vs-experiment comparison.

        Parameters
        ----------
        hypothesis:
            The final Rank #1 hypothesis selected by the
            AI Co-Scientist.

        experiment_result:
            The standardized result returned by ExperimentRunner.
        """

        started = time.perf_counter()

        result: Dict[str, Any] = {
            "success": False,
            "status": "not_started",
            "hypothesis_id": self._get_hypothesis_field(
                hypothesis,
                "hypothesis_id",
            ),
            "hypothesis_title": self._get_hypothesis_field(
                hypothesis,
                "title",
            ),
            "paper_result": None,
            "experiment_result": None,
            "comparability": None,
            "metric_comparison": None,
            "explanation": None,
            "errors": [],
            "comparison_seconds": None,
        }

        try:
            # ----------------------------------------------------
            # 1. Validate experiment result
            # ----------------------------------------------------

            extracted_experiment = (
                self.extract_experiment_results(
                    experiment_result
                )
            )

            result[
                "experiment_result"
            ] = extracted_experiment

            if not extracted_experiment.get(
                "success",
                False,
            ):
                result[
                    "status"
                ] = "experiment_result_unavailable"

                result[
                    "errors"
                ].extend(
                    extracted_experiment.get(
                        "errors",
                        [],
                    )
                )

                return result

            # ----------------------------------------------------
            # 2. Extract paper result from hypothesis evidence
            # ----------------------------------------------------

            paper_result = (
                self.extract_paper_results(
                    hypothesis
                )
            )

            result[
                "paper_result"
            ] = paper_result

            if not paper_result.get(
                "success",
                False,
            ):
                result[
                    "status"
                ] = "paper_result_unavailable"

                result[
                    "errors"
                ].extend(
                    paper_result.get(
                        "errors",
                        [],
                    )
                )

                return result

            # ----------------------------------------------------
            # 3. Check compatibility
            # ----------------------------------------------------

            comparability = (
                self.check_comparability(
                    paper_result,
                    extracted_experiment,
                    hypothesis,
                )
            )

            result[
                "comparability"
            ] = comparability

            if not comparability.get(
                "comparable",
                False,
            ):
                result[
                    "status"
                ] = "not_comparable"

                result[
                    "errors"
                ].append(
                    comparability.get(
                        "reason",
                        "The results are not comparable.",
                    )
                )

                return result

            # ----------------------------------------------------
            # 4. Calculate numerical differences
            # ----------------------------------------------------

            metric_comparison = (
                self.compare_metrics(
                    paper_result,
                    extracted_experiment,
                )
            )

            result[
                "metric_comparison"
            ] = metric_comparison

            if not metric_comparison.get(
                "success",
                False,
            ):
                result[
                    "status"
                ] = "no_common_metrics"

                result[
                    "errors"
                ].extend(
                    metric_comparison.get(
                        "errors",
                        [],
                    )
                )

                return result

            # ----------------------------------------------------
            # 5. Scientific explanation
            # ----------------------------------------------------

            explanation = (
                self.explain_difference(
                    hypothesis,
                    paper_result,
                    extracted_experiment,
                    metric_comparison,
                )
            )

            result[
                "explanation"
            ] = explanation

            # ----------------------------------------------------
            # 6. Complete
            # ----------------------------------------------------

            result[
                "success"
            ] = True

            result[
                "status"
            ] = "completed"

        except Exception as error:
            result[
                "status"
            ] = "comparison_error"

            result[
                "errors"
            ].append(
                str(error)
            )

        finally:
            result[
                "comparison_seconds"
            ] = (
                time.perf_counter()
                - started
            )

        return result

    # ============================================================
    # Human-Readable Summary
    # ============================================================

    @staticmethod
    def format_comparison(
        comparison_result: Dict[str, Any],
    ) -> str:
        """
        Convert a comparison result into a simple human-readable
        summary suitable for logs or UI output.
        """

        if not isinstance(
            comparison_result,
            dict,
        ):
            return "Invalid comparison result."

        status = comparison_result.get(
            "status",
            "unknown",
        )

        if status != "completed":
            errors = comparison_result.get(
                "errors",
                [],
            )

            message = (
                f"Experiment comparison status: {status}."
            )

            if errors:
                message += (
                    " "
                    + " ".join(
                        str(error)
                        for error in errors
                    )
                )

            return message

        lines = [
            "Experiment Comparison",
            "=====================",
        ]

        title = comparison_result.get(
            "hypothesis_title"
        )

        if title:
            lines.append(
                f"Rank #1 Hypothesis: {title}"
            )

        metric_comparison = (
            comparison_result.get(
                "metric_comparison",
                {},
            )
        )

        metrics = metric_comparison.get(
            "metrics",
            {},
        )

        for metric_name, values in metrics.items():
            paper_value = (
                values["paper"] * 100
            )

            experiment_value = (
                values["experiment"] * 100
            )

            difference = (
                values[
                    "difference_percentage_points"
                ]
            )

            sign = "+" if difference > 0 else ""

            lines.append(
                f"{metric_name}: "
                f"Paper={paper_value:.2f}%, "
                f"Experiment={experiment_value:.2f}%, "
                f"Difference={sign}{difference:.2f} pp"
            )

        explanation = (
            comparison_result.get(
                "explanation"
            )
            or {}
        )

        assessment = explanation.get(
            "overall_assessment"
        )

        if assessment:
            lines.append("")
            lines.append(
                f"Assessment: {assessment}"
            )

        recommendation = explanation.get(
            "recommendation"
        )

        if recommendation:
            lines.append(
                f"Recommendation: {recommendation}"
            )

        return "\n".join(lines)