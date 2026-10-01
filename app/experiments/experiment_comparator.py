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

    Important design principle:

        The paper's reported evaluation metrics define the reference
        measurements or constraints used for scientific comparison.

    The comparator therefore does NOT restrict comparisons to a fixed
    list such as accuracy/precision/recall/F1. Any finite numerical
    metric reported by both the paper and automated experiment may be
    considered, subject to compatibility checks.
    """

    # ============================================================
    # Metric aliases
    # ============================================================

    # These aliases are used only to normalize common metric names.
    #
    # They are NOT a whitelist of comparable metrics.
    #
    # This allows metrics such as:
    #   - handshake_latency_ms
    #   - certificate_size_bytes
    #   - throughput_mbps
    #   - overhead_percent
    #   - recovery_time_ms
    #   - accuracy
    #   - f1
    #   - reward
    #   - risk
    #
    # to pass through the comparator.
    METRIC_ALIASES = {
        "accuracy": [
            "accuracy",
            "acc",
        ],
        "precision_weighted": [
            "precision_weighted",
            "weighted_precision",
            "weighted precision",
        ],
        "recall_weighted": [
            "recall_weighted",
            "weighted_recall",
            "weighted recall",
        ],
        "f1_weighted": [
            "f1_weighted",
            "weighted_f1",
            "weighted f1",
            "f1-score",
            "f1 score",
        ],
    }

    NON_SCIENTIFIC_TIMING_METRICS = {
        "training_seconds",
        "evaluation_seconds",
        "total_execution_seconds",
        "execution_seconds",
    }

    def __init__(
        self,
        llm_model: Optional[str] = None,
        paper_library: Optional[ChromaPaperLibrary] = None,
    ):
        self.llm_model = llm_model

        self.paper_library = (
            paper_library
            or ChromaPaperLibrary()
        )

        self.paper_reader = PaperReader(
            paper_library=self.paper_library,
            llm_callable=self._call_llm,
        )

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

        # Do not silently interpret booleans as numerical metrics.
        if isinstance(value, bool):
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

        Examples:

            "F1 Score"              -> "f1_score"
            "Weighted F1"           -> "weighted_f1"
            "Handshake Latency"     -> "handshake_latency"
            "Certificate Size (B)"  -> "certificate_size_b"
        """

        if name is None:
            return ""

        return re.sub(
            r"[^a-z0-9]+",
            "_",
            str(name).strip().lower(),
        ).strip("_")

    @classmethod
    def _canonical_metric_name(
        cls,
        name: Any,
    ) -> str:
        """
        Convert common metric aliases to a canonical metric name.

        Unknown metrics are simply normalized and preserved.

        This is intentionally NOT a whitelist.
        """

        normalized = cls._normalise_metric_name(name)

        if not normalized:
            return ""

        for canonical_name, aliases in cls.METRIC_ALIASES.items():
            candidates = [
                canonical_name,
                *aliases,
            ]

            normalized_candidates = {
                cls._normalise_metric_name(alias)
                for alias in candidates
            }

            if normalized in normalized_candidates:
                return canonical_name

        return normalized

    @staticmethod
    def _normalise_metric_value(
        value: Any,
        unit: Optional[str] = None,
        value_type: Optional[str] = None,
        metric_name: Optional[str] = None,
    ) -> Optional[float]:
        """
        Normalize a numerical metric without assuming that every
        metric is a percentage.

        Percentage conversion is performed only when the metric
        metadata explicitly indicates a percentage.

        This prevents errors such as:

            latency = 5.2 ms
            being incorrectly converted to 0.052

        or:

            certificate_size = 1200 bytes
            being incorrectly converted to 12.
        """

        number = ExperimentComparator._safe_float(value)

        if number is None:
            return None

        normalized_unit = (
            str(unit or "")
            .strip()
            .lower()
        )

        normalized_value_type = (
            str(value_type or "")
            .strip()
            .lower()
        )

        percentage_units = {
            "%",
            "percent",
            "percentage",
            "percentage_point",
            "percentage_points",
        }

        percentage_types = {
            "percentage",
            "percent",
            "proportion",
        }

        # if (
        #     normalized_unit in percentage_units
        #     or normalized_value_type in percentage_types
        # ):
        #     if number > 1.0 and number <= 100.0:
        #         return number / 100.0

        normalized_metric_name = (
            ExperimentComparator._normalise_metric_name(
                metric_name
            )
        )

        percentage_metric_names = {
            "accuracy",
            "acc",
            "precision",
            "precision_weighted",
            "weighted_precision",
            "weighted precision",
            "recall",
            "recall_weighted",
            "weighted_recall",
            "weighted recall",
            "f1",
            "f1_score",
            "f1_weighted",
            "weighted_f1",
            "weighted f1",
            "f1-score",
            "f1 score",
        }

        is_percentage = (
            normalized_unit in percentage_units
            or normalized_value_type in percentage_types
            or normalized_metric_name in percentage_metric_names
        )

        if is_percentage and number > 1.0 and number <= 100.0:
            return number / 100.0

        return number

    @staticmethod
    def _normalise_reference_value(
        value: Any,
        metric_name: Optional[str] = None,
        definition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Normalize a structured reference metric while preserving
        its scientific semantic meaning.
        """

        definition = (
            definition
            if isinstance(definition, dict)
            else {}
        )

        if isinstance(value, dict):
            normalized = dict(value)
        else:
            normalized = {
                "value": value,
            }

        unit = (
            normalized.get("unit")
            or definition.get("unit")
            or None
        )

        value_type = str(
            normalized.get("value_type")
            or definition.get("value_type")
            or "measured_value"
        ).strip().lower()

        relation = str(
            normalized.get("relation")
            or definition.get("relation")
            or "exact"
        ).strip().lower()

        allowed_value_types = {
            "measured_value",
            "upper_bound",
            "lower_bound",
            "range",
            "qualitative_result",
            "configuration",
            "unknown",
        }

        allowed_relations = {
            "exact",
            "less_than",
            "less_than_or_equal",
            "greater_than",
            "greater_than_or_equal",
            "range",
            "none",
        }

        if value_type not in allowed_value_types:
            value_type = "unknown"

        if relation not in allowed_relations:
            relation = "none"

        normalized["unit"] = unit
        normalized["value_type"] = value_type
        normalized["relation"] = relation

        if "source_text" not in normalized:
            normalized["source_text"] = (
                definition.get("source_text", "")
                or ""
            )

        # Preserve optional range information.
        for field in (
            "lower_value",
            "upper_value",
            "lower_unit",
            "upper_unit",
        ):
            if field not in normalized and field in definition:
                normalized[field] = definition[field]

        return normalized

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
    def _clamp_metric(
        value: Any,
    ) -> Optional[float]:
        """
        Convert a metric to a finite value and clamp percentage-style
        proportions to the valid [0, 1] range.
        """

        number = ExperimentComparator._safe_float(value)

        if number is None:
            return None

        # Convert percentage-style values such as 95.2 -> 0.952
        if 1.0 < number <= 100.0:
            number /= 100.0

        return max(0.0, min(1.0, number))

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
            "latency",
            "throughput",
            "overhead",
            "certificate",
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

        for index, source in enumerate(
            evidence_sources,
            start=1,
        ):
            if not isinstance(source, dict):
                continue

            title = str(
                source.get("title")
                or ""
            )

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
                prioritized_sources.append(
                    formatted_source
                )

            elif any(
                keyword in searchable_text
                for keyword in abstract_keywords
            ):
                abstract_sources.append(
                    formatted_source
                )

            else:
                other_sources.append(
                    formatted_source
                )

        ordered_sources = (
            prioritized_sources
            + abstract_sources
            + other_sources
        )

        if not ordered_sources:
            return "No usable scientific evidence was provided."

        return "\n\n---\n\n".join(
            ordered_sources
        )

    def _load_paper_chunks(
        self,
        evidence_sources: List[Dict[str, Any]],
    ) -> List[Any]:
        """
        Load indexed chunks for the evidence sources.
        """

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
                source_chunks = (
                    self.paper_library.get_source_chunks(
                        source_id
                    )
                )

                chunks.extend(
                    source_chunks
                )

            except Exception as error:
                print(
                    f"[ExperimentComparator] "
                    f"Failed to load source {source_id}: {error}"
                )

        return chunks

    @staticmethod
    def _normalise_reference_experiment(
        reference_experiment: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Normalize the reference experiment structure produced by
        PaperReader / ExperimentOrchestrator.

        Returns an empty dictionary when no reference experiment
        is available.
        """

        if not isinstance(
            reference_experiment,
            dict,
        ):
            return {}

        if not reference_experiment:
            return {}

        normalized = dict(
            reference_experiment
        )

        sources = normalized.get(
            "sources",
            [],
        )

        if isinstance(
            sources,
            dict,
        ):
            sources = [sources]

        if not isinstance(
            sources,
            list,
        ):
            sources = []

        normalized["sources"] = [
            source
            for source in sources
            if isinstance(source, dict)
        ]

        normalized["available"] = bool(
            normalized.get("available")
            and normalized["sources"]
        )

        normalized["source_count"] = len(
            normalized["sources"]
        )

        return normalized

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
        """
        Load full paper text from evidence sources.
        """

        loaded_sources = []

        for source in evidence_sources:
            source_copy = dict(source)

            url = self._get_paper_url(
                source
            )

            if url:
                try:
                    paper_text = (
                        self.paper_reader.read_paper(
                            url
                        )
                    )

                    if paper_text:
                        source_copy["content"] = (
                            paper_text
                        )

                        source_copy[
                            "paper_retrieved"
                        ] = True

                    else:
                        source_copy[
                            "paper_retrieved"
                        ] = False

                        source_copy[
                            "paper_retrieval_error"
                        ] = (
                            "Paper was downloaded but "
                            "no text was extracted."
                        )

                except Exception as error:
                    source_copy[
                        "paper_retrieved"
                    ] = False

                    source_copy[
                        "paper_retrieval_error"
                    ] = str(error)

            loaded_sources.append(
                source_copy
            )

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

        arXiv evidence may provide:

            https://arxiv.org/html/2603.11006v1

        or:

            https://arxiv.org/abs/2603.11006v1

        Both are converted to the corresponding PDF URL.
        """

        if not isinstance(
            source,
            dict,
        ):
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

        elif "arxiv.org/abs/" in url:
            url = url.replace(
                "arxiv.org/abs/",
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

5. Preserve the numerical value as reported in the paper.

6. Do NOT automatically convert milliseconds, bytes, seconds,
   throughput, rates, percentages, or other units into another unit.

7. If a percentage is explicitly reported, preserve the percentage
   value as the numerical value unless the paper explicitly gives
   another representation.

8. If the chunk contains methodology, background, or setup only,
   return found_relevant_result as false.

9. If the chunk contains numerical results, extract only values
   explicitly stated in the chunk.

10. Extract ANY relevant quantitative evaluation metric, not only
    accuracy, precision, recall, or F1.

11. Other quantitative results may include:
    - latency
    - handshake latency
    - certificate size
    - overhead
    - throughput
    - recovery time
    - response time
    - memory usage
    - computational cost
    - risk
    - reward
    - security rate
    - unauthorized-access rate
    - constraint violations
    - optimization objective
    - other explicitly reported numerical measurements

12. The result must be relevant to the selected hypothesis.

13. If multiple metrics are explicitly reported in the same chunk,
    extract all relevant metrics.

14. If the paper provides a metric with a unit, preserve the unit
    in metric_definitions.

15. If the paper indicates whether higher or lower values are
    desirable, preserve that direction in metric_definitions.

16. Do not infer a direction when the paper does not explicitly
    establish one.

17. Do not combine numbers from unrelated experiments merely because
    they appear in the same chunk.

Return exactly this JSON structure:

{
    "found_relevant_result": false,
    "result_type": "none",
    "metrics": {},
    "metric_definitions": {},
    "evidence_quote": null,
    "reason": "",
    "chunk_id": null,
    "page_start": null,
    "page_end": null
}

The "metrics" object must contain only explicitly reported numerical
values.

Example:

{
    "found_relevant_result": true,
    "result_type": "network_performance",
    "metrics": {
        "handshake_latency_ms": 5.2,
        "certificate_size_bytes": 1200
    },
    "metric_definitions": {
        "handshake_latency_ms": {
            "unit": "ms",
            "value_type": "numeric",
            "direction": "lower_is_better"
        },
        "certificate_size_bytes": {
            "unit": "bytes",
            "value_type": "numeric",
            "direction": "lower_is_better"
        }
    },
    "evidence_quote": "The measured handshake latency was 5.2 ms...",
    "reason": "The chunk reports explicit quantitative evaluation results.",
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

        extracted_results = []

        for chunk_index, chunk in enumerate(
            chunks
        ):
            chunk_text = str(
                getattr(
                    chunk,
                    "text",
                    "",
                )
                or ""
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
evaluation results relevant to the selected hypothesis.

If it does not, return found_relevant_result as false.

If it does, extract ALL explicitly reported quantitative
evaluation results relevant to the selected hypothesis.

Preserve the numerical values and units exactly as reported.

Do not calculate, estimate, infer, or fabricate missing values.
"""

            try:
                response = self._call_llm(
                    system_prompt,
                    user_prompt,
                )

                parsed = self._extract_json(
                    response
                )

                if not isinstance(
                    parsed,
                    dict,
                ):
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

                if not isinstance(
                    metrics,
                    dict,
                ):
                    metrics = {}

                # Keep only finite numerical values.
                normalized_metrics = {}

                metric_definitions = parsed.get(
                    "metric_definitions",
                    {},
                )

                if not isinstance(
                    metric_definitions,
                    dict,
                ):
                    metric_definitions = {}

                for raw_name, raw_value in metrics.items():
                    if not isinstance(
                        raw_name,
                        str,
                    ):
                        continue

                    metric_definition = (
                        metric_definitions.get(
                            raw_name,
                            {},
                        )
                    )

                    if not isinstance(
                        metric_definition,
                        dict,
                    ):
                        metric_definition = {}

                    safe_value = (
                        self._normalise_metric_value(
                            raw_value,
                            unit=metric_definition.get(
                                "unit"
                            ),
                            value_type=metric_definition.get(
                                "value_type"
                            ),
                            metric_name=raw_name,
                        )
                    )

                    if safe_value is None:
                        continue

                    normalized_name = (
                        self._canonical_metric_name(
                            raw_name
                        )
                    )

                    if not normalized_name:
                        continue

                    normalized_metrics[
                        normalized_name
                    ] = safe_value

                    if raw_name != normalized_name:
                        if raw_name in metric_definitions:
                            metric_definitions[
                                normalized_name
                            ] = metric_definitions[
                                raw_name
                            ]

                parsed["metrics"] = (
                    normalized_metrics
                )

                parsed[
                    "metric_definitions"
                ] = metric_definitions

                if not normalized_metrics:
                    continue

                extracted_results.append(
                    parsed
                )

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

        combined_metrics: Dict[str, float] = {}
        combined_definitions: Dict[str, Dict[str, Any]] = {}

        for result in extracted_results:
            metrics = result.get(
                "metrics",
                {},
            )

            definitions = result.get(
                "metric_definitions",
                {},
            )

            if not isinstance(
                definitions,
                dict,
            ):
                definitions = {}

            for metric_name, value in metrics.items():
                if metric_name not in combined_metrics:
                    combined_metrics[
                        metric_name
                    ] = value

                    definition = definitions.get(
                        metric_name
                    )

                    if isinstance(
                        definition,
                        dict,
                    ):
                        combined_definitions[
                            metric_name
                        ] = definition

        # ========================================================
        # Build combined result
        # ========================================================

        first_result = extracted_results[0]

        return {
            "success": True,
            "status": "results_found",
            "found_relevant_result": True,
            "result_type": first_result.get(
                "result_type",
                "other_quantitative_result",
            ),
            "metrics": combined_metrics,
            "metric_definitions": combined_definitions,
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
                    "metric_definitions": result.get(
                        "metric_definitions",
                        {},
                    ),
                    "evidence_quote": result.get(
                        "evidence_quote"
                    ),
                }
                for result in extracted_results
            ],
        }

    def _extract_results_from_reference_experiment(
        self,
        reference_experiment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract paper results from the reference experiment already
        prepared by ExperimentOrchestrator.

        The reference experiment may contain scientific metrics such as:

            accuracy
            F1
            latency
            risk
            reward
            overhead
            recovery time
            throughput
            certificate size
            etc.

        The comparator preserves all finite numerical metrics.

        Comparability is determined later by check_comparability().
        """

        reference_experiment = (
            self._normalise_reference_experiment(
                reference_experiment
            )
        )

        if not reference_experiment.get(
            "available"
        ):
            return {
                "success": False,
                "status": "reference_experiment_unavailable",
                "metrics": {},
                "errors": [
                    "No extracted reference experiment was provided."
                ],
            }

        combined_metrics: Dict[str, float] = {}

        combined_reference_metrics: Dict[
            str,
            Dict[str, Any],
        ] = {}

        combined_definitions: Dict[
            str,
            Dict[str, Any],
        ] = {}

        extracted_sources: List[
            Dict[str, Any]
        ] = []

        combined_reference_conditions: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for source in reference_experiment.get(
            "sources",
            [],
        ):
            details = source.get(
                "experiment_details",
                {},
            )

            if not isinstance(
                details,
                dict,
            ):
                details = {}

            # --------------------------------------------------------
            # Collect explicit reference metrics.
            # --------------------------------------------------------

            source_metrics: Dict[str, Any] = {}

            reference_metrics = details.get(
                "reference_metrics",
                {},
            )

            if isinstance(
                reference_metrics,
                dict,
            ):
                source_metrics.update(
                    reference_metrics
                )

            source_conditions = details.get(
                "reference_conditions",
                {},
            )

            if isinstance(
                source_conditions,
                dict,
            ):
                for condition_name, condition_value in (
                    source_conditions.items()
                ):
                    normalized_condition_name = (
                        self._canonical_metric_name(
                            condition_name
                        )
                    )

                    if not normalized_condition_name:
                        continue

                    if normalized_condition_name not in combined_reference_conditions:
                        combined_reference_conditions[
                            normalized_condition_name
                        ] = condition_value

            # --------------------------------------------------------
            # Some PaperReader versions may store numerical metrics
            # directly inside "metrics".
            # --------------------------------------------------------

            reference_condition_names = set()

            if isinstance(
                source_conditions,
                dict,
            ):
                reference_condition_names = {
                    self._canonical_metric_name(name)
                    for name in source_conditions
                    if self._canonical_metric_name(name)
                }
            
            experiment_metrics = details.get(
                "metrics",
                [],
            )

            if isinstance(
                experiment_metrics,
                dict,
            ):
                for metric_name, value in (
                    experiment_metrics.items()
                ):
                    normalized_metric_name = (
                        self._canonical_metric_name(
                            metric_name
                        )
                    )

                    if normalized_metric_name in reference_condition_names:
                        continue

                    if metric_name not in source_metrics:
                        source_metrics[
                            metric_name
                        ] = value

            # --------------------------------------------------------
            # Metric metadata.
            # --------------------------------------------------------

            metric_definitions = details.get(
                "metric_definitions",
                {},
            )

            if not isinstance(
                metric_definitions,
                dict,
            ):
                metric_definitions = {}

            normalized_source_metrics: Dict[
                str,
                float,
            ] = {}

            normalized_source_definitions: Dict[
                str,
                Dict[str, Any],
            ] = {}

            # --------------------------------------------------------
            # Normalize and preserve all finite numerical metrics.
            # --------------------------------------------------------

            for metric_name, value in (
                source_metrics.items()
            ):
                if not isinstance(
                    metric_name,
                    str,
                ):
                    continue

                normalized_name = (
                    self._canonical_metric_name(
                        metric_name
                    )
                )

                if not normalized_name:
                    continue

                definition = metric_definitions.get(
                    metric_name,
                    metric_definitions.get(
                        normalized_name,
                        {},
                    ),
                )

                if not isinstance(
                    definition,
                    dict,
                ):
                    definition = {}

                reference_value = (
                    self._normalise_reference_value(
                        value,
                        metric_name=metric_name,
                        definition=definition,
                    )
                )

                reference_numeric_value = self._safe_float(
                    reference_value.get("value")
                )

                value_type = reference_value.get(
                    "value_type",
                    "unknown",
                )

                # Configuration and qualitative statements are not numerical
                # evaluation metrics.
                if value_type in {
                    "configuration",
                    "qualitative_result",
                    "unknown",
                }:
                    continue

                if reference_numeric_value is None:
                    continue

                reference_value["value"] = (
                    self._normalise_metric_value(
                        reference_numeric_value,
                        unit=reference_value.get("unit"),
                        value_type=(
                            definition.get("value_type")
                            if definition
                            else None
                        ),
                        metric_name=metric_name,
                    )
                )

                if reference_value["value"] is None:
                    continue

                normalized_source_metrics[
                    normalized_name
                ] = reference_value

                if definition:
                    normalized_source_definitions[
                        normalized_name
                    ] = definition

                # Keep `metrics` numeric for backward compatibility.
                numeric_value = reference_value.get(
                    "value"
                )

                # Preserve the structured semantic reference separately.
                if normalized_name not in combined_reference_metrics:
                    combined_reference_metrics[
                        normalized_name
                    ] = dict(reference_value)

                    if definition:
                        combined_definitions[
                            normalized_name
                        ] = definition

                # Preserve the first explicitly reported numeric value.
                if normalized_name not in combined_metrics:
                    combined_metrics[
                        normalized_name
                    ] = numeric_value

            # --------------------------------------------------------
            # Preserve source-level metadata.
            # --------------------------------------------------------

            extracted_sources.append(
                {
                    "source_url": source.get(
                        "source_url"
                    ),
                    "source_type": source.get(
                        "source_type",
                        "scientific_paper",
                    ),
                    "models": (
                        details.get("models")
                        or details.get("models_or_systems")
                        or []
                    ),
                    "datasets": (
                        details.get("datasets")
                        or details.get("datasets_or_testbeds")
                        or []
                    ),
                    "baselines": details.get("baselines", []),
                    "metrics": details.get(
                        "metrics",
                        [],
                    ),
                    "primary_metrics": details.get(
                        "primary_metrics",
                        [],
                    ),
                    "hyperparameters": details.get(
                        "hyperparameters",
                        {},
                    ),
                    "training_details": details.get(
                        "training_details",
                        {},
                    ),
                    "reference_metrics": reference_metrics,
                    "reference_conditions": details.get(
                        "reference_conditions",
                        {},
                    ),
                    "metric_definitions": metric_definitions,
                    "normalized_metrics": normalized_source_metrics,
                    "normalized_metric_definitions": (
                        normalized_source_definitions
                    ),
                    "results_text": source.get(
                        "results_text",
                        "",
                    ),
                }
            )

        # ------------------------------------------------------------
        # A reference experiment is usable if it contains any valid
        # numerical scientific results.
        #
        # Do NOT require those metrics to be directly comparable here.
        # Comparability is checked separately.
        # ------------------------------------------------------------

        if not combined_metrics:
            return {
                "success": False,
                "status": "no_reference_metrics",
                "metrics": {},
                "metric_definitions": {},
                "sources": extracted_sources,
                "errors": [
                    "The extracted reference experiment did not "
                    "contain usable quantitative metrics."
                ],
            }

        # ------------------------------------------------------------
        # Collect model and dataset information.
        # ------------------------------------------------------------

        models: List[str] = []
        datasets: List[str] = []

        for source in extracted_sources:
            source_models = source.get(
                "models",
                [],
            )

            source_datasets = source.get(
                "datasets",
                [],
            )

            if isinstance(
                source_models,
                str,
            ):
                source_models = [
                    source_models
                ]

            if isinstance(
                source_datasets,
                str,
            ):
                source_datasets = [
                    source_datasets
                ]

            for model in source_models:
                if (
                    model
                    and model not in models
                ):
                    models.append(
                        model
                    )

            for dataset in source_datasets:
                if (
                    dataset
                    and dataset not in datasets
                ):
                    datasets.append(
                        dataset
                    )

        return {
            "success": True,
            "status": "reference_results_available",
            "metrics": combined_metrics,
            "metric_definitions": combined_definitions,
            "reference_metrics": combined_metrics,
            "reference_conditions": combined_reference_conditions,
            "models": models,
            "datasets": datasets,
            "sources": extracted_sources,
        }

    def extract_paper_results(
        self,
        hypothesis: Any,
        reference_experiment: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Extract numerical results reported by the scientific evidence
        supporting the selected hypothesis.

        When a pre-extracted reference experiment is provided by
        ExperimentOrchestrator, use it directly to avoid downloading
        and parsing the same papers again.

        If no usable pre-extracted reference experiment is available,
        fall back to indexed PaperChunk objects from ChromaPaperLibrary.

        The LLM must NOT invent or calculate missing values.
        """

        started = time.perf_counter()

        # ========================================================
        # Use pre-extracted reference experiment when available
        # ========================================================

        if reference_experiment:
            print(
                "\n===== USING PRE-EXTRACTED REFERENCE EXPERIMENT ====="
            )

            reference_result = (
                self._extract_results_from_reference_experiment(
                    reference_experiment
                )
            )

            reference_result[
                "extraction_seconds"
            ] = (
                time.perf_counter()
                - started
            )

            return reference_result

        # ========================================================
        # Get evidence supporting the selected hypothesis
        # ========================================================

        evidence_sources = (
            self._get_hypothesis_evidence(
                hypothesis
            )
        )

        print(
            "\n===== HYPOTHESIS EVIDENCE SOURCES ====="
        )

        print(
            json.dumps(
                evidence_sources,
                indent=2,
                default=str,
            )
        )

        print(
            "=======================================\n"
        )

        if not evidence_sources:
            return {
                "success": False,
                "status": "no_evidence",
                "comparable": False,
                "metrics": {},
                "metric_definitions": {},
                "errors": [
                    "The selected hypothesis has no attached evidence sources."
                ],
                "extraction_seconds": (
                    time.perf_counter()
                    - started
                ),
            }

        # ========================================================
        # Get hypothesis information
        # ========================================================

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
                "metric_definitions": {},
                "errors": [
                    "No indexed paper chunks were found for "
                    "the evidence sources supporting the hypothesis."
                ],
                "extraction_seconds": (
                    time.perf_counter()
                    - started
                ),
            }

        print(
            "\n===== PAPER CHUNKS USED ====="
        )

        print(
            f"Number of chunks: {len(chunks)}"
        )

        for chunk in chunks:
            print(
                f"source_id={getattr(chunk, 'source_id', None)}, "
                f"chunk_id={getattr(chunk, 'chunk_id', None)}, "
                f"page={getattr(chunk, 'page', None)}, "
                f"page_start={getattr(chunk, 'page_start', None)}, "
                f"page_end={getattr(chunk, 'page_end', None)}, "
                f"section={getattr(chunk, 'section', None)}"
            )

        print(
            "=============================\n"
        )

        # ========================================================
        # Extract result from paper chunks
        # ========================================================

        try:
            parsed = (
                self._extract_results_from_chunks(
                    chunks,
                    hypothesis_title,
                    hypothesis_text,
                )
            )

            if not isinstance(
                parsed,
                dict,
            ):
                return {
                    "success": False,
                    "status": "invalid_extraction_result",
                    "comparable": False,
                    "metrics": {},
                    "metric_definitions": {},
                    "errors": [
                        "Paper-result extraction did not return "
                        "a valid result."
                    ],
                    "extraction_seconds": (
                        time.perf_counter()
                        - started
                    ),
                }

            if not parsed.get(
                "success",
                False,
            ):
                parsed[
                    "extraction_seconds"
                ] = (
                    time.perf_counter()
                    - started
                )

                return parsed

            metrics = parsed.get(
                "metrics",
                {},
            )

            if not isinstance(
                metrics,
                dict,
            ):
                metrics = {}

            metric_definitions = parsed.get(
                "metric_definitions",
                {},
            )

            if not isinstance(
                metric_definitions,
                dict,
            ):
                metric_definitions = {}

            normalized_metrics: Dict[
                str,
                float,
            ] = {}

            normalized_definitions: Dict[
                str,
                Dict[str, Any],
            ] = {}

            for raw_name, raw_value in (
                metrics.items()
            ):
                if not isinstance(
                    raw_name,
                    str,
                ):
                    continue

                normalized_name = (
                    self._canonical_metric_name(
                        raw_name
                    )
                )

                if not normalized_name:
                    continue

                definition = metric_definitions.get(
                    raw_name,
                    metric_definitions.get(
                        normalized_name,
                        {},
                    ),
                )

                if not isinstance(
                    definition,
                    dict,
                ):
                    definition = {}

                safe_value = (
                    self._normalise_metric_value(
                        raw_value,
                        unit=definition.get(
                            "unit"
                        ),
                        value_type=definition.get(
                            "value_type"
                        ),
                        metric_name=raw_name,
                    )
                )

                if safe_value is None:
                    continue

                normalized_metrics[
                    normalized_name
                ] = safe_value

                if definition:
                    normalized_definitions[
                        normalized_name
                    ] = definition

            parsed["metrics"] = (
                normalized_metrics
            )

            parsed[
                "metric_definitions"
            ] = normalized_definitions

            if not normalized_metrics:
                return {
                    "success": False,
                    "status": "no_valid_paper_metrics",
                    "comparable": False,
                    "metrics": {},
                    "metric_definitions": {},
                    "errors": [
                        "Paper-result extraction found relevant "
                        "content but no usable numerical metrics."
                    ],
                    "extraction_seconds": (
                        time.perf_counter()
                        - started
                    ),
                }

            parsed["success"] = True

            parsed[
                "extraction_seconds"
            ] = (
                time.perf_counter()
                - started
            )

            return parsed

        except Exception as error:
            error_message = str(
                error
            )

            if (
                "timed out"
                in error_message.lower()
            ):
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
                "metric_definitions": {},
                "errors": [
                    message
                ],
                "extraction_seconds": (
                    time.perf_counter()
                    - started
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
        Extract numerical metrics collected by ExperimentRunner.

        Unlike the previous implementation, this method does not
        restrict experiment results to accuracy/precision/recall/F1.

        Any finite numerical metric returned by ExperimentRunner is
        preserved.

        This is necessary because the experiment should eventually
        reproduce the evaluation metrics reported by the supporting
        paper.
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

        # Optional metric metadata produced by the experiment.
        metric_definitions = outputs.get(
            "metric_definitions",
            experiment_result.get(
                "metric_definitions",
                {},
            ),
        )

        if not isinstance(
            metric_definitions,
            dict,
        ):
            metric_definitions = {}

        normalized_metrics: Dict[
            str,
            float,
        ] = {}

        normalized_definitions: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for raw_name, raw_value in (
            metrics.items()
        ):
            if not isinstance(
                raw_name,
                str,
            ):
                continue

            normalized_name = (
                self._canonical_metric_name(
                    raw_name
                )
            )

            if normalized_name in self.NON_SCIENTIFIC_TIMING_METRICS:
                continue

            definition = metric_definitions.get(
                raw_name,
                metric_definitions.get(
                    normalized_name,
                    {},
                ),
            )

            if not isinstance(
                definition,
                dict,
            ):
                definition = {}

            safe_value = (
                self._normalise_metric_value(
                    raw_value,
                    unit=definition.get(
                        "unit"
                    ),
                    value_type=definition.get(
                        "value_type"
                    ),
                    metric_name=raw_name,
                )
            )

            if safe_value is None:
                continue

            normalized_metrics[
                normalized_name
            ] = safe_value

            if definition:
                normalized_definitions[
                    normalized_name
                ] = definition

        if not normalized_metrics:
            return {
                "success": False,
                "status": "no_valid_metrics",
                "metrics": {},
                "raw_metrics": metrics,
                "errors": [
                    "ExperimentRunner returned no valid finite "
                    "numerical metrics."
                ],
            }

        return {
            "success": True,
            "status": "completed",
            "metrics": normalized_metrics,
            "metric_definitions": normalized_definitions,
            "raw_metrics": metrics,
            "run_directory": experiment_result.get(
                "run_directory"
            ),
            "metrics_path": outputs.get(
                "metrics_path"
            ),
            "experiment_summary": (
                outputs.get("experiment_summary")
                or outputs.get("summary")
            ),
            "training_history": outputs.get(
                "training_history"
            ),
            "checkpoint": (
                outputs.get("checkpoint")
                or outputs.get("checkpoint_path")
            ),
        }

    # ============================================================
    # Compatibility Check
    # ============================================================

    @staticmethod
    def _normalise_unit(
        unit: Any,
    ) -> str:
        """
        Normalize a metric unit for compatibility checks.
        """

        if unit is None:
            return ""

        return re.sub(
            r"[^a-z0-9%]+",
            "",
            str(unit).strip().lower(),
        )

    def _check_metric_units(
        self,
        common_metrics: List[str],
        paper_definitions: Dict[str, Any],
        experiment_definitions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check whether metric units are compatible.

        Missing units do not automatically make a metric invalid.
        Instead, they generate a warning.

        Explicitly different units cause that metric to be excluded
        from the directly comparable set.
        """

        comparable_metrics = []
        incompatible_metrics = []
        warnings = []

        for metric_name in common_metrics:
            paper_definition = (
                paper_definitions.get(
                    metric_name,
                    {},
                )
            )

            experiment_definition = (
                experiment_definitions.get(
                    metric_name,
                    {},
                )
            )

            if not isinstance(
                paper_definition,
                dict,
            ):
                paper_definition = {}

            if not isinstance(
                experiment_definition,
                dict,
            ):
                experiment_definition = {}

            paper_unit = self._normalise_unit(
                paper_definition.get(
                    "unit"
                )
            )

            experiment_unit = self._normalise_unit(
                experiment_definition.get(
                    "unit"
                )
            )

            if (
                paper_unit
                and experiment_unit
                and paper_unit != experiment_unit
            ):
                incompatible_metrics.append(
                    metric_name
                )

                warnings.append(
                    f"Metric '{metric_name}' has incompatible "
                    f"units: paper='{paper_unit}', "
                    f"experiment='{experiment_unit}'."
                )

                continue

            if (
                not paper_unit
                or not experiment_unit
            ):
                warnings.append(
                    f"Metric '{metric_name}' does not have "
                    "complete unit metadata; numerical comparison "
                    "assumes the values use compatible units."
                )

            comparable_metrics.append(
                metric_name
            )

        return {
            "comparable_metrics": comparable_metrics,
            "incompatible_metrics": incompatible_metrics,
            "warnings": warnings,
        }

    def check_comparability(
        self,
        paper_result: Dict[str, Any],
        experiment_result: Dict[str, Any],
        hypothesis: Any,
    ) -> Dict[str, Any]:
        """
        Determine whether the paper result and automated experiment
        can reasonably be compared.

        A metric is considered a candidate when it is reported by
        both the paper and experiment.

        The method does NOT use a fixed metric whitelist.
        """

        paper_metrics = paper_result.get(
            "metrics",
            {},
        )

        experiment_metrics = experiment_result.get(
            "metrics",
            {},
        )

        if not isinstance(
            paper_metrics,
            dict,
        ):
            paper_metrics = {}

        if not isinstance(
            experiment_metrics,
            dict,
        ):
            experiment_metrics = {}

        common_metrics = sorted(
            set(paper_metrics)
            & set(experiment_metrics)
        )

        paper_definitions = paper_result.get(
            "metric_definitions",
            {},
        )

        reference_metrics = paper_result.get(
            "reference_metrics",
            {},
        )

        if not isinstance(
            reference_metrics,
            dict,
        ):
            reference_metrics = {}

        valid_paper_metrics = set()

        for metric_name in paper_metrics:
            reference_value = (
                reference_metrics.get(
                    metric_name
                )
            )

            if isinstance(
                reference_value,
                dict,
            ):
                value_type = str(
                    reference_value.get(
                        "value_type",
                        "",
                    )
                    or ""
                ).strip().lower()

                if value_type in {
                    "configuration",
                    "qualitative_result",
                    "unknown",
                }:
                    continue

            valid_paper_metrics.add(
                metric_name
            )

        common_metrics = sorted(
            valid_paper_metrics
            & set(experiment_metrics)
        )

        experiment_definitions = (
            experiment_result.get(
                "metric_definitions",
                {},
            )
        )

        if not isinstance(
            paper_definitions,
            dict,
        ):
            paper_definitions = {}

        if not isinstance(
            experiment_definitions,
            dict,
        ):
            experiment_definitions = {}

        unit_check = (
            self._check_metric_units(
                common_metrics,
                paper_definitions,
                experiment_definitions,
            )
        )

        comparable_metrics = (
            unit_check[
                "comparable_metrics"
            ]
        )

        incompatible_metrics = (
            unit_check[
                "incompatible_metrics"
            ]
        )

        warnings = list(
            unit_check[
                "warnings"
            ]
        )

        if not comparable_metrics:
            reason = (
                "No common numerical evaluation metrics "
                "with compatible units were found between "
                "the paper and experiment."
            )

            if incompatible_metrics:
                reason += (
                    " The common metrics had incompatible "
                    "units."
                )

            return {
                "comparable": False,
                "comparison_level": "none",
                "reason": reason,
                "common_metrics": [],
                "incompatible_metrics": (
                    incompatible_metrics
                ),
                "warnings": warnings,
            }

        # --------------------------------------------------------
        # Support both the new PaperReader format and the
        # legacy paper-result format used by existing tests.
        # --------------------------------------------------------

        paper_models = paper_result.get(
            "models",
            [],
        )

        if not paper_models:
            legacy_model = paper_result.get(
                "model_name"
            )

            if legacy_model:
                paper_models = [
                    legacy_model
                ]

        if isinstance(
            paper_models,
            str,
        ):
            paper_models = [
                paper_models
            ]

        paper_datasets = paper_result.get(
            "datasets",
            [],
        )

        if not paper_datasets:
            legacy_dataset = paper_result.get(
                "dataset"
            )

            if legacy_dataset:
                paper_datasets = [
                    legacy_dataset
                ]

        if isinstance(
            paper_datasets,
            str,
        ):
            paper_datasets = [
                paper_datasets
            ]

        paper_model = ", ".join(
            str(model)
            for model in paper_models
            if model
        )

        paper_dataset = ", ".join(
            str(dataset)
            for dataset in paper_datasets
            if dataset
        )

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

        experiment_dataset = ""

        if isinstance(
            experiment_summary,
            dict,
        ):
            experiment_dataset = str(
                experiment_summary.get(
                    "dataset",
                    "",
                )
                or experiment_summary.get(
                    "dataset_name",
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

        # --------------------------------------------------------
        # Dataset compatibility
        # --------------------------------------------------------

        dataset_warning = None

        if paper_dataset:
            normalized_dataset = (
                paper_dataset.lower()
            )

            if (
                "5g-nidd"
                not in normalized_dataset
                and "5g nidd"
                not in normalized_dataset
            ):
                dataset_warning = (
                    "The paper appears to report results using "
                    "a dataset or testbed different from the "
                    "offline 5G-NIDD dataset used by the "
                    "automated experiment."
                )

                warnings.append(
                    dataset_warning
                )

        # --------------------------------------------------------
        # Comparison level
        #
        # Same dataset does not automatically mean identical
        # experimental methodology. "direct" therefore means
        # dataset compatibility only, not full reproduction.
        # --------------------------------------------------------

        comparison_level = (
            "direct"
            if paper_dataset
            and (
                "5g-nidd"
                in paper_dataset.lower()
                or "5g nidd"
                in paper_dataset.lower()
            )
            else "partial"
        )

        return {
            "comparable": bool(
                comparable_metrics
            ),
            "common_metrics": comparable_metrics,
            "incompatible_metrics": (
                incompatible_metrics
            ),
            "paper_model": paper_model,
            "experiment_model": experiment_model,
            "paper_dataset": paper_dataset,
            "experiment_dataset": experiment_dataset,
            "comparison_level": comparison_level,
            "dataset_warning": dataset_warning,
            "warnings": warnings,
            "hypothesis": hypothesis_text,
        }

    # ============================================================
    # Numerical Comparison
    # ============================================================

    @staticmethod
    def _evaluate_reference_constraint(
        experiment_value: float,
        reference_value: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate an automated measurement against a semantic
        reference value.

        The reference may represent:

            exact measured value
            upper bound
            lower bound
            range

        Returns a structured comparison without treating bounds
        as exact measurements.
        """

        reference_numeric = (
            ExperimentComparator._safe_float(
                reference_value.get("value")
            )
        )

        value_type = str(
            reference_value.get(
                "value_type",
                "unknown",
            )
            or "unknown"
        ).strip().lower()

        relation = str(
            reference_value.get(
                "relation",
                "exact",
            )
            or "exact"
        ).strip().lower()

        result = {
            "reference_value": reference_numeric,
            "reference_value_type": value_type,
            "reference_relation": relation,
            "constraint_satisfied": None,
            "difference_from_reference": None,
            "comparison_interpretation": "",
        }

        if reference_numeric is None:
            result[
                "comparison_interpretation"
            ] = (
                "The reference value is not numerical and "
                "cannot be numerically evaluated."
            )

            return result

        if value_type == "upper_bound":
            if relation == "less_than":
                satisfied = (
                    experiment_value
                    < reference_numeric
                )

            else:
                satisfied = (
                    experiment_value
                    <= reference_numeric
                )

            result[
                "constraint_satisfied"
            ] = satisfied

            result[
                "difference_from_reference"
            ] = (
                experiment_value
                - reference_numeric
            )

            result[
                "comparison_interpretation"
            ] = (
                "The automated measurement was evaluated "
                "against the reported upper bound; the bound "
                "was not treated as the paper's exact measurement."
            )

            return result

        if value_type == "lower_bound":
            if relation == "greater_than":
                satisfied = (
                    experiment_value
                    > reference_numeric
                )

            else:
                satisfied = (
                    experiment_value
                    >= reference_numeric
                )

            result[
                "constraint_satisfied"
            ] = satisfied

            result[
                "difference_from_reference"
            ] = (
                experiment_value
                - reference_numeric
            )

            result[
                "comparison_interpretation"
            ] = (
                "The automated measurement was evaluated "
                "against the reported lower bound; the bound "
                "was not treated as the paper's exact measurement."
            )

            return result

        if value_type == "range":
            lower = (
                ExperimentComparator._safe_float(
                    reference_value.get(
                        "lower_value"
                    )
                )
            )

            upper = (
                ExperimentComparator._safe_float(
                    reference_value.get(
                        "upper_value"
                    )
                ))

            if (
                lower is not None
                and upper is not None
            ):
                satisfied = (
                    lower
                    <= experiment_value
                    <= upper
                )

                result[
                    "constraint_satisfied"
                ] = satisfied

                result[
                    "comparison_interpretation"
                ] = (
                    "The automated measurement was evaluated "
                    "for inclusion within the reported reference range."
                )

                return result

        if value_type == "measured_value":
            difference = (
                experiment_value
                - reference_numeric
            )

            result[
                "difference_from_reference"
            ] = difference

            result[
                "constraint_satisfied"
            ] = None

            result[
                "comparison_interpretation"
            ] = (
                "The automated measurement was compared "
                "numerically with the explicitly reported "
                "reference measurement. The reference value "
                "was not treated as a pass/fail constraint."
            )

            return result

        result[
            "comparison_interpretation"
        ] = (
            "The semantic type of the reference value does not "
            "support a direct numerical comparison."
        )

        return result

    def compare_metrics(
        self,
        paper_result: Dict[str, Any],
        experiment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate metric differences using Python.

        Difference:

            experiment - paper

        For arbitrary numerical metrics:

            difference
                = experiment - paper

            relative_difference
                = (experiment - paper) / paper

        The comparator does not assume that every metric is a
        percentage.

        Examples:

            latency:
                paper = 5.2 ms
                experiment = 5.7 ms
                difference = +0.5 ms

            accuracy:
                paper = 0.88
                experiment = 0.91
                difference = +0.03
        """

        paper_metrics = paper_result.get(
            "metrics",
            {},
        )

        experiment_metrics = experiment_result.get(
            "metrics",
            {},
        )

        if not isinstance(
            paper_metrics,
            dict,
        ):
            paper_metrics = {}

        if not isinstance(
            experiment_metrics,
            dict,
        ):
            experiment_metrics = {}

        common_metrics = sorted(
            set(paper_metrics)
            & set(experiment_metrics)
        )

        comparisons: Dict[
            str,
            Dict[str, Any],
        ] = {}

        paper_definitions = paper_result.get(
            "metric_definitions",
            {},
        )

        if not isinstance(
            paper_definitions,
            dict,
        ):
            paper_definitions = {}

        reference_metrics = paper_result.get(
                "reference_metrics",
                {},
            )

        if not isinstance(
            reference_metrics,
            dict,
        ):
            reference_metrics = {}

        comparisons = {}

        for metric_name in common_metrics:
            paper_definition = (
                paper_definitions.get(
                    metric_name,
                    {},
                )
            )

            if not isinstance(
                paper_definition,
                dict,
            ):
                paper_definition = {}

            # Prefer structured semantic reference information.
            reference_value = (
                reference_metrics.get(
                    metric_name
                )
            )

            if isinstance(
                reference_value,
                dict,
            ):
                reference_value = (
                    self._normalise_reference_value(
                        reference_value,
                        metric_name=metric_name,
                        definition=paper_definition,
                    )
                )
            else:
                reference_value = {
                    "value": paper_metrics.get(
                        metric_name
                    ),
                    "unit": paper_definition.get(
                        "unit"
                    ),
                    "value_type": (
                        paper_definition.get(
                            "value_type"
                        )
                        or "measured_value"
                    ),
                    "relation": (
                        paper_definition.get(
                            "relation"
                        )
                        or "exact"
                    ),
                    "source_text": (
                        paper_definition.get(
                            "source_text",
                            "",
                        )
                    ),
                }

            reference_numeric = (
                self._safe_float(
                    reference_value.get(
                        "value"
                    )
                )
            )

            experiment_value = (
                self._safe_float(
                    experiment_metrics.get(
                        metric_name
                    )
                )
            )

            if (
                reference_numeric is None
                or experiment_value is None
            ):
                continue

            semantic_comparison = (
                self._evaluate_reference_constraint(
                    experiment_value,
                    reference_value,
                )
            )

            value_type = (
                reference_value.get(
                    "value_type",
                    "measured_value",
                )
            )

            relation = (
                reference_value.get(
                    "relation",
                    "exact",
                )
            )

            difference = (
                semantic_comparison.get(
                    "difference_from_reference"
                )
            )

            relative_difference = None

            if (
                difference is not None
                and abs(reference_numeric) > 1e-12
            ):
                relative_difference = (
                    difference
                    / reference_numeric
                )

            difference_percentage_points = None

            unit = str(
                paper_definition.get(
                    "unit",
                    reference_value.get(
                        "unit",
                        "",
                    ),
                )
                or ""
            ).strip().lower()

            value_type_normalized = str(
                value_type or ""
            ).strip().lower()

            is_percentage = (
                unit in {
                    "%",
                    "percent",
                    "percentage",
                    "percentage_point",
                    "percentage_points",
                }
                or value_type_normalized in {
                    "percentage",
                    "percent",
                    "proportion",
                }
                or metric_name in {
                    "accuracy",
                    "precision",
                    "precision_weighted",
                    "recall",
                    "recall_weighted",
                    "f1",
                    "f1_score",
                    "f1_weighted",
                }
            )

            if (
                is_percentage
                and difference is not None
            ):
                difference_percentage_points = (
                    difference * 100.0
                )

            comparisons[
                metric_name
            ] = {
                "paper": reference_numeric,
                "experiment": experiment_value,
                "difference": difference,
                "absolute_difference": (
                    abs(difference)
                    if difference is not None
                    else None
                ),
                "relative_difference": (
                    relative_difference
                ),
                "relative_difference_percent": (
                    relative_difference * 100.0
                    if relative_difference is not None
                    else None
                ),
                "higher_than_paper": (
                    experiment_value
                    > reference_numeric
                ),
                "same_as_paper": (
                    abs(difference) < 1e-9
                    if difference is not None
                    else False
                ),
                "difference_percentage_points": (
                    difference_percentage_points
                ),

                # NEW semantic information
                "reference_value": reference_numeric,
                "reference_value_type": value_type,
                "reference_relation": relation,
                "reference_unit": (
                    reference_value.get(
                        "unit"
                    )
                ),
                "reference_lower_value": (
                    reference_value.get(
                        "lower_value"
                    )
                ),
                "reference_upper_value": (
                    reference_value.get(
                        "upper_value"
                    )
                ),
                "constraint_satisfied": (
                    semantic_comparison.get(
                        "constraint_satisfied"
                    )
                ),
                "comparison_interpretation": (
                    semantic_comparison.get(
                        "comparison_interpretation",
                        "",
                    )
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

        satisfied = [
            name
            for name, values in comparisons.items()
            if values.get(
                "constraint_satisfied"
            ) is True
        ]

        not_satisfied = [
            name
            for name, values in comparisons.items()
            if values.get(
                "constraint_satisfied"
            ) is False
        ]

        inconclusive = [
            name
            for name, values in comparisons.items()
            if values.get(
                "constraint_satisfied"
            ) is None
        ]

        average_difference = (
            sum(
                item[
                    "difference"
                ]
                for item in comparisons.values()
            )
            / len(comparisons)
        )

        return {
            "success": True,
            "status": "compared",
            "metrics": comparisons,
            "satisfied_metrics": satisfied,
            "not_satisfied_metrics": not_satisfied,
            "inconclusive_metrics": inconclusive,
            "average_difference": average_difference,
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

Explain the difference between a published scientific result
and an automatically reproduced experiment.

IMPORTANT:

1. Do not invent numerical results.

2. Do not change any metric values supplied to you.

3. Do not claim that the automated experiment reproduced the
   paper exactly unless the evidence supports that conclusion.

4. Do not assume that higher values are always better.
   The meaning depends on the metric.

5. Consider metric direction when it is explicitly supplied:
   - higher_is_better
   - lower_is_better

5A. Reference metrics may represent different semantic types:

    - measured_value
    - upper_bound
    - lower_bound
    - range
    - qualitative_result

    Do NOT interpret an upper/lower bound as an exact published
    measurement.

    For example:

        paper latency < 80 ms

    means the paper reported an upper bound of 80 ms, not that
    the paper measured exactly 80 ms.

6. Reference conditions such as number of users, UE count, batch
   size, hardware, or testbed configuration are experimental
   conditions, not performance measurements.

7. If a comparison is against a bound, explain whether the automated
   measurement satisfies or violates the reported constraint.

8. Do not describe a bound comparison as a numerical difference
   between two exact measurements.

9. Consider differences in:
   - dataset version
   - dataset source
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
   - implementation details
   - measurement methodology

10. If the paper metric and automated metric are not measuring
   exactly the same construct, identify that limitation.

11. If the evidence is insufficient to identify the exact cause,
   explicitly say so.

12. Distinguish confirmed facts from plausible explanations.

13. Do not describe an experiment as a faithful reproduction
    when it uses a substantially different evaluation protocol.

14. Return ONLY valid JSON.

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
    self._json_safe(
        paper_result
    ),
    indent=2,
    ensure_ascii=False,
    default=str,
    allow_nan=False,
)}

Automated Experiment Result:

{json.dumps(
    self._json_safe(
        experiment_result
    ),
    indent=2,
    ensure_ascii=False,
    default=str,
    allow_nan=False,
)}

Python-calculated Metric Comparison:

{json.dumps(
    self._json_safe(
        metric_comparison
    ),
    indent=2,
    ensure_ascii=False,
    default=str,
    allow_nan=False,
)}

Explain the difference scientifically.

Do not recalculate or modify the numerical values.

Do not invent missing experimental details.
"""

        try:
            response = self._call_llm(
                system_prompt,
                user_prompt,
            )

            print(
                "\n===== COMPARISON EXTRACTION RESPONSE ====="
            )

            print(
                repr(response)
            )

            print(
                "=====================================\n"
            )

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
        reference_experiment: Optional[
            Dict[str, Any]
        ] = None,
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

        reference_experiment:
            Optional paper experiment information already extracted
            by ExperimentOrchestrator / PaperReader.
        """

        started = time.perf_counter()

        result: Dict[str, Any] = {
            "success": False,
            "status": "not_started",
            "hypothesis_id": (
                self._get_hypothesis_field(
                    hypothesis,
                    "hypothesis_id",
                )
            ),
            "hypothesis_title": (
                self._get_hypothesis_field(
                    hypothesis,
                    "title",
                )
            ),
            "paper_result": None,
            "reference_experiment": None,
            "experiment_result": None,
            "comparability": None,
            "metric_comparison": None,
            "explanation": None,
            "errors": [],
            "comparison_seconds": None,
        }

        try:
            result[
                "reference_experiment"
            ] = (
                self._normalise_reference_experiment(
                    reference_experiment
                )
                if reference_experiment
                else None
            )

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
                    hypothesis,
                    reference_experiment=(
                        reference_experiment
                    ),
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

                # The experiment completed and the paper was read
                # successfully, so this is a valid scientific
                # comparison outcome rather than an execution error.
                result[
                    "success"
                ] = True

                result[
                    "reason"
                ] = comparability.get(
                    "reason",
                    "The paper and automated experiment "
                    "do not report compatible metrics.",
                )

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

        This method does not assume that all metrics are percentages.
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

        paper_result = (
            comparison_result.get(
                "paper_result",
                {},
            )
            or {}
        )

        experiment_result = (
            comparison_result.get(
                "experiment_result",
                {},
            )
            or {}
        )

        paper_definitions = paper_result.get(
            "metric_definitions",
            {},
        )

        experiment_definitions = (
            experiment_result.get(
                "metric_definitions",
                {},
            )
        )

        if not isinstance(
            paper_definitions,
            dict,
        ):
            paper_definitions = {}

        if not isinstance(
            experiment_definitions,
            dict,
        ):
            experiment_definitions = {}

        for metric_name, values in (
            metrics.items()
        ):
            paper_value = values.get(
                "paper"
            )

            experiment_value = values.get(
                "experiment"
            )

            difference = values.get(
                "difference"
            )

            relative_difference = values.get(
                "relative_difference_percent"
            )

            definition = paper_definitions.get(
                metric_name,
                {},
            )

            if not isinstance(
                definition,
                dict,
            ):
                definition = {}

            unit = (
                definition.get(
                    "unit"
                )
                or experiment_definitions.get(
                    metric_name,
                    {},
                ).get(
                    "unit"
                )
                if isinstance(
                    experiment_definitions.get(
                        metric_name,
                        {},
                    ),
                    dict,
                )
                else definition.get(
                    "unit"
                )
            )

            unit_text = (
                f" {unit}"
                if unit
                else ""
            )

            reference_value_type = str(
                values.get(
                    "reference_value_type",
                    "",
                )
                or ""
            ).strip().lower()

            reference_relation = str(
                values.get(
                    "reference_relation",
                    "",
                )
                or ""
            ).strip().lower()

            constraint_satisfied = values.get(
                "constraint_satisfied"
            )

            reference_unit = (
                values.get(
                    "reference_unit"
                )
                or ""
            )

            if reference_value_type in {
                "upper_bound",
                "lower_bound",
                "range",
            }:
                reference_value = values.get(
                    "reference_value"
                )

                if reference_value_type == "upper_bound":
                    relation_text = (
                        "<"
                        if reference_relation == "less_than"
                        else "<="
                    )

                    reference_text = (
                        f"{relation_text} "
                        f"{reference_value:g}"
                    )

                elif reference_value_type == "lower_bound":
                    relation_text = (
                        ">"
                        if reference_relation == "greater_than"
                        else ">="
                    )

                    reference_text = (
                        f"{relation_text} "
                        f"{reference_value:g}"
                    )

                else:
                    lower_value = (
                        values.get(
                            "reference_lower_value"
                        )
                    )

                    upper_value = (
                        values.get(
                            "reference_upper_value"
                        )
                    )

                    if (
                        lower_value is not None
                        and upper_value is not None
                    ):
                        reference_text = (
                            f"{lower_value:g}–"
                            f"{upper_value:g}"
                        )
                    else:
                        reference_text = (
                            f"range around "
                            f"{reference_value:g}"
                        )

                if reference_unit:
                    reference_text += (
                        f" {reference_unit}"
                    )

                if constraint_satisfied is True:
                    constraint_text = "Satisfied"
                elif constraint_satisfied is False:
                    constraint_text = "Not satisfied"
                else:
                    constraint_text = "Inconclusive"

                lines.append(
                    f"{metric_name}: "
                    f"Reference={reference_text}, "
                    f"Experiment={experiment_value}, "
                    f"Constraint={constraint_text}"
                )

                continue

            is_percentage = (
                metric_name in {
                    "accuracy",
                    "precision",
                    "precision_weighted",
                    "recall",
                    "recall_weighted",
                    "f1",
                    "f1_score",
                    "f1_weighted",
                }
                or str(
                    definition.get("unit", "")
                    or ""
                ).lower() in {
                    "%",
                    "percent",
                    "percentage",
                    "percentage_point",
                    "percentage_points",
                }
                or str(
                    definition.get("value_type", "")
                    or ""
                ).lower() in {
                    "percentage",
                    "percent",
                    "proportion",
                }
            )

            if is_percentage:
                paper_text = (
                    f"{paper_value * 100:.2f}%"
                    if paper_value is not None
                    else "N/A"
                )

                experiment_text = (
                    f"{experiment_value * 100:.2f}%"
                    if experiment_value is not None
                    else "N/A"
                )

                difference_percentage_points = values.get(
                    "difference_percentage_points"
                )

                if difference_percentage_points is not None:
                    difference_text = (
                        f"{difference_percentage_points:.2f} pp"
                    )
                elif difference is not None:
                    difference_text = (
                        f"{difference * 100:.2f} pp"
                    )
                else:
                    difference_text = "N/A"

                unit_text = ""

            else:
                paper_text = (
                    f"{paper_value:.4f}"
                    if paper_value is not None
                    else "N/A"
                )

                experiment_text = (
                    f"{experiment_value:.4f}"
                    if experiment_value is not None
                    else "N/A"
                )

                difference_text = (
                    f"{difference:+.4f}"
                    if difference is not None
                    else "N/A"
                )

            if relative_difference is not None:
                relative_text = (
                    f"{relative_difference:+.2f}%"
                )
            else:
                relative_text = "N/A"

            lines.append(
                f"{metric_name}: "
                f"Paper={paper_text}{unit_text}, "
                f"Experiment={experiment_text}{unit_text}, "
                f"Difference={difference_text}{unit_text}, "
                f"Relative={relative_text}"
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