"""
Paper Reader

Retrieves research papers from evidence URLs and extracts relevant
experimental/evaluation evidence and structured experiment details
for use by the automated experiment pipeline.

Workflow:

    Evidence URL
        |
        v
    Resolve PDF URL
        |
        v
    Download PDF
        |
        v
    Extract text
        |
        v
    Identify experimental/evaluation content
        |
        v
    Extract structured experiment details
        |
        +--> metrics
        +--> metric_definitions
        +--> reference_metrics
        +--> results
"""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Any, Dict, List, Optional
from html import unescape
from urllib.parse import urljoin, urlparse

import requests
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PaperReader:
    """
    Reads scientific papers and extracts experimental evidence.

    The structured output is intentionally metric-agnostic.  The reader
    does not assume that papers only evaluate accuracy, precision, recall,
    or F1.  Metrics such as latency, throughput, certificate size,
    communication overhead, memory usage, runtime, security rates, and
    other quantitative measurements are supported.
    """

    USER_AGENT = (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/153.0 Safari/537.36 "
        "AI-Co-Scientist/1.0"
    )

    # Common units used in scientific experiments.
    UNIT_ALIASES = {
        "%": "%",
        "percent": "%",
        "percentage": "%",
        "pct": "%",
        "ms": "ms",
        "millisecond": "ms",
        "milliseconds": "ms",
        "s": "s",
        "sec": "s",
        "second": "s",
        "seconds": "s",
        "us": "us",
        "μs": "us",
        "µs": "us",
        "microsecond": "us",
        "microseconds": "us",
        "ns": "ns",
        "nanosecond": "ns",
        "nanoseconds": "ns",
        "b": "bytes",
        "byte": "bytes",
        "bytes": "bytes",
        "kb": "KB",
        "kib": "KiB",
        "mb": "MB",
        "mib": "MiB",
        "gb": "GB",
        "gib": "GiB",
        "bps": "bps",
        "kbps": "Kbps",
        "mbps": "Mbps",
        "gbps": "Gbps",
        "tbps": "Tbps",
        "hz": "Hz",
        "khz": "kHz",
        "mhz": "MHz",
        "ghz": "GHz",
        "packets/s": "packets/s",
        "requests/s": "requests/s",
        "req/s": "requests/s",
        "samples/s": "samples/s",
        "fps": "FPS",
        "joules": "J",
        "joule": "J",
        "j": "J",
    }

    # Metrics that are normally proportions/percentages.
    #
    # This is only used to help interpret numeric values.  It does NOT
    # restrict which metrics can be extracted.
    PERCENTAGE_METRIC_KEYWORDS = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_score",
        "auc",
        "roc_auc",
        "success_rate",
        "error_rate",
        "detection_rate",
        "false_positive_rate",
        "false_negative_rate",
        "true_positive_rate",
        "true_negative_rate",
        "percentage",
        "percent",
        "proportion",
        "ratio",
    )

    def __init__(
        self,
        timeout: int = 30,
        max_text_length: int = 30000,
        llm_callable=None,
        paper_library=None,
    ):
        self.timeout = timeout
        self.max_text_length = max_text_length
        self.llm_callable = llm_callable
        self.paper_library = paper_library

        # Limit the amount of indexed paper text sent to the LLM.
        self.max_indexed_text_length = 12000

        # Keep enough output capacity for structured JSON containing
        # several metrics and experiment details.
        self.extraction_max_tokens = 4096

    # ============================================================
    # URL Normalization
    # ============================================================

    def _normalise_paper_url(self, url: str) -> str:
        """
        Normalize common scientific-paper URLs into downloadable
        PDF URLs where deterministic conversion is available.

        Handles:
        - arXiv /html/
        - arXiv /abs/
        """
        if not url:
            return url

        url = url.strip()

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

    def _download_source(
        self,
        url: str,
    ) -> Dict[str, Any]:
        """
        Download an evidence URL and determine whether it contains
        PDF or HTML content.

        Returns:
            {
                "url": final_url,
                "content_type": "pdf" | "html",
                "content": bytes,
            }
        """
        if not url:
            raise ValueError("Paper URL is required.")

        original_url = url.strip()

        normalized_url = self._normalise_paper_url(
            original_url
        )

        response = requests.get(
            normalized_url,
            timeout=self.timeout,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": (
                    "application/pdf,"
                    "text/html,"
                    "application/xhtml+xml,"
                    "*/*"
                ),
            },
            allow_redirects=True,
        )

        response.raise_for_status()

        content = response.content

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        # PDF detected from content type or magic bytes.
        if (
            "application/pdf" in content_type
            or content.startswith(b"%PDF")
        ):
            return {
                "url": response.url,
                "content_type": "pdf",
                "content": content,
            }

        # HTML detected from content type.
        if (
            "text/html" in content_type
            or "application/xhtml+xml" in content_type
        ):
            return {
                "url": response.url,
                "content_type": "html",
                "content": content,
            }

        # Some websites incorrectly report Content-Type.
        preview = content[:500].lower()

        if (
            b"<html" in preview
            or b"<!doctype html" in preview
        ):
            return {
                "url": response.url,
                "content_type": "html",
                "content": content,
            }

        raise ValueError(
            "Unsupported evidence content type. "
            f"URL: {original_url}. "
            f"Resolved URL: {response.url}. "
            f"Content-Type: {content_type}."
        )

    def _resolve_pdf_url(self, url: str) -> str:
        """
        Resolve an evidence URL to a downloadable PDF URL.

        Handles:
        - Direct PDF URLs
        - arXiv URLs
        - DOI/article URLs that redirect to an HTML article page

        The method first checks whether the supplied URL already returns
        a PDF. If it does not, it attempts to identify a PDF link from
        the resulting HTML page.
        """
        if not url:
            raise ValueError("Paper URL is required.")

        original_url = url.strip()
        normalized_url = self._normalise_paper_url(original_url)

        response = requests.get(
            normalized_url,
            timeout=self.timeout,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": (
                    "application/pdf,text/html,"
                    "application/xhtml+xml"
                ),
            },
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if (
            "application/pdf" in content_type
            or response.content.startswith(b"%PDF")
        ):
            return response.url

        html = response.text

        pdf_patterns = [
            r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
            r'href=["\']([^"\']*/pdf/[^"\']*)["\']',
            r'content=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        ]

        for pattern in pdf_patterns:
            matches = re.findall(
                pattern,
                html,
                flags=re.IGNORECASE,
            )

            for candidate in matches:
                candidate = candidate.strip()

                if not candidate:
                    continue

                candidate_url = urljoin(
                    response.url,
                    candidate,
                )

                if ".pdf" in candidate_url.lower():
                    logger.info(
                        "[PaperReader] Resolved PDF URL: %s",
                        candidate_url,
                    )
                    return candidate_url

        raise ValueError(
            "Could not resolve a PDF URL from the evidence page. "
            f"Original URL: {original_url}. "
            f"Resolved URL: {response.url}. "
            f"Content-Type: {content_type}."
        )

    def _extract_html_text(
        self,
        html: str,
    ) -> str:
        """
        Extract readable text from an HTML article/page.

        This is intentionally lightweight and does not require BeautifulSoup.
        Script, style, navigation, and other non-content elements are removed.
        """
        if not html:
            return ""

        text = html

        # Remove script/style/noscript blocks.
        text = re.sub(
            r"<(script|style|noscript|svg)[^>]*>.*?</\1>",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Preserve rough paragraph/heading boundaries.
        text = re.sub(
            r"</(p|div|section|article|h1|h2|h3|h4|h5|h6|li|tr)>",
            "\n",
            text,
            flags=re.IGNORECASE,
        )

        # Remove remaining HTML tags.
        text = re.sub(
            r"<[^>]+>",
            " ",
            text,
        )

        # Decode HTML entities.
        text = unescape(text)

        # Normalize whitespace.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n\s*\n+",
            "\n\n",
            text,
        )

        return text.strip()

    # ============================================================
    # PDF Download
    # ============================================================

    def download_paper(self, url: str) -> bytes:
        """
        Download a research paper PDF.

        Supports:
        - Direct PDF URLs
        - arXiv abstract/HTML URLs
        - DOI/article URLs that redirect to an HTML article page
        """
        if not url:
            raise ValueError("Paper URL is required.")

        original_url = url.strip()

        pdf_url = self._resolve_pdf_url(original_url)

        logger.info(
            "[PaperReader] Downloading PDF from: %s",
            pdf_url,
        )

        response = requests.get(
            pdf_url,
            timeout=self.timeout,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "application/pdf",
            },
            allow_redirects=True,
        )

        response.raise_for_status()

        content = response.content

        if not content.startswith(b"%PDF"):
            content_type = response.headers.get(
                "Content-Type",
                "",
            )

            preview = content[:100].decode(
                "utf-8",
                errors="replace",
            )

            raise ValueError(
                "The resolved evidence URL did not return a PDF. "
                f"Original URL: {original_url}. "
                f"PDF URL: {pdf_url}. "
                f"Content-Type: {content_type}. "
                f"Response preview: {preview}"
            )

        return content

    # ============================================================
    # PDF Text Extraction
    # ============================================================

    def extract_text(self, pdf_bytes: bytes) -> str:
        """
        Extract text from all pages of a PDF.
        """
        if not pdf_bytes:
            return ""

        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)

        pages: List[str] = []

        for page in reader.pages:
            try:
                text = page.extract_text()
            except Exception as exc:
                logger.warning(
                    "[PaperReader] Failed to extract PDF page: %s",
                    exc,
                )
                continue

            if text:
                pages.append(text)

        return "\n".join(pages)

    # ============================================================
    # Results Section Detection
    # ============================================================

    def find_experimental_content(self, text: str) -> str:
        """
        Identify and preserve experimental/evaluation portions of
        the paper.

        The LLM is explicitly instructed NOT to summarize, paraphrase,
        calculate, normalize, or remove numerical evidence.

        The goal is to preserve enough original text for the next
        structured extraction stage to recover exact metrics.
        """
        if not text or not text.strip():
            return ""

        if self.llm_callable is None:
            logger.warning(
                "[PaperReader] No LLM callable configured. "
                "Returning full paper text."
            )
            return text[: self.max_text_length]

        paper_text = text[: self.max_text_length]

        prompt = f"""
You are extracting experimental evidence from a scientific research paper.

Your task is to identify the portions of the paper containing actual
experimental, evaluation, performance, benchmark, or measured-result
evidence.

The next processing stage will extract exact numerical metrics from
your output. Therefore, LOSSLESS preservation of quantitative evidence
is more important than producing a short answer.

============================================================
WHAT TO INCLUDE
============================================================

Include text containing:

- experimental evaluation
- measured results
- quantitative findings
- performance measurements
- numerical comparisons
- experimental tables
- experimental figures when their numerical values are described
  in the text
- baseline comparisons
- latency
- throughput
- overhead
- accuracy
- precision
- recall
- F1-score
- AUC
- runtime
- execution time
- memory
- communication cost
- certificate size
- key size
- computational cost
- energy consumption
- packet loss
- error rate
- attack/detection rate
- security metrics
- scalability measurements
- resource utilization
- or ANY OTHER measured quantitative metric.

Also include the experimental setup when it is necessary to correctly
interpret the reported numerical result.

============================================================
CRITICAL PRESERVATION RULES
============================================================

1. DO NOT SUMMARIZE.

2. DO NOT PARAPHRASE.

3. DO NOT REWRITE numerical findings.

4. Preserve the ORIGINAL wording from the paper whenever possible.

5. Preserve EVERY numerical value associated with an experiment.

6. Preserve the COMPLETE sentence containing a numerical result.

7. Preserve the COMPLETE table row or surrounding table text when
   numerical table values are available in the extracted text.

8. Preserve ALL units exactly as written.

Examples:
- %
- ms
- s
- μs
- bytes
- KB
- MB
- Mbps
- Gbps
- packets/s

9. Preserve baseline values and proposed-method values.

10. Preserve both values when a paper says something such as:
    "Latency increased from 5 ms to 7 ms."

11. Preserve percentages exactly as reported.

12. Do NOT convert percentages to decimals.

13. Do NOT convert units.

14. Do NOT calculate new metrics.

15. Do NOT infer numerical values.

16. Do NOT replace numerical results with words such as:
    "higher", "lower", "better", or "comparable".

17. If the paper reports a factor such as:
    "three-fold decrease"
    preserve the original wording.

18. If a numerical value is associated with a baseline comparison,
    preserve the baseline context.

19. If a numerical value appears in an experimental table, preserve
    the table context if it is available in the extracted text.

20. Do not remove numbers merely because they appear in a paragraph
    instead of a section called "Results".

============================================================
RELEVANT SECTION TYPES
============================================================

Look for experimental evidence under headings such as:

- Evaluation
- Experiments
- Experimental Results
- Experimental Evaluation
- Performance Evaluation
- Performance Analysis
- Results
- Results and Discussion
- Evaluation Results
- Benchmark
- Experimental Setup
- Implementation and Evaluation
- Performance Study
- Case Study

Do NOT rely only on section headings. Use the CONTENT to determine
whether the text contains actual experimental evidence.

============================================================
EXCLUDE
============================================================

Exclude:

- Introduction
- General background
- Literature review
- Related work
- General motivation
- Future work
- References
- Author biographies

A conclusion paragraph should normally be excluded if it only repeats
earlier results. However, if it contains a quantitative result that
does NOT appear elsewhere in the supplied text, preserve it.

============================================================
IMPORTANT
============================================================

Do not invent missing values.

Do not calculate values.

Do not normalize values.

Do not convert units.

Do not summarize the paper.

Return ONLY the selected experimental/evaluation text.

PAPER TEXT:

{paper_text}
"""

        try:
            logger.info(
                "PaperReader experimental-content LLM call: reasoning=off"
            )

            result = self.llm_callable(
                prompt,
                temperature=0.0,
                reasoning="off",
                max_tokens=self.extraction_max_tokens,
            )

            if not result:
                logger.warning(
                    "[PaperReader] LLM returned no experimental content."
                )
                return paper_text

            extracted = str(result).strip()

            if not extracted:
                return paper_text

            logger.info(
                "[PaperReader] Extracted experimental content length: %d",
                len(extracted),
            )

            print(
                "\n===== PAPER RELEVANT EXPERIMENTAL CONTENT ====="
            )
            print(extracted[:10000])
            print(
                "================================================\n"
            )

            return extracted

        except Exception as exc:
            logger.warning(
                "[PaperReader] Failed to identify experimental "
                "content with LLM: %s",
                exc,
            )
            return paper_text

    # ============================================================
    # JSON Response Extraction
    # ============================================================

    def _extract_json(self, response: Any) -> Dict[str, Any]:
        """
        Extract a JSON object from an LLM response.

        Handles:
        - Direct dictionaries
        - Plain JSON strings
        - JSON wrapped in Markdown fences
        - JSON embedded in surrounding text
        """
        if response is None:
            return {}

        if isinstance(response, dict):
            return response

        text = str(response).strip()

        if not text:
            return {}

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        ).strip()

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return {}

        json_text = text[start : end + 1]

        try:
            parsed = json.loads(json_text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            return {}

        return {}

    # ============================================================
    # Numeric / Metric Helpers
    # ============================================================

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """
        Convert a numeric value into float.

        Supports:
        - int
        - float
        - numeric strings
        - strings such as "88%"
        - strings such as "5.2 ms"

        This helper is intentionally conservative and should only be
        used on fields that are expected to contain numeric values.
        """
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            try:
                number = float(value)

                if number != number:
                    return None

                if number in (float("inf"), float("-inf")):
                    return None

                return number

            except (TypeError, ValueError):
                return None

        if not isinstance(value, str):
            return None

        text = value.strip()

        if not text:
            return None

        text = text.replace(",", "")

        match = re.search(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eE][-+]?\d+)?",
            text,
        )

        if not match:
            return None

        try:
            number = float(match.group(0))

            if number != number:
                return None

            if number in (float("inf"), float("-inf")):
                return None

            return number

        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalise_metric_name(cls, name: Any) -> str:
        """
        Convert a metric name into a stable machine-readable form.

        Examples:
            "Handshake Latency" -> "handshake_latency"
            "F1-score" -> "f1_score"
            "Certificate Size (bytes)" -> "certificate_size_bytes"
        """
        if name is None:
            return ""

        text = str(name).strip().lower()

        if not text:
            return ""

        replacements = {
            "%": " percent ",
            "μ": "u",
            "µ": "u",
            "/": " per ",
            "-": "_",
            "–": "_",
            "—": "_",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(
            r"[\(\)\[\]\{\},:;]+",
            " ",
            text,
        )

        text = re.sub(
            r"[^a-z0-9_]+",
            "_",
            text,
        )

        text = re.sub(
            r"_+",
            "_",
            text,
        )

        return text.strip("_")

    @classmethod
    def _normalise_unit(cls, unit: Any) -> str:
        """
        Normalize a unit while preserving its meaning.

        Unknown units are retained rather than discarded.
        """
        if unit is None:
            return ""

        text = str(unit).strip()

        if not text:
            return ""

        key = text.lower().strip()

        if key in cls.UNIT_ALIASES:
            return cls.UNIT_ALIASES[key]

        return text

    @classmethod
    def _infer_unit_from_metric_name(
        cls,
        metric_name: str,
    ) -> str:
        """
        Infer an obvious unit from the metric name.

        This only handles explicit names such as:
            latency_ms
            certificate_size_bytes
            throughput_mbps
            accuracy_percent

        It does not invent units for ambiguous metrics.
        """
        normalized = cls._normalise_metric_name(metric_name)

        explicit_patterns = [
            (r"(?:^|_)ms(?:_|$)", "ms"),
            (r"(?:^|_)milliseconds?(?:_|$)", "ms"),
            (r"(?:^|_)sec(?:_|$)", "s"),
            (r"(?:^|_)seconds?(?:_|$)", "s"),
            (r"(?:^|_)us(?:_|$)", "us"),
            (r"(?:^|_)microseconds?(?:_|$)", "us"),
            (r"(?:^|_)ns(?:_|$)", "ns"),
            (r"(?:^|_)bytes?(?:_|$)", "bytes"),
            (r"(?:^|_)kb(?:_|$)", "KB"),
            (r"(?:^|_)kib(?:_|$)", "KiB"),
            (r"(?:^|_)mb(?:_|$)", "MB"),
            (r"(?:^|_)mib(?:_|$)", "MiB"),
            (r"(?:^|_)gb(?:_|$)", "GB"),
            (r"(?:^|_)gib(?:_|$)", "GiB"),
            (r"(?:^|_)kbps(?:_|$)", "Kbps"),
            (r"(?:^|_)mbps(?:_|$)", "Mbps"),
            (r"(?:^|_)gbps(?:_|$)", "Gbps"),
            (r"(?:^|_)bps(?:_|$)", "bps"),
            (r"(?:^|_)percent(?:_|$)", "%"),
            (r"(?:^|_)percentage(?:_|$)", "%"),
            (r"(?:^|_)pct(?:_|$)", "%"),
        ]

        for pattern, unit in explicit_patterns:
            if re.search(pattern, normalized):
                return unit

        return ""

    @classmethod
    def _infer_value_type(
        cls,
        metric_name: str,
        unit: str = "",
        definition: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Infer a broad value type for a metric.

        Possible values:
        - percentage
        - proportion
        - factor
        - count
        - measurement
        - numeric
        """
        if definition:
            explicit = definition.get("value_type")

            if explicit:
                return str(explicit).strip().lower()

        if unit == "%":
            return "percentage"

        normalized = cls._normalise_metric_name(metric_name)

        if "percent" in normalized or "percentage" in normalized:
            return "percentage"

        if "proportion" in normalized:
            return "proportion"

        if "fold" in normalized or "factor" in normalized:
            return "factor"

        if normalized.endswith("_count") or normalized == "count":
            return "count"

        for keyword in cls.PERCENTAGE_METRIC_KEYWORDS:
            if keyword in normalized:
                return "percentage"

        if unit:
            return "measurement"

        return "numeric"

    @classmethod
    def _infer_direction(
        cls,
        metric_name: str,
        definition: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Determine the expected direction of a metric when it is
        scientifically obvious.

        Returns:
            higher_is_better
            lower_is_better
            unknown
        """
        if definition:
            direction = definition.get("direction")

            if direction:
                normalized = str(direction).strip().lower()

                if normalized in {
                    "higher_is_better",
                    "higher",
                    "maximize",
                    "maximise",
                    "max",
                }:
                    return "higher_is_better"

                if normalized in {
                    "lower_is_better",
                    "lower",
                    "minimize",
                    "minimise",
                    "min",
                }:
                    return "lower_is_better"

                if normalized in {
                    "unknown",
                    "neutral",
                    "none",
                }:
                    return "unknown"

        normalized = cls._normalise_metric_name(metric_name)

        lower_keywords = (
            "latency",
            "delay",
            "overhead",
            "runtime",
            "execution_time",
            "response_time",
            "processing_time",
            "waiting_time",
            "memory_usage",
            "memory_consumption",
            "energy",
            "energy_consumption",
            "power_consumption",
            "packet_loss",
            "error",
            "loss",
            "cost",
            "certificate_size",
            "key_size",
            "message_size",
            "communication_cost",
            "false_positive",
            "false_negative",
        )

        higher_keywords = (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "throughput",
            "stability",
            "efficiency",
            "detection_rate",
            "true_positive",
            "true_negative",
        )

        if any(
            keyword in normalized
            for keyword in lower_keywords
        ):
            return "lower_is_better"

        if any(
            keyword in normalized
            for keyword in higher_keywords
        ):
            return "higher_is_better"

        return "unknown"

    @classmethod
    def _normalise_metric_definition(
        cls,
        metric_name: str,
        definition: Any,
    ) -> Dict[str, Any]:
        """
        Normalize a single metric definition.
        """
        normalized_name = cls._normalise_metric_name(
            metric_name
        )

        if isinstance(definition, dict):
            result = dict(definition)
        elif definition is None:
            result = {}
        else:
            result = {
                "description": str(definition),
            }

        display_name = (
            result.get("display_name")
            or result.get("name")
            or metric_name
        )

        unit = cls._normalise_unit(
            result.get("unit")
        )

        if not unit:
            unit = cls._infer_unit_from_metric_name(
                normalized_name
            )

        value_type = cls._infer_value_type(
            normalized_name,
            unit,
            result,
        )

        direction = cls._infer_direction(
            normalized_name,
            result,
        )

        result["display_name"] = str(display_name)
        result["unit"] = unit
        result["value_type"] = value_type
        result["direction"] = direction

        return result

    @classmethod
    def _normalise_metric_definitions(
        cls,
        definitions: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Normalize an arbitrary metric_definitions object.
        """
        if not isinstance(definitions, dict):
            return {}

        normalized: Dict[str, Dict[str, Any]] = {}

        for name, definition in definitions.items():
            metric_name = cls._normalise_metric_name(name)

            if not metric_name:
                continue

            normalized[metric_name] = (
                cls._normalise_metric_definition(
                    metric_name,
                    definition,
                )
            )

        return normalized

    @classmethod
    def _normalise_metric_value(
        cls,
        metric_name: str,
        value: Any,
        definition: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """
        Normalize a metric value without changing arbitrary physical
        measurements.

        Important:
            95 ms remains 95.
            1250 bytes remains 1250.
            88% becomes 0.88 only when the metric is clearly a
            percentage/proportion and the value is represented as
            0-100.

        This allows the Comparator to use one normalized representation
        for percentage metrics while preserving physical measurements.
        """
        number = cls._safe_float(value)

        if number is None:
            return None

        definition = definition or {}

        unit = cls._normalise_unit(
            definition.get("unit")
        )

        value_type = cls._infer_value_type(
            metric_name,
            unit,
            definition,
        )

        original_text = (
            str(value).lower()
            if isinstance(value, str)
            else ""
        )

        explicit_percentage = (
            "%" in original_text
            or unit == "%"
            or value_type in {
                "percentage",
                "proportion",
            }
        )

        if explicit_percentage:
            if 1 < number <= 100:
                return number / 100.0

        return number
    
    @classmethod
    def _normalise_reference_value(
        cls,
        value: Any,
        metric_name: str,
        metric_definition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Normalize a structured reference metric while preserving
        its scientific meaning.

        Unlike _normalise_metric_value(), this method does NOT reduce
        the reference to a single float. It preserves whether the
        reported value is:

        - an exact measured value
        - an upper/lower bound
        - a range
        - a configuration value
        - a qualitative result

        This prevents statements such as "less than 80 ms" from being
        incorrectly converted into an exact measurement of 80 ms.
        """
        definition = metric_definition or {}

        if isinstance(value, dict):
            normalized = dict(value)
        else:
            normalized = {
                "value": value,
            }

        raw_value = normalized.get("value")

        unit = (
            normalized.get("unit")
            or definition.get("unit")
            or cls._infer_unit_from_metric_name(metric_name)
        )

        unit = cls._normalise_unit(unit)

        value_type = str(
            normalized.get("value_type")
            or "unknown"
        ).strip().lower()

        relation = str(
            normalized.get("relation")
            or "none"
        ).strip().lower()

        allowed_value_types = {
            "measured_value",
            "upper_bound",
            "lower_bound",
            "range",
            "configuration",
            "qualitative_result",
            "unknown",
        }

        if value_type not in allowed_value_types:
            value_type = "unknown"

        allowed_relations = {
            "exact",
            "less_than",
            "less_than_or_equal",
            "greater_than",
            "greater_than_or_equal",
            "range",
            "none",
        }

        if relation not in allowed_relations:
            relation = "none"

        # Normalize numeric values, but preserve the semantic wrapper.
        numeric_value = cls._safe_float(raw_value)

        if numeric_value is not None:
            value = numeric_value
        else:
            value = raw_value

        source_text = normalized.get(
            "source_text",
            "",
        )

        if source_text is None:
            source_text = ""

        result = {
            "value": value,
            "unit": unit or None,
            "value_type": value_type,
            "relation": relation,
            "source_text": str(source_text),
        }

        # Preserve optional fields if supplied by the LLM.
        for key in (
            "lower_value",
            "upper_value",
            "lower_unit",
            "upper_unit",
        ):
            if key in normalized:
                result[key] = normalized[key]

        return result

    @classmethod
    def _normalise_metrics(
        cls,
        metrics: Any,
        metric_definitions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Normalize a metrics dictionary.

        Arbitrary metric names are accepted.
        """
        if not isinstance(metrics, dict):
            return {}

        definitions = cls._normalise_metric_definitions(
            metric_definitions or {}
        )

        normalized: Dict[str, float] = {}

        for name, value in metrics.items():
            metric_name = cls._normalise_metric_name(name)

            if not metric_name:
                continue

            definition = definitions.get(
                metric_name,
                {},
            )

            number = cls._normalise_metric_value(
                metric_name,
                value,
                definition,
            )

            if number is None:
                continue

            normalized[metric_name] = number

        return normalized

    # ============================================================
    # Result Validation / Recovery
    # ============================================================

    @classmethod
    def _recover_reference_metrics_from_results(
        cls,
        details: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Recover structured reference metrics from result entries when
        possible.

        This is a safety net for LLM responses where the model correctly
        places a numerical result in `results` but accidentally leaves
        `reference_metrics` empty.

        The recovered values preserve scientific semantics instead of
        being reduced to plain floats.

        Important:
            This method only uses explicitly structured numeric fields
            already present in the result objects. It does NOT attempt
            to calculate, estimate, or infer values from prose.
        """
        recovered: Dict[str, Dict[str, Any]] = {}

        results = details.get("results")

        if not isinstance(results, list):
            return recovered

        metric_definitions = cls._normalise_metric_definitions(
            details.get("metric_definitions", {})
        )

        for result in results:
            if not isinstance(result, dict):
                continue

            metrics = result.get("metrics")

            if not isinstance(metrics, dict):
                continue

            for name, value in metrics.items():
                metric_name = cls._normalise_metric_name(name)

                if not metric_name:
                    continue

                definition = metric_definitions.get(
                    metric_name,
                    {},
                )

                # Preserve structured semantic information if the result
                # already contains it.
                if isinstance(value, dict):
                    normalized_value = cls._normalise_reference_value(
                        value,
                        metric_name,
                        definition,
                    )

                else:
                    # A plain numeric result entry represents an explicitly
                    # reported measurement, but we must not invent a bound,
                    # threshold, or other semantic relation.
                    normalized_value = cls._normalise_reference_value(
                        {
                            "value": value,
                            "unit": (
                                definition.get("unit")
                                or cls._infer_unit_from_metric_name(
                                    metric_name
                                )
                                or None
                            ),
                            "value_type": "measured_value",
                            "relation": "exact",
                            "source_text": "",
                        },
                        metric_name,
                        definition,
                    )

                if normalized_value.get("value") is not None:
                    recovered[metric_name] = normalized_value

        return recovered

    @classmethod
    def _validate_and_normalise_details(
        cls,
        details: Dict[str, Any],
        raw_text: str,
    ) -> Dict[str, Any]:
        """
        Normalize the structured experiment extraction response.

        This method ensures the output has a stable schema and that
        reference_metrics contains only numeric values.
        """
        if not isinstance(details, dict):
            return {}

        required_defaults = {
            "experiment_objective": "",
            "experimental_setup": [],
            "models_or_systems": [],
            "datasets_or_testbeds": [],
            "baselines": [],
            "configurations": [],
            "metrics": [],
            "metric_definitions": {},
            "hyperparameters": {},
            "training_details": {},
            "reference_metrics": {},
            "reference_conditions": {},
            "results": [],
            "experiment_notes": [],
        }

        for key, default in required_defaults.items():
            if key not in details:
                details[key] = default

        metric_definitions = cls._normalise_metric_definitions(
            details.get("metric_definitions", {})
        )

        # Normalize explicitly reported reference metrics while
        # preserving their scientific semantics.
        reference_metrics: Dict[str, Dict[str, Any]] = {}

        raw_reference_metrics = details.get(
            "reference_metrics",
            {},
        )

        if isinstance(raw_reference_metrics, dict):
            for name, value in raw_reference_metrics.items():
                metric_name = cls._normalise_metric_name(name)

                if not metric_name:
                    continue

                definition = metric_definitions.get(
                    metric_name,
                    {},
                )

                normalized_value = cls._normalise_reference_value(
                    value,
                    metric_name,
                    definition,
                )

                # Keep only entries that actually contain a value.
                if normalized_value.get("value") is not None:
                    reference_metrics[metric_name] = normalized_value

        # Recover structured metrics from result objects if present.
        recovered_metrics = cls._recover_reference_metrics_from_results(
            details
        )

        for name, value in recovered_metrics.items():
            reference_metrics.setdefault(
                name,
                value,
            )

        # Normalize reference conditions while preserving their
        # configuration semantics.
        reference_conditions: Dict[str, Dict[str, Any]] = {}

        raw_reference_conditions = details.get(
            "reference_conditions",
            {},
        )

        if isinstance(raw_reference_conditions, dict):
            for name, value in raw_reference_conditions.items():
                condition_name = cls._normalise_metric_name(name)

                if not condition_name:
                    continue

                if isinstance(value, dict):
                    normalized_condition = dict(value)
                else:
                    normalized_condition = {
                        "value": value,
                    }

                normalized_condition.setdefault(
                    "value_type",
                    "configuration",
                )
                normalized_condition.setdefault(
                    "unit",
                    None,
                )
                normalized_condition.setdefault(
                    "source_text",
                    "",
                )

                if normalized_condition.get("value") is not None:
                    reference_conditions[condition_name] = (
                        normalized_condition
                    )

        details["reference_conditions"] = reference_conditions

        # Normalize the list of metric names.
        metric_names: List[str] = []

        raw_metrics = details.get("metrics", [])

        if isinstance(raw_metrics, dict):
            raw_metrics = list(raw_metrics.keys())

        if isinstance(raw_metrics, list):
            for metric in raw_metrics:
                if isinstance(metric, dict):
                    name = (
                        metric.get("name")
                        or metric.get("metric")
                        or metric.get("metric_name")
                    )
                else:
                    name = metric

                normalized_name = cls._normalise_metric_name(
                    name
                )

                if normalized_name:
                    metric_names.append(
                        normalized_name
                    )

        # Every reference metric is also a metric.
        for metric_name in reference_metrics:
            if metric_name not in metric_names:
                metric_names.append(metric_name)

        details["metrics"] = metric_names
        details["metric_definitions"] = metric_definitions
        details["reference_metrics"] = reference_metrics

        # Ensure common list/object fields have stable types.
        list_fields = [
            "experimental_setup",
            "models_or_systems",
            "datasets_or_testbeds",
            "baselines",
            "configurations",
            "results",
            "experiment_notes",
        ]

        for field in list_fields:
            if not isinstance(details.get(field), list):
                details[field] = (
                    [details[field]]
                    if details[field] not in (None, "")
                    else []
                )

        object_fields = [
            "hyperparameters",
            "training_details",
            "reference_conditions",
            "metric_definitions",
            "reference_metrics",
        ]

        for field in object_fields:
            if not isinstance(details.get(field), dict):
                details[field] = {}

        details["raw_text"] = raw_text

        return details

    # ============================================================
    # Experiment Detail Extraction
    # ============================================================

    def extract_experiment_details(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Extract structured experimental details from scientific paper
        text.

        The LLM is required to:

        - extract arbitrary metrics;
        - preserve explicitly reported numeric values;
        - provide metric definitions and units when supported;
        - distinguish actual experimental results from contextual
          numbers;
        - avoid calculating or inventing values.
        """
        if not text or not text.strip():
            return {}

        if self.llm_callable is None:
            logger.warning(
                "[PaperReader] No LLM callable configured. "
                "Cannot perform structured experiment extraction."
            )

            return {
                "raw_text": text,
                "models_or_systems": [],
                "datasets_or_testbeds": [],
                "metrics": [],
                "metric_definitions": {},
                "reference_metrics": {},
                "hyperparameters": {},
                "training_details": {},
                "results": [],
                "experiment_notes": [],
            }

        prompt = f"""
Extract structured experimental information from the scientific paper
text below.

The supplied text has already been filtered toward experimental,
evaluation, benchmark, and results-related content.

Your primary objective is to identify EXACTLY what the paper measured
and EXACTLY what numerical experimental results it reported.

============================================================
OUTPUT FORMAT
============================================================

Your ENTIRE response MUST be exactly ONE valid JSON object.

Do NOT:

- explain your answer
- provide reasoning
- describe what you are doing
- repeat the instructions
- use Markdown
- use ```json fences
- add text before or after the JSON
- invent information
- calculate values
- estimate values
- convert units
- convert percentages to decimals

Use exactly this structure:

{{
  "experiment_objective": "",
  "experimental_setup": [],
  "models_or_systems": [],
  "datasets_or_testbeds": [],
  "baselines": [],
  "configurations": [],
  "metrics": [],
  "metric_definitions": {{}},
  "hyperparameters": {{}},
  "training_details": {{}},
  "reference_metrics": {{
    "<metric_name>": {{
        "value": <number>,
        "unit": "<unit or null>",
        "value_type": "measured_value|upper_bound|lower_bound|range|qualitative_result|unknown",
        "relation": "exact|less_than|less_than_or_equal|greater_than|greater_than_or_equal|range|none",
        "source_text": "<short original wording>"
    }}
  }},
  "reference_conditions": {{
    "<condition_name>": {{
        "value": <number|string|list>,
        "unit": "<unit or null>",
        "value_type": "configuration",
        "source_text": "<short original wording>"
    }}
  }},
  "results": [],
  "experiment_notes": []
}}

============================================================
METRICS
============================================================

"metrics" must contain ALL metrics explicitly evaluated or measured
by this paper.

Do NOT limit the list to:

- accuracy
- precision
- recall
- F1

Other valid metrics include, for example:

- handshake latency
- certificate size
- key generation time
- encapsulation time
- decapsulation time
- throughput
- throughput stability
- communication overhead
- computational overhead
- memory usage
- runtime
- execution time
- packet loss
- detection rate
- false-positive rate
- unauthorized access
- energy consumption
- resource utilization
- scalability
- security measurements

Extract whatever metrics the paper actually evaluates.

============================================================
METRIC DEFINITIONS
============================================================

"metric_definitions" should describe metrics when the supplied text
provides enough information.

Use this structure:

"metric_definitions": {{
  "metric_name": {{
    "display_name": "",
    "description": "",
    "unit": "",
    "direction": "",
    "value_type": ""
  }}
}}

Possible "direction" values:

- "higher_is_better"
- "lower_is_better"
- "unknown"

Possible "value_type" values include:

- "percentage"
- "proportion"
- "factor"
- "count"
- "measurement"
- "numeric"

IMPORTANT:

Do not invent units or optimization directions.

For example:

"handshake_latency_ms": {{
  "display_name": "Handshake latency",
  "description": "TLS handshake latency",
  "unit": "ms",
  "direction": "lower_is_better",
  "value_type": "measurement"
}}

============================================================
REFERENCE METRICS
============================================================

"reference_metrics" MUST contain explicit numerical experimental
results reported by the paper.

The values MUST be JSON numbers, not strings.

Correct:

"reference_metrics": {{
  "accuracy": 88
}}

Correct:

"reference_metrics": {{
  "handshake_latency_ms": 5.2,
  "certificate_size_bytes": 1250
}}

Correct:

"reference_metrics": {{
  "throughput_stability_percent": 35
}}

Incorrect:

"reference_metrics": {{
  "accuracy": "88%"
}}

Incorrect:

"reference_metrics": {{
  "latency": "5.2 ms"
}}

============================================================
CRITICAL REFERENCE METRIC RULES
============================================================

1. Only extract numerical values explicitly reported in the supplied
   experimental text.

2. Do NOT invent missing values.

3. Do NOT calculate a value.

4. Do NOT derive a percentage from two other values.

5. Do NOT convert units.

6. Do NOT convert percentages to decimals.

7. Preserve the numerical value exactly as reported.

8. If the paper reports 88%, use:

   "accuracy": 88

9. If the paper reports 5.2 ms, use:

   "handshake_latency_ms": 5.2

10. If the paper reports 1250 bytes, use:

    "certificate_size_bytes": 1250

11. If the paper reports 35% better throughput stability, use:

    "throughput_stability_percent": 35

12. If the paper reports a factor such as:

    "three-fold decrease in unauthorized access"

    preserve the factor without converting it into a percentage:

    "unauthorized_access_reduction_factor": 3

13. If the paper reports multiple values for the same metric, use
    descriptive names.

For example:

"Latency increased from 5 ms to 7 ms."

should become:

"reference_metrics": {{
  "latency_baseline_ms": 5,
  "latency_proposed_ms": 7
}}

14. If a metric is named but the paper does not provide a numerical
    result, include it in "metrics" and, where possible,
    "metric_definitions", but DO NOT put it into "reference_metrics".

15. If the paper reports a range such as "5-7 ms", do not invent a
    single value such as 6 ms.

16. If the paper reports a mean and standard deviation, preserve them
    separately if possible.

Example:

"latency_mean_ms": 5.2,
"latency_std_ms": 0.4

17. Do not treat a year, sample count, number of authors, section
    number, citation number, or other contextual number as a metric
    unless it is explicitly being evaluated or measured.

18. Do not treat numerical claims about another paper as this paper's
    own experimental result.

============================================================
RESULTS
============================================================

"results" should contain concise descriptions of actual experimental
findings.

Every result containing an explicit numerical measurement should also
have a corresponding entry in "reference_metrics" whenever it can be
represented directly as a numeric metric without calculation.

Example:

"results": [
  "The proposed method achieved 88% accuracy in trust detection.",
  "The method achieved 35% better throughput stability."
]

Corresponding:

"reference_metrics": {{
  "accuracy": 88,
  "throughput_stability_percent": 35
}}

Qualitative results may remain only in "results".

Example:

"results": [
  "The proposed system showed comparable performance to the baseline."
]

Do NOT put "comparable" into "reference_metrics".

============================================================
MULTIPLE EXPERIMENTS
============================================================

If the paper contains multiple experiments, preserve distinct metrics
using descriptive names when necessary.

For example:

"latency_baseline_ms": 5,
"latency_proposed_ms": 7

or:

"accuracy_dataset_a": 92,
"accuracy_dataset_b": 89

Do not overwrite one result with another.

============================================================
EXPERIMENTAL SETUP
============================================================

Extract only information explicitly supported by the supplied text.

Include:

- model/system
- dataset
- testbed
- hardware
- software framework
- baseline
- experimental configuration
- evaluation protocol
- important hyperparameters
- training details

Do not invent missing information.

============================================================
REFERENCE VALUE SEMANTICS
============================================================

For every explicitly reported numerical value that may be useful
for downstream comparison, determine what the value represents.

Do NOT treat every number as an exact measured result.

Each reference value should be classified using:

"value_type":

- "measured_value"
  An explicitly reported measured experimental result.

- "upper_bound"
  A statement such as "less than 80 ms", "<80 ms",
  "below 5%", or equivalent.

- "lower_bound"
  A statement such as "more than 90%", ">90%",
  "above 1 Gbps", or equivalent.

- "range"
  A reported interval such as "10-20 ms" or "between 10 and 20 ms".

- "configuration"
  A value describing an experimental setup or condition,
  such as "500 UEs", "100 users", "batch size 64", or
  "trained for 50 epochs".

- "qualitative_result"
  A qualitative finding without a directly reported numerical
  measurement.

- "unknown"
  Use this only when the semantic meaning cannot be determined
  reliably from the supplied evidence.

Also determine:

"relation":

- "exact"
- "less_than"
- "less_than_or_equal"
- "greater_than"
- "greater_than_or_equal"
- "range"
- "none"

Preserve the original wording that establishes the meaning.

For example:

"less than 80 ms overhead for 500 UEs"

must NOT be represented simply as:

"latency_overhead_ms": 80

Instead, represent the latency reference as an upper bound and
represent 500 UEs as an experimental configuration.

The numerical value itself must remain unchanged. Do not calculate,
estimate, or normalize the value.

============================================================
IMPORTANT DISTINCTION
============================================================

A reference value is NOT automatically an experiment metric.

Distinguish between:

1. measured result
2. threshold/bound
3. experimental configuration
4. qualitative finding

A configuration value such as "500 UEs" must not be treated as a
performance metric.

A threshold such as "less than 80 ms" must not be treated as an
exact measured value of 80 ms.

A qualitative statement such as "effective detection" must not be
converted into an invented numerical detection rate.

============================================================
SCIENTIFIC FIDELITY
============================================================

1. Distinguish this paper's own experiment from background claims.

2. Distinguish measured results from assumptions.

3. Do not treat another paper's reported result as this paper's result.

4. Do not calculate metrics.

5. Do not infer a value merely because a concept is mentioned.

6. Do not convert units.

7. Do not normalize percentages.

8. Preserve baseline comparisons.

9. Preserve enough context to understand what each metric means.

10. If information is unavailable, use an empty list, empty object,
    or empty string.

============================================================
PAPER EXPERIMENTAL TEXT
============================================================

{text}
"""

        try:
            logger.info(
                "PaperReader experiment-details LLM call: reasoning=off"
            )

            response = self.llm_callable(
                prompt,
                temperature=0.0,
                reasoning="off",
                max_tokens=self.extraction_max_tokens,
            )

            print("\n[PaperReader] RAW LLM EXPERIMENT RESPONSE:")
            print("=" * 70)
            print(str(response))
            print("=" * 70)

            details = self._extract_json(response)

            print("\n[PaperReader] PARSED EXPERIMENT JSON:")
            print("=" * 70)
            print(details)
            print("=" * 70)

            if not details:
                logger.warning(
                    "[PaperReader] LLM returned invalid experiment JSON."
                )
                return {}

            details = self._validate_and_normalise_details(
                details,
                text,
            )

            logger.info(
                "[PaperReader] Extracted %d reference metrics.",
                len(details.get("reference_metrics", {})),
            )

            print("\n[PaperReader] NORMALISED REFERENCE METRICS:")
            print("=" * 70)
            print(
                json.dumps(
                    details.get("reference_metrics", {}),
                    indent=2,
                )
            )
            print("=" * 70)

            return details

        except Exception as exc:
            logger.exception(
                "[PaperReader] Experiment extraction failed: %s",
                exc,
            )
            return {}


    def _determine_results_status(
        self,
        experiment_details: Dict[str, Any],
    ) -> str:
        """
        Determine whether the paper contains usable experimental results.
        """
        if not experiment_details:
            return "no_results"

        reference_metrics = experiment_details.get(
            "reference_metrics",
            {},
        )

        results = experiment_details.get(
            "results",
            [],
        )

        if (
            isinstance(reference_metrics, dict)
            and reference_metrics
        ):
            return "available"

        if (
            isinstance(results, list)
            and results
        ):
            return "qualitative_only"

        return "no_results"

    # ============================================================
    # Complete Paper Reading
    # ============================================================

    def read_paper(
        self,
        url: str,
    ) -> str:
        """
        Read an evidence URL.

        Supports:
        - direct PDF URLs
        - arXiv /abs/ URLs
        - arXiv /html/ URLs
        - DOI/article URLs that return PDF
        - normal HTML articles/pages
        """
        source = self._download_source(url)

        content_type = source["content_type"]
        content = source["content"]

        if content_type == "pdf":
            full_text = self.extract_text(content)

        elif content_type == "html":
            html = content.decode(
                "utf-8",
                errors="replace",
            )

            full_text = self._extract_html_text(
                html
            )

        else:
            return ""

        if not full_text.strip():
            return ""

        return self.find_experimental_content(
            full_text
        )

    # ============================================================
    # Indexed Chunk Preparation
    # ============================================================

    def _prepare_indexed_text(
        self,
        chunks: List[Any],
    ) -> str:
        """
        Prepare indexed paper chunks for LLM processing.

        Chunks are kept in their existing order and the total text
        passed to the LLM is capped to prevent excessively large
        prompts.
        """
        if not chunks:
            return ""

        chunk_texts: List[str] = []
        total_length = 0

        for chunk in chunks:
            if hasattr(chunk, "page_content"):
                text = chunk.page_content

            elif isinstance(chunk, dict):
                text = (
                    chunk.get("page_content")
                    or chunk.get("text")
                    or chunk.get("content")
                    or ""
                )

            else:
                text = str(chunk)

            if not text or not text.strip():
                continue

            text = text.strip()

            remaining = (
                self.max_indexed_text_length
                - total_length
            )

            if remaining <= 0:
                break

            if len(text) > remaining:
                text = text[:remaining]

            chunk_texts.append(text)

            total_length += len(text)

            if total_length >= self.max_indexed_text_length:
                break

        return "\n\n".join(chunk_texts)

    # ============================================================
    # Read + Extract Experiment
    # ============================================================

    def read_experiment_reference(
        self,
        url: str,
        source: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Read a scientific paper and extract its experimental
        methodology and reported results.

        Preferred path:
            Existing ChromaPaperLibrary indexed chunks.

        Fallback path:
            Download and parse the PDF directly.

        The returned structure is:

        {
            "source_url": ...,
            "source_id": ...,
            "source_type": "scientific_paper",
            "results_text": ...,
            "experiment_details": {
                ...
                "metrics": [...],
                "metric_definitions": {...},
                "reference_metrics": {...},
                "results": [...]
            },
            "indexed": True/False
        }
        """
        if not url:
            return {}

        original_url = url.strip()

        # Keep this normalization here for logging/debugging and
        # compatibility with existing callers.
        pdf_url = self._normalise_paper_url(
            original_url
        )

        logger.info(
            "[PaperReader] Reading experiment reference: %s",
            original_url,
        )

        if pdf_url != original_url:
            logger.info(
                "[PaperReader] Normalized paper URL: %s",
                pdf_url,
            )

        # ========================================================
        # Preferred path: existing indexed paper chunks
        # ========================================================

        source_id = None

        if isinstance(source, dict):
            source_id = (
                source.get("source_id")
                or source.get("id")
            )

        if source_id and self.paper_library is not None:
            try:
                chunks = self.paper_library.get_source_chunks(
                    str(source_id)
                )

                if chunks:
                    logger.info(
                        "[PaperReader] Using %d indexed chunks "
                        "for source %s.",
                        len(chunks),
                        source_id,
                    )

                    indexed_text = self._prepare_indexed_text(
                        chunks
                    )

                    if indexed_text.strip():
                        logger.info(
                            "[PaperReader] Prepared %d characters "
                            "of indexed text for source %s.",
                            len(indexed_text),
                            source_id,
                        )

                        experimental_text = (
                            self.find_experimental_content(
                                indexed_text
                            )
                        )

                        if not experimental_text.strip():
                            experimental_text = indexed_text[
                                : self.max_text_length
                            ]

                        experiment_details = (
                            self.extract_experiment_details(
                                experimental_text
                            )
                        )

                        results_status = self._determine_results_status(
                            experiment_details
                        )

                        return {
                            "source_url": original_url,
                            "source_id": str(source_id),
                            "source_type": "scientific_paper",
                            "results_text": experimental_text,
                            "experiment_details": experiment_details,
                            "indexed": True,
                        }

                    logger.warning(
                        "[PaperReader] Indexed source %s contained "
                        "no usable text. Falling back to PDF.",
                        source_id,
                    )

            except Exception as exc:
                logger.warning(
                    "[PaperReader] Failed to read indexed source "
                    "%s: %s. Falling back to PDF.",
                    source_id,
                    exc,
                )

        # ========================================================
        # Fallback path: PDF workflow
        # ========================================================

        try:
            experimental_text = self.read_paper(
                original_url
            )

        except Exception as exc:
            logger.error(
                "[PaperReader] Failed to read paper %s: %s",
                original_url,
                exc,
            )
            return {}

        if not experimental_text.strip():
            return {}

        experiment_details = (
            self.extract_experiment_details(
                experimental_text
            )
        )

        return {
            "source_url": original_url,
            "source_type": "scientific_paper",
            "results_text": experimental_text,
            "experiment_details": experiment_details,
            "indexed": False,
        }