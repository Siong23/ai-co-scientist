"""
Paper Reader

Retrieves research papers from evidence URLs and extracts relevant text
and experiment details for use by the automated experiment pipeline.
"""

import io
import json
import re
from typing import Any, Dict, List, Optional

import requests
from pypdf import PdfReader
from responses import logger


class PaperReader:
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
        # This prevents very large prompts when a paper contains
        # many indexed chunks.
        self.max_indexed_text_length = 12000

    # ============================================================
    # URL Normalization
    # ============================================================

    def _normalise_paper_url(self, url: str) -> str:
        """
        Normalize common scientific-paper URLs into downloadable
        PDF URLs where a deterministic conversion is available.
        """

        if not url:
            return url

        url = url.strip()

        # arXiv HTML page -> PDF
        if "arxiv.org/html/" in url:
            url = url.replace(
                "arxiv.org/html/",
                "arxiv.org/pdf/",
            )

        # arXiv abstract page -> PDF
        elif "arxiv.org/abs/" in url:
            url = url.replace(
                "arxiv.org/abs/",
                "arxiv.org/pdf/",
            )

        return url

    # ============================================================
    # PDF Download
    # ============================================================

    def download_paper(self, url: str) -> bytes:
        """
        Download a research paper PDF from the provided URL.
        """

        if not url:
            raise ValueError("Paper URL is required.")

        url = self._normalise_paper_url(url)

        response = requests.get(
            url,
            timeout=self.timeout,
            headers={
                "User-Agent": (
                    "AI-Co-Scientist-ExperimentComparator/1.0"
                )
            },
        )

        response.raise_for_status()

        content = response.content

        if not content.startswith(b"%PDF"):
            content_type = response.headers.get("Content-Type", "")
            preview = content[:100].decode(
                "utf-8",
                errors="replace",
            )

            raise ValueError(
                "The evidence URL did not return a PDF. "
                f"URL: {url}. "
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
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    # ============================================================
    # Results Section Detection
    # ============================================================

    def find_experimental_content(self, text: str) -> str:
        """
        Identify and extract the parts of a scientific paper that
        contain experimental evaluation, quantitative results,
        performance measurements, or empirical findings.

        Section names are not hard-coded because scientific papers
        may use different structures and terminology.
        """

        if not text or not text.strip():
            return ""

        if self.llm_callable is None:
            print(
                "[PaperReader] No LLM callable configured. "
                "Returning full paper text."
            )
            return text[:self.max_text_length]

        prompt = f"""
You are analyzing a scientific research paper.

Your task is to identify the parts of the paper that contain
experimental evaluation, empirical results, quantitative
measurements, performance comparisons, or experimental findings.

IMPORTANT:
- Do not assume the section is called "Results".
- Section names may vary between papers.
- Use the CONTENT and CONTEXT of the paper to identify the
relevant experimental/results sections.
- Do not invent results.
- Do not calculate or modify numerical values.
- Include the relevant experimental setup when it is necessary
to understand the reported results.
- Exclude unrelated background, introduction, and literature review.

Return ONLY the relevant text from the paper.
Preserve the original wording and numerical values.

PAPER TEXT:
{text[:self.max_text_length]}
"""

        try:
            logger.info(
                "PaperReader experimental-content LLM call: reasoning=off"
            )

            result = self.llm_callable(
                prompt,
                temperature=0.0,
                reasoning="off",
                max_tokens=2048,
            )

            if not result:
                print(
                    "[PaperReader] LLM returned no results section."
                )
                return text[:self.max_text_length]

            return str(result).strip()

        except Exception as exc:
            print(
                "[PaperReader] Failed to identify results section "
                f"with LLM: {exc}"
            )
            return text[:self.max_text_length]


    # ============================================================
    # JSON Response Extraction
    # ============================================================

    def _extract_json(self, response: Any) -> Dict[str, Any]:
        """
        Extract a JSON object from an LLM response.

        Handles:
        - Direct dictionaries
        - Plain JSON strings
        - JSON wrapped in markdown code fences
        - JSON embedded in surrounding text
        """

        if response is None:
            return {}

        # LLM callable may already return a dictionary.
        if isinstance(response, dict):
            return response

        text = str(response).strip()

        if not text:
            return {}

        # Remove markdown code fences such as:
        # ```json
        # {...}
        # ```
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

        # First try parsing the entire response.
        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # If the LLM added explanatory text around the JSON,
        # locate the outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return {}

        json_text = text[start:end + 1]

        try:
            parsed = json.loads(json_text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            return {}

        return {}

    
    # ============================================================
    # Experiment Detail Extraction
    # ============================================================

    def extract_experiment_details(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Extract experiment details from scientific paper text
        using the configured LLM.

        The LLM must only extract information explicitly supported
        by the provided text.
        """

        if not text or not text.strip():
            return {}

        if self.llm_callable is None:
            print(
                "[PaperReader] No LLM callable configured. "
                "Cannot perform structured experiment extraction."
            )
            return {
                "raw_text": text,
                "models": [],
                "datasets": [],
                "metrics": [],
                "hyperparameters": {},
                "training_details": {},
                "experiment_notes": [],
            }

        prompt = f"""
Extract experimental information from the scientific paper text below.

OUTPUT REQUIREMENT:
Your ENTIRE response must be exactly ONE valid JSON object.

Do NOT:
- explain your answer
- provide reasoning
- describe what you are doing
- repeat the instructions
- use Markdown
- use ```json fences
- add text before or after the JSON
- invent information

The JSON object MUST use exactly this structure:

{{
  "experiment_objective": "",
  "experimental_setup": [],
  "models_or_systems": [],
  "datasets_or_testbeds": [],
  "baselines": [],
  "configurations": [],
  "metrics": [],
  "hyperparameters": {{}},
  "training_details": {{}},
  "reference_metrics": {{}},
  "results": [],
  "experiment_notes": []
}}

Rules:

1. Extract ONLY information explicitly supported by the paper text.
2. Do NOT invent missing values.
3. Do NOT calculate any results.
4. Preserve numerical values and ranges exactly as stated.
5. Preserve units exactly as stated.
6. Distinguish measured results from assumptions.
7. Include baseline and comparison configurations when stated.
8. Include domain-specific metrics such as latency, throughput,
   overhead, runtime, memory, communication cost, energy, or
   security measurements when explicitly reported.
9. If information is unavailable, use an empty list, empty object,
   or empty string as appropriate.
10. Do not treat background or related-work claims as experimental
    results unless the text explicitly identifies them as results
    of this paper.
11. Keep each result concise.
12. Do not copy large sections of the paper into the JSON.

PAPER EXPERIMENTAL TEXT:

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
                max_tokens=2048,
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

            required_keys = [
                "experiment_objective",
                "experimental_setup",
                "models_or_systems",
                "datasets_or_testbeds",
                "baselines",
                "configurations",
                "metrics",
                "hyperparameters",
                "training_details",
                "reference_metrics",
                "results",
                "experiment_notes",
            ]

            if not details:
                print(
                    "[PaperReader] LLM returned invalid experiment JSON."
                )
                return {}

            missing_keys = [
                key for key in required_keys
                if key not in details
            ]

            if missing_keys:
                print(
                    "[PaperReader] Experiment JSON is missing keys: "
                    + ", ".join(missing_keys)
                )
                return {}

            details["raw_text"] = text

            return details
            
        except Exception as exc:
            print(
                "[PaperReader] Experiment extraction failed: "
                f"{exc}"
            )
            return {}
    
    # ============================================================
    # Complete Paper Reading
    # ============================================================

    def read_paper(self, url: str) -> str:
        """
        Download a paper and identify the experimental content.
        """
        url = self._normalise_paper_url(url)
        pdf_bytes = self.download_paper(url)
        full_text = self.extract_text(pdf_bytes)

        if not full_text.strip():
            return ""

        return self.find_experimental_content(full_text)

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

        If an indexed paper source is available, reuse the existing
        ChromaPaperLibrary chunks instead of downloading and parsing
        the PDF again. Fall back to the existing PDF-based workflow
        when indexed chunks are unavailable.
        """
        if not url:
            return {}

        url = self._normalise_paper_url(url)

        # ============================================================
        # Preferred path: use existing indexed paper chunks
        # ============================================================
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
                        "[PaperReader] Using %d indexed chunks for source %s.",
                        len(chunks),
                        source_id,
                    )

                    indexed_text = self._prepare_indexed_text(chunks)

                    if indexed_text.strip():
                        logger.info(
                            "[PaperReader] Prepared %d characters of indexed text "
                            "for source %s.",
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
                                :self.max_text_length
                            ]

                        experiment_details = (
                            self.extract_experiment_details(
                                experimental_text
                            )
                        )

                        return {
                            "source_url": url,
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
                    "[PaperReader] Failed to read indexed source %s: %s. "
                    "Falling back to PDF.",
                    source_id,
                    exc,
                )

        # ============================================================
        # Fallback path: existing PDF download/extraction workflow
        # ============================================================
        experimental_text = self.read_paper(url)

        if not experimental_text.strip():
            return {}

        experiment_details = self.extract_experiment_details(
            experimental_text
        )

        return {
            "source_url": url,
            "source_type": "scientific_paper",
            "results_text": experimental_text,
            "experiment_details": experiment_details,
            "indexed": False,
        }
