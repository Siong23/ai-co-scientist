"""
Paper Reader

Retrieves research papers from evidence URLs and extracts text from
PDF documents for use by the Experiment Comparator.
"""

import io
import re

import requests
from pypdf import PdfReader


class PaperReader:
    """
    Retrieve and extract relevant text from research papers.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_text_length: int = 30000,
    ):
        self.timeout = timeout
        self.max_text_length = max_text_length

    # ========================================================
    # Download Paper
    # ========================================================

    def download_paper(self, url: str) -> bytes:
        """
        Download a PDF from the supplied URL.
        """
        if not url:
            raise ValueError("Paper URL is required.")

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

        # Verify that the downloaded content is actually a PDF.
        if not content.startswith(b"%PDF"):
            content_type = response.headers.get("Content-Type", "")
            preview = content[:100].decode(
                "utf-8",
                errors="replace",
            )

            raise ValueError(
                "The evidence URL did not return a PDF. "
                f"Content-Type: {content_type}. "
                f"Response preview: {preview}"
            )

        return content

    # ========================================================
    # Extract PDF Text
    # ========================================================

    def extract_text(self, pdf_bytes: bytes) -> str:
        """
        Extract text from a PDF document.
        """

        if not pdf_bytes:
            return ""

        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)
    
    # ========================================================
    # Find Relevant Sections
    # ========================================================

    def find_results_section(self, text: str) -> str:
        """
        Locate the most relevant results/evaluation section in extracted
        scientific paper text without depending on a specific section number.
        """

        if not text:
            return ""

        lines = text.splitlines()

        # Keywords that commonly identify result/evaluation sections.
        section_keywords = (
            "results",
            "result",
            "evaluation",
            "experimental results",
            "experimental evaluation",
            "experiments",
            "performance evaluation",
            "performance results",
            "results and discussion",
            "results and discussions",
            "discussion and results",
            "discussion and discussions",
        )

        candidates = []

        for index, line in enumerate(lines):
            cleaned = re.sub(r"\s+", " ", line).strip()

            if not cleaned:
                continue

            lowered = cleaned.lower()

            # Ignore very long normal paragraphs.
            if len(cleaned) > 100:
                continue

            # Remove common section numbering.
            heading_text = re.sub(
                r"^(?:[ivxlcdm]+|\d+(?:\.\d+)*)[\.\)]?\s*",
                "",
                lowered,
            ).strip()

            # Only accept exact heading matches.
            # This avoids matching ordinary sentences such as:
            # "evaluation. Performance improved..."
            if heading_text in section_keywords:
                has_section_number = bool(
                    re.match(
                        r"^\s*(?:[ivxlcdm]+|\d+(?:\.\d+)*)[\.\)]?\s+",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                )

                is_uppercase_heading = (
                    cleaned.upper() == cleaned
                    and any(character.isalpha() for character in cleaned)
                )

                if has_section_number or is_uppercase_heading:
                    candidates.append(index)

        if not candidates:
            print(
                "[PaperReader] No Results/Evaluation section heading matched."
            )
            return text[: self.max_text_length]

        # Prefer the earliest strong Results/Evaluation section.
        start_line = candidates[0]

        print(
            f"[PaperReader] Matched Results section: "
            f"{repr(lines[start_line])}"
        )

        relevant_text = "\n".join(lines[start_line:])

        print(
            f"[PaperReader] Extracting paper from line "
            f"{start_line} ({len(lines)} total lines)."
        )

        return relevant_text[: self.max_text_length]
    
    # ========================================================
    # Read Paper
    # ========================================================

    def read_paper(self, url: str) -> str:
        """
        Download a paper and extract its relevant text.

        Args:
            url: Paper PDF URL.

        Returns:
            Relevant extracted paper text.

        Raises:
            ValueError: If the URL is invalid.
            requests.RequestException: If the paper cannot be downloaded.
        """

        pdf_bytes = self.download_paper(url)

        full_text = self.extract_text(pdf_bytes)

        if not full_text.strip():
            return ""

        return self.find_results_section(full_text)
