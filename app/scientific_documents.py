"""Structured scientific-document parsing primitives with a pypdf fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class DocumentElement:
    """One source-faithful element recovered from a scientific document."""

    element_type: str
    text: str
    page: int
    section: str = "Unknown"
    subsection: str = ""
    section_path: tuple[str, ...] = ("Unknown",)


@dataclass(frozen=True)
class ExtractedPaper:
    """Parsed pages/elements plus an honest record of parser truncation."""

    pages: tuple[tuple[int, str], ...]
    total_pages: int
    truncated: bool
    elements: tuple[DocumentElement, ...] = ()
    parser: str = "pypdf"


class ScientificDocumentParser(Protocol):
    """Interface for optional structured parsers used before the pypdf fallback."""

    name: str

    def parse(self, pdf_path: Path, *, max_pages: int) -> ExtractedPaper:
        """Parse a PDF into source-faithful document elements."""


_SECTION_NAMES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "literature review": "Literature Review",
    "materials and methods": "Methods",
    "materials & methods": "Methods",
    "methodology": "Methods",
    "methods": "Methods",
    "method": "Methods",
    "experiments": "Experiments",
    "experimental results": "Results",
    "results": "Results",
    "results and discussion": "Results and Discussion",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "acknowledgements": "Acknowledgements",
    "acknowledgments": "Acknowledgements",
    "references": "References",
    "bibliography": "References",
    "appendix": "Appendix",
}
_SUBSECTION_NAMES = {
    "dataset": "Dataset",
    "datasets": "Datasets",
    "experimental setup": "Experimental Setup",
    "experiment setup": "Experimental Setup",
    "main results": "Main Results",
    "ablation": "Ablation",
    "ablations": "Ablations",
    "ablation study": "Ablation Study",
    "implementation details": "Implementation Details",
}
_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<number>(?:\d+(?:\.\d+)*)|(?:[IVXLC]+))[.)]?\s+(?P<title>[^.!?]{1,100})\s*$",
    re.IGNORECASE,
)
_ATOMIC_ELEMENT_TYPES = {"table", "figure_caption", "equation", "code_block"}


def _normalized_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().rstrip(":").casefold()


def _display_heading(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().rstrip(":")
    if normalized.isupper():
        return normalized.title()
    return normalized


def _heading_path(line: str, current_section: str) -> tuple[str, str] | None:
    """Return (section, subsection) for conservative scientific headings."""

    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    numbered = _NUMBERED_HEADING.match(stripped)
    heading = numbered.group("title") if numbered else stripped
    normalized = _normalized_heading(heading)
    if normalized in _SECTION_NAMES:
        return _SECTION_NAMES[normalized], ""
    if normalized in _SUBSECTION_NAMES and current_section != "Unknown":
        return current_section, _SUBSECTION_NAMES[normalized]
    if numbered:
        number = numbered.group("number")
        display = _display_heading(heading)
        numeric_depth = number.count(".") + 1 if number[0].isdigit() else 1
        if numeric_depth > 1 and current_section != "Unknown":
            return current_section, display
        if len(display.split()) <= 10:
            return display, ""
    if stripped.isupper() and len(stripped.split()) <= 8 and not any(character in stripped for character in ".!?="):
        return _display_heading(stripped), ""
    return None


def _element_type(text: str) -> str:
    stripped = text.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    if re.match(r"^(?:table)\s+[A-Z0-9IVXLC]+[.:\s]", first_line, re.IGNORECASE):
        return "table"
    if re.match(r"^(?:figure|fig\.)\s+[A-Z0-9IVXLC]+[.:\s]", first_line, re.IGNORECASE):
        return "figure_caption"
    lines = stripped.splitlines()
    if len(lines) >= 2:
        numeric_rows = sum(len(re.findall(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?!\w)", line)) >= 2 for line in lines)
        if numeric_rows >= max(2, len(lines) // 2):
            return "table"
    if len(stripped) <= 500 and re.search(r"(?:=|≤|≥|≈|∑|∫|√|\^|_[A-Za-z0-9])", stripped):
        prose_words = re.findall(r"[A-Za-z]{3,}", stripped)
        if len(prose_words) <= 12:
            return "equation"
    if len(lines) >= 2 and sum(bool(re.search(r"[{};]|\b(?:def|class|return|import)\b", line)) for line in lines) >= 2:
        return "code_block"
    return "paragraph"


def recover_document_elements(pages: Sequence[tuple[int, str]]) -> tuple[DocumentElement, ...]:
    """Recover conservative section/paragraph structure from page text."""

    elements: list[DocumentElement] = []
    section = "Unknown"
    subsection = ""

    def append_buffer(buffer: list[str], page: int) -> None:
        text = "\n".join(buffer).strip()
        if not text:
            return
        path = tuple(item for item in (section, subsection) if item) or ("Unknown",)
        elements.append(
            DocumentElement(
                element_type=_element_type(text),
                text=text,
                page=page,
                section=section,
                subsection=subsection,
                section_path=path,
            )
        )

    for page, text in pages:
        buffer: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                append_buffer(buffer, page)
                buffer = []
                continue
            heading_path = _heading_path(line, section)
            if heading_path is not None:
                append_buffer(buffer, page)
                buffer = []
                section, subsection = heading_path
                continue
            buffer.append(line)
        append_buffer(buffer, page)

    # Keep a table caption and immediately following table rows together when
    # pypdf exposes them as separate blocks on the same page.
    merged: list[DocumentElement] = []
    for element in elements:
        if (
            merged
            and merged[-1].element_type == "table"
            and element.element_type == "table"
            and merged[-1].page == element.page
            and merged[-1].section_path == element.section_path
        ):
            previous = merged.pop()
            merged.append(
                DocumentElement(
                    element_type="table",
                    text=f"{previous.text}\n{element.text}",
                    page=element.page,
                    section=element.section,
                    subsection=element.subsection,
                    section_path=element.section_path,
                )
            )
        else:
            merged.append(element)
    return tuple(merged)


class PypdfScientificParser:
    """Dependency-light fallback parser with conservative structure recovery."""

    name = "pypdf"

    def parse(self, pdf_path: Path, *, max_pages: int) -> ExtractedPaper:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages: list[tuple[int, str]] = []
        total_pages = len(reader.pages)
        for page_number, page in enumerate(reader.pages[:max_pages], start=1):
            raw_text = (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
            lines = [line.rstrip() for line in raw_text.split("\n")]
            text = "\n".join(lines)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append((page_number, text))
        page_tuple = tuple(pages)
        return ExtractedPaper(
            pages=page_tuple,
            total_pages=total_pages,
            truncated=total_pages > max_pages,
            elements=recover_document_elements(page_tuple),
            parser=self.name,
        )


def split_text_by_boundaries(text: str, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """Split paragraph text at sentence/line/word boundaries before hard cuts."""

    normalized = text.strip()
    if not normalized:
        return []
    maximum = max(1, int(max_chars))
    overlap = min(max(0, int(overlap_chars)), maximum - 1)
    chunks: list[str] = []
    start = 0
    minimum_boundary = max(start + 1, maximum // 2)
    while start < len(normalized):
        hard_end = min(start + maximum, len(normalized))
        end = hard_end
        if hard_end < len(normalized):
            boundary_floor = start + minimum_boundary
            for separator in ("\n", ". ", "? ", "! ", " "):
                boundary = normalized.rfind(separator, boundary_floor, hard_end)
                if boundary >= boundary_floor:
                    end = boundary + (2 if separator in {". ", "? ", "! "} else 1)
                    break
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        next_start = max(start + 1, end - overlap)
        if next_start > start and next_start < end and not normalized[next_start - 1].isspace():
            following_space = normalized.find(" ", next_start, end)
            if following_space != -1:
                next_start = following_space + 1
        start = next_start
    return chunks


@dataclass(frozen=True)
class ElementChunk:
    """A section-bounded raw chunk assembled from one or more elements."""

    raw_text: str
    page_start: int
    page_end: int
    section: str
    subsection: str
    section_path: tuple[str, ...]
    element_type: str


def chunk_document_elements(
    elements: Sequence[DocumentElement],
    *,
    max_chars: int,
    overlap_chars: int = 0,
) -> tuple[ElementChunk, ...]:
    """Chunk by section, paragraph, sentence, and finally hard boundaries."""

    maximum = max(1, int(max_chars))
    overlap = min(max(0, int(overlap_chars)), maximum - 1)
    chunks: list[ElementChunk] = []
    pending: list[DocumentElement] = []

    def flush_pending() -> None:
        if not pending:
            return
        first = pending[0]
        chunks.append(
            ElementChunk(
                raw_text="\n\n".join(element.text for element in pending),
                page_start=min(element.page for element in pending),
                page_end=max(element.page for element in pending),
                section=first.section,
                subsection=first.subsection,
                section_path=first.section_path,
                element_type=first.element_type if len(pending) == 1 else "paragraph",
            )
        )
        pending.clear()

    for element in elements:
        text = element.text.strip()
        if not text:
            continue
        path_changed = bool(pending and pending[0].section_path != element.section_path)
        atomic = element.element_type in _ATOMIC_ELEMENT_TYPES
        if path_changed or atomic:
            flush_pending()
        if atomic:
            pieces = [text] if len(text) <= maximum else split_text_by_boundaries(text, maximum, 0)
            for piece in pieces:
                chunks.append(
                    ElementChunk(
                        raw_text=piece,
                        page_start=element.page,
                        page_end=element.page,
                        section=element.section,
                        subsection=element.subsection,
                        section_path=element.section_path,
                        element_type=element.element_type,
                    )
                )
            continue
        if len(text) > maximum:
            flush_pending()
            for piece in split_text_by_boundaries(text, maximum, overlap):
                chunks.append(
                    ElementChunk(
                        raw_text=piece,
                        page_start=element.page,
                        page_end=element.page,
                        section=element.section,
                        subsection=element.subsection,
                        section_path=element.section_path,
                        element_type=element.element_type,
                    )
                )
            continue
        candidate_length = len(text) + sum(len(item.text) for item in pending) + (2 * len(pending))
        if pending and candidate_length > maximum:
            flush_pending()
        pending.append(element)
    flush_pending()
    return tuple(chunks)
