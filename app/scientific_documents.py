"""Structured scientific-document parsing primitives with a pypdf fallback."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
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
    "acknowledgement": "Acknowledgements",
    "acknowledgements": "Acknowledgements",
    "acknowledgment": "Acknowledgements",
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
# Section numbers have at most two digits per level, so page numbers printed
# before a running header ("812 Journal ...") do not match. Roman numerals must
# be uppercase and end in a period, as in "IV. RESULTS".
_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<number>\d{1,2}(?:\.\d{1,2}){0,3}|[IVX]{1,4})(?P<punctuation>[.)]?)\s+(?P<title>[^.!?]{1,100})\s*$"
)
_ROMAN_NUMERALS = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}
# pypdf splits small-caps headings after their first letter: "R ELATED W ORK".
_SMALL_CAPS_SPLIT = re.compile(r"\b([A-Z])\s(?=[A-Z]{2,}\b)")
# A and I are also English words, so a lone split on them is left alone.
_UNAMBIGUOUS_SMALL_CAPS_SPLIT = re.compile(r"\b([B-HJ-Z])\s(?=[A-Z]{2,}\b)")
_EDGE_LINE_COUNT = 3
_ATOMIC_ELEMENT_TYPES = {"table", "figure_caption", "equation", "code_block"}


def _normalized_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().rstrip(":").casefold()


def _display_heading(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().rstrip(":")
    if normalized.isupper():
        return normalized.title()
    return normalized


def _join_small_caps(value: str) -> str:
    return _SMALL_CAPS_SPLIT.sub(r"\1", value)


def _display_title(title: str) -> str:
    # A lone split on A or I can be a real word ("A BRIEF OVERVIEW"), so those
    # are rejoined only when the title shows the small-caps pattern repeatedly.
    if title.isupper():
        if len(_SMALL_CAPS_SPLIT.findall(title)) >= 2:
            title = _join_small_caps(title)
        else:
            title = _UNAMBIGUOUS_SMALL_CAPS_SPLIT.sub(r"\1", title)
    return _display_heading(title)


def _known_heading(candidates: Sequence[str], current_section: str) -> tuple[str, str] | None:
    for candidate in candidates:
        for form in (candidate, _join_small_caps(candidate)):
            normalized = _normalized_heading(form)
            if normalized in _SECTION_NAMES:
                return _SECTION_NAMES[normalized], ""
            if normalized in _SUBSECTION_NAMES and current_section != "Unknown":
                return current_section, _SUBSECTION_NAMES[normalized]
    return None


def _is_math_symbol(character: str) -> bool:
    return unicodedata.category(character) == "Sm" or 0x1D400 <= ord(character) <= 0x1D7FF


def _plausible_heading_title(title: str) -> bool:
    """Reject figure labels, formulas, and sentence fragments posing as titles."""

    stripped = title.strip()
    visible = re.sub(r"\s", "", stripped)
    ascii_letters = sum(character.isascii() and character.isalpha() for character in visible)
    first_letter = next((character for character in stripped if character.isalpha()), "")
    return (
        bool(re.search(r"[A-Za-z]{3,}", stripped))
        and ascii_letters >= 0.6 * len(visible)
        and len(stripped.split()) <= 10
        and not any(character in stripped for character in ",=")
        and not any(_is_math_symbol(character) for character in stripped)
        and first_letter.isupper()
    )


def _heading_path(
    line: str,
    current_section: str,
    last_top_number: int | None = None,
) -> tuple[str, str, int | None] | None:
    """Return (section, subsection, top-level number) for a conservative heading.

    Known section names always count. Other titles count only when they are
    numbered in sequence: figure labels, formulas, running headers, and list
    items otherwise open spurious sections that fragment every later chunk.
    """

    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    numbered = _NUMBERED_HEADING.match(stripped)
    title = numbered.group("title").strip() if numbered else stripped
    top_number: int | None = None
    depth = 0
    if numbered:
        number = numbered.group("number")
        if number[0].isdigit():
            parts = [int(part) for part in number.split(".")]
            top_number, depth = parts[0], len(parts)
        elif numbered.group("punctuation") == "." and number in _ROMAN_NUMERALS:
            top_number, depth = _ROMAN_NUMERALS[number], 1

    known = _known_heading((title, stripped), current_section)
    if known is not None:
        return known[0], known[1], top_number if depth == 1 else last_top_number
    # Journal names and numbered entries inside a bibliography are not headings.
    if current_section == "References" or depth == 0 or not _plausible_heading_title(title):
        return None
    if depth == 1:
        if last_top_number is None:
            in_sequence = top_number <= 3
        else:
            in_sequence = last_top_number < top_number <= last_top_number + 2
        return (_display_title(title), "", top_number) if in_sequence else None
    if top_number != last_top_number or current_section == "Unknown":
        return None
    return current_section, _display_title(title), last_top_number


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


def _edge_line_key(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().casefold()))


def _repeated_edge_lines(pages: Sequence[tuple[int, str]]) -> set[str]:
    """Find running headers and footers repeated at the edges of many pages."""

    page_counts: Counter[str] = Counter()
    for _page, text in pages:
        lines = [line for line in text.splitlines() if line.strip()]
        edge_lines = lines[:_EDGE_LINE_COUNT] + lines[-_EDGE_LINE_COUNT:]
        page_counts.update({_edge_line_key(line) for line in edge_lines})
    minimum_pages = max(3, int(0.3 * len(pages)))
    return {key for key, count in page_counts.items() if count >= minimum_pages}


def recover_document_elements(pages: Sequence[tuple[int, str]]) -> tuple[DocumentElement, ...]:
    """Recover conservative section/paragraph structure from page text."""

    elements: list[DocumentElement] = []
    section = "Unknown"
    subsection = ""
    last_top_number: int | None = None
    running_lines = _repeated_edge_lines(pages)

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
        lines = text.splitlines()
        non_blank = [index for index, line in enumerate(lines) if line.strip()]
        edge_indexes = set(non_blank[:_EDGE_LINE_COUNT] + non_blank[-_EDGE_LINE_COUNT:])
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if index in edge_indexes and _edge_line_key(line) in running_lines:
                continue
            if not line:
                append_buffer(buffer, page)
                buffer = []
                continue
            heading_path = _heading_path(line, section, last_top_number)
            if heading_path is not None:
                append_buffer(buffer, page)
                buffer = []
                section, subsection, last_top_number = heading_path
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
    combine_under_chars: int = 0,
    standalone_min_chars: int = 0,
    excluded_sections: Sequence[str] = (),
) -> tuple[ElementChunk, ...]:
    """Chunk by section, paragraph, sentence, and finally hard boundaries.

    A section change closes the pending chunk only once it holds
    ``combine_under_chars``, so short sections fill one chunk instead of
    becoming fragments. Tables, captions, equations, and code shorter than
    ``standalone_min_chars`` stay with the surrounding text. The defaults keep
    every section and atomic element in chunks of its own.
    """

    maximum = max(1, int(max_chars))
    overlap = min(max(0, int(overlap_chars)), maximum - 1)
    combine_under = max(0, int(combine_under_chars))
    standalone_min = max(0, int(standalone_min_chars))
    excluded = set(excluded_sections)
    chunks: list[ElementChunk] = []
    pending: list[DocumentElement] = []

    def flush_pending() -> None:
        if not pending:
            return
        # A chunk that spans sections is labelled by the one that contributes
        # the most text; the first such section wins a tie.
        weights: dict[tuple[str, ...], int] = {}
        for element in pending:
            weights[element.section_path] = weights.get(element.section_path, 0) + len(element.text)
        dominant_path = max(weights, key=weights.__getitem__)
        label = next(element for element in pending if element.section_path == dominant_path)
        chunks.append(
            ElementChunk(
                raw_text="\n\n".join(element.text for element in pending),
                page_start=min(element.page for element in pending),
                page_end=max(element.page for element in pending),
                section=label.section,
                subsection=label.subsection,
                section_path=label.section_path,
                element_type=pending[0].element_type if len(pending) == 1 else "paragraph",
            )
        )
        pending.clear()

    for element in elements:
        if element.section in excluded:
            continue
        text = element.text.strip()
        if not text:
            continue
        path_changed = bool(pending and pending[-1].section_path != element.section_path)
        pending_length = sum(len(item.text) for item in pending) + 2 * max(0, len(pending) - 1)
        if path_changed and pending_length >= combine_under:
            flush_pending()
        atomic = element.element_type in _ATOMIC_ELEMENT_TYPES and len(text) >= standalone_min
        if atomic:
            flush_pending()
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
