"""Turn uploaded files into clean text sections, then into chunks.

Step 1 of the pipeline. Each file type is read with its own library:
  - PDF  -> pypdf        (one section per page)
  - DOCX -> python-docx  (one section per heading)
  - PPTX -> python-pptx  (one section per slide, including speaker notes)
Every section remembers where it came from (page/slide number), so the
app can later show "Source: slide 4" chips.
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx"}
LEGACY_EXTENSIONS = {".doc": ".docx", ".ppt": ".pptx"}
MIN_TEXT_CHARS = 200
HEADING_STYLES = {"Title", "Heading 1", "Heading 2", "Heading 3"}


class ParseError(Exception):
    """Raised when a file can't be read; the message is shown to the user."""


@dataclass
class Section:
    """A piece of a document with its origin.

    location is "page N" or "slide N" (DOCX has no pages, so we use the
    section number). heading is the nearest title, if any.
    """

    text: str
    source: str
    file_type: str
    location: str
    heading: str = ""


def clean_text(text: str) -> str:
    """Collapse runs of spaces/tabs and blank lines so the text is compact."""
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def check_extension(filename: str) -> str:
    """Return the lowercase extension, or raise a friendly ParseError."""
    ext = Path(filename).suffix.lower()
    if ext in LEGACY_EXTENSIONS:
        raise ParseError(
            f"Old '{ext}' files aren't supported. Please open it and 'Save As' {LEGACY_EXTENSIONS[ext]}."
        )
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError("Unsupported file type. Please upload a .pdf, .docx or .pptx file.")
    return ext


def parse_pdf(path: Path, source: str) -> tuple[list[Section], list[str]]:
    """Read a PDF page by page. PDFs carry no reliable headings, so none are returned."""
    reader = PdfReader(str(path))
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            sections.append(Section(text, source, "pdf", f"page {page_number}"))
    return sections, []


def _iter_docx_blocks(doc: Any) -> list[Paragraph | Table]:
    """Return paragraphs and tables in the order they appear in the document.

    python-docx gives paragraphs and tables as separate lists; walking the
    underlying XML body keeps their real order.
    """
    blocks: list[Paragraph | Table] = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            blocks.append(Paragraph(child, doc))
        elif child.tag.endswith("}tbl"):
            blocks.append(Table(child, doc))
    return blocks


def _table_to_text(table: Table) -> str:
    """Flatten a table to text, one row per line, cells separated by ' | '."""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_docx(path: Path, source: str) -> tuple[list[Section], list[str]]:
    """Read a Word file, grouping paragraphs under the most recent heading."""
    doc = DocxDocument(str(path))
    sections: list[Section] = []
    headings: list[str] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        """Turn the collected text into a Section and reset the buffer."""
        text = clean_text("\n".join(buffer))
        if text:
            location = f"section {len(sections) + 1}"
            sections.append(Section(text, source, "docx", location, current_heading))
        buffer.clear()

    for block in _iter_docx_blocks(doc):
        if isinstance(block, Table):
            buffer.append(_table_to_text(block))
            continue
        text = block.text.strip()
        if not text:
            continue
        style = block.style.name if block.style is not None else ""
        if style in HEADING_STYLES:
            flush()
            current_heading = text
            headings.append(text)
            buffer.append(text)
        else:
            buffer.append(text)
    flush()
    return sections, headings


def parse_pptx(path: Path, source: str) -> tuple[list[Section], list[str]]:
    """Read a PowerPoint: title = heading, other text = body, plus speaker notes."""
    prs = Presentation(str(path))
    sections: list[Section] = []
    headings: list[str] = []
    for slide_number, slide in enumerate(prs.slides, start=1):
        title_shape = slide.shapes.title
        title = title_shape.text.strip() if title_shape is not None and title_shape.has_text_frame else ""
        parts = [title] if title else []
        for shape in slide.shapes:
            if shape is title_shape or not shape.has_text_frame:
                continue
            body = shape.text_frame.text.strip()
            if body:
                parts.append(body)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"Notes: {notes}")
        text = clean_text("\n".join(parts))
        if text:
            sections.append(Section(text, source, "pptx", f"slide {slide_number}", title))
        if title:
            headings.append(title)
    return sections, headings


def dedupe_keep_order(items: list[str]) -> list[str]:
    """Remove duplicates (ignoring case) while keeping the first occurrence."""
    seen: set[str] = set()
    result = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def parse_document(path: Path, source: str) -> tuple[list[Section], list[str]]:
    """Parse any supported file and return (sections, headings).

    `source` is the original file name shown to the user. Raises ParseError
    with a human-readable message if the file can't be used.
    """
    ext = check_extension(source)
    parsers = {".pdf": parse_pdf, ".docx": parse_docx, ".pptx": parse_pptx}
    try:
        sections, headings = parsers[ext](path, source)
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"Could not read this file. Is it damaged or password-protected? ({exc})") from exc

    total_chars = sum(len(s.text) for s in sections)
    if total_chars < MIN_TEXT_CHARS:
        raise ParseError("No readable text found. Is this a scanned PDF?")
    return sections, dedupe_keep_order(headings)


def chunk_sections(
    sections: list[Section], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[dict[str, Any]]:
    """Split sections into overlapping chunks, copying metadata onto each.

    Why chunk? Embedding models and LLM prompts work best on short passages.
    The overlap means a sentence cut at a boundary still appears whole in
    one of the two neighbouring chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for section in sections:
        meta = asdict(section)
        meta.pop("text")
        for piece in splitter.split_text(section.text):
            chunks.append({**meta, "text": piece, "chunk_index": len(chunks)})
    return chunks
